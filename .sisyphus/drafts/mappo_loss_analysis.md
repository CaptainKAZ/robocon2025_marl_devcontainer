# MAPPO Loss 计算详解

**生成时间**: 2026-03-06  
**项目**: RoboCon 2025 MARL Basketball Layup  
**算法**: MAPPO (Multi-Agent PPO)

---

## 目录

1. [核心结论](#1-核心结论)
2. [Loss 计算架构](#2-loss-计算架构)
3. [GAE 计算](#3-gae-计算)
4. [Critic Loss 计算](#4-critic-loss-计算)
5. [Policy Loss 计算](#5-policy-loss-计算)
6. [参数共享与分组训练](#6-参数共享与分组训练)
7. [代码流程分析](#7-代码流程分析)

---

## 1. 核心结论

### 1.1 GAE 计算方式

**GAE 是按 minibatch 分组计算的，而不是整体计算**

```python
# 关键代码位置：mappo.py:573-594
def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
    if self.minibatch_advantage:
        # 按 minibatch 计算 GAE（推荐，节省内存）
        increment = -(
            -self.experiment.config.train_minibatch_size(self.on_policy)
            // batch.shape[1]
        )
    else:
        # 按 full batch 计算 GAE
        increment = batch.batch_size[0] + 1
    
    # 对每个 minibatch 计算 value target
    while last_start_index < batch.shape[0]:
        minimbatch = batch[last_start_index:start_index]
        with torch.no_grad():
            loss.value_estimator(
                minimbatch,
                params=loss.critic_network_params,
                target_params=loss.target_critic_network_params,
            )
```

**配置参数**：`minibatch_advantage: true` (默认)

---

### 1.2 Loss 计算方式

**Loss 是按智能体组（attacker/defender）分开计算，然后独立优化的**

| 组别 | Loss Module | 优化器 | 参数 |
|------|------------|--------|------|
| **attacker** | ClipPPOLoss | 独立优化器 | attacker actor + critic 参数 |
| **defender** | ClipPPOLoss | 独立优化器 | defender actor + critic 参数 |

---

## 2. Loss 计算架构

### 2.1 整体架构

```
Experiment (experiment.py)
    │
    ├─ Attacker Group
    │   ├─ Actor Model (分散式，参数可选共享)
    │   ├─ Critic Model (集中式，参数可选共享)
    │   └─ ClipPPOLoss (独立 loss module)
    │       ├─ Policy Loss (clipped surrogate)
    │       └─ Value Loss (MSE/smooth_l1)
    │
    └─ Defender Group
        ├─ Actor Model (分散式，参数可选共享)
        ├─ Critic Model (集中式，参数可选共享)
        └─ ClipPPOLoss (独立 loss module)
            ├─ Policy Loss (clipped surrogate)
            └─ Value Loss (MSE/smooth_l1)
```

---

### 2.2 Loss Module 创建流程

```python
# 步骤 1: 为每个组创建独立的 Loss Module
def _get_loss(self, group: str, policy_for_loss: TensorDictModule, continuous: bool):
    loss_module = ClipPPOLoss(
        actor=policy_for_loss,              # 该组的 actor
        critic=self.get_critic(group),      # 该组的 critic (centralized)
        clip_epsilon=self.clip_epsilon,     # PPO clip 参数
        entropy_coeff=self.entropy_coef,    # entropy bonus 系数
        critic_coeff=self.critic_coef,      # critic loss 权重
        loss_critic_type=self.loss_critic_type,  # "l1", "l2", "smooth_l1"
        normalize_advantage=False,          # 是否归一化 advantage
    )
    
    # 设置 TensorDict keys（按组区分）
    loss_module.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target"),
        value=(group, "state_value"),
        sample_log_prob=(group, "log_prob"),
    )
    
    # 配置 GAE value estimator
    loss_module.make_value_estimator(
        ValueEstimators.GAE, 
        gamma=self.experiment_config.gamma, 
        lmbda=self.lmbda
    )
    
    return loss_module, False  # False = 不使用 target network
```

**关键点**：
- 每个组有独立的 `ClipPPOLoss` 实例
- TensorDict keys 包含组名前缀，避免冲突
- Critic 是 centralized（使用全局状态）
- Actor 是 decentralized（使用局部观察）

---

## 3. GAE 计算

### 3.1 GAE 公式

**Generalized Advantage Estimation (GAE)**：

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**参数**：
- `gamma` (γ): 折扣因子，默认 0.995
- `lmbda` (λ): GAE 参数，默认 0.95

---

### 3.2 GAE 计算代码位置

```python
# 文件：mappo.py:573-594
def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
    """
    在训练前处理 batch 数据，计算 GAE
    """
    loss = self.get_loss_and_updater(group)[0]
    
    # 决定 GAE 计算的 batch 大小
    if self.minibatch_advantage:
        # 按 minibatch 计算（节省内存）
        increment = -(
            -self.experiment.config.train_minibatch_size(self.on_policy)
            // batch.shape[1]
        )
    else:
        # 按 full batch 计算（更准确）
        increment = batch.batch_size[0] + 1
    
    # 分批次计算 GAE
    last_start_index = 0
    start_index = increment
    minibatches = []
    
    while last_start_index < batch.shape[0]:
        minimbatch = batch[last_start_index:start_index]
        minibatches.append(minimbatch)
        
        with torch.no_grad():
            # 关键：调用 value_estimator 计算 GAE
            loss.value_estimator(
                minimbatch,
                params=loss.critic_network_params,
                target_params=loss.target_critic_network_params,
            )
        
        last_start_index = start_index
        start_index += increment
    
    # 合并所有 minibatch
    batch = torch.cat(minibatches, dim=0)
    return batch
```

**TensorDict 输出**：
- `(group, "advantage")`: GAE advantage 估计
- `(group, "value_target")`: Value target（用于 critic loss）

---

### 3.3 Minibatch vs Full Batch GAE

| 方式 | 内存占用 | 准确性 | 推荐场景 |
|------|---------|--------|---------|
| **Minibatch GAE** | 低 | 略低（边界效应） | 大规模训练、内存受限 |
| **Full Batch GAE** | 高 | 高 | 小规模训练、精确优化 |

**配置**：`minibatch_advantage: true` (默认)

---

## 4. Critic Loss 计算

### 4.1 Critic Loss 公式

**Value Function Loss**：

$$
L_{\text{critic}} = \frac{1}{N} \sum_{i=1}^{N} \left( V(s_i) - V^{\text{target}}_i \right)^2
$$

**可选损失函数**（`loss_critic_type`）：

| 类型 | 公式 | 特点 |
|------|------|------|
| `"l2"` (MSE) | $(V - V^{\text{target}})^2$ | 标准选择，对异常值敏感 |
| `"l1"` (MAE) | $\|V - V^{\text{target}}\|$ | 鲁棒，但对小误差不敏感 |
| `"smooth_l1"` (Huber) | $\begin{cases} 0.5x^2 & \text{if } \|x\| < 1 \\ \|x\| - 0.5 & \text{otherwise} \end{cases}$ | **推荐**，平衡鲁棒性和敏感性 |

---

### 4.2 Critic Loss 代码实现

```python
# TorchRL ClipPPOLoss 内部实现
def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
    # 1. 获取 critic 预测值
    value = self.critic(tensordict)
    
    # 2. 获取 value target (从 GAE 计算得到)
    value_target = tensordict.get(self.value_target_key)
    
    # 3. 计算 value loss
    if self.loss_critic_type == "l2":
        loss_value = (value - value_target).pow(2).mean()
    elif self.loss_critic_type == "l1":
        loss_value = (value - value_target).abs().mean()
    elif self.loss_critic_type == "smooth_l1":
        diff = value - value_target
        loss_value = torch.where(
            diff.abs() < 1.0,
            0.5 * diff.pow(2),
            diff.abs() - 0.5
        ).mean()
    
    # 4. 加权
    loss_value = loss_value * self.critic_coeff
    
    return TensorDict({"loss_critic": loss_value}, [])
```

---

### 4.3 Centralized Critic

**关键特性**：Critic 使用全局状态，而非局部观察

```python
# 文件：mappo.py:611-658
def get_critic(self, group: str) -> TensorDictModule:
    # 判断是否使用全局状态
    if self.state_spec is not None:
        input_has_agent_dim = False
        critic_input_spec = self.state_spec  # 使用全局状态！
    else:
        input_has_agent_dim = True
        critic_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
    
    # 创建 critic model
    value_module = self.critic_model_config.get_model(
        input_spec=critic_input_spec,
        output_spec=critic_output_spec,
        n_agents=n_agents,
        centralised=True,  # 关键：Centralized Critic
        input_has_agent_dim=input_has_agent_dim,
        agent_group=group,
        share_params=self.share_param_critic,
        device=self.device,
        action_spec=self.action_spec,
        name=f"{group}_critic"
    )
    
    # 如果参数共享，扩展输出到所有智能体
    if self.share_param_critic:
        expand_module = TensorDictModule(
            lambda value: value.unsqueeze(-2).expand(
                *value.shape[:-1], n_agents, 1
            ),
            in_keys=["state_value"],
            out_keys=[(group, "state_value")],
        )
        value_module = TensorDictSequential(value_module, expand_module)
    
    return value_module
```

**Centralized Critic 的意义**：
- **训练阶段**：Critic 可以看到全局状态，准确评估价值
- **执行阶段**：Actor 只需要局部观察，实现分散式执行
- **参数共享**：可以选择所有智能体共享一个 critic 参数

---

## 5. Policy Loss 计算

### 5.1 Policy Loss 公式

**Clipped Surrogate Objective**：

$$
L_{\text{policy}} = -\mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

**参数**：
- `clip_epsilon` (ε): PPO clip 范围，默认 0.2
- `advantage` (Â): GAE 估计的优势函数

---

### 5.2 Entropy Bonus

$$
L_{\text{entropy}} = -\beta \cdot \mathbb{E}_t \left[ H(\pi_\theta(\cdot | s_t)) \right]
$$

**参数**：
- `entropy_coef` (β): entropy bonus 系数，默认 0.01

**作用**：鼓励探索，防止策略过早收敛

---

### 5.3 Policy Loss 代码实现

```python
# TorchRL ClipPPOLoss 内部实现
def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
    # 1. 获取 action log prob
    log_prob = tensordict.get(self.sample_log_prob_key)
    old_log_prob = tensordict.get(self.old_log_prob_key)
    
    # 2. 计算 importance ratio
    ratio = torch.exp(log_prob - old_log_prob)
    
    # 3. 获取 advantage
    advantage = tensordict.get(self.advantage_key)
    
    # 4. 计算 clipped surrogate loss
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
    loss_policy = -torch.min(surr1, surr2).mean()
    
    # 5. 计算 entropy bonus
    dist = self.actor.get_dist(tensordict)
    entropy = dist.entropy().mean()
    loss_entropy = -self.entropy_coeff * entropy
    
    # 6. 合并 policy loss
    loss_objective = loss_policy + loss_entropy
    
    return TensorDict({
        "loss_objective": loss_objective,
        "loss_entropy": loss_entropy,
    }, [])
```

---

### 5.4 最终 Loss 组合

```python
# 文件：mappo.py:598-605
def process_loss_vals(self, group: str, loss_vals: TensorDictBase) -> TensorDictBase:
    """
    合并 policy loss 和 entropy loss
    """
    loss_vals.set(
        "loss_objective", 
        loss_vals["loss_objective"] + loss_vals["loss_entropy"]
    )
    del loss_vals["loss_entropy"]
    return loss_vals
```

**最终 Loss 输出**：
- `loss_objective`: Policy loss + Entropy loss
- `loss_critic`: Value function loss

---

## 6. 参数共享与分组训练

### 6.1 参数共享配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `share_param_actor` | True (defender) / False (attacker) | Actor 参数共享 |
| `share_param_critic` | True | Critic 参数共享 |

**配置示例**（`clear_restore.py`）：

```python
# Attacker 配置
attacker_config = MappoConfig(
    share_param_actor=False,  # A1 和 A2 有不同的策略
    share_param_critic=True,  # 共享 critic 参数
    ...
)

# Defender 配置
defender_config = MappoConfig(
    share_param_actor=True,   # D1 和 D2 共享策略
    share_param_critic=True,  # 共享 critic 参数
    ...
)
```

---

### 6.2 分组训练机制

```python
# 文件：experiment.py (训练循环)
for group in self.group_map.keys():
    # 1. 获取该组的 loss module
    loss, updater = self.algorithm.get_loss_and_updater(group)
    
    # 2. 获取该组的参数
    params = self.algorithm.get_parameters(group)
    
    # 3. 计算该组的 loss
    loss_vals = loss(batch)
    
    # 4. 反向传播和优化
    for loss_name, loss_value in loss_vals.items():
        optimizer = self.optimizers[group][loss_name]
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**关键点**：
- 每个组有独立的优化器
- 每个组有独立的梯度更新
- 可以选择性冻结某些组的训练（通过 `train_group_map` 控制）

---

### 6.3 参数提取

```python
# 文件：mappo.py:230-234
def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
    """
    提取该组 loss 对应的参数
    """
    return {
        "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
        "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
    }
```

**参数分组**：
- `loss_objective`: Actor 参数（policy network）
- `loss_critic`: Critic 参数（value network）

---

## 7. 代码流程分析

### 7.1 完整训练流程

```
1. 数据收集 (Collection)
   ├─ 环境交互
   ├─ 存储 (group, "observation"), (group, "action"), (group, "reward")
   └─ 存储 (group, "log_prob") (旧策略的 log prob)

2. 数据预处理 (Preprocessing)
   ├─ 调用 process_batch(group, batch)
   ├─ 计算 GAE advantage: (group, "advantage")
   └─ 计算 value target: (group, "value_target")

3. Loss 计算 (Loss Computation)
   ├─ 遍历每个组
   ├─ 调用 loss_module(batch)
   ├─ 返回 loss_objective (policy loss)
   └─ 返回 loss_critic (value loss)

4. 梯度更新 (Gradient Update)
   ├─ 反向传播
   ├─ 优化器更新
   └─ 清零梯度
```

---

### 7.2 关键代码位置

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| **Loss Module 创建** | `mappo.py` | 201-228 | 为每个组创建 ClipPPOLoss |
| **GAE 计算** | `mappo.py` | 573-594 | process_batch 方法 |
| **Critic 创建** | `mappo.py` | 611-658 | 创建 centralized critic |
| **参数提取** | `mappo.py` | 230-234 | 提取 actor/critic 参数 |
| **Loss 处理** | `mappo.py` | 598-605 | 合并 entropy loss |
| **训练循环** | `experiment.py` | - | 优化器循环 |

---

### 7.3 TensorDict Keys 流转

```
Collection 阶段:
  (group, "observation")      # 观察输入
  (group, "action")           # 动作输出
  (group, "log_prob")         # 旧策略 log prob
  ("next", group, "reward")   # 奖励
  ("next", group, "done")     # 终止标志

Preprocessing 阶段:
  (group, "advantage")        # GAE advantage
  (group, "value_target")     # Value target

Loss 计算阶段:
  (group, "state_value")      # Critic 预测值
  loss_objective              # Policy loss
  loss_critic                 # Value loss
```

---

## 8. 总结

### 8.1 核心要点

1. **GAE 计算是分组进行的**：
   - 按 minibatch 计算（默认）或 full batch 计算
   - 每个组独立计算 advantage

2. **Loss 是按智能体组分开计算的**：
   - Attacker 和 Defender 有独立的 loss module
   - 每个组有独立的优化器和梯度更新

3. **Centralized Critic, Decentralized Actor**：
   - Critic 使用全局状态（训练时）
   - Actor 使用局部观察（训练和执行时）

4. **参数共享可配置**：
   - Actor 参数共享：可选（attacker 不共享，defender 共享）
   - Critic 参数共享：默认共享

5. **标准 MAPPO Loss**：
   - Policy Loss: Clipped Surrogate Objective
   - Value Loss: MSE / L1 / Smooth L1
   - Entropy Bonus: 鼓励探索

---

### 8.2 推荐配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `clip_epsilon` | 0.2 | PPO clip 范围 |
| `entropy_coef` | 0.01 | Entropy bonus 系数 |
| `critic_coef` | 1.0 | Critic loss 权重 |
| `loss_critic_type` | `"smooth_l1"` | Value loss 类型 |
| `lmbda` | 0.95 | GAE lambda |
| `gamma` | 0.99 | 折扣因子 |
| `minibatch_advantage` | `true` | Minibatch GAE |

---

**文档版本**: 1.0  
**最后更新**: 2026-03-06