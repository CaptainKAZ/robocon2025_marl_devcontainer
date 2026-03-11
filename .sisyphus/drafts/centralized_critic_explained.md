# Centralized Critic 如何评价队伍表现

**生成时间**: 2026-03-06  
**核心问题**: Critic 评价的是整支队伍还是个体智能体？如何做到的？

---

## 🎯 核心答案

**Critic 评价的是整支队伍的综合表现，通过全局状态学习团队价值，然后为每个智能体提供相同的价值估计。**

**关键机制**：
- **输入**：全局状态（所有智能体的信息）
- **输出**：单个团队价值 → 复制给所有智能体
- **不是平均化，而是复制！**

---

## 1. Critic 架构详解

### 1.1 两种参数共享模式

从 `mappo.py:611-658` 的代码：

```python
def get_critic(self, group: str) -> TensorDictModule:
    n_agents = len(self.group_map[group])
    
    # ========================================
    # 关键：两种输出模式
    # ========================================
    if self.share_param_critic:  # 默认 True
        # 模式 1：输出单个团队价值
        critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
    else:
        # 模式 2：输出每个 agent 独立的值
        critic_output_spec = Composite({
            group: Composite(
                {"state_value": Unbounded(shape=(n_agents, 1))},
                shape=(n_agents,),
            )
        })
    
    # ========================================
    # 输入：全局状态
    # ========================================
    if self.state_spec is not None:
        # 使用全局状态（关键！）
        critic_input_spec = self.state_spec
    else:
        # 使用所有 agent 的观察拼接
        critic_input_spec = Composite({group: self.observation_spec[group]})
    
    # 创建 critic model
    value_module = self.critic_model_config.get_model(
        input_spec=critic_input_spec,
        output_spec=critic_output_spec,
        centralised=True,  # 关键：Centralized Critic
        share_params=self.share_param_critic,
        ...
    )
    
    # ========================================
    # 如果参数共享，扩展输出
    # ========================================
    if self.share_param_critic:
        # 核心：将单个值复制到所有 agent
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

---

### 1.2 数据流详解

#### **模式 1：share_param_critic=True（默认，推荐）**

```
全局状态输入 (23维)
    [A1位置, A1速度, A1状态, 
     A2位置, A2速度, 
     D1位置, D1速度, 
     D2位置, D2速度, 
     投篮点, 篮筐, 时间]
         ↓
    Centralized Critic (Attention Model)
    - 3层 Attention
    - 全局态势理解
    - 学习团队协作价值
         ↓
    单个团队价值 V(s_global)
    shape: [batch, 1]
         ↓
    expand_module (复制)
    unsqueeze(-2) → [batch, 1, 1]
    expand → [batch, n_agents, 1]
         ↓
    输出给所有 agent
    (attacker, "state_value") = [batch, 2, 1]
    - A1 的值 = V(s_global)
    - A2 的值 = V(s_global)  # 相同！
```

**关键点**：
- ✅ **单个网络**：所有 agent 共享一个 critic 网络
- ✅ **单个输出**：网络输出一个标量值
- ✅ **复制机制**：通过 `expand()` 复制给所有 agent
- ❌ **不是平均化**：没有计算 agent 价值的平均

---

#### **模式 2：share_param_critic=False**

```
全局状态输入 (23维)
         ↓
    Centralized Critic
    - 每个 agent 有独立的网络参数
    - 或者共享部分参数，但最终 MLP 独立
         ↓
    每个 agent 独立的值
    shape: [batch, n_agents, 1]
         ↓
    输出
    (attacker, "state_value") = [batch, 2, 1]
    - A1 的值 = V₁(s_global)
    - A2 的值 = V₂(s_global)  # 可能不同
```

**区别**：
- 参数是否共享
- 输出是单个值还是多个值
- 但都是基于全局状态

---

## 2. 为什么是"复制"而不是"平均"？

### 2.1 代码证据

```python
# 关键代码：mappo.py:649-656
expand_module = TensorDictModule(
    lambda value: value.unsqueeze(-2).expand(
        *value.shape[:-1], n_agents, 1
    ),
    in_keys=["state_value"],
    out_keys=[(group, "state_value")],
)
```

**解析**：
1. `value` 的 shape 是 `[batch, 1]`
2. `unsqueeze(-2)` → `[batch, 1, 1]`
3. `expand(*value.shape[:-1], n_agents, 1)` → `[batch, n_agents, 1]`
4. **这是复制操作，不是求和或平均！**

---

### 2.2 数学表达

**团队价值函数**：

$$
V_{\text{team}}(s) = V_\theta(s_{\text{global}})
$$

**每个 agent 的值**：

$$
V_{a_1}(s) = V_{\text{team}}(s) \\
V_{a_2}(s) = V_{\text{team}}(s)
$$

**不是平均化**：

$$
\text{❌ 错误理解: } V_{a_i} = \frac{1}{N} \sum_{j=1}^{N} V_j(s)
$$

$$
\text{✅ 正确理解: } V_{a_i} = V_{\text{team}}(s) \quad \forall i \in \{1, 2\}
$$

---

## 3. 为什么设计成"团队价值"？

### 3.1 MAPPO 的核心思想

**Centralized Training, Decentralized Execution (CTDE)**：

| 阶段 | Critic | Actor |
|------|--------|-------|
| **训练** | 看到全局状态 | 看到局部观察 |
| **执行** | 不需要 | 只需要局部观察 |

**价值函数的意义**：
- **团队价值** = 从当前全局状态出发，整个团队的期望累积奖励
- **不是个体价值** = 不是单个 agent 能获得的奖励

---

### 3.2 篮球场景的直觉

**进攻方团队价值**：

$$
V_{\text{attacker}}(s) = \mathbb{E}\left[ \sum_{t=0}^T \gamma^t r_{\text{team}}(t) \mid s_0 = s \right]
$$

其中：
- $r_{\text{team}}$ = 进攻方团队奖励（A1 和 A2 共享）
- 团队奖励取决于：A1 是否得分、A2 是否有效掩护、配合是否成功

**关键理解**：
- A1 的成功 = 团队的成功
- A2 的成功 = 团队的成功
- **因此 A1 和 A2 应该有相同的价值估计**

---

### 3.3 为什么不是个体价值？

假设我们为每个 agent 计算独立的价值：

```
❌ 错误设计：
V_A1(s) = 期望 A1 获得的奖励
V_A2(s) = 期望 A2 获得的奖励

问题：
1. 在合作任务中，个体奖励不独立
2. A1 的成功依赖于 A2 的配合
3. 分离的价值函数难以学习协作
```

**正确设计**：

```
✅ 正确设计：
V_team(s) = 期望团队获得的奖励

优势：
1. 明确团队目标
2. 学习协作行为
3. 避免"自私"策略
```

---

## 4. Loss 如何使用这些值？

### 4.1 Value Target 计算（GAE）

```python
# 对每个 agent 独立计算
for agent in [A1, A2]:
    # TD residual
    δ_agent = r_agent + γ * V_team(s') - V_team(s)
    
    # GAE
    A_agent = Σ (γλ)^l * δ_{t+l}
    
    # Value target
    V_target_agent = A_agent + V_team(s)
```

**关键**：
- 虽然 $V_{\text{team}}$ 相同，但 $r_{\text{agent}}$ 不同
- 因此 $V_{\text{target}}$ 会不同
- 每个 agent 有独立的价值目标

---

### 4.2 Critic Loss 计算

```python
# 对每个 agent 独立计算
for agent in [A1, A2]:
    # Critic 预测（相同）
    V_pred_agent = V_team(s)  # 来自同一个 critic
    
    # Value target（不同）
    V_target_agent = ...  # 从 GAE 计算得到
    
    # Loss
    loss_agent = smooth_l1(V_pred_agent, V_target_agent)

# 总 loss = 平均所有 agent 的 loss
loss_critic = mean([loss_A1, loss_A2])
```

**关键机制**：
1. **预测值相同**：$V_{\text{pred}}^{A1} = V_{\text{pred}}^{A2} = V_{\text{team}}(s)$
2. **目标值不同**：$V_{\text{target}}^{A1} \neq V_{\text{target}}^{A2}$（因为奖励不同）
3. **梯度信号不同**：每个 agent 的 loss 提供不同的梯度

---

## 5. 参数共享的影响

### 5.1 share_param_critic=True（默认）

**优势**：
- ✅ 参数效率高（单个网络）
- ✅ 强制团队价值概念
- ✅ 更容易学习协作
- ✅ 更稳定（梯度来自多个 agent 的平均）

**劣势**：
- ❌ 无法区分 agent 角色
- ❌ 所有 agent 必须有相同的价值理解

---

### 5.2 share_param_critic=False

**优势**：
- ✅ 每个 agent 可以有不同价值理解
- ✅ 可以学习异质角色

**劣势**：
- ❌ 参数量增加
- ❌ 可能学习"自私"策略
- ❌ 训练更不稳定

---

## 6. 实际训练效果

### 6.1 在您的篮球场景中

**进攻方（attacker）**：
- `share_param_critic=True`
- A1 和 A2 看到相同的价值
- 学习团队协作（A1 得分 = 团队成功）

**防守方（defender）**：
- `share_param_critic=True`
- D1 和 D2 看到相同的价值
- 学习团队防守（阻止得分 = 团队成功）

---

### 6.2 梯度流

```
Loss computation:
    ├─ Agent A1:
    │  V_pred = V_team(s)  # 相同
    │  V_target_A1 = ...    # 不同（因为奖励不同）
    │  loss_A1 = smooth_l1(V_pred, V_target_A1)
    │  gradient_A1 = ∂loss_A1/∂θ_critic
    │
    └─ Agent A2:
       V_pred = V_team(s)  # 相同
       V_target_A2 = ...    # 不同
       loss_A2 = smooth_l1(V_pred, V_target_A2)
       gradient_A2 = ∂loss_A2/∂θ_critic

Optimizer update:
    θ_critic ← θ_critic - lr * (gradient_A1 + gradient_A2) / 2
```

**关键**：
- 每个 agent 提供独立的梯度信号
- 梯度平均后更新同一个 critic 网络
- 网络学习到的是"平均"的团队价值

---

## 7. 总结

### 7.1 核心机制

| 维度 | 说明 |
|------|------|
| **输入** | 全局状态（所有 agent 信息） |
| **输出** | 单个团队价值 |
| **分配** | 复制给所有 agent（不是平均） |
| **参数** | 所有 agent 共享 critic 参数 |
| **目标** | 每个 agent 有独立的 value target |
| **梯度** | 来自所有 agent 的平均 |

---

### 7.2 关键理解

1. **Critic 评价的是团队表现**，不是个体
2. **输出是单个值，复制给所有 agent**
3. **不是平均化，是复制**
4. **每个 agent 的 value target 不同**（因为奖励不同）
5. **梯度信号来自所有 agent 的平均**

---

### 7.3 为什么这样设计？

**MAPPO 的哲学**：
- 团队成功 = 个体成功
- 协作比个人英雄主义更重要
- 共享的价值函数强制团队意识

**在篮球场景中**：
- A1 得分 = 进攻方成功 = A2 成功
- 防守成功 = D1 和 D2 共同成功
- 因此使用团队价值是合理的

---

## 8. 配置建议

### 8.1 推荐配置

```yaml
# attacker (异质角色，协作任务)
share_param_critic: True   # 团队价值
share_param_actor: False   # 独立策略

# defender (同质角色)
share_param_critic: True   # 团队价值
share_param_actor: True    # 共享策略
```

---

### 8.2 何时使用 share_param_critic=False？

**适用场景**：
- 竞争性团队内部（每个 agent 有自己的目标）
- 强异质角色（不同 agent 有完全不同的任务）
- 非合作任务

**不适用您的篮球场景**，因为：
- 进攻方和防守方内部是合作的
- 团队成功 = 个体成功
- 应该使用团队价值

---

**文档版本**: 1.0  
**最后更新**: 2026-03-06