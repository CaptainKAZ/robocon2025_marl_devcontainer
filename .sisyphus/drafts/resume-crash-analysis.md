# Draft: 训练恢复崩溃问题分析

## 问题描述

**现象**: 使用 `autoresume.sh cold` 训练后，resume 时模型完全崩溃
**矛盾**: 使用 `visualize_critic_fixed.py` 可视化时，模型加载和运行正常
**关键推断**: 保存没问题，问题出在加载到训练阶段

---

## 代码路径分析

### 1. autoresume.sh 调用逻辑

```bash
# 第一次运行
python3 clear_restore.py -m cold    # 从零开始

# 后续运行
python3 clear_restore.py -m cont    # 继续训练
```

### 2. 训练脚本 clear_restore.py 的加载方式

**cont 模式**:
```python
# 设置 restore_file，让 Experiment 自己恢复完整状态
experiment_config.restore_file = checkpoint_path

# 创建新的实验对象
experiment = Experiment(
    task=new_task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=114514,
    config=experiment_config,
    callbacks=[WinRateReportDebounced()]
)

# experiment.run() 内部会调用 _load_experiment()
```

**关键流程**:
1. `Experiment.__init__()` → 调用 `_setup()`
2. `_setup()` → 调用 `_setup_name()` → 检查 `restore_file`
3. **然后** `_setup()` → 调用 `_load_experiment()` 加载状态

### 3. 可视化脚本的加载方式

```python
exp = Experiment.reload_from_file(checkpoint_path)
```

**关键流程**:
1. `reload_from_file()` → 读取 `config.pkl` 恢复所有配置
2. 创建 Experiment 对象（使用保存时的配置）
3. `_load_experiment()` 加载状态

---

## 关键差异分析

### 差异 1: 配置来源不同

| 方面 | 训练恢复 (cont) | 可视化 |
|------|-----------------|--------|
| task | `new_task = LayupTask.LAYUP.get_from_yaml()` | 从 `config.pkl` 加载 |
| algorithm_config | 新创建的 `MappoConfig` | 从 `config.pkl` 加载 |
| model_config | 新创建的 `SequenceModelConfig` | 从 `config.pkl` 加载 |
| seed | 硬编码 `114514` | 从 `config.pkl` 加载 |
| callbacks | 新创建 `[WinRateReportDebounced()]` | 从 `config.pkl` 加载 |

### 差异 2: 初始化顺序不同

**训练恢复**:
```
Experiment.__init__()
  → _setup()
    → _setup_name()      # 设置 folder_name
    → _setup_task()      # 创建环境
    → _setup_algorithm() # 创建 losses, optimizers, buffers
    → _setup_collector()
    → _setup_logger()
    → _on_setup()
  → _load_experiment()   # 加载状态覆盖
```

**可视化**:
```
reload_from_file()
  → 从 pickle 加载所有配置
  → Experiment.__init__() (使用保存的配置)
    → _setup() (同上)
    → _load_experiment()
```

---

## 潜在问题点

### 问题 1: 新配置 vs 保存的配置

训练恢复时创建**新的**配置对象，可能与保存时的配置不一致：

```python
# 训练脚本中硬编码的配置
attacker_algorithm_config = MappoConfig.get_from_yaml()
attacker_algorithm_config.share_param_actor = False
attacker_algorithm_config.share_param_critic = True

# 如果 YAML 文件或默认值变化，可能导致不匹配
```

### 问题 2: Callback 状态不保存

`WinRateReportDebounced` callback 的状态**不会**保存到 checkpoint：
- `fix_attacker_mode` / `fix_defender_mode` 状态丢失
- 训练模式判断逻辑可能重置

### 问题 3: 环境变量依赖

```python
# cont 模式设置
os.environ['VMAS_INITIAL_SHOT_THRESHOLD'] = '0.2'

# 但这个环境变量不会从 checkpoint 恢复
# 如果环境逻辑依赖这个变量，可能导致状态不一致
```

### ~~问题 4: n_iters_performed 恢复~~ ✅ 已确认正确恢复

```python
# state_dict 保存
state = OrderedDict(
    total_time=self.total_time,
    total_frames=self.total_frames,
    n_iters_performed=self.n_iters_performed,
    mean_return=self.mean_return,
)

# load_state_dict 中确实恢复了（第 1368-1371 行）
self.total_time = state_dict["state"]["total_time"]
self.total_frames = state_dict["state"]["total_frames"]
self.n_iters_performed = state_dict["state"]["n_iters_performed"]
self.mean_return = state_dict["state"]["mean_return"]
```

**确认**: 这个不是问题点。✅

### 问题 5: train_group_map 不保存

```python
# callback 中修改训练组
train_map = self.experiment.train_group_map
if fix_attacker_mode:
    if "attacker" in train_map:
        del train_map["attacker"]

# 但 train_group_map 不在 state_dict 中
# 恢复后会重置为默认值
```

### 问题 6: 收集器 (collector) 状态

```python
# state_dict 中保存
if not self.config.collect_with_grad:
    state_dict.update({"collector": self.collector.state_dict()})

# 但加载时需要检查 collector 状态是否正确恢复
# 特别是环境状态、随机种子等
```

---

## checkpoint 完整结构

```python
{
    "state": {
        "total_time": ...,
        "total_frames": ...,
        "n_iters_performed": ...,
        "mean_return": ...
    },
    "loss_attacker": {...},        # actor + critic 网络参数
    "loss_defender": {...},
    "buffer_attacker": {...},      # replay buffer (如果 off-policy)
    "buffer_defender": {...},
    "collector": {...},            # 环境收集器状态
    "grad_scaler_attacker": {...}, # AMP 缩放器
    "grad_scaler_defender": {...},
    "optimizer_attacker": {...},   # 优化器状态
    "optimizer_defender": {...},
    "lr_scheduler_attacker": {...}, # 学习率调度器
    "lr_scheduler_defender": {...}
}
```

**注意**: 
- Callback 状态 **不保存**
- train_group_map **不保存**
- 环境变量 **不保存**

---

## 建议的诊断实验

### 实验 1: 检查 total_frames 是否恢复

```python
# 在 _load_experiment() 后添加打印
print(f"Loaded total_frames: {self.total_frames}")
print(f"Loaded n_iters: {self.n_iters_performed}")
```

### 实验 2: 对比训练恢复和可视化的配置

```python
# 在两种方式加载后，打印配置对比
print("=== Config Comparison ===")
print(f"Task config: {experiment.task.config}")
print(f"Algorithm config: {experiment.algorithm_config}")
print(f"Model config: {experiment.model_config}")
```

### 实验 3: 检查优化器状态

```python
# 打印优化器状态的关键值
for group in experiment.optimizers:
    for name, opt in experiment.optimizers[group].items():
        print(f"Optimizer {group}/{name}:")
        print(f"  state keys: {opt.state.keys()}")
        print(f"  param_groups: {opt.param_groups}")
```

### 实验 4: 检查网络参数一致性

```python
# 加载后检查 actor 参数是否一致
import torch

# 训练恢复
exp_train = ...  # cont 模式加载

# 可视化加载
exp_viz = Experiment.reload_from_file(checkpoint_path)

# 对比参数
for name, param_train in exp_train.losses["attacker"].actor_network.named_parameters():
    param_viz = exp_viz.losses["attacker"].actor_network.state_dict()[name]
    if not torch.allclose(param_train, param_viz):
        print(f"MISMATCH: {name}")
```

### 实验 5: 最小复现测试

创建一个简单的脚本：
```python
# 1. cold 训练 10 iterations
# 2. 保存 checkpoint
# 3. cont 恢复
# 4. 检查:
#    - total_frames 是否正确
#    - actor 输出是否一致
#    - optimizer 状态是否存在
```

---

## 高概率问题猜测

### ~~猜测 1: n_iters_performed 未恢复导致学习率问题~~ ❌ 已排除

已确认 `total_frames` 和 `n_iters_performed` 正确恢复。

### 猜测 2: 配置不一致导致网络结构不匹配

训练恢复时创建新配置，如果与保存时不一致：
- 网络结构可能不同
- `load_state_dict` 静默失败（没有检查返回值）
- 部分参数未加载

### 猜测 3: GRU 隐藏状态未正确初始化

GRU 等循环网络的隐藏状态：
- 可视化时可能使用 `is_init=True` 正确初始化
- 训练恢复时可能使用错误的初始隐藏状态

---

## 待验证的问题

1. `load_state_dict` 是否真的没有恢复 `state` 字段中的 `total_frames` 等？
2. 配置对象创建时，是否与保存时完全一致？
3. `reload_from_file` 和 `restore_file` 方式的 `_setup()` 调用是否有区别？
4. Callback 的 `train_group_map` 修改是否会在恢复后生效？

---

## 下一步行动

等待用户选择：
1. 执行哪个诊断实验
2. 是否需要更深入探索某个特定方向
3. 是否需要查看更多代码细节