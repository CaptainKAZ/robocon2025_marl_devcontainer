# AGENTS.md

**生成时间**: 2026-03-05
**Commit**: e07bbd88
**Branch**: master

始终使用中文简体回复

---

## 概述

RoboCon 2025 多智能体强化学习 (MARL) 项目，训练篮球上篮场景的竞争性智能体。项目基于两个 **forked git 子模块** 并进行重大定制修改：

- **BenchMARL** (forked): TorchRL 的 MARL 训练库 - 定制支持混合动作空间、异步评估、自定义模型
- **VectorizedMultiAgentSimulator/VMAS** (forked): 向量化 2D 物理模拟器 - 扩展自定义上篮场景

核心环境：`vmas/layup` - 2v2 攻防对抗（进攻方投篮得分，防守方封盖）

---

## 项目结构

```
.
├── BenchMARL/                          # Forked & Modified
│   ├── clear_restore.py                # 主训练入口 ⭐
│   ├── health_check.py                 # 神经网络健康监控回调
│   ├── autoresume.sh                   # 自动恢复训练脚本
│   ├── benchmarl/
│   │   ├── algorithms/mappo.py         # ✨ 修改: 混合动作支持
│   │   ├── environments/layup/
│   │   │   └── common.py               # ✨ 修改: FlattenHybridAction 变换
│   │   ├── models/attention.py         # ✨ 修改: 参数共享修复
│   │   ├── experiment/experiment.py    # ✨ 修改: 异步评估 worker
│   │   └── conf/                       # Hydra 配置
│   └── outputs/                        # 训练输出和检查点
│
├── VectorizedMultiAgentSimulator/      # Forked & Modified
│   └── vmas/scenarios/
│       ├── layup.py                    # ✨ 自定义: 完整游戏逻辑
│       └── layup_jit.py                # ✨ 自定义: JIT 奖励计算
│
└── .devcontainer/                      # DevContainer 配置
```

---

## 快速开始

| 任务 | 位置 | 说明 |
|------|------|------|
| 训练 | `BenchMARL/clear_restore.py` | 主入口，包含实验配置、模型设置、检查点加载 |
| 环境逻辑 | `VectorizedMultiAgentSimulator/vmas/scenarios/layup.py` | 游戏规则、奖励函数、终止条件 |
| JIT 优化 | `layup_jit.py` | 向量化奖励计算（3-5x 加速） |
| 检查点恢复 | `clear_restore.py:339-350` | 取消注释并修改 `restore_file_path` |
| 配置文件 | `benchmarl/conf/` | Hydra YAML 配置（实验、算法、模型） |

---

## 🔥 核心定制修改

### 1. Hybrid Action Space Support (Continuous + Discrete)

**Modified Files:**
- `BenchMARL/benchmarl/environments/layup/common.py`
- `BenchMARL/benchmarl/algorithms/mappo.py`
- `BenchMARL/benchmarl/experiment/experiment.py`

**What Changed:**
The original BenchMARL only supported pure continuous or pure discrete actions. We added full support for **hybrid action spaces** (e.g., velocity control + discrete brake trigger).

**Key Implementation:**
```python
# common.py - FlattenHybridAction Transform
class FlattenHybridAction(Transform):
    """
    Converts hybrid actions (continuous + discrete) into flat tensor for VMAS.
    - Forward: Agent -> Env (flattens [vx, vy] + [trigger] -> [vx, vy, trigger_float])
    - Inverse: Env -> Agent (identity)
    """
```

**How It Works:**
1. Actor outputs `CompositeSpec` with `{"continuous": ..., "discrete": ...}`
2. `FlattenHybridAction._inv_call()` concatenates them before sending to VMAS
3. VMAS receives flat `[vx, vy, brake_signal]` tensor
4. `mappo.py._get_composite_policy()` constructs multi-distribution policy with proper log-prob summation

**Why This Matters:**
Enables agents to learn both continuous movement control AND discrete decision-making (when to brake/shoot) simultaneously.

---

### 2. Asynchronous CPU-based Evaluation Worker

**Modified Files:**
- `BenchMARL/benchmarl/experiment/experiment.py` (lines 326-475, 537-568, 996-1023, 1054-1072)

**What Changed:**
Original BenchMARL blocked training during evaluation rollouts. We implemented a **persistent background evaluation process** that runs on CPU while training continues on GPU.

**Architecture:**
```
Main Process (GPU)              Evaluation Process (CPU)
    │                                    │
    ├─ Training Loop                     ├─ Initialize env once
    │   ├─ Collect data                  │
    │   ├─ Train models                  ├─ Wait for weights
    │   │                                 │
    │   └─ [Every N iters]               ├─ Receive new weights
    │       └─ Send weights ──queue──>   │
    │           (non-blocking)            ├─ Run rollouts
    │                                     │
    │                                     └─ Log results
```

**Benefits:**
- Zero training overhead (evaluation runs in parallel)
- Forced CPU evaluation prevents GPU memory issues
- Always evaluates latest weights (queue maxsize=1)

---

### 3. Custom Attention Model with Parameter Sharing Fix

**Modified Files:**
- `BenchMARL/benchmarl/models/attention.py`

**What Changed:**
Fixed critical bugs in the original Attention model related to parameter sharing and dimension handling for both centralised and decentralised settings.

**Key Fixes:**
1. **Parameter Sharing for Encoders**: When `share_params=False`, each agent now correctly gets independent encoder networks (lines 124-156)
2. **Dimension Handling**: Proper dimension manipulation for:
   - `input_has_agent_dim=True/False`
   - `output_has_agent_dim=True/False`
   - `centralised=True/False`
3. **Global State Input**: Added `_forward_global_input()` to handle centralised critic using global state (lines 368-480)

**Architecture:**
```
Input (entity features)
  → Entity Encoders (type-specific + ID embedding for grouped features)
  → Transformer Attention Layers (multi-head self-attention)
  → Feature Aggregation (ego embedding OR flatten all tokens)
  → Concat Global Features
  → Final MLP → Output
```

---

### 4. JIT-Compiled Reward Calculation

**Modified Files:**
- `VectorizedMultiAgentSimulator/vmas/scenarios/layup.py`
- `VectorizedMultiAgentSimulator/vmas/scenarios/layup_jit.py`

**What Changed:**
Moved all reward/termination logic from Python to a single JIT-compatible function (`calculate_rewards_and_dones_jit`) for 3-5x speedup.

**Performance:**
- Pre-computes interaction matrices (distances, collisions, velocity diffs)
- Fully vectorized operations (no Python loops)
- Compatible with `torch.jit.script` or `torch.compile`

---

## 🏀 Layup Game Rules (Detailed)

### Environment Setup
- **Field**: 8m (width) × 15m (length) rectangle
- **Teams**: 2v2 (2 attackers vs 2 defenders)
- **Objective**: Attackers score by shooting in the designated spot; Defenders prevent scoring
- **Time Limit**: 15 seconds per episode
- **Physics**: Holonomic dynamics with max velocity 5 m/s, max acceleration 3 m/s²

### Agent Roles

**Attacker 1 (A1)** - Ball Handler:
- Starts at bottom-left corner
- Must navigate to shooting spot (radius 1.2m, randomly positioned in opponent's half)
- Wins by making a successful shot

**Attacker 2 (A2)** - Screener:
- Starts in random position in own half
- Role: Set screens to free A1, disrupt defenders
- Shares team reward

**Defenders (D1, D2)**:
- Start in opponent's half (grid formation)
- Must prevent A1 from shooting or force timeout
- Can press but must avoid fouls
- **Cannot cross midline for more than 20 frames** (2 seconds)

### Shooting Mechanics

**Shooting Trigger Conditions** (all must be met for 10 consecutive frames):
1. A1 is inside shooting spot (`dist < 1.2m` AND `y > 0`)
2. A1 velocity `< 0.2 m/s` (configurable, uses curriculum learning)
3. A1 action magnitude `< 2.0` (not accelerating hard)
4. A1 is braking (in current implementation, brake is disabled but logic exists)

**Shot Success Determination:**
```python
# Block Factor Calculation (vectorized for all defenders)
shot_vector = basket_pos - a1_pos
blocker_vector = defender_pos - a1_pos

# 1. Defender must be between A1 and basket
proj_ratio = dot(blocker_vector, shot_vector) / ||shot_vector||²
is_between = (0 < proj_ratio < 1)

# 2. Defender must be close to shot line
perpendicular_dist = ||blocker_vector - proj_ratio * shot_vector||
line_blocking = exp(-perpendicular_dist² / (2 * 0.3²))

# 3. Defender must be close to A1
proximity_gate = sigmoid(25 * (threshold - dist_to_a1))

# Total block factor (clamped to [0, 1])
block_factor = line_blocking * is_between * proximity_gate
total_block = sum(all_defenders' block_factors)

# Shot succeeds if total_block < 0.5
shot_made = (total_block < 0.5)
```

### Termination Conditions & Rewards

#### Win Conditions (Attackers)

**Code 1: Shot Made** (`termination_reason=1`)
```python
# Attacker Rewards
A1_reward = base_score * (1 - block_factor)           # Up to 6000
           + time_bonus * (1 - block_factor)          # Up to 6000
           + spacing_bonus                            # Based on distance from defenders
           + velocity_stillness_bonus                 # Slower = better
           + shoot_score                              # Fixed 6000

A2_reward = base_score * (1 - block_factor)
           + screen_bonus                             # If positioned well
           + spacing_bonus * 3
           + time_bonus

# Defender Penalties
Defender_reward = block_contribution * 4000           # Small consolation
                + force_far_reward                     # If A1 shot from edge
                + positioning_reward
                + area_control_reward
                - 3000                                 # Base penalty for allowing shot
                + delay_bonus                          # Longer game = better
```

**Code 2: Opponent Foul** (`termination_reason=2`)
```python
# High-speed collision (rel_velocity > 0.5 m/s) or
# Defender touched A1 during shot preparation (reading frames > 0)

Attacker_reward = +dynamic_foul_magnitude            # 6000 + velocity_penalty
Defender_penalty = -dynamic_foul_magnitude
```

**Code 3: Opponent Wall Collision** (`termination_reason=3`)
```python
# Defender pushed into wall for 20+ consecutive frames
Attacker_reward = +standard
Defender_penalty = -11000
```

**Code 4: Opponent Overextension** (`termination_reason=4`)
```python
# Defender crossed midline (y < 0) for 20+ frames
Attacker_reward = +standard
Defender_penalty = -12000 (very harsh)
```

**Code 5: Opponent Friendly Fire** (`termination_reason=5`)
```python
# Defender collided with teammate at high speed
Attacker_reward = +standard
Both_defenders_penalty = -dynamic_foul_magnitude
```

#### Loss Conditions (Attackers)

**Code 11: Shot Blocked** (`termination_reason=11`)
```python
# Shot attempted but total_block_factor >= 0.5
Defender_reward = +block_contribution_reward + other_bonuses
Attacker_reward = per shot made, but very low due to high block factor
```

**Code 12: Attack Timeout** (`termination_reason=12`)
```python
# Time ran out (15 seconds expired)

# A1 in spot
if in_spot:
    A1_reward = +200 - velocity_penalty - action_penalty
else:
    A1_reward = -100 - 400 * distance_to_spot

# A2 penalty if still in own half (stalling)
if A2.y < 0:
    A2_penalty = -100 * |A2.y|  # Deeper = worse

Defender_reward = +9000 (huge success)
```

**Code 13: Team Foul** (`termination_reason=13`)
```python
# Attacker caused high-speed collision
Attacker_penalty = -dynamic_foul_magnitude
Defender_reward = +dynamic_foul_magnitude * 0.6  # Teammate bonus
```

**Code 14: Team Wall Collision** (`termination_reason=14`)
```python
# Attacker stuck on wall 20+ frames
Attacker_penalty = -11000
```

**Code 15: Team Friendly Fire** (`termination_reason=15`)
```python
# Attacker collided with A2
Both_attackers_penalty = -dynamic_foul_magnitude
```

### Dense Reward Shaping (Every Step)

All dense rewards are scaled by `dense_reward_factor=0.1` and timestep `0.005`.

**Universal Penalties** (all agents):
- Out-of-bounds soft penalty: `-3000 * smooth_penetration * (velocity + 1)`
- Action magnitude: `-0.1 * ||action||`
- Brake usage: `-0.1 * is_braking`
- Conflicting action: `-1.0 * ||action|| * is_braking`
- Action jerk: `-0.01 * ||action - prev_action||`
- Collision penalty: `-5.0 * rel_velocity` (active) or `-0.1 * rel_velocity` (passive)
- Proximity penalty: `-60 * smooth_penetration` when too close to others

**A1 Specific**:
- Gaussian attraction to spot: `+600 * exp(-dist²/0.6²)`
- Speed reward toward spot: `+k * velocity_projection` (k scales with initial distance)
- In-spot reward: `+3.0 * (1.5 - dist/R_spot)`
- Velocity stillness (in spot): `+20 * exp(-velocity²/0.4²)`
- Action stillness (in spot): `+10 * exp(-action²/0.3²)`
- Block penalty: `-70 * block_factor`
- Hesitation penalty (outside spot, moving slow): `-400 * (1 - speed/1.5)`
- Separation reward (when blocked): `+60 * speed_away_from_nearest_defender`
- Tangential movement (when pressured): `+120 * lateral_speed * pressure_gate`

**A2 Specific**:
- Screen position reward: `+200 * exp(-dist_to_ideal²) * position_gate * spacing_gate`
- Interference reward: `+400 * exp(-dist_to_defender²)`
- Repulsion reward: `+800 * defender_retreat_speed` (if A2 near defender)
- Shot line blocking penalty: `-90 * blocking_factor`
- **Crossing midline gate**: A2 only gets positive rewards if `y >= 0`

**Defenders**:
- Positioning reward: `+190 * exp(-dist_to_ideal_pos²) * behind_gate`
- Pressure reward: `+30 * (1 - dist_to_a1/range)²`
- Spot control: `+30 * (-a1_radial_velocity)` (slowing A1's approach)
- Gaussian spot attraction: `+30 * exp(-dist_to_spot²/R_spot²)`
- A1 penetration penalty: `-5.0 * a1_depth²`
- Overextension penalty: `-240 * max(0, -y)` (越过中线惩罚)

**Time Pressure** (after 8 second grace period):
- Attacker penalty: `-1.6 * (elapsed - 8)²` if A1 not in spot
- Defender bonus: `+0.1 * (elapsed - 8)²`

---

## Main Entry Point

**Primary training script**: [BenchMARL/clear_restore.py](BenchMARL/clear_restore.py)

This is the main script for training the layup scenario. It contains:
- Experiment configuration
- Model architecture setup (Attention + GRU sequence models)
- Ensemble algorithm configuration (separate configs for attackers and defenders)
- Custom callback for curriculum learning based on win rate
- Checkpoint loading utilities

### Running Training

```bash
# Run the main training script
cd /workspaces/robocon2025_marl_devcontainer/BenchMARL
python clear_restore.py
```

The script will:
1. Create a timestamped output folder in `outputs/YYYY-MM-DD_HH-MM-SS/`
2. Train attackers and defenders using MAPPO algorithm
3. Apply curriculum learning via `WinRateReportDebounced` callback
4. Save checkpoints during training
5. Run asynchronous evaluation on CPU in background

---

## Key Configuration Details

### Algorithm Configuration

The script uses **EnsembleAlgorithmConfig** with separate MAPPO configurations:

**Attacker**:
```python
share_param_actor = False   # Each attacker has its own policy
share_param_critic = True   # Shared value function
```

**Defender**:
```python
share_param_actor = True    # Defenders share policy parameters
share_param_critic = True   # Shared value function
```

### Model Architecture

**Attacker Model**: SequenceModel with Attention + GRU
- Attention layer: `benchmarl/conf/model/layers/attention_attacker.yaml`
- GRU layer for temporal processing
- Intermediate size: 128

**Defender Model**: SequenceModel with Attention + GRU
- Attention layer: `benchmarl/conf/model/layers/attention_defender.yaml`
- GRU layer for temporal processing
- Intermediate size: 128

**Critic Model**: Attention-based
- Config: `benchmarl/conf/model/layers/attention_critic.yaml`
- Uses global state instead of individual observations

### Experiment Configuration

Key hyperparameters from `base_experiment.yaml`:
- Learning rate: `0.00005`
- Gamma: `0.995`
- Max iterations: `4000`
- Gradient clipping: `2.0` (norm clipping)
- Training device: `cuda`
- Sampling device: `cpu`
- Buffer device: `cpu`

---

## Curriculum Learning System

The training uses a **win-rate-based curriculum** via `WinRateReportDebounced` callback:

**Mechanism**:
- If attacker win rate < 30%: Train only attackers (freeze defenders)
- If attacker win rate > 70%: Train only defenders (freeze attackers)
- If win rate between 42-58%: Train both groups normally

**Hysteresis/Debouncing**:
- Once entering "fix_attacker" mode, stays until win rate ≥ 58%
- Once entering "fix_defender" mode, stays until win rate ≤ 42%
- Prevents oscillation and provides stable training periods
- First 20 iterations always train both groups (warm-up period)

**Additional Curriculum**: Shot velocity threshold
- Training envs start with lenient threshold (1.2 m/s)
- Gradually decreases to target (0.2 m/s) as agents learn
- Eval envs always use target difficulty

---

## Common Development Tasks

### Training from Scratch

```bash
python BenchMARL/clear_restore.py
```

### Loading from Checkpoint

Uncomment and modify lines 339-350 in `clear_restore.py`:

```python
restore_file_path = find_latest_checkpoint(checkpoint_pattern)
experiment_config.restore_file = restore_file_path
```

The checkpoint utilities can find:
- Latest file in a directory: `find_latest_file(path, "*.pt")`
- Latest matching pattern: `find_latest_checkpoint("outputs/**/checkpoints/*.pt")`

### Partial Weight Loading

The script includes commented code (lines 417-463) for:
- Loading full model weights (actor + critic)
- Loading only actor weights (keeping fresh critic)
- Filtering state_dict by key prefix

### Monitoring Training

The callback system prints detailed statistics every batch:
- Termination reason breakdown with percentages
- Win rate calculation
- Current training mode (normal/fix_attacker/fix_defender)

### Using Health Check

Import and add `HealthCheckCallback` to monitor neural network health:

```python
from health_check import HealthCheckCallback

experiment = Experiment(
    ...,
    callbacks=[
        WinRateReportDebounced(),
        HealthCheckCallback(log_every_n_steps=50)
    ]
)
```

This monitors:
- Dead neurons (dormancy detection)
- Gradient flow issues
- GELU activation statistics
- Compatible with vmap-based models

---

## Performance Optimizations

The code enables TF32 for faster training on Ampere GPUs:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

JIT-compiled reward calculation provides 3-5x speedup over pure Python.

---

## Directory Structure

```
.
├── BenchMARL/                          # Forked & Modified
│   ├── clear_restore.py                # Main training entry point
│   ├── health_check.py                 # Neural network monitoring callback
│   ├── autoresume.sh                   # Auto-resume training script
│   ├── benchmarl/
│   │   ├── algorithms/
│   │   │   └── mappo.py                # ✨ Modified: Hybrid action support
│   │   ├── environments/layup/
│   │   │   └── common.py               # ✨ Modified: FlattenHybridAction transform
│   │   ├── models/
│   │   │   ├── attention.py            # ✨ Modified: Parameter sharing fixes
│   │   │   ├── gru.py                  # GRU model
│   │   │   └── mamba.py                # Mamba model (experimental)
│   │   ├── experiment/
│   │   │   └── experiment.py           # ✨ Modified: Async evaluation worker
│   │   ├── conf/
│   │   │   ├── experiment/base_experiment.yaml
│   │   │   ├── algorithm/mappo.yaml
│   │   │   └── model/layers/
│   │   │       ├── attention_attacker.yaml
│   │   │       ├── attention_defender.yaml
│   │   │       └── attention_critic.yaml
│   │   └── experiment/callback.py      # Base callback class
│   └── outputs/                        # Training outputs and checkpoints
│
├── VectorizedMultiAgentSimulator/      # Forked & Modified
│   └── vmas/scenarios/
│       ├── layup.py                    # ✨ Custom: Full game logic
│       └── layup_jit.py                # ✨ Custom: JIT reward calculation
│
└── .devcontainer/
    ├── Dockerfile                      # PyTorch 2.9.1 + CUDA 13.0
    └── devcontainer.json               # GPU-enabled container config
```

---

## Important Implementation Details

### Agent Grouping

The layup environment uses TorchRL's group-based MARL API:
- Group "attacker": Offensive agents
- Group "defender": Defensive agents
- Each group has independent loss modules, models, and training states
- Groups can be trained independently via `experiment.train_group_map`

### Ensemble Configuration

When using `EnsembleAlgorithmConfig` and `EnsembleModelConfig`:
- Each group can have different algorithm hyperparameters
- Each group can have different model architectures
- Parameter sharing is configured per-group

### Callback Integration

Custom callbacks can access:
- `self.experiment`: The Experiment instance
- `self.experiment.train_group_map`: Controls which groups are trained
- `self.experiment.n_iters_performed`: Current iteration count
- Batch data via `batch.get((group, key, ...))` in `on_batch_collected()`

### TensorDict Navigation

Batch data structure:
```python
# Access done flags
done = batch.get(("next", "done"))

# Access group-specific info
info = batch.get(("next", "attacker", "info", "termination_reason"))

# Shape: [time_steps, num_envs, num_agents, feature_dim]
```

---

## Git Submodules

Both BenchMARL and VMAS are **forked** git submodules with significant modifications. When making changes:

```bash
# Update submodules to latest
git submodule update --remote

# Check submodule status
git submodule status

# Changes in submodules are tracked in their own repositories
# The parent repo only tracks the commit hash of each submodule
```

**Important**: Do not merge upstream changes without careful review, as we have made breaking modifications to both libraries.

---

## GPU Configuration

DevContainer settings:
- Full GPU access with `--gpus=all`
- 8GB shared memory (`--shm-size=8gb`)
- CUDA 13.0 with cuDNN 9
- PyTorch 2.9.1

---

## Troubleshooting

### Checkpoint Loading Issues

The script includes `print_dict_paths()` utility to inspect checkpoint structure:
```python
checkpoint = torch.load(restore_file_path)
print_dict_paths(checkpoint)  # Shows all keys and types
```

### Win Rate Not Updating

Check that:
- Environments are actually terminating (`total_dones_in_batch > 0`)
- The `termination_reason` info is being set in the layup environment
- Win codes `{1,2,3,4,5}` match your environment's reward logic

### Model Architecture Mismatches

When loading checkpoints with different architectures:
- Use `strict=False` in `load_state_dict()`
- Filter state_dict by key prefix to load partial weights
- Check model layer names match between checkpoint and current model

### Hybrid Action Errors

If you see errors related to action specs:
- Ensure `FlattenHybridAction` transform is applied in `common.py`
- Check that `mappo.py` uses `_get_composite_policy()` for composite action specs
- Verify action spec in environment matches `{"continuous": ..., "discrete": ...}`

---

## Alternative Model Configurations

The script includes commented alternatives:

```python
# Mamba-based model (experimental)
# model_config = MambaConfig.get_from_yaml()

# Simple MLP baseline
# model_config = MlpConfig.get_from_yaml()
# critic_model_config = model_config
```

To switch models, comment/uncomment the relevant configuration blocks in `clear_restore.py`.
