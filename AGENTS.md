# AGENTS.md

**最后更新**: 2026-10-08（单组自对弈 + 混合动作 + 判罚重构版本）
**适用 commit**: 父仓库 `b1cf605` / BenchMARL `c9901f9` (branch `self-play`) / VMAS `921f83f` (branch `balance`)
**分支**: master

始终使用中文简体回复。

> ⚠️ **本文件是"给 AI 代理看的工作手册"，与 `README.md`（给人看的项目门面）分工不同**：
> README 讲"这是什么项目、怎么跑起来、历史数据"；AGENTS.md 讲"当前代码的真实状态、
> 容易踩的坑、必须遵守的工作纪律"。**做技术判断前先读本文件，数值细节以代码为准**
> （标注了文件:行；代码是唯一权威）。

---

## 0. 三句话现状

1. **任务**：RoboCon 2025 篮球上篮 2v2 —— A1 持球、A2 掩护、D1/D2 防守，抢分靠"投篮命中 / 造对手犯规"，
   用 MARL 让策略自己涌现出配合（挡拆、延误、造犯规、卡位）。
2. **技术栈**：BenchMARL（fork）+ VMAS（fork）子模块，**单组自对弈 MAPPO**
   （四方同组、共享主干、角色条件化 Attention + GRU），**混合动作空间**（2 连续速度 + 1 离散"按住读条"键）。
3. **当前进度**：v43e 在 20 秒/200 步的困难环境（R_spot=0.9m + 感知噪声）里，末窗胜率 ~76-78%、码 1（投篮命中）~55-60%；
   **训练已停，准备迁云端**（最新存档 `checkpoint_192000000.pt` = iter 640）。

---

## 1. 项目结构（当前）

```
.
├── AGENTS.md                       # 本文件（代理工作手册）
├── README.md                       # 项目门面（含评测 GIF、人机大战截图）
├── CLAUDE.md                       # 早期版本的同源说明，内容同 AGENTS.md 旧版，参考价值有限
├── docs/assets/                    # README 用图：eval_latest.gif / human_game.png / dashboard_codes.png
├── .opencode/                      # 全部运维/取证脚本（111 份，见 §9）
├── .devcontainer/                  # Dockerfile + devcontainer.json（--privileged / GPU / 8GB shm）
│
├── BenchMARL/                      # fork: CaptainKAZ/BenchMARL，branch self-play
│   ├── clear_restore.py            # ★ 唯一训练入口（配置 + 回调 + checkpoint 恢复）
│   ├── health_check.py             # 神经网络健康监控回调（可选）
│   ├── benchmarl/
│   │   ├── algorithms/mappo.py     # ★ 复合动作策略 / mask 注入 / 分布 / 损失
│   │   ├── models/attention.py     # ★ 角色条件化 Attention（RoleConditionalMLP / RoleFiLM）
│   │   ├── models/gru.py           # ★ cuDNN 融合 GRU + 角色条件化动作头
│   │   ├── models/common.py        # SequenceModelConfig / fp16→fp32 / cpu bf16 采样开关
│   │   ├── environments/layup/common.py   # ★ FlattenHybridAction / 动作掩码 / 单组 group_map
│   │   ├── environments/vmas/layup.py     # LayupTask（真正被加载的任务配置入口）
│   │   ├── experiment/experiment.py       # ★ 训练循环 / 重叠采集 / 评测 worker / 优化器 / 安全网
│   │   ├── threadlocal_state.py    # torchrl 进程级全局态 → threading.local（重叠采集安全）
│   │   ├── muon.py                 # Muon 优化器（USE_MUON=1 启用，默认 AdamW）
│   │   └── conf/                   # Hydra 配置（见 §8）
│   ├── liveview/                   # 实时观察窗（8765 面板：逐帧回放 + 终局码曲线 + 日志）
│   ├── human_game/                 # 人机大战（8766，浏览器里手动操控任意球员）
│   └── outputs/                    # 训练输出 / checkpoint / 评测视频
│
└── VectorizedMultiAgentSimulator/  # fork: CaptainKAZ/VectorizedMultiAgentSimulator，branch balance
    ├── AGENTS.md                   # ⚠️ 2026-03-06 的奖励详解，数值已过期（见 §12）
    └── vmas/
        ├── scenarios/layup.py      # ★ 游戏逻辑 / 观测 / 感知噪声 / 动作处理 / 渲染
        ├── scenarios/layup_jit.py  # ★ 奖励与判罚计算（普通函数，非 torch.jit）
        └── simulator/controllers/velocity_controller.py  # 速度 PID（3 通道动作只取前 2 维）
```

---

## 2. 快速开始

### 2.1 容器（rootless Docker，必须 root 身份运行）

```bash
# 服务器/机器重启后容器会退出（ExitCode 255），先起来：
docker start robocon2025-marl

# 从零重建（特权模式只能在创建时给，docker start 无法追加）：
bash .opencode/start_container.sh      # --privileged --gpus=all --shm-size=8gb + 依赖两级自愈
```

- 容器内工作目录 `/home/vscode/workspace/BenchMARL`；`/home/vscode/workspace` 就是 host 的仓库根（bind mount rw），**host 改文件容器立刻可见**。
- 容器是 **privileged** ⇒ 容器内有 root、可用 ptrace/py-spy 做性能取证。
- 关键：**host PID ≠ 容器内 PID**，用 py-spy 前先 `docker exec ... readlink -f /proc/$p/exe` 确认真是 `/opt/conda/bin/python3.11`。

### 2.2 训练（唯一的入口是 `clear_restore.py`）

```bash
# 冷启动（从零，走 VMAS_INITIAL_SHOT_THRESHOLD=1.2 的宽松课程）
docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  'LIVE_VIEW=1 OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 MALLOC_ARENA_MAX=2 \
   python clear_restore.py -m cold --max-iters 150 > outputs/train_vXX.log 2>&1'

# 续跑（-c 指定 checkpoint；阈值强制 0.2 = 困难环境）
docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  'LIVE_VIEW=1 OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 MALLOC_ARENA_MAX=2 \
   python clear_restore.py -m cont -c <ckpt.pt> --max-iters 700 > outputs/train_vXX.log 2>&1'
```

- **不要直接 `python clear_restore.py` 就完事**：先用 `.opencode/resume_v43f.sh` 这类脚本（含"已有训练在跑"守卫 + 启动校验打印）。
- **当前架构与所有旧 checkpoint 不兼容**（旧的是 attacker/defender 双组）：`-m cont` 会 mmap 预检，缺 `loss_agents` 直接报错要求 cold。
- 长任务一定要把日志镜像到 host 并打开：`ln -sf <log> /tmp/opencode/train_vXX.log && code /tmp/opencode/train_vXX.log`。

### 2.3 面板与人机大战

```bash
# 实时观察窗（8765）：逐帧回放 + 10 个终局码曲线 + 日志跟随
nohup bash BenchMARL/liveview/start_dashboard.sh &      # 浏览器 http://localhost:8765/
# 人机大战（8766）：加载最新权重，你手动操控任一球员
bash BenchMARL/human_game/start_game.sh                  # http://localhost:8766/
```

### 2.4 常用环境变量

| 变量 | 作用 |
|---|---|
| `OVERLAP_COLLECTION=1` | 采集与训练重叠（默认关；见 §7.1 的坑） |
| `SAMPLING_AUTOCAST_BF16=1` | CPU 采样走 bf16（采集 192→157 ms/步） |
| `LIVE_VIEW=1` | 打开 `LiveViewCallback`，写 `outputs/live/live_env.json` |
| `DELAY_CONSUME=1` | 诊断臂：串行采集 + 延迟一轮消费（隔离"1 次更新滞后"） |
| `ATTENTION_COMPILE_MODE=off` | 关 attention 的 torch.compile（改编译开关的 run 必须先剥离 `_orig_mod.` 键！） |
| `NAN_DEBUG=1` / `NAN_SAVE=1` | 非有限 loss 时打印/落盘现场（见 §7.3） |
| `USE_MUON=1` | 用 Muon 优化器取代 AdamW |
| `TL_STATE=0` | 关掉 threadlocal 补丁（仅在需要复现旧行为时） |
| `ACTOR_LR_MULT=<x>` | 给 `loss_objective` 单独乘 lr 倍率（用户已要求保持 1.0） |

---

## 3. 环境（vmas/layup）规格

> 权威代码：`VectorizedMultiAgentSimulator/vmas/scenarios/layup.py`（逻辑/观测/动作/渲染）
> 与 `.../layup_jit.py`（奖励与判罚，**普通函数，不是 torch.jit**）。

### 3.1 场地与时间

| 项 | 值 |
|---|---|
| 场地 | 8 m（宽 x）× 15 m（长 y）；中线 y=0；篮筐在 +y 端（半径 0.1 m） |
| 单局 | 20 s = 200 步（`dt=0.1`，`max_steps=200`） |
| 物理 | holonomic，`v_max=5 m/s`、`a_max=3 m/s²`，球员半径 0.3 m |
| 投篮点 | **R_spot = 0.9 m 固定**，位置每局随机（必须在对方半场 y>0）；"圈内"= `dist ≤ R 且 y>0` |
| 初始位置 | `fixed_init: false` ⇒ A2/D1/D2 随机（A1 仍固定 (−3.40,−6.90) 一带，以代码为准） |

### 3.2 动作空间（混合 + 掩码）★ 这是本项目最容易被误解的地方

- A1：**3 通道** —— 前 2 维连续（归一化速度指令 ∈(-1,1)）+ 第 3 维离散"按住读条"键（0/1）。
- A2/D1/D2：网络**只输出 2 维**，由动作适配层补 0 到 3 维，环境侧**忽略第 3 通道**。
- **动作掩码**：`get_action_mask()` 返回 `[batch, n_agents, 2]` bool —— "不按"恒允许，"按下"仅当 **A1 在圈内**；
  在 mappo 里以 `logits.masked_fill(~mask, -1e9)` 注入（非法动作概率≈0）。
- **按键语义**：按住 = 刹车蓄力（期望速度置 0，由速度 PID 保持 v≈0，**不直接改 `state.vel`**）；
  **读条 10 帧满自动出手**；松手/出圈/不满足静止条件 → 读条清零。
- **10 帧读条不可降低**（模拟机械结构准备过程，防守方可在读条期补救）—— 这是用户硬约束。
- 动作是线性映射（旧版的 1.5 次方整形与 0.2 死区**已删除**，因为整形会破坏读条静止条件）。
- ⚠️ 速度上限：连续动作经 tanh 落在 (-1,1)，`process_action` 里**必须乘 `v_max` 还原**成 ±5 m/s
  （历史上漏乘导致"策略只能跑 1 m/s、4 秒内零过中线"）。

### 3.3 观测与全局状态

- **观测 41 维**（每个 agent 一份，权威布局见 `layup.py` 的 `observation()`）：
  `self[0:9]`（pos/[4,7.5] ×2、vel/5 ×2、上帧原始动作/5 ×2、is_in_spot、读条进度、剩余时间）
  `+ 队友[9:17] + 对手1[17:25] + 对手2[25:33]`（每个 8 维：rel_pos/[8,15]、rel_vel/10、abs_pos/[4,7.5]、abs_vel/5）
  `+ 投篮点[33:37] + 篮筐[37:41]`。
- **全局 state 23 维**：只给 critic（特权信息），不由网络看到。
- **感知噪声**（`_refresh_perception`，在 `post_step`/`reset_world_at` 刷新）：
  `σ_pos = k_perception_noise(0.02)·d + floor(0.05)`、`σ_vel = k_perception_noise_vel(0.01)·d + floor(0.02)`；
  **同一份含噪量同时用于相对与绝对量**（避免自相矛盾）；本体感受、投篮点、篮筐、全局 state **不加噪**；
  实测开销 0.91 ms/步（0.18 s/轮，可忽略）。
- 观测/state 以 **fp16 存储**，模型入口转 fp32（`models/{mlp,attention,gru}.py`）。

### 3.4 终局码（权威表）★ 面板/统计/分析必须严格用这套码

| 码 | 含义 | 归属 | 说明 |
|---|---|---|---|
| **1** | 投篮命中 | 攻方胜 | 读条满且被封盖系数判定为进 |
| 11 | 投篮被盖 | 攻方负 | 读条满但被封盖 |
| 12 | 进攻超时 | 攻方负 | 200 步未出手 |
| **2** | 对手犯规 | 攻方胜 | 读条期被防守者触碰，或防守者被判主动犯规 |
| 13 | 己方犯规 | 攻方负 | A1/A2 被判主动犯规 |
| **3** | 对手失误-撞墙 | 攻方胜 | 防守者撞墙（首碰 >0.5 m/s 立判） |
| 14 | 己方失误-撞墙 | 攻方负 | 攻方撞墙 |
| **4** | 对手失误-越线 | 攻方胜 | 防守者越过中线 >5 帧 |
| **5** | 对手友军误伤 | 攻方胜 | 防守者撞自己队友 |
| 15 | 己方友军误伤 | 攻方负 | 攻方撞自己队友 |

- `WIN_CODES = {1,2,3,4,5}` ⇒ **胜率 = 这 5 个码的占比**（注意 11/12 都是"输"，别只看码 1）。
- `termination_reason` 是**全局广播**：batch 里 `("next","agents","info","termination_reason")` 四个 agent 值相同，
  取 `[...,0,:]` 即可；**该键不在 replay buffer 里**（被 `_get_excluded_keys` 排除）⇒ 事后分析要么查日志块，要么重跑 rollout。

### 3.5 奖励结构（缩放：稠密 ×0.0005，终局 ×0.005）

- **命中/终局**：A1 ≈ +95~+133（含 shoot_score/base/time）；被盖 `k_blocked_shot_penalty=12000` ⇒ A1 −60、
  A2 `k_blocked_shot_penalty_a2=18000` ⇒ −90（用户要求"A1 少扣、A2 多扣"）。
- **造犯规**：`R_foul=8000` + `k_foul_vel_penalty=800`×相对速度 ⇒ 被撞方 +40~54；主动方 −40~54；
  攻方主动犯规额外 ×`k_attacker_active_foul_scale=1.5`；主动方的队友按 `k_teammate_foul_share=0.3` 分摊；
  守方被犯规另有 `defender_fouled_bonus=2500`。造犯规奖金 `k_foul_drawing_bonus` 需自身速度 <0.3 m/s（站定）。
- **防守延误**：`k_def_delay_bonus=14000`，`R_delay = k · r^1.5`（r = 剩余时间比例），
  并加 `k_def_delay_floor=0.5` **保底**（"延误永远好过不延误"）；出手放投罚不再被延误完全对冲：
  `shot_penalty = 9000 · (1 − 0.10·contest)`。
- **A1 进攻引导**：投篮点速度奖励（比例于初始距离归一化）、高斯吸引（σ=0.5R）、圈内驻留、
  被遮挡横移残量 `k_a1_tangential_reward=150`（乘前进方向归一化，原地摇摆=0）、
  走廊清空 `k_a1_lane_clear_reward=600`、读条进度奖励 + 中断惩罚、时间压力（8 秒后递增惩罚）。
- **A2 掩护（结果导向）**：`k_a2_lane_clear_reward=400`、`k_a2_body_check=300`、
  `k_ideal_screen_pos=200`、`spread/support/immediate_draw` 三通道（`k_a2_draw_immediate=300`）、
  硬门控 `between_gate`、唯一反制项 `k_a2_shot_line_penalty=90`（**偏弱，见 §11.1**）。
- **防守**：越线惩罚 `k_overextend_penalty`、站位/贴防/延缓奖励、被逼超时奖。
- **超时**：圈内 −25 / 圈外 −50；A2 停滞 `k_a2_stalling_penalty_timeup`。

### 3.6 判罚规则（2026-10 重构版）★ 判罚直接决定策略走向

1. **读条期保护（A1 专属）**：只要 A1 在读条，任何**碰到 A1 的防守者**单独判防守犯规（码 2），**不设速度门槛**；
   实现用逐防守者掩码（`collision_matrix[:, 0, n_attackers:]`，`layup_jit.py` 约 :300-334）。
   **读条期触碰绝对不允许判给攻方**（用户硬约束）。
2. **普通碰撞判责 = 对称"接近分量"**：`approach_i = clamp(v_i·n)`、`approach_j = clamp(-v_j·n)`，
   大者为主动方；平手用整体速度决胜；门槛 `foul_approach_threshold=0.35`（低于门槛算"擦身"，不判）。
   旧版按"速度方向 XOR"判责会把正常突破判成进攻犯规（终局碰撞里 86% 是擦身）⇒ 进攻期望变负 ⇒ 策略不进攻。
3. **篮球式豁免**：`violation_D = clamp(|v_D|−0.4) + clamp(approach_D−0.4)`；
   若 `approach_atk − k_legal_def_exempt(1.2)·violation_D ≤ approach_def` ⇒ **责任归防守**（码 2）。
   即"防守方移动中碰撞 = 防守犯规，站稳被撞 = 进攻犯规"。
4. **越线**：防守者越过中线累计 **>5 帧** ⇒ 码 4（原 20 帧）。
5. **撞墙**：**首碰且速度 > `v_wall_crash_threshold`(0.5 m/s) 立即判负**（原"连续 20 帧"规则保留为兜底）。
6. **友军误伤**：同队高速相撞，罚则 ×`k_friendly_fire_scale=2.0`。

---

## 4. 算法与模型

### 4.1 算法：单组自对弈 MAPPO

- **一个组** `agents`（四方同组，`MarlGroupMapType.ALL_IN_ONE_GROUP`），一套 loss、一套优化器。
- `MappoConfig(share_param_actor=True, share_param_critic=False)`，`clip_epsilon=0.2`、`lmbda=0.95`、
  `gamma=0.995`、`entropy_coef=0.01`、`critic_coef=1.0`、`normalize_advantage=True`
  （`normalize_advantage_exclude_dims=[-2]`，保护最后一维不做标准化）。
- 课程学习**已彻底删除**（不再有 fix_attacker/fix_defender 的 58%/42% 迟滞）。
  历史教训：冻结一方时另一方会去"薅固定对手的羊毛"，冻结期=技能退化期。

### 4.2 网络：角色条件化 Attention + GRU

- **actor**：`SequenceModelConfig([Attention(attention_agents.yaml), GRU(gru.yaml)], intermediate_sizes=[256])`。
  - Attention 主干**四方共享**，`role_ids=[0,1,2,2]`、`n_roles=3`、`role_embedding_dim=32`。
  - 角色注入两条路径：① 角色嵌入投影后加到 ego token；② `RoleFiLM` 在**每个 attention block** 输出上做
    `x·(1+scale_role)+shift_role`（`to_film` 零初始化 ⇒ 训练初期恒等，不破坏既有行为）。
  - 队友/对手分离：`encoder_groups.players` 里拆出 `opponent1_embed` / `opponent2_embed`（八维特征各自编码），
    解决"分不清两个防守者"的老问题。
  - 输出头：`RoleConditionedMLP`（共享 trunk + **每角色一个 2 层头**），
    `role_out_dims=[6,4,4]`（A1 = 4 连续参数 + 2 离散 logits；A2/D 只 4 参数，补 0 到 6）。
  - GRU 之后**动作参数头也角色条件化**（`gru.py` 同样接 `RoleConditionedMLP`）。
- **critic**：独立网络（Attention，输入 23 维全局 state，`share_params_override: True` 一次前向出 4 个价值）。
  actor/critic **不共享主干**：输入模态不同（per-agent obs vs 全局 state）+ 需要单优化器/一次反传（否则主干双步更新）。
- **分布**：连续位 `SafeTanhNormal` + `BoundedNormalParamExtractor`
  （`loc = loc_bound·tanh(raw/loc_bound)`，`loc_bound=3.0`；`scale = 0.01+(1.0−0.01)·sigmoid(raw)`)；
  `SafeTanhNormal.log_prob` 前把 value clamp 到边界内侧 0.1%（根治 `0×∞ → NaN`）；
  离散位 `Categorical` + `logits.masked_fill(~mask, -1e9)`。
- **梯度冲突监控**：`clear_restore.py` 的 `GradConflictCallback`（每 10 iter）打印共享主干上
  **攻方/守方梯度余弦相似度**与 per-role 头权重范数。历史读数 cos ≈ −0.12 ~ +0.02（冲突不重）。

### 4.3 参数规模（参考）

| 部件 | 规模 |
|---|---|
| 单组 actor（Attention+GRU+角色头） | ≈ 1.8-2.3 M |
| critic | ≈ 1.4 M |
| MLP 时代（对照，已废弃） | ≈ 1.7 M |

---

## 5. 性能事实（都是实测，别重复踩）

- 稳态 **~55-75 s/轮**（1500 环境 × 200 步 = 30 万帧/轮）。**GPU 功耗被 Afterburner 限过，别拿耗时跟历史比**。
- **采集与训练大致各占一半**；重叠（`OVERLAP_COLLECTION=1`）时 `iter ≈ max(采集, 训练)`。
- 采集耗时里 **~90% 是 CPU 策略前向**；`SAMPLING_AUTOCAST_BF16=1` 能把采集从 192 ms/步降到 157 ms/步（1.22×）。
- **CUDA 采样比 CPU 慢 ~1.75×**（历史实测）⇒ `sampling_device: cpu`。
- py-spy 火焰图分类：collector 线程 ~52%（大头是 `nn.Linear.forward`）、主训练线程 ~36%
  （`_optimizer_loop` ~27%）、其他 ~12%（含面板 socket accept ~6%；LiveView 回调仅 0.5%，可忽略）。
- **评测 worker 的视频渲染是最大的额外开销**（评测期间抢 CPU，iter 从 65→86 s 都发生过）；
  关评测/降分辨率能显著提速，但用户要看视频 ⇒ 保留。
- `on_policy_minibatch_size=12000` 是实测最优（opt_loops 57→41 s，−25%；20000 会 OOM）。
  ⚠️ 恢复 checkpoint 会带回旧的 buffer batch_size ⇒ 已加 `_configured_buffer_batch_size` 纠正（日志打 `[BufferGuard]`）。
- **已判死、别重试**：VMAS 侧 `torch.compile` / `torch.jit.script`（奖励 87× 慢、env.step 24× 慢、
  `torch.jit.script` 报 `Tensor cannot be used as a tuple`）；CUDA 采样；fork 多进程采集
  （fork 子进程一碰 CUDA 就 `Cannot re-initialize CUDA in forked subprocess`）；
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（已被 torch 标 deprecated 且实测无收益）。
- WSL 显存：训练峰值 ~11-12 GB / 12 GB 上限；**顶破后会走系统内存 ⇒ 速度骤降**（曾把 iter 拉到 100 s）。
  现实做法：别与训练并发跑重脚本（见 §10）。

---

## 6. 实验历史与数据（胜率 / 码 1 命中率，均为训练日志批窗口统计）

| 版本 | 配置要点 | 末窗胜率 | 末窗码 1 | 备注 |
|---|---|---|---|---|
| v26 | Attention+GRU 双组冷启动 | 55.4% | 27.4% | 窗口峰值码 1 达 40% |
| v32 | cont from v28 | 62.6% | 46.7% | 交叉对打攻守两端领先 v28/v29 |
| v36 | **单组化 + 角色条件化** cold | 78.8% | 70.5% | 早期墙/越线主导 → 后转投篮 |
| v39 | 加感知噪声 | — | — | **崩坏**：非有限 loss 967 次、策略冻结 |
| v41 | R_spot 0.9 的困难环境 | 36.2% | 12.2% | 难度上来后胜率回落（预期） |
| v43b | 判罚 A+（旧参数）+ 感知噪声 | 79.6% | 52.5% | 码 2 26.6%、码 13 10.2% |
| v43e | **判罚再放宽（0.4/0.4/1.2）** | 76-78% | **55-60%** | 码 2 降到 17.5%（相对 −34%），攻方改走"投进"路线 |

**判罚经济学**（iter 570 缓存取证）：码 13 里"被撞者"实得 ≈ +54.5 > 逼超时 +45 > 正常出手 ≈ −45~+26
⇒ 旧判罚下"站着等被撞"比进攻还赚；这正是 v38 时代 A1 退缩（对方半场占比掉到 21.7%）的根因。
终局碰撞帧里 **86.4% 是擦身**（approach 中位 0.19）⇒ 已用 `foul_approach_threshold=0.35` + 篮球式豁免修正。

**交叉对打方法学**（`.opencode/crossplay*.py`）：同一 Experiment 内换权重、统计 `('next','done')` 与
`('next','agents','info','termination_reason')`；每臂 ~1500-9000 局，1σ≈1.2-1.6pp；
必须设 `VMAS_INITIAL_SHOT_THRESHOLD=0.2`（否则评测环境跑阈值课程、给早期 cell 送温暖）；
旧代 checkpoint 动作维度不同时要把动作叶子换成 2 维 `Bounded` 并钉住 `loc_bound`。

---

## 7. 训练工程与踩坑（每条都是真金白银换来的）

### 7.1 重叠采集是数据污染源
`_CollectionPrefetcher` 与主线程并发时，torchrl 的**进程级全局态**
（`_interaction_type` / `_skip_existing` / `recurrent_mode_state_manager`）会被两个线程互相改写，
症状是"重叠臂数据系统性码 1 虚高、码 12 偏低 ⇒ 胜率虚高、误触发课程"。
修法：`benchmarl/threadlocal_state.py` 把这三个态换成 `threading.local`（`TL_STATE=0` 可关），
并在 `experiment.py` 开头安装钩子。另外 `trigger_next()` 在 `policy.train()` 之前 ⇒ 预取批天然滞后 1 次更新
（用 `DELAY_CONSUME=1` 验证过：**纯滞后不会崩坏**，所以病毒是并发而不是滞后）。

### 7.2 BufferGuard：跨配置恢复会 CUDA 越界
checkpoint 里的 buffer 形状与当前 `n_envs/frames_per_batch` 不匹配时，`load_state_dict` 会写越界
（`device-side assert triggered` / `IndexKernel.cu:111`）。
修法：`_buffer_state_is_compatible` 检查形状，不兼容就打印 `[BufferGuard]` 并**跳过 buffer 恢复**
（on-policy 缓冲可丢）。

### 7.3 NaN / 崩坏治理
- `_optimizer_loop` 会跳过非有限 loss / 梯度并打印 `[SafeTrain]`（累计计数）。
- `NAN_DEBUG=1` 打印 subdata 非有限叶子；`NAN_SAVE=1` 落盘现场；`_dump_nonfinite_grads` 打印梯度统计。
- **已知现象**：坏模式一旦闭环会自持（lr=0 冻结回放也能复现），表现是动作全饱和 ±1、
  墙碰撞率冲到 88%、回合奖励 −68；**回滚到好档可以跳出**（所以 checkpoint 要密、要留档）。
- 排查经验：**NaN 不在持久化数据里**（离线扫 checkpoint 全 finite）⇒ 产生于训练时的 loss 计算。

### 7.4 两个静默失效的框架 bug（历史全量 run 都受影响）
- `entropy_coeff` / `critic_coeff` **双 f 拼写**：torchrl 0.8.1 的 `ClipPPOLoss` 只认单 f，
  且 `PPOLoss.__init__` 不转发多余 kwargs ⇒ 配置被静默丢弃，实际一直用 0.01/1.0。已修 + 启动打印 `[LossCoef]`。
- **优化器重复参数**：actor 参数同时进了 `loss_objective` 与 `loss_critic`，每步更新两次（等效 lr×2）。
  已按 `id(p)` 去重（`[Optimizer] agents: 参数去重 204 -> 103`）。

### 7.5 改编译开关的 run 必须先处理 `_orig_mod.`
`torch.compile` 会把模块包成 `OptimizedModule`，state_dict 键带 `_orig_mod.`。
关编译加载会**静默丢弃**这些键（只留 warning）⇒ 策略权重没加载、训练看起来"完全没学会"。
现成工具：`.opencode/make_eager_ckpt.py`（剥离键名）。

### 7.6 checkpoint 策略
`checkpoint_interval` 现为 **1.5M 帧 = 每 5 轮**（原来 3M，因 GPU 偶发崩溃改为更密）；
`keep_checkpoints_num: 10`。存档路径 `<run_dir>/<exp_name>/checkpoints/checkpoint_<frames>.pt`，
其中 `frames = iter × 300000`（iter 640 ⇒ `checkpoint_192000000.pt`）。

### 7.7 GPU 偶发故障（已发生 ≥4 次）
症状：`torch.AcceleratorError: CUDA error: unknown error`（栈底常在
`adam.py::_multi_tensor_adam` → `torch._foreach_lerp_`，或 `experiment.py::_optimizer_loop`），
容器 `ExitCode=255`、`OOMKilled=false`。与"探针抢内存"叠加时会更容易触发（**所以 §10 禁止并发**）。
处置：`docker start` → `.opencode/resume_v43X.sh` 从最近存档续跑（几乎零损失）。

---

## 8. 配置速查表（写代码/调参前先看这里，改完记得同步文档）

`BenchMARL/benchmarl/conf/experiment/base_experiment.yaml`

| 键 | 值 | 说明 |
|---|---|---|
| `sampling_device` / `train_device` / `buffer_device` | `cpu` / `cuda` / `cpu` | CPU 采样比 CUDA 快 |
| `gamma` / `lr` / `clip_grad_val` | 0.995 / 5e-05 / 2.0 | |
| `max_n_iters` | 4000 | 实际由 `--max-iters` 决定 |
| `on_policy_collected_frames_per_batch` | 300000 | = 1500 env × 200 步 |
| `on_policy_n_envs_per_worker` | 1500 | |
| `on_policy_n_minibatch_iters` / `on_policy_minibatch_size` | 10 / 12000 | 12000 是实测最优 |
| `evaluation` / `evaluation_interval` / `evaluation_episodes` | True / 4500000 / 8 | 异步 CPU 评测 + 出视频 |
| `checkpoint_interval` / `keep_checkpoints_num` | 1500000 / 10 | 每 5 轮存档（抗崩） |
| `loggers` | `[csv]` | tensorboard 已关（省内存） |

`BenchMARL/benchmarl/conf/algorithm/mappo.yaml`：`clip_epsilon=0.2`、`entropy_coef=0.01`、
`critic_coef=1.0`（**单 f，别写错**）、`lmbda=0.95`、`normalize_advantage=True` +
`normalize_advantage_exclude_dims=[-2]`、`bounded_tanh_params=True`、`loc_bound=3.0`、
`scale_min=0.01`、`scale_max=1.0`。

`BenchMARL/benchmarl/conf/task/vmas/layup.yaml`：`max_steps=200`、`fixed_init=false`、
`fixed_spot=false`、`history_frames=0`（改用 GRU 后不再堆帧）、
`k_perception_noise=0.02` + floor 0.05、`k_perception_noise_vel=0.01` + floor 0.02。

`conf/model/layers/`：`attention_agents.yaml`（actor：`role_ids=[0,1,2,2]`、`n_roles=3`、
`role_embedding_dim=32`、`ego_pool=mean`、embedding 128 / 4 头 / 2 层）、
`gru.yaml`（`hidden_size=256`、`role_out_dims=[6,4,4]`、`compile: False`）、
`attention_critic.yaml`（critic，`share_params_override: True`）。

---

## 9. 工具链（`BenchMARL/liveview`、`BenchMARL/human_game`、`.opencode/`）

### 9.1 实时观察窗（8765）
- `liveview/live_view.py`：`LiveViewCallback` 从 batch 里切**完整回合**（`is_init` 起点 → 其后第一个 `next.done`），
  **严格使用真实终局码**（`CODE_NAMES` / `WIN_CODES`），写 `outputs/live/live_env.json`；
  `--from-ckpt` 走 `offline_from_ckpt()`（临时建 Experiment、真实 rollout 取码，比 buffer 可靠）。
- `liveview/live_view.html`：单页三格 —— 逐帧回放（含读条条、遮挡黑圆、终局肇事红圈）+ 10 条终局码曲线 + 日志跟随。
- `liveview/serve_dashboard.py` + `start_dashboard.sh`：8765 面板（自读日志、`/curve.json`）。
- `liveview/live_html_check.py`：无头浏览器验收（22 项）。
- 坑：顶层 `let` 变量不挂 `window`（探针要直接用裸标识符）；改 HTML 后浏览器要 Ctrl+F5。

### 9.2 人机大战（8766）
`human_game/game_server.py`（+ `game.html`、`tcp_relay.py`、`start_game.sh`）：浏览器里操控任一球员与策略对打/合作；
按住 = 跟随/吸引，A1 松手 = 原地刹车蓄力，读条 10/10 自动出手；`/api/meta` 会回报环境参数。

### 9.3 `.opencode/` 脚本（按用途）

| 类别 | 脚本 |
|---|---|
| 容器/训练 | `start_container.sh`、`resume_v4*.sh`、`watch_v4*.sh` |
| **取证（下结论前必跑）** | `forensics_p0.py`（buffer+advantage 深析）、`probe_contact_econ.py`（接触经济学/终局码分布）、`probe_foul_rule.py`（合成判罚案例）、`probe_buf_codes.py`、`parse_logs.py`/`cmp_blocks.py`（日志块统计） |
| 性能 | `spy_analyze.py`（py-spy 火焰图按线程聚合）、`speed_arm4.py` + `run_speed4.sh` + `poll_speed4.sh`、`probe_noise_cost2.py` |
| checkpoint/崩溃 | `nan_scan_ckpt.py`、`make_eager_ckpt.py`（剥 `_orig_mod.`）、`probe_weight_snapshot.py` |
| 交叉对打 | `crossplay.py`、`crossplay_v3.py`（+ `render_matrix.py`、`poll_xp.sh`） |
| 面板验收 | `run_live_checks.sh`、`check_curve_codes.py`、`check_curve_logic.js` |
| 视频/截图 | `mp4_to_gif.py`、`mp4s_to_gif.py`、`snap_end_frame.py` |

---

## 10. 工作纪律（用户硬约束 + 团队协议，**违反等于返工**）

1. **始终中文简体**。
2. **下结论前必须做 advantage + replay buffer 的深度取证**（`forensics_p0.py` 之类），
   不许只看日志统计块就下结论（历史上因此错过"A1 退缩"的真因）。
3. **分析类探针绝对不许与训练并发**：曾经因为探针抢内存把训练挤崩（用户明确批评过）。
   探针一律 `nice -n 19 OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2`，且**先确认没有训练在跑**。
4. **10 帧读条不可降低**；**读条期触碰不允许判攻方**；**投篮点位置必须随机**（半径固定 0.9 m）；
   **batch 不能太小**（阻碍探索）；**贴身防守的负期望最多只能软化 ~10%**。
5. **未明确要求不要启动训练**；要跑之前先说明"跑什么、多久、看什么"，被授权无人值守看护时才可自动续跑。
6. **长任务日志必须镜像到 host 并打开**：`/tmp/opencode/train_vXX.log` + `code`。
7. **轮询一律用循环脚本**（写 `poll_*.sh` 放后台，别反复发 tool call），单次 sleep 不超过 30 秒。
8. **脚本放项目 `.opencode/`**（可复用），一次性产物放 `/tmp/opencode/`。
9. **面板与统计严格按终局码颜色/名称**，禁止"其它/未知"这种糊弄输出。
10. GPU 功耗被限过、机器会偶发重启 ⇒ **耗时只跟同环境同期数据比**；容器掉了先 `docker start`。

---

## 11. 已知问题与开放问题

### 11.1 A2"先占住封盖位、A1 站在其身后出手"的必胜套路（**用户提出，尚未修**）
- 现象：A2 卡在防守者与 A1 之间的出手线路上，防守者被规则和物理同时锁死 ⇒ 攻方近乎无解。
- 代码原因（`layup_jit.py` 约 :860-980）：A2 占位收益 `lane_clear 400` + `body_check 300` +
  `ideal_screen_pos 200` + `draw_immediate 300`，而唯一反制 `shot_line_penalty=90`（上限 −0.045/步）⇒ **净赚**；
  且 `block_factor_a1` **只统计防守者位置**（队友挡视线不减命中）、读条期触碰无门槛、防守者绕不过 A2。
- 候选修法：**F1** 提高挡视线罚（90→3000~4000、垂距 σ 0.15→0.35~0.45 m、距 A1 σ 0.6→1.2 m）；
  **F2** 命中判定计入队友遮挡 / 判非法掩护（治根）；**F3** 给守方活路（`def_proximity_threshold` 0.9→1.2、
  `block_gate_k` 25→15）；**F4** 收紧 A2 的 `lane_clear` 权重。**（建议 F1+F3，等用户拍板）**

### 11.2 其他开放项
- **感知噪声范围未定**：投篮点/篮筐是否加噪？A1 本体感受是否加噪？
- **贴身防守的期望仍未真正为负**（D 贴 A1<1.2 m 时 advantage 仍 +0.15）。
- **训练伙伴问题**：v37 攻 vs v38 守能刷到 94% 平坦 —— 不是轮数不够，是"防守没见过的风格"；
  长期方向是**历史对手池 / league**（用户尚未选）。
- **崩坏治理**：需要训练内自动检测 + 自动回滚（现在靠人看日志 + 回滚好档）。
- **checkpoint 保留策略**：是否只留最近 2~3 档？MLP 时代（v14/v15）存档是否清理？
- **迁云端**：起点 `checkpoint_192000000.pt`(iter 640)，需要迁移清单（依赖、启动命令、要带走的文件）。
- **GPU 偶发 `unknown error`**：本机 WSL 环境问题，云端应重测。

---

## 12. 其他文档的状态（别被旧文档误导）

| 文件 | 状态 |
|---|---|
| `AGENTS.md`（本文件） | ✅ 2026-10-08 重写，反映当前架构 |
| `README.md` | ✅ 项目门面（含评测 GIF、人机大战截图、实验数据表） |
| `VectorizedMultiAgentSimulator/AGENTS.md` | ✅ 2026-10-08 重写为**当前实现**的奖励/判罚系统详解：缩放体系、读条与封盖公式、终局码权威表、判罚六条与发放账本、稠密项逐条公式+系数、感知噪声、调参索引（键→行号）、历史沿革与已知漏洞。查数值仍以代码为准，但本文已与代码对齐 |
| `CLAUDE.md` | ✅ 已收拢为指向本文件与 `README.md` 的简短指针（旧内容已删） |
| `BenchMARL/`、`VMAS/` 内部 README | 上游原版，未改 |

---

## 13. 排障速查

| 症状 | 处置 |
|---|---|
| 容器不在了 / `Exited (255)` | `docker start robocon2025-marl`；训练从最近存档 `-m cont -c <ckpt>` 续 |
| `-m cont` 报 checkpoint 缺 `loss_agents` | 旧双组架构存档，不兼容 ⇒ 只能 cold |
| 训练"完全没学会"（码 1 恒 0） | 检查是否改了编译开关而没剥 `_orig_mod.` 键 / 动作是否漏乘 `v_max` |
| 出现大量 `[SafeTrain]` | 崩坏前兆；看 `[NanCase]` 现场，必要时回滚到上一个好档 |
| 训练变慢（iter >90 s） | 查是否与探针/评测并发、显存是否顶破 12 GB（走系统内存） |
| `git push` 连不上 GitHub | 本机 DNS 曾把 `github.com` 解析到不可达 IP；开代理，或
`git -c http.curloptResolve=github.com:443:140.82.112.3 push ...`（已验证可用） |
