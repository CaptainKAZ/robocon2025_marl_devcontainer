# CLAUDE.md

> 本文件原本是项目说明的旧副本（2026-03 内容，描述的是 attacker/defender 双组 + 胜率课程 + MLP 时代）。
> **该内容已全部过期**，为避免误导已清空，请以以下两份文档为准：

| 文档 | 用途 |
|---|---|
| **`AGENTS.md`** | ⭐ 给 AI 代理的工作手册：当前架构、环境/动作/观测/终局码/奖励/判罚的权威说明、配置速查、踩坑记录、工具链、工程纪律、开放问题 |
| `README.md` | 项目门面（给人看）：项目介绍、快速开始、环境设计、实验历史与数据、工具链、FAQ |

**查数值（奖励权重、判罚阈值、超参数）一律以代码为准**：

- `VectorizedMultiAgentSimulator/vmas/scenarios/layup.py` —— 场景逻辑、`h_params`、观测、感知噪声、动作处理、渲染
- `VectorizedMultiAgentSimulator/vmas/scenarios/layup_jit.py` —— 奖励与判罚计算（读条、封盖、犯规判责、终局码）
- `BenchMARL/clear_restore.py` —— 训练入口与实验配置
- `BenchMARL/benchmarl/conf/**` —— Hydra 配置（experiment / algorithm / task / model layers）

📘 另推荐：`VectorizedMultiAgentSimulator/AGENTS.md` —— **当前实现的奖励/判罚系统详解**
（2026-10-08 按代码重写）：缩放体系、读条与封盖公式、终局码权威表、判罚六条与发放账本、
稠密奖励逐条公式与系数、感知噪声、调参索引（键 → `layup.py` 行号）、历史沿革与已知漏洞。
