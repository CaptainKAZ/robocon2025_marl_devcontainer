#!/bin/bash
# [v43d] 判罚参数放宽后的新一轮：从 v43c 的最新存档 iter570 续跑。
# 相对 v43c 的唯一区别 = layup.py 的三个合法防守位参数：
#   foul_legal_def_speed 0.3 -> 0.4、foul_legal_def_approach 0.3 -> 0.4、k_legal_def_exempt 1.5 -> 1.2
#   （foul_approach_threshold 按用户清单保持 0.35）
# 其余与 v43c 同参：LIVE_VIEW=1、OVERLAP_COLLECTION=1、bf16 采样、不开 expandable_segments。
set -u
cd /home/proton/robocon2025_marl_devcontainer

CKPT="outputs/2026-10-06_11-10-07/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/checkpoints/checkpoint_171000000.pt"
LOG=outputs/train_v43d.log

# 守卫：只在真的没有 python 训练进程时启动（pgrep -f 会匹配 bash 包装自身，故按 readlink 过滤）
BUSY=$(docker exec robocon2025-marl bash -lc 'for p in $(pgrep -f clear_restore.py); do exe=$(readlink -f /proc/$p/exe 2>/dev/null); case "$exe" in *python*) echo $p;; esac; done' 2>/dev/null)
if [ -n "${BUSY:-}" ]; then echo "已有训练在跑（pid $BUSY），退出"; exit 1; fi

docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "LIVE_VIEW=1 OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 python clear_restore.py -m cont -c $CKPT --max-iters 700 > $LOG 2>&1"

sleep 25
echo "=== 启动校验（$LOG）==="
docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "grep -E '\[BufferGuard\]|\[Optimizer\]|\[LossCoef\]|\[LiveView\]|Will restore|阈值' $LOG | head -10; echo '...'; tail -2 $LOG"
echo "=== 确认新判罚参数已生效（任务 yaml 是否覆盖）==="
grep -nE "foul_legal_def|k_legal_def_exempt|foul_approach_threshold" \
  /home/proton/robocon2025_marl_devcontainer/BenchMARL/benchmarl/conf/task/vmas/layup.yaml || echo "(任务 yaml 未覆盖这三项 -> 使用 layup.py 新默认值)"
