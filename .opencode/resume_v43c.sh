#!/bin/bash
# [v43c] 训练在 iter576 因 CUDA 错误（unknown error）崩掉后，从最新存档 iter570 续跑到 600。
# 与 v43b 同参：不开 expandable_segments（已验证无用且已被 torch 标 deprecated）。
set -u
cd /home/proton/robocon2025_marl_devcontainer

CKPT="outputs/2026-10-06_11-10-07/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/checkpoints/checkpoint_171000000.pt"
LOG=outputs/train_v43c.log

# 守卫：只在真的没有 python 训练进程时启动（pgrep -f 会匹配 bash 包装自身，故按 readlink 过滤）
BUSY=$(docker exec robocon2025-marl bash -lc 'for p in $(pgrep -f clear_restore.py); do exe=$(readlink -f /proc/$p/exe 2>/dev/null); case "$exe" in *python*) echo $p;; esac; done' 2>/dev/null)
if [ -n "${BUSY:-}" ]; then echo "已有训练在跑（pid $BUSY），退出"; exit 1; fi

docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "LIVE_VIEW=1 OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 python clear_restore.py -m cont -c $CKPT --max-iters 600 > $LOG 2>&1"

sleep 25
echo "=== 启动校验（$LOG）==="
docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "grep -E '\[BufferGuard\]|\[Optimizer\]|\[LossCoef\]|\[LiveView\]|Will restore' $LOG | head -8; echo '...'; tail -2 $LOG"
