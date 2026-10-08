#!/bin/bash
# [v43f] v43e 在 iter640 再次死于偶发 GPU 故障（CUDA error: unknown error，容器 ExitCode=255）。
# 由于 checkpoint_interval 已改成 1.5M 帧（每 5 轮），iter640 的存档刚刚写完 => 本次几乎零损失。
# 参数与 v43d/v43e 完全一致（含新判罚 0.4/0.4/1.2）。
set -u
cd /home/proton/robocon2025_marl_devcontainer

CKPT="outputs/2026-10-08_10-31-21/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/checkpoints/checkpoint_192000000.pt"
LOG=outputs/train_v43f.log

BUSY=$(docker exec robocon2025-marl bash -lc 'for p in $(pgrep -f clear_restore.py); do exe=$(readlink -f /proc/$p/exe 2>/dev/null); case "$exe" in *python*) echo $p;; esac; done' 2>/dev/null)
if [ -n "${BUSY:-}" ]; then echo "已有训练在跑（pid $BUSY），退出"; exit 1; fi

docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "LIVE_VIEW=1 OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 python clear_restore.py -m cont -c $CKPT --max-iters 700 > $LOG 2>&1"

sleep 25
echo "=== 启动校验（$LOG）==="
docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "grep -E '\[BufferGuard\]|\[Optimizer\]|\[LossCoef\]|\[LiveView\]|Will restore' $LOG | head -8; echo '...'; tail -2 $LOG | tr -d '\r' | cut -c1-160"
