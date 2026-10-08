#!/bin/bash
# [v43] 速度实验结束后恢复训练：从 iter460 的存档继续跑到 600 轮。
# 用法（host）: bash .opencode/resume_v43.sh [ALLOC_CONF]
#   默认沿用 expandable_segments:True；若实验证明它拖慢采集，传空串关掉：
#   bash .opencode/resume_v43.sh ""            # 关闭 expandable_segments
set -u
cd /home/proton/robocon2025_marl_devcontainer

ALLOC=${1-expandable_segments:True}
CKPT="outputs/2026-10-06_10-37-05/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/checkpoints/checkpoint_138000000.pt"

docker exec robocon2025-marl bash -lc 'pgrep -f "python clear_restore.py" >/dev/null && { echo "已有训练在跑，退出"; exit 1; }'

docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "LIVE_VIEW=1 OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 PYTORCH_CUDA_ALLOC_CONF=$ALLOC python clear_restore.py -m cont -c $CKPT --max-iters 600 > outputs/train_v43b.log 2>&1"

sleep 20
echo "=== 启动校验（train_v43b.log）==="
docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  'grep -E "\[BufferGuard\]|\[Optimizer\]|\[LossCoef\]|\[LiveView\]|Will restore" outputs/train_v43b.log | head -8; echo "..."; tail -2 outputs/train_v43b.log'
