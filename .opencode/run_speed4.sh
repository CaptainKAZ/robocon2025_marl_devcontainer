#!/bin/bash
# [速度诊断] 采集侧 A/B：定位 v43 比 v42 慢 ~8.5s/轮的原因。
# 用法（host）: nohup bash .opencode/run_speed4.sh > /tmp/opencode/run_speed4.log 2>&1 &
# 前置：先停掉训练与看护（否则显存 7.2G + 8.4G 超 12G 会走内存，测量失真）。
set -u
cd /home/proton/robocon2025_marl_devcontainer

cp -f .opencode/speed_arm4.py /tmp/opencode/speed_arm4.py 2>/dev/null
docker cp .opencode/speed_arm4.py robocon2025-marl:/tmp/speed_arm4.py >/dev/null

COMMON='OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 VMAS_INITIAL_SHOT_THRESHOLD=0.2'

arm() {  # TAG  EVAL  LIVEVIEW  ALLOC_CONF
  local tag=$1 ev=$2 lv=$3 alloc=${4:-}
  echo "=== ARM $tag  eval=$ev liveview=$lv alloc=[${alloc:-default}]  $(date +%H:%M:%S) ==="
  docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
    "ARM_TAG=$tag ARM_EVAL=$ev LIVE_VIEW=$lv ARM_ITERS=3 PYTORCH_CUDA_ALLOC_CONF=$alloc $COMMON python /tmp/speed_arm4.py > outputs/speed4_$tag.log 2>&1"
  echo "--- ARM $tag 结束 $(date +%H:%M:%S) ---"
}

arm A 1 1 "expandable_segments:True"   # v43 现场复刻
arm B 1 0 ""                           # 去掉 expandable_segments + 观察窗
arm C 0 0 ""                           # 再关掉评测 worker（≈ 之前 mb_arm 的配置）

echo "ALL ARMS DONE $(date +%H:%M:%S)"
