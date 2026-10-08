#!/usr/bin/env bash
# minibatch 规模 A/B（串行）：6000 -> 12000 -> 20000，各 3 轮，冻结权重续跑
set -u
C=robocon2025-marl
run() {
  local mb=$1
  echo "=== [mb_ab] start MB=$mb $(date +%H:%M:%S) ==="
  docker exec -w /home/vscode/workspace/BenchMARL "$C" bash -lc \
    "MB_SIZE=$mb ARM_ITERS=3 OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 python /tmp/mb_arm.py > outputs/mb_arm_$mb.log 2>&1"
  echo "=== [mb_ab] done MB=$mb rc=$? $(date +%H:%M:%S) ==="
}
run 6000
run 12000
run 20000
echo "all done $(date +%H:%M:%S)" > /tmp/opencode/mb_ab_done.txt
