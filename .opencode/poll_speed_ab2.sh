#!/bin/bash
# 轮询 v2 完成并汇总
for i in $(seq 1 120); do
  if docker exec robocon2025-marl test -f /tmp/speed_arm2_done.txt 2>/dev/null; then break; fi
  sleep 20
done
for arm in D E; do
  echo "=== ARM $arm ==="
  docker exec robocon2025-marl bash -lc "grep -E '\[Arm2\]|collection time|Traceback|Error' /home/vscode/workspace/BenchMARL/outputs/speed_$arm.log | tail -8"
  docker exec robocon2025-marl bash -lc "grep -oE '[0-9]+/200 \[[0-9:]+<[0-9:]+, *[0-9.]+s/it\]' /home/vscode/workspace/BenchMARL/outputs/speed_$arm.log | tail -1"
done
echo SPEED_AB2_POLL_DONE
