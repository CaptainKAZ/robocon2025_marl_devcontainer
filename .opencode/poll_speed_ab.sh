#!/bin/bash
# 轮询加速验证完成并汇总采集耗时
for i in $(seq 1 150); do
  if docker exec robocon2025-marl test -f /tmp/speed_ab_done.txt 2>/dev/null; then break; fi
  sleep 20
done
for arm in A B C; do
  echo "=== ARM $arm ==="
  docker exec robocon2025-marl bash -lc "grep -E '\[Arm\]|collection time' /home/vscode/workspace/BenchMARL/outputs/speed_$arm.log | tail -6"
done
echo "=== 末行 s/it ==="
for arm in A B C; do
  docker exec robocon2025-marl bash -lc "grep -oE '[0-9]+/200 \[[0-9:]+<[0-9:]+, *[0-9.]+s/it\]' /home/vscode/workspace/BenchMARL/outputs/speed_$arm.log | tail -1"
done
echo SPEED_AB_POLL_DONE
