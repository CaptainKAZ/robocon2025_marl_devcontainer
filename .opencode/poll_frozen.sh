#!/bin/bash
# 轮询冻结权重判决实验，结束后输出关键行（循环脚本，避免手打多次轮询）
LOG_HOST=/home/proton/robocon2025_marl_devcontainer/BenchMARL/outputs/frozen_lr0.log
MIRROR=/tmp/opencode/frozen_lr0.log
for i in $(seq 1 120); do
  if ! docker exec robocon2025-marl bash -lc 'ps -eo args | grep -q "[f]rozen_collect_lr0.py"'; then
    break
  fi
  sleep 20
done
cp "$LOG_HOST" "$MIRROR" 2>/dev/null
echo "===== frozen_lr0 关键行 ====="
grep -E "\[PerIter\]|\[BufferGuard\]|\[Frozen\]|Traceback|RuntimeError|Error" "$LOG_HOST" | tail -40
echo "===== 末尾 12 行 ====="
tail -12 "$LOG_HOST"