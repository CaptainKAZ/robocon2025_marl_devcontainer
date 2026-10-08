#!/usr/bin/env bash
# 轮询 frozen_nan 复现：等 [NaNCase] 出现或进程结束；状态写入 /tmp/opencode/frozen_nan_watch_status.txt
LOG=/home/proton/robocon2025_marl_devcontainer/BenchMARL/outputs/frozen_nan.log
STATUS=/tmp/opencode/frozen_nan_watch_status.txt
mkdir -p /tmp/opencode
for i in $(seq 1 150); do
  if [ -f "$LOG" ] && grep -q "\[NaNCase\]" "$LOG" 2>/dev/null; then
    echo "$(date +%H:%M:%S) NaN_CASE_FOUND" > "$STATUS"; break
  fi
  if [ -f "$LOG" ] && grep -q "\[Frozen\] DONE" "$LOG" 2>/dev/null; then
    echo "$(date +%H:%M:%S) DONE_NO_CASE" > "$STATUS"; break
  fi
  niter=$(grep -c "\[PerIter\]" "$LOG" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) waiting iters=$niter" > "$STATUS"
  sleep 20
done
