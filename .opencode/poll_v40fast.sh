#!/usr/bin/env bash
# 轮询 v40fast：等首个 [PROF] 行（说明一轮跑完）或报错
LOG=/home/proton/robocon2025_marl_devcontainer/BenchMARL/outputs/train_v40fast.log
STATUS=/tmp/opencode/v40fast_status.txt
mkdir -p /tmp/opencode
for i in $(seq 1 60); do
  if [ -f "$LOG" ] && grep -qE "Traceback|RuntimeError|KeyError|ValueError|Error:" "$LOG" 2>/dev/null; then
    echo "$(date +%H:%M:%S) ERROR" > "$STATUS"; break
  fi
  if [ -f "$LOG" ] && grep -q "\[PROF\]\[agents\]" "$LOG" 2>/dev/null; then
    it=$(grep -oE "[0-9]+/150 \[[0-9:]+<[0-9:]+" "$LOG" | tail -1)
    prof=$(grep "\[PROF\]\[agents\]" "$LOG" | tail -1)
    ct=$(grep "collection time" "$LOG" | tail -1)
    echo "$(date +%H:%M:%S) FIRST_ITER_OK | $it | $prof | $ct" > "$STATUS"; break
  fi
  n=$(grep -c "collection time" "$LOG" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) waiting collected=$n" > "$STATUS"
  sleep 20
done
