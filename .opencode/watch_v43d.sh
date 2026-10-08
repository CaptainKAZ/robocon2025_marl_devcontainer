#!/usr/bin/env bash
# v43d 看护：只等训练结束 -> 窗口报告 + 健康标记，**不自动续跑**（崩过一次，续跑交回人工决定）
set -u
HOST=/home/proton/robocon2025_marl_devcontainer
STATUS=/tmp/opencode/v43d_watch_status.txt
LOG="$HOST/BenchMARL/outputs/train_v43d.log"

while ps -eo args | grep -q "[c]lear_restore.py"; do
  prog=$(grep -oE "[0-9]+/[0-9]+ \[" "$LOG" 2>/dev/null | tail -1 | tr -d ' [')
  echo "$(date +%H:%M:%S) running ${prog:-?}" > "$STATUS"
  sleep 25
done
echo "$(date +%H:%M:%S) ended; analyzing" > "$STATUS"
python3 "$HOST/.opencode/v36_report.py" "$LOG" v43d > /tmp/opencode/v43d_windows.txt 2>&1 || true
tb=$(grep -c "Traceback" "$LOG" 2>/dev/null | head -1); tb=${tb:-0}
st=$(grep -c "\[SafeTrain\]" "$LOG" 2>/dev/null | head -1); st=${st:-0}
echo "$(date +%H:%M:%S) ended tb=$tb safetrain=$st (不自动续跑) 窗口报告=/tmp/opencode/v43d_windows.txt" >> "$STATUS"
