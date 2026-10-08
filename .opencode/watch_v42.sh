#!/usr/bin/env bash
# v40 看护：逐代等待训练结束 -> 健康判定 -> 窗口报告 + forensics -> 健康则自动续跑 150 轮（最多 4 代）
set -u
HOST=/home/proton/robocon2025_marl_devcontainer
OPC=$HOST/.opencode
STATUS=/tmp/opencode/v42_watch_status.txt
WOPC=/tmp/opencode
mkdir -p "$WOPC"
cd "$HOST/BenchMARL" || exit 1

docker cp "$OPC/forensics_p0.py" robocon2025-marl:/tmp/forensics_p0.py >/dev/null 2>&1 || true

gen=0
LOG="$HOST/BenchMARL/outputs/train_v42.log"
while [ "$gen" -le 3 ]; do
  while ps -eo args | grep -q "[c]lear_restore.py"; do
    prog=$(grep -oE "[0-9]+/[0-9]+ \[" "$LOG" 2>/dev/null | tail -1 | tr -d ' [')
    echo "$(date +%H:%M:%S) gen=$gen running ${prog:-?}" > "$STATUS"
    sleep 25
  done
  echo "$(date +%H:%M:%S) gen=$gen ended; analyzing" > "$STATUS"

  if [ -f "$OPC/v36_report.py" ]; then
    python3 "$OPC/v36_report.py" "$LOG" "v42g$gen" > "$WOPC/v42g${gen}_windows.txt" 2>&1 || true
  fi

  tb=$(grep -c "Traceback" "$LOG" 2>/dev/null | head -1); tb=${tb:-0}
  st=$(grep -c "\[SafeTrain\]" "$LOG" 2>/dev/null | head -1); st=${st:-0}
  wall=$(grep "己方失误-撞墙 (码 14)" "$LOG" 2>/dev/null | tail -1 | grep -oE "\([0-9.]+%" | tr -d '(%')
  wall=${wall:-0}

  verdict=HEALTHY
  if [ "$tb" -gt 0 ]; then
    verdict=CRASH_TRACEBACK
  elif awk -v w="$wall" 'BEGIN{ exit !(w+0>40) }'; then
    verdict=WALL_COLLAPSE
  elif [ "$st" -gt 20 ]; then
    verdict=SAFETRAIN_SPAM
  fi
  echo "$(date +%H:%M:%S) gen=$gen verdict=$verdict tb=$tb safetrain=$st wall14=$wall" > "$STATUS"

  CKPT=$(ls -t "$HOST"/BenchMARL/outputs/*/mappo_layup_sequencemodel__*/checkpoints/checkpoint_*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    rel=${CKPT#"$HOST"/BenchMARL/}
    docker cp "$CKPT" robocon2025-marl:/tmp/ck_v42g$gen.pt >/dev/null 2>&1 || true
    docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
      "PYTHONPATH=/home/vscode/workspace/BenchMARL FPB=300000 NENV=1500 python /tmp/forensics_p0.py /tmp/ck_v42g$gen.pt v42g$gen /tmp/forensics_v42g$gen.txt" \
      > "$WOPC/forensics_v42g${gen}_run.log" 2>&1 || true
    docker cp robocon2025-marl:/tmp/forensics_v42g$gen.txt "$WOPC/forensics_v42g$gen.txt" >/dev/null 2>&1 || true
  fi

  if [ "$verdict" != "HEALTHY" ] || [ -z "$CKPT" ]; then
    echo "$(date +%H:%M:%S) stop: verdict=$verdict ckpt=${CKPT:-none}" >> "$STATUS"
    break
  fi

  next=$((gen+1))
  nlog="$HOST/BenchMARL/outputs/train_v42g${next}.log"
  echo "$(date +%H:%M:%S) auto-continue -> train_v42g${next}.log (max-iters $((450+150*next)))" >> "$STATUS"
  docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
    "LIVE_VIEW=1 ACTOR_LR_MULT=2 OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 python clear_restore.py -m cont -c $rel --max-iters $((450+150*next)) > outputs/train_v42g${next}.log 2>&1"
  sleep 20
  gen=$next
  LOG="$nlog"
done
echo "$(date +%H:%M:%S) watch loop exited (gen=$gen)" >> "$STATUS"
