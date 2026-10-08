#!/usr/bin/env bash
# v38 投篮学习看护：50 轮内学不会投篮 -> 自动跑 buffer 取证，结果写到 .opencode/v38_diag_*.txt
# 状态: .opencode/v38_shot_status.txt（每次覆盖）、v38_shot_history.txt（按 iter 追加）
# 完成标记: .opencode/v38_diag_READY.txt
set -u
cd /home/proton/robocon2025_marl_devcontainer

LOG=BenchMARL/outputs/train_v38.log
STATUS=.opencode/v38_shot_status.txt
HIST=.opencode/v38_shot_history.txt
READY=.opencode/v38_diag_READY.txt
DIAG1=.opencode/v38_diag_done.txt
DIAG2=.opencode/v38_diag_late_done.txt
RUNROOT=BenchMARL/outputs/2026-10-03_05-39-14
LOGWATCH=/tmp/opencode/watch_v38shot.log

echo "[watch] 启动 $(date '+%F %T') 日志=$LOG" >> "$LOGWATCH"

latest_ckpt () {
  find "$RUNROOT" -name "checkpoint_*.pt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2
}

run_diag () {  # $1 = tag
  local tag="$1" ck_host ck_ctr out
  ck_host=$(latest_ckpt)
  out=".opencode/v38_diag_${tag}.txt"
  if [ -z "$ck_host" ]; then
    echo "NO_CHECKPOINT $(date '+%F %T') tag=$tag" >> "$READY"
    return 1
  fi
  ck_ctr="${ck_host#BenchMARL/}"
  docker cp .opencode/forensics_button.py robocon2025-marl:/tmp/forensics_button.py >/dev/null 2>&1
  {
    echo "=== v38 投篮门诊 [$tag] $(date '+%F %T') ==="
    echo "ckpt: $ck_host"
    echo
    echo "--- ① 日志窗口表 ---"
    python3 .opencode/v36_report.py "$LOG" "v38@$tag" 2>&1 | tail -40
    echo
    echo "--- ② 最近 10 批细则 ---"
    grep -aE "^(  - |总胜利|胜率)" "$LOG" | tail -40
    echo
    echo "--- ③ buffer 取证（进圈/按键/读条/出手/advantage）---"
    docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
      "PYTHONPATH=/home/vscode/workspace/BenchMARL OMP_NUM_THREADS=6 python /tmp/forensics_button.py '$ck_ctr' '$tag' /tmp/fx_button_$tag.txt" 2>&1
    echo
    echo "--- ④ 训练进程/日志尾 ---"
    docker exec robocon2025-marl bash -lc 'ps -eo args | grep "[c]lear_restore.py" | head -1' 2>&1
    tail -c 600 "$LOG" | tr '\r' '\n' | tail -5
  } > "$out" 2>&1
  echo "DIAG_READY tag=$tag file=$out $(date '+%F %T')" >> "$READY"
  echo "[watch] 取证完成 tag=$tag -> $out" >> "$LOGWATCH"
}

last_iter=-1
stale=0
while true; do
  sleep 30
  alive=$(docker exec robocon2025-marl bash -lc 'ps -eo args | grep "[c]lear_restore.py" | head -1' 2>/dev/null)
  out=$(python3 .opencode/watch_shot_parse.py "$LOG" "$STATUS" 2>/dev/null)
  iter=$(printf '%s\n' "$out" | sed -n 's/^ITER=//p')
  dec=$(printf '%s\n' "$out" | sed -n 's/^DECISION=//p')
  iter=${iter:-0}

  if [ "$iter" != "$last_iter" ]; then
    tail -1 "$STATUS" >> "$HIST"
    last_iter=$iter
    stale=0
  else
    stale=$((stale + 1))
  fi

  case "$dec" in
    diagnose)
      if [ ! -f "$DIAG1" ]; then run_diag "it${iter}"; touch "$DIAG1"; fi
      ;;
    diagnose_late)
      if [ ! -f "$DIAG2" ]; then run_diag "it${iter}late"; touch "$DIAG2"; fi
      ;;
    crash)
      echo "TRAIN_CRASH iter=$iter $(date '+%F %T')" >> "$READY"
      echo "[watch] 检测到 Traceback，退出" >> "$LOGWATCH"
      break
      ;;
  esac

  if [ -z "$alive" ]; then
    if [ "$iter" -ge 149 ] 2>/dev/null; then
      echo "TRAIN_DONE iter=$iter $(date '+%F %T')" >> "$READY"
      break
    fi
    if [ "$stale" -ge 10 ]; then
      echo "TRAIN_STOPPED iter=$iter $(date '+%F %T')" >> "$READY"
      break
    fi
  fi
done
echo "[watch] 退出 $(date '+%F %T')" >> "$LOGWATCH"
