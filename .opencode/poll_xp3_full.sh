#!/usr/bin/env bash
# 交叉对打 v3 轮询：等容器内 /tmp/xp3_full_done.txt（det 先、random 后），
# 完成后取回两份 JSON 到项目 .opencode/，并打印两个日志尾。
set -uo pipefail
C=robocon2025-marl
ROOT=/home/proton/robocon2025_marl_devcontainer
for i in $(seq 1 160); do
  if docker exec "$C" test -f /tmp/xp3_full_done.txt 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] poll#$i done marker found"
    break
  fi
  if ! docker exec "$C" bash -lc 'ps -eo args | grep "[c]rossplay_v3.py" >/dev/null' 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] poll#$i 进程已不在且无完成标记（疑似中断）"
    break
  fi
  echo "[$(date +%H:%M:%S)] poll#$i 运行中..."
  sleep 30
done
docker cp "$C":/tmp/xp3_det.json "$ROOT/.opencode/xp3_det.json" 2>/dev/null || echo "(无 xp3_det.json)"
docker cp "$C":/tmp/xp3_random.json "$ROOT/.opencode/xp3_random.json" 2>/dev/null || echo "(无 xp3_random.json)"
echo "---- det log tail ----"
docker exec "$C" bash -lc 'tail -8 /tmp/xp3_det.log' 2>/dev/null || true
echo "---- random log tail ----"
docker exec "$C" bash -lc 'tail -8 /tmp/xp3_random.log' 2>/dev/null || true
echo "=== XP3 FULL FINISHED ==="
