#!/bin/bash
# host 侧轮询：等容器内交叉对打完成 → 取回 JSON → 渲染报告
cd "$(dirname "$0")/.." || exit 1
STATUS=/tmp/opencode/xp42_poll_status.txt
echo "waiting $(date)" > "$STATUS"
for i in $(seq 1 300); do
  if docker exec robocon2025-marl bash -lc 'test -f /tmp/xp42_done.txt' >/dev/null 2>&1; then
    docker cp robocon2025-marl:/tmp/xp42_det.json .opencode/ >/dev/null 2>&1
    docker cp robocon2025-marl:/tmp/xp42_random.json .opencode/ >/dev/null 2>&1
    docker cp robocon2025-marl:/tmp/xp42_det.log /tmp/opencode/ >/dev/null 2>&1
    docker cp robocon2025-marl:/tmp/xp42_random.log /tmp/opencode/ >/dev/null 2>&1
    docker cp robocon2025-marl:/tmp/xp42_run.log /tmp/opencode/ >/dev/null 2>&1
    python3 .opencode/render_xp42.py > .opencode/xp42_report.md 2>&1
    echo "DONE $(date)" > "$STATUS"
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT $(date)" > "$STATUS"
