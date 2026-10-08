#!/usr/bin/env bash
# 轮询方案 B 冒烟：等 SMOKE_SPECB_DONE 或 Traceback
STATUS=/tmp/opencode/smoke_specb_status.txt
for i in $(seq 1 24); do
  out=$(docker exec robocon2025-marl bash -lc 'cat /tmp/p0_smoke_specb.log' 2>/dev/null)
  if echo "$out" | grep -q "SMOKE_SPECB_DONE"; then
    echo "$(date +%H:%M:%S) DONE" > "$STATUS"; break
  fi
  if echo "$out" | grep -q "Traceback"; then
    echo "$(date +%H:%M:%S) TRACEBACK" > "$STATUS"; break
  fi
  echo "$(date +%H:%M:%S) waiting" > "$STATUS"
  sleep 20
done
