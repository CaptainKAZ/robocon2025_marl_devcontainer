#!/bin/bash
set -u
DEST="/home/proton/robocon2025_marl_devcontainer/.opencode"
for i in $(seq 1 45); do
  if docker exec robocon2025-marl test -f /tmp/xp3_done.txt 2>/dev/null; then
    for f in xp3_random.json xp3_det.json xp3_random.log xp3_det.log; do
      docker cp robocon2025-marl:/tmp/$f "$DEST/$f" >/dev/null 2>&1
    done
    echo "=== xp3 random ==="; grep -a "\[cell\]" "$DEST/xp3_random.log" || tail -5 "$DEST/xp3_random.log"
    echo "=== xp3 det ===";    grep -a "\[cell\]" "$DEST/xp3_det.log" || tail -5 "$DEST/xp3_det.log"
    echo "[poll] done"
    exit 0
  fi
  sleep 30
done
echo "[poll] timeout"; docker exec robocon2025-marl bash -lc 'tail -3 /tmp/xp3_random.log; echo ---; tail -3 /tmp/xp3_det.log'
