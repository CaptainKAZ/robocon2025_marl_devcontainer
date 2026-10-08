#!/usr/bin/env bash
# 等容器内 run_xp42.sh 全部跑完（run log 出现 "all done"）-> 取回两个 JSON -> 评估。
# 轮询用循环，每次 sleep 20s，最多 20 次（≈7 分钟）。后台运行。
set -u
cd "$(dirname "$0")/.." || exit 1          # 仓库根

DONE=0
for i in $(seq 1 20); do
  if docker exec robocon2025-marl bash -lc 'grep -q "all done" /tmp/xp42_run.log' 2>/dev/null; then
    DONE=1; echo "[poll] run log 出现 'all done'（第 $i 次检查）"; break
  fi
  sleep 20
done
if [ "$DONE" -ne 1 ]; then
  echo "[poll] 超时：run_xp42.sh 仍未结束"
  docker exec robocon2025-marl bash -lc 'tail -3 /tmp/xp42_run.log; grep -E "^\[cell\]|^\[done\]" /tmp/xp42_random.log'
  exit 1
fi

docker cp robocon2025-marl:/tmp/xp42_det.json    .opencode/xp42_det.json
docker cp robocon2025-marl:/tmp/xp42_random.json .opencode/xp42_random.json
echo "=============== 交叉对打评估 (A=iter300 / B=iter450) ==============="
python3 .opencode/eval_xp42.py
