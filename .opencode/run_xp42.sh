#!/bin/bash
# 容器内：跑 iter300 vs iter450 的交叉对打（det + random），各 192 envs × 3 rounds
set -u
cd /home/vscode/workspace/BenchMARL
export OMP_NUM_THREADS=6 ATTENTION_COMPILE_MODE=off PYTHONPATH=/home/vscode/workspace/BenchMARL
export VMAS_INITIAL_SHOT_THRESHOLD=0.2
rm -f /tmp/xp42_done.txt
for m in det random; do
  echo "[run] $m start $(date)" >> /tmp/xp42_run.log
  python /tmp/crossplay_v42.py --mode "$m" --envs 192 --rounds 3 --out "/tmp/xp42_$m.json" > "/tmp/xp42_$m.log" 2>&1
  echo "[run] $m exit=$? $(date)" >> /tmp/xp42_run.log
done
echo done > /tmp/xp42_done.txt
echo "[run] all done $(date)" >> /tmp/xp42_run.log
