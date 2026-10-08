#!/bin/bash
set -u
export PYTHONPATH=/home/vscode/workspace/BenchMARL
export ATTENTION_COMPILE_MODE=off
export OMP_NUM_THREADS=6
cd /home/vscode/workspace/BenchMARL
rm -f /tmp/xp3_done.txt /tmp/xp3_det.json
python /tmp/crossplay_v3.py --mode det --envs 192 --rounds 2 --cells v36:v36,v37:v37,v36:v37,v37:v36 --out /tmp/xp3_det.json > /tmp/xp3_det.log 2>&1
touch /tmp/xp3_done.txt
