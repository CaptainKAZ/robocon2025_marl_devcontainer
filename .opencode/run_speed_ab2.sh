#!/bin/bash
# 加速验证 v2：fork + 多进程采集（2/4 workers），各 3 轮
cd /home/vscode/workspace/BenchMARL || exit 1
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 ARM_ITERS=3 OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 MP_START_METHOD=fork
mkdir -p /tmp/speed_arm2
rm -f /tmp/speed_arm2_done.txt
echo "[AB2] ARM D fork + 2 workers x 750 envs" >&2
ARM_WORKERS=2 ARM_ENVS_PER_WORKER=750 python /tmp/speed_arm2.py > outputs/speed_D.log 2>&1
echo "[AB2] ARM E fork + 4 workers x 375 envs" >&2
ARM_WORKERS=4 ARM_ENVS_PER_WORKER=375 python /tmp/speed_arm2.py > outputs/speed_E.log 2>&1
echo DONE > /tmp/speed_arm2_done.txt
