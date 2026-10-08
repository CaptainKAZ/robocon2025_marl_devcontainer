#!/bin/bash
# 加速验证 v3：fork + 关编译（绕过 forked 子进程里的 dynamo CUDA 初始化）
cd /home/vscode/workspace/BenchMARL || exit 1
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 ARM_ITERS=3 OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 MP_START_METHOD=fork ATTENTION_COMPILE_MODE=off
mkdir -p /tmp/speed_arm2
rm -f /tmp/speed_arm3_done.txt
echo "[AB3] ARM F fork+compile_off 2 workers" >&2
ARM_WORKERS=2 ARM_ENVS_PER_WORKER=750 python /tmp/speed_arm2.py > outputs/speed_F.log 2>&1
echo "[AB3] ARM G fork+compile_off 4 workers" >&2
ARM_WORKERS=4 ARM_ENVS_PER_WORKER=375 python /tmp/speed_arm2.py > outputs/speed_G.log 2>&1
echo DONE > /tmp/speed_arm3_done.txt
