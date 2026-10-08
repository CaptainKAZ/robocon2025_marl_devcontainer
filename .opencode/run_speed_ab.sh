#!/bin/bash
# 加速验证：三臂各 3 轮（冻结权重 lr=0），只比采集耗时
cd /home/vscode/workspace/BenchMARL || exit 1
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 ARM_ITERS=3 OVERLAP_COLLECTION=1
mkdir -p /tmp/speed_arm
rm -f /tmp/speed_ab_done.txt
echo "[AB] ARM A default (compile on, bf16 on)" >&2
SAMPLING_AUTOCAST_BF16=1 python /tmp/speed_arm.py > outputs/speed_A.log 2>&1
echo "[AB] ARM B compile off" >&2
ATTENTION_COMPILE_MODE=off SAMPLING_AUTOCAST_BF16=1 python /tmp/speed_arm.py > outputs/speed_B.log 2>&1
echo "[AB] ARM C bf16 off (fp32 sampling)" >&2
SAMPLING_AUTOCAST_BF16=0 python /tmp/speed_arm.py > outputs/speed_C.log 2>&1
echo DONE > /tmp/speed_ab_done.txt
