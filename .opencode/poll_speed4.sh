#!/bin/bash
# [速度诊断] 轮询 run_speed4.sh 的三个臂，完成后打印各臂的墙钟/采集时间汇总。
# 用法（host）: nohup bash .opencode/poll_speed4.sh > /tmp/opencode/poll_speed4.log 2>&1 &
set -u
cd /home/proton/robocon2025_marl_devcontainer
RUNLOG=/tmp/opencode/run_speed4.log

for i in $(seq 1 60); do
  if grep -q "ALL ARMS DONE" "$RUNLOG" 2>/dev/null; then
    echo "=== ALL ARMS DONE (poll #$i) ==="
    break
  fi
  sleep 30
done

for tag in A B C; do
  echo ""
  echo "########## ARM $tag ##########"
  docker exec -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
    "f=outputs/speed4_$tag.log; [ -f \$f ] || { echo '(无日志)'; exit 0; }; \
     grep -oE '\[Arm:$tag\] iter=[0-9]+ dt=[0-9.]+s cuda_max=[0-9]+MB' \$f; \
     echo '-- collection time 行 --'; grep -oE 'collection time: [0-9.]+s' \$f | head -6; \
     echo '-- opt_loops --'; grep -oE 'opt_loops=[0-9.]+s' \$f | head -6; \
     echo '-- 结尾 --'; tail -3 \$f"
done
echo ""
echo "POLL_SPEED4 DONE"
