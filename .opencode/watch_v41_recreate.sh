#!/usr/bin/env bash
# v40c 的 150 轮跑完 -> 重建「特权」容器 -> 带 LIVE_VIEW=1 续跑 v41 -> 再挂看护
# 用户 2026-10-03 定案："以后起 docker 都用 privilege=1"，按"150 轮结束一次做完"执行。
set -u
HOST=/home/proton/robocon2025_marl_devcontainer
OPC=$HOST/.opencode
STATUS=/tmp/opencode/v41_orch_status.txt
cd "$HOST/BenchMARL" || exit 1

LOG=$HOST/BenchMARL/outputs/train_v40c.log

# 0) 先停掉旧看护：否则它会在“旧容器”里抢先续跑，与重建冲突
for p in $(pgrep -f "[w]atch_v40c.sh"); do kill -9 "$p" 2>/dev/null || true; done
echo "$(date +%H:%M:%S) orchestrator up; waiting for v40c to finish" > "$STATUS"

# 1) 等训练结束
while ps -eo args | grep -q "[c]lear_restore.py"; do
  prog=$(grep -oE "[0-9]+/[0-9]+ \[" "$LOG" 2>/dev/null | tail -1 | tr -d ' [')
  echo "$(date +%H:%M:%S) waiting ${prog:-?} (still training)" > "$STATUS"
  sleep 25
done
echo "$(date +%H:%M:%S) training ended; writing window report" > "$STATUS"

# 2) 窗口报告存档（仅留档，不阻塞）
if [ -f "$OPC/v36_report.py" ]; then
  python3 "$OPC/v36_report.py" "$LOG" "v40c_final" > /tmp/opencode/v40c_final_windows.txt 2>&1 || true
fi

# 3) 找最新 checkpoint
CKPT=$(ls -t "$HOST"/BenchMARL/outputs/*/mappo_layup_sequencemodel__*/checkpoints/checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "$(date +%H:%M:%S) no checkpoint -> abort" >> "$STATUS"; exit 1; fi
rel=${CKPT#"$HOST"/BenchMARL/}
echo "$(date +%H:%M:%S) ckpt=$rel ; recreating privileged container" > "$STATUS"

# 4) 重建特权容器
bash "$OPC/start_container.sh" > /tmp/opencode/container_recreate.log 2>&1
PRIV=$(docker inspect robocon2025-marl --format '{{.HostConfig.Privileged}}' 2>/dev/null)
if [ "$PRIV" != "true" ]; then
  echo "$(date +%H:%M:%S) recreate FAILED (Privileged=$PRIV, see /tmp/opencode/container_recreate.log)" >> "$STATUS"
  exit 1
fi
echo "$(date +%H:%M:%S) privileged container ready; relaunching scripts" > "$STATUS"

# 5) 新容器 /tmp 是空的，重新投放容器内脚本
for f in forensics_p0.py live_view.py; do
  docker cp "$OPC/$f" "robocon2025-marl:/tmp/$f" >/dev/null 2>&1 || true
done

# 6) 带 LIVE_VIEW=1 续跑（v41）
docker exec -d -w /home/vscode/workspace/BenchMARL robocon2025-marl bash -lc \
  "LIVE_VIEW=1 ACTOR_LR_MULT=2 OVERLAP_COLLECTION=1 MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 SAMPLING_AUTOCAST_BF16=1 python clear_restore.py -m cont -c $rel --max-iters 300 > outputs/train_v41.log 2>&1"
sleep 30

# 7) 胜率曲线：不再起 matplotlib 守护 —— 由 host 侧面板 /curve.json 直接解析日志、前端 Chart.js 渲染

# 8) 生成并挂 v41 看护（其自动续跑同样带 LIVE_VIEW=1）
sed -e 's|v40c_watch_status|v41_watch_status|' \
    -e 's|train_v40c\.log|train_v41.log|' \
    -e 's|v40g|v41g|g' \
    "$OPC/watch_v40c.sh" > "$OPC/watch_v41.sh"
chmod +x "$OPC/watch_v41.sh"
nohup bash "$OPC/watch_v41.sh" > /tmp/opencode/watch_v41.out 2>&1 &

echo "$(date +%H:%M:%S) v41 started with LIVE_VIEW=1 (--max-iters 300); watch_v41 armed" > "$STATUS"
