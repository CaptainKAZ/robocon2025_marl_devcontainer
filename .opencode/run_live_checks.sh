#!/bin/bash
# 无头验收（host 缺 libnspr4/libnss3/libasound 等系统库，用 snap gnome runtime + playwright 自带 firefox bundle 顶替）
# 用法: bash .opencode/run_live_checks.sh [http://127.0.0.1:8765/]
set -u
cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH=/snap/gnome-46-2404/153/usr/lib/x86_64-linux-gnu:/home/proton/.cache/ms-playwright/firefox-1511/firefox${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "########## 1) 曲线：全部结束原因 ##########"
python3 .opencode/check_curve_codes.py "$@"
rc1=$?

echo
echo "########## 2) 整页验收（22 项） ##########"
python3 BenchMARL/liveview/live_html_check.py
rc2=$?

echo
echo "汇总: 曲线 rc=$rc1  整页 rc=$rc2"
exit $(( rc1 + rc2 ))
