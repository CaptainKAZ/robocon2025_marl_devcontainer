# -*- coding: utf-8 -*-
"""v38 投篮学习看护：解析训练日志的批次块，判定"50 轮内是否学会投篮"。

用法: python3 watch_shot_parse.py <log> <status_path>

判定规则（在脚本里写死，便于事后核对）：
  - 最近 10 批窗口的码1（命中）均值 >= 3%            -> learned（学会了）
  - iter >= 50 且 码1 最近 < 1% 且 码11(被盖) < 1% 且
    历史最好出手率(码1+码11) < 2%                     -> diagnose（没学会，取证）
  - iter >= 80 且 码1 最近 < 3%（但有零星出手）        -> diagnose_late（补一次取证）
  - 其它 iter >= 50 的情况                            -> attempting（有出手端倪，继续观察）
  - 日志出现 Traceback                                -> crash
  - iter < 50                                         -> wait
"""
import re
import sys
import time

LOG = sys.argv[1]
STATUS = sys.argv[2]

text = open(LOG, encoding="utf-8", errors="replace").read()
blocks = text.split("===== 回合结束原因统计 =====")[1:]
code_re = re.compile(r"-\s*(胜利|失败):\s*([^(]*?)\(码 (\d+)\):\s*(\d+) 次 \(([\d.]+)%\)")
win_re = re.compile(r"胜率:\s*([\d.]+)%")

rows = []
for b in blocks:
    codes = {int(m.group(3)): float(m.group(5)) for m in code_re.finditer(b)}
    m = win_re.search(b)
    rows.append({"codes": codes, "win": float(m.group(1)) if m else None})

n = len(rows)
recent = rows[-10:]
avg = lambda f: (sum(f(r) for r in recent) / len(recent)) if recent else 0.0
c1_recent = avg(lambda r: r["codes"].get(1, 0.0))
c11_recent = avg(lambda r: r["codes"].get(11, 0.0))
c12_recent = avg(lambda r: r["codes"].get(12, 0.0))
c13_recent = avg(lambda r: r["codes"].get(13, 0.0))
win_recent = avg(lambda r: r["win"] or 0.0)
best1 = max((r["codes"].get(1, 0.0) for r in rows), default=0.0)
best_attempt = max((r["codes"].get(1, 0.0) + r["codes"].get(11, 0.0) for r in rows), default=0.0)

if "Traceback" in text:
    decision = "crash"
elif n >= 80 and c1_recent < 3.0:
    decision = "diagnose_late" if best_attempt >= 2.0 else "diagnose"
elif n >= 50 and c1_recent < 1.0 and c11_recent < 1.0 and best_attempt < 2.0:
    decision = "diagnose"
elif c1_recent >= 3.0:
    decision = "learned"
elif n >= 50:
    decision = "attempting"
else:
    decision = "wait"

line = (f"{time.strftime('%H:%M:%S')} iter={n} win={win_recent:.2f} c1={c1_recent:.2f} c11={c11_recent:.2f} "
        f"c12={c12_recent:.1f} c13={c13_recent:.1f} best1={best1:.2f} best_attempt={best_attempt:.2f} -> {decision}")
with open(STATUS, "w", encoding="utf-8") as f:
    f.write(line + "\n")

print(f"ITER={n}")
print(f"DECISION={decision}")
print(f"C1R={c1_recent:.2f}")
print(f"BEST1={best1:.2f}")
