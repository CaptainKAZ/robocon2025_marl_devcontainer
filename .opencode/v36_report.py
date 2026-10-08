# -*- coding: utf-8 -*-
"""批次日志窗口统计（胜率 + 各终局码占比），输出 markdown。

用法: python3 v36_report.py <log_path> <label> [out_path]
"""
import re
import sys

LOG = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
OUT = sys.argv[3] if len(sys.argv) > 3 else None

text = open(LOG, encoding="utf-8", errors="replace").read()
blocks = text.split("===== 回合结束原因统计 =====")[1:]
code_re = re.compile(r"-\s*(胜利|失败):\s*([^(]*?)\(码 (\d+)\):\s*(\d+) 次 \(([\d.]+)%\)")
done_re = re.compile(r"总胜利次数:\s*(\d+) / 总结束次数:\s*(\d+)")

rows = []
for b in blocks:
    codes = {int(m.group(3)): float(m.group(5)) / 100.0 for m in code_re.finditer(b)}
    dm = done_re.search(b)
    win = int(dm.group(1)) / max(int(dm.group(2)), 1) if dm else None
    rows.append({"codes": codes, "win": win})

codes_all = sorted({c for r in rows for c in r["codes"]})
w = 10
lines = [f"### {LABEL} 日志分析：共 {len(rows)} 个批次块\n"]
lines.append("| 窗口(批次) | 胜率 | " + " | ".join(f"码{c}%" for c in codes_all) + " |")
lines.append("|---" * (len(codes_all) + 2) + "|")
for i in range(0, len(rows), w):
    chunk = rows[i:i + w]
    if not chunk:
        continue
    win = [r["win"] for r in chunk if r["win"] is not None]
    win_s = f"{100 * sum(win) / len(win):.1f}" if win else "-"
    cells = [f"{100 * sum(r['codes'].get(c, 0.0) for r in chunk) / len(chunk):.1f}" for c in codes_all]
    lines.append(f"| {i + 1}-{i + len(chunk)} | {win_s} | " + " | ".join(cells) + " |")

out = "\n".join(lines)
if OUT:
    open(OUT, "w", encoding="utf-8").write(out)
else:
    print(out)
