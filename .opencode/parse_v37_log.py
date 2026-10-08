"""解析 clear_restore 训练日志的"回合结束原因统计"块（逐批终端码分布）。
用法: python parse_v37_log.py <log_path> <lo> <hi>
"""
import re
import sys

p, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
txt = open(p, errors="ignore").read()
blocks = txt.split("回合结束原因统计")
rx = re.compile(r"码\s*(\d+)\):\s*(\d+)\s*次\s*\(([\d.]+)%\)")
print("blocks:", len(blocks) - 1)
for i, b in enumerate(blocks[1:], 1):
    if lo <= i <= hi:
        codes = {int(m.group(1)): float(m.group(3)) for m in rx.finditer(b)}
        tot = re.search(r"总结束次数:\s*(\d+)", b)
        win = re.search(r"胜率:\s*([\d.]+)%", b)
        row = " ".join(f"码{k}:{codes.get(k, 0):.2f}" for k in (1, 2, 3, 4, 5, 11, 12, 13, 14, 15))
        print(f"#{i} tot={tot.group(1) if tot else '?'} win={win.group(1) if win else '?'} {row}")
