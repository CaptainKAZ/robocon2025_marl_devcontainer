"""打印 scalars CSV 的最后 N 行（iter,value），并可选按奇偶对齐打印多列。
用法: python dump_csv.py <csv> <last_n> [<csv2> ...]
"""
import sys


def load(p):
    rows = []
    for ln in open(p):
        ln = ln.strip()
        if not ln or ln.startswith("iter"):
            continue
        parts = ln.replace(",", " ").split()
        try:
            it, val = int(float(parts[0])), float(parts[1])
            rows.append((it, val))
        except Exception:
            pass
    return rows


n = int(sys.argv[2])
for p in sys.argv[1:]:
    if p == sys.argv[1] or p == sys.argv[2]:
        continue
    name = p.split("/")[-1].replace(".csv", "")
    try:
        rows = load(p)[-n:]
        s = " ".join(f"{it}:{v:.4g}" for it, v in rows)
        print(f"[{name}] {s}")
    except Exception as e:
        print(f"[{name}] ERROR {e}")
