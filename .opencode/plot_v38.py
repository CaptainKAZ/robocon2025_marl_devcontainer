"""v38 训练日志分析：胜率/终局码 vs 迭代 + 10 批窗口表 + 图片渲染。
用法（容器内）：python plot_v38.py [log] [out.png]
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = sys.argv[1] if len(sys.argv) > 1 else "outputs/train_v38.log"
OUTPNG = sys.argv[2] if len(sys.argv) > 2 else "outputs/_analysis/v38_winrate.png"

txt = Path(LOG).read_text(errors="ignore")

blocks = []
cur = None
for line in txt.splitlines():
    m = re.search(r"本批次数据中有\s*(\d+)\s*个回合结束", line)
    if m:
        cur = {"dones": int(m.group(1)), "codes": {}}
        blocks.append(cur)
        continue
    if cur is not None:
        m2 = re.search(r"码\s*(\d+)\):\s*(\d+)\s*次\s*\(([0-9.]+)%\)", line)
        if m2:
            cur["codes"][int(m2.group(1))] = float(m2.group(3))
            continue
        m3 = re.search(r"胜率:\s*([0-9.]+)%", line)
        if m3:
            cur["win"] = float(m3.group(1))
            cur = None

n = len(blocks)
x = list(range(1, n + 1))
win = [b.get("win", float("nan")) for b in blocks]
c1 = [b["codes"].get(1, 0.0) for b in blocks]
c11 = [b["codes"].get(11, 0.0) for b in blocks]
c12 = [b["codes"].get(12, 0.0) for b in blocks]
c13 = [b["codes"].get(13, 0.0) for b in blocks]
dones = [b["dones"] for b in blocks]


def ma(series, k=10):
    out = []
    for i in range(len(series)):
        lo = max(0, i - k + 1)
        out.append(sum(series[lo : i + 1]) / (i - lo + 1))
    return out


win_ma, c1_ma, c12_ma, c13_ma = ma(win), ma(c1), ma(c12), ma(c13)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])
fig.suptitle(f"v38 cold 150 iters - win rate vs iteration (1 iter = 300k frames, 1500 envs x 200 steps)", fontsize=13)

ax1.plot(x, win, color="tab:blue", alpha=0.25, lw=1)
ax1.plot(x, win_ma, color="tab:blue", lw=2.5, label="win rate (10-iter MA)")
ax1.axhline(58.0, color="gray", ls="--", lw=1, alpha=0.7, label="58% (old curriculum freeze line)")
best_i = max(range(n), key=lambda i: win_ma[i])
ax1.annotate(f"peak MA {win_ma[best_i]:.1f}% @iter {best_i+1}", xy=(best_i + 1, win_ma[best_i]),
             xytext=(best_i + 1 - 25, min(99, win_ma[best_i] + 8)), arrowprops=dict(arrowstyle="->", color="tab:blue"))
ax1.annotate(f"final MA {win_ma[-1]:.1f}%", xy=(n, win_ma[-1]), xytext=(n - 22, win_ma[-1] - 12),
             arrowprops=dict(arrowstyle="->", color="tab:blue"))
ax1.set_ylabel("win rate (%)")
ax1.set_ylim(0, 102)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right", fontsize=9)

ax2.plot(x, c1, color="tab:green", alpha=0.25, lw=1)
ax2.plot(x, c1_ma, color="tab:green", lw=2.5, label="code 1 shot made (10-iter MA)")
ax2.plot(x, c12_ma, color="tab:orange", lw=1.8, label="code 12 timeout (MA)")
ax2.plot(x, c13_ma, color="tab:red", lw=1.8, label="code 13 own foul (MA)")
ax2.set_ylabel("share of episodes (%)")
ax2.set_xlabel("iteration")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper right", fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.96])
Path(OUTPNG).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPNG, dpi=140)
print(f"[saved] {OUTPNG}  ({n} batches parsed)")

print("\n== 10-iter windows (win% | code1% | code11% | code12% | code13% | dones) ==")
W = 10
for s in range(0, n, W):
    e = min(n, s + W)
    w = sum(win[s:e]) / (e - s)
    k1 = sum(c1[s:e]) / (e - s)
    k11 = sum(c11[s:e]) / (e - s)
    k12 = sum(c12[s:e]) / (e - s)
    k13 = sum(c13[s:e]) / (e - s)
    dn = sum(dones[s:e])
    print(f"iters {s+1:3d}-{e:3d} | win {w:5.1f} | c1 {k1:5.1f} | c11 {k11:4.1f} | c12 {k12:5.1f} | c13 {k13:5.1f} | dones {dn}")
