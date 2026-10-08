#!/usr/bin/env python3
"""把 live_env.json（10 局）画成 2x5 胶片图：无浏览器时的兜底查看方式，也用于验证坐标映射。

用法：python render_live_preview.py --json outputs/live/live_env.json --out outputs/live/live_preview.png
     可选 --at end|0.5|mid 决定每格画在回合的哪个时刻（默认 end=终局瞬间）
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow

COLORS = ["#e6550d", "#fdae6b", "#3182bd", "#6baed6"]
# 容器里的 matplotlib 没有中文字体：把中文类别映射成 ASCII，避免胶片图全是方框
ASCII_OUTCOME = {
    "命中": "MADE", "出手被盖": "BLOCKED", "超时": "TIMEOUT",
    "造犯规(A1)": "FOUL_DRAWN(A1)", "造犯规(A2)": "FOUL_DRAWN(A2)",
    "攻方犯规(A1)": "OFF_FOUL(A1)", "攻方犯规(A2)": "OFF_FOUL(A2)",
    "攻方重罚": "OFF_HEAVY", "其它": "OTHER",
}


def frame_ax(ax, ep, f, t, S=1.0):
    ax.set_xlim(-4, 4)
    ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((-4, -7.5), 8, 15, fill=False, ec="#c9d1d9", lw=1))
    ax.axhline(0, color="#b9c2cc", ls="--", lw=0.8)
    ax.add_patch(Circle(ep["spot"], 0.9, color="#4caf50", alpha=0.15))
    ax.add_patch(Circle(ep["spot"], 0.9, color="#2e7d32", fill=False, lw=1))
    ax.add_patch(Circle(ep["basket"], 0.35, color="#e6550d", fill=False, lw=1.2))
    # 轨迹
    for a in range(4):
        xs = [ep["frames"][i][a*4] for i in range(t + 1)]
        ys = [ep["frames"][i][a*4+1] for i in range(t + 1)]
        ax.plot(xs, ys, color=COLORS[a], alpha=0.25, lw=1)
    for a in range(4):
        x, y, vx, vy = f[a*4], f[a*4+1], f[a*4+2], f[a*4+3]
        ax.add_patch(Circle((x, y), 0.34, color=COLORS[a], zorder=3))
        ax.annotate("A1A2D1D2"[a*2:a*2+2], (x, y), fontsize=5, ha="center", va="center", zorder=4)
        if vx*vx + vy*vy > 0.01:
            ax.add_patch(FancyArrow(x, y, vx*0.12, vy*0.12, width=0.02,
                                    head_width=0.22, color=COLORS[a], zorder=2))
    prog, in_spot = f[16], f[17]
    ax.add_patch(Rectangle((-0.9, 6.4), 1.8, 0.25, color="#d0d7de"))
    ax.add_patch(Rectangle((-0.9, 6.4), 1.8*prog, 0.25, color="#2e7d32"))
    ax.set_title(f"{ASCII_OUTCOME.get(ep['outcome'], ep['outcome'])} | row{ep['row']} len{ep['n']} | step {t} rem{f[18]:.1f}s charge{prog*100:.0f}%"
                 + (" INSPOT" if in_spot > 0.5 else ""), fontsize=6.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--at", default="end", help="end | 0..1 的比例")
    args = ap.parse_args()

    d = json.load(open(args.json))
    eps = d["episodes"]
    cols = 5
    rows = (len(eps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 4.3), dpi=110)
    axes = list(axes.flatten())
    for k, ax in enumerate(axes):
        if k >= len(eps):
            ax.axis("off")
            continue
        ep = eps[k]
        last = ep["n"] - 1
        t = last if args.at == "end" else int(round(float(args.at) * last))
        frame_ax(ax, ep, ep["frames"][t], t)
    fig.suptitle(f"live_env.json | iter {d['iter']} | {len(eps)} complete episodes (is_init -> done)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out)
    print(f"[preview] wrote {args.out}")


if __name__ == "__main__":
    main()
