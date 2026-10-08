#!/usr/bin/env python3
"""渲染交叉对打 v3 结果：读 .opencode/xp3_{det,random}.json，打印 4 单元矩阵 + 终局构成。

用法：python3 .opencode/render_xp3.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {
    "det": os.path.join(ROOT, ".opencode", "xp3_det.json"),
    "random": os.path.join(ROOT, ".opencode", "xp3_random.json"),
}
REASON_NAMES = {0: "未终局", 1: "shot", 2: "opp_foul", 3: "opp_wall", 4: "opp_cross", 5: "opp_ff",
                11: "blocked", 12: "timeout", 13: "own_foul", 14: "own_wall", 15: "own_ff"}


def show(tag, path):
    if not os.path.exists(path):
        print(f"[{tag}] 缺文件：{path}")
        return
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    print(f"\n===== {tag} 口径（envs={d['envs']} rounds={d['rounds']}）=====")
    print(f"{'单元':22s} {'胜率':>7s} {'码1':>7s} {'局数':>6s}   终局构成")
    for key, st in d["cells"].items():
        eps = max(1, st["episodes"])
        wr = 100.0 * st["win"] / eps
        sr = 100.0 * st["shot"] / eps
        top = sorted(st["reasons"].items(), key=lambda kv: -kv[1])[:5]
        top_s = " ".join(f"{REASON_NAMES.get(int(c), c)}:{v}" for c, v in top)
        print(f"{key:22s} {wr:6.1f}% {sr:6.1f}% {st['episodes']:6d}   {top_s}")


def main():
    for tag, path in FILES.items():
        show(tag, path)
    print("\nRENDER DONE")


if __name__ == "__main__":
    main()
