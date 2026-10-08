#!/usr/bin/env python3
"""渲染 iter300(A) vs iter450(B) 交叉对打矩阵：.opencode/xp42_{det,random}.json -> markdown 报告。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REASON = {1: "命中", 2: "造犯规", 3: "守方撞墙", 4: "守方越线", 5: "守方误伤",
          11: "被盖", 12: "超时", 13: "己方犯规", 14: "己方撞墙", 15: "己方误伤", 0: "未判"}
LABEL = {"A": "iter300(上一轮)", "B": "iter450(最新)"}


def load(mode):
    p = os.path.join(HERE, f"xp42_{mode}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def cell_txt(d):
    if not d:
        return "—"
    wr = 100.0 * d["win"] / max(1, d["episodes"])
    sr = 100.0 * d["shot"] / max(1, d["episodes"])
    return f"{wr:5.1f}% / 码1 {sr:4.1f}%  (n={d['episodes']})"


def main():
    out = []
    out.append("# 交叉对打：iter300（上一轮，A） vs iter450（最新，B）")
    out.append("")
    out.append("单组共享 actor（role_ids [0,1,2,2]）⇒ 无法只换攻/守权重；本实验两份 policy 各前向一次、按 agent 行拼接动作。")
    out.append("格 = 攻方胜率 / 码1（命中）占比；X@Y = 攻方用 X、守方用 Y。")
    out.append("")
    for mode in ("det", "random"):
        d = load(mode)
        if not d:
            out.append(f"## {mode}：数据缺失")
            continue
        out.append(f"## {mode} 口径（{d['envs']} envs × {d['rounds']} rounds = {d['envs']*d['rounds']} 局/格）")
        out.append("")
        out.append("| 攻方 \\ 守方 | A (iter300) | B (iter450) |")
        out.append("|---|---|---|")
        for a in ("A", "B"):
            row = [f"**{a}** ({LABEL[a].split('(')[1][:-1]})"]
            for b in ("A", "B"):
                row.append(cell_txt(d["cells"].get(f"{a}@{b}")))
            out.append("| " + " | ".join(row) + " |")
        out.append("")
        out.append("终局构成（各格 top5）：")
        for k, v in d["cells"].items():
            top = sorted(v["reasons"].items(), key=lambda kv: -kv[1])[:5]
            s = " ".join(f"{REASON.get(int(c), c)}×{n}" for c, n in top)
            out.append(f"- `{k}`：{s}")
        out.append("")
    out.append("## 读法")
    out.append("- 自对各格（A@A / B@B）是同一份权重自打，作为该档的基线。")
    out.append("- `A@B` 越低于 `A@A` ⇒ B 的防守更能遏制 A 的进攻；`B@A` 越高 ⇒ B 的进攻越能打穿 A 的防守。")
    out.append("- 注意：本实验里的 A 是 v41 末档（奖励改动前），B 是 v42 末档（A2 新通道 + 被盖拆分之后）；两档都只训练到各自阶段，不代表收敛上限。")
    print("\n".join(out))


if __name__ == "__main__":
    main()
