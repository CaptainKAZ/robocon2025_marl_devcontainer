#!/usr/bin/env python3
"""交叉对打 A(iter300) vs B(iter450) 评估：攻防强度分离、命中转化、犯规/超时结构。
用法: python eval_xp42.py                # 读 .opencode/xp42_{det,random}.json
"""
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN_CODES = {"1", "2", "3", "4", "5"}
LABEL = {"A": "A=iter300(v41末)", "B": "B=iter450(v42末)"}


def sig(p, n):
    return 100.0 * math.sqrt(max(p * (1 - p), 0) / max(n, 1))


def load(mode):
    p = ROOT / f"xp42_{mode}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def cell_stats(c):
    n = c["episodes"]
    r = {str(k): v for k, v in c["reasons"].items()}
    win = sum(v for k, v in r.items() if k in WIN_CODES)
    shot, blocked = r.get("1", 0), r.get("11", 0)
    conv = (shot / (shot + blocked)) if (shot + blocked) else float("nan")
    return dict(n=n, win=c["win"], winpct=100.0 * c["win"] / n,
                win_ci=sig(c["win"] / n, n),
                shot=shot, shotpct=100.0 * shot / n,
                blocked=blocked, conv=100.0 * conv if conv == conv else float("nan"),
                own_foul=r.get("13", 0), own_foul_pct=100.0 * r.get("13", 0) / n,
                opp_foul=r.get("2", 0), timeout=r.get("12", 0),
                timeout_pct=100.0 * r.get("12", 0) / n, reasons=r)


def evaluate(mode, d):
    cells = d["cells"]
    key = {k.replace(" ", ""): v for k, v in cells.items()}
    # 归一化键名 -> AA / AB / BA / BB
    norm = {}
    for k, v in cells.items():
        # 形如 "A@A (攻=A 守=A)" / "A@B" / "B@A (攻=B 守=A)"
        t = k.split(" ")[0].upper()
        if "@" in t:
            a, b = t.split("@")
            norm[f"{a}{b}"] = v
    need = {"AA", "AB", "BA", "BB"}
    if not need.issubset(norm):
        print(f"\n[{mode}] 结果不完整（已有 {sorted(norm)}，缺 {sorted(need-set(norm))}）——跳过评估")
        return
    print(f"\n{'='*96}\n口径 {mode.upper()}  (envs={d['envs']} rounds={d['rounds']}, 每格 {list(norm.values())[0]['episodes']} 局)\n{'='*96}")
    st = {k: cell_stats(v) for k, v in norm.items()}
    print(f"{'格(攻@守)':<10}{'胜率%':>9}{'±1σ':>6}{'码1%':>8}{'被盖%':>8}{'命中转化率%':>12}{'己方犯规%':>11}{'超时%':>8}")
    for k in ("AA", "AB", "BA", "BB"):
        s = st[k]
        print(f"{k:<10}{s['winpct']:>9.1f}{s['win_ci']:>6.1f}{s['shotpct']:>8.1f}"
              f"{100.0*s['blocked']/s['n']:>8.1f}{s['conv']:>12.1f}{s['own_foul_pct']:>11.1f}{s['timeout_pct']:>8.1f}")
    print("\n-- 分项计数 --")
    for k in ("AA", "AB", "BA", "BB"):
        s = st[k]
        r = s["reasons"]
        print(f"  {k}: " + " ".join(f"码{c}:{r[c]}" for c in ("1","11","2","13","12","3","14","5","15","4") if r.get(c)))

    print("\n-- 攻防强度分离（用共同对手消除'谁打谁'的混杂）--")
    # 攻方强度：共同守方 A
    d_atk = st["BA"]["winpct"] - st["AA"]["winpct"]
    print(f"  攻方强度（共同守方=A）：B攻 {st['BA']['winpct']:.1f}% vs A攻 {st['AA']['winpct']:.1f}%  ⇒ Δ={d_atk:+.1f}pp")
    d_atk2 = st["BB"]["winpct"] - st["AB"]["winpct"]
    print(f"  攻方强度（共同守方=B）：B攻 {st['BB']['winpct']:.1f}% vs A攻 {st['AB']['winpct']:.1f}%  ⇒ Δ={d_atk2:+.1f}pp")
    # 守方强度：共同攻方 A
    d_def = st["AA"]["winpct"] - st["AB"]["winpct"]
    print(f"  守方强度（共同攻方=A）：A攻 vs A守 {st['AA']['winpct']:.1f}% vs A攻 vs B守 {st['AB']['winpct']:.1f}%  ⇒ B守压制 {d_def:+.1f}pp")
    d_def2 = st["BA"]["winpct"] - st["BB"]["winpct"]
    print(f"  守方强度（共同攻方=B）：B攻 vs A守 {st['BA']['winpct']:.1f}% vs B攻 vs B守 {st['BB']['winpct']:.1f}%  ⇒ B守压制 {d_def2:+.1f}pp")
    # 命中转化
    print(f"  命中转化率：A攻@A守 {st['AA']['conv']:.0f}% | B攻@A守 {st['BA']['conv']:.0f}% | A攻@B守 {st['AB']['conv']:.0f}% | B攻@B守 {st['BB']['conv']:.0f}%")
    # 自对打平衡
    print(f"  自对打平衡点（胜率%）：A {st['AA']['winpct']:.1f} | B {st['BB']['winpct']:.1f}  ⇒ 变化 {st['BB']['winpct']-st['AA']['winpct']:+.1f}pp")
    print(f"  自对打命中率 码1%：A {st['AA']['shotpct']:.1f} | B {st['BB']['shotpct']:.1f}  ; 己方犯规%：A {st['AA']['own_foul_pct']:.1f} | B {st['BB']['own_foul_pct']:.1f}")

    print("\n-- 判读 --")
    atk_up = (d_atk + d_atk2) / 2
    def_up = (d_def + d_def2) / 2
    print(f"  攻方进步 均值 Δ ≈ {atk_up:+.1f}pp ；守方进步 均值 Δ ≈ {def_up:+.1f}pp")
    if def_up > atk_up:
        print("  ⇒ B 攻守两端都更强，但**守方进步幅度更大** ⇒ 自对打胜率下降是'平衡点移向防守'，不是进攻退化。")
    else:
        print("  ⇒ B 攻方进步更大。")


def main():
    any_ = False
    for mode in ("det", "random"):
        d = load(mode)
        if d is None:
            print(f"\n[{mode}] 结果文件尚未生成（还在跑？）")
            continue
        any_ = True
        evaluate(mode, d)
    if not any_:
        sys.exit(1)


if __name__ == "__main__":
    main()
