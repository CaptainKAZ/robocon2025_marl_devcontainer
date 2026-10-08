#!/usr/bin/env python3
"""验收：胜率曲线是否把**全部 10 个结束原因**都画出来并可单独开关。

用法: python3 .opencode/check_curve_codes.py [http://127.0.0.1:8765/]
产物: .opencode/curve_all_codes.png（截图）
退出码: 0 全过 / 1 有 FAIL
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
CODES = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
SHOT = Path(__file__).resolve().parent / "curve_all_codes.png"

fails = []


def chk(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + str(extra)) if extra else ''}")
    if not ok:
        fails.append(name)


with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1680, "height": 940})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(BASE, wait_until="networkidle")
    pg.wait_for_timeout(3500)

    # ① 10 个复选框都在
    for n in CODES:
        chk(f"复选框 code{n} 存在", pg.locator(f"#code{n}").count() == 1)

    # ② 曲线数据里 10 个码都有序列
    info = pg.evaluate(
        """() => {
            const j = (typeof curveData !== 'undefined') ? curveData : null;
            if (!j || !j.points || !j.points.length) return {n: 0, keys: []};
            const keys = Object.keys(j.points[0]);
            return {n: j.points.length, keys};
        }"""
    )
    missing = [f"c{n}" for n in CODES if f"c{n}" not in info["keys"]]
    chk("曲线 JSON 含全部 c1..c15 字段", not missing, f"缺 {missing}")

    # ③ 图上真的画了 10 条码曲线（+ 胜率/5窗均/参考线）
    labels = pg.evaluate(
        """() => {
            const c = (typeof curveChart !== 'undefined') ? curveChart : null;
            return c ? c.data.datasets.map(d => d.label) : [];
        }"""
    )
    drawn = [l for l in labels if l.startswith("码")]
    chk("图上码曲线条数 = 10", len(drawn) == 10, f"{len(drawn)} 条: {drawn}")
    chk("含胜率与 5 窗均", "胜率" in labels and "胜率(5窗均)" in labels, labels[:4])

    # ④ 取消勾选一条后，图上的条数真的减少（开关生效）
    pg.uncheck("#code3")
    pg.wait_for_timeout(1200)
    labels2 = pg.evaluate("() => curveChart.data.datasets.map(d => d.label)")
    chk("取消 code3 后条数 -1", len(labels2) == len(labels) - 1, f"{len(labels)} -> {len(labels2)}")
    pg.check("#code3")
    pg.wait_for_timeout(1200)
    labels3 = pg.evaluate("() => curveChart.data.datasets.map(d => d.label)")
    chk("重新勾选 code3 后恢复", len(labels3) == len(labels), f"{len(labels3)}")

    chk("无 JS 报错", not errors, errors[:3])

    pg.locator("#pane-curve").screenshot(path=str(SHOT))
    print(f"截图: {SHOT}")
    b.close()

print("\n结论:", "全部通过" if not fails else f"{len(fails)} 项失败: {fails}")
sys.exit(1 if fails else 0)
