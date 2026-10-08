#!/usr/bin/env python3
"""观察窗「自动连播」验收（Playwright）。

用法：
    LD_LIBRARY_PATH=/tmp/opencode/pwlibs/root/usr/lib/x86_64-linux-gnu \
    python3 .opencode/live_autoadv_check.py [base_url]

检查：
  ① 有「播完自动播下一局」复选框且默认勾选
  ② 末帧停留后自动切下一局（sel 0→1）
  ③ 继续自动切（1→2）
  ④ 取消勾选：停在本局末帧（playing=False，sel 不变）
  ⑤ 重新勾选：自动续播
"""
import sys, time
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
RES = []


def chk(name, ok, extra=""):
    RES.append((name, bool(ok)))
    print(("PASS  " if ok else "FAIL  ") + name + (("   " + str(extra)) if extra else ""))


def force_near_end(page, ep=0):
    page.evaluate(
        "(i) => { selectEp(i); t = Math.max(0, curLen() - 2); acc = 0; endAt = 0; playing = true; setPlayLabel(); }",
        ep,
    )


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1500, "height": 950})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(BASE, wait_until="domcontentloaded", timeout=25000)
        pg.wait_for_function("() => typeof D !== 'undefined' && D && D.episodes && D.episodes.length >= 2", timeout=30000)
        n_eps = pg.evaluate("D.episodes.length")
        print(f"地址={BASE}  局数={n_eps}  iter={pg.evaluate('D.iter')}")

        # ① 复选框
        box = pg.query_selector("#auto")
        chk("① 存在「自动连播」复选框", box is not None)
        if box is not None:
            chk("① 默认勾选", pg.evaluate("document.getElementById('auto').checked"))
            chk("① autoAdv 初始为 true", pg.evaluate("autoAdv === true"))

        # ② 末帧 -> 自动下一局
        s0 = pg.evaluate("sel")
        force_near_end(pg)
        time.sleep(6.2)
        s1 = pg.evaluate("sel")
        chk("② 末帧停留后自动切下一局", s1 == (s0 + 1) % n_eps, f"sel {s0} -> {s1}")
        chk("② 切换后仍在播放", pg.evaluate("playing === true"))

        # ③ 再切一局
        force_near_end(pg, s1)
        time.sleep(6.2)
        s2 = pg.evaluate("sel")
        chk("③ 继续自动连播", s2 == (s1 + 1) % n_eps, f"sel {s1} -> {s2}")

        # ④ 取消勾选 -> 停住
        pg.evaluate("""() => { const c = document.getElementById('auto'); c.checked = false; c.onchange({target:c}); }""")
        time.sleep(0.2)
        chk("④ 取消勾选后 autoAdv=false", pg.evaluate("autoAdv === false"))
        force_near_end(pg, s2)
        time.sleep(6.2)
        s4 = pg.evaluate("sel")
        chk("④ 取消后停在本局末帧（不切换）", s4 == s2, f"sel {s2} -> {s4}")
        chk("④ 取消后播放停止", pg.evaluate("playing === false"))

        # ⑤ 勾回 -> 续播
        pg.evaluate("""() => { const c = document.getElementById('auto'); c.checked = true; c.onchange({target:c}); }""")
        time.sleep(0.3)
        chk("⑤ 勾回后自动续播", pg.evaluate("playing === true"))

        chk("⑥ 无 JS 报错", len(errs) == 0, errs[:2])
        b.close()

    ok = sum(1 for _, o in RES if o)
    print(f"\n==== {ok}/{len(RES)} 通过 ====")
    if ok != len(RES):
        print("失败项：" + " | ".join(n for n, o in RES if not o))
    sys.exit(0 if ok == len(RES) else 1)


if __name__ == "__main__":
    main()
