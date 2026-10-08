"""汇总 mb_ab 三臂日志：每臂打印 minibatch 采样形状 / 训练侧耗时 / 单轮墙钟 / 显存峰值。

用法：
  python3 .opencode/mb_report.py            # 读 .opencode/mb_ab_logs/*.log
  python3 .opencode/mb_report.py <log> ...  # 直接给日志路径
"""

import glob
import re
import sys

PAT_HEAD = re.compile(r"\[MbArm\] MB=(\d+) iters=(\d+) frames=(\d+) envs=(\d+) n_minibatch_iters=(\d+) overlap=(\d) bf16=(\d)")
PAT_ITER = re.compile(r"\[MbArm\] iter=(\d+) iter_dt=([\d.]+)s cuda_max_alloc=([\d.]+)GB cuda_max_resv=([\d.]+)GB sample_obs=(.*)$")
PAT_PROF = re.compile(r"\[PROF\]\[agents\].*?opt_loops=([\d.]+)s")
PAT_COLL = re.compile(r"collection time:? ([\d.]+)")
PAT_DONE = re.compile(r"\[MbArm\] DONE MB=(\d+) total=([\d.]+)s")


def parse(path):
    out = {"path": path, "mb": None, "iters": [], "opt_loops": [], "coll": [], "total": None, "errors": []}
    for line in open(path, errors="replace"):
        m = PAT_HEAD.search(line)
        if m:
            out["mb"] = int(m.group(1))
            out["head"] = m.groups()
            continue
        m = PAT_ITER.search(line)
        if m:
            out["iters"].append(
                {
                    "iter": int(m.group(1)),
                    "dt": float(m.group(2)),
                    "alloc": float(m.group(3)),
                    "resv": float(m.group(4)),
                    "obs": m.group(5).strip(),
                }
            )
            continue
        m = PAT_PROF.search(line)
        if m:
            out["opt_loops"].append(float(m.group(1)))
            continue
        m = PAT_COLL.search(line)
        if m:
            out["coll"].append(float(m.group(1)))
            continue
        m = PAT_DONE.search(line)
        if m:
            out["total"] = float(m.group(2))
            continue
        if "Traceback" in line or "Error" in line:
            out["errors"].append(line.strip()[:160])
    return out


def agg(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


def main():
    paths = sys.argv[1:] or sorted(glob.glob(".opencode/mb_ab_logs/*.log"))
    if not paths:
        print("(没有日志；先跑 run_mb_ab.sh 并把日志拷到 .opencode/mb_ab_logs/)")
        return
    print(f"{'MB':>6} | {'sample_obs':<18} | {'opt_loops(s)':>28} | {'collection(s)':>13} | {'iter_dt(s)':>11} | {'cuda峰值(GB)':>12}")
    for p in paths:
        d = parse(p)
        iters = [i for i in d["iters"]]
        obs = iters[-1]["obs"] if iters else "?"
        # 跳过第 1 轮（含暖机/首次编译）
        opt = d["opt_loops"][1:] or d["opt_loops"]
        coll = d["coll"][1:] or d["coll"]
        dt = [i["dt"] for i in iters][1:] or [i["dt"] for i in iters]
        alloc = max((i["alloc"] for i in iters), default=0.0)
        detail = "/".join(f"{x:.1f}" for x in d["opt_loops"])
        print(
            f"{d['mb']!s:>6} | {obs:<18} | {agg(opt):>10.1f}  [{detail}] | "
            f"{agg(coll):>13.1f} | {agg(dt):>11.1f} | {alloc:>12.2f}"
        )
        if d["errors"]:
            print(f"        !! {len(d['errors'])} error line(s), e.g. {d['errors'][0]}")
        if d["total"]:
            print(f"        total={d['total']:.1f}s")


if __name__ == "__main__":
    main()
