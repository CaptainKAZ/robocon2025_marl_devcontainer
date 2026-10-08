"""扫描 checkpoint 内所有浮点张量中的非有限值（NaN / ±Inf）。

用法（容器内）:
    python /tmp/nan_scan_ckpt.py <ckpt_path> [<ckpt_path2> ...]

输出: 每个 checkpoint 一行汇总 + 每个含非有限值的叶子张量的 dtype/shape/数量/finite 区间。
"""

import sys

import torch
from collections.abc import Mapping

MAX_DEPTH = 14


def scan_tensor(t, path, results):
    if not torch.is_tensor(t):
        return
    if not t.is_floating_point():
        return
    try:
        finite = torch.isfinite(t)
    except Exception as e:  # noqa: BLE001
        results.append(f"  {path}: isfinite() 失败: {e}")
        return
    n_bad = int((~finite).sum())
    if n_bad == 0:
        return
    good = t[finite]
    if good.numel() > 0:
        gmin, gmax = float(good.min()), float(good.max())
    else:
        gmin = gmax = float("nan")
    frac = n_bad / max(t.numel(), 1)
    results.append(
        f"  {path}: dtype={t.dtype} shape={tuple(t.shape)} "
        f"nonfinite={n_bad}/{t.numel()} ({frac:.3%}) finite[min,max]=[{gmin:.4g},{gmax:.4g}]"
    )


def walk(obj, path, results, depth=0):
    if depth > MAX_DEPTH:
        return
    if torch.is_tensor(obj):
        scan_tensor(obj, path, results)
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            walk(v, f"{path}/{k}", results, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]", results, depth + 1)


def main():
    for ckpt_path in sys.argv[1:]:
        print(f"\n===== {ckpt_path} =====", flush=True)
        ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        if isinstance(ck, Mapping):
            top = [k for k in ck.keys()]
            print("顶层键:", top)
        results = []
        walk(ck, "", results)
        if results:
            print(f"发现 {len(results)} 个含非有限值的张量:")
            for line in results[:150]:
                print(line)
        else:
            print("未发现任何非有限张量（全部 finite）")
        buf = ck.get("buffer_agents") if isinstance(ck, Mapping) else None
        if buf is not None:
            try:
                st = buf["_storage"]["_storage"]
                state = st["state"]
                print("buffer state shape:", tuple(state.shape), "| rows:", int(state.shape[0]))
            except Exception as e:  # noqa: BLE001
                print("buffer 信息读取失败:", e)


if __name__ == "__main__":
    main()
