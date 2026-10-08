"""快速探针：打印 v38 checkpoint 的 buffer 存储结构（键/形状/dtype），供离线预筛脚本使用。
用法: python probe_v38_buffer.py <ckpt_path>
"""
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=False)
print("ckpt top keys:", list(ck.keys()))
buf = ck["buffer_agents"]
print("buffer_agents keys:", list(buf.keys()))
st = buf["_storage"]["_storage"]
print("storage type:", type(st).__name__)


def walk(d, prefix="", depth=0):
    if depth > 2:
        return
    try:
        ks = list(d.keys())
    except Exception:
        print(prefix, "->", type(d).__name__)
        return
    for k in ks:
        v = d[k]
        if hasattr(v, "shape"):
            print(f"{prefix}{k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"{prefix}{k}: {type(v).__name__}")
            walk(v, prefix + str(k) + ".", depth + 1)


walk(st)
print("--- storage.next ---")
walk(st["next"], "next.")
print("--- writer/sampler ---")
for k in ("_writer", "_sampler", "_batch_size", "_rng"):
    if k in buf:
        v = buf[k]
        if hasattr(v, "shape"):
            print(f"{k}: shape={tuple(v.shape)}")
        else:
            print(f"{k}: {type(v).__name__} {v if not hasattr(v, 'keys') else list(v.keys())}")
