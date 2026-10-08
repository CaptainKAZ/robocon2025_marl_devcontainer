"""探 buffer 里动作的真实量纲：ckpt60M 的 buffer_agents 中 action.continuous/logits/discrete 的统计。"""
import sys

import torch

ck_path = sys.argv[1]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
st = ck["buffer_agents"]["_storage"]["_storage"]

ac = st["agents"]["action"]["continuous"].float()
print(f"[action.continuous] shape={tuple(ac.shape)} dtype={ac.dtype}")
for d in range(ac.shape[-1]):
    x = ac[..., d]
    print(f"  dim{d}: min={x.min():+.3f} max={x.max():+.3f} mean={x.mean():+.3f} absmean={x.abs().mean():.3f} "
          f"p99={torch.quantile(x.flatten(), 0.99):+.3f}")
n = ac.norm(dim=-1)
print(f"  norm: mean={n.mean():.3f} p50={n.median():.3f} p90={torch.quantile(n.flatten(),0.9):.3f} max={n.max():.3f}")
print(f"  frac>5: {(n > 5).float().mean():.4f} | frac>7.07: {(n > 7.07).float().mean():.4f} | frac>10: {(n > 10).float().mean():.4f}")
print(f"  A1 norm mean: {n[...,0].mean():.3f} | A2: {n[...,1].mean():.3f} | D1: {n[...,2].mean():.3f} | D2: {n[...,3].mean():.3f}")

if "logits" in st["agents"]:
    lg = st["agents"]["logits"].float()
    print(f"[logits] shape={tuple(lg.shape)} min={lg.min():+.2f} max={lg.max():+.2f} absmean={lg.abs().mean():.3f}")

if "action" in st["agents"] and "discrete" in st["agents"]["action"]:
    di = st["agents"]["action"]["discrete"].float()
    print(f"[action.discrete] shape={tuple(di.shape)} mean={di.mean():.4f} A1mean={di[...,0].mean():.4f}")

# 看看其他相关键
print("[keys under agents]", list(st["agents"].keys()))
print("[keys under agents.action]", list(st["agents"]["action"].keys()) if "action" in st["agents"] else "?")
