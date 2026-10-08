# -*- coding: utf-8 -*-
"""速度/动作幅值画像：对比不同 checkpoint 里四角色的移动快慢与命令大小。

用法（容器内）:
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/probe_speed_profile.py <ckpt> <label>
兼容两种 buffer：单组按键版（action 是 dict：continuous/discrete）与旧的 2 维连续版。
"""
import sys

import torch

CKPT, LABEL = sys.argv[1], sys.argv[2]
TLIM = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0


def get(d, *keys):
    cur = d
    for k in keys:
        if hasattr(cur, "items") and k in cur.keys():
            cur = cur[k]
        else:
            return None
    return cur


def max_run(x):
    run = torch.zeros(x.shape[0], dtype=torch.long)
    best = torch.zeros_like(run)
    for t in range(x.shape[1]):
        run = torch.where(x[:, t], run + 1, torch.zeros_like(run))
        best = torch.maximum(best, run)
    return best


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
st = ck["buffer_agents"]["_storage"]["_storage"]
obs = get(st, "agents", "observation").float()
E, T = obs.shape[0], obs.shape[1]

act = get(st, "agents", "action", "continuous")
if act is None:
    a2 = get(st, "agents", "action")
    act = a2 if (a2 is not None and hasattr(a2, "shape") and a2.dim() == 4) else None
act = act.float() if act is not None else None

print(f"\n##### {LABEL}  ({CKPT.split('/')[-1]})  E={E} T={T} #####")
names = ["A1", "A2", "D1", "D2"]
for r, nm in enumerate(names):
    v = obs[:, :, r, 2:4] * 5.0
    spd = v.norm(dim=-1)
    line = (f"  {nm}: 速度 mean={spd.mean():.2f} p50={spd.quantile(0.5):.2f} p90={spd.quantile(0.9):.2f} "
            f"p99={spd.quantile(0.99):.2f} max={spd.max():.2f} | >2m/s={100.0*(spd>2).float().mean():.1f}% "
            f">3m/s={100.0*(spd>3).float().mean():.1f}%")
    if act is not None:
        m = act[:, :, r, :2].norm(dim=-1)
        line += (f" || 命令 mean={m.mean():.2f} p90={m.quantile(0.9):.2f} p99={m.quantile(0.99):.2f} "
                 f"max={m.max():.2f} | >=2.9(触顶)={100.0*(m>=2.9).float().mean():.1f}%")
    print(line)

# A1 速度随时间分桶（按剩余时间）
a1 = obs[:, :, 0, :]
t_rem = a1[..., 8] * TLIM  # obs 里的剩余时间比例 × 每局时长
spd1 = (a1[..., 2:4] * 5.0).norm(dim=-1)
print(f"  A1 平均速度按剩余时间分桶（{TLIM:.0f}s 一局）:")
for lo in [x * TLIM / 5 for x in range(5)]:
    m = (t_rem > lo) & (t_rem <= lo + 4)
    if m.any():
        print(f"    剩余 {lo:.0f}-{lo + TLIM/5:.0f}s: 速度 mean={spd1[m].mean():.2f} p90={spd1[m].quantile(0.9):.2f} 帧占比={100.0*m.float().mean():.1f}%")
