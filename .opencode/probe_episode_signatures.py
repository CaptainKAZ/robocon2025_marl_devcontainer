#!/usr/bin/env python3
"""完整回合的终局签名统计：用于把 buffer 里的终局映射到终止码（观察窗的类别标签）。

用法：python probe_episode_signatures.py <ckpt> [iter_label]
"""
import sys

import numpy as np
import torch

ck_path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "?"

ck = torch.load(ck_path, map_location="cpu", mmap=True, weights_only=False)
st = ck["buffer_agents"]["_storage"]["_storage"]
state = st["state"]
done = st["next"]["done"][..., 0].bool().cpu().numpy()
ini = st["is_init"][..., 0].bool().cpu().numpy()
rew = st["next"]["agents"]["reward"]
if rew.dim() == 4:
    rew = rew[..., 0]
rew = rew.float().cpu().numpy()          # [E,T,4]

E, T = done.shape
eps = []
for e in range(E):
    starts = np.nonzero(ini[e])[0]
    if starts.size == 0:
        continue
    ends = np.nonzero(done[e])[0]
    if ends.size == 0:
        continue
    for s in starts:
        after = ends[ends >= s]
        if after.size:
            eps.append((e, int(s), int(after[0])))

print(f"[{label}] 完整回合数 = {len(eps)} / {E} 行 (T={T})")

s_np = state.float().numpy()             # [E,T,23]  (加载较慢但一次性)
sig = {}
for (e, a, b) in eps:
    prog = float(s_np[e, b, 5])
    trem = float(s_np[e, b, 22]) * 20.0
    r = rew[e, b]                        # [4] A1,A2,D1,D2
    key = (round(prog, 2), round(trem, 0), round(float(r[0]), 0))
    d = sig.setdefault(key, [0, np.zeros(4)])
    d[0] += 1
    d[1] += r

print(f"{'prog':>5} {'t_rem':>6} {'rA1':>6} {'n':>5}  {'rA2':>7} {'rD1':>7} {'rD2':>7}")
for key, (n, rsum) in sorted(sig.items(), key=lambda kv: -kv[1][0])[:22]:
    rm = rsum / n
    print(f"{key[0]:>5.2f} {key[1]:>6.0f} {key[2]:>6.0f} {n:>5}  {rm[1]:>7.1f} {rm[2]:>7.1f} {rm[3]:>7.1f}")
print(f"[{label}] 唯一签名数 = {len(sig)}")
