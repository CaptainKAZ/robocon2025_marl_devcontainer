"""调试：终局步进度/按键/奖励联合分布 + 回合长度 + 首次过线步分布。
用法: python dbg_prescreen.py <ckpt_path>
"""
import collections
import statistics
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=False)
st = ck["buffer_agents"]["_storage"]["_storage"]
obs = st["agents"]["observation"].float()
rew = st["next"]["agents"]["reward"].float()[..., 0]
done = st["next"]["done"][..., 0]
init = st["is_init"][..., 0]
act_d = st["agents"]["action"]["discrete"]

E, T = done.shape
print("E,T=", E, T)
idx = torch.nonzero(done)
e, t = idx[:, 0], idx[:, 1]
pv = obs[e, t, 0, 7]
r1 = rew[e, t, 0]
rd = rew[e, t, 2:4].mean(-1)
pr = act_d[e, t, 0]
print("terminals:", len(e))

cnt = collections.Counter((pv * 20).round().clamp(0, 20).int().tolist())
print("progress hist (x0.05):", sorted(cnt.items()))
print("press rate at terminal:", round(float(pr.float().mean()), 3))

for lo, hi, name in [(60, 1e9, "r1>=60"), (20, 60, "20<=r1<60"), (-20, 20, "-20<=r1<20"), (-1e9, -20, "r1<-20")]:
    m = (r1 >= lo) & (r1 < hi)
    print(f"{name}: n={int(m.sum())} rd_mean={float(rd[m].mean()):.1f} rd_p50={float(rd[m].median()):.1f} "
          f"press={float(pr[m].float().mean()):.2f} prog_mean={float(pv[m].mean()):.3f}")

m = rd >= 30
print("rd>=30: n=", int(m.sum()), "r1 mean", round(float(r1[m].mean()), 1),
      "r1/20 bins:", sorted(collections.Counter((r1[m] / 20).round().int().tolist()).items()))

# 回合段长度（真实起点=行首+每个 init）
seg_lens = []
for ee in range(E):
    s = 0
    for tt_ in range(T):
        if init[ee, tt_]:
            if tt_ > s:
                seg_lens.append(tt_ - s)
            s = tt_
    seg_lens.append(T - s)
print("segments:", len(seg_lens), "mean len", round(statistics.mean(seg_lens), 1),
      "p50", statistics.median(seg_lens),
      "share>=150:", round(sum(1 for x in seg_lens if x >= 150) / len(seg_lens), 3))

# 段起点 A1 位置
start_pos = []
for ee in range(0, E, 100):
    for tt_ in range(T):
        if tt_ == 0 or init[ee, tt_]:
            start_pos.append(obs[ee, tt_, 0, 0:2].tolist())
            break
sp = torch.tensor(start_pos)
print("A1 start pos sample n=", len(start_pos), "mean", sp.mean(0).tolist())

# 首次过线步（段长>=40）
cross = []
for ee in range(E):
    s = 0
    for tt_ in range(T + 1):
        if tt_ == T or (tt_ > s and init[ee, tt_]):
            L = tt_ - s
            if L >= 40:
                ys = obs[ee, s:tt_, 0, 1] * 7.5
                nz = (ys > 0).nonzero()
                cross.append(int(nz[0]) if len(nz) else -1)
            s = tt_
vals = [c for c in cross if c >= 0]
print("segments>=40:", len(cross), "crossed:", len(vals), "never:", len(cross) - len(vals))
if vals:
    import numpy as np
    v = np.array(vals)
    print("first-cross step p10/25/50/75/90:", np.percentile(v, [10, 25, 50, 75, 90]).round(1).tolist())
    for b in [(0, 39), (40, 79), (80, 119), (120, 199)]:
        print(f"  cross in {b}: {int(((v >= b[0]) & (v <= b[1])).sum())}")
