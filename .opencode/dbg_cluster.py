"""聚类调试：终局步 (A1奖励, 防守均值奖励) 2D 直方图，映射终局码簇。
用法: python dbg_cluster.py <ckpt_path>
"""
import collections
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=False)
st = ck["buffer_agents"]["_storage"]["_storage"]
obs = st["agents"]["observation"].float()
rew = st["next"]["agents"]["reward"].float()[..., 0]
done = st["next"]["done"][..., 0]

idx = torch.nonzero(done)
e, t = idx[:, 0], idx[:, 1]
pv = obs[e, t, 0, 7]
r1 = rew[e, t, 0]
rd = rew[e, t, 2:4].mean(-1)
print("terminals:", len(e))

cnt2 = collections.Counter(zip((r1 / 20).round().int().tolist(), (rd / 10).round().int().tolist()))
print("top (r1b20, rdb10) -> n:")
for (a, b), n in sorted(cnt2.items(), key=lambda kv: -kv[1])[:28]:
    print(f"  r1~{a*20:+4d} rd~{b*10:+4d} : n={n}")

print("--- 按 rd 分箱 ---")
for lo, hi, name in [(-1e9, -20, "rd<-20"), (-20, 0, "-20..0"), (0, 20, "0..20"), (20, 30, "20..30"), (30, 40, "30..40"), (40, 1e9, "rd>=40")]:
    m = (rd >= lo) & (rd < hi)
    if int(m.sum()) == 0:
        continue
    h = collections.Counter((r1[m] / 20).round().int().tolist())
    print(f"{name}: n={int(m.sum())} r1mean={float(r1[m].mean()):.1f} r1hist={sorted(h.items())}")

print("--- attempts (prog==0.9) ---")
m = (pv - 0.9).abs() < 1e-4
print("n=", int(m.sum()), "r1 hist:", sorted(collections.Counter((r1[m] / 20).round().int().tolist()).items()),
      "rdmean=", round(float(rd[m].mean()), 1))
print("--- non-attempts ---")
m2 = ~m
print("n=", int(m2.sum()), "r1 hist:", sorted(collections.Counter((r1[m2] / 20).round().int().tolist()).items()))
