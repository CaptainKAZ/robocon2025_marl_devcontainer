"""v37（无按键、旧映射）buffer 控制/行为探针：|u| 命令分布、旧映射生效速度、实际速度、过线时机、读条占比。
用法: python probe_v37_control.py <ckpt_path>
"""
import sys

import torch

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False, mmap=False)
buf = ck["buffer_agents"]
st = buf["_storage"]["_storage"]


def tree(d, p="", depth=0):
    if depth > 3 or not hasattr(d, "keys"):
        return
    for kk in d.keys():
        v = d[kk]
        if hasattr(v, "shape"):
            print(f"{p}{kk}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"{p}{kk}: {type(v).__name__}")
            tree(v, p + str(kk) + ".", depth + 1)


tree(st)


def find(d, key, depth=0):
    if depth > 3 or not hasattr(d, "keys"):
        return None
    if key in d.keys():
        return d[key]
    for kk in d.keys():
        r = find(d[kk], key, depth + 1)
        if r is not None:
            return r
    return None


obs = find(st, "observation")
act = find(st, "action")
init = find(st, "is_init")
print("obs:", None if obs is None else tuple(obs.shape), "act:", None if act is None else tuple(act.shape))
if obs is None or act is None:
    sys.exit(2)
obs = obs.float()
act = act.float()
u = act[:, :, 0, :]
mag = u.norm(dim=-1)
print(f"[control] |u| mean={mag.mean():.3f} p50={mag.median():.3f} p90={mag.quantile(0.9):.3f} max={mag.max():.2f}")
v_old = u * (mag / 5.0).clamp(min=0).pow(0.5).unsqueeze(-1)
n_old = v_old.norm(dim=-1)
dz = n_old < 0.2
eff = torch.where(dz, torch.zeros_like(n_old), n_old)
print(f"[control] 旧映射生效速度 mean={eff.mean():.3f} p50={eff.median():.3f} | 死区占比={dz.float().mean():.3f} | >=1.0 m/s 占比={(n_old >= 1.0).float().mean():.3f}")
spd = obs[:, :, 0, 2:4].norm(dim=-1) * 5.0
print(f"[control] 实际速度 mean={spd.mean():.3f} p50={spd.median():.3f}")
pos = obs[:, :, 0, 0:2] * torch.tensor([4.0, 7.5])
E, T = pos.shape[0], pos.shape[1]
if init is not None:
    init = init[..., 0].bool() if init.dim() == 3 else init.bool()
else:
    init = torch.zeros(E, T, dtype=torch.bool)
cross, never, n4, c4 = [], 0, 0, 0
for e in range(E):
    s = 0
    for t in range(T + 1):
        if t == T or (t > s and init[e, t]):
            L = t - s
            if L >= 40:
                ys = pos[e, s:t, 1]
                nz = (ys > 0).nonzero()
                if len(nz):
                    fc = int(nz[0])
                    cross.append(fc)
                    if fc < 40:
                        c4 += 1
                else:
                    never += 1
                n4 += 1
            s = t
print(f"[cross] 段>=40: {n4} 过线: {len(cross)} 从不过线: {never} 4s内过线: {c4}")
if cross:
    v = torch.tensor(cross).float()
    print("first-cross p25/50/75:", [round(x, 1) for x in torch.quantile(v, torch.tensor([0.25, 0.5, 0.75])).tolist()])
in_spot = obs[:, :, 0, 6] > 0.5
prog1 = obs[:, :, 0, 7]
print(f"[behavior] 圈内占比={in_spot.float().mean():.3f} 读条中占比={(prog1 > 0).float().mean():.3f}")

# 奖励稀释对照
r_all = st["next"]["agents"]["reward"].float()[..., 0]  # (E,T,4)
r1 = r_all[:, :, 0]
print(f"[reward] A1 每步 mean={r1.mean():.4f} p50={r1.median():.4f} | 全体每步 mean={r_all.mean():.4f}")
done = st["next"]["done"][..., 0].bool()
rm = r1[done]
print(f"[reward] 终局一步 A1 mean={rm.mean():.3f} p50={rm.median():.3f} (n={rm.numel()})")
tot = []
for e in range(E):
    s = 0
    for t in range(T + 1):
        if t == T or (t > s and init[e, t]):
            if t - s >= 40:
                tot.append(r1[e, s:t].sum().item())
            s = t
tot_t = torch.tensor(tot)
print(f"[reward] 完整段 A1 总奖励 mean={tot_t.mean():.3f} p50={tot_t.median():.3f} (n={tot_t.numel()})")
