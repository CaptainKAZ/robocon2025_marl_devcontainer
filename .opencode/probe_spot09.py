"""R_spot 0.9 变更的环境级验证探针。

检查：
  ① h_params['R_spot'] == 0.9 及全部派生量（gaussian_sigma=0.45 / screen_pos_sigma=0.9 /
     def_gaussian_spot_sigma=0.9 / repulsion_proximity_threshold=0.9）
  ② spot 采样边界：y ∈ [0.9, 4.65]、|x| ≤ 3.55，且圈不越出场地（spot_y - R >= 0）
  ③ is_in_spot 边界：A1 距圈心 0.85m → 在圈内；0.95m → 在圈外（旧 1.2 判定应失效）
  ④ 按键 mask 与 is_in_spot 一致
用法（容器内，工作目录 BenchMARL）：python /tmp/probe_spot09.py
"""
import os
import torch

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

from benchmarl.environments import LayupTask  # noqa: E402

E = 8
task = LayupTask.LAYUP.get_from_yaml()
env = task.get_env_fun(num_envs=E, continuous_actions=True, seed=0, device=torch.device("cpu"))()

scen = None
for path in ("scenario", "unwrapped.scenario", "_env.scenario", "env.scenario"):
    o = env
    ok = True
    for p in path.split("."):
        if hasattr(o, p):
            o = getattr(o, p)
        else:
            ok = False
            break
    if ok:
        scen = o
        break
assert scen is not None, "scenario 未找到"
hp = scen.h_params
R = float(hp["R_spot"])
print(f"[1] R_spot = {R}")
for k in ("gaussian_sigma", "screen_pos_sigma", "def_gaussian_spot_sigma", "repulsion_proximity_threshold"):
    print(f"    {k} = {float(hp[k])}")

td = env.reset()
try:
    A0 = env.action_spec["agents", "action"].zero()
    print(f"[dbg] 动作模板 = {type(A0).__name__}")
except Exception as e:  # noqa: BLE001
    print(f"[dbg] spec.zero() 失败: {e}")
    A0 = None


def zero_action(a):
    if a is None:
        return None
    if hasattr(a, "keys"):  # 复合 TensorDict
        for k in a.keys():
            a[k].zero_()
    else:
        a.zero_()
    return a


# ---- ② 采样边界（多次 reset）----
ys, xs, margins = [], [], []
for r in range(5):
    td = env.reset()
    sp = scen.spot_center.state.pos.detach().clone()  # [E,2]
    ys.append(sp[:, 1]); xs.append(sp[:, 0]); margins.append(sp[:, 1] - R)
ys = torch.cat(ys); xs = torch.cat(xs); margins = torch.cat(margins)
print(f"[2] spot 采样 x∈[{xs.min():+.3f},{xs.max():+.3f}] y∈[{ys.min():.3f},{ys.max():.3f}] "
      f"| y-R min={margins.min():.4f} (>=0 表示圈不出场) | n={ys.numel()}")

# ---- ③ is_in_spot 边界 + ④ mask ----
sp = scen.spot_center.state.pos
# 选一个圈心 y 足够高的 world（避免 y>0 门控干扰）
cands = (sp[:, 1] > 2.0).nonzero().flatten()
i = int(cands[0]) if len(cands) else 0
c = sp[i].clone()
print(f"[3] 测试 world #{i}：spot=({c[0]:+.3f},{c[1]:+.3f})")

a1 = scen.world.agents[0]


def place_and_step(dist):
    a1.set_pos(c + torch.tensor([dist, 0.0]), batch_index=i)
    a1.set_vel(torch.zeros(2), batch_index=i)
    t = td.copy() if hasattr(td, "copy") else td
    act = zero_action(A0.clone()) if A0 is not None else None
    if act is not None:
        t.set(("agents", "action"), act)
    t = env.step(t)
    in_spot = bool(scen.is_in_spot_a1[i])
    mask = scen.get_action_mask()
    m = bool(mask[i, 0, 1])
    print(f"    距圈心 {dist:.2f}m -> is_in_spot={in_spot} | mask[可按]={m} | 一致={in_spot == m}")
    return in_spot, m


for d in (0.85, 0.95):
    place_and_step(d)

print("[4] L300 探针结束")
