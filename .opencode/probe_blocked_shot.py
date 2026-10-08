"""被盖惩罚对照（按键时代版）：A1 用 k_blocked_shot_penalty，A2 用 k_blocked_shot_penalty_a2。

用手动摆位制造"封盖"来验证两个参数各自生效，不改奖励代码。
坑：动作 td 必须在已 reset 的 td 上改，不能每步都 env.reset()（会清掉读条计数）。
"""
import os, torch
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
from benchmarl.environments import LayupTask

task = LayupTask.LAYUP.get_from_yaml()
NAMES = ["A1", "A2", "D1", "D2"]


def set_action(td, press=False):
    td.set(("agents", "action", "continuous"), torch.zeros(1, 4, 2))
    disc = torch.zeros(1, 4, 1, dtype=torch.long)
    disc[0, 0, 0] = 1 if press else 0          # 只有 A1 有按键
    td.set(("agents", "action", "discrete"), disc)
    return td


def run(tag, a1_penalty, blocker=True, a2_penalty=None):
    env = task.get_env_fun(num_envs=1, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    sc = env.scenario
    td = env.reset()
    sc.h_params["k_blocked_shot_penalty"] = a1_penalty
    if a2_penalty is not None:
        sc.h_params["k_blocked_shot_penalty_a2"] = a2_penalty
    basket = sc.basket.state.pos[0].clone().float()
    spot = sc.spot_center.state.pos[0].clone().float()
    unit = (basket - spot) / (basket - spot).norm()
    d1_xy = (spot + 0.7 * unit) if blocker else (spot + 2.5 * unit)
    d2_xy = spot + torch.tensor([3.0, -2.5])
    a2_xy = spot + torch.tensor([-3.0, -0.5])
    if a2_xy[1] <= 0:
        a2_xy[1] = 0.5
    place = {0: spot, 1: a2_xy, 2: d1_xy, 3: d2_xy}
    for i, xy in place.items():
        ag = env.world.agents[i]
        ag.set_pos(xy, batch_index=0)
        ag.set_vel(torch.zeros(2), batch_index=0)

    # 空走一步：刷新 is_in_spot（按键有效性依赖"上一帧在圈内"）
    td = env.step(set_action(td, press=False))
    for i, xy in place.items():                      # 复位（空走可能让位置漂一点点）
        ag = env.world.agents[i]
        ag.set_pos(xy, batch_index=0)
        ag.set_vel(torch.zeros(2), batch_index=0)
    sc.a1_still_frames_counter[0] = 9                # 再按一下 = 第 10 帧出手
    try:
        sc.p_vels.zero_()
    except Exception:
        pass

    td = env.step(set_action(td, press=True))
    code = td.get(("next", "agents", "info", "termination_reason")).flatten().tolist()
    tr = td.get(("next", "agents", "info", "terminal_reward")).flatten().tolist()
    rw = td.get(("next", "agents", "reward")).flatten().tolist()
    k1 = sc.h_params["k_blocked_shot_penalty"]
    k2 = sc.h_params["k_blocked_shot_penalty_a2"]
    print(f"[{tag}] 码={[int(c) for c in code]}  k_A1={k1:.0f} k_A2={k2:.0f}")
    print(f"[{tag}] terminal_reward: " + " ".join(f"{n}={v:+.2f}" for n, v in zip(NAMES, tr)))
    print(f"[{tag}] step_reward:     " + " ".join(f"{n}={v:+.2f}" for n, v in zip(NAMES, rw)))


run("封盖 A1=12000 A2=18000", 12000.0, blocker=True, a2_penalty=18000.0)
run("封盖 A1=0     A2=18000", 0.0, blocker=True, a2_penalty=18000.0)
run("封盖 A1=12000 A2=12000（旧）", 12000.0, blocker=True, a2_penalty=12000.0)
run("空位投 A1=12000 A2=18000", 12000.0, blocker=False, a2_penalty=18000.0)
print("BLOCKED SHOT PROBE DONE")
