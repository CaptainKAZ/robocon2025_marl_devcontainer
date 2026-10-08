"""[读条期触碰] 验证：只有"真正碰到 A1 的防守者"被罚 -R_foul，另一名防守者不受影响。

用法: python /tmp/probe_charge_foul.py
"""
import os, torch
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
from benchmarl.environments import LayupTask

B = 3
# c0: 只有 D1 碰到 A1；c1: D1/D2 都碰到；c2: 都不碰（不触发）
CASES = [
    ("c0 只有 D1 碰到 A1（读条中）", True, False),
    ("c1 D1 与 D2 都碰到 A1（读条中）", True, True),
    ("c2 无人碰到（对照）", False, False),
]


def set_action(td, press_a1):
    td.set(("agents", "action", "continuous"), torch.zeros(B, 4, 2))
    disc = torch.zeros(B, 4, 1, dtype=torch.long)
    if press_a1:
        disc[:, 0, 0] = 1
    td.set(("agents", "action", "discrete"), disc)
    return td


def place(env, b, spot, d1_on, d2_on):
    def put(i, xy, v):
        ag = env.world.agents[i]
        ag.set_pos(torch.tensor(xy, dtype=torch.float32), batch_index=b)
        ag.set_vel(torch.tensor(v, dtype=torch.float32), batch_index=b)
    put(0, spot, (0.0, 0.0))
    put(1, (spot[0] - 1.0, spot[1] - 2.0), (0.0, 0.0))
    put(2, (spot[0] + 0.5, spot[1]) if d1_on else (spot[0] - 1.0, spot[1] + 2.0), (0.0, 0.0))
    put(3, (spot[0], spot[1] - 0.5) if d2_on else (spot[0] + 1.0, spot[1] + 2.0), (0.0, 0.0))


def main():
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=B, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    sc = env.scenario
    td = env.reset()
    spots = []
    for b in range(B):
        c = sc.spot_center.state.pos[b].tolist()
        spots.append((float(c[0]), float(c[1])))
        place(env, b, spots[b], CASES[b][1], CASES[b][2])
    td = env.step(set_action(td, press_a1=False))   # 暖机：刷新 is_in_spot_a1
    for b in range(B):
        place(env, b, spots[b], CASES[b][1], CASES[b][2])   # 位置不变，重申一遍保险
        sc.a1_still_frames_counter[b] = 5                   # 已读条 5 帧
    td = env.step(set_action(td, press_a1=True))
    code = td.get(("next", "agents", "info", "termination_reason"))[..., 0, 0].tolist()
    rw = td.get(("next", "agents", "reward")).reshape(B, -1).tolist()
    print(f"[cfg] R_foul={sc.h_params['R_foul']}  max_score={sc.h_params['max_score']}")
    for b, (name, d1, d2) in enumerate(CASES):
        print(f"  {name}")
        print(f"     终局码={code[b]}  奖励 A1/A2/D1/D2 = {[round(x, 2) for x in rw[b]]}")
    env.close()


if __name__ == "__main__":
    main()
