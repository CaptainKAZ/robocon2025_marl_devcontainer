"""投篮按键：mask 写入是否随 step 刷新（根键 + next 键）"""
import os
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "1.2")
import torch
from benchmarl.environments import LayupTask


def main():
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=2, continuous_actions=True, seed=0, device="cpu")()
    td = env.reset()
    sc = env.scenario
    for b in range(2):
        sc.a1.set_pos(sc.spot_center.state.pos[b].clone(), batch_index=b)
        sc.a1.set_vel(torch.zeros(2), b)

    def step(td):
        td.set(("agents", "action", "continuous"), torch.zeros(2, 4, 2))
        td.set(("agents", "action", "discrete"), torch.zeros(2, 4, 1, dtype=torch.long))
        return env.step(td)

    for f in range(3):
        td = step(td)
        root = td[("agents", "action_mask")][0, 0, 1].item()
        nxt = td[("next", "agents", "action_mask")][0, 0, 1].item()
        print(f"帧{f}: is_in_spot={sc.is_in_spot_a1.tolist()} "
              f"scenario_mask={sc.get_action_mask()[0, 0, 1].item()} "
              f"root_mask={root} next_mask={nxt}")

    print("DBG2 DONE")
    env.close()


if __name__ == "__main__":
    main()
