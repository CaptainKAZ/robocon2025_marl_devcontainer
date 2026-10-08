"""投篮按键：mask 写入口径排查"""
import os
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "1.2")
import torch
from benchmarl.environments import LayupTask


def main():
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=2, continuous_actions=True, seed=0, device="cpu")()
    base = getattr(env, "_env", None)
    print("env type:", type(env).__name__, "| base type:", type(base).__name__)
    print("base is VmasEnvWithState:", hasattr(base, "_write_action_mask"))

    td = env.reset()
    sc = env.scenario
    for b in range(2):
        sc.a1.set_pos(sc.spot_center.state.pos[b].clone(), batch_index=b)
        sc.a1.set_vel(torch.zeros(2), b)

    def step_zero(td):
        td.set(("agents", "action", "continuous"), torch.zeros(2, 4, 2))
        td.set(("agents", "action", "discrete"), torch.zeros(2, 4, 1, dtype=torch.long))
        return env.step(td)

    for f in range(3):
        td = step_zero(td)
        sc_mask = sc.get_action_mask()
        td_mask = td[("agents", "action_mask")]
        print(
            f"帧 {f}: is_in_spot={sc.is_in_spot_a1.tolist()} | "
            f"scenario.get_action_mask A1可按={sc_mask[0,0,1].item()} | "
            f"td[agents,action_mask] A1可按={td_mask[0,0,1].item()} "
            f"sum={int(td_mask.sum())} same_obj={sc_mask.data_ptr()==td_mask.data_ptr()}"
        )

    print("td keys:", list(td.keys()))
    print("group td keys:", list(td[("agents",)].keys()))
    print("BUTTON MASK DBG DONE")
    env.close()


if __name__ == "__main__":
    main()
