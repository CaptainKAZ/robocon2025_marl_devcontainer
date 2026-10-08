"""投篮按键：环境级冒烟（动作 spec / mask / 按下降刹车 / 按住读条出手 / 非 A1 忽略）

跑法（容器内）：
  cd /home/vscode/workspace/BenchMARL && python /tmp/probe_button_env.py
"""
import os
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "1.2")

import torch

from benchmarl.environments import LayupTask


def show_spec(spec, prefix=""):
    for k in spec.keys():
        v = spec[k]
        if hasattr(v, "keys") and not hasattr(v, "space"):
            print(f"{prefix}{k}/")
            show_spec(v, prefix + "  ")
        else:
            print(f"{prefix}{k}: {type(v).__name__} shape={tuple(getattr(v, 'shape', ()))}")


def main():
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=4, continuous_actions=True, seed=0, device="cpu")()

    print("=" * 70)
    print("[1] action spec（应当只有 agents 组；continuous 2 维 + discrete 1 维）")
    print("=" * 70)
    show_spec(env.full_action_spec_unbatched)

    print()
    print("=" * 70)
    print("[2] reset -> mask")
    print("=" * 70)
    td = env.reset()
    for group, item in td.items() if hasattr(td, "items") else []:
        pass
    mask = td.get(("agents", "action_mask"))
    print("mask shape:", tuple(mask.shape), "dtype:", mask.dtype)
    print("mask[0]:\n", mask[0])
    print("A1 可按(初始，应当 False):", bool(mask[0, 0, 1]))

    sc = env.scenario
    E = 4
    # 把 A1 放到投篮点中心（并清速），其余 agent 保持
    for b in range(E):
        sc.a1.set_pos(sc.spot_center.state.pos[b].clone(), batch_index=b)
        sc.a1.set_vel(torch.zeros(2), b)
    sc.a1_still_frames_counter[:] = 0
    sc.a1_press[:] = False
    sc.p_vels.zero_()

    print()
    print("=" * 70)
    print("[3] 不按按键：12 帧后读条应仍为 0、无出手")
    print("=" * 70)
    for f in range(12):
        td.set(("agents", "action", "continuous"), torch.zeros(E, 4, 2))
        td.set(("agents", "action", "discrete"), torch.zeros(E, 4, 1, dtype=torch.long))
        out = env.step(td)
        td = out
        tr = out.get(("next", "agents", "info", "termination_reason")) if ("next", "agents", "info", "termination_reason") in out.keys(True) else None
        if f in (0, 1, 11):
            print(f"  帧 {f}: counter={sc.a1_still_frames_counter[0].item()} press={bool(sc.a1_press[0])} maskA1可按={bool(out[('agents','action_mask')][0,0,1])}")
    print("  12 帧后 counter:", sc.a1_still_frames_counter.tolist())

    print()
    print("=" * 70)
    print("[4] 按住按键：读条应逐帧 +1，第 10 帧出手（终局）")
    print("=" * 70)
    for b in range(E):
        sc.a1.set_pos(sc.spot_center.state.pos[b].clone(), batch_index=b)
        sc.a1.set_vel(torch.zeros(2), b)
    sc.a1_still_frames_counter[:] = 0
    sc.p_vels.zero_()
    # 让防守者远离投篮线，避免被判封盖以外的干扰
    done_frames = []
    for f in range(14):
        td.set(("agents", "action", "continuous"), torch.zeros(E, 4, 2))
        disc = torch.zeros(E, 4, 1, dtype=torch.long)
        disc[:, 0, 0] = 1  # A1 按住
        td.set(("agents", "action", "discrete"), disc)
        out = env.step(td)
        td = out
        cnt = sc.a1_still_frames_counter[0].item()
        fired = bool(out.get(("next", "agents", "info", "win_in_step"))[0, 0].item()) if ("next", "agents", "info", "win_in_step") in out.keys(True) else None
        print(f"  帧 {f}: counter={cnt} fired(win_in_step)={fired}")
        if fired:
            done_frames.append(f)
            break
    codes = out.get(("next", "agents", "info", "termination_reason"))[0, :, 0].tolist() if ("next", "agents", "info", "termination_reason") in out.keys(True) else None
    print("  终局码:", codes)

    print()
    print("=" * 70)
    print("[5] 按下瞬间刹车：给 A1 一个速度，按下后应当为 0")
    print("=" * 70)
    env2 = task.get_env_fun(num_envs=2, continuous_actions=True, seed=1, device="cpu")()
    td2 = env2.reset()
    sc2 = env2.scenario
    for b in range(2):
        sc2.a1.set_pos(sc2.spot_center.state.pos[b].clone(), batch_index=b)
        sc2.a1.set_vel(torch.zeros(2), b)
    # 先空走一步，让 is_in_spot_a1 更新为 True（它在 pre_step 里刷新）
    td2.set(("agents", "action", "continuous"), torch.zeros(2, 4, 2))
    td2.set(("agents", "action", "discrete"), torch.zeros(2, 4, 1, dtype=torch.long))
    td2 = env2.step(td2)
    print("  空走一步后 is_in_spot_a1:", sc2.is_in_spot_a1.tolist(),
          "maskA1可按:", bool(td2[("agents", "action_mask")][0, 0, 1]))
    for b in range(2):
        sc2.a1.set_vel(torch.tensor([4.0, 3.0]), b)
    print("  按下前 vel:", sc2.a1.state.vel[0].tolist())
    td2.set(("agents", "action", "continuous"), torch.zeros(2, 4, 2))
    disc = torch.zeros(2, 4, 1, dtype=torch.long)
    disc[:, 0, 0] = 1
    td2.set(("agents", "action", "discrete"), disc)
    env2.step(td2)
    print("  按下后 vel:", sc2.a1.state.vel[0].tolist(), "（应当 ≈0）")

    print()
    print("=" * 70)
    print("[6] 非 A1 按下（第 3 通道=1）：应被忽略（A2/D 不动、不推进 A1 读条）")
    print("=" * 70)
    sc.a1_still_frames_counter[:] = 0
    td.set(("agents", "action", "continuous"), torch.zeros(E, 4, 2))
    disc = torch.zeros(E, 4, 1, dtype=torch.long)
    disc[:, 1, 0] = 1
    disc[:, 2, 0] = 1
    td.set(("agents", "action", "discrete"), disc)
    env.step(td)
    print("  A1 counter（应 0）:", sc.a1_still_frames_counter.tolist())

    print()
    print("BUTTON ENV PROBE DONE")
    env.close()
    env2.close()


if __name__ == "__main__":
    main()
