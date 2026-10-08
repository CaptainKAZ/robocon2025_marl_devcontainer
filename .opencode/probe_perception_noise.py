"""[感知噪声] 验证"队友/对手观测"的误差随距离增大，且与模型一致。

用法: python /tmp/probe_perception_noise.py [k]   # k: 位置噪声系数（默认取 h_params）
观测块布局（每个非自身实体 8 维）:
    rel_pos/[W, L]  rel_vel/(2*v_max)  abs_pos/[W/2, L/2]  abs_vel/v_max
队友 9:17 / opp1 17:25 / opp2 25:33  ->  abs_pos 分别 13:15 / 21:23 / 29:31
"""
import os, sys, torch
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
from benchmarl.environments import LayupTask

B, STEPS = 96, 60
ABS_DIV = torch.tensor([4.0, 7.5])
BLOCKS = {"teammate": (9, 13), "opp1": (17, 21), "opp2": (25, 29)}   # (块起点, abs_pos 起点)


def main():
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=B, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    sc = env.scenario
    hp = sc.h_params
    print(f"[cfg] k_pos={hp['k_perception_noise']} floor_pos={hp['perception_noise_floor']} "
          f"k_vel={hp['k_perception_noise_vel']} floor_vel={hp['perception_noise_vel_floor']}")
    others = {"teammate": None, "opp1": None, "opp2": None}

    td = env.reset()
    rec = {k: {"d": [], "e": []} for k in BLOCKS}
    for _ in range(STEPS):
        td.set(("agents", "action", "continuous"), torch.randn(B, 4, 2) * 0.4)
        td.set(("agents", "action", "discrete"), torch.zeros(B, 4, 1, dtype=torch.long))
        td = env.step(td)
        obs = td.get(("next", "agents", "observation"))
        for a in range(4):
            if a < 2:
                names = {"teammate": 1 - a, "opp1": 2, "opp2": 3}
            else:
                names = {"teammate": 3 if a == 2 else 2, "opp1": 0, "opp2": 1}
            self_pos = env.world.agents[a].state.pos
            for key, (blk, absp) in BLOCKS.items():
                idx = names[key]
                seen = obs[:, a, absp:absp + 2] * ABS_DIV
                true = env.world.agents[idx].state.pos
                rec[key]["d"].append((true - self_pos).norm(dim=-1))
                rec[key]["e"].append((seen - true).norm(dim=-1))

    print("\n-- 观测位置误差 vs 距离 --")
    for key in BLOCKS:
        d = torch.cat(rec[key]["d"]); e = torch.cat(rec[key]["e"])
        print(f"  [{key}] n={d.numel()}  |误差| mean={e.mean():.4f} m")
        edges = [(0, 1), (1, 2), (3, 5), (5, 8), (8, 12)]
        for lo, hi in edges:
            m = (d >= lo) & (d < hi)
            if m.sum() < 20:
                continue
            pred = hp["perception_noise_floor"] + hp["k_perception_noise"] * d[m].mean()
            # 2D 高斯 -> |误差| 的均值 = sigma * sqrt(pi/2)
            print(f"    d∈[{lo},{hi}) n={int(m.sum()):6d}  d_mean={d[m].mean():5.2f}  "
                  f"实测|误差|={e[m].mean():.4f}  模型预测={pred * 1.2533:.4f}")
    env.close()


if __name__ == "__main__":
    main()
