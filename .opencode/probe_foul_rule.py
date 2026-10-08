"""[方案A / A+] 碰撞犯规门槛与"合法防守位豁免"的对照验证。

各案例占一个 batch（n_envs=B），手动摆位 + 直接写 p_vels（判罚用的就是这个量，
= 上一物理步结束时的速度），跑一步看终局码与奖励。
判责数学（layup_jit.py 条件3）：
    approach_i = clamp(v_i·n, 0), approach_j = clamp(-v_j·n, 0)
    approach_max = max(approach_i, approach_j)                 # 门槛用它
    对称基线: active = argmax(approach)；平手 -> |v| 大者
    合法防守位豁免(仅跨队): violation_D = max(0,|v_D|-v_legal) + max(0,approach_D-a_legal)
                          score_A = approach_A - k_exempt * violation_D
                          score_A <= approach_D  ->  责任归防守方
"""
import os, torch
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
from benchmarl.environments import LayupTask

NAMES = ["A1", "A2", "D1", "D2"]

# A1 在原点，D1 在 (0,0.5)（接触，n = A1->D1 = +y）
# id, A1vel, D1vel
CASES = [
    ("b0 擦身横滑（反向侧擦）", (2.0, 0.0), (-2.0, 0.0)),
    ("b1 撞站定防守（合法占位）", (0.0, 1.0), (0.0, 0.0)),
    ("b2 追尾（防守慢速后撤 0.25）", (0.0, 1.0), (0.0, 0.25)),
    ("b3 防守迎面撞（A1 站定）", (0.0, 0.0), (0.0, 1.0)),
    ("b4 防守横向移动 1.5（非法占位）", (0.0, 1.0), (1.5, 0.0)),
    ("b5 防守同向后撤 1.0（非法占位）", (0.0, 1.0), (0.0, 1.0)),
    ("b6 A1 侧向、防守慢速 0.25", (1.5, 0.0), (0.0, 0.25)),
]
B = len(CASES)


def set_action(td):
    td.set(("agents", "action", "continuous"), torch.zeros(B, 4, 2))
    td.set(("agents", "action", "discrete"), torch.zeros(B, 4, 1, dtype=torch.long))
    return td


def main():
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=B, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    sc = env.scenario
    hp = sc.h_params
    print(f"[cfg] approach_th={hp['foul_approach_threshold']}  v_legal={hp['foul_legal_def_speed']}  "
          f"a_legal={hp['foul_legal_def_approach']}  k_exempt={hp['k_legal_def_exempt']}")
    td = env.reset()
    for b, (_, a1v, d1v) in enumerate(CASES):
        place = {0: (torch.tensor([0.0, 0.0]), torch.tensor(a1v)),
                 2: (torch.tensor([0.0, 0.5]), torch.tensor(d1v)),
                 1: (torch.tensor([-3.0, -5.0]), torch.zeros(2)),
                 3: (torch.tensor([3.0, 5.0]), torch.zeros(2))}
        for i, (xy, v) in place.items():
            ag = env.world.agents[i]
            ag.set_pos(xy, batch_index=b)
            ag.set_vel(v, batch_index=b)
    for b, (_, a1v, d1v) in enumerate(CASES):
        sc.p_vels[b, 0] = torch.tensor(a1v, dtype=sc.p_vels.dtype)
        sc.p_vels[b, 2] = torch.tensor(d1v, dtype=sc.p_vels.dtype)
    td = env.step(set_action(td))
    code = td.get(("next", "agents", "info", "termination_reason"))[..., 0, 0].tolist()
    rw = td.get(("next", "agents", "reward")).reshape(B, -1).tolist()

    print("\n-- 结果 --")
    for b, (name, a1v, d1v) in enumerate(CASES):
        vi, vj = torch.tensor(a1v), torch.tensor(d1v)
        n = torch.tensor([0.0, 1.0])
        ai = max(0.0, float((vi * n).sum()))
        aj = max(0.0, float(-(vj * n).sum()))
        dv = float((vi - vj).norm())
        amax = max(ai, aj)
        viol = max(0.0, float(vj.norm()) - hp["foul_legal_def_speed"]) + \
            max(0.0, aj - hp["foul_legal_def_approach"])
        score_a = ai - hp["k_legal_def_exempt"] * viol
        exempt = score_a <= aj
        gate = amax > hp["foul_approach_threshold"]
        who = "D1（进攻方豁免->码2）" if (exempt and gate) else ("A1（码13）" if gate else "不判（擦身/轻碰）")
        print(f"  {name}")
        print(f"     |Δv|={dv:4.2f} a_A1={ai:4.2f} a_D1={aj:4.2f} a_max={amax:4.2f} | "
              f"violation={viol:4.2f} score_A={score_a:5.2f} -> 责任={who}")
        print(f"     实测 终局码={code[b]} 奖励={[round(x, 2) for x in rw[b]]}")
    env.close()


if __name__ == "__main__":
    main()
