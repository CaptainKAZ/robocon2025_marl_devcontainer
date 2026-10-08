"""[判罚经济取证] 用当前策略跑一个真实采集批，按**真实终局码**统计碰撞类终局的几何与接近分量，
量化"守方被判撞人"的代价结构，并给出门槛收紧/放宽的一阶影响估计。

为什么要跑真实 rollout：replay buffer 里没有 termination_reason（被 `_get_excluded_keys` 排除），
只有采集批里才有权威终局码。

用法（容器内，cwd=/home/vscode/workspace/BenchMARL）：
  PROBE_NENV=1500 python /tmp/probe_contact_econ.py <ckpt> <out.txt>

判责数学（layup_jit.py 条件3，方案A+）：
  n = (pos_D - pos_A1)/|·|
  approach_A1 = clamp(v_A1·n, 0)     approach_D = clamp(-v_D·n, 0)
  approach_max = max(两者)           # 门槛 foul_approach_threshold=0.35 用它
  violation_D = max(0,|v_D|-0.3) + max(0,approach_D-0.3)
  score_A = approach_A1 - 1.5*violation_D ; score_A <= approach_D -> 责任归防守
"""
import os
import sys

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig

CKPT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/contact_econ.txt"
NENV = int(os.environ.get("PROBE_NENV", "1500"))

SCA = torch.tensor([4.0, 7.5])    # 绝对位置
SCR = torch.tensor([8.0, 15.0])   # 相对位置
SCV = 5.0                          # 绝对速度
SCVR = 10.0                        # 相对速度
AGENT_R = 0.3
CONTACT = 2 * AGENT_R              # 0.6 m 接触阈值

A_T = 0.35   # foul_approach_threshold 现值
LEGAL_V, LEGAL_A, K_EXEMPT = 0.3, 0.3, 1.5

OUTL = []


def p(*a):
    s = " ".join(str(x) for x in a)
    OUTL.append(s)
    print(s, flush=True)


def q(x, name, unit=""):
    x = x.flatten().float()
    if x.numel() == 0:
        p(f"    {name}: (空)")
        return
    t = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9]))
    p(f"    {name}: p10={t[0]:.3f} p50={t[1]:.3f} p90={t[2]:.3f} mean={x.mean():.3f}{unit}")


def build_exp():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_n_envs_per_worker = NENV
    cfg.on_policy_collected_frames_per_batch = NENV * 200
    cfg.max_n_iters = 1
    cfg.evaluation = False
    cfg.render = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.restore_file = None
    cfg.save_folder = "/tmp/probe_contact_econ"
    os.makedirs(cfg.save_folder, exist_ok=True)
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    task = LayupTask.LAYUP.get_from_yaml()
    acfg = MappoConfig.get_from_yaml()
    acfg.share_param_actor = True
    acfg.share_param_critic = False
    mc = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"),
            GruConfig.get_from_yaml("benchmarl/conf/model/layers/gru.yaml"),
        ],
        intermediate_sizes=[256],
    )
    cc = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    return Experiment(task=task, algorithm_config=acfg, model_config=mc, critic_model_config=cc,
                      seed=0, config=cfg, callbacks=[])


def main():
    os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
    exp = build_exp()
    try:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        for g in exp.group_map.keys():
            exp.losses[g].load_state_dict(ck[f"loss_{g}"])
        exp.collector.update_policy_weights_()
        exp.policy.eval()
        b = next(iter(exp.collector))

        obs = b.get(("agents", "observation")).float()
        rew = b.get(("next", "agents", "reward")).float()
        done = b.get(("next", "done")).float()
        code = b.get(("next", "agents", "info", "termination_reason")).float()
        act = b.get(("agents", "action", "continuous")).float()
        # 统一成 (T,B,A,·)
        if obs.shape[1] != NENV and obs.shape[0] == NENV:      # (B,T,A,F)
            swap = lambda x: x.transpose(0, 1)
            obs, rew, done, code, act = (swap(x) for x in (obs, rew, done, code, act))
        T, B = obs.shape[0], obs.shape[1]
        p(f"[批] T={T} B={B} codes={tuple(code.shape)}")
        code = code.reshape(T, B, 4, -1)[..., 0, 0]
        done = done.reshape(T, B, -1)[..., 0]
        rew = rew.reshape(T, B, 4, -1)[..., 0]

        # ---------- 1. 终局码分布 ----------
        term = done.bool()
        c = code[term].long()
        n = int(term.sum())
        p(f"\n[1] 真实终局码分布（{n} 局）")
        names = {1: "码1 命中(攻胜)", 2: "码2 防守犯规(攻胜)", 3: "码3 防守撞墙(攻胜)",
                 4: "码4 防守越线(攻胜)", 5: "码5 防守友军(攻胜)", 11: "码11 被盖(守胜)",
                 12: "码12 超时(守胜)", 13: "码13 己方犯规(守胜)", 14: "码14 己方撞墙(守胜)",
                 15: "码15 己方友军(守胜)"}
        win = 0
        for k in sorted(names):
            m = int((c == k).sum())
            if m:
                flag = "  <== 碰撞" if k in (2, 13, 5, 15) else ""
                p(f"    {names[k]:22s} {m:6d}  {100*m/n:5.1f}%{flag}")
        win = int(sum((c == k).sum() for k in (1, 2, 3, 4, 5)))
        p(f"    -> 攻方胜率 {100*win/n:.1f}%")

        # ---------- 2. 碰撞类终局的几何（A1 视角） ----------
        lt = obs[..., 0, :]            # A1 自己的观测（含对手绝对位置/速度）
        # A1 本体
        a1_pos = lt[..., 0:2] * SCA
        a1_vel = lt[..., 2:4] * SCV
        a1_in_spot = lt[..., 6] > 0.5
        a1_prog = lt[..., 7]
        rem_t = lt[..., 8]
        d_pos = [lt[..., 21:23] * SCA, lt[..., 29:31] * SCA]   # opp1/opp2 绝对位置
        d_vel = [lt[..., 23:25] * SCV, lt[..., 31:33] * SCV]
        dist = [ (d_pos[i] - a1_pos).norm(dim=-1) for i in range(2) ]

        def approaches(i):
            vec = d_pos[i] - a1_pos
            nn = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)
            ap_a1 = (a1_vel * nn).sum(-1).clamp(min=0)
            ap_d = (-(d_vel[i]) * nn).sum(-1).clamp(min=0)
            return ap_a1, ap_d

        p(f"\n[2] 碰撞类终局的接触几何（终局前一步的 A1 视角）")
        for k in (2, 13, 5, 15):
            m = (term & (code == k))
            if int(m.sum()) == 0:
                continue
            p(f"  --- {names[k]}：{int(m.sum())} 局 ---")
            for i in (0, 1):
                ap_a1, ap_d = approaches(i)
                md = dist[i][m]
                # 只统计"真接触"的那一侧（取近的一侧）
                near = (dist[0] <= dist[1]).float()
                sel = m & ((dist[0] <= dist[1]) if i == 0 else (dist[0] > dist[1]))
                if int(sel.sum()) == 0:
                    continue
                apm = torch.maximum(ap_a1, ap_d)[sel]
                p(f"    D{i+1} 为最近防守：{int(sel.sum())} 局 | 距离 p50={md.min().item():.3f}(min) "
                  f"| approach_max p50={apm.median():.3f} | <0.35 占 {100*float((apm < A_T).float().mean()):.1f}% "
                  f"| <0.5 占 {100*float((apm < 0.5).float().mean()):.1f}% "
                  f"| <1.0 占 {100*float((apm < 1.0).float().mean()):.1f}%")
                p(f"      读条中(prog>0) 占 {100*float((a1_prog[sel] > 0).float().mean()):.1f}% "
                  f"| A1 在圈内 占 {100*float(a1_in_spot[sel].float().mean()):.1f}% "
                  f"| 剩余时间 p50={rem_t[sel].median():.1f}s")
            # 责任归属（用方案A+数学复算，看当时实际判给谁的原因）
            if k == 2:
                ap_a1_0, ap_d_0 = approaches(0)
                ap_a1_1, ap_d_1 = approaches(1)
                sel = m & (dist[0] <= dist[1])
                v_abs_d = d_vel[0].norm(dim=-1)
                viol = (v_abs_d - LEGAL_V).clamp(min=0) + (ap_d_0 - LEGAL_A).clamp(min=0)
                score_a = ap_a1_0 - K_EXEMPT * viol
                p(f"      豁免触发（score_A<=approach_D，判守方）占 {100*float((score_a <= ap_d_0)[sel].float().mean()):.1f}%")

        # ---------- 3. 非终局：防守方靠近 A1 的稠密代价 & 是否敢贴防 ----------
        p(f"\n[3] 非终局步：防守贴近 A1 的稠密奖励代价（按最近防守距离分箱）")
        nt = ~term
        dmin = torch.minimum(dist[0], dist[1])
        bins = [(0.0, 0.6), (0.6, 0.9), (0.9, 1.5), (1.5, 3.0), (3.0, 99.0)]
        for lo, hi in bins:
            m = nt & (dmin >= lo) & (dmin < hi)
            if int(m.sum()) == 0:
                continue
            p(f"    最近防守 ∈ [{lo},{hi}) m: 步占比 {100*float(m.float().sum())/float(nt.float().sum()):5.1f}% "
              f"| D 单步奖励 mean={rew[..., 2][m].mean():+.3f}/{rew[..., 3][m].mean():+.3f} "
              f"| A1 单步奖励 mean={rew[..., 0][m].mean():+.3f}")

        # ---------- 4. A1 读条期间的贴防意愿（守方是否"不敢进圈"） ----------
        p(f"\n[4] A1 在投篮区且读条中（prog>0）的步：守方距离分布")
        chg = nt & a1_in_spot & (a1_prog > 0)
        if int(chg.sum()) > 0:
            p(f"    读条中步数 {int(chg.sum())}（占全部步 {100*float(chg.float().sum())/float(nt.float().sum()):.2f}%）")
            q(dmin[chg], "最近防守距离", " m")
            for th in (0.6, 0.9, 1.5):
                p(f"      最近防守 < {th} m 占 {100*float((dmin[chg] < th).float().mean()):.1f}%")
        c2 = nt & a1_in_spot & (a1_prog <= 0)
        if int(c2.sum()) > 0:
            q(dmin[c2], "圈内未读条时最近防守距离", " m")

    finally:
        try:
            exp.close()
        except Exception as e:
            p(f"[close] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
    with open(OUT, "w") as f:
        f.write("\n".join(OUTL) + "\n")
    print(f"\n写出 {OUT}")
