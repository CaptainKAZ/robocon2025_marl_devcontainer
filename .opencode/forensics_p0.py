# -*- coding: utf-8 -*-
"""[P0 单组架构] buffer + advantage 取证（4 角色，单组 'agents'）。

用法（容器内，cwd=/home/vscode/workspace/BenchMARL）：
  PYTHONPATH=/home/vscode/workspace/BenchMARL FPB=300000 NENV=2000 \
    python /tmp/forensics_p0.py <ckpt> <label> <out.txt>

观测布局（41 维，layup.py:1211-1295）：
  self      [0:2]pos/[4,7.5] [2:4]vel/5 [4:6]上帧动作/5 [6]in_spot [7]读条进度 [8]剩余时间
  teammate  [9:11]rel_pos/[8,15] [11:13]rel_vel/10 [13:15]abs_pos/[4,7.5] [15:17]abs_vel/5
  opp1      [17:19]rel_pos [19:21]rel_vel [21:23]abs_pos [23:25]abs_vel
  opp2      [25:27]rel_pos [27:29]rel_vel [29:31]abs_pos [31:33]abs_vel
  spot      [33:35]rel_pos [35:37]abs_pos
  basket    [37:39]rel_pos [39:41]abs_pos
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
LABEL = sys.argv[2] if len(sys.argv) > 2 else "p0"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/fx_p0.txt"
FPB = int(os.environ.get("FPB", 300000))
NENV = int(os.environ.get("NENV", 2000))

SCA = torch.tensor([4.0, 7.5])   # 绝对位置归一化因子
SCR = torch.tensor([8.0, 15.0])  # 相对位置归一化因子
ROLES = ["A1", "A2", "D1", "D2"]

OUTL = []


def p(*a):
    s = " ".join(str(x) for x in a)
    OUTL.append(s)
    print(s, flush=True)


def sh(mask, tot):
    tot = tot.float().sum()
    return f"{100 * float(mask.float().sum()) / max(float(tot), 1):.1f}%"


def st(x):
    x = x.flatten().float()
    if x.numel() == 0:
        return "(空)"
    q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))
    return f"mean={x.mean():+8.3f} std={x.std():7.3f} p50={q[1]:+8.3f} p95={q[2]:+8.3f} pos={(x > 0).float().mean():.2f}"


def build_exp():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = FPB
    cfg.on_policy_n_envs_per_worker = NENV
    cfg.on_policy_minibatch_size = 15000
    cfg.max_n_iters = 1
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/fx_p0"
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
            GruConfig.get_from_yaml(),
        ],
        intermediate_sizes=[256],
    )
    cc = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    return Experiment(task=task, algorithm_config=acfg, model_config=mc, critic_model_config=cc,
                      seed=0, config=cfg, callbacks=[])


def main():
    exp = build_exp()
    try:
        buf = exp.replay_buffers["agents"]
        ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
        p(f"[ckpt] {os.path.basename(CKPT)}  n_iters={ck.get('state', {}).get('n_iters_performed')}")
        p(f"[ckpt] buffer groups={[k for k in ck.keys() if k.startswith('buffer_')]}")
        buf.load_state_dict(ck["buffer_agents"])
        td = buf._storage._storage
        shape = tuple(td.shape)
        n = 1
        for s in shape:
            n *= s

        def flat(key):
            v = td.get(key)
            return v.reshape(n, *v.shape[len(shape):])

        term = flat(("next", "done")).reshape(-1).bool()
        rew = flat(("next", "agents", "reward")).reshape(-1, 4).float()
        er = flat(("next", "agents", "episode_reward")).reshape(-1, 4).float()
        adv = flat(("agents", "advantage")).reshape(-1, 4).float()
        obs = flat(("agents", "observation")).float()      # (N,4,41)
        lt = obs[:, :, -41:]
        N = int(term.numel())
        ones = torch.ones(N, dtype=torch.bool)
        nterm = float(term.sum())

        def geom(ai):
            f = lt[:, ai]
            pos = f[:, 0:2] * SCA
            vel = f[:, 2:4] * 5.0
            spd = vel.norm(dim=-1)
            o_rel = torch.stack([f[:, 17:19] * SCR, f[:, 25:27] * SCR], 1)
            o_vel = torch.stack([f[:, 23:25] * 5.0, f[:, 31:33] * 5.0], 1)
            d = o_rel.norm(dim=-1)
            dmin, near = d.min(dim=1)
            unit = o_rel / (d.unsqueeze(-1) + 1e-6)
            sel = near.view(-1, 1, 1).expand(-1, 1, 2)
            u = unit.gather(1, sel).squeeze(1)
            dv = o_vel.gather(1, sel).squeeze(1)
            o_abs = torch.stack([f[:, 19:21] * SCA, f[:, 27:29] * SCA], 1)
            return dict(pos=pos, vel=vel, spd=spd, dmin=dmin, u=u,
                        v_to_opp=(vel * u).sum(-1), def_appr=(dv * (-u)).sum(-1),
                        d_tm=(f[:, 9:11] * SCR).norm(dim=-1),
                        tm_abs=f[:, 13:15] * SCA, o_abs=o_abs,
                        in_spot=f[:, 6] > 0.5, prog=f[:, 7], time=f[:, 8],
                        d_spot=(f[:, 33:35] * SCR).norm(dim=-1),
                        basket_rel=f[:, 37:39] * SCR)

        g = [geom(i) for i in range(4)]

        p("\n" + "=" * 100)
        p(f"{LABEL}   N={N}  terminals={int(nterm)} ({100 * nterm / N:.2f}% 帧)")
        p("=" * 100)

        # ---------- 全局行为（4 角色） ----------
        p("\n[全局行为] 角色: y均值 | 对方半场% | 速度 | 到最近对手 | 朝对手速度 | 步奖励(非终局)")
        for i, nm in enumerate(ROLES):
            gg = g[i]
            half = sh(gg["pos"][:, 1] > 0, ones) if i < 2 else sh(gg["pos"][:, 1] < 0, ones)
            p(f"  {nm}: y={gg['pos'][:, 1].mean():+6.2f}±{gg['pos'][:, 1].std():5.2f} | 对方半场={half:>6s} | "
              f"v={gg['spd'].mean():.2f} | d_opp={gg['dmin'].mean():.2f} | v_to_opp={gg['v_to_opp'].mean():+.2f} | "
              f"rw={rew[~term, i].mean():+.4f}")

        # ---------- 关键行为 ----------
        a1, a2 = g[0], g[1]
        p("\n[关键行为]")
        p(f"  A1: 圈内={sh(a1['in_spot'], ones)} | 读条>0={sh(a1['prog'] > 0, ones)} | "
          f"圈内&速度<0.5={sh(a1['in_spot'] & (a1['spd'] < 0.5), ones)} | 接触(<0.6m)={sh(a1['dmin'] < 0.6, ones)} | "
          f"到投篮点={a1['d_spot'].mean():.2f}")
        p(f"  A2: 过中线={sh(a2['pos'][:, 1] > 0, ones)} | 到A1={a2['d_tm'].mean():.2f} | "
          f"到防守={a2['dmin'].mean():.2f} | <1.5m={sh(a2['dmin'] < 1.5, ones)}")
        # A2 掩护几何: 是否在防守与 A1 之间（用 A2 自己的 obs 重建）
        tm = a2["tm_abs"] - a2["pos"]                 # A2->A1
        o_abs = a2["o_abs"]                            # (N,2,2)
        bt = tm.unsqueeze(1) - o_abs                   # A1 - 对手
        denom = (bt ** 2).sum(-1)
        proj = ((a2["pos"].unsqueeze(1) - o_abs) * bt).sum(-1) / (denom + 1e-6)
        perp = ((a2["pos"].unsqueeze(1) - o_abs) - proj.unsqueeze(-1) * bt).norm(dim=-1)
        between = (proj > 0) & (proj < 1) & (perp < 1.2)
        p(f"  A2 掩护: 在任一防守-A1连线间(1.2m)={sh(between.any(dim=1), ones)} | "
          f"两防守都满足={sh(between.all(dim=1), ones)}")
        d1, d2 = g[2], g[3]
        p(f"  D1: 越线(y<0)={sh(d1['pos'][:, 1] < 0, ones)} | 贴A1<1.2m={sh(d1['dmin'] < 1.2, ones)} | "
          f"D2: 越线={sh(d2['pos'][:, 1] < 0, ones)} | 贴A1<1.2m={sh(d2['dmin'] < 1.2, ones)}")
        # 封盖代理: A1 到篮筐连线，防守垂距
        a1_pos = a1["pos"]
        basket_abs = a1_pos + a1["basket_rel"]
        shot = basket_abs - a1_pos
        for i, nm in [(2, "D1"), (3, "D2")]:
            dv = g[i]["pos"] - a1_pos
            t = (dv * shot).sum(-1) / (shot.pow(2).sum(-1) + 1e-6)
            perp = (dv - t.unsqueeze(-1) * shot).norm(dim=-1)
            bp = (t > 0) & (t < 1)
            line = torch.exp(-perp.pow(2) / (2 * 0.3 ** 2)) * bp.float()
            p(f"  封盖代理 {nm}: 在线上(bp>0.2)={sh(line > 0.2, ones)} | 读到线上={sh(line > 0.2, ones)}")

        # ---------- 终局构成（4 角色 episode_reward 符号规则） ----------
        tsec = lt[:, 0, 8] * 15.0
        a_neg = (er[:, 0] <= -10) | (er[:, 1] <= -10)
        d_neg = (er[:, 2] <= -10) | (er[:, 3] <= -10)
        timeout = term & (tsec <= 0.2)
        a_loss = term & ~timeout & a_neg & ~d_neg
        d_loss = term & ~timeout & d_neg & ~a_neg
        both_loss = term & ~timeout & a_neg & d_neg
        no_rew = term & ~timeout & ~a_neg & ~d_neg
        # 命中代理: 攻方两人 ep_reward 都很高 且 有时间富余
        shot_made = term & ~timeout & (er[:, 0] >= 40) & (er[:, 1] >= 40)
        p("\n[终局构成] 4 角色 episode_reward 符号分类（近似口径）:")
        for nm, m in [("码12 超时", timeout), ("码13 攻方犯规", a_loss), ("码3/4/5 守方失误", d_loss),
                      ("码5 友军(守方)", both_loss), ("码4 无奖励", no_rew), ("命中代理(码1)", shot_made)]:
            if int(m.sum()) == 0:
                p(f"  {nm:16s} 0")
                continue
            p(f"  {nm:16s} {int(m.sum()):6d} {100 * float(m.sum()) / max(nterm, 1):6.1f}%  | "
              f"A1 {er[m, 0].mean():+7.2f} A2 {er[m, 1].mean():+7.2f} "
              f"D1 {er[m, 2].mean():+7.2f} D2 {er[m, 3].mean():+7.2f}")

        # ---------- advantage 分组 ----------
        p("\n[advantage 分组]")
        for i, nm in enumerate(ROLES):
            a = adv[:, i]
            p(f"  ---- {nm}: {st(a)} ----")
            gg = g[i]
            p(f"    对方半场(内): {st(a[gg['pos'][:, 1] > 0] if i < 2 else a[gg['pos'][:, 1] < 0])}")
            p(f"    最近对手<0.6m: {st(a[gg['dmin'] < 0.6])}")
            p(f"    最近对手[0.6,1.5): {st(a[(gg['dmin'] >= 0.6) & (gg['dmin'] < 1.5)])}")
            p(f"    最近对手>=2m: {st(a[gg['dmin'] >= 2])}")
            p(f"    靠近对手(v>0): {st(a[gg['v_to_opp'] > 0])} | 远离(v<0): {st(a[gg['v_to_opp'] < 0])}")
        p(f"  ---- A1 专项 ----")
        p(f"    圈内: {st(adv[:, 0][a1['in_spot']])}")
        p(f"    读条 [0,0.3): {st(adv[:, 0][(a1['prog'] > 0) & (a1['prog'] < 0.3)])}")
        p(f"    读条 [0.3,0.6): {st(adv[:, 0][(a1['prog'] >= 0.3) & (a1['prog'] < 0.6)])}")
        p(f"    读条 [0.6,0.9): {st(adv[:, 0][(a1['prog'] >= 0.6) & (a1['prog'] < 0.9)])}")
        p(f"    读条 >=0.9: {st(adv[:, 0][a1['prog'] >= 0.9])}")
        p(f"    到投篮点<12m 分箱: <2m {st(adv[:, 0][a1['d_spot'] < 2])} | [2,5) {st(adv[:, 0][(a1['d_spot'] >= 2) & (a1['d_spot'] < 5)])} | >=5m {st(adv[:, 0][a1['d_spot'] >= 5])}")
        p(f"  ---- A2 专项 ----")
        p(f"    过中线: {st(adv[:, 1][a2['pos'][:, 1] > 0])} | 未过: {st(adv[:, 1][a2['pos'][:, 1] < 0])}")
        p(f"    掩护连线间: {st(adv[:, 1][between.any(dim=1)])}")
        p(f"    到A1<3m: {st(adv[:, 1][a2['d_tm'] < 3])} | >=5m: {st(adv[:, 1][a2['d_tm'] >= 5])}")
        p(f"  ---- D 专项 ----")
        p(f"    D1 越线: {st(adv[:, 2][d1['pos'][:, 1] < 0])} | D2 越线: {st(adv[:, 3][d2['pos'][:, 1] < 0])}")
        p(f"    D1 贴A1<1.2m: {st(adv[:, 2][d1['dmin'] < 1.2])}")
        p(f"    D2 贴A1<1.2m: {st(adv[:, 3][d2['dmin'] < 1.2])}")
        p(f"    剩余时间<3s: D1 {st(adv[:, 2][tsec < 3])} | D2 {st(adv[:, 3][tsec < 3])}")

        # 相关性
        p("\n[相关性 corr(adv, x)]")
        for i, nm in enumerate(ROLES):
            gg = g[i]
            for xn, x in [("dmin", gg["dmin"]), ("v_to_opp", gg["v_to_opp"]), ("y", gg["pos"][:, 1]), ("spd", gg["spd"])]:
                xx = x.float()
                aa = adv[:, i]
                c = ((xx - xx.mean()) * (aa - aa.mean())).mean() / (xx.std() * aa.std() + 1e-8)
                p(f"  {nm} corr({xn}, adv) = {c:+.4f}")

        with open(OUT, "w") as fh:
            fh.write("\n".join(OUTL) + "\n")
        p(f"\n[forensics_p0] written: {OUT}")
    finally:
        exp.close()


if __name__ == "__main__":
    main()
