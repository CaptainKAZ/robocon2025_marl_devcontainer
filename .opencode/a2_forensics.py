"""A2（掩护者）行为取证：从 checkpoint 的 replay buffer 重建几何、复算 A2 稠密项、分状态看 advantage。

用法（容器内）:
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/a2_forensics.py <ckpt1> <label1> [<ckpt2> <label2> ...]
"""
import glob
import sys

import torch

# ---------------- 常量（与 layup.py/conf 当前值一致） ----------------
ABS = torch.tensor([4.0, 7.5])     # 位置归一化 (W/2, L/2)
REL = torch.tensor([8.0, 15.0])    # 相对位置归一化 (W, L)
VMAX = 5.0
RELVEL = 2 * VMAX
R_SPOT = 0.9
AGENT_R = 0.3
# A2 稠密项参数
K_LC, SIG_LC = 400.0, 1.0
K_SCREEN, SIG_SCREEN = 30.0, 0.9
K_INTERF = 40.0
K_BODY, SIG_BODY = 300.0, 0.5
K_REPUL, REP_RANGE = 200.0, 0.9
K_LINE, SIG_LINE_PERP = 90.0, 0.15
K_FRIEND, FRIEND_RANGE = 1000.0, 0.9
K_STALL = 100.0
K_DRAW, DRAW_RANGE, DRAW_STILL = 150.0, 1.8, 0.3
DRAW_ATT_SCALE = 2.0
# 封盖几何
BLOCK_SIGMA, BLOCK_GATE_K, DEF_PROX = 0.30, 25.0, 0.9


def _st(storage, path):
    x = storage
    for k in path:
        x = x[k]
    return x


def load_buf(ckpt):
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    st = ck["buffer_agents"]["_storage"]["_storage"]
    obs = _st(st, ["agents", "observation"]).float()[:, :, 1, :]   # A2 视角
    adv = _st(st, ["agents", "advantage"]).float()[:, :, 1, 0]     # (E,T)
    rew = _st(st, ["next", "agents", "reward"]).float()            # (E,T,4,1) 全员
    done = _st(st, ["next", "done"]).bool()
    ep_rew = _st(st, ["next", "agents", "episode_reward"]).float()
    act = _st(st, ["agents", "action", "continuous"]).float()[:, :, 1, :]
    return dict(obs=obs, adv=adv, rew=rew, done=done, ep_rew=ep_rew, act=act)


def geometry(d):
    o = d["obs"]
    d["posA2"] = o[..., 0:2] * ABS
    d["velA2"] = o[..., 2:4] * VMAX
    d["A1_pos"] = o[..., 13:15] * ABS
    d["A1_vel"] = o[..., 15:17] * VMAX
    d["D1_pos"] = o[..., 21:23] * ABS
    d["D2_pos"] = o[..., 29:31] * ABS
    d["spot"] = o[..., 35:37] * ABS
    d["basket"] = o[..., 39:41] * ABS
    d["in_spot"] = o[..., 6]
    d["prog"] = o[..., 7]
    d["time"] = o[..., 8]  # t_remaining/20
    return d


def a2_terms(d):
    a2, a1 = d["posA2"], d["A1_pos"]
    velA2 = d["velA2"]
    D = torch.stack([d["D1_pos"], d["D2_pos"]], dim=2)          # (E,T,2,2)
    vD = torch.stack([d["obs"][..., 23:25] * VMAX,
                      d["obs"][..., 31:33] * VMAX], dim=2)      # D 速度
    E, T = a1.shape[0], a1.shape[1]
    a2e = a2.unsqueeze(2)                                        # (E,T,1,2)

    d_a2_def = (a2e - D).norm(dim=-1)                            # (E,T,2)
    d_a1_def = (a1.unsqueeze(2) - D).norm(dim=-1)                # (E,T,2)
    d_a2_a1 = (a2 - a1).norm(dim=-1)                             # (E,T)

    # --- 封盖因子（与 jit 相同） ---
    ap = (d["basket"] - a1).unsqueeze(2)                         # (E,T,1,2)
    ad = D - a1.unsqueeze(2)
    ratio = (ad * ap).sum(-1) / (ap.pow(2).sum(-1) + 1e-6)
    between = (ratio > 0) & (ratio < 1)
    closest = a1.unsqueeze(2) + ratio.unsqueeze(-1) * ap
    perp = (D - closest).norm(dim=-1)
    gate = torch.sigmoid(BLOCK_GATE_K * (DEF_PROX - d_a1_def))
    bf = torch.exp(-perp.pow(2) / (2 * BLOCK_SIGMA ** 2)) * between.float() * gate   # (E,T,2)
    bf_sum = bf.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # --- A2 专属稠密项（缩放前） ---
    threat = torch.sigmoid(BLOCK_GATE_K * (DEF_PROX - d_a1_def))
    lane_clear = K_LC * ((1 - bf) * torch.exp(-d_a2_def.pow(2) / (2 * SIG_LC ** 2)) * threat).sum(-1)

    to_a1 = (a1.unsqueeze(2) - D)
    unit = to_a1 / (to_a1.norm(dim=-1, keepdim=True) + 1e-6)
    ideal = D + 0.9 * unit
    d_ideal = (a2e - ideal).norm(dim=-1)
    vec_ad = D - a2e
    vec_aa = a1.unsqueeze(2) - a2e
    pos_gate = torch.sigmoid(-7.0 * (vec_ad * vec_aa).sum(-1))
    spacing_gate = torch.sigmoid(7.0 * (d_a2_a1.unsqueeze(-1) - d_a2_def))
    screen = (K_SCREEN * torch.exp(-d_ideal.pow(2) / (2 * SIG_SCREEN ** 2)) * pos_gate * spacing_gate).max(-1).values
    inter = (K_INTERF * torch.exp(-d_a2_def.pow(2) / (2 * SIG_SCREEN ** 2))).max(-1).values
    body = K_BODY * (bf / bf_sum * torch.exp(-d_a2_def.pow(2) / (2 * SIG_SCREEN ** 2))).sum(-1) * \
        torch.exp(-velA2.pow(2).sum(-1) / (2 * SIG_BODY ** 2))

    away = -(vD * unit).sum(-1)                                  # 防守者背离 A1 的速度分量
    rep = (K_REPUL * away.clamp(min=0) * (d_a2_def < REP_RANGE).float()).max(-1).values

    shot = d["basket"] - a1
    a2v = a2 - a1
    r2 = (a2v * shot).sum(-1) / (shot.pow(2).sum(-1) + 1e-6)
    between2 = (r2 > 0) & (r2 < 1)
    perp2 = (a2v - r2.unsqueeze(-1) * shot).norm(dim=-1)
    line = K_LINE * between2.float() * torch.exp(-perp2.pow(2) / (2 * SIG_LINE_PERP ** 2)) * \
        torch.exp(-d_a2_a1.pow(2) / (2 * AGENT_R ** 2))

    friend = K_FRIEND * (FRIEND_RANGE - d_a2_a1).clamp(min=0)

    # 站立造犯规（通用项里 A2 的一份）：防守者朝 A2 的接近速度
    spd_a2 = velA2.norm(dim=-1)
    still = (spd_a2 < DRAW_STILL).float()
    act_norm = d["act"].norm(dim=-1)
    to_stand = (act_norm < DRAW_STILL).float()
    app_va = (-(vD * ((a2e - D) / (d_a2_def.unsqueeze(-1) + 1e-6))).sum(-1)).clamp(min=0)  # D 朝 A2 速度
    draw = K_DRAW * DRAW_ATT_SCALE * (app_va * (d_a2_def < DRAW_RANGE).float()).sum(-1) * still * to_stand

    role = lane_clear + screen + inter + body + rep - line
    stall = (a2[..., 1] < 0)
    dense_a2_specific = torch.where(stall, -K_STALL * a2[..., 1].abs(), role)

    d.update(dict(d_a2_def=d_a2_def, d_a1_def=d_a1_def, d_a2_a1=d_a2_a1, bf=bf, bf_sum=bf_sum.sum(-1),
                  lane_clear=lane_clear, screen=screen, inter=inter, body=body, rep=rep, line=line,
                  friend=friend, draw=draw, role=role, stall=stall, dense_a2_specific=dense_a2_specific,
                  pos_gate=pos_gate, spacing_gate=spacing_gate, d_ideal=d_ideal, still=still))
    return d


def f(x):
    return f"{x.mean():+.3f}"


def q(x, prob):
    return torch.quantile(x.flatten().float()[: 4_000_000], prob)


def pct(x):
    return f"{100 * x.float().mean():.1f}%"


def report(label, d):
    a2 = d["posA2"]; a1 = d["A1_pos"]
    adv2 = d["adv"]
    y = a2[..., 1]
    nearD = d["d_a2_def"].min(-1).values
    out = []
    p = out.append
    p(f"\n{'='*72}\n{label}\n{'='*72}")
    p("---- 位置 / 空间 ----")
    p(f"  A2 y: mean={y.mean():+.2f} p10={q(y,0.1):+.2f} p50={q(y,0.5):+.2f} p90={q(y,0.9):+.2f}")
    p(f"  在进攻半场 y>0: {pct(y>0)} | 己方半场 y<0: {pct(y<0)} | 深退 y<-2: {pct(y<-2)} | y<-4: {pct(y<-4)}")
    p(f"  过中线门控生效(全部稠密被替换为停顿罚): {pct(d['stall'])}")
    p(f"  A2-A1 距离: mean={d['d_a2_a1'].mean():.2f} p50={q(d['d_a2_a1'],0.5):.2f} | "
      f"<1.2m {pct(d['d_a2_a1']<1.2)} <2m {pct(d['d_a2_a1']<2)} <3m {pct(d['d_a2_a1']<3)}")
    p(f"  A2-最近防守: mean={nearD.mean():.2f} | 接触<0.66 {pct(nearD<0.66)} <0.9 {pct(nearD<0.9)} "
      f"<1.5 {pct(nearD<1.5)} <1.8 {pct(nearD<1.8)}")
    p("---- 掩护几何 ----")
    p(f"  到理想掩护位距离: mean={d['d_ideal'].min(-1).values.mean():.2f} | 位置门开启 {pct(d['pos_gate']>0.5)} "
      f"| 间距门开启 {pct(d['spacing_gate']>0.5)} | 两者同时(pos_gate>0.5&spacing>0.5) "
      f"{pct((d['pos_gate']>0.5)&(d['spacing_gate']>0.5))}")
    p(f"  走廊清空项 lane_clear: mean={f(d['lane_clear'])} p90={q(d['lane_clear'],0.9):.2f} "
      f">10 占比 {pct(d['lane_clear']>10)}")
    p(f"  掩护站位 screen: mean={f(d['screen'])} >5 占比 {pct(d['screen']>5)}")
    p(f"  干扰 inter: mean={f(d['inter'])} | 卡位 body: mean={f(d['body'])} | 排斥 rep: mean={f(d['rep'])}")
    p(f"  挡线 line: 生效占比 {pct(d['line']>1)} mean={f(d['line'])}")
    p(f"  让路 friend: mean={f(d['friend'])} (>50 占比 {pct(d['friend']>50)})")
    p(f"  站立造犯规 draw(模拟值): mean={f(d['draw'])} >50 占比 {pct(d['draw']>50)}")
    p(f"  A2 专属稠密合计(缩放前, 含停顿门控): mean={f(d['dense_a2_specific'])}")
    p("---- A1 读条期间 A2 在干什么 ----")
    ch = d["prog"] > 0
    p(f"  读条帧占比 {pct(ch)}")
    if ch.any():
        p(f"  读条期 A2-A1 距离 mean={d['d_a2_a1'][ch].mean():.2f} | A2-最近防守 mean={nearD[ch].mean():.2f} "
          f"| 贴防<1.2 {pct(nearD[ch]<1.2)}")
        p(f"  读条期 screen>5 {pct(d['screen'][ch]>5)} | 挡线中 {pct(d['line'][ch]>1)} | 位置门开 {pct(d['pos_gate'][ch]>0.5)}")
    p("---- 动作 ----")
    u2 = d['act'].norm(dim=-1)
    p(f"  |u_A2| mean={u2.mean():.3f} p90={q(u2,0.9):.3f} | 近静止<0.3 {pct(u2<0.3)}")
    p("---- A2 advantage ----")
    a = lambda name, m: p(f"  {name:34s} mean={adv2[m].mean():+7.3f} std={adv2[m].std():6.3f} n={int(m.sum())}")

    a("全体", torch.ones_like(adv2, dtype=torch.bool))
    a("y>0 (进攻半场)", y > 0)
    a("y<0 (己方半场)", y < 0)
    a("A2-A1 <2m", d["d_a2_a1"] < 2)
    a("A2-A1 2-5m", (d["d_a2_a1"] >= 2) & (d["d_a2_a1"] < 5))
    a("A2-A1 >5m", d["d_a2_a1"] >= 5)
    a("近防守<0.9m", nearD < 0.9)
    a("近防守 0.9-1.8", (nearD >= 0.9) & (nearD < 1.8))
    a("近防守 >1.8m", nearD >= 1.8)
    a("screen>5 (好掩护位)", d["screen"] > 5)
    a("lane_clear>10", d["lane_clear"] > 10)
    a("挡线中(line>1)", d["line"] > 1)
    a("读条期(prog>0)", ch)
    a("读条期&近防守<1.2", ch & (nearD < 1.2))
    a("读条期&screen>5", ch & (d["screen"] > 5))
    a("contact<0.66 & 静止", (nearD < 0.66) & (d["still"] > 0.5))
    p("  相关性 corr(adv, x):")
    xs = [("y", y), ("d(A2,A1)", d["d_a2_a1"]), ("d(A2,近防守)", nearD), ("lane_clear", d["lane_clear"]),
          ("screen", d["screen"]), ("inter", d["inter"]), ("body", d["body"]), ("rep", d["rep"]),
          ("line", d["line"]), ("friend", d["friend"]), ("draw", d["draw"]),
          ("prog", d["prog"]), ("bf_sum", d["bf_sum"])]
    for nm, x in xs:
        xx = x.flatten().float(); aa = adv2.flatten()
        c = ((xx - xx.mean()) * (aa - aa.mean())).mean() / (xx.std() * aa.std() + 1e-8)
        p(f"    {nm:16s} {c:+.4f}")
    p("---- 终局（按 done 帧 + 奖励签名分类） ----")
    done = d["done"].unsqueeze(-1)                     # (E,T,1)
    r = d["rew"]                                       # (E,T,4,1)
    ep = d["ep_rew"]
    idx = done.expand_as(r).nonzero()                  # (n,4)
    if idx.numel() > 0:
        bb, tt, aa4, _ = idx.T
        r1 = r[bb, tt, 0, 0]; r2 = r[bb, tt, 1, 0]
        rd = r[bb, tt, 2:, 0].mean(-1)
        prog_end = d["prog"][bb, tt]
        time_end = d["time"][bb, tt]
        y2end = y[bb, tt]
        cls = torch.full_like(r1, 5)                   # 5=其它
        cls = torch.where(prog_end >= 0.85, torch.where(r1 >= 30, torch.zeros_like(cls), torch.ones_like(cls)), cls)
        cls = torch.where((prog_end < 0.85) & (time_end <= 0.03), torch.full_like(cls, 2), cls)
        cls = torch.where((cls == 5) & (r2 >= 35) & (r1 <= 5), torch.full_like(cls, 3), cls)   # 造犯规(A2被撞)
        cls = torch.where((cls == 5) & (r1 <= -35), torch.full_like(cls, 4), cls)                # 攻方犯规
        names = ["码1命中", "码11被盖", "码12超时", "码2造犯规(A2被动)", "码13/15攻方犯规", "其它"]
        p(f"  终局事件 {len(r1)} 个")
        for ci, nm in enumerate(names):
            m = cls == ci
            if m.any():
                p(f"    {nm:22s} n={int(m.sum()):5d} ({100*m.float().mean():5.1f}%) | A2终局奖励 {r2[m].mean():+7.2f} "
                  f"| A1 {r1[m].mean():+7.2f} | 防守 {rd[m].mean():+7.2f} | A2末帧y {y2end[m].mean():+.2f} "
                  f"| 读条末 {prog_end[m].mean():.2f}")
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    outs = []
    for i in range(0, len(args), 2):
        ckpt, label = args[i], args[i + 1]
        d = a2_terms(geometry(load_buf(ckpt)))
        outs.append(report(label, d))
    txt = "\n".join(outs)
    print(txt)
    with open("/tmp/a2_forensics_out.txt", "w") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
