# -*- coding: utf-8 -*-
"""在 iter300 的真实 buffer 上重算 [2026-10-04 A2 改造] 各项的实发量级（不改任何东西）。
用法: python a2_newterms_probe.py <ckpt.pt>
"""
import sys
import torch

ABS = torch.tensor([4.0, 7.5])
REL = torch.tensor([8.0, 15.0])
V = 5.0
BASKET = torch.tensor([0.0, 6.9])
SIG2_SCREEN = 2 * 0.9**2          # screen_pos_sigma=0.9
SIG2_BETWEEN = 2 * 0.45**2        # a2_between_gate_sigma=0.45
SIG2_BODY = 2 * 0.5**2            # a2_body_check_still_sigma=0.5
SIG2_BLOCK = 2 * 0.30**2          # block_sigma=0.30
SIG2_SUP_LANE = 2 * 1.0**2
SIG2_SUP_CLEAR = 2 * 0.35**2
DRAW_RANGE = 1.8
DRAW_CAP = 3.0
DRAW_GAIN = 300.0
SPREAD_POS, SPREAD_NEG, SPREAD_CAP = 200.0, 60.0, 1.0
SUPPORT_GAIN = 150.0


def stat(name, x):
    x = x.flatten().float()
    if x.numel() == 0:
        print(f"  {name:44s} (空)")
        return
    q = torch.quantile(x, torch.tensor([0.5, 0.9]))
    print(f"  {name:44s} mean={x.mean():>9.3f} p50={q[0]:>8.3f} p90={q[1]:>8.3f} >0={100*(x>0).float().mean():5.1f}%")


def main():
    ckpt = sys.argv[1]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = ck["buffer_agents"]["_storage"]["_storage"]
    obs = st["agents"]["observation"][:, :, 1, :].float()
    o = obs[..., -41:]

    a2_pos = o[..., 0:2] * ABS
    a2_vel = o[..., 2:4] * V
    a1_pos = o[..., 13:15] * ABS
    d1_pos, d1_vel = o[..., 21:23] * ABS, o[..., 23:25] * V
    d2_pos, d2_vel = o[..., 29:31] * ABS, o[..., 31:33] * V
    spot = o[..., 35:37] * ABS

    def stack(a, b):
        return torch.stack([a, b], dim=-2)

    Dpos = stack(d1_pos, d2_pos)          # (...,2,2)
    Dvel = stack(d1_vel, d2_vel)

    def norm(x):
        return torch.linalg.norm(x, dim=-1)

    # --- 几何 ---
    def_to_a1 = a1_pos.unsqueeze(-2) - Dpos                       # A1 - D
    dist_da1 = norm(def_to_a1)                                    # (...,2)
    unit_def_a1 = def_to_a1 / (dist_da1.unsqueeze(-1) + 1e-6)
    rel_a2_def = a2_pos.unsqueeze(-2) - Dpos
    proj = (rel_a2_def * unit_def_a1).sum(-1)
    perp = norm(rel_a2_def - proj.unsqueeze(-1) * unit_def_a1)
    between = (
        torch.exp(-perp.pow(2) / SIG2_BETWEEN)
        * torch.sigmoid(7.0 * proj)
        * torch.sigmoid(7.0 * (dist_da1 - proj))
    )
    dist_a2_def = norm(a2_pos.unsqueeze(-2) - Dpos)
    gauss = torch.exp(-dist_a2_def.pow(2) / SIG2_SCREEN)

    # --- block factor（复刻） ---
    ap = (BASKET - a1_pos)                                        # (...,2)
    ad = Dpos - a1_pos.unsqueeze(-2)
    ratio = (ad * ap.unsqueeze(-2)).sum(-1) / (ap.pow(2).sum(-1, keepdim=True) + 1e-6)
    is_between = (ratio > 0) & (ratio < 1)
    closest = a1_pos.unsqueeze(-2) + ratio.unsqueeze(-1) * ap.unsqueeze(-2)
    dist_perp = norm(Dpos - closest)
    prox_gate = torch.sigmoid(25.0 * (0.9 - dist_da1))
    bf = torch.exp(-dist_perp.pow(2) / SIG2_BLOCK) * is_between.float() * prox_gate
    total_bf = torch.clamp(bf.sum(-1), 0, 1)

    # --- inter（40，门控前后） ---
    inter_old, _ = torch.max(40.0 * gauss, dim=-1)
    inter_new, _ = torch.max(40.0 * gauss * between, dim=-1)

    # --- body（300，门控前后） ---
    block_share = bf / (bf.sum(-1, keepdim=True) + 1e-6)
    still = torch.exp(-a2_vel.pow(2).sum(-1) / SIG2_BODY)
    body_old = 300.0 * (block_share * gauss).sum(-1) * still
    body_new = 300.0 * (block_share * gauss * between).sum(-1) * still

    # --- screen（30 -> 200） ---
    ideal = Dpos + 0.9 * unit_def_a1
    d_ideal_sq = (a2_pos.unsqueeze(-2) - ideal).pow(2).sum(-1)
    vec_a2_def = Dpos - a2_pos.unsqueeze(-2)
    vec_a2_a1 = a1_pos.unsqueeze(-2) - a2_pos.unsqueeze(-2)
    pos_gate = torch.sigmoid(-7.0 * (vec_a2_def * vec_a2_a1).sum(-1))
    spacing = torch.sigmoid(7.0 * (norm(vec_a2_a1) - norm(vec_a2_def)))
    scr_pot_old = 30.0 * torch.exp(-d_ideal_sq / SIG2_SCREEN) * pos_gate * spacing
    scr_old, _ = torch.max(scr_pot_old, dim=-1)
    scr_pot_new = 200.0 * torch.exp(-d_ideal_sq / SIG2_SCREEN) * pos_gate * spacing
    scr_new, _ = torch.max(scr_pot_new, dim=-1)

    # --- spread（防守者远离 A1，A1 过线后） ---
    sep = (Dvel * (-unit_def_a1)).sum(-1)                         # (...,2) >0 = 远离 A1
    pidx = torch.argmin(dist_da1, dim=-1)
    primary = sep.gather(-1, pidx.unsqueeze(-1)).squeeze(-1)
    crossed = (a1_pos[..., 1] > 0).float()
    spread = crossed * (
        SPREAD_POS * torch.clamp(primary, min=0.0)
        + SPREAD_NEG * torch.clamp(primary, min=-SPREAD_CAP, max=0.0)
    )

    # --- support（走廊 + 通道清空） ---
    corr = spot - a1_pos
    corr_len2 = corr.pow(2).sum(-1, keepdim=True) + 1e-6
    rel_c = a2_pos - a1_pos
    proj_c = torch.clamp((rel_c * corr).sum(-1, keepdim=True) / corr_len2, 0.0, 1.0)
    perp_c = norm(rel_c - proj_c * corr)
    near_c = torch.exp(-perp_c.pow(2) / SIG2_SUP_LANE)
    clear = torch.exp(-total_bf.pow(2) / SIG2_SUP_CLEAR)
    support = SUPPORT_GAIN * near_c * clear * crossed

    # --- draw（防守者朝 A2 冲来，即时） ---
    toward = -vec_a2_def / (dist_a2_def.unsqueeze(-1) + 1e-6)
    appr = torch.clamp((Dvel * toward).sum(-1), min=0.0, max=DRAW_CAP)
    falloff = torch.clamp(1.0 - dist_a2_def / DRAW_RANGE, min=0.0)
    draw, _ = torch.max(DRAW_GAIN * appr * falloff, dim=-1)

    print(f"\n===== A2 新项在 iter300 buffer 上的实发量级（{obs.shape[0]}×{obs.shape[1]} 帧）=====")
    print(f"  A1 过线帧占比: {100*(a1_pos[...,1]>0).float().mean():.1f}% | A2 过线: {100*(a2_pos[...,1]>=0).float().mean():.1f}%")
    stat("between 门 (0~1)", between)
    stat("inter 旧 (40*高斯)", inter_old)
    stat("inter 新 (×between 门)", inter_new)
    stat("body 旧 (300)", body_old)
    stat("body 新 (×between 门)", body_new)
    stat("screen 旧 (30)", scr_old)
    stat("screen 新 (200)", scr_new)
    stat("spread 拉开空间 (±200/60)", spread)
    stat("  └ 正部 200*max(sep,0)", SPREAD_POS * torch.clamp(primary, min=0.0))
    stat("  └ 负部 60*clamp(sep,-1,0)", SPREAD_NEG * torch.clamp(primary, min=-SPREAD_CAP, max=0.0))
    stat("support 走廊支援 (150)", support)
    stat("  └ near_corr 走廊垂距门", near_c)
    stat("  └ clear 通道清空门", clear)
    stat("draw 即时造撞 (300)", draw)
    stat("  └ appr 接近速度 (m/s)", appr.max(-1).values)
    stat("  └ falloff 距离衰减", falloff.max(-1).values)
    stat("total_bf 总封盖因子", total_bf)

    oh = torch.clamp(primary, min=0.0)
    primary_neg = torch.clamp(primary, min=-SPREAD_CAP, max=0.0)
    print("\n  近防守者<0.9m 占比: %.1f%% | 主盯防者远离(sep>0) 占比: %.1f%% | sep>0.5 占比: %.1f%%"
          % (100*(dist_a2_def.min(-1).values < 0.9).float().mean(),
             100*(primary > 0).float().mean(),
             100*(primary > 0.5).float().mean()))
    print("  防守者朝 A2 接近(appr>0.1 且 1.8m 内) 占比: %.1f%%"
          % (100*((appr > 0).any(-1) & (dist_a2_def.min(-1).values < DRAW_RANGE)).float().mean()))

    # --- 汇总（原始分/步 & 缩放后） ---
    d_inter = inter_new.mean() - inter_old.mean()
    d_body = body_new.mean() - body_old.mean()
    d_screen = scr_new.mean() - scr_old.mean()
    s_spread, s_support, s_draw = spread.mean(), support.mean(), draw.mean()
    raw_delta = (d_inter + d_body + d_screen + s_spread + s_support + s_draw).item()
    print("\n  ---- 每步净变化（原始分 -> 最终分, ×0.0005）----")
    for n, v in [("inter", d_inter), ("body", d_body), ("screen", d_screen),
                 ("spread", s_spread), ("support", s_support), ("draw", s_draw)]:
        print(f"    {n:8s} Δraw/步 = {v.item():+8.3f}  ->  {v.item()*0.0005:+.5f}/步")
    print(f"    合计     Δraw/步 = {raw_delta:+8.3f}  ->  {raw_delta*0.0005:+.5f}/步(最终)")
    ep_len = 84.0  # 之前实测完整回合长度中位数
    print(f"    （按每局 ~{ep_len:.0f} 步粗估：{raw_delta*0.0005*ep_len:+.2f} 分/局）")
    print(f"\n  被盖事件（终局）：k_blocked_shot_penalty 12000 -> 18000，A1/A2 各多扣 {(18000-12000)*0.005:.0f} 分/次")


if __name__ == "__main__":
    main()
