# -*- coding: utf-8 -*-
"""按键时代单组 buffer 取证：A1 进圈 / 按键 / 读条 / 出手 + advantage 分组。

用法（容器内）:
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/forensics_button.py <ckpt> <label> [out.txt]
要点：
  - 单组 buffer 存储是 OrderedDict，必须方括号访问；
  - 终局只在 ("next","done")，根级 done 全 0；
  - T=200 = 一整局（t_limit=20s），每行恰好一局；
  - 出手判定：终局帧 progress >= 0.9；命中/被盖用 A1 终局奖励区分（命中 ~+145，被盖 ~+2.6）。
"""
import sys

import torch

CKPT = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
OUT = sys.argv[3] if len(sys.argv) > 3 else None

lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def tree(d, prefix="", depth=0):
    if depth > 3 or not hasattr(d, "items"):
        return
    for k, v in d.items():
        if hasattr(v, "items"):
            P(f"  {prefix}{k}/")
            tree(v, prefix + "  ", depth + 1)
        else:
            try:
                P(f"  {prefix}{k} {tuple(v.shape)} {v.dtype}")
            except Exception:
                P(f"  {prefix}{k} {type(v)}")


def get(d, *keys):
    cur = d
    for k in keys:
        if hasattr(cur, "items") and k in cur.keys():
            cur = cur[k]
        else:
            return None
    return cur


def sq4(x):
    """把 (..., k, 1) 压成 (..., k)。"""
    if x is None:
        return None
    if x.dim() >= 2 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def max_run(x):
    """每行最大连续 True 长度，x: (E,T) bool。"""
    run = torch.zeros(x.shape[0], dtype=torch.long)
    best = torch.zeros(x.shape[0], dtype=torch.long)
    for t in range(x.shape[1]):
        run = torch.where(x[:, t], run + 1, torch.zeros_like(run))
        best = torch.maximum(best, run)
    return best


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    st = ck["buffer_agents"]["_storage"]["_storage"]

    P(f"##### {LABEL} #####")
    P(f"ckpt: {CKPT}")
    P("[keytree]")
    tree(st)

    obs = get(st, "agents", "observation")
    done = get(st, "next", "done")
    rew = get(st, "next", "agents", "reward")
    adv = get(st, "agents", "advantage")
    if obs is None or done is None:
        P("!! 缺少 observation/done，无法分析")
        return

    obs = obs.float()
    done = sq4(done).bool()
    rew = sq4(rew).float() if rew is not None else None
    adv = sq4(adv).float() if adv is not None else None

    E, T = obs.shape[0], obs.shape[1]
    P(f"[shape] obs={tuple(obs.shape)} done={tuple(done.shape)} rew={None if rew is None else tuple(rew.shape)} adv={None if adv is None else tuple(adv.shape)}")

    a1 = obs[:, :, 0, :]
    in_spot = a1[..., 6] > 0.5
    prog = a1[..., 7]
    t_rem = a1[..., 8]
    pos_y = a1[..., 1] * 7.5
    vel = a1[..., 2:4] * 5.0
    spd = vel.norm(dim=-1)

    mask = get(st, "agents", "action_mask")
    if mask is None:
        mask = get(st, "next", "agents", "action_mask")
    a1_mask = None
    if mask is not None:
        mask_b = mask.bool()
        a1_mask = mask_b[:, :, 0, 1] if mask_b.dim() == 4 else mask_b[:, :, 0]
    disc = get(st, "agents", "action", "discrete")
    press = None
    if disc is not None:
        disc = sq4(disc).float()
        press = disc[:, :, 0] > 0.5

    n = E * T
    P("[帧级]")
    P(f"  A1 在圈帧占比        {100.0*in_spot.float().mean():.2f}%")
    P(f"  A1 过中线帧占比      {100.0*(pos_y > 0).float().mean():.2f}%")
    P(f"  A1 平均速度          {spd.mean():.2f} m/s")
    P(f"  A1 读条>0 帧占比     {100.0*(prog > 0.01).float().mean():.2f}%")
    P(f"  A1 读条 0.25/0.5/0.75 以上帧占比  "
      f"{100.0*(prog >= 0.25).float().mean():.3f}% / {100.0*(prog >= 0.5).float().mean():.3f}% / {100.0*(prog >= 0.75).float().mean():.3f}%")
    if a1_mask is not None:
        P(f"  mask 允许按帧占比    {100.0*a1_mask.float().mean():.3f}%  (n={int(a1_mask.sum())})")
    if press is not None:
        P(f"  按下帧占比           {100.0*press.float().mean():.3f}%  (n={int(press.sum())})")
        if a1_mask is not None:
            allowed = int(a1_mask.sum())
            hit = int((press & a1_mask).sum())
            P(f"  允许且按下           {hit} / {allowed} = {100.0*hit/max(allowed,1):.1f}%")
        never = (~press).any(dim=1)
        P(f"  整局从未按键的局数    {int(never.sum())} / {E} = {100.0*never.float().mean():.1f}%")
        mr = max_run(press)
        P(f"  最长连续按键长度分布  0:{int((mr==0).sum())} 1-4:{int(((mr>=1)&(mr<=4)).sum())} "
          f"5-9:{int(((mr>=5)&(mr<=9)).sum())} >=10:{int((mr>=10).sum())}  (max={int(mr.max())})")

    P("[局级]")
    P(f"  总局数               {E}")
    P(f"  进过圈的局数         {int(in_spot.any(dim=1).sum())} = {100.0*in_spot.any(dim=1).float().mean():.1f}%")
    max_prog = prog.max(dim=1).values
    # 注意 fp16：9/10=0.9 存成 0.89990 < 0.9，所以出手判定阈值用 0.85
    P(f"  局最大读条>=0.85 局数 {int((max_prog >= 0.85).sum())}")
    P(f"  局最大读条>=0.5 局数  {int((max_prog >= 0.5).sum())}")
    P(f"  局最大读条>0 局数     {int((max_prog > 0.01).sum())}")

    ti = done.nonzero()
    n_term = ti.shape[0]
    P(f"  终局帧数             {n_term}")
    if n_term > 0:
        te = ti[:, 0]
        tt = ti[:, 1]
        t_prog = prog[te, tt]
        t_rew1 = rew[te, tt, 0] if rew is not None else None
        t_in = in_spot[te, tt]
        attempt = t_prog >= 0.85   # fp16 下 9/10 -> 0.8999
        P(f"  出手局(终局读条>=0.85) {int(attempt.sum())} = {100.0*attempt.float().mean():.2f}%")
        if t_rew1 is not None and attempt.any():
            made = attempt & (t_rew1 > 60)
            blocked = attempt & (t_rew1 <= 60)
            P(f"    其中命中(奖励>60)   {int(made.sum())}   被盖(<=60) {int(blocked.sum())}")
            P(f"    出手时 A1 奖励      mean={t_rew1[attempt].mean():.2f} p10={t_rew1[attempt].quantile(0.1):.2f} p90={t_rew1[attempt].quantile(0.9):.2f}")
        if t_rew1 is not None:
            for lbl, m in [("命中", attempt & (t_rew1 > 60)), ("被盖", attempt & (t_rew1 <= 60)),
                           ("超时", (~attempt) & (t_rew1 < -20)), ("其它", (~attempt) & (t_rew1 >= -20))]:
                if m.any():
                    P(f"    {lbl}: n={int(m.sum())}  终局读条 mean={t_prog[m].mean():.2f}  终局在圈占比={100.0*t_in[m].float().mean():.1f}%")

    if adv is not None and rew is not None:
        P("[advantage / reward 分角色]")
        for r, nm in enumerate(["A1", "A2", "D1", "D2"]):
            P(f"  {nm}: adv mean={adv[:,:,r].mean():.3f} std={adv[:,:,r].std():.3f} | "
              f"步奖励 mean={rew[:,:,r].mean():.4f} | 终局奖励 mean={rew[done.bool()][:,r].mean():.2f}" if n_term > 0 else
              f"  {nm}: adv mean={adv[:,:,r].mean():.3f}")
        P("[A1 advantage 分组]")
        a1a = adv[:, :, 0]
        for lbl, m in [("在圈内", in_spot), ("读条>0", prog > 0.01), ("按键", press if press is not None else torch.zeros_like(in_spot)),
                       ("圈内且按键", (in_spot & press) if press is not None else in_spot), ("圈外", ~in_spot)]:
            if m.any():
                P(f"  {lbl:12s} n={int(m.sum()):6d}  adv mean={a1a[m].mean():.3f} std={a1a[m].std():.3f} pos={100.0*(a1a[m]>0).float().mean():.0f}%")

    if OUT:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    P("FORENSICS_DONE")


if __name__ == "__main__":
    main()
