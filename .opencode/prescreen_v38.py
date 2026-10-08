"""离线预筛 v2（v38 checkpoint buffer 分析）——按钮使用 / 新旧映射速度 / 奖励稀释 / 被盖 EV。

修正 v1 的切段 bug：buffer 的每一行(env)通常从"局中"开始（init[0] 多为 False），
真实回合起点 = init 位置；只有 [init_k .. init_{k+1}-1] 且以 done 结束的段才是完整回合。
终局事件统计直接用全部 done 位置（不依赖起点）。

用法: python prescreen_v38.py <ckpt> <label> [out]
"""
import sys

import torch


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=False)
    st = ck["buffer_agents"]["_storage"]["_storage"]
    return {
        "obs": st["agents"]["observation"].float(),             # (E,T,4,41) fp16->fp32
        "act_c": st["agents"]["action"]["continuous"].float(),   # (E,T,4,2) 命令速度 u (m/s)
        "act_d": st["agents"]["action"]["discrete"],             # (E,T,4) 0/1
        "mask": st["agents"]["action_mask"],                     # (E,T,4,2)
        "rew": st["next"]["agents"]["reward"].float()[..., 0],   # (E,T,4)
        "done": st["next"]["done"][..., 0],                      # (E,T)
        "init": st["is_init"][..., 0],                           # (E,T)
        "adv": st["agents"]["advantage"].float()[..., 0],        # (E,T,4)
    }


def stats(name, x):
    x = x.flatten()
    if x.numel() == 0:
        print(f"    {name:40s} (空)")
        return
    q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9]))
    print(f"    {name:40s} mean={x.mean():>8.3f} p50={q[1]:>8.3f} p90={q[2]:>8.3f}")


def main():
    path, label = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    d = load(path)
    obs, act_c, act_d, mask = d["obs"], d["act_c"], d["act_d"], d["mask"]
    rew, done, init, adv = d["rew"], d["done"], d["init"], d["adv"]
    E, T = done.shape
    pct = lambda x: f"{100.0 * float(x):.1f}%"

    pos = obs[..., 0:2] * torch.tensor([4.0, 7.5])
    vel = obs[..., 2:4] * 5.0
    spd = vel.norm(dim=-1)
    in_spot = obs[..., 6] > 0.5
    prog = obs[..., 7]
    t_rem = obs[..., 8]          # 剩余时间（t_limit=20s 归一化）
    prog1 = prog[..., 0]
    press = act_d[..., 0] == 1   # (E,T) A1
    allowed = mask[..., 0, 1]    # (E,T) A1 允许按下

    P(f"\n{'='*78}\n[{label}]  {path}\n{'='*78}")
    P(f"局部行={E} 步={T} 步间隔=0.1s")

    # ---------- 切段 ----------
    segs = []        # (e, s, en, complete)  真实起点段（含被行尾截断的"头"）
    term_events = []  # (e, t)
    for e in range(E):
        ii = torch.nonzero(init[e]).flatten().tolist()
        for t in torch.nonzero(done[e]).flatten().tolist():
            term_events.append((e, t))
        for k, s in enumerate(ii):
            en = ii[k + 1] - 1 if k + 1 < len(ii) else T - 1
            complete = (k + 1 < len(ii)) and bool(done[e, en])
            segs.append((e, s, en, complete))
    n_seg = len(segs)
    n_comp = sum(1 for *_, c in segs if c)
    n_term = len(term_events)

    # ---------- A. 终局事件分类（全部 done 步） ----------
    cls_count = {"made": 0, "blocked": 0, "timeout": 0, "opp_foul": 0, "own_mistake": 0, "other": 0}
    cls_r1 = {k: [] for k in cls_count}
    cls_rd = {k: [] for k in cls_count}
    cls_trem = {k: [] for k in cls_count}
    for e, t in term_events:
        pe = prog1[e, t].item()
        r1 = rew[e, t, 0].item()
        rd = rew[e, t, 2:4].mean().item()
        # [v3 修正] 终局步读条进度=0.9（9/10 帧）而非 1.0；rd>=40 已用 buffer 对齐验证 == 码12 超时
        if pe >= 0.85:
            k = "made" if r1 >= 30 else "blocked"
        elif rd >= 40:
            k = "timeout"
        elif r1 >= 25:
            k = "opp_foul"
        elif r1 <= -35:
            k = "own_mistake"
        else:
            k = "other"
        cls_count[k] += 1
        cls_r1[k].append(r1)
        cls_rd[k].append(rd)
        cls_trem[k].append(obs[e, t, 0, 8].item() * 20.0)

    P(f"\n[A] 终局事件 {n_term} 个（真实起点段 {n_seg} 个，其中完整回合 {n_comp} 个）")
    for k in ("made", "blocked", "timeout", "opp_foul", "own_mistake", "other"):
        n = cls_count[k]
        P(f"    {k:8s} n={n:5d}（{pct(n/max(1,n_term))}） | A1奖励 mean={sum(cls_r1[k])/max(1,n):+7.2f} | 防守奖励 mean={sum(cls_rd[k])/max(1,n):+7.2f} | 出手时剩余 {sum(cls_trem[k])/max(1,n):.1f}s")
    att = cls_count["made"] + cls_count["blocked"]
    if att:
        P(f"    出手 {att} 次: 命中率≈{pct(cls_count['made']/att)} 被盖率≈{pct(cls_count['blocked']/att)}")

    # ---------- B. 按钮使用（A1） ----------
    dpg = prog1 - torch.cat([torch.zeros_like(prog1[:, :1]), prog1[:, :-1]], dim=1)
    dpg[init] = 0.0
    productive = press & (dpg > 0)
    wasted = press & (~(dpg > 0))
    invalid = press & (~allowed)
    P(f"\n[B] 按钮使用（A1）")
    P(f"    允许步占比={pct(allowed.float().mean())} | 按下步占比（全体）={pct(press.float().mean())} | 允许时按下率={pct((press & allowed).float().sum()/max(1,allowed.float().sum()))}")
    P(f"    非法按下（不允许却按，mask 校验）={int(invalid.sum())}")
    P(f"    按键中: 读条推进步={int(productive.sum())}（{pct(productive.float().sum()/max(1,press.float().sum()))}） 空按步={int(wasted.sum())}（{pct(wasted.float().sum()/max(1,press.float().sum()))}）")
    # 按键连续段长度（全体步）
    runs_hist = {"1": 0, "2-4": 0, "5-9": 0, "10-19": 0, ">=20": 0}
    for e in range(E):
        idx = torch.nonzero(press[e]).flatten().tolist()
        if not idx:
            continue
        cur = 1
        for i in range(1, len(idx)):
            if idx[i] == idx[i - 1] + 1:
                cur += 1
            else:
                runs_hist["1" if cur == 1 else "2-4" if cur <= 4 else "5-9" if cur <= 9 else "10-19" if cur <= 19 else ">=20"] += 1
                cur = 1
        runs_hist["1" if cur == 1 else "2-4" if cur <= 4 else "5-9" if cur <= 9 else "10-19" if cur <= 19 else ">=20"] += 1
    P(f"    按键连续段长度分布（全体 {sum(runs_hist.values())} 段）: {runs_hist}")
    # 段内：读条开始/完整读条/中断
    starts_seg = completed_seg = interrupt_seg = 0
    seg_with_attempt = seg_with_press = 0
    for e, s, en, c in segs:
        pseg = prog1[e, s : en + 1]
        prev_seg = torch.cat([torch.zeros(1), pseg[:-1]])
        n_start = int(((pseg > 0) & (prev_seg == 0)).sum())
        starts_seg += n_start
        if n_start:
            seg_with_press += 1
        att_seg = bool(done[e, en]) and prog1[e, en].item() >= 0.85
        if att_seg:
            completed_seg += 1
            seg_with_attempt += 1
        interrupt_seg += max(0, n_start - (1 if att_seg else 0))
    P(f"    真实起点段 {n_seg}: 有按键的段={pct(seg_with_press/n_seg)} 有出手的段={pct(seg_with_attempt/n_seg)}")
    P(f"    读条开始 {starts_seg} 次（每段 {starts_seg/max(1,n_seg):.2f}）| 完成出手 {completed_seg} 次 | 中断≈{interrupt_seg} 次")

    # ---------- C. 新旧映射速度对比 ----------
    u = act_c[..., 0, :]
    mag = u.norm(dim=-1)
    v_old = u * (mag / 5.0).clamp(min=0).pow(0.5).unsqueeze(-1)
    n_old = v_old.norm(dim=-1)
    dz = n_old < 0.2
    v_old[dz] = 0

    def clampn(v, m=5.0):
        n = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return v * (m / n).clamp(max=1.0)

    v_old, v_new = clampn(v_old), clampn(u)
    n_old, n_new = v_old.norm(dim=-1), v_new.norm(dim=-1)
    P(f"\n[C] 新旧映射速度对比（A1 命令 u 同一分布）")
    stats("|u| 命令模长", mag)
    stats("旧映射 |v_old|", n_old)
    stats("新映射 |v_new|", n_new)
    P(f"    旧映射死区清零占比={pct(dz.float().mean())} | <0.5 m/s: 旧={pct((n_old<0.5).float().mean())} 新={pct((n_new<0.5).float().mean())} | <0.2: 旧={pct((n_old<0.2).float().mean())} 新={pct((n_new<0.2).float().mean())}")
    P(f"    |v_new|>|v_old| 占比={pct((n_new>n_old).float().mean())} | 非死区步平均放大={float((n_new[~dz]/(n_old[~dz]+1e-6)).mean()):.2f}×")
    stats("实际速度 |vel|（A1 obs）", spd[..., 0])

    # ---------- D. 奖励稀释 ----------
    P(f"\n[D] 奖励稀释（A1）")
    tot = rew[..., 0]
    for a, b in [(0, 49), (50, 99), (100, 149), (150, 199)]:
        seg = tot[:, a : b + 1]
        P(f"    步 {a:>3}-{b:<3} 每步平均={seg.mean():+.4f}")
    comp_r = torch.tensor([rew[e, s : en + 1, 0].sum().item() for e, s, en, c in segs if c])
    comp_r1t = torch.tensor([rew[e, en, 0].item() for e, s, en, c in segs if c])
    stats("完整回合 A1 总奖励（稠密+终局）", comp_r)
    stats("完整回合 A1 终局一步奖励", comp_r1t)
    # 过中线时机（真实起点段，只取长度>=40）
    blocked_4s = n_4s = cross_4s = never_cross = 0
    first_cross = []
    first_cross_full = []
    for e, s, en, c in segs:
        if en - s + 1 < 40:
            continue
        n_4s += 1
        ys = pos[e, s : en + 1, 0, 1]
        nz = (ys > 0).nonzero()
        if len(nz):
            fc = int(nz[0].item())
            first_cross_full.append(fc)
            if fc < 40:
                cross_4s += 1
        else:
            never_cross += 1
        if not len((ys[:40] > 0).nonzero()):
            blocked_4s += 1
        else:
            first_cross.append(int((ys[:40] > 0).nonzero()[0].item()))
    P(f"    开局 4s（真实起点段 {n_4s}）从未过中线={pct(blocked_4s/max(1,n_4s))} | 4s 内过线={cross_4s}")
    if first_cross_full:
        fc_t = torch.tensor(first_cross_full).float()
        q = torch.quantile(fc_t, torch.tensor([0.25, 0.5, 0.75]))
        P(f"    全程: 过线段={len(first_cross_full)}/{n_4s}（{pct(len(first_cross_full)/max(1,n_4s))}） 首次过线步 p25/p50/p75={q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}（99=9.9s）")
    else:
        P(f"    全程: 过线段=0/{n_4s}")
    P(f"    A1 步速<0.2 占比={pct((spd[...,0]<0.2).float().mean())} | 圈内占比={pct(in_spot[...,0].float().mean())} | 读条中占比={pct((prog1>0).float().mean())}")

    # ---------- E. 出手/被盖 EV ----------
    P(f"\n[E] 出手 EV")
    if att:
        P(f"    出手率（真实起点段）={pct(seg_with_attempt/n_seg)} | 每段出手={seg_with_attempt/max(1,n_seg):.3f} 次")
        P(f"    出手命中率≈{pct(cls_count['made']/att)} 被盖率≈{pct(cls_count['blocked']/att)}")
        # 按剩余时间分箱
        for lo, hi, nm in [(12, 20.5, ">12s"), (6, 12, "6-12s"), (0, 6, "<6s")]:
            m_n = m_m = m_b = 0
            for e, t in term_events:
                if prog1[e, t].item() < 0.85:
                    continue
                tr = obs[e, t, 0, 8].item() * 20.0
                if lo <= tr < hi:
                    r1 = rew[e, t, 0].item()
                    if r1 >= 60:
                        m_m += 1
                    else:
                        m_b += 1
                    m_n += 1
            if m_n:
                P(f"    剩余 {nm:6s}: 出手 {m_n:4d}（占出手 {pct(m_n/att)}） 命中率={pct(m_m/m_n)} 被盖率={pct(m_b/m_n)}")
        # 被盖代价
        at = torch.tensor(cls_r1["made"] + cls_r1["blocked"])
        mc = torch.tensor(cls_r1["made"])
        bl = torch.tensor(cls_r1["blocked"])
        stats("出手 A1 奖励（全部）", at)
        stats("命中 A1 奖励", mc)
        stats("被盖 A1 奖励", bl)
    rd_at = torch.tensor(cls_rd["made"] + cls_rd["blocked"])
    rd_to = torch.tensor(cls_rd["timeout"])
    stats("防守者奖励@出手", rd_at)
    stats("防守者奖励@超时", rd_to)

    # ---------- F. advantage 速览 ----------
    P(f"\n[F] advantage 速览")
    roles = ["A1", "A2", "D1", "D2"]
    charging = prog1 > 0
    for i, rn in enumerate(roles):
        a = adv[..., i]
        P(f"    {rn}: mean={a.mean():+.3f} std={a.std():.3f} | 读条中={a[charging].mean():+.3f} | 圈内={a[in_spot[...,0]].mean():+.3f}")
    a1 = adv[..., 0]
    P(f"    A1 读条中 std={a1[charging].std():.2f} vs 非读条 std={a1[~charging].std():.2f} | D1 读条中={adv[...,2][charging].mean():+.3f}")

    if out:
        with open(out, "w") as f:
            f.write("\n".join(lines))
        print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
