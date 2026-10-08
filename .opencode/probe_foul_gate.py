#!/usr/bin/env python3
"""量化：接触对里 |Δv| 与"责任人接近分量"的差异。
用法: python probe_foul_gate.py <ckpt> [标签]
说明：用 buffer 的 state（23 维：A1pos/vel、A2、D1、D2、spot、basket、time）重建所有位置与速度，
      对 dist<0.6m 的接触对计算 |Δv| 与 approach 分量，比较"当前门槛(|Δv|>0.5)"与
      "提案门槛(责任方接近分量>0.5)"会覆盖哪些接触。
      注意：buffer 存的是该帧的观测速度，而判罚用的是上一物理步的 p_vels，两者相差一帧，统计意义足够。
"""
import sys
import torch

POS_DIV = torch.tensor([4.0, 7.5])      # W/2, L/2
VEL_DIV = 5.0                            # v_max
CONTACT_R = 0.6                          # 2*agent_radius
V_TH = 0.5
EPS = 1e-3


def _get(obj, key, default=None):
    try:
        v = obj[key]
        return v
    except Exception:
        return default


def main():
    ckpt = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else ""

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    buf = ck["buffer_agents"]
    st = buf["_storage"]["_storage"] if "_storage" in buf else buf
    if hasattr(st, "keys"):
        print("[keys]", list(st.keys())[:20])
    state = _get(st, "state")
    assert state is not None, "找不到 state"
    state = state.float()
    nxt = _get(st, "next")
    done = _get(nxt, "done") if nxt is not None else None
    if done is None:
        done = _get(st, "done")
    done = done.float()
    while done.dim() > 2:
        done = done[..., 0]
    is_init = _get(st, "is_init")
    if is_init is not None:
        is_init = is_init.bool()
        while is_init.dim() > 2:
            is_init = is_init[..., 0]
    # 某些实现把 done 放在 next 下、is_init 在根；归一化到 (E,T)
    if done.shape[0] != state.shape[0]:
        done = done.transpose(0, 1)
    if is_init is not None and is_init.shape[0] != state.shape[0]:
        is_init = is_init.transpose(0, 1)

    E, T, _ = state.shape
    pos = torch.stack([state[..., 0:2], state[..., 6:8], state[..., 10:12], state[..., 14:16]], dim=-2) * POS_DIV
    vel = torch.stack([state[..., 2:4], state[..., 8:10], state[..., 12:14], state[..., 16:18]], dim=-2) * VEL_DIV
    print(f"[{label}] state={tuple(state.shape)} pos={tuple(pos.shape)} done={tuple(done.shape)} "
          f"is_init={'None' if is_init is None else tuple(is_init.shape)}")

    # 有效帧：从 is_init 之后到下一次 done（不含 done 之后）
    # 关键：终局帧（碰撞判罚就发生在这里）必须保留 —— done[t] 表示"第 t 步结束了回合"，
    # 但同一行里会有多局（is_init 重新拉起），所以用 active 而不是永久 seen。
    valid = torch.zeros(E, T, dtype=torch.bool)
    active = torch.zeros(E, dtype=torch.bool)
    for t in range(T):
        if is_init is not None:
            active = active | is_init[:, t]
        else:
            active = torch.ones_like(active)   # 无 is_init 时保守处理
        valid[:, t] = active
        active = active & (~done[:, t].bool())   # 本帧仍有效，下一帧才失效
    term = done.bool()
    n_term = int((valid & term).sum())
    print(f"[{label}] 有效帧 {valid.sum().item()} / {E*T}（含终局帧 {n_term}）")

    rows = []
    for i in range(4):
        for j in range(i + 1, 4):
            rel = pos[..., j, :] - pos[..., i, :]
            d = rel.norm(dim=-1)
            contact = (d < CONTACT_R) & valid
            if not contact.any():
                continue
            n = rel / (d.unsqueeze(-1) + 1e-6)
            vi, vj = vel[..., i, :], vel[..., j, :]
            ai = torch.clamp((vi * n).sum(-1), min=0.0)
            aj = torch.clamp(-(vj * n).sum(-1), min=0.0)
            dv = (vi - vj).norm(dim=-1)
            tied = (ai - aj).abs() <= EPS
            i_active = torch.where(tied, vi.norm(dim=-1) >= vj.norm(dim=-1), ai > aj)
            a_act = torch.where(i_active, ai, aj)
            a_max = torch.maximum(ai, aj)
            rows.append((f"{i}-{j}", contact, term[contact], dv[contact], a_act[contact], a_max[contact],
                         i_active[contact], ai[contact], aj[contact],
                         vi.norm(dim=-1)[contact], vj.norm(dim=-1)[contact], (i < 2) != (j < 2), i < 2))

    tot_pairs = sum(int(r[1].sum()) for r in rows)
    print(f"\n[{label}] 接触对总数 {tot_pairs}（只计有效帧；其中终局帧上的接触 "
          f"{sum(int(r[2].sum()) for r in rows)}）")
    if tot_pairs == 0:
        return
    dv_all = torch.cat([r[3] for r in rows])
    aa_all = torch.cat([r[4] for r in rows])
    am_all = torch.cat([r[5] for r in rows])
    term_flags = torch.cat([r[2] for r in rows])

    def q(x, name):
        if x.numel() == 0:
            print(f"  {name:28s} (空)")
            return
        qs = torch.quantile(x, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
        print(f"  {name:28s} mean={x.mean():6.3f} p10={qs[0]:5.2f} p25={qs[1]:5.2f} p50={qs[2]:5.2f} "
              f"p75={qs[3]:5.2f} p90={qs[4]:5.2f} p99={qs[5]:6.2f}")

    print("\n-- 接触对上的分布 --")
    q(dv_all, "|Δv| 相对速度")
    q(aa_all, "责任人接近分量")
    q(am_all, "max(接近分量)")

    def gate_report(sel, name):
        dv, aa = dv_all[sel], aa_all[sel]
        n = dv.numel()
        if n == 0:
            print(f"\n-- {name}：无接触对 --")
            return
        cur = dv > V_TH
        new = aa > V_TH
        print(f"\n-- 门槛对比 [{name}]（分母 {n} 个接触对）--")
        print(f"  当前门槛 |Δv|>{V_TH}                     : {int(cur.sum())} ({100*cur.float().mean():.1f}%)")
        print(f"  提案门槛 责任方接近>{V_TH}               : {int(new.sum())} ({100*new.float().mean():.1f}%)")
        print(f"  两者都为真（真正的对撞/冲刺）            : {int((cur&new).sum())} ({100*(cur&new).float().mean():.1f}%)")
        graze = cur & ~new
        chase = new & ~cur
        print(f"  ① |Δv|>{V_TH} 但接近≤{V_TH}（擦身/横滑；提案会免责）: {int(graze.sum())} "
              f"（占当前判罚的 {100*graze.sum()/max(int(cur.sum()),1):.1f}%）")
        print(f"  ② 接近>{V_TH} 但 |Δv|≤{V_TH}（追击型：对方在逃，Δv 被抵消；当前漏判）: {int(chase.sum())}")
        if graze.any():
            q(dv[graze], "   ①组 |Δv|")
            q(aa[graze], "   ①组 接近分量")
        if chase.any():
            q(dv[chase], "   ②组 |Δv|")
            q(aa[chase], "   ②组 接近分量")

    gate_report(torch.ones_like(dv_all, dtype=torch.bool), "全部有效帧")
    gate_report(term_flags, "终局帧（真正触发判罚的地方）")

    # 阈值扫描（终局帧）：把"严重碰撞"的标准从 |Δv| 换成接近分量，会有多少判罚
    dv_t, aa_t = dv_all[term_flags], aa_all[term_flags]
    print("\n-- 终局帧阈值扫描（分母 %d 个接触对）--" % dv_t.numel())
    print("  接近阈值 V :  只按 approach>V   |  |Δv|>0.5 且 approach>V")
    for Vg in (0.2, 0.3, 0.35, 0.4, 0.5):
        print(f"    {Vg:4.1f}    :  {int((aa_t > Vg).sum()):6d} ({100*(aa_t>Vg).float().mean():5.1f}%) | "
              f"{int(((dv_t > 0.5) & (aa_t > Vg)).sum()):6d} ({100*((dv_t>0.5)&(aa_t>Vg)).float().mean():5.1f}%)")
    print(f"  当前口径 |Δv|>0.5（无 approach 条件）: {int((dv_t>0.5).sum())} ({100*(dv_t>0.5).float().mean():.1f}%)  "
          f"| 其中 approach≤0.2 的：{int(((dv_t>0.5)&(aa_t<=0.2)).sum())}")

    # 终局帧上：当前门槛命中者的"队伍归属"拆分（决定 码13 会不会大幅消失）
    print("\n-- 终局帧：归属拆分（旧门槛 |Δv|>0.5  /  新门槛 approach>0.35）--")
    tot_old = tot_new = tot_old_att = tot_new_att = 0
    for (name, contact, tcontact, dv, aa, am, ia, _ai, _aj, _si, _sj, _cross, _iatt) in rows:
        # 该 pair 在终局帧上的命中（contact 已只含有效帧）
        m = tcontact
        if not m.any():
            continue
        hit = dv[m] > V_TH
        if not hit.any():
            continue
        act = ia[m][hit]
        i_idx, j_idx = [int(x) for x in name.split("-")]
        atk = {0, 1}
        i_is_att = i_idx in atk
        resp_is_att = torch.where(act, torch.full_like(act, i_is_att), torch.full_like(act, not i_is_att))
        # 新门槛：同一批接触对里 approach>0.35 的
        hit_new = aa[m] > 0.35
        act_new = ia[m][hit_new]
        resp_is_att_new = torch.where(act_new, torch.full_like(act_new, i_is_att), torch.full_like(act_new, not i_is_att))
        cross = (i_is_att != (j_idx in atk))
        n_att, n_new, n_new_att = int(resp_is_att.sum()), int(hit_new.sum()), int(resp_is_att_new.sum())
        tot_old += int(hit.sum()); tot_old_att += n_att
        tot_new += n_new; tot_new_att += n_new_att
        print(f"  pair {name}（{'跨队' if cross else '同队'}）: 旧门槛命中 {int(hit.sum())}（攻方主动 {n_att}）"
              f" | 新门槛命中 {n_new}（攻方主动 {n_new_att}）")
    print(f"  == 合计: 旧 {tot_old}（攻方主动 {tot_old_att}） -> 新 {tot_new}（攻方主动 {tot_new_att}）"
          f"，判罚量降到 {100*tot_new/max(tot_old,1):.1f}%")

    # ---- [方案A+] 合法防守位豁免：新判责 vs 对称判责（终局帧、门内接触对）----
    V_LEGAL, A_LEGAL, K_EXEMPT, TH = 0.3, 0.3, 1.5, 0.35
    n_gate = n_sym_att = n_new_att = n_flip_att2def = n_flip_def2att = 0
    flip_examples = []
    for (name, contact, tcontact, dv, am, _aa, _ia, ai, aj, si, sj, cross, iatt) in [
            (r[0], r[1], r[2], r[3], r[5], r[4], r[6], r[7], r[8], r[9], r[10], r[11], r[12]) for r in rows]:
        m = tcontact
        if not m.any():
            continue
        A, B_, S1, S2 = ai[m], aj[m], si[m], sj[m]
        amax = torch.maximum(A, B_)
        gate = amax > TH
        tied = (A - B_).abs() <= 1e-3
        i_sym = torch.where(tied, S1 >= S2, A > B_)
        act_sym_is_i = i_sym
        cross_t = torch.full_like(A, bool(cross)).bool()
        iatt_t = torch.full_like(A, bool(iatt)).bool()
        if bool(iatt & cross):            # i 是攻方、j 是防守方
            ap_atk, ap_def, sp_def = A, B_, S2
            def_is_i = torch.zeros_like(A).bool()
        else:                             # j 是攻方（同队项稍后由 cross_t 过滤）
            ap_atk, ap_def, sp_def = B_, A, S1
            def_is_i = torch.ones_like(A).bool()
        viol = torch.clamp(sp_def - V_LEGAL, min=0.0) + torch.clamp(ap_def - A_LEGAL, min=0.0)
        exempt = cross_t & ((ap_atk - K_EXEMPT * viol) <= ap_def)
        act_new_is_i = torch.where(exempt, def_is_i, act_sym_is_i)
        g = gate
        sym_att = ((act_sym_is_i & iatt_t) | (~act_sym_is_i & (~iatt_t))) & cross_t
        new_att = ((act_new_is_i & iatt_t) | (~act_new_is_i & (~iatt_t))) & cross_t
        n_gate += int(g.sum())
        n_sym_att += int((g & sym_att).sum())
        n_new_att += int((g & new_att).sum())
        n_flip_att2def += int((g & sym_att & ~new_att).sum())
        n_flip_def2att += int((g & (~sym_att) & new_att).sum())
        idx = (g & exempt & cross).nonzero().flatten()[:3].tolist()
        for t in idx:
            flip_examples.append((name, float(amax[t]), float(ap_atk[t]), float(ap_def[t]), float(sp_def[t])))
    print("\n-- [方案A+] 合法防守位豁免（终局帧、门内接触对）--")
    print(f"  门内接触对 {n_gate}；对称判责下攻方担责 {n_sym_att} -> 新判责 {n_new_att}"
          f"（豁免翻转 {n_flip_att2def} 个；反向翻转 {n_flip_def2att} 个）")
    for (nm, amax, ap_atk, ap_def, sp_def) in flip_examples[:6]:
        print(f"    例 {nm}: a_max={amax:.2f} 攻接近={ap_atk:.2f} 守接近={ap_def:.2f} 守速度={sp_def:.2f} -> 责任归防守")


if __name__ == "__main__":
    main()
