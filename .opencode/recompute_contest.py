"""离线重算 v3：新规则（延误保底 + 干扰加成 + 放投罚 ≤10% 减免）对出手回合防守奖励的影响。
   纯 buffer 张量重算，不建环境。checkpoint_24000000.pt (iter 80)。"""
import torch

CK = "outputs/2026-10-02_15-07-00/mappo_layup_sequencemodel__2ac16b45_26_10_02-15_07_00/checkpoints/checkpoint_24000000.pt"
SCR = torch.tensor([8.0, 15.0])
AG = 0.3
DEF_PROX = 3 * AG            # 0.9
PROX_THR = AG * 2.3          # 0.69
SIGMA = 0.30
GATE_K = 25.0
K_DELAY = 14000.0
K_PEN = 9000.0
DISCOUNT = 0.10
SCALE = 0.005
FLOORS = [0.0, 0.3, 0.5]

ck = torch.load(CK, map_location="cpu", weights_only=False)
st = ck["buffer_agents"]["_storage"]["_storage"]
nx = st["next"]
print("top keys:", list(st.keys()))
print("next keys:", list(nx.keys()))

obs = st["agents"]["observation"]
ng = nx["agents"] if "agents" in nx.keys() else nx
rw = ng["reward"]
term = nx["done"] if "done" in nx.keys() else st["done"]
top_done_sum = int(st["done"].float().sum())
next_done_sum = int(nx["done"].float().sum()) if "done" in nx.keys() else -1
print("done sums: top=", top_done_sum, "next=", next_done_sum)

if term.dim() == 3 and term.shape[-1] == 1:
    term = term[..., 0]
obs = obs.reshape(-1, 4, 41).float()
rw = rw.reshape(-1, 4).float()
term = term.reshape(-1).bool()
n_all = term.numel()
print("shapes after flatten:", tuple(obs.shape), tuple(rw.shape), tuple(term.shape))
print(f"帧数 {n_all} | 终局 {int(term.sum())} ({100*float(term.float().mean()):.2f}%) | "
      f"reward mean={rw.mean():.4f} min={rw.min():.2f} max={rw.max():.2f}")


def contest(defrel, basket_rel):
    denom = float((basket_rel ** 2).sum()) + 1e-6
    t = float((defrel * basket_rel).sum()) / denom
    perp2 = float(((defrel - t * basket_rel) ** 2).sum())
    d = float(defrel.norm())
    between = (t > 0) and (t < 1)
    soft = torch.sigmoid(torch.tensor(GATE_K * (DEF_PROX - d))).item()
    blocker = between and (perp2 < PROX_THR ** 2)
    c = torch.exp(torch.tensor(-perp2 / (2 * SIGMA ** 2))).item() * (1.0 if blocker else 0.0) * soft
    return min(max(c, 0.0), 1.0)


SHOT, TIMEUP = [], []
s = 0
for i in range(n_all):
    if not term[i]:
        continue
    er = rw[s:i + 1].sum(0)          # (4,)
    a1 = obs[i, 0]
    basket_rel = a1[37:39] * SCR
    r = float((1.0 - a1[8]).clamp(0, 1))
    c1 = contest(a1[17:19] * SCR, basket_rel)
    c2 = contest(a1[25:27] * SCR, basket_rel)
    rec = (r, c1, c2, float(rw[i, 2]), float(rw[i, 3]))
    if er[0] >= 40 and er[1] >= 40:
        SHOT.append(rec)
    elif er[2] > 20 and r > 0.9:
        TIMEUP.append(rec)
    s = i + 1

n_ep = int(term.sum())
ns = len(SHOT)
rs = torch.tensor([x[0] for x in SHOT])
c1 = torch.tensor([x[1] for x in SHOT])
c2 = torch.tensor([x[2] for x in SHOT])
old1 = torch.tensor([x[3] for x in SHOT])
old2 = torch.tensor([x[4] for x in SHOT])
print(f"\n终局 {n_ep}：出手回合 {ns} ({100*ns/max(n_ep,1):.1f}%) | 超时 {len(TIMEUP)} | 其他 {n_ep-ns-len(TIMEUP)}")
print(f"出手时 r: mean={rs.mean():.3f} p50={rs.median():.3f}")

for nm, cc in [("D1", c1), ("D2", c2)]:
    q = torch.quantile(cc, torch.tensor([0.10, 0.50, 0.90]))
    print(f"  contest {nm}: mean={cc.mean():.3f} p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f} | "
          f"c<0.05={100*float((cc<0.05).float().mean()):.1f}% | c>0.3={100*float((cc>0.3).float().mean()):.1f}%")
print(f"  两防守者都几乎无干扰(c<0.05)的出手: {100*float(((c1<0.05)&(c2<0.05)).float().mean()):.1f}%")

base = 70.0 * torch.pow(rs, 1.5)   # 旧延误项（scaled）
print(f"\n旧延误项(per defender, scaled) mean={base.mean():.2f} | 旧出手回合防守奖励: D1={old1.mean():+.2f} D2={old2.mean():+.2f}")
print("\n[各保底值下的重算]")
for f in FLOORS:
    g1 = base * (f + (1.0 - f) * c1)
    g2 = base * (f + (1.0 - f) * c2)
    d1 = (g1 - base) + 4.5 * c1
    d2 = (g2 - base) + 4.5 * c2
    dtot = d1 + d2
    new1 = old1 + d1
    new2 = old2 + d2
    print(f"  floor={f:.2f}: 新延误 D1={g1.mean():.2f} D2={g2.mean():.2f} | "
          f"Δ D1={d1.mean():+.2f} D2={d2.mean():+.2f} | 新出手奖励 D1={new1.mean():+.2f} D2={new2.mean():+.2f} | "
          f"每局合计 Δ={float(dtot.sum())/n_ep:+.2f}")
print("DONE")
