"""[交叉对打 v3] 新规则环境里：v38/v39（按键） vs v37（旧连续动作，无按键）。

用法（容器内）：
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/crossplay_v3.py --mode random --envs 192 --rounds 3 --out /tmp/xp3_random.json
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/crossplay_v3.py --mode det    --envs 192 --rounds 3 --out /tmp/xp3_det.json

设计：
- 对局全部在【新规则环境】里跑（混合动作 spec：continuous[4,2] + discrete[4,1]，按键，线性映射，200 步，被盖 -60）。
- v38 侧：直接调用复合动作 policy（按键 = 学到的离散位，mask 生效）。
- v37 侧：只输出 2 维连续命令（其原生非复合 policy）；按键适配 =「站住即读条」：
    press = (A1 在圈内) 且 (|command| < 2.0)   # 复刻旧规则 is_ready_to_shoot 的动作阈值
  其防守方忽略第 3 通道（新环境本身也忽略）。
- 两个 policy 固定权重（各装一次）；每个 cell 只换"谁控制攻方/守方"。
"""
import argparse
import json
import os
import time

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch
from types import SimpleNamespace
from torchrl.data import Bounded
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

# 复合动作策略必须关掉 lp 聚合（否则 ProbabilisticActor 报
# "composite_lp_aggregate is set to True but log_prob_keys were passed"）。
# 与 BenchMARL/benchmarl/experiment/experiment.py:449/802 同款处理。
set_composite_lp_aggregate(False).set()

from benchmarl.environments import LayupTask
from benchmarl.environments.layup.common import VmasEnvWithState
from benchmarl.algorithms import MappoConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.utils import _add_rnn_transforms
from torchrl.envs.utils import MarlGroupMapType

BASE = "/home/vscode/workspace/BenchMARL/outputs"
CKPTS = {
    "v38": f"{BASE}/2026-10-03_05-39-14/mappo_layup_sequencemodel__a35f3c2b_26_10_03-05_39_14/checkpoints/checkpoint_45000000.pt",
    "v37": f"{BASE}/2026-10-02_18-54-30/mappo_layup_sequencemodel__8f108ad3_26_10_02-16_39_27/checkpoints/checkpoint_63000000.pt",
}
WIN_CODES = {1, 2, 3, 4, 5}
REASON_NAMES = {1: "shot", 2: "opp_foul", 3: "opp_wall", 4: "opp_cross", 5: "opp_ff",
                11: "blocked", 12: "timeout", 13: "own_foul", 14: "own_wall", 15: "own_ff"}
GROUP_SINGLE = {"agents": ["attacker_1", "attacker_2", "defender_1", "defender_2"]}
PRESS_THRESH = 2.0  # 旧规则 a_shot_threshold


def strip_orig(k):
    return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")


def extract_actor(ck, group):
    pfx = "actor_network_params."
    sd = {}
    for k, v in ck[f"loss_{group}"].items():
        if isinstance(v, torch.Tensor) and k.startswith(pfx):
            sd[strip_orig(k[len(pfx):])] = v
    return sd


def install(policy, actor_sd, prefix=None):
    psd = policy.state_dict()
    nkeys = list(psd.keys())
    if prefix is None:
        best, best_hit = None, -1
        for pfx in ("module.0.", "module.1.", ""):
            hits = 0
            for k, v in actor_sd.items():
                nk = strip_orig(k)
                c = [pk for pk in nkeys if pk.startswith(pfx) and strip_orig(pk).endswith(nk)]
                if len(c) == 1 and tuple(psd[c[0]].shape) == tuple(v.shape):
                    hits += 1
            if hits > best_hit:
                best, best_hit = pfx, hits
        prefix = best
    chosen = {}
    hit = miss = amb = 0
    amb_detail = []
    for k, v in actor_sd.items():
        nk = strip_orig(k)
        cands = [pk for pk in nkeys if pk.startswith(prefix) and strip_orig(pk).endswith(nk)]
        if len(cands) == 1:
            pk = cands[0]
            if tuple(psd[pk].shape) == tuple(v.shape):
                chosen[pk] = v; hit += 1
            else:
                amb += 1; amb_detail.append((k, len(cands)))
        elif not cands:
            miss += 1
        else:
            amb += 1; amb_detail.append((k, len(cands)))
    if amb_detail:
        print(f"[install] ambiguous/mismatch keys ({len(amb_detail)}):", flush=True)
        for k, c in amb_detail[:8]:
            print(f"    {k}  (候选 {c})", flush=True)
    full = dict(psd); full.update(chosen)
    policy.load_state_dict(full, strict=True)
    return prefix, hit, miss, amb


def shell_for(env, algo_cfg, model_cfg, critic_cfg, cfg, task, with_mask=True):
    return SimpleNamespace(
        config=cfg, algorithm_config=algo_cfg, model_config=model_cfg,
        critic_model_config=critic_cfg, task=task, group_map=task.group_map(env),
        continuous_actions=True, seed=0, on_policy=algo_cfg.on_policy(),
        observation_spec=task.observation_spec(env), action_spec=task.action_spec(env),
        info_spec=task.info_spec(env), state_spec=task.state_spec(env),
        action_mask_spec=(task.action_mask_spec(env) if with_mask else None),
    )


def build_new(task, cfg, n_envs, ckpt):
    """v38/v39：新规则（复合动作 + 按键 mask）"""
    attn = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml")
    model_cfg = SequenceModelConfig([attn, GruConfig.get_from_yaml()], intermediate_sizes=[256])
    critic_cfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    a_cfg = MappoConfig.get_from_yaml(); a_cfg.share_param_actor = True; a_cfg.share_param_critic = False
    a_cfg.loc_bound = 3.0   # 保真：v37/v38 训练时 loc_bound=3.0（A+B 之后才改成 5.0）

    env_spec = task.get_env_fun(num_envs=n_envs, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    shell = shell_for(env_spec, a_cfg, model_cfg, critic_cfg, cfg, task, with_mask=True)
    algo = a_cfg.get_algorithm(shell)
    env = _add_rnn_transforms(lambda: env_spec, task.group_map(env_spec), model_cfg)()
    policy = algo.get_policy_for_collection().to("cpu")
    policy.eval()
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    pfx, h, m, amb = install(policy, extract_actor(ck, "agents"))
    print(f"[build] new(v38) policy: prefix={pfx} hit={h} miss={m} amb={amb}", flush=True)
    return policy, env, model_cfg


def build_old(task, cfg, n_envs, ckpt):
    """v37：非复合 2 维连续动作（旧 spec）"""
    attn = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml")
    g = GruConfig.get_from_yaml(); g.role_out_dims = []      # v37 的按角色头宽度一致 = 4
    model_cfg = SequenceModelConfig([attn, g], intermediate_sizes=[256])
    critic_cfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    a_cfg = MappoConfig.get_from_yaml(); a_cfg.share_param_actor = True; a_cfg.share_param_critic = False
    a_cfg.loc_bound = 3.0   # 保真：v37 训练时 loc_bound=3.0

    env_spec = VmasEnvWithState(scenario="layup", num_envs=n_envs, continuous_actions=True, seed=0,
                                device="cpu", clamp_actions=True, group_map=GROUP_SINGLE, **dict(task.config))
    shell = shell_for(env_spec, a_cfg, model_cfg, critic_cfg, cfg, task, with_mask=False)
    # v37 原生动作 = 2 维连续（旧规则，无按键通道）：把 group 叶子 spec 换成 2 维有界连续，
    # 否则动作头按 3 维建出 6 个参数，与 v37 权重（4 个参数）的 6 个角色头键形状不符。
    _leaf = shell.action_spec[("agents", "action")]
    _spec_2d = shell.action_spec.clone()
    _spec_2d[("agents", "action")] = Bounded(
        low=_leaf.space.low[..., :2].clone(),
        high=_leaf.space.high[..., :2].clone(),
        shape=tuple(_leaf.shape[:-1]) + (2,),
        dtype=_leaf.dtype, device=_leaf.device,
    )
    shell.action_spec = _spec_2d
    algo = a_cfg.get_algorithm(shell)
    env = _add_rnn_transforms(lambda: env_spec, GROUP_SINGLE, model_cfg)()
    policy = algo.get_policy_for_collection().to("cpu")
    policy.eval()
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    pfx, h, m, amb = install(policy, extract_actor(ck, "agents"))
    print(f"[build] old(v37) policy: prefix={pfx} hit={h} miss={m} amb={amb}", flush=True)
    return policy, env


class Driver:
    """把两边的动作拼成一个 4-agent 的复合动作：[continuous(4,2), discrete(4,1)]"""

    def __init__(self, p_new, p_old, td_old, a_side, d_side, dbg=False):
        self.p_new, self.p_old, self.td_old = p_new, p_old, td_old
        self.a_side, self.d_side = a_side, d_side
        self.dbg = dbg
        self._n = 0

    def __call__(self, td):
        obs = td.get(("agents", "observation"))
        is_init = td.get("is_init")

        out_old = None
        if self.a_side == "v37" or self.d_side == "v37":
            tmpl = self.td_old
            tmpl.set(("agents", "observation"), obs)
            tmpl.set("is_init", is_init)
            ii = is_init.bool() if is_init.dim() == 1 else is_init.bool().reshape(is_init.shape[0], -1)[:, 0]
            if ii.any():
                s = tmpl.get(("agents", "_hidden_gru_1"), None)
                if s is not None:
                    s[ii] = 0
            out_old = self.p_old(tmpl)
            ns = out_old.get(("next", "agents", "_hidden_gru_1"), None)
            if ns is not None:
                tmpl.set(("agents", "_hidden_gru_1"), ns.detach())
            if self.dbg and self._n == 1:
                d = tmpl.get(("agents", "_hidden_gru_1")).abs().max().item()
                print(f"    [state] old 第二步 |h|max = {d:.6f}", flush=True)

        # 总是调用新侧策略：确保复合动作容器存在（否则 (v37,v37) 单元里
        # td 里根本没有 ("agents","action") 键）。之后按边覆盖具体通道即可。
        self.p_new(td)
        act = td.get(("agents", "action"))
        cont = act.get("continuous")
        disc = act.get("discrete")

        if self.a_side == "v37":
            a_old = out_old.get(("agents", "action"))          # [E,4,2]
            cont[:, :2] = a_old[:, :2]
            mask = td.get(("agents", "action_mask"))[:, 0, 1]  # A1 在圈内才允许
            press = (mask & (a_old[:, 0, :].norm(dim=-1) < PRESS_THRESH)).to(disc.dtype)
            tgt = disc[:, 0].reshape(press.shape)
            tgt[:] = press
        else:
            d0 = disc[:, 0].reshape(disc[:, 0].shape)
            if self.dbg and self._n == 0:
                print(f"    [act] new disc A1 按下比例 = {float((d0 > 0).float().mean()):.3f}", flush=True)
        if self.d_side == "v37":
            a_old = out_old.get(("agents", "action"))
            cont[:, 2:] = a_old[:, 2:]
            tgt = disc[:, 2].reshape(disc[:, 2].shape)
            tgt.zero_()

        if self._n == 0:
            self.a0_fp = (float(cont.mean()), float(cont.std()), float(disc.float().mean()))
        self._n += 1
        return td


def run_rollouts(env, driver, max_steps, n_rounds, mode):
    et = ExplorationType.DETERMINISTIC if mode == "det" else ExplorationType.RANDOM
    reasons = {}
    n_eps = n_win = n_shot = 0
    t0 = time.time()
    with torch.no_grad():
        with set_exploration_type(et):
            for _ in range(n_rounds):
                roll = env.rollout(max_steps=max_steps, policy=driver,
                                   auto_cast_to_device=True, break_when_any_done=False)
                done = roll.get(("next", "done"))
                tr = roll.get(("next", "agents", "info", "termination_reason"))
                E, T = done.shape[0], done.shape[1]
                for e in range(E):
                    idx = done[e, :, 0].nonzero(as_tuple=True)[0]
                    code = 0
                    if idx.numel() > 0:
                        row = tr[e, int(idx[0])].flatten()
                        nz = row[row > 0]
                        code = int(nz[0]) if nz.numel() else 0
                    reasons[code] = reasons.get(code, 0) + 1
                    n_eps += 1
                    if code in WIN_CODES:
                        n_win += 1
                    if code == 1:
                        n_shot += 1
    return dict(episodes=n_eps, win=n_win, shot=n_shot, reasons=reasons, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "det"])
    ap.add_argument("--envs", type=int, default=192)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/tmp/xp3_result.json")
    ap.add_argument("--v38-ckpt", default=None, help="v38 侧 checkpoint 覆盖（训练轮数对照实验用）")
    ap.add_argument("--cells", default="", help="只跑指定 cell（逗号分隔，如 v37_atk_vs_v38_def）")
    args = ap.parse_args()

    torch.manual_seed(0)
    task = LayupTask.LAYUP.get_from_yaml()
    cfg = ExperimentConfig.get_from_yaml()
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    max_steps = task.max_steps(None)

    n = 16 if args.smoke else args.envs
    rounds = 1 if args.smoke else args.rounds
    print(f"[cfg] mode={args.mode} envs={n} rounds={rounds} max_steps={max_steps} threshold=0.2", flush=True)

    p_new, env, _ = build_new(task, cfg, n, args.v38_ckpt or CKPTS["v38"])
    p_old, env_old = build_old(task, cfg, n, CKPTS["v37"])
    td_old = env_old.reset()

    cells = [("v38", "v38"), ("v37", "v37"), ("v38", "v37"), ("v37", "v38")]
    if args.cells:
        want = set(args.cells.split(","))
        cells = [c for c in cells if f"{c[0]}_atk_vs_{c[1]}_def" in want]
    print(f"[cfg] v38_ckpt={args.v38_ckpt or CKPTS['v38']}", flush=True)
    out = {"mode": args.mode, "envs": n, "rounds": rounds, "cells": {}}
    for a_side, d_side in cells:
        driver = Driver(p_new, p_old, td_old.clone(), a_side, d_side, dbg=args.smoke)
        stat = run_rollouts(env, driver, max_steps, rounds, args.mode)
        key = f"{a_side}_atk_vs_{d_side}_def"
        out["cells"][key] = stat
        wr = 100.0 * stat["win"] / max(1, stat["episodes"])
        sr = 100.0 * stat["shot"] / max(1, stat["episodes"])
        top = sorted(stat["reasons"].items(), key=lambda kv: -kv[1])[:4]
        top_s = " ".join(f"{REASON_NAMES.get(c, c)}:{v}" for c, v in top)
        print(f"[cell] {key:22s} win={wr:5.1f}% 码1={sr:4.1f}% eps={stat['episodes']} "
              f"({stat['seconds']}s) fp={driver.a0_fp} | {top_s}", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
