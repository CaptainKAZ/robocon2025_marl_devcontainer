"""[兼容层交叉对打 v2] 单组环境里混搭：v36（单组共享主干）vs v34/v35（旧两组分组模式）。

用法（容器内）：
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/crossplay_v2.py --mode random --envs 256 --rounds 4 --out /tmp/xp2_random.json
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/crossplay_v2.py --mode det    --envs 256 --rounds 4 --out /tmp/xp2_det.json

设计：
- 所有对局都在【单组环境】(ALL_IN_ONE_GROUP, 与 v36/v37 训练一致) 里跑。
- v36 侧：直接调用单组 policy（状态存在 env td 的 ('agents','_hidden_gru_1') 里，原样）。
- 旧分组侧：用两份独立的两组 env 模板 td，把单组 obs 拆成 attacker/defender 两半喂进去，
  动作再拼回 ('agents','action')。状态在两模板 td 里原地滚动。
- 任一侧为 v36 时另一侧可用旧权重；两侧都是旧权重时用同一个 duo policy 的两棵子树。
"""
import argparse
import json
import os
import time

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch
from types import SimpleNamespace
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.environments import LayupTask
from benchmarl.environments.layup.common import VmasEnvWithState
from benchmarl.algorithms import MappoConfig, EnsembleAlgorithmConfig
from benchmarl.models import EnsembleModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.utils import _add_rnn_transforms

BASE = "/home/vscode/workspace/BenchMARL/outputs"
SINGLE_RUN = "2026-10-02_15-07-00/mappo_layup_sequencemodel__2ac16b45_26_10_02-15_07_00"
DUO_RUN = "ensemblealgorithm_layup_ensemblemodel__9d8e5c85_26_09_30-07_04_09"

CKPTS = {
    "v36": f"{BASE}/{SINGLE_RUN}/checkpoints/checkpoint_30000000.pt",
    "v34": f"{BASE}/2026-10-02_08-47-52/{DUO_RUN}/checkpoints/checkpoint_86250000.pt",
    "v35": f"{BASE}/2026-10-02_10-25-09/{DUO_RUN}/checkpoints/checkpoint_131250000.pt",
}
DUO_SIDES = ["v34", "v35"]
WIN_CODES = {1, 2, 3, 4, 5}
REASON_NAMES = {1: "shot", 2: "opp_foul", 3: "opp_wall", 4: "opp_cross", 5: "opp_ff",
                11: "blocked", 12: "timeout", 13: "own_foul", 14: "own_wall", 15: "own_ff"}
GROUP_SINGLE = {"agents": ["attacker_1", "attacker_2", "defender_1", "defender_2"]}
GROUP_DUO = {"attacker": ["attacker_1", "attacker_2"], "defender": ["defender_1", "defender_2"]}


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
    for k, v in actor_sd.items():
        nk = strip_orig(k)
        cands = [pk for pk in nkeys if pk.startswith(prefix) and strip_orig(pk).endswith(nk)]
        if len(cands) == 1:
            pk = cands[0]
            if tuple(psd[pk].shape) == tuple(v.shape):
                chosen[pk] = v; hit += 1
            else:
                amb += 1
        elif not cands:
            miss += 1
        else:
            amb += 1
    full = dict(psd); full.update(chosen)
    policy.load_state_dict(full, strict=True)
    return prefix, hit, miss, amb


def shell_for(env, algo_cfg, model_cfg, critic_cfg, cfg, task):
    return SimpleNamespace(
        config=cfg, algorithm_config=algo_cfg, model_config=model_cfg,
        critic_model_config=critic_cfg, task=task, group_map=task.group_map(env),
        continuous_actions=True, seed=0, on_policy=algo_cfg.on_policy(),
        observation_spec=task.observation_spec(env), action_spec=task.action_spec(env),
        info_spec=task.info_spec(env), state_spec=task.state_spec(env),
        action_mask_spec=task.action_mask_spec(env),
    )


def build_single(task, cfg, n_envs):
    from benchmarl.algorithms import MappoConfig as MP
    attn = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml")
    attn.encoders_per_role = False   # v36 训练时的取值
    model_cfg = SequenceModelConfig([attn, GruConfig.get_from_yaml()], intermediate_sizes=[256])
    critic_cfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    a_cfg = MP.get_from_yaml(); a_cfg.share_param_actor = True; a_cfg.share_param_critic = False

    env_spec = task.get_env_fun(num_envs=4, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    env_spec = TransformedEnv(env_spec, Compose(*task.get_env_transforms(env_spec)).clone())
    shell = shell_for(env_spec, a_cfg, model_cfg, critic_cfg, cfg, task)
    algo = a_cfg.get_algorithm(shell)
    env = _add_rnn_transforms(lambda: env_spec, task.group_map(env_spec), model_cfg)()
    policy = algo.get_policy_for_collection().to("cpu")
    policy.eval()
    ck = torch.load(CKPTS["v36"], map_location="cpu", mmap=True, weights_only=False)
    pfx, h, m, amb = install(policy, extract_actor(ck, "agents"))
    print(f"[build] v36 单组 policy: prefix={pfx} hit={h} miss={m} amb={amb}", flush=True)
    return policy, model_cfg


def build_duo(task, cfg, n_envs):
    g = lambda: (lambda x: (setattr(x, "role_ids", []), setattr(x, "n_roles", 0), x)[-1])(GruConfig.get_from_yaml())
    att = SequenceModelConfig([AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_attacker.yaml"), g()], intermediate_sizes=[256])
    dfn = SequenceModelConfig([AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_defender.yaml"), g()], intermediate_sizes=[256])
    model_cfg = EnsembleModelConfig({"attacker": att, "defender": dfn})
    critic_cfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    a_cfg = MappoConfig.get_from_yaml(); a_cfg.share_param_actor = False; a_cfg.share_param_critic = False
    d_cfg = MappoConfig.get_from_yaml(); d_cfg.share_param_actor = True;  d_cfg.share_param_critic = True
    algo_cfg = EnsembleAlgorithmConfig({"attacker": a_cfg, "defender": d_cfg})

    env_spec = VmasEnvWithState(scenario="layup", num_envs=n_envs, continuous_actions=True, seed=0,
                                device="cpu", clamp_actions=True, group_map=GROUP_DUO, **dict(task.config))
    env_spec = TransformedEnv(env_spec, Compose(*task.get_env_transforms(env_spec)).clone())
    shell = shell_for(env_spec, algo_cfg, model_cfg, critic_cfg, cfg, task)
    algo = algo_cfg.get_algorithm(shell)
    env = _add_rnn_transforms(lambda: env_spec, task.group_map(env_spec), model_cfg)()
    policy = algo.get_policy_for_collection().to("cpu")
    policy.eval()
    cks = {n: torch.load(p, map_location="cpu", mmap=True, weights_only=False) for n, p in CKPTS.items() if n in DUO_SIDES}
    for side, ck in cks.items():
        pfx0, h0, m0, a0 = install(policy, extract_actor(ck, "attacker"), "module.0.")
        pfx1, h1, m1, a1 = install(policy, extract_actor(ck, "defender"), "module.1.")
        print(f"[build] duo {side} 权重就绪: atk({h0}/{m0}/{a0}) def({h1}/{m1}/{a1})", flush=True)
    return policy, env, cks


class Driver:
    """env.rollout 的 policy：把两边的动作拼成一个 4-agent 动作。"""

    def __init__(self, p_single, p_duo, td_duo, a_side, d_side, dbg=False):
        self.p_single, self.p_duo, self.td_duo = p_single, p_duo, td_duo
        self.a_side, self.d_side = a_side, d_side
        self.dbg = dbg
        self._n = 0

    def __call__(self, td):
        obs = td.get(("agents", "observation"))
        is_init = td.get("is_init")
        out_d = None
        if self.a_side != "v36" or self.d_side != "v36":
            tmpl = self.td_duo
            tmpl.set(("attacker", "observation"), obs[:, :2])
            tmpl.set(("defender", "observation"), obs[:, 2:])
            tmpl.set("is_init", is_init)
            # 新开局的行：GRU 状态清零（模仿 env primer）
            ii = is_init.bool() if is_init.dim() == 1 else is_init.bool().reshape(is_init.shape[0], -1)[:, 0]
            if ii.any():
                for k in (("attacker", "_hidden_gru_1"), ("defender", "_hidden_gru_1")):
                    s = tmpl.get(k)
                    if s is not None:
                        s[ii] = 0
            out_d = self.p_duo(tmpl)
            # gru.py:627 把新状态写到 ("next", ...)，搬回模板供下一步使用
            for k in (("attacker", "_hidden_gru_1"), ("defender", "_hidden_gru_1")):
                ns = out_d.get(("next", *k), None)
                if ns is not None:
                    tmpl.set(k, ns.detach())
            if self.dbg and self._n == 1:
                d = tmpl.get(("attacker", "_hidden_gru_1")).abs().max().item()
                print(f"    [state] duo 第二步状态 |h|max = {d:.6f}", flush=True)
        out_s = None
        if self.a_side == "v36" or self.d_side == "v36":
            out_s = self.p_single(td)
        act_a = out_s.get(("agents", "action"))[:, :2] if self.a_side == "v36" else out_d.get(("attacker", "action"))
        act_d = out_s.get(("agents", "action"))[:, 2:] if self.d_side == "v36" else out_d.get(("defender", "action"))
        td.set(("agents", "action"), torch.cat([act_a, act_d], dim=-2))
        if self._n == 0:
            self.a0_fp = (float(act_a.mean()), float(act_a.std()), float(act_d.mean()), float(act_d.std()))
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
                done = roll.get(("next", "done"))                                  # [E,T,1]（实测本环境 rollout 为 env 在前、时间在后）
                tr = roll.get(("next", "agents", "info", "termination_reason"))    # [E,T,4,1]
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
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/tmp/xp2_result.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    task = LayupTask.LAYUP.get_from_yaml()
    cfg = ExperimentConfig.get_from_yaml()
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    max_steps = task.max_steps(None)

    n = args.envs
    print(f"[cfg] mode={args.mode} envs={n} rounds={args.rounds} max_steps={max_steps} threshold=0.2", flush=True)

    p_single, m36 = build_single(task, cfg, n)
    p_duo, env_duo, cks = build_duo(task, cfg, n)

    env_base = task.get_env_fun(num_envs=n, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    env_base = TransformedEnv(env_base, Compose(*task.get_env_transforms(env_base)).clone())
    env = _add_rnn_transforms(lambda: env_base, task.group_map(env_base), m36)()
    td_duo = env_duo.reset()

    cells = [("v36", "v36"), ("v35", "v35"), ("v36", "v35"), ("v35", "v36")] if args.smoke \
        else [(a, d) for a in ["v36"] + DUO_SIDES for d in ["v36"] + DUO_SIDES]

    out = {"mode": args.mode, "envs": n, "rounds": args.rounds, "cells": {}}
    for a_side, d_side in cells:
        # 装载 duo 权重（v36 侧的对侧用 v35 填充，动作会被丢弃）
        a_ck = cks.get(a_side) or cks["v35"]
        d_ck = cks.get(d_side) or cks["v35"]
        install(p_duo, extract_actor(a_ck, "attacker"), "module.0.")
        install(p_duo, extract_actor(d_ck, "defender"), "module.1.")
        driver = Driver(p_single, p_duo, td_duo.clone(), a_side, d_side, dbg=args.smoke)
        stat = run_rollouts(env, driver, max_steps, args.rounds, args.mode)
        key = f"{a_side}_atk_vs_{d_side}_def"
        out["cells"][key] = stat
        wr = 100.0 * stat["win"] / max(1, stat["episodes"])
        sr = 100.0 * stat["shot"] / max(1, stat["episodes"])
        top = sorted(stat["reasons"].items(), key=lambda kv: -kv[1])[:4]
        top_s = " ".join(f"{REASON_NAMES.get(c, c)}:{v}" for c, v in top)
        print(f"[cell] {key:22s} win={wr:5.1f}% 码1={sr:4.1f}% eps={stat['episodes']} "
              f"({stat['seconds']}s) fp(a={driver.a0_fp[0]:+.2f}/{driver.a0_fp[1]:.2f} d={driver.a0_fp[2]:+.2f}/{driver.a0_fp[3]:.2f}) | {top_s}", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
