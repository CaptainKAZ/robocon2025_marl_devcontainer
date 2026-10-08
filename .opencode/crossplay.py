"""Cross-play: 单进程内换权重做多代交叉对打.

用法:
  python crossplay.py --probe          # 只探测 policy/ckpt 键映射
  python crossplay.py --mode random    # 随机采样口径（训练口径）
  python crossplay.py --mode det       # 确定性口径（评测口径）
"""
import argparse
import json
import os
import time
from types import SimpleNamespace
from pathlib import Path

import torch

from benchmarl.algorithms import MappoConfig, EnsembleAlgorithmConfig
from benchmarl.models import EnsembleModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.utils import _add_rnn_transforms

from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type

BASE = "/home/vscode/workspace/BenchMARL/outputs"
RUN = "ensemblealgorithm_layup_ensemblemodel__9d8e5c85_26_09_30-07_04_09"
CKPTS = {
    "v28": f"{BASE}/2026-09-30_07-04-09/{RUN}/checkpoints/checkpoint_33750000.pt",
    "v29": f"{BASE}/2026-09-30_15-32-23/{RUN}/checkpoints/checkpoint_49500000.pt",
    "v32": f"{BASE}/2026-10-01_12-42-22/{RUN}/checkpoints/checkpoint_56250000.pt",
    "v34": f"{BASE}/2026-10-02_08-47-52/{RUN}/checkpoints/checkpoint_86250000.pt",
    "v35": f"{BASE}/2026-10-02_10-25-09/{RUN}/checkpoints/checkpoint_131250000.pt",
}
WIN_CODES = {1, 2, 3, 4, 5}
REASON_NAMES = {1: "shot", 2: "opp_foul", 3: "opp_wall", 4: "opp_cross", 5: "opp_ff",
                11: "blocked", 12: "timeout", 13: "own_foul", 14: "own_wall", 15: "own_ff"}


def build(mode, num_envs, seed=0, device="cpu"):
    # 交叉对打统一用目标难度（与 cont 训练的评测口径一致），避免阈值课程给前面的对局送温暖
    os.environ["VMAS_INITIAL_SHOT_THRESHOLD"] = "0.2"
    task = LayupTask.LAYUP.get_from_yaml()
    cfg = ExperimentConfig.get_from_yaml()

    a_cfg = MappoConfig.get_from_yaml(); a_cfg.share_param_actor = False; a_cfg.share_param_critic = False
    d_cfg = MappoConfig.get_from_yaml(); d_cfg.share_param_actor = True;  d_cfg.share_param_critic = True
    algo_cfg = EnsembleAlgorithmConfig({"attacker": a_cfg, "defender": d_cfg})

    att = SequenceModelConfig(
        [AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_attacker.yaml"),
         GruConfig.get_from_yaml()], intermediate_sizes=[256])
    dfn = SequenceModelConfig(
        [AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_defender.yaml"),
         GruConfig.get_from_yaml()], intermediate_sizes=[256])
    model_cfg = EnsembleModelConfig({"attacker": att, "defender": dfn})
    critic_cfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

    continuous_actions = bool(cfg.prefer_continuous_actions)
    env = task.get_env_fun(num_envs=num_envs, continuous_actions=continuous_actions,
                           seed=seed, device=torch.device(device))()
    env = TransformedEnv(env, Compose(*task.get_env_transforms(env)).clone()).to(device)

    shell = SimpleNamespace(
        config=cfg, algorithm_config=algo_cfg, model_config=model_cfg,
        critic_model_config=critic_cfg, task=task, group_map=task.group_map(env),
        continuous_actions=continuous_actions, seed=seed, on_policy=algo_cfg.on_policy(),
        observation_spec=task.observation_spec(env), action_spec=task.action_spec(env),
        info_spec=task.info_spec(env), state_spec=task.state_spec(env),
        action_mask_spec=task.action_mask_spec(env),
    )
    algorithm = algo_cfg.get_algorithm(shell)
    for a in algorithm.algorithms_map.values():
        a.device = torch.device(device)

    group_map = task.group_map(env)
    if model_cfg.is_rnn:
        env = _add_rnn_transforms(lambda: env, group_map, model_cfg)()
    env = algorithm.process_env_fun(lambda: env)()
    policy = algorithm.get_policy_for_collection().to(device)
    policy.eval()
    return env, policy, task


def strip_orig(k):
    return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")


def extract_actor(ck, group):
    pfx = "actor_network_params."
    sd = {}
    for k, v in ck[f"loss_{group}"].items():
        if isinstance(v, torch.Tensor) and k.startswith(pfx):
            sd[strip_orig(k[len(pfx):])] = v
    return sd


def _norm(k):
    return k.replace("._orig_mod.", ".").replace("_orig_mod.", "")


def _detect_prefix(policy, actor_sd):
    """探测该组 actor 对应 policy 的哪棵子树（'module.0.' / 'module.1.'）"""
    psd = policy.state_dict()
    nkeys = list(psd.keys())
    best, best_hit = None, -1
    for pfx in ("module.0.", "module.1."):
        hits = 0
        for k, v in actor_sd.items():
            nk = _norm(k)
            c = [pk for pk in nkeys if pk.startswith(pfx) and _norm(pk).endswith(nk)]
            if len(c) == 1 and tuple(psd[c[0]].shape) == tuple(v.shape):
                hits += 1
        if hits > best_hit:
            best, best_hit = pfx, hits
    return best, best_hit


def install(policy, actor_sd, prefix=None):
    """把某组 actor 权重写进 policy。返回 (前缀, 命中数, 未命中数, 歧义数)."""
    psd = policy.state_dict()
    nkeys = list(psd.keys())
    if prefix is None:
        prefix, _ = _detect_prefix(policy, actor_sd)
    chosen = {}
    hit = miss = amb = 0
    for k, v in actor_sd.items():
        nk = _norm(k)
        cands = [pk for pk in nkeys if pk.startswith(prefix) and _norm(pk).endswith(nk)]
        if len(cands) == 1:
            pk = cands[0]
            if tuple(psd[pk].shape) == tuple(v.shape):
                chosen[pk] = v
                hit += 1
            else:
                amb += 1
        elif not cands:
            miss += 1
        else:
            amb += 1
    full = dict(psd)
    full.update(chosen)
    policy.load_state_dict(full, strict=True)
    return prefix, hit, miss, amb


def run_rollouts(env, policy, task, max_steps, n_rounds, mode):
    et = ExplorationType.DETERMINISTIC if mode == "det" else ExplorationType.RANDOM
    reasons = {}
    n_eps = 0
    n_win = 0
    n_shot = 0
    t0 = time.time()
    with torch.no_grad():
        with set_exploration_type(et):
            for _ in range(n_rounds):
                roll = env.rollout(max_steps=max_steps, policy=policy,
                                   auto_cast_to_device=True, break_when_any_done=False)
                done = roll.get(("next", "done")).reshape(roll.batch_size[0], -1)  # [E,T]
                tr = roll.get(("next", "attacker", "info", "termination_reason"))
                tr = tr.reshape(roll.batch_size[0], roll.numel() // roll.batch_size[0], -1)
                for e in range(done.shape[0]):
                    idx = done[e].nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        code = 0
                    else:
                        t = int(idx[0])
                        row = tr[e, t].flatten()
                        nz = row[row > 0]
                        code = int(nz[0]) if nz.numel() else 0
                    reasons[code] = reasons.get(code, 0) + 1
                    n_eps += 1
                    if code in WIN_CODES:
                        n_win += 1
                    if code == 1:
                        n_shot += 1
    dt = time.time() - t0
    return dict(episodes=n_eps, win=n_win, shot=n_shot, reasons=reasons, seconds=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--mode", default="random", choices=["random", "det"])
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="/tmp/crossplay_result.json")
    args = ap.parse_args()

    env, policy, task = build(args.mode, args.envs)
    max_steps = task.max_steps(env)

    if args.probe:
        psd = policy.state_dict()
        print(f"policy state_dict 键数: {len(psd)}")
        keys = list(psd.keys())
        for k in keys[:40]:
            print("  ", tuple(psd[k].shape), k)
        print("  ...")
        ck = torch.load(CKPTS["v35"], map_location="cpu", mmap=True, weights_only=False)
        for grp in ("attacker", "defender"):
            sd = extract_actor(ck, grp)
            print(f"\n{grp}: actor 张量数 {len(sd)}；键样例: {list(sd.keys())[:3]}")
            pfx, hit, miss, amb = install(policy, sd)
            print(f"  install -> prefix={pfx} hit={hit} miss={miss} amb={amb}")
        return

    models = {}
    for name, path in CKPTS.items():
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        models[name] = {"attacker": extract_actor(ck, "attacker"),
                        "defender": extract_actor(ck, "defender")}
        del ck
        print(f"[load] {name}: att={len(models[name]['attacker'])} def={len(models[name]['defender'])}", flush=True)

    names = list(CKPTS.keys())
    if args.quick:
        names = ["v34", "v35"]
    rounds = 2 if args.quick else args.rounds
    att_pfx, a_hits = _detect_prefix(policy, models["v35"]["attacker"])
    def_pfx, d_hits = _detect_prefix(policy, models["v35"]["defender"])
    print(f"[prefix] attacker={att_pfx}(hits={a_hits}) defender={def_pfx}(hits={d_hits})", flush=True)
    result = {"mode": args.mode, "envs": args.envs, "rounds": rounds, "matrix": {}}
    t_start = time.time()
    for att_name in names:
        for def_name in names:
            t0 = time.time()
            _, h1, m1, a1 = install(policy, models[att_name]["attacker"], prefix=att_pfx)
            _, h2, m2, a2 = install(policy, models[def_name]["defender"], prefix=def_pfx)
            policy.eval()
            stat = run_rollouts(env, policy, task, max_steps, rounds, args.mode)
            wpct = 100.0 * stat["win"] / max(1, stat["episodes"])
            spct = 100.0 * stat["shot"] / max(1, stat["episodes"])
            result["matrix"][f"{att_name}@{def_name}"] = stat
            top = sorted(stat["reasons"].items(), key=lambda x: -x[1])[:4]
            top_s = " ".join(f"{REASON_NAMES.get(c, c)}:{v}" for c, v in top)
            print(f"[{args.mode}] att={att_name} def={def_name} | n={stat['episodes']} "
                  f"win={wpct:5.1f}% shot={spct:5.1f}% | {top_s} | install h/m/a={h1}/{m1}/{a1},{h2}/{m2}/{a2} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    result["total_seconds"] = time.time() - t_start
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"\nDONE {result['total_seconds']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
