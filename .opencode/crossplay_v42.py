"""[交叉对打 v42] 同架构两代对打：iter300（v41 末，奖励改动前） vs iter450（v42 末，改动后）。

用法（容器内）：
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/crossplay_v42.py --mode det    --envs 192 --rounds 3 --out /tmp/xp42_det.json
  PYTHONPATH=/home/vscode/workspace/BenchMARL python /tmp/crossplay_v42.py --mode random --envs 192 --rounds 3 --out /tmp/xp42_random.json

设计：
- 单组共享 actor（role_ids [0,1,2,2]）⇒ 无法只换"攻方"或"守方"的权重；
  本脚本让两份 policy 各自前向一次（各带自己的 GRU 隐状态模板），再按 agent 行拼接动作：
  攻方行 0:2 取 a_side 的输出，守方行 2:4 取 d_side 的输出。
- 环境统一为新规则（复合动作 {continuous[4,2], discrete[4,1]} + 按键 mask），两边同 spec。
- cells：A@A / A@B / B@A / B@B，其中 X@Y = 攻方用 X、守方用 Y。
"""
import argparse
import json
import os
import time

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch
from types import SimpleNamespace
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs.utils import ExplorationType, set_exploration_type

set_composite_lp_aggregate(False).set()

from benchmarl.environments import LayupTask
from benchmarl.algorithms import MappoConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.common import SequenceModelConfig
from benchmarl.experiment import ExperimentConfig
from benchmarl.utils import _add_rnn_transforms

BASE = "/home/vscode/workspace/BenchMARL/outputs"
CKPTS = {
    # 上一轮（v41 末 = iter300，A2 新奖励 + 被盖拆分之前）
    "A300": f"{BASE}/2026-10-03_16-47-10/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/checkpoints/checkpoint_90000000.pt",
    # 最新一轮（v42 末 = iter450，A2 新奖励 + 被盖拆分之后）
    "B450": f"{BASE}/2026-10-04_03-32-31/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/checkpoints/checkpoint_135000000.pt",
}
WIN_CODES = {1, 2, 3, 4, 5}
REASON_NAMES = {1: "shot", 2: "opp_foul", 3: "opp_wall", 4: "opp_cross", 5: "opp_ff",
                11: "blocked", 12: "timeout", 13: "own_foul", 14: "own_wall", 15: "own_ff"}


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
        if len(cands) == 1 and tuple(psd[cands[0]].shape) == tuple(v.shape):
            chosen[cands[0]] = v
            hit += 1
        elif not cands:
            miss += 1
        else:
            amb += 1
    full = dict(psd)
    full.update(chosen)
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


def build_single(task, cfg, n_envs, ckpt, label):
    """单组共享 actor（新规则：复合动作 + 按键 mask）。返回 policy；env 在外部另建。"""
    attn = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml")
    model_cfg = SequenceModelConfig([attn, GruConfig.get_from_yaml()], intermediate_sizes=[256])
    critic_cfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    a_cfg = MappoConfig.get_from_yaml()
    a_cfg.share_param_actor = True
    a_cfg.share_param_critic = False
    a_cfg.loc_bound = 3.0  # 保真：v41/v42 训练时 loc_bound=3.0

    env_spec = task.get_env_fun(num_envs=n_envs, continuous_actions=True, seed=0,
                                device=torch.device("cpu"))()
    shell = shell_for(env_spec, a_cfg, model_cfg, critic_cfg, cfg, task)
    algo = a_cfg.get_algorithm(shell)
    policy = algo.get_policy_for_collection().to("cpu")
    policy.eval()
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    pfx, h, m, amb = install(policy, extract_actor(ck, "agents"))
    print(f"[build] {label} prefix={pfx} hit={h} miss={m} amb={amb}  <- {os.path.basename(ckpt)}", flush=True)
    return policy


class Driver2:
    """两份 policy 各前向一次（各带自己的 GRU 隐状态模板），按 agent 行拼接动作。"""

    def __init__(self, pA, pB, tmplB, a_side, d_side, dbg=False):
        self.pA, self.pB, self.tmplB = pA, pB, tmplB
        self.a_side, self.d_side = a_side, d_side
        self.dbg = dbg
        self._n = 0
        self.a0_fp = None
        self.n_press = 0.0
        self.n_step = 0

    def __call__(self, td):
        obs = td.get(("agents", "observation"))
        is_init = td.get("is_init")

        # --- B 侧：自己的模板，隐状态每一步都推进（看的是同一份真实观测流） ---
        tmpl = self.tmplB
        tmpl.set(("agents", "observation"), obs)
        tmpl.set("is_init", is_init)
        # 关键：mask 必须每步同步（A1 只有"在圈内"才允许按键）。
        # 不回填的话 tmpl 里是 reset 时的旧 mask（A1 不在圈内 ⇒ 恒 False），
        # B 侧 A1 永远按不下投篮键 ⇒ B 的码1 会被错算成 0。
        _mask = td.get(("agents", "action_mask"), None)
        if _mask is not None:
            tmpl.set(("agents", "action_mask"), _mask)
        st = tmpl.get(("agents", "_hidden_gru_1"), None)
        if st is not None and is_init is not None:
            ii = is_init.reshape(is_init.shape[0], -1)[:, 0].bool()
            if ii.any():
                st[ii] = 0
        outB = self.pB(tmpl)
        nsB = outB.get(("next", "agents", "_hidden_gru_1"), None)
        if nsB is not None and st is not None:
            st.copy_(nsB.detach())

        # --- A 侧：直接跑在 rollout 的 td 上（顺带写 env 侧 RNN 状态） ---
        self.pA(td)
        act = td.get(("agents", "action"))
        cont = act.get("continuous")
        disc = act.get("discrete")
        aB = outB.get(("agents", "action"))

        if self.a_side == "B":
            cont[:, :2] = aB.get("continuous")[:, :2]
            disc[:, :2] = aB.get("discrete")[:, :2]
        if self.d_side == "B":
            cont[:, 2:] = aB.get("continuous")[:, 2:]
            disc[:, 2:] = aB.get("discrete")[:, 2:]

        if self._n == 0:
            self.a0_fp = (round(float(cont.mean()), 3), round(float(cont.std()), 3),
                          round(float(disc.float().mean()), 3))
            if self.dbg:
                print(f"    [fp] cont mean/std={self.a0_fp[0]}/{self.a0_fp[1]} disc 按下均值={self.a0_fp[2]}", flush=True)
        # 统计 A1 按键率（诊断用；fingerprint 只测了第 0 步，无法反映全程）
        self.n_press += float(disc[:, 0].float().sum())
        self.n_step += int(disc.shape[0])
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
                                   auto_cast_to_device=False, break_when_any_done=False)
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
    ap.add_argument("--out", default="/tmp/xp42_result.json")
    ap.add_argument("--ckpt-a", default=None, help="A 侧（上一轮）checkpoint 覆盖")
    ap.add_argument("--ckpt-b", default=None, help="B 侧（最新一轮）checkpoint 覆盖")
    ap.add_argument("--cells", default="", help="只跑指定 cell（逗号分隔，如 A@B,B@A）")
    args = ap.parse_args()

    torch.manual_seed(0)
    task = LayupTask.LAYUP.get_from_yaml()
    cfg = ExperimentConfig.get_from_yaml()
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    max_steps = task.max_steps(None)

    n = 16 if args.smoke else args.envs
    rounds = 1 if args.smoke else args.rounds
    ckA = args.ckpt_a or CKPTS["A300"]
    ckB = args.ckpt_b or CKPTS["B450"]
    print(f"[cfg] mode={args.mode} envs={n} rounds={rounds} max_steps={max_steps} threshold=0.2", flush=True)
    print(f"[cfg] A(上一轮)={ckA}", flush=True)
    print(f"[cfg] B(最新)  ={ckB}", flush=True)

    pA = build_single(task, cfg, n, ckA, "A(v41末/iter300)")
    pB = build_single(task, cfg, n, ckB, "B(v42末/iter450)")

    # 环境：与训练同款（同 spec + RNN transforms）
    env_spec = task.get_env_fun(num_envs=n, continuous_actions=True, seed=0, device=torch.device("cpu"))()
    env_model_cfg = SequenceModelConfig([AttentionConfig.get_from_yaml(
        "benchmarl/conf/model/layers/attention_agents.yaml"), GruConfig.get_from_yaml()],
        intermediate_sizes=[256])
    env = _add_rnn_transforms(lambda: env_spec, task.group_map(env_spec), env_model_cfg)()
    tmplB = env.reset()
    if args.smoke:
        print(f"[tmpl] keys={list(tmplB.keys(True))}", flush=True)

    cells = [("A", "A"), ("B", "B"), ("A", "B"), ("B", "A")]
    if args.cells:
        want = set(args.cells.split(","))
        cells = [c for c in cells if f"{c[0]}@{c[1]}" in want]
    out = {"mode": args.mode, "envs": n, "rounds": rounds, "ckpt_a": ckA, "ckpt_b": ckB, "cells": {}}
    for a_side, d_side in cells:
        env.reset()
        driver = Driver2(pA, pB, tmplB, a_side, d_side, dbg=args.smoke)
        stat = run_rollouts(env, driver, max_steps, rounds, args.mode)
        key = f"{a_side}@{d_side}"
        out["cells"][key] = stat
        wr = 100.0 * stat["win"] / max(1, stat["episodes"])
        sr = 100.0 * stat["shot"] / max(1, stat["episodes"])
        top = sorted(stat["reasons"].items(), key=lambda kv: -kv[1])[:5]
        top_s = " ".join(f"{REASON_NAMES.get(c, c)}:{v}" for c, v in top)
        pr = 100.0 * driver.n_press / max(1, driver.n_step)
        print(f"[cell] {key} (攻={a_side} 守={d_side}) win={wr:5.1f}% 码1={sr:4.1f}% eps={stat['episodes']} "
              f"({stat['seconds']}s) fp={driver.a0_fp} A1按键率={pr:.2f}% | {top_s}", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
