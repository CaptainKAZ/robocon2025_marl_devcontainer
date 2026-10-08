"""[P0 单组化] 角色/观测/动作顺序探针：用真实 collector 前向确认 agent 顺序、role 映射、
动作幅度分布，以及一个 batch 内各 agent 的终局构成。

用法（容器内，cwd=/home/vscode/workspace/BenchMARL）:
    python /tmp/p0_probe_roles.py
"""

import os

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = int(os.environ.get("PROBE_FRAMES", "1500"))
    cfg.on_policy_n_envs_per_worker = int(os.environ.get("PROBE_ENVS", "10"))
    cfg.on_policy_minibatch_size = 5000
    cfg.max_n_iters = 1
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/p0_probe"
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"

    task = LayupTask.LAYUP.get_from_yaml()
    algo = MappoConfig.get_from_yaml()
    algo.share_param_actor = True
    algo.share_param_critic = False
    mc = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml(
                "benchmarl/conf/model/layers/attention_agents.yaml"
            ),
            GruConfig.get_from_yaml(),
        ],
        intermediate_sizes=[256],
    )
    cc = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

    exp = Experiment(task=task, algorithm_config=algo, model_config=mc,
                     critic_model_config=cc, seed=114514, config=cfg, callbacks=[])
    env = exp.test_env
    print("[PROBE] env.agent_names =", getattr(env, "agent_names", None))
    print("[PROBE] env.group_map =", getattr(env, "group_map", None))

    # reset 后的初始位置（确认 agent 顺序）
    td = env.reset()
    obs = td.get(("agents", "observation"))
    print("[PROBE] reset obs shape:", tuple(obs.shape))
    for i in range(obs.shape[-2]):
        p = obs[0, i, 0:2] * torch.tensor([4.0, 7.5])
        print(f"   agent{i}: pos=({p[0]:+.2f},{p[1]:+.2f})")

    # 用真实 collector 采一个 batch（随机初始化策略的行为）
    batch = next(iter(exp.collector))
    loc = batch.get(("agents", "loc"))
    scale = batch.get(("agents", "scale"))
    act = batch.get(("agents", "action"))
    print("[PROBE] loc shape", tuple(loc.shape), "scale", tuple(scale.shape))
    print("[PROBE] |loc| per agent:", [f"{v:.3f}" for v in loc.abs().mean(dim=(0, 1, 3))])
    print("[PROBE] |action| per agent:", [f"{v:.3f}" for v in act.abs().mean(dim=(0, 1, 3))])
    print("[PROBE] scale per agent:", [f"{v:.3f}" for v in scale.mean(dim=(0, 1, 3))])

    # 终局构成：termination_reason 形状与各 code 计数
    done = batch.get(("next", "done"))
    info = batch.get(("next", "agents", "info", "termination_reason"))
    print("[PROBE] done shape", tuple(done.shape), "info shape", tuple(info.shape))
    done_flat = done.reshape(-1).bool()
    # 逐 agent 的 code（若 info 是 [T,B,4,1]，取每步第 i 个 agent）
    if info.dim() == 4:
        codes_all = info.reshape(-1, info.shape[-2])
        print("[PROBE] per-agent code count (index=agent):")
        for i in range(codes_all.shape[-1]):
            vals, counts = torch.unique(codes_all[done_flat, i], return_counts=True)
            top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: -kv[1])[:6]
            print(f"   agent{i}: {top}")
    vals, counts = torch.unique(info[..., 0].reshape(-1)[done_flat], return_counts=True)
    top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: -kv[1])[:8]
    print("[PROBE] agent0(取[...,0,:]) code count:", top)

    # done 时刻各 agent 的 y 坐标（判断谁在越线/撞墙）
    nxt_obs = batch.get(("next", "agents", "observation"))
    posy = nxt_obs[..., 1] * 7.5
    print("[PROBE] done 时刻 y 均值（每 agent）:",
          [f"{v:.2f}" for v in posy.reshape(-1, posy.shape[-2])[done_flat].mean(dim=0)])
    exp.close()
    print("[PROBE] DONE")


if __name__ == "__main__":
    main()
