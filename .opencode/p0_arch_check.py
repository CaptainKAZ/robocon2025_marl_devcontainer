"""[P0-B] 新架构结构核对 + 1 iter 数据流冒烟（CPU）。

用法（容器内，cwd=/home/vscode/workspace/BenchMARL）:
    python /tmp/p0_arch_check.py
"""

import os

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch
from torch import nn

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig, RoleConditionedMLP
from benchmarl.models.gru import GruConfig
from clear_restore import WinRateReportSimple


def summarize(root, label):
    print(f"[ARCH] ===== {label} =====")
    n_params = sum(p.numel() for p in root.parameters())
    print(f"[ARCH]   参数总量: {n_params/1e6:.3f}M")
    mlps = [m for m in root.modules() if isinstance(m, RoleConditionedMLP)]
    films = [m for m in root.modules() if m.__class__.__name__ == "RoleFiLM"]
    print(f"[ARCH]   RoleConditionedMLP x{len(mlps)} | RoleFiLM x{len(films)}")
    for i, m in enumerate(mlps):
        head0 = m.heads[0]
        layers = sum(1 for _ in head0.modules() if isinstance(_, nn.Linear))
        print(f"[ARCH]     [{i}] trunk_in={m.trunk[0].in_features} heads={len(m.heads)} head_linear_layers={layers}")
    for m in root.modules():
        if m.__class__.__name__ == "Attention":
            first_layer = m.attention_layers[0]
            shared_attn = not isinstance(first_layer, nn.ModuleList)
            enc_shared = all(
                not isinstance(v, nn.ModuleList) for v in m.encoders.values()
            )
            if getattr(m, "use_role_encoders", False):
                enc_mode = "per_role"
            elif enc_shared:
                enc_mode = "shared"
            else:
                enc_mode = "per_agent"
            print(
                f"[ARCH]     Attention: share_params={m.share_params} "
                f"spec_share={getattr(m,'_spec_share_params',None)} out_has_agent={m.output_has_agent_dim} "
                f"in_has_agent={m.input_has_agent_dim} role_enabled={getattr(m,'_role_enabled',None)} "
                f"enc_mode={enc_mode} attn_shared={shared_attn} n_role_heads={getattr(m,'n_roles',0)}"
            )
            print(f"[ARCH]       encoders: { {k: type(v).__name__ for k, v in m.encoders.items()} }")


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = 1500
    cfg.on_policy_n_envs_per_worker = 10
    cfg.on_policy_minibatch_size = 500
    cfg.max_n_iters = 1
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/p0_arch_check"
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"

    task = LayupTask.LAYUP.get_from_yaml()
    algo = MappoConfig.get_from_yaml()
    algo.share_param_actor = True
    algo.share_param_critic = False

    model_config = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml(
                os.environ.get(
                    "ARCH_AGENTS_YAML",
                    "benchmarl/conf/model/layers/attention_agents.yaml",
                )
            ),
            GruConfig.get_from_yaml(),
        ],
        intermediate_sizes=[256],
    )
    critic_config = AttentionConfig.get_from_yaml(
        "benchmarl/conf/model/layers/attention_critic.yaml"
    )

    exp = Experiment(
        task=task,
        algorithm_config=algo,
        model_config=model_config,
        critic_model_config=critic_config,
        seed=0,
        config=cfg,
        callbacks=[WinRateReportSimple()],
    )

    loss = exp.losses["agents"]
    print("[ARCH] loss children:", [n for n, _ in loss.named_children()])
    summarize(loss.actor_network, "actor_network")
    for name in ("critic_network", "value_network", "critic"):
        mod = getattr(loss, name, None)
        if mod is not None:
            summarize(mod, name)
            break

    try:
        exp.run()
    finally:
        for group, buf in exp.replay_buffers.items():
            s = buf.sample()
            obs = s.get((group, "observation"))
            act = s.get((group, "action"))
            adv = s.get((group, "advantage"))
            vt = s.get((group, "value_target"))
            sv = s.get((group, "state_value"), None)
            if sv is None:
                sv = s.get(("next", group, "state_value"), None)
            print(f"[ARCH] group={group} obs={tuple(obs.shape)} act={tuple(act.shape)} "
                  f"adv={tuple(adv.shape)} vt={tuple(vt.shape)} state_value={None if sv is None else tuple(sv.shape)}")
            print(f"[ARCH] adv finite={bool(torch.isfinite(adv).all())} "
                  f"vt finite={bool(torch.isfinite(vt).all())}")
        exp.close()
        print("[ARCH] DONE")


if __name__ == "__main__":
    main()
