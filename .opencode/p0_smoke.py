"""[P0 单组化] 小规模数据流冒烟：验证单组 4 agent + 共享主干 + 角色头 全链路跑通。

用法（容器内，cwd=/home/vscode/workspace/BenchMARL）:
    python /tmp/p0_smoke.py
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
from clear_restore import WinRateReportSimple


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = 1500
    cfg.on_policy_n_envs_per_worker = 10
    cfg.on_policy_minibatch_size = 500
    cfg.max_n_iters = 3
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/p0_smoke"
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"

    task = LayupTask.LAYUP.get_from_yaml()

    algo = MappoConfig.get_from_yaml()
    algo.share_param_actor = True
    algo.share_param_critic = False

    model_config = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml(
                "benchmarl/conf/model/layers/attention_agents.yaml"
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
    print("[SMOKE] train_group_map:", list(exp.train_group_map.keys()))
    print("[SMOKE] losses:", list(exp.losses.keys()))
    try:
        exp.run()
    finally:
        # 抽样检查 buffer 内容
        for group, buf in exp.replay_buffers.items():
            s = buf.sample()
            obs = s.get((group, "observation"))
            act = s.get((group, "action"))
            adv = s.get((group, "advantage"))
            vt = s.get((group, "value_target"))
            print(f"[SMOKE] group={group} obs={tuple(obs.shape)} act={tuple(act.shape)}")
            print(
                f"[SMOKE] adv finite={bool(torch.isfinite(adv).all())} "
                f"mean={adv.float().mean():.4f} | vt finite={bool(torch.isfinite(vt).all())} "
                f"mean={vt.float().mean():.4f}"
            )
            print(
                f"[SMOKE] done_sum={int(s.get(('next', 'done')).sum())} "
                f"n_agents={obs.shape[-3]}"
            )
        exp.close()
    print("[SMOKE] P0 SMOKE DONE")


if __name__ == "__main__":
    main()
