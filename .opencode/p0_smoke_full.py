"""[P0 单组化] 真实规模短跑冒烟：验证 2000 envs/300k frames 全链路 + eval worker + 视频。

用法（容器内，cwd=/home/vscode/workspace/BenchMARL）:
    OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 python /tmp/p0_smoke_full.py
"""

import os

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from clear_restore import WinRateReportSimple


def main():
    cfg = ExperimentConfig.get_from_yaml()  # 真实规模（2000 envs / 300k frames）
    cfg.max_n_iters = 3
    # eval worker 冒烟：首个 batch 后立刻评测，4 集、开渲染（验证视频链路）
    cfg.evaluation = True
    cfg.evaluation_interval = cfg.on_policy_collected_frames_per_batch
    cfg.evaluation_episodes = 4
    cfg.evaluation_deterministic_actions = True
    cfg.render = True
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/p0_smoke_full"

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
        seed=114514,
        config=cfg,
        callbacks=[WinRateReportSimple()],
    )
    print(f"[FULL] envs={cfg.on_policy_n_envs_per_worker} frames/batch="
          f"{cfg.on_policy_collected_frames_per_batch} minibatch={cfg.on_policy_minibatch_size}")
    print("[FULL] OVERLAP_COLLECTION =", os.environ.get("OVERLAP_COLLECTION", "0"))
    exp.run()
    exp.close()
    print("[FULL] P0 FULL SMOKE DONE")


if __name__ == "__main__":
    main()
