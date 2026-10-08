"""加速验证单臂：冻结权重（lr=0）从 v39 iter200 续跑 ARM_ITERS 轮，只测采集耗时。

用法（容器内）：
  ARM_ITERS=3 OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 python /tmp/speed_arm.py
"""

import os

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig

CKPT = (
    "outputs/2026-10-03_10-33-56/mappo_layup_sequencemodel__a35f3c2b_26_10_03-05_39_14/"
    "checkpoints/checkpoint_60000000.pt"
)


def main():
    iters = int(os.environ.get("ARM_ITERS", "3"))
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_iters = 200 + iters          # 从 iter200 起再跑 iters 轮
    cfg.restore_file = os.path.abspath(CKPT)
    cfg.lr = 0.0
    cfg.weight_decay = 0.0
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/speed_arm"

    print(f"[Arm] iters={iters} frames={cfg.on_policy_collected_frames_per_batch} "
          f"envs={cfg.on_policy_n_envs_per_worker} overlap={os.environ.get('OVERLAP_COLLECTION','0')} "
          f"compile={os.environ.get('ATTENTION_COMPILE_MODE','(default)')} "
          f"bf16={os.environ.get('SAMPLING_AUTOCAST_BF16','0')}", flush=True)

    task = LayupTask.LAYUP.get_from_yaml()
    acfg = MappoConfig.get_from_yaml()
    acfg.share_param_actor = True
    acfg.share_param_critic = False
    acfg.loc_bound = 5.0

    model = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"),
            GruConfig.get_from_yaml("benchmarl/conf/model/layers/gru.yaml"),
        ],
        intermediate_sizes=[256],
    )
    critic = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

    exp = Experiment(
        task=task, algorithm_config=acfg, model_config=model,
        critic_model_config=critic, seed=0, config=cfg, callbacks=[],
    )
    try:
        exp.run()
    finally:
        exp.close()
        print("[Arm] DONE", flush=True)


if __name__ == "__main__":
    main()
