"""探测 spawn 多进程采集需要的可 pickle 对象（策略 / env_func / 采集器）。

用法（容器内）：ATTENTION_COMPILE_MODE=off python /tmp/probe_pickle.py
"""

import os
import pickle

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
os.environ.setdefault("ATTENTION_COMPILE_MODE", "off")

from benchmarl.algorithms import MappoConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = 2000
    cfg.on_policy_n_envs_per_worker = 10
    cfg.on_policy_minibatch_size = 1000
    cfg.max_n_iters = 1
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/probe_pickle"
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    cfg.buffer_device = "cpu"

    task = LayupTask.LAYUP.get_from_yaml()
    acfg = MappoConfig.get_from_yaml()
    acfg.share_param_actor = True
    acfg.share_param_critic = False
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
        targets = {
            "policy": exp.policy,
            "env_func": exp.env_func,
            "collector": getattr(exp, "collector", None),
        }
        for name, obj in targets.items():
            if obj is None:
                print(f"[pickle] {name}: None")
                continue
            try:
                pickle.dumps(obj)
                print(f"[pickle] {name}: OK")
            except Exception as e:
                print(f"[pickle] {name}: FAIL {type(e).__name__}: {str(e)[:300]}")
    finally:
        exp.close()
        print("PROBE_PICKLE_DONE")


if __name__ == "__main__":
    main()
