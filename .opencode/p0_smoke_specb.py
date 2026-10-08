"""方案 B（spec 级恢复 ±v_max）冒烟：确认 hybrid 连续 spec 恢复 ±5、动作落在 ±5、log_prob 正常。"""

import os

os.environ["VMAS_INITIAL_SHOT_THRESHOLD"] = "0.2"

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig


class Probe(Callback):
    def on_batch_collected(self, batch):
        try:
            spec = self.experiment.action_spec[("agents", "action")]
            cont = spec["continuous"]
            lo = torch.as_tensor(cont.space.low).flatten()
            hi = torch.as_tensor(cont.space.high).flatten()
            print(f"[probe] hybrid continuous space low={lo[:3].tolist()} high={hi[:3].tolist()} shape={tuple(cont.shape)}")
        except Exception as e:  # noqa: BLE001
            print("[probe] spec err:", e)
        try:
            act = batch.get(("agents", "action"))
            c = act.get("continuous")
            print(f"[probe] action.continuous min={c.min():.3f} max={c.max():.3f} absmean={c.abs().mean():.3f}")
        except Exception as e:  # noqa: BLE001
            print("[probe] act err:", e)
        try:
            loc = batch.get(("agents", "params", "continuous", "loc"))
            sc = batch.get(("agents", "params", "continuous", "scale"))
            lp = batch.get(("agents", "log_prob"))
            print(f"[probe] loc=[{loc.min():.3f},{loc.max():.3f}] scale=[{sc.min():.4f},{sc.max():.4f}] "
                  f"log_prob=[{lp.min():.3f},{lp.max():.3f}]")
        except Exception as e:  # noqa: BLE001
            print("[probe] params err:", e)


def main():
    os.makedirs("/tmp/p0_smoke_specb", exist_ok=True)
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = 2000
    cfg.on_policy_n_envs_per_worker = 10
    cfg.on_policy_minibatch_size = 500
    cfg.max_n_iters = 3
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.loggers = []
    cfg.save_folder = "/tmp/p0_smoke_specb"
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"

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
        task=task,
        algorithm_config=acfg,
        model_config=model,
        critic_model_config=critic,
        seed=0,
        config=cfg,
        callbacks=[Probe()],
    )
    exp.run()
    exp.close()
    print("SMOKE_SPECB_DONE")


if __name__ == "__main__":
    main()
