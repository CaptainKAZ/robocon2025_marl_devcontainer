"""投篮按键：算法级 tiny 冒烟（复合动作 + mask + role_out_dims=[6,4,4]）"""
import os
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "1.2")
import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback


class Probe(Callback):
    def __init__(self):
        super().__init__()
        self._done = False

    def on_batch_collected(self, batch):
        try:
            keys = [str(k) for k in batch.keys(True) if "mask" in str(k)]
            done = batch.get(("next", "done")).squeeze(-1).bool()
            n = int(done.sum())
            tr = batch.get(("next", "agents", "info", "termination_reason"))[..., 0, :].squeeze(-1)
            print(f"[SMOKE] mask keys={keys} dones={n}")
            if n:
                r = tr[done]
                print(f"[SMOKE] 码分布: " + ", ".join(f"{c}:{int((r==c).sum())}" for c in sorted({int(x) for x in r})))
            if not self._done:
                self._done = True
                m = batch.get(("agents", "action_mask"))
                print(f"[SMOKE] 当前 mask 形状={tuple(m.shape)} A1可按比例={float(m[..., 0, 1].float().mean()):.4f}")
                a = batch.get(("agents", "action", "discrete"))
                print(f"[SMOKE] 离散动作形状={tuple(a.shape)} 取值={sorted({int(x) for x in a.flatten()})}")
        except Exception as e:
            print("[SMOKE] cb err", repr(e))


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = 2000   # 10 envs -> T = 200
    cfg.on_policy_n_envs_per_worker = 10
    cfg.on_policy_minibatch_size = 500
    cfg.on_policy_n_minibatch_iters = 2
    cfg.max_n_iters = 3
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/p0_smoke_button"
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    cfg.loggers = []

    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=2, continuous_actions=True, seed=0, device="cpu")()
    print("[SMOKE] 动作 spec =", env.action_spec)
    print("[SMOKE] action_mask_spec =", task.action_mask_spec(env))
    env.close()

    acfg = MappoConfig.get_from_yaml()
    acfg.share_param_actor = True
    acfg.share_param_critic = False
    model_config = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"),
            GruConfig.get_from_yaml(),
        ],
        intermediate_sizes=[256],
    )
    critic_config = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
    exp = Experiment(task=task, algorithm_config=acfg, model_config=model_config,
                     critic_model_config=critic_config, seed=0, config=cfg, callbacks=[Probe()])
    print("[SMOKE] group_map =", exp.train_group_map)
    exp.run()

    b = exp.replay_buffers["agents"]
    td = b.sample()
    print("[SMOKE] buffer 采样键:", [str(k) for k in td.keys(True) if "action" in str(k) or "mask" in str(k)])
    adv = td.get(("agents", "advantage"))
    print(f"[SMOKE] adv={tuple(adv.shape)} finite={bool(torch.isfinite(adv).all())}")
    for name, net in [("actor", exp.losses["agents"].actor_network), ("critic", exp.losses["agents"].critic_network)]:
        n = sum(p.numel() for p in net.parameters())
        print(f"[SMOKE] {name} 参数量={n/1e6:.3f}M")
    exp.close()
    print("[SMOKE] DONE")


if __name__ == "__main__":
    main()
