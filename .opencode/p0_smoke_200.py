import os, torch
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "1.2")
from benchmarl.algorithms import MappoConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.environments.layup.common import LayupClass

class Win(Callback):
    def on_batch_collected(self, batch):
        try:
            done = batch.get(("next", "done")).squeeze(-1).bool()
            tr = batch.get(("next", "agents", "info", "termination_reason"))[..., 0, :].squeeze(-1)
            n = int(done.sum())
            if n:
                r = tr[done]
                print(f"[SMOKE] dones={n} 码1={float(((r==1).float()).mean())*100:.1f}% 唯码={sorted({int(x) for x in r if int(x)>0})}")
        except Exception as e:
            print("[SMOKE] cb err", e)

cfg = ExperimentConfig.get_from_yaml()
cfg.on_policy_collected_frames_per_batch = 2000      # 10 envs -> T = 200
cfg.on_policy_n_envs_per_worker = 10
cfg.on_policy_minibatch_size = 500
cfg.on_policy_n_minibatch_iters = 2
cfg.max_n_iters = 3
cfg.render = False
cfg.evaluation = False
cfg.checkpoint_interval = 0
cfg.checkpoint_at_end = False
cfg.create_json = False
cfg.save_folder = "/tmp/p0_smoke_200"
cfg.train_device = "cpu"; cfg.sampling_device = "cpu"
cfg.loggers = []

task = LayupTask.LAYUP.get_from_yaml()
acfg = MappoConfig.get_from_yaml(); acfg.share_param_actor = True; acfg.share_param_critic = False
model_config = SequenceModelConfig(
    model_configs=[AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"), GruConfig.get_from_yaml()],
    intermediate_sizes=[256],
)
critic_config = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")
exp = Experiment(task=task, algorithm_config=acfg, model_config=model_config, critic_model_config=critic_config,
                 seed=0, config=cfg, callbacks=[Win()])
print("[SMOKE] group_map =", exp.train_group_map, "T =", cfg.on_policy_collected_frames_per_batch // cfg.on_policy_n_envs_per_worker)
exp.run()
try:
    b = exp.replay_buffers["agents"]
    td = b.sample()
    obs = td.get(("agents", "observation"))
    adv = td.get(("agents", "advantage"))
    print(f"[SMOKE] sample obs={tuple(obs.shape)} adv={tuple(adv.shape)} finite={bool(torch.isfinite(adv).all())}")
except Exception as e:
    print("[SMOKE] buffer check err:", e)
exp.close()
print("[SMOKE] DONE")
