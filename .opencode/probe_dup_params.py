"""单组 actor/critic 网络里是否有重复注册的参数（优化器 duplicate parameter 警告溯源）"""
import os
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "1.2")
import torch
from collections import Counter
from benchmarl.algorithms import MappoConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig


def report(name, params):
    by_id = Counter(id(p) for p in params)
    dup = {k: v for k, v in by_id.items() if v > 1}
    print(f"[DUP] {name}: 参数张量 {len(params)} 个，唯一 {len(by_id)} 个，重复 id 数 {len(dup)}")
    if dup:
        idset = set(dup)
        names = [n for n, p in params if id(p) in idset]
        print(f"[DUP] {name} 重复参数名: {names[:12]}")


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = 400
    cfg.on_policy_n_envs_per_worker = 2
    cfg.on_policy_minibatch_size = 200
    cfg.on_policy_n_minibatch_iters = 1
    cfg.max_n_iters = 1
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/p0_dup_probe"
    cfg.train_device = "cpu"
    cfg.sampling_device = "cpu"
    cfg.loggers = []

    task = LayupTask.LAYUP.get_from_yaml()
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
                     critic_model_config=critic_config, seed=0, config=cfg, callbacks=[])

    loss = exp.losses["agents"]
    report("actor_network", list(loss.actor_network.named_parameters()))
    report("critic_network", list(loss.critic_network.named_parameters()))
    report("critic_network_params_holder", list(loss.critic_network_params.named_parameters()))
    for k, m in loss.named_children():
        report(f"loss.{k}", list(m.named_parameters()))
    opts = exp.optimizers["agents"]
    print("[DUP] optimizers[agents] 类型:", type(opts).__name__, "键:", list(opts.keys()) if hasattr(opts, "keys") else "?")
    for key, opt in (opts.items() if hasattr(opts, "items") else enumerate(opts)):
        n = sum(len(g["params"]) for g in opt.param_groups)
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        print(f"[DUP] optimizer[{key}] {type(opt).__name__}: groups={len(opt.param_groups)} params={n} unique={len(set(ids))}")
    # 直接查 mappo 的 _get_parameters 输出
    for g, params in exp.algorithm._get_parameters(exp.losses).items():
        ids = [id(p) for p in params]
        print(f"[DUP] _get_parameters[{g}]: n={len(ids)} unique={len(set(ids))}")
    exp.close()
    print("[DUP] DONE")


if __name__ == "__main__":
    main()
