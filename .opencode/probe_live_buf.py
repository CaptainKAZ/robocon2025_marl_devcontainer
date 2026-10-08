"""探查 replay buffer 的底层 storage 结构与 state 布局（供 live_view 取一条环境序列用）。"""
import os
import torch

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

from benchmarl.algorithms import MappoConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig

CKPT = os.environ.get("CKPT", "outputs/2026-10-03_13-24-18/"
                               "mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/"
                               "checkpoints/checkpoint_6000000.pt")


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.on_policy_collected_frames_per_batch = 300000
    cfg.on_policy_n_envs_per_worker = 1500
    cfg.on_policy_minibatch_size = 6000
    cfg.max_n_iters = 1
    cfg.render = False
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/probe_live"
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
        ], intermediate_sizes=[256])
    critic = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

    exp = Experiment(task=task, algorithm_config=acfg, model_config=model,
                     critic_model_config=critic, seed=0, config=cfg, callbacks=[])
    try:
        buf = exp.replay_buffers["agents"]
        print("[buf] type:", type(buf))
        ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=False)
        print("[ck] top keys:", list(ck.keys()))
        buf.load_state_dict(ck["buffer_agents"])
        print("[buf] after load: len=", len(buf))
        st = buf._storage
        print("[storage] type:", type(st))
        print("[storage] attrs:", [a for a in dir(st) if not a.startswith("_")][:30])
        inner = getattr(st, "_storage", None)
        print("[inner] type:", type(inner))
        if hasattr(inner, "keys"):
            ks = list(inner.keys())
            print("[inner] keys:", ks[:20])
            for k in ("state", ("agents", "observation"), ("next", "done"), ("next", "agents", "reward")):
                try:
                    v = inner.get(k) if hasattr(inner, "get") else inner[k]
                    print("   ", k, "->", None if v is None else tuple(v.shape))
                except Exception as e:
                    print("   ", k, "ERR", type(e).__name__, e)
        # state 各维统计，确认布局/量纲
        S = inner["state"] if isinstance(inner, dict) else inner.get("state")
        if S is not None:
            S = S.float()
            print("[state] shape", tuple(S.shape))
            print("[state] 每维 min/max:")
            flat = S.reshape(-1, S.shape[-1])
            mn, mx = flat.amin(0), flat.amax(0)
            for d in range(S.shape[-1]):
                print(f"   dim{d:2d}: min={mn[d]:+.3f} max={mx[d]:+.3f}")
            print("[state] dim4(in_spot) 取值:", torch.unique(flat[:, 4])[:6].tolist())
            print("[state] dim5(progress) 分位:", torch.quantile(flat[:, 5], torch.tensor([0.5, 0.9, 0.99])).tolist())
            print("[state] dim18:20(spot) 样例:", flat[:3, 18:20].tolist())
            print("[state] dim20:22(basket) 样例:", flat[:3, 20:22].tolist())
            print("[state] dim22(time) 分位:", torch.quantile(flat[:, 22], torch.tensor([0.0, 0.5, 1.0])).tolist())
    finally:
        exp.close()
        print("PROBE_LIVE_DONE")


if __name__ == "__main__":
    main()
