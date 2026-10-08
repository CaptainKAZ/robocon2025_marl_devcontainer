"""minibatch 规模 A/B：冻结权重从 v41 iter450 续跑 ARM_ITERS 轮，测"训练耗时 + 显存 + 单轮墙钟"。

用法（容器内，cwd=/home/vscode/workspace/BenchMARL）：
  MB_SIZE=12000 ARM_ITERS=3 OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 python /tmp/mb_arm.py

输出关键字：
  [MbArm] 每轮 iter_dt / cuda 峰值
  [PROF][agents] ... opt_loops=  训练侧耗时（experiment.py 打印）
  collection time:               采集侧耗时
"""

import os
import time

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch

from benchmarl.algorithms import MappoConfig
from benchmarl.models import SequenceModelConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.gru import GruConfig
from benchmarl.environments import LayupTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback

CKPT = (
    "outputs/2026-10-04_03-32-31/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/"
    "checkpoints/checkpoint_135000000.pt"
)
START_ITER = 450


class MbProbe(Callback):
    """每轮打印：墙钟间隔、CUDA 峰值占用、本组采样 batch 形状。"""

    def __init__(self):
        super().__init__()
        self.last = None

    def on_batch_collected(self, batch):
        now = time.time()
        dt = 0.0 if self.last is None else now - self.last
        self.last = now
        alloc = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        resv = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        try:
            buf = self.experiment.replay_buffers["agents"]
            sampled = buf.sample()
            shp = tuple(sampled.get(("agents", "observation")).shape)
        except Exception as exc:  # noqa: BLE001
            shp = f"(sample failed: {exc})"
        print(
            f"[MbArm] iter={self.experiment.n_iters_performed} iter_dt={dt:.1f}s "
            f"cuda_max_alloc={alloc:.2f}GB cuda_max_resv={resv:.2f}GB sample_obs={shp}",
            flush=True,
        )


def main():
    iters = int(os.environ.get("ARM_ITERS", "3"))
    mb = int(os.environ.get("MB_SIZE", "6000"))
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_iters = START_ITER + iters
    cfg.restore_file = os.path.abspath(CKPT)
    cfg.lr = 0.0
    cfg.weight_decay = 0.0
    cfg.on_policy_minibatch_size = mb
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = f"/tmp/mb_arm_{mb}"
    os.makedirs(cfg.save_folder, exist_ok=True)

    frames = cfg.on_policy_collected_frames_per_batch
    envs = cfg.on_policy_n_envs_per_worker
    print(
        f"[MbArm] MB={mb} iters={iters} frames={frames} envs={envs} "
        f"n_minibatch_iters={cfg.on_policy_n_minibatch_iters} "
        f"overlap={os.environ.get('OVERLAP_COLLECTION', '0')} "
        f"bf16={os.environ.get('SAMPLING_AUTOCAST_BF16', '0')}",
        flush=True,
    )

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
        callbacks=[MbProbe()],
    )

    # 恢复 checkpoint 会把 buffer 的 _batch_size 一起带回来（6000/200=30 条序列），
    # 这样 minibatch 变大只会减少优化步数、每步采样量不变。这里按当前配置重新对齐。
    seq_len = -(-frames // (envs * cfg.n_workers))
    want = max(1, -(-mb // seq_len))
    for gname, buf in exp.replay_buffers.items():
        old = getattr(buf, "_batch_size", None)
        buf._batch_size = want
        print(f"[MbArm] buffer[{gname}] _batch_size: {old} -> {want} (seq_len={seq_len})", flush=True)

    try:
        t0 = time.time()
        exp.run()
        print(f"[MbArm] DONE MB={mb} total={time.time() - t0:.1f}s", flush=True)
    finally:
        exp.close()


if __name__ == "__main__":
    main()
