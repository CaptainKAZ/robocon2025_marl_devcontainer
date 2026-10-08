"""[速度诊断] 采集侧 A/B：定位 v43（74-88 s/it）比 v42（65.5 s/it）慢 ~8.5s/轮的原因。

三个候选因素（其余环境与 v43 完全一致：1500 envs / 300k frames / MB=12000 / 单组 agents / ckpt=iter460）：
  1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True （v42 没有）
  2. LIVE_VIEW=1 观察窗回调（v42 没有）
  3. 评测 worker（两组都开，但只有评测轮才会渲染视频）

用法（容器内，cwd=BenchMARL）:
  ARM_TAG=A ARM_EVAL=1 LIVE_VIEW=1 ARM_ITERS=3 python /tmp/speed_arm4.py
"""
import os
import time
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import torch  # noqa: E402
from benchmarl.algorithms import MappoConfig  # noqa: E402
from benchmarl.models.attention import AttentionConfig  # noqa: E402
from benchmarl.models.gru import GruConfig  # noqa: E402
from benchmarl.models.common import SequenceModelConfig  # noqa: E402
from benchmarl.environments import LayupTask  # noqa: E402
from benchmarl.experiment import Experiment, ExperimentConfig, Callback  # noqa: E402

TAG = os.environ.get("ARM_TAG", "arm")
N = int(os.environ.get("ARM_ITERS", "3"))
EVAL = os.environ.get("ARM_EVAL", "1") == "1"
LV = os.environ.get("LIVE_VIEW", "0") == "1"
START = int(os.environ.get("ARM_START", "460"))
CKPT = os.environ.get(
    "ARM_CKPT",
    "outputs/2026-10-06_10-37-05/mappo_layup_sequencemodel__6a32dad8_26_10_03-13_24_18/checkpoints/checkpoint_138000000.pt",
)


class DtProbe(Callback):
    """每轮打印墙钟间隔 + CUDA 峰值；第一轮为参考（含评测与编译暖机）。"""

    def __init__(self):
        super().__init__()
        self.t0 = None
        self.n = 0

    def on_batch_collected(self, batch):
        now = time.time()
        dt = 0.0 if self.t0 is None else now - self.t0
        self.t0 = now
        cuda_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        print(
            f"[Arm:{TAG}] iter={START + self.n} dt={dt:.1f}s cuda_max={cuda_mb:.0f}MB",
            flush=True,
        )
        self.n += 1


cfg = ExperimentConfig.get_from_yaml()
cfg.max_n_iters = START + N
cfg.restore_file = CKPT
cfg.lr = 0.0                      # 冻结权重：只测时间，不动策略
cfg.evaluation = EVAL
cfg.checkpoint_interval = 0
cfg.checkpoint_at_end = False
cfg.save_folder = f"/tmp/speed4_{TAG}"
os.makedirs(cfg.save_folder, exist_ok=True)

callbacks = [DtProbe()]
if LV:
    import sys

    sys.path.insert(0, "/home/vscode/workspace/BenchMARL/liveview")
    from live_view import LiveViewCallback

    callbacks.append(LiveViewCallback(out_dir=f"/tmp/speed4_live_{TAG}"))
    print(f"[Arm:{TAG}] LiveViewCallback 已挂载", flush=True)

task = LayupTask.LAYUP.get_from_yaml()
acfg = MappoConfig.get_from_yaml()
acfg.share_param_actor = True
acfg.share_param_critic = False
mcfg = SequenceModelConfig(
    model_configs=[
        AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"),
        GruConfig.get_from_yaml(),
    ],
    intermediate_sizes=[256],
)
ccfg = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

print(
    f"[Arm:{TAG}] eval={EVAL} live_view={LV} mb={cfg.on_policy_minibatch_size} "
    f"alloc={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(default)')}",
    flush=True,
)
exp = Experiment(
    task=task,
    algorithm_config=acfg,
    model_config=mcfg,
    critic_model_config=ccfg,
    seed=114514,
    config=cfg,
    callbacks=callbacks,
)
try:
    exp.run()
finally:
    exp.close()
print(f"ARM {TAG} DONE", flush=True)
