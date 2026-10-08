"""冻结权重（lr=0）判决实验：从 v39 iter200 续跑 8 轮，检验逐批交替是否来自训练动力学。

用法（容器内）：
  OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 python /tmp/frozen_collect_lr0.py > outputs/frozen_lr0.log 2>&1

逻辑：
  - 加载 v39 checkpoint_60000000.pt（iter200），lr=0 / weight_decay=0 ⇒ 权重完全冻结；
  - 采集管线保持与真实训练一致（overlap prefetcher + bf16 + 1500 envs/300k frames）；
  - 逐迭代打印 collection 端统计（episode_reward、终局码分布、按键率）。
  若交替仍在 ⇒ 采集管线/环境侧；若消失 ⇒ 权重双稳（训练动力学）。
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
from benchmarl.experiment.callback import Callback

CKPT = (
    "outputs/2026-10-03_10-33-56/mappo_layup_sequencemodel__a35f3c2b_26_10_03-05_39_14/"
    "checkpoints/checkpoint_60000000.pt"
)
CKPT_ABS = os.path.abspath(CKPT)

CODE_NAMES = {
    1: "c1made", 2: "c2oppfoul", 3: "c3wallD", 4: "c4lineD", 5: "c5ffD",
    11: "c11block", 12: "c12timeout", 13: "c13ownfoul", 14: "c14wallA", 15: "c15ffA",
}


class PerIterStats(Callback):
    """逐迭代打印 collection 端统计（权重冻结，只观察采集数据）。"""

    @staticmethod
    def _first_agent(t):
        """把 [T,B,4,1]（或带 1 尾维的变体）逐步取第 0 个 agent/特征，得到 [T,B]。"""
        while t.dim() > 2:
            t = t[..., 0]
        return t

    def on_batch_collected(self, batch):
        try:
            done = batch.get(("next", "done"))
            mask = self._first_agent(done).bool()            # [T, B]

            tr = self._first_agent(batch.get(("next", "agents", "info", "termination_reason")))
            rew = self._first_agent(batch.get(("next", "agents", "reward")))  # A1 每步

            ep = batch.get(("next", "agents", "episode_reward"))
            ep = self._first_agent(ep) if ep is not None else None            # A1 回合累计

            disc = batch.get(("agents", "action", "discrete"))
            disc = self._first_agent(disc).float() if disc is not None else None

            n = int(mask.sum().item())
            parts = [f"n_done={n}"]
            if n > 0:
                codes = torch.bincount(tr[mask].long(), minlength=16).tolist()
                parts.append(" ".join(f"{CODE_NAMES.get(i, 'c' + str(i))}:{codes[i]}"
                                      for i in range(16) if codes[i] > 0))
                if ep is not None:
                    e = ep[mask]
                    parts.append(f"ep(A1)_mean={float(e.mean()):+.3f} p50={float(e.median()):+.3f}")
            parts.append(f"step_rew(A1)={float(rew.mean()):+.4f}")
            if disc is not None:
                parts.append(f"press_rate={float(disc.mean()):.3f}")
            print("[PerIter] " + " | ".join(parts), flush=True)
        except Exception as e:
            import traceback
            print(f"[PerIter] ERROR {e}\n{traceback.format_exc()}", flush=True)


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_iters = 208            # 从 iter200 起再跑 8 轮
    cfg.restore_file = CKPT_ABS
    cfg.lr = 0.0                     # 冻结权重
    cfg.weight_decay = 0.0
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/frozen_lr0"

    print(f"[Frozen] ckpt={CKPT_ABS}", flush=True)
    print(f"[Frozen] lr={cfg.lr} wd={cfg.weight_decay} max_n_iters={cfg.max_n_iters} "
          f"frames={cfg.on_policy_collected_frames_per_batch} envs={cfg.on_policy_n_envs_per_worker} "
          f"overlap={os.environ.get('OVERLAP_COLLECTION', '0')}", flush=True)

    task = LayupTask.LAYUP.get_from_yaml()
    acfg = MappoConfig.get_from_yaml()
    acfg.share_param_actor = True
    acfg.share_param_critic = False
    # [复现保真] v39 运行时 yaml 的 loc_bound=5.0（2026-10-03 曾 3.0→5.0；此后 yaml 已回 3.0）
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
        critic_model_config=critic, seed=0, config=cfg, callbacks=[PerIterStats()],
    )
    try:
        exp.run()
    finally:
        exp.close()
        print("[Frozen] DONE", flush=True)


if __name__ == "__main__":
    main()
