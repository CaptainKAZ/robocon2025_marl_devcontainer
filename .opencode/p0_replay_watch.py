"""带仪表的回放实验 v2：从 v39 checkpoint_60000000.pt（iter200）续跑，lr=0（权重冻结），
逐迭代打印【原始量纲】的动作/参数/物理速度/观测统计，抓捕 iter~203 的"采集崩坏"瞬间。

用法（容器内）：
  OVERLAP_COLLECTION=1 SAMPLING_AUTOCAST_BF16=1 python /tmp/p0_replay_watch.py > outputs/replay_watch.log 2>&1
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

SCALE_ABS = torch.tensor([4.0, 7.5])
SCALE_REL = torch.tensor([8.0, 15.0])

CODE_NAMES = {
    1: "made", 2: "oppfoul", 3: "wallD", 4: "lineD", 5: "ffD",
    11: "block", 12: "timeout", 13: "ownfoul", 14: "wallA", 15: "ffA",
}


class Watch(Callback):
    @staticmethod
    def _fa(t):
        while t is not None and t.dim() > 2:
            t = t[..., 0]
        return t

    def on_batch_collected(self, batch):
        it = self.experiment.n_iters_performed
        try:
            done = self._fa(batch.get(("next", "done"))).bool()
            tr = self._fa(batch.get(("next", "agents", "info", "termination_reason")))
            ep = self._fa(batch.get(("next", "agents", "episode_reward")))

            obs = batch.get(("agents", "observation"))
            a1 = obs[..., 0, :].float() if obs.dim() >= 3 else obs.float()
            a1pos = a1[..., 0:2] * SCALE_ABS
            spot_rel = a1[..., 33:35] * SCALE_REL
            d_spot = torch.linalg.norm(spot_rel, dim=-1)
            in_spot = ((d_spot <= 1.2) & (a1pos[..., 1] > 0)).float().mean().item()
            prog = a1[..., 7]
            charging = (prog > 0).float().mean().item()
            prog_max = float(prog.max())

            # 物理速度（obs self[2:4] 归一化 /5）
            a1vel = a1[..., 2:4] * 5.0
            spd_mean = float(a1vel.norm(dim=-1).mean())
            spd_max = float(a1vel.norm(dim=-1).max())

            # 原始动作（不降维）
            cont = batch.get(("agents", "action", "continuous")).float()
            disc = batch.get(("agents", "action", "discrete"))
            cont_info = (f"cont{tuple(cont.shape)}[{float(cont.min()):+.2f},{float(cont.max()):+.2f}]"
                         f" absm={float(cont.abs().mean()):.3f}")
            if cont.dim() >= 3:
                a1c = cont[..., 0, :]
                cont_info += f" A1norm={float(a1c.norm(dim=-1).mean()):.3f}"

            loc = batch.get(("agents", "params", "continuous", "loc"), default=None)
            sc = batch.get(("agents", "params", "continuous", "scale"), default=None)
            par_info = ""
            if loc is not None:
                lf = loc.float()
                par_info += f" loc{tuple(lf.shape)}[{float(lf.min()):+.2f},{float(lf.max()):+.2f}]"
            if sc is not None:
                sf = sc.float()
                par_info += f" scale[{float(sf.min()):.3f},{float(sf.max()):.3f}]"

            # 键扫描（只打一次，找 hidden state）
            key_info = ""
            if it <= 200:
                ks = [str(k) for k in batch.keys(True, True)]
                key_info = " [keys] " + str([k for k in ks if any(s in k.lower() for s in ("hidden", "params", "state"))][:20])

            n = int(done.sum().item())
            codes = torch.bincount(tr[done].long(), minlength=16).tolist() if n > 0 else [0] * 16
            code_str = " ".join(f"{CODE_NAMES.get(i, i)}:{codes[i]}" for i in range(16) if codes[i] > 0)
            ep_mean = float(ep[done].mean()) if n > 0 and ep is not None else float("nan")

            print(f"[Watch] it={it} n={n} ep(A1)={ep_mean:+.2f} | {code_str}"
                  f" | in_spot={in_spot*100:.1f}% charging={charging*100:.1f}% prog_max={prog_max:.2f}"
                  f" | speed={spd_mean:.2f}/{spd_max:.2f} | {cont_info}{par_info} | press={float(disc.float().mean()):.3f}"
                  f"{key_info}", flush=True)
        except Exception as e:
            import traceback
            print(f"[Watch] it={it} ERROR {e}\n{traceback.format_exc()}", flush=True)


def main():
    cfg = ExperimentConfig.get_from_yaml()
    cfg.max_n_iters = 204            # 从 iter200 起再跑 4 轮（覆盖崩坏点）
    cfg.restore_file = CKPT_ABS
    cfg.lr = 0.0
    cfg.weight_decay = 0.0
    cfg.evaluation = False
    cfg.checkpoint_interval = 0
    cfg.checkpoint_at_end = False
    cfg.create_json = False
    cfg.save_folder = "/tmp/replay_watch"

    print(f"[Replay] ckpt={CKPT_ABS} lr=0 max_n_iters={cfg.max_n_iters}", flush=True)

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
    critic = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/critic_agents.yaml") \
        if os.path.exists("benchmarl/conf/model/layers/critic_agents.yaml") \
        else AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

    exp = Experiment(
        task=task, algorithm_config=acfg, model_config=model,
        critic_model_config=critic, seed=0, config=cfg, callbacks=[Watch()],
    )
    try:
        exp.run()
    finally:
        exp.close()
        print("[Replay] DONE", flush=True)


if __name__ == "__main__":
    main()
