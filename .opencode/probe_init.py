import os
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
import torch
from benchmarl.environments import LayupTask

task = LayupTask.LAYUP.get_from_yaml()
cfg = dict(task.config)
print("cfg fixed_init =", cfg.get("fixed_init"), "| fixed_spot =", cfg.get("fixed_spot"))

env_fun = task.get_env_fun(8, True, 0, "cpu")
env = env_fun()
td = env.reset()
obs = td.get(("agents", "observation"))
print("obs shape:", tuple(obs.shape))
o = obs.reshape(obs.shape[0], -1, 4, 41)[:, 0]
pos = o[:, :, :2] * torch.tensor([4.0, 7.5])
for i, nm in enumerate(["A1", "A2", "D1", "D2"]):
    p = pos[:, i]
    xs = " ".join(f"{v:+.2f}" for v in p[:6, 0])
    ys = " ".join(f"{v:+.2f}" for v in p[:6, 1])
    print(f"{nm}: x=[{xs}] y=[{ys}]")
mid = (pos[:, 2] + pos[:, 3]) / 2
print("A2->mid(D1,D2) dist:", " ".join(f"{d:.2f}" for d in (pos[:, 1] - mid).norm(dim=-1).tolist()))
env.close()
print("DONE")
