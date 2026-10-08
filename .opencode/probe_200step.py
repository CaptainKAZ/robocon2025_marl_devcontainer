import os, torch
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
from benchmarl.environments import LayupTask
from benchmarl.experiment import ExperimentConfig

task = LayupTask.LAYUP.get_from_yaml()
cfg = ExperimentConfig.get_from_yaml()
F = cfg.on_policy_collected_frames_per_batch
E = cfg.on_policy_n_envs_per_worker
print(f"[cfg] frames={F} envs={E} -> rollout T = {F // E}")
print(f"[cfg] task.max_steps = {task.max_steps(None)}")

env = task.get_env_fun(num_envs=2, continuous_actions=True, seed=0, device=torch.device("cpu"))()
scen = None
for path in ("scenario", "unwrapped.scenario", "_env.scenario", "env.scenario"):
    o = env
    ok = True
    for p in path.split("."):
        if hasattr(o, p):
            o = getattr(o, p)
        else:
            ok = False
            break
    if ok:
        scen = o
        break
print("[env] scenario:", type(scen).__name__ if scen is not None else "NOT FOUND")
if scen is not None:
    hp = scen.h_params
    print(
        "[env] t_limit=%s max_steps=%s k_blocked_shot_penalty=%s grace=%s"
        % (hp["t_limit"], scen.max_steps, hp.get("k_blocked_shot_penalty"), hp["time_penalty_grace_period"])
    )

td = env.reset()
n = 0
code = None
a0 = td.get(("agents", "action"), None)
if a0 is None:
    obs0 = td.get(("agents", "observation"))
    a0 = torch.zeros(obs0.shape[0], obs0.shape[1], 2)
print("[dbg] 零动作形状:", tuple(a0.shape))
for t in range(400):
    td.set(("agents", "action"), torch.zeros_like(a0))
    td = env.step(td)
    n += 1
    d = td.get(("next", "done"))
    if bool(d.any()):
        tr = td.get(("next", "agents", "info", "termination_reason")).flatten()
        code = sorted({int(x) for x in tr if int(x) > 0})
        break
rew = td.get(("next", "agents", "reward"))
print(f"[rollout] 零动作终局于第 {n} 步, 码={code}")
print(
    "[rollout] 终局奖励 A1=%.2f A2=%.2f D1=%.2f D2=%.2f"
    % (float(rew[0, 0]), float(rew[0, 1]), float(rew[0, 2]), float(rew[0, 3]))
)
