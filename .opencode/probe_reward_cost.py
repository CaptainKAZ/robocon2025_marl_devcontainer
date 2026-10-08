"""[性能归因] env.step 里的"奖励/判罚函数"（含碰撞判责）到底占多少？

用法: python /tmp/probe_reward_cost.py
"""

import os
import time

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch

from benchmarl.environments import LayupTask

B, STEPS, WARM = 1500, 25, 3

task = LayupTask.LAYUP.get_from_yaml()
env = task.get_env_fun(num_envs=B, continuous_actions=True, seed=0, device=torch.device("cpu"))()
sc = env.scenario
orig = sc.jitted_reward_calculator
acc = {"t": 0.0, "n": 0}


def timed(*a, **k):
    t0 = time.perf_counter()
    out = orig(*a, **k)
    acc["t"] += time.perf_counter() - t0
    acc["n"] += 1
    return out


sc.jitted_reward_calculator = timed


def act(td):
    td.set(("agents", "action", "continuous"), torch.randn(B, 4, 2) * 0.4)
    td.set(("agents", "action", "discrete"), torch.zeros(B, 4, 1, dtype=torch.long))
    return env.step(td)


td = env.reset()
for _ in range(WARM):
    td = act(td)

acc["t"] = 0.0
acc["n"] = 0
t0 = time.perf_counter()
for _ in range(STEPS):
    td = act(td)
step_ms = (time.perf_counter() - t0) / STEPS * 1000
rew_ms = acc["t"] / STEPS * 1000
print(
    f"env.step = {step_ms:.2f} ms/step（{B} envs）\n"
    f"  其中 奖励/判罚函数 = {rew_ms:.2f} ms/step（调用 {acc['n']} 次 = {acc['n']/STEPS:.1f} 次/步）"
    f" -> 占 env.step 的 {rew_ms/step_ms*100:.1f}%"
)
env.close()
