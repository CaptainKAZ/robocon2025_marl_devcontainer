"""[感知噪声] 真实开销测量（修正 probe_noise_cost.py 的假开关）：
  旧探针只把 sigma 系数置 0，但 `_noise_enabled` 在 build 时就定死了，randn 分支照跑 -> 测出来的差值没有意义。
  本探针直接翻 `_noise_enabled` 开关，并单独计时 `_refresh_perception()`。

用法: python /tmp/probe_noise_cost2.py
"""
import os, time, torch

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
from benchmarl.environments import LayupTask

B, STEPS, CALLS = 1500, 25, 200
ABS_DIV = torch.tensor([4.0, 7.5])


def build():
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=B, continuous_actions=True, seed=0,
                           device=torch.device("cpu"))()
    return env


def obs_err(env):
    td = env.reset()
    obs = td.get(("agents", "observation"))
    seen = obs[:, 0, 21:23] * ABS_DIV
    return (seen - env.world.agents[2].state.pos).norm(dim=-1).mean().item()


def step_cost(env, steps=STEPS):
    td = env.reset()
    t0 = time.time()
    for _ in range(steps):
        td.set(("agents", "action", "continuous"), torch.randn(B, 4, 2) * 0.4)
        td.set(("agents", "action", "discrete"), torch.zeros(B, 4, 1, dtype=torch.long))
        td = env.step(td)
    return (time.time() - t0) / steps * 1000


def call_cost(env, calls=CALLS):
    sc = env.scenario
    sc._refresh_perception()
    t0 = time.time()
    for _ in range(calls):
        sc._refresh_perception()
    return (time.time() - t0) / calls * 1000


env = build()
sc = env.scenario
print(f"[cfg] _noise_enabled={getattr(sc, '_noise_enabled', None)}  h_params: "
      f"k_perception_noise={sc.h_params.get('k_perception_noise')} floor={sc.h_params.get('perception_noise_floor')} "
      f"k_vel={sc.h_params.get('k_perception_noise_vel')} floor_vel={sc.h_params.get('perception_noise_vel_floor')}")

for flag in (True, False):
    sc._noise_enabled = flag
    err = obs_err(env)
    cc = call_cost(env)
    st = step_cost(env)
    print(f"  _noise_enabled={str(flag):5s}  观测误差 mean={err:.4f} m | _refresh_perception={cc:.3f} ms/call "
          f"| env.step={st:.2f} ms/step")

env.close()
print("DONE")
