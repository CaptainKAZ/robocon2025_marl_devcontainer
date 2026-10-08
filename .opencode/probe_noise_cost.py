"""[感知噪声] 开关 + 性能：对比"噪声开(默认)/关"的单步耗时；关闭时应看到零观测误差。

用法: python /tmp/probe_noise_cost.py
"""
import os, time, torch
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")
from benchmarl.environments import LayupTask

B, STEPS = 1500, 25
ABS_DIV = torch.tensor([4.0, 7.5])
OFF = dict(k_perception_noise=0.0, perception_noise_floor=0.0,
           k_perception_noise_vel=0.0, perception_noise_vel_floor=0.0)


def run(tag, off=False):
    task = LayupTask.LAYUP.get_from_yaml()
    env = task.get_env_fun(num_envs=B, continuous_actions=True, seed=0,
                           device=torch.device("cpu"))()
    if off:
        for k, val in OFF.items():
            env.scenario.h_params[k] = val
    td = env.reset()
    obs = td.get(("agents", "observation"))
    seen = obs[:, 0, 21:23] * ABS_DIV
    err = (seen - env.world.agents[2].state.pos).norm(dim=-1)
    t0 = time.time()
    for _ in range(STEPS):
        td.set(("agents", "action", "continuous"), torch.randn(B, 4, 2) * 0.4)
        td.set(("agents", "action", "discrete"), torch.zeros(B, 4, 1, dtype=torch.long))
        td = env.step(td)
    dt = (time.time() - t0) / STEPS
    print(f"  [{tag}] {dt*1000:.1f} ms/step（{B} envs）  首帧观测误差 mean={err.mean():.4f} m")
    env.close()
    return dt


print("-- 单步耗时与开关 --")
d_on = run("噪声开(默认)")
d_off = run("噪声关", off=True)
print(f"  => 噪声开销 {(d_on-d_off)*1000:+.1f} ms/step（{(d_on/d_off-1)*100:+.1f}%）")
