"""采集耗时拆分：env.step vs 策略前向（CPU，1500 envs，bf16）。
用法: PROBE_ENVS=1500 PROBE_STEPS=40 python /tmp/probe_speed_split.py
"""
import os, time, torch

os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

from benchmarl.environments import LayupTask

NE = int(os.environ.get("PROBE_ENVS", "1500"))
STEPS = int(os.environ.get("PROBE_STEPS", "40"))

print(f"[cfg] threads={torch.get_num_threads()} envs={NE} steps={STEPS}")
task = LayupTask.LAYUP.get_from_yaml()
env = task.get_env_fun(num_envs=NE, continuous_actions=True, seed=0, device="cpu")()
try:
    for tr in (task.get_env_transforms(env) or []):
        env.append_transform(tr)
except Exception as e:
    print("[warn] transforms:", e)
env = env.to("cpu")

td = env.reset()

def step_once(td):
    td = env.rand_action(td)
    return env.step(td)

# warmup
for _ in range(3):
    tdx = step_once(td)
    td = tdx

# --- 1) env.step + rand_action（无策略） ---
t0 = time.time()
for _ in range(STEPS):
    tdx = step_once(td)
    td = tdx
dt_env = (time.time() - t0) / STEPS
print(f"[split] env.step(+rand_action) = {dt_env*1000:.1f} ms/step  -> {NE/dt_env/1000:.2f}k fps(env only)")

# --- 2) 纯 reset 成本 ---
t0 = time.time()
for _ in range(STEPS):
    td = env.reset()
dt_reset = (time.time() - t0) / STEPS
print(f"[split] env.reset = {dt_reset*1000:.1f} ms/step")

# --- 3) 策略前向（用随机初始化的当前架构） ---
try:
    from benchmarl.algorithms import MappoConfig
    from benchmarl.models.common import SequenceModelConfig
    from benchmarl.models.attention import AttentionConfig
    from benchmarl.models.gru import GruConfig
    acfg = MappoConfig.get_from_yaml()
    acfg.share_param_actor = True
    acfg.share_param_critic = False
    mc = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_agents.yaml"),
            GruConfig.get_from_yaml(),
        ],
        intermediate_sizes=[256],
    )
    from benchmarl.models.common import get_model
    model = get_model(
        model_config=mc,
        input_spec=env.observation_spec,
        output_spec=env.action_spec,
        action_spec=env.action_spec,
        agent_group=list(env.group_map.keys())[0],
        input_has_agent_dim=True,
        n_agents=len(env.group_map[list(env.group_map.keys())[0]]),
        centralised=False,
        share_params=True,
        device=torch.device("cpu"),
    ).to("cpu").eval()
    td0 = env.reset()
    use_bf16 = os.environ.get("PROBE_BF16", "1") == "1"
    with torch.no_grad():
        for _ in range(2):
            if use_bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(td0.select(*model.in_keys))
            else:
                out = model(td0.select(*model.in_keys))
        t0 = time.time()
        for _ in range(STEPS):
            if use_bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(td0.select(*model.in_keys))
            else:
                out = model(td0.select(*model.in_keys))
        dt_pol = (time.time() - t0) / STEPS
    n_agents = len(env.group_map[list(env.group_map.keys())[0]])
    print(f"[split] policy.forward(bf16={use_bf16}) = {dt_pol*1000:.1f} ms/step (rows={NE*n_agents})")
    print(f"[split] 合计估计 = {(dt_env+dt_pol)*1000:.1f} ms/step")
except Exception as e:
    import traceback; traceback.print_exc()
    print("[warn] policy probe failed:", e)

print("PROBE_SPLIT_DONE")
