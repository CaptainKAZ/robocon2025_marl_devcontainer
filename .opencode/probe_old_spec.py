"""探查：v37 侧（旧规则连续动作）的 action spec 结构与 2 维替换可行性。"""
import os
os.environ.setdefault("VMAS_INITIAL_SHOT_THRESHOLD", "0.2")

import torch
from benchmarl.environments import LayupTask
from benchmarl.environments.layup.common import VmasEnvWithState

GROUP_SINGLE = {"agents": ["attacker_1", "attacker_2", "defender_1", "defender_2"]}

task = LayupTask.LAYUP.get_from_yaml()
env = VmasEnvWithState(scenario="layup", num_envs=4, continuous_actions=True, seed=0,
                       device="cpu", clamp_actions=True, group_map=GROUP_SINGLE, **dict(task.config))
spec = task.action_spec(env)
print("[spec] type:", type(spec).__name__)
print("[spec] keys:", list(spec.keys()))
leaf = spec[("agents", "action")]
print("[leaf] type:", type(leaf).__name__, "shape:", tuple(leaf.shape), "dtype:", leaf.dtype)
print("[leaf] has space:", hasattr(leaf, "space"))
if hasattr(leaf, "space"):
    print("[leaf] space type:", type(leaf.space).__name__)
    print("[leaf] low:", leaf.space.low, "high:", leaf.space.high)

from torchrl.data import Bounded
try:
    nl = Bounded(low=leaf.space.low[..., :2].clone(),
                 high=leaf.space.high[..., :2].clone(),
                 shape=tuple(leaf.shape[:-1]) + (2,),
                 dtype=leaf.dtype, device=leaf.device)
    print("[new] Bounded ok, shape:", tuple(nl.shape), "low:", nl.space.low, "high:", nl.space.high)
except Exception as e:
    print("[new] Bounded FAIL:", repr(e))

spec2 = spec.clone()
try:
    spec2[("agents", "action")] = nl
    print("[clone+replace] ok ->", tuple(spec2[("agents", "action")].shape))
except Exception as e:
    print("[clone+replace] FAIL:", repr(e))

spec3 = task.action_spec(env)
print("[after] env leaf shape (是否被污染):", tuple(spec3[("agents", "action")].shape))
print("[after] spec2 leaf shape:", tuple(spec2[("agents", "action")].shape))
print("DONE")
