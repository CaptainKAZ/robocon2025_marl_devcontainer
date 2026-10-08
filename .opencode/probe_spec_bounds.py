"""探测原始（未经 FlattenHybridAction 变换）动作 spec 的叶子类型与 bounds。"""

import copy

from benchmarl.environments import LayupTask
from benchmarl.environments.layup.common import VmasEnvWithState
from torchrl.envs.utils import MarlGroupMapType


def dump(name, spec):
    print(f"--- {name} --- type={type(spec).__name__} shape={getattr(spec, 'shape', None)}")
    try:
        print("  top keys:", list(spec.keys()))
    except Exception as e:  # noqa: BLE001
        print("  keys err", e)
    for key in [("agents", "action"), "agents"]:
        try:
            leaf = spec[key]
        except Exception:
            continue
        print(f"  spec[{key!r}] -> {type(leaf).__name__} shape={getattr(leaf, 'shape', None)}")
        space = getattr(leaf, "space", None)
        print(f"     .space={type(space).__name__} low={getattr(space, 'low', None)} high={getattr(space, 'high', None)}")
        print(f"     .low={getattr(leaf, 'low', None)} .high={getattr(leaf, 'high', None)}")
        break


def main():
    task = LayupTask.LAYUP.get_from_yaml()
    cfg = copy.deepcopy(task.config)
    raw = VmasEnvWithState(
        scenario=task.name.lower(),
        num_envs=2,
        continuous_actions=True,
        seed=0,
        device="cpu",
        clamp_actions=True,
        group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
        **cfg,
    )
    dump("raw.action_spec", raw.action_spec)
    try:
        dump("raw.full_action_spec", raw.full_action_spec)
    except Exception as e:  # noqa: BLE001
        print("raw.full_action_spec 不存在:", e)

    env = task.get_env_fun(2, True, 0, "cpu")()
    print("transformed env:", type(env).__name__)
    leaf = env.action_spec[("agents", "action")]
    print("hybrid continuous space:", leaf["continuous"].space)


if __name__ == "__main__":
    main()
