import torch

CK = "outputs/2026-10-02_15-07-00/mappo_layup_sequencemodel__2ac16b45_26_10_02-15_07_00/checkpoints/checkpoint_24000000.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
print("buffer_agents keys:", list(ck["buffer_agents"].keys()))
st = ck["buffer_agents"]["_storage"]["_storage"]
print("storage type:", type(st).__name__)
print("top keys:", list(st.keys()))


def walk(d, prefix="", depth=0):
    if depth > 2 or not hasattr(d, "keys"):
        return
    for k in d.keys():
        try:
            v = d[k]
        except Exception as e:
            print(f"{prefix}{k}: ERR {e}")
            continue
        if hasattr(v, "keys"):
            print(f"{prefix}{k}: {type(v).__name__} <- {list(v.keys())}")
            walk(v, prefix + "   ", depth + 1)
        else:
            print(f"{prefix}{k}: {type(v).__name__} {tuple(v.shape)} {v.dtype}")


walk(st)
