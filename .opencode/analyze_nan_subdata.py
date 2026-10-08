import torch

PATH = "/tmp/nan_case_iter204_agents_loss_objective.pt"

td = torch.load(PATH, map_location="cpu", weights_only=False)


def walk(node, prefix=""):
    try:
        keys = list(node.keys())
    except AttributeError:
        return
    for k in keys:
        try:
            v = node.get(k)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(v, "keys"):
            walk(v, prefix + str(k) + "/")
        elif torch.is_tensor(v):
            fin = torch.isfinite(v)
            nbad = int((~fin).sum())
            if v.is_floating_point() and v.numel():
                fv = v[fin]
                mn = float(fv.min()) if fv.numel() else float("nan")
                mx = float(fv.max()) if fv.numel() else float("nan")
                print(f"{prefix}{k}: {tuple(v.shape)} {v.dtype} nonfinite={nbad} min={mn:.5g} max={mx:.5g}")
            else:
                print(f"{prefix}{k}: {tuple(v.shape)} {v.dtype} nonfinite={nbad}")


print("=== 全部叶子 ===")
walk(td)

print()
print("=== 关键量分布 ===")
for key in [
    ("agents", "log_prob"),
    ("agents", "sample_log_prob"),
    ("agents", "advantage"),
    ("agents", "value_target"),
    ("agents", "action"),
    ("agents", "loc"),
    ("agents", "scale"),
    ("agents", "logits"),
]:
    try:
        v = td.get(key)
    except Exception:  # noqa: BLE001
        v = None
    if v is None:
        print(f"{key}: <缺失>")
        continue
    if hasattr(v, "keys"):
        for sub in v.keys():
            t = v.get(sub)
            if torch.is_tensor(t) and t.is_floating_point() and t.numel():
                q = torch.quantile(t.flatten().float(), torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99, 1.0]))
                print(f"{key}.{sub}: {tuple(t.shape)} q0={q[0]:.4g} q0.1%={q[1]:.4g} q1%={q[2]:.4g} q50={q[3]:.4g} q99={q[4]:.4g} q100={q[5]:.4g}")
    elif torch.is_tensor(v) and v.numel():
        q = torch.quantile(v.flatten().float(), torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99, 1.0]))
        print(f"{key}: {tuple(v.shape)} q0={q[0]:.4g} q0.1%={q[1]:.4g} q1%={q[2]:.4g} q50={q[3]:.4g} q99={q[4]:.4g} q100={q[5]:.4g}")
