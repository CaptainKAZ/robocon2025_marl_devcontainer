import glob, torch, os
cks = sorted(glob.glob("/home/vscode/workspace/BenchMARL/outputs/2026-10-03_14-12-27/*/checkpoints/checkpoint_*.pt"),
             key=os.path.getmtime)
ck_path = cks[-1]
print("ckpt =", ck_path)
ck = torch.load(ck_path, map_location="cpu", mmap=True, weights_only=False)
st = ck["buffer_agents"]["_storage"]["_storage"]

def walk(obj, prefix="", depth=0):
    if depth > 3: return
    try:
        keys = list(obj.keys())
    except Exception:
        return
    for k in keys:
        try:
            v = obj.get(k) if hasattr(obj, "get") else obj[k]
        except Exception as e:
            print(f"{prefix}{k}: ERR {e}"); continue
        if hasattr(v, "keys"):
            print(f"{prefix}{k}: <td keys={len(list(v.keys()))}>")
            walk(v, prefix + "  ", depth + 1)
        else:
            shp = tuple(v.shape) if hasattr(v, "shape") else None
            print(f"{prefix}{k}: {type(v).__name__} {shp}")

print("---- 顶层 ----")
print(list(st.keys()))
print("---- next 子树 ----")
walk(st.get("next"), "  ")
print("---- 是否有 termination_reason ----")
for path in [("next","agents","info","termination_reason"), ("agents","info","termination_reason"),
             ("next","agents","info"), ("next","info","termination_reason")]:
    cur = st
    ok = True
    for k in path:
        try:
            cur = cur.get(k) if hasattr(cur,"get") else cur[k]
        except Exception:
            ok = False; break
        if cur is None: ok = False; break
    print(path, "->", (tuple(cur.shape), cur.dtype) if ok and hasattr(cur,"shape") else ("FOUND" if ok else "missing"))
