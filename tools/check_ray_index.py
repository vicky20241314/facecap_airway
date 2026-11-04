"""Equivalence check: RayIndex replaces the rtree per-ray query without changing semantics."""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np, trimesh
from pathlib import Path
from head_rgbd.ray_index import RayIndex
from head_rgbd.geometry_c import sample_depth, first_hits

run = Path(r"runtime\outputs\speedup_full_v1")
seq = json.loads((run / "prepared" / "sequence.json").read_text(encoding="utf-8"))
rep = json.loads((run / "reconstruction" / "dense" / "report.json").read_text(encoding="utf-8"))
rows = rep["frames"]
z = np.load(run / "reconstruction" / "dense" / "parameters.npz", allow_pickle=False)
V, F = z["dense_vertices"], z["dense_faces"]
print("mesh", V.shape, F.shape)


def load(path):
    zz = np.load(path, allow_pickle=False)
    return {k: zz[k] for k in zz.files}


rng = np.random.default_rng(20261101)
allpts = []
for idx in (4, 19, 0):
    row = rows[idx]
    d = load(run / "prepared" / next(r["file"] for r in seq["frames"] if r["id"] == row["id"]))
    pts, uv, st = sample_depth(d, 3000, rng, True, stride=3)
    allpts.append(pts)
    print("frame", row["id"], row["view"], "rays", len(pts))
pts = np.concatenate(allpts)

mesh = trimesh.Trimesh(V[4], F, process=False)
t0 = time.perf_counter(); h_ref, r_ref, f_ref = first_hits(mesh, pts); t_ref = time.perf_counter() - t0

t0 = time.perf_counter(); idx = RayIndex(V[4], F); t_build = time.perf_counter() - t0
t0 = time.perf_counter(); h_new, r_new, f_new = idx.first_hits(pts); t_new = time.perf_counter() - t0

print(f"\nrays={len(pts)}  rtree={t_ref*1000:.1f}ms  index_build={t_build*1000:.1f}ms  query={t_new*1000:.1f}ms")
print(f"per ray: rtree {t_ref/len(pts)*1e6:.1f}us  new {(t_build+t_new)/len(pts)*1e6:.1f}us")

ref = {int(r): (int(f), h) for r, f, h in zip(r_ref, f_ref, h_ref)}
new = {int(r): (int(f), h) for r, f, h in zip(r_new, f_new, h_new)}
only_ref = set(ref) - set(new)
only_new = set(new) - set(ref)
common = set(ref) & set(new)
face_same = sum(1 for r in common if ref[r][0] == new[r][0])
dist = np.array([np.linalg.norm(ref[r][1] - new[r][1]) for r in common])
print(f"\nhits: rtree={len(ref)} new={len(new)} common={len(common)}")
print(f"only-in-rtree={len(only_ref)} only-in-new={len(only_new)}")
print(f"same face: {face_same}/{len(common)}")
print(f"hit point diff: max={dist.max()*1e6:.3f} um  mean={dist.mean()*1e6:.3f} um")
if only_ref:
    sample = sorted(only_ref)[:3]
    print("rtree-only ray ids:", sample)
if only_new:
    sample = sorted(only_new)[:3]
    print("new-only ray ids:", sample)
