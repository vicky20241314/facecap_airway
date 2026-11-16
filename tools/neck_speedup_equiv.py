"""Same-input A/B equivalence checks for the neck speed-up kernels.

Every check compares the optimised implementation against the *original*
implementation, which is reproduced here verbatim as the reference.  The
reference lives in this test only; production keeps one code path.

Run:
    python -m tools.neck_speedup_equiv --fit <dense dir> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi          # noqa: E402
from head_rgbd import neck_remesh as nr             # noqa: E402
from head_rgbd import neck_stage as ns              # noqa: E402


# --------------------------------------------------------------------------
# reference implementations (pre-optimisation, copied verbatim)
# --------------------------------------------------------------------------

def ref_edge_records(faces):
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    balance = np.bincount(inverse, weights=signs, minlength=len(edges))
    return edges, inverse, counts, signs, balance


def ref_has_duplicate_face(faces):
    srt = np.sort(np.asarray(faces, np.int64), axis=1)
    _, cnt = np.unique(srt, axis=0, return_counts=True)
    return bool((cnt > 1).any())


def ref_valence(F, n):
    v = np.zeros(n, dtype=np.int64)
    for k in range(3):
        np.add.at(v, F[:, k], 1)
    return v


def ref_section_segments(V, F, ax, h0):
    centre, axis = ax["centre_mm"], ax["axis"]
    e1, e2 = ax["lateral"], ax["anterior"]
    tri = (V[F] * 1000.0 - centre) @ np.column_stack([e1, e2, axis])
    s = tri[:, :, 2] - h0
    lo, hi = s.min(axis=1), s.max(axis=1)
    ids = np.flatnonzero((lo <= 0) & (hi >= 0))
    if not len(ids):
        return np.zeros((0, 2, 2))
    tri = tri[ids][:, :, :2]
    s = s[ids]
    pts = []
    for i in range(3):
        j = (i + 1) % 3
        t0, t1 = s[:, i], s[:, j]
        cross = ((t0 <= 0) & (t1 > 0)) | ((t1 <= 0) & (t0 > 0))
        p = np.full((len(ids), 2), np.nan)
        den = np.where(np.abs(t0 - t1) > 1e-12, t0 - t1, 1.0)
        w = np.where(cross, t0 / den, 0.0)
        p[cross] = tri[cross, i] + w[cross, None] * (tri[cross, j] - tri[cross, i])
        pts.append(p)
    segs = []
    for i in range(3):
        j = (i + 1) % 3
        a, b = pts[i], pts[j]
        ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        if ok.any():
            segs.append(np.stack([a[ok], b[ok]], axis=1))
    return np.vstack(segs) if segs else np.zeros((0, 2, 2))


def ref_radial_field(V, F, ax, h_chin, curve, n_theta=ns.N_THETA,
                     h_lo=0.02, h_hi=1.06):
    hs_ = np.arange(h_lo, h_hi + 1e-9, ns.H_STEP / h_chin) * h_chin
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    R = np.full((len(hs_), n_theta), np.nan)
    for i, h0 in enumerate(hs_):
        ca, cb = curve(float(h0))
        segs = ref_section_segments(V, F, ax, float(h0))
        if len(segs) == 0:
            continue
        R[i] = ns.ray_radius(segs, th, (float(ca), float(cb)))
    for i in range(R.shape[0]):
        row = R[i]
        bad = ~np.isfinite(row)
        if bad.any() and (~bad).sum() > 0.65 * n_theta:
            idx = np.arange(n_theta)
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad], period=n_theta)
            R[i] = row
    return hs_, th, R


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def _rand_mesh(rng, n_v, n_f):
    F = rng.integers(0, n_v, size=(n_f, 3))
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return F[ok]


def check_topology(F, rng, reps=6):
    rows = []
    meshes = [F] + [_rand_mesh(rng, int(F.max()) + 1, len(F) // 4)
                    for _ in range(reps)]
    # a mesh with a deliberately duplicated face and a degenerate face
    dup = np.vstack([F[:200], F[:1]])
    meshes.append(dup)
    for k, M in enumerate(meshes):
        a = mi.edge_records(M)
        b = ref_edge_records(M)
        same = all(np.array_equal(x, y) for x, y in zip(a, b))
        dt = [str(x.dtype) for x in a] == [str(x.dtype) for x in b]
        n = int(M.max()) + 1
        v_ok = np.array_equal(nr.valence(M, n), ref_valence(M, n))
        d_new = mi.has_duplicate_face(M)
        d_ref = ref_has_duplicate_face(M)
        rows.append({"mesh": k, "faces": int(len(M)), "edge_records_identical": bool(same),
                     "dtypes_identical": bool(dt), "valence_identical": bool(v_ok),
                     "dup_new": bool(d_new), "dup_ref": bool(d_ref),
                     "dup_identical": bool(d_new == d_ref)})
    return rows


def ref_edge_map(F):
    F = np.asarray(F)
    d = {}
    for k in range(3):
        i, j = F[:, k], F[:, (k + 1) % 3]
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        for a, b, r in zip(lo.tolist(), hi.tolist(), range(len(F))):
            d.setdefault((a, b), []).append(r)
    return d


def ref_face_max_dihedral(V, F):
    from head_rgbd.neck_bridge import dihedral_deg
    _e, _a = dihedral_deg(V, F)
    _emap = {}
    for kk in range(len(_e)):
        _emap[(int(_e[kk, 0]), int(_e[kk, 1]))] = abs(float(_a[kk]))
    out = np.zeros(len(F), float)
    for r in range(len(F)):
        tri = F[r].tolist()
        out[r] = max(_emap.get((min(tri[s], tri[(s + 1) % 3]),
                                max(tri[s], tri[(s + 1) % 3])), 0.0)
                     for s in range(3))
    return out


def check_edge_map(F, rng, reps=3):
    from head_rgbd.neck_bridge import face_max_dihedral_deg
    V = rng.normal(size=(int(F.max()) + 1, 3))
    rows = []
    for k, M in enumerate([F] + [_rand_mesh(rng, int(F.max()) + 1, len(F) // 3)
                                 for _ in range(reps)]):
        t0 = time.perf_counter()
        new = nr.edge_map(M)
        t_new = time.perf_counter() - t0
        t0 = time.perf_counter()
        ref = ref_edge_map(M)
        t_ref = time.perf_counter() - t0
        # dict order is not part of the contract; the per-edge row order is
        same_keys = set(new) == set(ref)
        same_rows = same_keys and all(new[k] == ref[k] for k in ref)
        Vm = rng.normal(size=(int(M.max()) + 1, 3))
        t0 = time.perf_counter()
        dn = face_max_dihedral_deg(Vm, M)
        t_dn = time.perf_counter() - t0
        t0 = time.perf_counter()
        dr = ref_face_max_dihedral(Vm, M)
        t_dr = time.perf_counter() - t0
        rows.append({"mesh": k, "faces": int(len(M)),
                     "edge_map_keys_identical": bool(same_keys),
                     "edge_map_row_order_identical": bool(same_rows),
                     "edge_map_s_new": round(t_new, 4),
                     "edge_map_s_ref": round(t_ref, 4),
                     "face_dihedral_bit_identical": bool(np.array_equal(dn, dr)),
                     "face_dihedral_max_diff": float(np.abs(dn - dr).max()),
                     "face_dihedral_s_new": round(t_dn, 4),
                     "face_dihedral_s_ref": round(t_dr, 4)})
    return rows


def check_sections(V, F, ax, h_chin, curve):
    tri = ns.section_triangles(V, F, ax)
    levels = np.linspace(0.05, 1.0, 9) * h_chin
    seg_max = 0.0
    seg_identical = True
    t_new = t_ref = 0.0
    for h0 in levels:
        t0 = time.perf_counter()
        a = ns.section_segments(V, F, ax, float(h0), tri=tri)
        t_new += time.perf_counter() - t0
        t0 = time.perf_counter()
        b = ref_section_segments(V, F, ax, float(h0))
        t_ref += time.perf_counter() - t0
        seg_identical &= bool(a.shape == b.shape and np.array_equal(a, b))
        if a.shape == b.shape and len(a):
            seg_max = max(seg_max, float(np.abs(a - b).max()))
    t0 = time.perf_counter()
    hn, tn, Rn = ns.radial_field(V, F, ax, h_chin, curve)
    t_new_field = time.perf_counter() - t0
    t0 = time.perf_counter()
    hr, tr, Rr = ref_radial_field(V, F, ax, h_chin, curve)
    t_ref_field = time.perf_counter() - t0
    finite = np.isfinite(Rr)
    return {
        "levels": len(levels),
        "segments_bit_identical": bool(seg_identical),
        "segments_max_abs_diff": seg_max,
        "segments_s_new": round(t_new, 4), "segments_s_ref": round(t_ref, 4),
        "field_shape_identical": bool(Rn.shape == Rr.shape),
        "field_nan_pattern_identical": bool(np.array_equal(np.isfinite(Rn), finite)),
        "field_bit_identical": bool(np.array_equal(Rn[finite], Rr[finite])),
        "field_max_abs_diff_mm": (float(np.abs(Rn[finite] - Rr[finite]).max())
                                  if finite.any() else 0.0),
        "field_s_new": round(t_new_field, 4), "field_s_ref": round(t_ref_field, 4),
        "field_speedup": (round(t_ref_field / t_new_field, 2) if t_new_field else None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)

    from tools import build_neck_stage as bns
    states_raw, F, _refs = bns.load_subject(args.fit)
    V = np.asarray(states_raw["closed"]["V"], float)
    F = np.asarray(F, np.int64)
    print(f"input {len(V)} vertices / {len(F)} faces", flush=True)

    topo = check_topology(F, rng)
    print(json.dumps(topo, indent=1), flush=True)

    emap = check_edge_map(F, rng)
    print(json.dumps(emap, indent=1), flush=True)

    _frame, ax, h_chin = ns.subject_frame(V, F, states_raw["closed"]["landmarks"])
    curve, _spine = ns.spine_curve(V, F, ax, h_chin)
    sec = check_sections(V, F, ax, h_chin, curve)
    print(json.dumps(sec, indent=1), flush=True)

    ok = (all(r["edge_records_identical"] and r["valence_identical"]
              and r["dup_identical"] and r["dtypes_identical"] for r in topo)
          and all(r["edge_map_keys_identical"] and r["edge_map_row_order_identical"]
                  and r["face_dihedral_bit_identical"] for r in emap)
          and sec["segments_bit_identical"] and sec["field_bit_identical"]
          and sec["field_nan_pattern_identical"])
    payload = {"schema": "neck_speedup.equiv.v1", "fit": str(args.fit),
               "topology": topo, "edge_map": emap, "sections": sec, "pass": bool(ok)}
    (args.out / "equiv.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())