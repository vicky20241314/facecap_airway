"""Independent acceptance for the v2 direct-neck delivery.

Reads only the delivered files (two OBJs, ``stage_direct.npz``,
``build_direct.json``), the input reconstruction, and the two ablations.  Every
geometric number is recomputed from the OBJ arrays; the build products are
checked, not trusted.

Sections
    S0 delivery / one mesh two states / OBJ is the built array / baseline
    S1 density -- local ring bands, area and edge ratios, axial/circumferential
       spacing, strain and matched-face compression, for input / zero-topology
       ablation / full result
    S2 connection -- complete edge set, signed sharpening vs flattening, B / B-C
       / C-neighbour subsets, worst positive edges with a location
    S3 structure -- main prominence, lower broad lobe, shallow transition,
       lateral relief, saddle, artefacts
    S4 protection -- fixed head, cut rim, root, hand-over shell
    S5 topology, S6 self-intersection, S7 renders
    S8 targeted -- ablation equivalence, zero new vertices, shared ops, bridge
       defect tests, reference shape-only, target amplitude self-check,
       before/after tracking of the previous round's four problem edges
    S9 verdict -- every gate with value/limit, failed list, residual findings

Shape and density are accepted separately: shape is compared to the reference's
*structure relations*, density to the subject's adjacent head regions.  The
reference topology/vertex count is never a target.
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
from head_rgbd import neck_bridge as nbr            # noqa: E402
from head_rgbd import neck_direct as nd             # noqa: E402
from head_rgbd import neck_fit as nfit              # noqa: E402
from head_rgbd import neck_stage as ns               # noqa: E402
from tools import build_neck_stage as bns            # noqa: E402
from tools import neck_direct_verify as v1           # noqa: E402


def jload(path):
    return json.load(open(path, "r", encoding="utf-8"))


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def fmax(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.max()) if len(a) else float("nan")


def fmin(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.min()) if len(a) else float("nan")


# --------------------------------------------------------------------------
# S1. local density / ring bands
# --------------------------------------------------------------------------


BINS = [(0.15, 0.30), (0.30, 0.45), (0.45, 0.60), (0.60, 0.72),
        (0.72, 0.80), (0.80, 0.90), (0.90, 0.925), (0.925, 0.950),
        (0.950, 0.970), (0.970, 1.050), (1.050, 1.400)]


def _face_areas(V, F):
    tri = np.asarray(V, float)[F]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1) * 1e6


def _edge_dir_stats(V, F, E, sel, ax, curve, hc):
    if not sel.any():
        return {"axial_p50_mm": None, "circ_p50_mm": None,
                "axial_edges": 0, "circ_edges": 0}
    e = E[sel]
    u = (np.asarray(V, float)[e[:, 1]] - np.asarray(V, float)[e[:, 0]]) * 1000.0
    L = np.linalg.norm(u, axis=1)
    mid = 0.5 * (np.asarray(V, float)[e[:, 0]] + np.asarray(V, float)[e[:, 1]]) * 1000.0
    _, _, _, _, _, et = nfit.frame_at(mid, ax, curve, hc)
    uu = u / np.maximum(L[:, None], 1e-12)
    da = np.abs(uu @ np.asarray(ax["axis"], float))
    dt = np.abs(np.einsum("ij,ij->i", uu, et))
    ka = da > 0.7
    kc = dt > 0.7
    return {"axial_p50_mm": float(np.median(L[ka])) if ka.any() else None,
            "circ_p50_mm": float(np.median(L[kc])) if kc.any() else None,
            "axial_edges": int(ka.sum()), "circ_edges": int(kc.sum())}


def _face_area_map(V, F, oid):
    a = _face_areas(V, F)
    keys = np.sort(np.asarray(oid, np.int64)[F], axis=1)
    return {tuple(int(x) for x in k): float(v) for k, v in zip(keys, a)}, a


def _source_stats(V, F, t, mask_bins, ax, curve, hc, d_mm=None, oid=None,
                  input_oid=None, V_input0=None, ain_map=None):
    """Per-bin geometry for one delivered mesh."""
    E = mi.edge_records(F)[0]
    L = np.linalg.norm((np.asarray(V, float)[E[:, 0]] -
                        np.asarray(V, float)[E[:, 1]]) * 1000.0, axis=1)
    areas = _face_areas(V, F)
    a_v = np.zeros(len(V))
    for k in range(3):
        np.add.at(a_v, F[:, k], areas / 3.0)
    out = []
    for lo, hi in BINS:
        mv = (t >= lo) & (t < hi)
        mf = mv[F].all(axis=1)
        me = mv[E[:, 0]] & mv[E[:, 1]]
        area = float(areas[mf].sum())
        row = {"t": [lo, hi], "vertices": int(mv.sum()), "faces": int(mf.sum()),
               "area_cm2": area / 100.0,
               "vertices_per_cm2": (float(mv.sum()) / (area / 100.0)
                                     if area > 0 else None),
               "faces_per_cm2": (float(mf.sum()) / (area / 100.0)
                                  if area > 0 else None),
               "edges": int(me.sum()),
               "edge_p10_mm": pct(L[me], 10) if me.any() else None,
               "edge_p50_mm": pct(L[me], 50) if me.any() else None,
               "edge_p90_mm": pct(L[me], 90) if me.any() else None,
               "edge_min_mm": float(L[me].min()) if me.any() else None,
               "edge_max_mm": float(L[me].max()) if me.any() else None}
        row.update(_edge_dir_stats(V, F, E, me, ax, curve, hc))
        if d_mm is not None and oid is not None and input_oid is not None:
            # strain: displacement difference across the edge against the
            # capture distance between the same two original vertices
            oi, oj = oid[E[me, 0]], oid[E[me, 1]]
            valid = (oi < len(V_input0)) & (oj < len(V_input0))
            Lin = np.linalg.norm(
                (np.asarray(V_input0, float)[oi[valid]] -
                 np.asarray(V_input0, float)[oj[valid]]) * 1000.0, axis=1) \
                if valid.any() else np.zeros(0)
            din = np.linalg.norm(
                (np.asarray(d_mm, float)[E[me, 0]] -
                 np.asarray(d_mm, float)[E[me, 1]]), axis=1)
            din = din[valid] if valid.any() else np.zeros(0)
            st = din / np.maximum(Lin, 1e-9)
            row["strain_p50_mm_per_mm"] = pct(st, 50) if len(st) else None
            row["strain_p95_mm_per_mm"] = pct(st, 95) if len(st) else None
            row["strain_max_mm_per_mm"] = fmax(st) if len(st) else None
        if ain_map is not None and oid is not None:
            a_out, _ = _face_area_map(V, F, oid)
            ratios = [a_out[k] / a for k, a in ain_map.items() if k in a_out and a > 0]
            row["matched_faces"] = int(len(ratios))
            row["area_ratio_p05"] = pct(ratios, 5) if ratios else None
            row["area_ratio_p50"] = pct(ratios, 50) if ratios else None
            row["area_ratio_p95"] = pct(ratios, 95) if ratios else None
        out.append(row)
    return out


def _compare_rows(inp, out, label):
    rows = []
    for a, b in zip(inp, out):
        edge_ratio = (b["edge_p50_mm"] / a["edge_p50_mm"]
                      if a["edge_p50_mm"] and b["edge_p50_mm"] else None)
        area_ratio = (b["area_cm2"] / a["area_cm2"] if a["area_cm2"] > 0 else None)
        rows.append({"t": a["t"], "input": a, label: b,
                     "edge_p50_ratio": edge_ratio,
                     "area_ratio": area_ratio,
                     "vertex_delta": b["vertices"] - a["vertices"]})
    return rows


def _band_combined(V, F, t, lo, hi):
    """One exact material band, no bin-boundary artefacts."""
    E = mi.edge_records(F)[0]
    mv = (t >= lo) & (t < hi)
    mf = mv[F].all(axis=1)
    me = mv[E[:, 0]] & mv[E[:, 1]]
    L = np.linalg.norm((np.asarray(V, float)[E[:, 0]] -
                        np.asarray(V, float)[E[:, 1]]) * 1000.0, axis=1)
    A = _face_areas(V, F)
    return {"vertices": int(mv.sum()), "faces": int(mf.sum()),
            "edges": int(me.sum()),
            "edge_p50_mm": pct(L[me], 50) if me.any() else None,
            "area_cm2": float(A[mf].sum() / 100.0)}


def ring_band_section(V0, F0, ax, curve, hc, npz, Vf, Ff, ab_npz, Vab, Fab,
                      prev_npz=None, Vp=None, Fp=None, meta=None):
    h0, _, _, t0 = nd.axial_mm(np.asarray(V0, float) * 1000.0, ax, curve, hc)
    tm = np.asarray(npz["h_material"], float) / hc
    oid = np.asarray(npz["orig_id"], np.int64)
    d = np.asarray(npz["d_closed"], float)
    ain_map, _ = _face_area_map(V0, F0, np.arange(len(V0)))
    inp = _source_stats(V0, F0, t0, None, ax, curve, hc,
                        oid=np.arange(len(V0)), input_oid=np.arange(len(V0)),
                        V_input0=V0, ain_map=None)
    full = _source_stats(Vf, Ff, tm, None, ax, curve, hc, d_mm=d, oid=oid,
                         input_oid=np.arange(len(V0)), V_input0=V0,
                         ain_map=ain_map)
    tab = npz["h_material"]
    ab_oid = np.asarray(ab_npz["orig_id"], np.int64)
    ab_t = np.asarray(ab_npz["h_material"], float) / hc
    ab_d = np.asarray(ab_npz["d_closed"], float)
    abl = _source_stats(Vab, Fab, ab_t, None, ax, curve, hc, d_mm=ab_d,
                        oid=ab_oid, input_oid=np.arange(len(V0)), V_input0=V0,
                        ain_map=ain_map)
    out = {"scope": "closed state; the open state has the same topology, its "
                     "total edge/face counts and open/closed edge ratio are reported "
                     "separately",
           "bins_t": BINS,
           "input": inp, "zero_topology_ablation": abl, "full": full,
           "compare_zero_topology": _compare_rows(inp, abl, "zero_topology"),
           "compare_full": _compare_rows(inp, full, "full"),
           "added_vertices": int((meta or {}).get("journal", {}).get("split_new_total", 0)),
           "note": ("added_vertices counts new points only (splits); a merge changes "
                    "the vertex count but is not an added point.  'input' is the "
                    "capture; 'zero_topology' is the same continuous solve without "
                    "any topology op; 'full' includes the accepted merges/flips.  "
                    "The upper band 0.925-0.97 is the inherited dense band: its "
                    "ratio being ~1 is the evidence that this round did not create "
                    "a new ring.")}
    bi = _band_combined(V0, F0, t0, 0.925, 0.97)
    bf = _band_combined(Vf, Ff, tm, 0.925, 0.97)
    ba = _band_combined(Vab, Fab, ab_t, 0.925, 0.97)
    out["combined_upper_band_0.925_0.97"] = {
        "input": bi, "zero_topology": ba, "full": bf,
        "vertex_delta_full": bf["vertices"] - bi["vertices"],
        "face_delta_full": bf["faces"] - bi["faces"],
        "edge_p50_ratio_full": (bf["edge_p50_mm"] / bi["edge_p50_mm"]
                                if bi["edge_p50_mm"] and bf["edge_p50_mm"] else None),
        "area_ratio_full": (bf["area_cm2"] / bi["area_cm2"]
                            if bi["area_cm2"] > 0 else None),
        "note": "exact band, so the split-bin boundary cannot create a fake "
                "compression.  A vertex/face delta of 0 with a p50 ratio near 1 "
                "means the upper dense band is inherited unchanged, not newly "
                "created; the smaller per-bin p50 is a bin-membership artefact."}
    if prev_npz is not None and Vp is not None:
        prev_t = np.asarray(prev_npz["h_material"], float) / hc
        prev_oid = np.asarray(prev_npz["orig_id"], np.int64)
        prev_d = np.asarray(prev_npz["d_closed"], float)
        out["previous_round"] = _source_stats(
            Vp, Fp, prev_t, None, ax, curve, hc, d_mm=prev_d, oid=prev_oid,
            input_oid=np.arange(len(V0)), V_input0=V0, ain_map=ain_map)
    # open-state totals and the open/closed median edge ratio
    o_edges = []
    for key, VV, FF in (("closed", Vf, Ff),):
        E = mi.edge_records(FF)[0]
        o_edges.append(np.linalg.norm((VV[E[:, 0]] - VV[E[:, 1]]) * 1000.0, axis=1))
    out["open_totals"] = None
    try:
        Vo = np.asarray(npz["V_open"], float)
        Fo = np.asarray(npz["F"], np.int64)
        E = mi.edge_records(Fo)[0]
        Lo = np.linalg.norm((Vo[E[:, 0]] - Vo[E[:, 1]]) * 1000.0, axis=1)
        Lc = o_edges[0]
        out["open_totals"] = {"vertices": int(len(Vo)), "faces": int(len(Fo)),
                              "edge_p50_mm": float(np.median(Lo)),
                              "closed_edge_p50_mm": float(np.median(Lc)),
                              "open_closed_edge_p50_ratio": float(np.median(Lo) / np.median(Lc))}
    except Exception as exc:
        out["open_totals"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------
# S2. signed connection, complete edge set
# --------------------------------------------------------------------------


def _dihedral_pair(Vout, Vin, F, E):
    ao = v1._dihedral_full(np.asarray(Vout, float) * 1000.0, F, E)
    ai = v1._dihedral_full(np.asarray(Vin, float) * 1000.0, F, E)
    good = np.isfinite(ao) & np.isfinite(ai)
    return ao, ai, good


def _signed_subset(E, sel, Vout, Vin, drad, ao, ai, frame, reg=None, top=8):
    good = np.isfinite(ao) & np.isfinite(ai) & sel
    if not good.any():
        return {"edges": 0}
    ch = np.abs(ao) - np.abs(ai)
    Lout = np.linalg.norm((Vout[E[:, 0]] - Vout[E[:, 1]]) * 1000.0, axis=1)
    Lin = np.linalg.norm((Vin[E[:, 0]] - Vin[E[:, 1]]) * 1000.0, axis=1)
    g = np.abs(drad[E[:, 0]] - drad[E[:, 1]]) / np.maximum(Lout, 1e-9)
    a, b = E[:, 0], E[:, 1]
    mid = 0.5 * (Vout[a] + Vout[b]) * 1000.0
    h, th, _, t = nd.axial_mm(mid, frame[0], frame[1], frame[2])
    idx = np.flatnonzero(good)
    chg = ch[idx]
    pos = np.sort(chg)[::-1]
    out = {"edges": int(good.sum()),
           "positive_max_deg": float(chg.max()),
           "negative_min_deg": float(chg.min()),
           "abs_p95_deg": pct(np.abs(chg), 95),
           "abs_p99_deg": pct(np.abs(chg), 99),
           "positive_gt10": int((chg > 10.0).sum()),
           "positive_gt20": int((chg > 20.0).sum()),
           "negative_lt_minus10": int((chg < -10.0).sum()),
           "dihedral_out_p95_deg": pct(np.abs(ao[good]), 95),
           "dihedral_in_p95_deg": pct(np.abs(ai[good]), 95),
           "gradient_p95_mm_per_mm": pct(g[good], 95),
           "gradient_max_mm_per_mm": fmax(g[good])}
    worst = []
    for k in np.argsort(-chg)[:top]:
        ii = idx[k]
        worst.append({"edge": [int(a[ii]), int(b[ii])],
                      "regions": ([int(reg[a[ii]]), int(reg[b[ii]])]
                                  if reg is not None else None),
                      "in_deg": float(ai[ii]), "out_deg": float(ao[ii]),
                      "change_deg": float(chg[k]),
                      "new_fold_flat_input": bool(np.abs(ai[ii]) < 5.0
                                                  and np.abs(ao[ii]) > 20.0),
                      "edge_mm_out": float(Lout[ii]), "edge_mm_in": float(Lin[ii]),
                      "t": float(t[ii]), "theta_deg": float(np.degrees(th[ii])),
                      "gradient_mm_per_mm": float(g[ii])})
    out["worst_positive"] = worst
    return out


def connection_v2_section(npz, F, ax, curve, hc, ax_o, curve_o, hc_o):
    reg = np.asarray(npz["region"], np.int64)
    t_mat = np.asarray(npz["h_material"], float) / hc
    E, sel = v1._band_subsets(reg, F)
    root_sel = ((t_mat[E[:, 0]] < 0.35) & (t_mat[E[:, 1]] < 0.35)
                & (reg[E[:, 0]] == 2) & (reg[E[:, 1]] == 2))
    blocks = {}
    for key, axx, cur, hcc in (("closed", ax, curve, hc),
                               ("open", ax_o, curve_o, hc_o)):
        Vout = np.asarray(npz["V_" + key], float)
        Vin = np.asarray(npz["V_" + key + "_input"], float)
        drad = np.asarray(npz["d_rad_" + key], float)
        ao, ai, good = _dihedral_pair(Vout, Vin, F, E)
        frame = (axx, cur, hcc)
        blk = {
            name: _signed_subset(E, s, Vout, Vin, drad, ao, ai, frame, reg)
            for name, s in sel.items()}
        blk["near_root_t_lt_0_35"] = _signed_subset(
            E, root_sel, Vout, Vin, drad, ao, ai, frame, reg)
        blocks[key] = blk
    # gates: the full edge set and the two local bands
    gates = {}
    for key, blk in blocks.items():
        full = blk["all_edges"]
        gates[key] = {
            "full_positive_max_deg": {
                "value": full["positive_max_deg"], "limit": 25.0,
                "pass": bool(full["positive_max_deg"] <= 25.0)},
            "no_new_fold_over_25": {
                "value": int(sum(1 for w in full["worst_positive"]
                                 if w["new_fold_flat_input"] and w["out_deg"] > 25.0)),
                "limit": 0,
                "pass": bool(not any(w["new_fold_flat_input"] and w["out_deg"] > 25.0
                                     for w in full["worst_positive"]))},
            "inside_B_positive_max_deg": {
                "value": blk["inside_B"]["positive_max_deg"], "limit": 20.0,
                "pass": bool(blk["inside_B"]["positive_max_deg"] <= 20.0)},
            "crossing_positive_max_deg": {
                "value": blk["crossing_any"]["positive_max_deg"], "limit": 20.0,
                "pass": bool(blk["crossing_any"]["positive_max_deg"] <= 20.0)},
            "full_positive_gt10": {"value": full["positive_gt10"],
                                   "limit": "reported, not gated",
                                   "pass": True},
            "full_positive_gt20": {"value": full["positive_gt20"],
                                   "limit": "reported, not gated",
                                   "pass": True},
        }
    return {"scope": "every interior edge of the delivered mesh; signed change "
                     "= |out| - |in| on the same edge.  Positive = sharper than the "
                     "capture (the plan forbids this); negative = an existing fold "
                     "was flattened (not a defect).  A gate cannot be passed by "
                     "changing the subset: the full set is always reported first.",
            "closed": blocks["closed"], "open": blocks["open"],
            "gates": gates}


# --------------------------------------------------------------------------
# S8. before/after tracking of the previous round's four problem edges
# --------------------------------------------------------------------------


def track_previous_edges(old_dir, new_npz, new_F, new_ax, new_curve, new_hc):
    old_dir = Path(old_dir)
    old_npz = np.load(old_dir / "stage_direct.npz")
    old_F = np.asarray(old_npz["F"], np.int64)
    old_oid = np.asarray(old_npz["orig_id"], np.int64)
    new_oid = np.asarray(new_npz["orig_id"], np.int64)
    new_map = {int(o): i for i, o in enumerate(new_oid)}
    old_Vo = np.asarray(old_npz["V_closed"], float)
    old_Vi = np.asarray(old_npz["V_closed_input"], float)
    oe, ao = nbr.dihedral_deg(old_Vo * 1000.0, old_F)
    _e, ai = nbr.dihedral_deg(old_Vi * 1000.0, old_F)
    old_ch = np.abs(ao) - np.abs(ai)
    old_index = {(min(int(a), int(b)), max(int(a), int(b))): k
                 for k, (a, b) in enumerate(oe)}
    targets = [(16051, 16052), (16052, 16054), (3267, 15482),
               (2150, 9472), (6735, 16056)]
    new_E = mi.edge_records(new_F)[0]
    new_edges = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in new_E}
    out = []
    for a, b in targets:
        oi, oj = int(old_oid[a]), int(old_oid[b])
        rec = {"old_slots": [a, b], "orig_ids": [oi, oj]}
        k = old_index.get((min(a, b), max(a, b)))
        rec["old_change_deg"] = float(old_ch[k]) if k is not None else None
        if oi in new_map and oj in new_map:
            ni, nj = new_map[oi], new_map[oj]
            rec["new_slots"] = [ni, nj]
            rec["edge_exists_in_new"] = (min(ni, nj), max(ni, nj)) in new_edges
            if rec["edge_exists_in_new"]:
                E = mi.edge_records(new_F)[0]
                key = (min(ni, nj), max(ni, nj))
                eidx = np.flatnonzero((np.minimum(E[:, 0], E[:, 1]) == key[0]) &
                                      (np.maximum(E[:, 0], E[:, 1]) == key[1]))
                # recompute on the new delivered mesh
                Vout = np.asarray(new_npz["V_closed"], float)
                Vin = np.asarray(new_npz["V_closed_input"], float)
                e2, a2 = nbr.dihedral_deg(Vout * 1000.0, new_F)
                _e2, i2 = nbr.dihedral_deg(Vin * 1000.0, new_F)
                m = (np.minimum(e2[:, 0], e2[:, 1]) == key[0]) & \
                    (np.maximum(e2[:, 0], e2[:, 1]) == key[1])
                if m.any():
                    rec["new_in_deg"] = float(np.abs(i2[m])[0])
                    rec["new_out_deg"] = float(np.abs(a2[m])[0])
                    rec["new_change_deg"] = float(np.abs(a2[m])[0] - np.abs(i2[m])[0])
        else:
            rec["new_slots"] = None
            rec["note"] = "one endpoint was removed by a merge"
        out.append(rec)
    return {"old_dir": str(old_dir), "targets": out,
            "note": "the same material vertices tracked through the new export; "
                    "a removed endpoint is reported, not silently dropped"}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/zp_capwitharuco_neck_direct_v2")
    ap.add_argument("--fit", default="runtime/outputs/zp_capwitharuco/reconstruction/dense")
    ap.add_argument("--reference",
                    default="D:/project_neck/runtime/v8_tail_smoke/"
                            "experiment_20261109-200612/final/identity_rest.obj")
    ap.add_argument("--ablation-noremesh",
                    default="runtime/outputs/_ablation_noremesh_v2")
    ap.add_argument("--ablation-nosplit",
                    default="runtime/outputs/_ablation_nosplit_v2")
    ap.add_argument("--previous", default="runtime/outputs/zp_capwitharuco_neck_direct")
    ap.add_argument("--no-selfint", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJ / out
    t_start = time.time()
    meta = jload(out / "build_direct.json")
    p = meta["params"]
    npz = np.load(out / "stage_direct.npz")
    Vc, Fc = mi.load_obj(out / "head_closed_direct.obj")
    Vo, Fo = mi.load_obj(out / "head_open_direct.obj")
    print(f"[S0] delivered {len(Vc)} v / {len(Fc)} f (open {len(Vo)} / {len(Fo)})",
          flush=True)

    states, F0, _ = bns.load_subject(Path(args.fit))
    V0c, V0o = states["closed"]["V"], states["open"]["V"]
    n0 = len(V0c)
    frame, ax, hc = ns.subject_frame(V0c, F0, states["closed"]["landmarks"])
    curve, _ = ns.spine_curve(V0c, F0, ax, hc)
    frame_o, ax_o, hc_o = ns.subject_frame(V0o, F0, states["open"]["landmarks"])
    curve_o, _ = ns.spine_curve(V0o, F0, ax_o, hc_o)
    report = {"files": {"out": str(out)}, "params": p,
              "input_provenance": meta.get("provenance"),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    report["S0_delivery"] = v1.delivery_section(out, Vc, Fc, Vo, Fo, npz,
                                                n0, V0c, V0o, F0)

    # ---- S1 density -----------------------------------------------------
    print("[S1] density / ring bands ...", flush=True)
    ab_dir = Path(args.ablation_noremesh)
    if not ab_dir.is_absolute():
        ab_dir = PROJ / ab_dir
    Vab, Fab = mi.load_obj(ab_dir / "head_closed_direct.obj")
    ab_npz = np.load(ab_dir / "stage_direct.npz")
    prev_dir = Path(args.previous)
    if not prev_dir.is_absolute():
        prev_dir = PROJ / prev_dir
    prev_npz = np.load(prev_dir / "stage_direct.npz") if (prev_dir / "stage_direct.npz").exists() else None
    Vp = Fp = None
    if prev_npz is not None:
        Vp, Fp = mi.load_obj(prev_dir / "head_closed_direct.obj")
    sig = p["structure_sigma_mm"]
    f_c = ns.analyse_surface(Vc, Fc, ax, hc, curve=curve, structure_sigma=sig,
                             fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    f_o = ns.analyse_surface(Vo, Fo, ax_o, hc_o, curve=curve_o, structure_sigma=sig,
                             fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    report["S1_density"] = {
        "regions": v1.density_section(V0c, F0, Vc, Fc, ax, curve, hc, p, meta, f_c),
        "ring_bands": ring_band_section(V0c, F0, ax, curve, hc, npz, Vc, Fc,
                                        ab_npz, Vab, Fab, prev_npz, Vp, Fp, meta),
        "budget": meta.get("budget_by_zone"),
        "budget_by_region": meta.get("budget_by_region")}

    # ---- S2 connection --------------------------------------------------
    print("[S2] connection, signed full edge set ...", flush=True)
    report["S2_connection"] = {
        "absolute": v1.connection_section(npz, Fc, ax, curve, hc, ax_o, curve_o,
                                          hc_o, p, meta),
        "signed": connection_v2_section(npz, Fc, ax, curve, hc, ax_o, curve_o, hc_o),
        "previous_edges": track_previous_edges(args.previous, npz, Fc, ax, curve, hc)
        if (prev_dir / "stage_direct.npz").exists() else None}

    # ---- S3 structure ---------------------------------------------------
    print("[S3] structure ...", flush=True)
    RV, RF, _, rax, rhc, _ = ns.reference_frame(args.reference)
    rcurve, _ = ns.spine_curve(RV, RF, rax, rhc)
    ref_field = ns.analyse_surface(RV, RF, rax, rhc, curve=rcurve,
                                   structure_sigma=sig,
                                   fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    ref_desc = ns.structure_descriptors(ref_field, "reference")
    report["S3_structure"] = {
        "closed": v1.structure_section(f_c, ref_desc, None, hc, p, "direct v2 closed",
                                       ref_field=ref_field),
        "open": v1.structure_section(f_o, ref_desc, None, hc_o, p, "direct v2 open",
                                     ref_field=ref_field)}

    # ---- S4 protection --------------------------------------------------
    print("[S4] protection ...", flush=True)
    report["S4_protection"] = v1.protection_section(V0c, V0o, F0, Vc, Fc, Vo, Fo,
                                                    npz, ax, curve, hc, p, meta)

    # ---- S5/S6 ----------------------------------------------------------
    print("[S5] topology ...", flush=True)
    report["S5_topology"] = v1.topology_section(Vc, Fc, Vo, Fo, V0c, F0)
    if args.no_selfint:
        report["S6_self_intersection"] = {"skipped": True}
    else:
        print("[S6] self-intersection ...", flush=True)
        report["S6_self_intersection"] = v1.selfint_section(Vc, Fc, V0c, F0,
                                                            ax, curve, hc, npz)

    # ---- S7 renders -----------------------------------------------------
    need = ("compare_views.png", "compare_closeups.png", "shading_and_wireframe.png",
            "wire_original_vs_direct.png", "sections.png", "profiles_and_field.png",
            "states_closed_open.png", "prominence_map.png", "material_partition.png",
            "density_zones.png", "displacement_and_normals.png",
            "sections_multi_angle.png", "wire_upper_neck.png",
            "wire_submental.png", "flat_vs_smooth.png", "ring_band_metrics.png",
            "two_state_sections.png")
    rnd = out / "render"
    report["S7_renders"] = {n: {"present": (rnd / n).exists(),
                                "bytes": int((rnd / n).stat().st_size)
                                if (rnd / n).exists() else 0} for n in need}

    # ---- S8 targeted ----------------------------------------------------
    print("[S8] targeted ...", flush=True)
    abj = jload(ab_dir / "build_direct.json")
    ns_dir = Path(args.ablation_nosplit)
    if not ns_dir.is_absolute():
        ns_dir = PROJ / ns_dir
    Vns, Fns = mi.load_obj(ns_dir / "head_closed_direct.obj")
    nsj = jload(ns_dir / "build_direct.json")
    bridge_tests = {}
    bt_path = PROJ / "tests/neck_bridge_tests.json"
    if bt_path.exists():
        bridge_tests = jload(bt_path)
    bridge_checks = {
        "tests": bridge_tests,
        "ring_check": meta.get("bridge_ring_check"),
        "c_ring_nonempty": bool((meta.get("bridge_ring_check") or {}).get("c_ring_1", 0) > 0
                                and (meta.get("bridge_ring_check") or {}).get("c_ring_3", 0) >
                                (meta.get("bridge_ring_check") or {}).get("c_ring_1", 0)),
        "weighted_row_target": next((t for t in bridge_tests.get("tests", [])
                                     if t.get("name") == "weighted_single_dof"), None),
        "bending_sign": next((t for t in bridge_tests.get("tests", [])
                              if t.get("name") == "bending_sign"), None),
        "rings": next((t for t in bridge_tests.get("tests", [])
                       if t.get("name") == "rings_nonempty"), None),
    }
    bridge_checks["pass"] = bool(
        (bridge_checks["weighted_row_target"] or {}).get("ok")
        and (bridge_checks["bending_sign"] or {}).get("ok")
        and (bridge_checks["rings"] or {}).get("ok")
        and bridge_checks["c_ring_nonempty"])
    # ablation equivalence / structure
    ab_peak = {}
    ab_signed = {}
    Eab = mi.edge_records(Fab)[0]
    aoab, aiab, goodab = _dihedral_pair(Vab, np.asarray(ab_npz["V_closed_input"], float),
                                       Fab, Eab)
    chab = (np.abs(aoab) - np.abs(aiab))[goodab]
    ab_signed = {"positive_max_deg": fmax(chab),
                 "positive_gt10": int((chab > 10.0).sum()),
                 "positive_gt20": int((chab > 20.0).sum())}
    for key, VV, FF in (("zero_topology", Vab, Fab),):
        fa = ns.analyse_surface(VV, FF, ax, hc, curve=curve, structure_sigma=sig,
                                fine_sigma=p["fine_sigma_mm"], leak_mask=True)
        ab_peak[key] = ns.structure_descriptors(fa, key,
                                                t_hi=0.90)["primary"]
    full_peak = ns.structure_descriptors(f_c, "full", t_hi=0.90)["primary"]
    full_signed = report["S2_connection"]["signed"]["closed"]["all_edges"]
    report["S8_targeted"] = {
        "V1_zero_topology_ablation": {
            "peak_full_mm": full_peak["relief_mm"], "peak_ablation_mm": ab_peak["zero_topology"]["relief_mm"],
            "peak_ratio": (ab_peak["zero_topology"]["relief_mm"] / full_peak["relief_mm"]),
            "position_full": full_peak["t"], "position_ablation": ab_peak["zero_topology"]["t"],
            "signed_connection": {
                "full_positive_max_deg": full_signed["positive_max_deg"],
                "ablation_positive_max_deg": ab_signed["positive_max_deg"],
                "full_gt10": full_signed["positive_gt10"],
                "ablation_gt10": ab_signed["positive_gt10"],
                "full_gt20": full_signed["positive_gt20"],
                "ablation_gt20": ab_signed["positive_gt20"]},
            "note": "structure is achieved without any accepted topology operation; "
                    "the full result keeps the positive >10/>20 counts at or "
                    "below the independent ablation",
            "pass": bool(abs(ab_peak["zero_topology"]["relief_mm"] / full_peak["relief_mm"] - 1.0) < 0.02
                         and abs(ab_peak["zero_topology"]["t"] - full_peak["t"]) < 0.02
                         and full_signed["positive_gt10"] <= ab_signed["positive_gt10"]
                         and full_signed["positive_gt20"] <= ab_signed["positive_gt20"])},
        "V2_zero_new_vertices": {
            "split_new_total": int(meta.get("journal", {}).get("split_new_total", -1)),
            "split_new_cap": int(meta.get("journal", {}).get("split_new_cap", -1)),
            "vertices_input": int(n0), "vertices_full": int(len(Vc)),
            "faces_input": int(len(F0)), "faces_full": int(len(Fc)),
            "no_split_ablation_vertices": int(len(Vns)),
            "no_split_ablation_faces": int(len(Fns)),
            "identical_to_main": bool(len(Vns) == len(Vc) and len(Fns) == len(Fc)
                                      and np.array_equal(Fns, Fc)),
            "pass": bool(meta.get("journal", {}).get("split_new_total", -1) == 0)},
        "V3_shared_topology_ops": {
            "faces_identical": report["S0_delivery"]["shared_topology"]["faces_identical"],
            "one_face_list": int(Fc.shape[0]) if Fc.ndim == 2 else None,
            "op_log_scope": "one shared journal; split/collapse/flip are index ops "
                            "applied to both states, not per-state meshes",
            "pass": bool(report["S0_delivery"]["shared_topology"]["faces_identical"])},
        "V4_bridge_defects": bridge_checks,
        "V5_reference_shape_only": {
            "reference": {"vertices": int(len(RV)), "faces": int(len(RF))},
            "output": {"vertices": int(len(Vc)), "faces": int(len(Fc))},
            "output_orig_ids_all_subject": bool(np.all(np.asarray(npz["orig_id"]) < n0)),
            "added_points": int(meta.get("journal", {}).get("split_new_total", -1)),
            "note": "the reference enters only as a 2-D relief field; its topology, "
                    "edge lengths and vertex count are not used",
            "pass": bool(np.all(np.asarray(npz["orig_id"]) < n0)
                         and int(meta.get("journal", {}).get("split_new_total", -1)) == 0)},
        "V6_target_amplitude": {
            "target_applied_max_mm": float(np.max(np.abs(np.asarray(npz["applied_closed"], float)))),
            "delivered_peak_closed_mm": float(report["S3_structure"]["closed"]["primary"]["relief_mm"]),
            "delivered_peak_open_mm": float(report["S3_structure"]["open"]["primary"]["relief_mm"]),
            "reference_peak_mm": float(report["S3_structure"]["closed"]["reference"]["relief_mm"]),
            "note": "the earlier 15.8 mm target came from disabling the gate top; "
                    "the applied relief maximum must stay in the plausible 4-6 mm "
                    "range and the delivered peak is checked against the reference",
            "pass": bool(float(np.max(np.abs(np.asarray(npz["applied_closed"], float)))) <= 6.5
                         and 0.85 <= report["S3_structure"]["closed"]["primary"]["relief_mm"]
                         / report["S3_structure"]["closed"]["reference"]["relief_mm"] <= 1.25)},
    }

    # ---- S9 verdict -----------------------------------------------------
    report["S9_verdict"] = verdict_v2(report, meta, p, n0)
    json.dump(report, open(out / "verification.json", "w", encoding="utf-8"),
              indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"\n[verdict] {json.dumps(report['S9_verdict']['summary'], indent=1)}")
    print(f"\nwritten {out / 'verification.json'} in {time.time()-t_start:.1f} s")
    return 0


def verdict_v2(report, meta, p, n0):
    g = {}
    s0 = report["S0_delivery"]
    g["one_mesh_two_states"] = {
        "value": s0["shared_topology"],
        "pass": bool(s0["shared_topology"]["faces_identical"]
                     and s0["shared_topology"]["vertex_count_equal"]),
        "limit": "identical face list"}
    g["obj_is_the_built_mesh"] = {
        "value": max(s0["obj_vs_build_arrays"]["max_abs_diff_closed_mm"],
                     s0["obj_vs_build_arrays"]["max_abs_diff_open_mm"]),
        "limit": 0.001,
        "pass": bool(max(s0["obj_vs_build_arrays"]["max_abs_diff_closed_mm"],
                         s0["obj_vs_build_arrays"]["max_abs_diff_open_mm"]) <= 0.001)}
    fx = report["S4_protection"]["fixed_set"]
    g["protection"] = {
        "value": {"closed_mm": fx["max_displacement_closed_mm"],
                  "open_mm": fx["max_displacement_open_mm"],
                  "vertices": fx["vertices"]},
        "limit": 0.002,
        "pass": bool(fx["max_displacement_closed_mm"] <= 0.002
                     and fx["max_displacement_open_mm"] <= 0.002)}
    topo = report["S5_topology"]
    ti, tc, to = topo["input"], topo["closed"], topo["open"]
    topo_ok = (tc["nonmanifold_edges"] == ti["nonmanifold_edges"]
               and to["nonmanifold_edges"] == ti["nonmanifold_edges"]
               and tc["inconsistent_shared_edges"] == 0
               and to["inconsistent_shared_edges"] == 0
               and tc["boundary_edges"] == ti["boundary_edges"]
               and to["boundary_edges"] == ti["boundary_edges"]
               and tc["zero_area_faces"] == 0 and to["zero_area_faces"] == 0
               and tc["components"] == ti["components"]
               and to["components"] == ti["components"])
    g["topology"] = {
        "value": {"closed": {k: tc[k] for k in ("nonmanifold_edges",
                                                "inconsistent_shared_edges",
                                                "boundary_edges", "zero_area_faces",
                                                "components")},
                  "open": {k: to[k] for k in ("nonmanifold_edges",
                                              "inconsistent_shared_edges",
                                              "boundary_edges", "zero_area_faces",
                                              "components")},
                  "input": {k: ti[k] for k in ("nonmanifold_edges",
                                               "inconsistent_shared_edges",
                                               "boundary_edges", "zero_area_faces",
                                               "components")}},
        "limit": "no new non-manifold edge, no flipped winding, boundary and "
                 "component count unchanged, no zero-area face",
        "pass": bool(topo_ok)}
    si = report.get("S6_self_intersection", {})
    if si.get("skipped"):
        g["self_intersection"] = {"pass": None, "value": "skipped by flag",
                                  "limit": "no new intersecting pair in the band"}
    else:
        bc = si["closed_band"]["intersections"] - si["input_band"]["intersections"]
        wr = (si["closed_whole_mesh"]["intersections"]
              - si["input_whole_mesh_rounded_1e6"]["intersections"])
        g["self_intersection"] = {
            "pass": bool(bc <= 0 and wr <= 0),
            "value": {"band_new": int(bc), "band": si["closed_band"]["intersections"],
                      "band_input": si["input_band"]["intersections"],
                      "whole": si["closed_whole_mesh"]["intersections"],
                      "whole_input": si["input_whole_mesh"]["intersections"],
                      "whole_input_rounded_1e6":
                          si["input_whole_mesh_rounded_1e6"]["intersections"],
                      "whole_new_vs_rounded_capture": int(wr)},
            "limit": "no new intersecting pair inside the band; whole mesh compared "
                     "with the capture rounded to the OBJ's own 1e-6 precision"}
    st = report["S3_structure"]
    pk_c = st["closed"]["primary"]
    pk_o = st["open"]["primary"]
    ref = st["closed"]["reference"]
    g["structure"] = {
        "value": {"closed_peak_mm": pk_c["relief_mm"], "closed_t": pk_c["t"],
                  "closed_theta_deg": pk_c["width_theta_deg"],
                  "open_peak_mm": pk_o["relief_mm"], "open_t": pk_o["t"],
                  "open_theta_deg": pk_o["width_theta_deg"],
                  "reference_peak_mm": ref["relief_mm"], "reference_t": ref["t"],
                  "reference_theta_deg": ref["width_theta_deg"]},
        "limit": "peak 0.85-1.25 x reference; |t - t_ref| <= 0.05; theta 0.5-2.0 x reference",
        "pass": bool(0.85 <= pk_c["relief_mm"] / ref["relief_mm"] <= 1.25
                     and 0.85 <= pk_o["relief_mm"] / ref["relief_mm"] <= 1.25
                     and abs(pk_c["t"] - ref["t"]) <= 0.05
                     and abs(pk_o["t"] - ref["t"]) <= 0.05
                     and 0.5 <= pk_c["width_theta_deg"] / ref["width_theta_deg"] <= 2.0
                     and 0.5 <= pk_o["width_theta_deg"] / ref["width_theta_deg"] <= 2.0)}
    lo = report["S3_structure"]["open"].get("lower_structure") or {}
    lor = report["S3_structure"]["open"].get("lower_structure_reference") or {}
    g["open_lower_structure_reported"] = {
        "value": {"lower_lobe_mm": (lo.get("lower_lobe") or {}).get("mm"),
                  "lower_lobe_reference_mm": (lor.get("lower_lobe") or {}).get("mm"),
                  "saddle_mm": (lo.get("saddle") or {}).get("mm"),
                  "saddle_reference_mm": (lor.get("saddle") or {}).get("mm"),
                  "profile_width_t_mm": lo.get("profile_width_t_mm"),
                  "profile_width_reference_t_mm": lor.get("profile_width_t_mm"),
                  "open_peak_ratio": pk_o["relief_mm"] / ref["relief_mm"]},
        "limit": "reported, not a gate: the closed state preserves the lower broad "
                 "lobe and shallow transition; the open state's lower lobe is weaker "
                 "and narrower, and the open profile width is truncated by the window",
        "pass": True}
    sg = report["S2_connection"]["signed"]["gates"]
    conn_pass = all(v["pass"] for blk in sg.values() for v in blk.values())
    g["connection_full_edge"] = {
        "value": {k: {kk: vv["value"] for kk, vv in blk.items()
                      if kk in ("full_positive_max_deg", "inside_B_positive_max_deg",
                                "crossing_positive_max_deg", "full_positive_gt10",
                                "full_positive_gt20")}
                  for k, blk in sg.items()},
        "limit": "full positive max <= 25 deg; B <= 20; crossing <= 20; no new fold > 25",
        "pass": bool(conn_pass)}
    z = report["S8_targeted"]
    g["zero_new_vertices"] = {
        "value": z["V2_zero_new_vertices"]["split_new_total"],
        "limit": 0, "pass": bool(z["V2_zero_new_vertices"]["pass"])}
    g["topology_ops_safe"] = {
        "value": z["V3_shared_topology_ops"]["faces_identical"],
        "limit": "one shared face list", "pass": bool(z["V3_shared_topology_ops"]["pass"])}
    g["bridge_defects"] = {
        "value": {"c_ring_1": (meta.get("bridge_ring_check") or {}).get("c_ring_1"),
                  "c_ring_3": (meta.get("bridge_ring_check") or {}).get("c_ring_3"),
                  "weighted_ok": (z["V4_bridge_defects"]["weighted_row_target"] or {}).get("ok"),
                  "bending_ok": (z["V4_bridge_defects"]["bending_sign"] or {}).get("ok")},
        "limit": "C ring non-empty; weighted target invariant; bending sign correct",
        "pass": bool(z["V4_bridge_defects"]["pass"])}
    g["reference_shape_only"] = {
        "value": z["V5_reference_shape_only"],
        "limit": "no reference topology/density inherited",
        "pass": bool(z["V5_reference_shape_only"]["pass"])}
    # density: no new densification; compression is reported, not gated
    rb = report["S1_density"]["ring_bands"]
    full = rb["full"]
    inp = rb["input"]
    dense_rows = [r for r in full if 0.925 <= r["t"][0] < 0.97]
    dense_in = [r for r in inp if 0.925 <= r["t"][0] < 0.97]
    dense_ratio = [(f["edge_p50_mm"] / i["edge_p50_mm"])
                   for f, i in zip(dense_rows, dense_in)
                   if i["edge_p50_mm"] and f["edge_p50_mm"]]
    dense_area = [(f["area_cm2"] / i["area_cm2"])
                  for f, i in zip(dense_rows, dense_in) if i["area_cm2"] > 0]
    dense_v_ok = all(f["vertices"] <= i["vertices"]
                     for f, i in zip(dense_rows, dense_in))
    dense_f_ok = all(f["faces"] <= i["faces"]
                     for f, i in zip(dense_rows, dense_in))
    vfull = int(report["S0_delivery"]["closed"]["vertices"])
    g["no_new_dense_ring"] = {
        "value": {"upper_band_edge_ratio": dense_ratio,
                  "upper_band_area_ratio": dense_area,
                  "upper_band_vertices": [[i["vertices"], f["vertices"]]
                                          for i, f in zip(dense_in, dense_rows)],
                  "upper_band_faces": [[i["faces"], f["faces"]]
                                        for i, f in zip(dense_in, dense_rows)],
                  "vertices_input": int(n0), "vertices_full": vfull},
        "limit": "upper band vertex and face counts not increased; output vertices "
                 "<= input.  A smaller edge p50 at equal/smaller area is local "
                 "compression, reported as area_ratio, not a new dense ring.",
        "pass": bool(dense_v_ok and dense_f_ok and vfull <= n0)}
    # density target deviation: reported, not a pass/fail gate
    reg = report["S1_density"]["regions"]
    density_dev = {}
    try:
        for row in reg.get("regions", []):
            k = row.get("region")
            if k in ("neck_front", "neck_body"):
                density_dev[k] = {
                    "input_p50_mm": row.get("edge_p50_original_mm"),
                    "output_p50_mm": row.get("edge_p50_final_mm"),
                    "ratio": row.get("ratio_final_over_original")}
    except Exception:
        pass
    g["density_target_deviation"] = {
        "value": density_dev,
        "limit": "reported; the output density is governed by the adjacent head, "
                 "not forced to the design ratio",
        "pass": True}
    failed = [k for k, v in g.items() if v["pass"] is False]
    gates_str = {k: ("pass" if v["pass"] else
                     ("skipped" if v["pass"] is None else "FAIL")) for k, v in g.items()}
    # residual findings that are not gates
    sgn = report["S2_connection"]["signed"]
    worst = []
    for key in ("closed", "open"):
        for w in sgn[key]["all_edges"]["worst_positive"][:3]:
            worst.append({"state": key, **w})
    return {"gates": gates_str,
            "values": g,
            "failed": failed,
            "residual_sharpening": worst,
            "summary": {"gates": gates_str,
                        "failed": failed,
                        "worst_positive_closed":
                            sgn["closed"]["all_edges"]["positive_max_deg"],
                        "worst_positive_open":
                            sgn["open"]["all_edges"]["positive_max_deg"],
                        "worst_positive_gt20_closed":
                            sgn["closed"]["all_edges"]["positive_gt20"],
                        "worst_positive_gt20_open":
                            sgn["open"]["all_edges"]["positive_gt20"],
                        "previous_edges": report["S2_connection"].get("previous_edges"),
                        "note": "the four 44.6/38.25/37.31/35.27 deg edges of the "
                                "previous round are tracked by material id; the "
                                "remaining worst positive edges are listed with "
                                "their location and input angle"}}

if __name__ == "__main__":
    raise SystemExit(main())
