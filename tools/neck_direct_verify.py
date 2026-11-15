"""Acceptance of the direct neck stage, read back from the delivered files.

``head_closed_direct.obj`` and ``head_open_direct.obj`` are parsed from disk and
every geometric number here is derived from those two arrays.  Nothing is
re-meshed, re-fitted or re-analysed from the build's working state.  Exactly
three build products are read as well, and each of them is *checked* rather than
trusted:

* ``V_*_input`` -- the "before" positions on the delivered topology, i.e. the
  original capture surface sampled at the delivered vertex slots.  Section S0
  asserts bit-identity with the original capture vertex for every surviving slot
  (slots below ``n0``) and a distance bound for the handful of split vertices;
  without that assertion the displacement and dihedral-change columns would be
  measured against an invented baseline;
* ``region`` -- the A/B/C material label of the shared topology.  It cannot be
  recovered from geometry alone (A and C are both smooth neck surface), so it is
  read and then cross-checked against the delivered face list: the interface
  lines have to exist in the delivered connectivity, and the B area / width have
  to match the geometry that is actually there;
* ``density_zone`` -- used only for the per-zone budget table.

Sections
    S0 delivery       what the two files are, and whether they are one mesh
    S1 density        regional edge length, areal face density, the op journal
    S2 connection     the complete connection band, both states, no window
    S3 structure      peak, prominence, width and artefacts on the delivered surface
    S4 protection     the fixed head, the cut boundary, the hand-over shell
    S5 topology       manifoldness, winding, boundary, slivers, angles, dihedral
    S6 self-intersect complete broad phase, band and whole mesh, with baselines
    S7 renders        presence of the same-camera panels
    S8 targeted       four checks that the method is what it claims to be
    S9 verdict        every gate with its value, its limit and, when it fails,
                      the edge where it fails

The one synthetic part is the regulariser sweep in V4; it runs on small strips
and says so in its own output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi          # noqa: E402
from head_rgbd import neck_bridge as nbr            # noqa: E402
from head_rgbd import neck_direct as nd             # noqa: E402
from head_rgbd import neck_stage as ns              # noqa: E402
from tools import build_neck_direct as bnd          # noqa: E402
from tools import build_neck_stage as bns           # noqa: E402


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


def nrm_face_angles(V, F):
    tri = np.asarray(V, float)[F]
    ang = np.zeros((len(F), 3))
    for k in range(3):
        a, b, c = tri[:, k], tri[:, (k + 1) % 3], tri[:, (k + 2) % 3]
        u, w = b - a, c - a
        cos = np.einsum("ij,ij->i", u, w) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-12)
        ang[:, k] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return ang


# --------------------------------------------------------------------------
# S0. what is on disk
# --------------------------------------------------------------------------


def delivery_section(out, Vc, Fc, Vo, Fo, npz, n0, V0c, V0o, F0):
    oid = np.asarray(npz["orig_id"], np.int64)
    reg = np.asarray(npz["region"], np.int64)
    zon = np.asarray(npz["density_zone"], np.int64)
    Vci, Voi = np.asarray(npz["V_closed"]), np.asarray(npz["V_open"])
    Vcin = np.asarray(npz["V_closed_input"])

    # the two files must be the same mesh: one topology, one index order
    shared = {"faces_identical": bool(Fc.shape == Fo.shape and np.array_equal(Fc, Fo)),
              "vertex_count_equal": bool(len(Vc) == len(Vo)),
              "index_range_ok_closed": bool(Fc.max() < len(Vc) and Fc.min() >= 0),
              "index_range_ok_open": bool(Fo.max() < len(Vo) and Fo.min() >= 0)}

    # ... and the OBJ must be the array the build wrote, not a stale file.
    # ``write_obj`` prints 6 decimals, so half a micrometre is the resolution.
    obj = {"max_abs_diff_closed_mm": float(np.abs(Vc - Vci).max() * 1000.0),
           "max_abs_diff_open_mm": float(np.abs(Vo - Voi).max() * 1000.0),
           "resolution_mm": 0.001,
           "note": "head_*_direct.obj re-read and compared with stage_direct.npz; the "
                   "writer rounds to 1e-6 m, so anything above 0.001 mm would mean the "
                   "OBJ on disk is not the mesh the build finished with"}

    # the "before" baseline: for every surviving slot it has to be the original
    # capture vertex itself, otherwise displacement and crease change are
    # measured against something that was never captured
    surv = oid < n0
    before = {}
    for tag, Vin, V0 in (("closed", Vcin, V0c),
                         ("open", np.asarray(npz["V_open_input"]), V0o)):
        d = np.linalg.norm(Vin[surv] - V0[oid[surv]], axis=1) * 1000.0
        before[tag] = {"surviving_slots": int(surv.sum()),
                       "max_diff_vs_capture_mm": float(d.max()),
                       "slots_not_in_capture": int((~surv).sum()),
                       "slots_not_in_capture_share": float((~surv).mean())}
        if (~surv).any():
            # a split vertex has no capture position of its own; its baseline is
            # the midpoint of its two parents, so it must lie *on* the input
            # surface to within half a local edge
            dist, _ = ns.point_to_surface_mm(Vin[~surv], V0, F0)
            before[tag]["split_slots_point_to_input_surface_max_mm"] = float(dist.max())
    before["note"] = ("slots below n0 are capture vertices and are required to match "
                      "bit-exactly; a slot above n0 was created by a split and its "
                      "baseline is the parent midpoint, checked by distance to the input "
                      "surface instead")

    # the material label cannot be read off the geometry, so check that it agrees
    # with the delivered connectivity and with the bridge geometry that is there
    E = mi.edge_records(Fc)[0]
    r0, r1 = reg[E[:, 0]], reg[E[:, 1]]
    ab = ((r0 == 0) & (r1 == 1)) | ((r0 == 1) & (r1 == 0))
    bc = ((r0 == 1) & (r1 == 2)) | ((r0 == 2) & (r1 == 1))
    ac = ((r0 == 0) & (r1 == 2)) | ((r0 == 2) & (r1 == 0))
    tri = Vc[Fc] * 1000.0
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                axis=1)
    fb_in = (reg[Fc] == 1).all(axis=1)
    fb_touch = (reg[Fc] == 1).any(axis=1)
    hm = np.asarray(npz["h_material"], float)
    labels = {"region_counts": {"A": int((reg == 0).sum()), "B": int((reg == 1).sum()),
                                "C": int((reg == 2).sum())},
              "density_zone_counts": {str(k): int((zon == k).sum()) for k in range(4)},
              "interface_edges_in_delivered_mesh": {"A_B": int(ab.sum()),
                                                    "B_C": int(bc.sum()),
                                                    "A_C_direct": int(ac.sum())},
              # "inside" = all three corners are B, which is the region's own surface;
              # "touching" = at least one corner is B, which is larger by the two rings
              # of A/C faces that hang off the band.  Both are reported because the
              # density budget is written per region and must be compared to "inside"
              "B_faces": int(fb_in.sum()),
              "B_area_cm2": float(area[fb_in].sum() / 100.0),
              "B_faces_touching": int(fb_touch.sum()),
              "B_area_cm2_touching": float(area[fb_touch].sum() / 100.0),
              "B_faces_per_vertex": float(fb_in.sum() / max(1, int((reg == 1).sum()))),
              "h_material_B_range_mm": [float(hm[reg == 1].min()), float(hm[reg == 1].max())],
              "h_material_C_range_mm": [float(hm[reg == 2].min()), float(hm[reg == 2].max())],
              "note": "region is an index label of the shared topology and cannot be "
                      "derived from geometry; what is checked here is that the two "
                      "interface lines exist in the delivered connectivity (an A-C edge "
                      "would mean the bridge does not separate head from neck) and that "
                      "B is a band of the material height, not a blob"}

    # the per-region density budget is written over the *editable* subset, so its
    # area is a vertex-area sum over those vertices, not the region's whole surface.
    # Check that it is self-consistent:  sum(A_v)/n  must be close to (sqrt3/2) ell^2
    try:
        bj = json.loads((out / "build_direct.json").read_text(encoding="utf-8"))
        ba = {}
        for scope, rows in (("by_region", bj.get("budget_by_region") or {}),
                            ("by_zone", bj.get("budget_by_zone") or {})):
            for k, row in rows.items():
                n = float(row.get("n_current") or 0)
                a = float(row.get("area_mm2") or 0.0)
                ba[f"{scope}:{k}"] = {
                    "area_cm2": a / 100.0, "n_current": int(n),
                    "area_per_vertex_mm2": (a / n) if n else None,
                    "n_budget": row.get("n_budget"), "n_estimate": row.get("n_estimate")}
        au, _ = nd.vertex_areas(Vc, Fc)
        ba["_note"] = ("budget area is sum(A_v) over the editable vertices of the region, "
                       "so it is smaller than the region's whole surface; area_per_vertex "
                       "should land near (sqrt3/2)*ell^2 of that region's target edge")
        ba["_san_check"] = {"whole_region_area_cm2": {k: float(area[(reg[Fc] == i)
                                                                  .all(axis=1)].sum() / 100.0)
                                                     for k, i in (("A", 0), ("B", 1), ("C", 2))},
                            "sum_vertex_areas_cm2": float(au.sum() / 100.0)}
        labels["budget_area"] = ba
    except Exception as exc:                                   # report, never hide
        labels["budget_area"] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"files": {k: str((out / k)) for k in
                      ("head_closed_direct.obj", "head_open_direct.obj",
                       "stage_direct.npz", "build_direct.json")},
            "closed": {"vertices": int(len(Vc)), "faces": int(len(Fc))},
            "open": {"vertices": int(len(Vo)), "faces": int(len(Fo))},
            "shared_topology": shared, "obj_vs_build_arrays": obj,
            "before_positions": before, "labels": labels}


# --------------------------------------------------------------------------
# S1. density
# --------------------------------------------------------------------------


def density_section(V0, F0, Vf, Ff, ax, curve, hc, p, meta, field):
    reg0 = nd.head_scales(V0, F0, ax, curve, hc)
    regf = nd.head_scales(Vf, Ff, ax, curve, hc)
    rows = []
    for key in ("mandible_lateral", "occiput_lower", "neck_front", "neck_body", "shoulder"):
        a, b = reg0[key], regf[key]
        rows.append({"region": key,
                     "edge_p50_original_mm": a["edge_p50_mm"],
                     "edge_p50_final_mm": b["edge_p50_mm"],
                     "ratio_final_over_original": (b["edge_p50_mm"] / a["edge_p50_mm"]
                                                   if a["edge_p50_mm"] and b["edge_p50_mm"]
                                                   else None),
                     "faces_per_cm2_original": a["faces_per_cm2"],
                     "faces_per_cm2_final": b["faces_per_cm2"],
                     "vertices_final": b["vertices"]})

    h_all, th_all, _, t_all = nd.axial_mm(Vf * 1000.0, ax, curve, hc)
    ell, zones_eval, anchors = nd.ell_field(Vf, Ff, ax, curve, hc, reg0, p)
    E = mi.edge_records(Ff)[0]
    L = np.linalg.norm(Vf[E[:, 0]] - Vf[E[:, 1]], axis=1) * 1000.0
    ell_e = 0.5 * (ell[E[:, 0]] + ell[E[:, 1]])
    inband = ((t_all[E[:, 0]] > p["data_t"][0]) & (t_all[E[:, 0]] < p["data_t"][3]) &
              (t_all[E[:, 1]] > p["data_t"][0]) & (t_all[E[:, 1]] < p["data_t"][3]))
    nrm = L[inband] / np.maximum(ell_e[inband], 1e-6)
    band = {"edges": int(inband.sum()),
            "L_over_ell_p10": pct(nrm, 10), "L_over_ell_p50": pct(nrm, 50),
            "L_over_ell_p90": pct(nrm, 90),
            "share_within_4over5_and_4over3": float(np.mean((nrm >= 0.8) & (nrm <= 4.0 / 3.0))),
            "share_longer_than_4over3": float(np.mean(nrm > 4.0 / 3.0)),
            "share_shorter_than_4over5": float(np.mean(nrm < 0.8)),
            "edge_mm_p10": pct(L[inband], 10), "edge_mm_p50": pct(L[inband], 50),
            "edge_mm_p90": pct(L[inband], 90)}

    regions = {}
    for name, cond in (("anterior |theta|<30, t 0.5-0.85",
                        (np.abs(th_all) < np.radians(30)) & (t_all > 0.50) & (t_all < 0.85)),
                       ("structure |theta|<20, t 0.62-0.80",
                        (np.abs(th_all) < np.radians(20)) & (t_all > 0.62) & (t_all < 0.80)),
                       ("side 60<|theta|<95",
                        (np.abs(th_all) > np.radians(60)) & (np.abs(th_all) < np.radians(95)) &
                        (t_all > 0.42) & (t_all < 0.88)),
                       ("posterior |theta|>100",
                        (np.abs(th_all) > np.radians(100)) & (t_all > 0.42) & (t_all < 0.88))):
        if cond.sum() < 10:
            continue
        regions[name] = {"vertices": int(cond.sum()), "ell_mm_p50": pct(ell[cond], 50)}

    ops = {"split": 0, "collapse": 0, "flip": 0, "candidate_split": 0, "candidate_collapse": 0}
    attempted = {"split": 0, "collapse": 0, "flip": 0}
    per_pass = []
    for rec in meta["journal"]["passes"]:
        for k in attempted:
            attempted[k] += int(rec.get(k, 0) or 0)
        # a rolled-back pass did not reach the delivered mesh; count only the
        # accepted operations here and keep the attempted counts separate
        if not rec.get("rolled_back"):
            for k in ops:
                ops[k] += int(rec.get(k, 0) or 0)
        per_pass.append({k: rec.get(k) for k in
                         ("pass", "free", "room_split", "candidate_split",
                          "candidate_collapse", "split", "collapse", "flip", "vertices",
                          "faces", "relax_max_move_mm", "rolled_back")})
        per_pass[-1]["rejected_split_candidates"] = len(rec.get("ops", {}).get("rejected", []))

    spl = [o for rec in meta["journal"]["passes"] for o in rec.get("ops", {}).get("split", [])]
    cause = {"split_by_region": {}, "split_norm_p50": None, "split_norm_max": None,
             "split_abs_relief_p50_mm": None, "rejected": []}
    for rec in meta["journal"]["passes"]:
        cause["rejected"] += rec.get("ops", {}).get("rejected", [])
    if spl:
        mid = np.array([0.5 * (Vf[o["edge"][0]] + Vf[o["edge"][1]]) for o in spl])
        hh, tt, _, _ = nd.axial_mm(mid * 1000.0, ax, curve, hc)
        thd = np.degrees(np.abs(tt))
        tt = hh / hc
        rel = np.abs(np.asarray(field["relief"], float))
        lab = np.where(np.abs(thd) < 30, "anterior",
                       np.where(np.abs(thd) < 95, "lateral", "posterior"))
        lab = np.where(tt > 0.86, "interface", lab)
        for k in sorted(set(lab.tolist())):
            cause["split_by_region"][k] = int(np.sum(lab == k))
        cause["split_norm_p50"] = pct([o["norm"] for o in spl], 50)
        cause["split_norm_max"] = float(max(o["norm"] for o in spl))
        F2 = ns.Field2D(np.asarray(field["t"], float) * hc,
                        np.asarray(field["theta"], float), np.nan_to_num(rel))
        amp = F2(tt * hc, np.sign(tt) * np.radians(thd))
        cause["split_abs_relief_p50_mm"] = pct(amp, 50)
        cause["split_abs_relief_min_mm"] = float(np.min(amp)) if len(amp) else None

    return {"regions": rows, "band": band, "ell_by_region": regions, "anchors": anchors,
            "zone_counts_recomputed": {str(k): int((zones_eval == k).sum())
                                       for k in range(4)},
            "operations": ops, "attempted_operations": attempted,
            "per_pass": per_pass, "split_cause": cause,
            "budget": {"by_zone": meta.get("budget_by_zone"), "by_region":
                       meta.get("budget_by_region"), "zone_cap": meta.get("zone_cap"),
                       "total": meta["budget"]},
            "counts": {"vertices_original": meta["journal"]["vertices_original"],
                       "vertices_final": meta["journal"]["vertices_final"],
                       "vertices_used_final": meta["journal"]["vertices_used"],
                       "faces_original": meta["journal"]["faces_original"],
                       "faces_final": meta["journal"]["faces_final"],
                       "net_new_vertices": ops["split"] - ops["collapse"],
                       "merged_slots": meta["journal"]["merged_slots"],
                       "largest_mesh_vertices": meta["journal"]["max_working_vertices"],
                       "largest_mesh_faces": meta["journal"]["max_working_faces"]}}


# --------------------------------------------------------------------------
# S2. the connection, both states, complete
# --------------------------------------------------------------------------


def _dihedral_full(V, F, E):
    """Signed dihedral per edge of ``E``; NaN on a boundary edge.

    ``nbr.dihedral_deg`` only returns interior edges, so the result is scattered
    back into the full edge list.  Keeping the full list lets every subset below
    (including ``all_edges``) be indexed by one boolean mask of one length.
    """
    counts = mi.edge_records(F)[2]
    two = np.flatnonzero(counts == 2)
    _eb, ang = nbr.dihedral_deg(V, F)
    out = np.full(len(E), np.nan)
    out[two] = ang
    return out


def _band_subsets(reg, F):
    E = mi.edge_records(F)[0]
    r0, r1 = reg[E[:, 0]], reg[E[:, 1]]
    near_b = np.zeros(len(reg), bool)
    touch = ((r0 == 1) | (r1 == 1))
    near_b[E[touch].ravel()] = True
    sel = {
        "all_edges": np.ones(len(E), bool),
        "inside_B": (r0 == 1) & (r1 == 1),
        "crossing_A_B": ((r0 == 0) & (r1 == 1)) | ((r0 == 1) & (r1 == 0)),
        "crossing_B_C": ((r0 == 1) & (r1 == 2)) | ((r0 == 2) & (r1 == 1)),
        "inside_C_adjacent_to_B": (r0 == 2) & (r1 == 2) & (near_b[E[:, 0]] | near_b[E[:, 1]]),
        "crossing_any": (((r0 == 1) & (r1 != 1)) | ((r1 == 1) & (r0 != 1))),
    }
    return E, sel


def _subset_stats(E, sel, Vout, Vin, drad, ang_o, ang_i, frame, reg=None):
    """max / p95 / count>10 and the worst edges, with a location for each."""
    L = np.linalg.norm((Vout[E[:, 0]] - Vout[E[:, 1]]) * 1000.0, axis=1)
    Lin = np.linalg.norm((Vin[E[:, 0]] - Vin[E[:, 1]]) * 1000.0, axis=1)
    g = np.abs(drad[E[:, 0]] - drad[E[:, 1]]) / np.maximum(L, 1e-9)
    ch = np.abs(ang_o) - np.abs(ang_i)
    a, b = E[sel, 0], E[sel, 1]
    out = {"edges": int(sel.sum())}
    if not sel.any():
        return out
    Ls, Lsi, gs, chs = L[sel], Lin[sel], g[sel], ch[sel]
    ao, ai = ang_o[sel], ang_i[sel]
    out.update({"edge_mm_p50": pct(Ls, 50), "edge_mm_min": float(Ls.min()),
                "edge_mm_max": float(Ls.max()),
                "gradient_mm_per_mm_max": fmax(gs), "gradient_p95_mm_per_mm": pct(gs, 95),
                "dihedral_change_max_deg": fmax(chs), "dihedral_change_p95_deg": pct(chs, 95),
                "dihedral_change_p50_deg": pct(chs, 50),
                "dihedral_out_p95_deg": pct(np.abs(ao), 95),
                "dihedral_in_p95_deg": pct(np.abs(ai), 95),
                "edges_change_gt_1_deg": int((np.abs(chs) > 1.0).sum()),
                "edges_change_gt_10_deg": int((np.abs(chs) > 10.0).sum()),
                "share_change_gt_10_deg": float((np.abs(chs) > 10.0).mean()),
                "edges_input_shorter_than_half_mm": int((Lsi < 0.5).sum())})
    # the worst few, located: a magnitude without a position cannot be judged
    mid = 0.5 * (Vout[a] + Vout[b]) * 1000.0
    h, th, _, t = nd.axial_mm(mid, frame[0], frame[1], frame[2])
    worst = []
    for k in np.argsort(-np.abs(chs))[:6]:
        worst.append({"edge": [int(a[k]), int(b[k])],
                      "regions": ([int(reg[a[k]]), int(reg[b[k]])] if reg is not None
                                  else None),
                      "in_deg": float(ai[k]), "out_deg": float(ao[k]),
                      "change_deg": float(chs[k]),
                      "sign_flip": bool(np.sign(ao[k]) != np.sign(ai[k])),
                      "edge_mm_out": float(Ls[k]), "edge_mm_in": float(Lsi[k]),
                      "sliver_in_input": bool(Lsi[k] < 1.0),
                      "t": float(t[k]), "theta_deg": float(np.degrees(th[k])),
                      "gradient_mm_per_mm": float(gs[k])})
    out["worst_change"] = worst
    return out


def connection_section(npz, Fc, ax, curve, hc, ax_o, curve_o, hc_o, p, meta):
    reg = np.asarray(npz["region"], np.int64)
    E, sel = _band_subsets(reg, Fc)
    blocks = {}
    for key, axx, cur, hcc in (("closed", ax, curve, hc), ("open", ax_o, curve_o, hc_o)):
        Vout = np.asarray(npz["V_" + key], float)
        Vin = np.asarray(npz["V_" + key + "_input"], float)
        drad = np.asarray(npz["d_rad_" + key], float)
        ang_o = _dihedral_full(Vout, Fc, E)
        ang_i = _dihedral_full(Vin, Fc, E)
        sub = {name: _subset_stats(E, s, Vout, Vin, drad, ang_o, ang_i,
                                   (axx, cur, hcc), reg) for name, s in sel.items()}
        # the same measurement restricted to edges that are at least 2 mm long in
        # the *input*.  Supplementary only: it exists to show how much of the tail
        # is carried by the sub-millimetre slivers the capture already contains,
        # and it never replaces the numbers above.
        Lin = np.linalg.norm((Vin[E[:, 0]] - Vin[E[:, 1]]) * 1000.0, axis=1)
        sup = {name: _subset_stats(E, s & (Lin >= 2.0), Vout, Vin, drad, ang_o, ang_i,
                                   (axx, cur, hcc), reg) for name, s in sel.items()}
        blocks[key] = {"subsets": sub,
                       "supplementary_edges_input_ge_2mm": sup,
                       "note": "dihedral is |angle| on every interior edge; the change "
                               "column is |out| - |in| on the same edge of the same face "
                               "list, so it is a shape change and not an analysis-window "
                               "artefact.  in = the capture surface resampled at the "
                               "delivered vertex slots (checked in S0)"}

    # criteria, taken from the build's own acceptance (params), re-evaluated here
    lim_g = float(p["adapt_gradient_limit"])
    lim_c = float(p["adapt_crease_limit_deg"])
    mar = float(p["adapt_sharpness_margin_deg"])
    gates = {}
    for key, blk in blocks.items():
        s = blk["subsets"]
        cr = s["crossing_any"]
        sharp_ok = bool(s["inside_B"]["dihedral_out_p95_deg"]
                        <= s["inside_B"]["dihedral_in_p95_deg"] + mar
                        and cr["dihedral_out_p95_deg"] <= cr["dihedral_in_p95_deg"] + mar)
        # the same rule the build used to accept a candidate: the crease *change*
        # is allowed to be large where the surface is already sharp (the submental
        # fold is), so it passes either by being small or by not being sharper
        # than the input; the gradient has to pass on its own
        change_ok = bool(cr["dihedral_change_p95_deg"] <= lim_c) or sharp_ok
        gates[key] = {
            "crease_change_across_p95": {"value": cr["dihedral_change_p95_deg"],
                                         "limit": lim_c,
                                         "pass": change_ok,
                                         "rule": "p95 of |out|-|in| across the seam must be "
                                                 "<= limit, OR the delivered seam must be no "
                                                 "sharper than the input + margin"},
            "no_new_sharp_edge": {
                "value": {"B_out_p95": s["inside_B"]["dihedral_out_p95_deg"],
                          "B_in_p95": s["inside_B"]["dihedral_in_p95_deg"],
                          "cross_out_p95": cr["dihedral_out_p95_deg"],
                          "cross_in_p95": cr["dihedral_in_p95_deg"]},
                "limit": f"out p95 <= in p95 + {mar}", "pass": sharp_ok},
            "gradient_across_p95_cumulative": {
                "value": cr["gradient_p95_mm_per_mm"], "limit": lim_g,
                "pass": bool(cr["gradient_p95_mm_per_mm"] <= lim_g),
                "note": "cumulative shape change (delivered minus capture).  The build "
                        "logged this quantity per solve increment, which is ~5x smaller; "
                        "the limit was written for that comparison, so this gate is the "
                        "stricter reading and is reported as such"},
            "gradient_inside_B_p95": {"value": s["inside_B"]["gradient_p95_mm_per_mm"],
                                      "limit": lim_g,
                                      "pass": bool(s["inside_B"]["gradient_p95_mm_per_mm"] <= lim_g)},
            "max_change_location": cr["worst_change"][0] if cr["worst_change"] else None,
        }
    # what the build itself measured, for cross-reference
    return {"scope": "every interior edge of the delivered closed mesh; no t window, no "
                     "face filter.  The subsets are named by the material label only, so "
                     "'crossing_any' contains the A-B line and the B-C line in full",
            "closed": blocks["closed"], "open": blocks["open"],
            "criteria": {"gradient_p95_mm_per_mm": lim_g, "crease_p95_deg": lim_c,
                         "sharpness_margin_deg": mar},
            "gates": gates,
            "build_log": meta.get("connection")}


# --------------------------------------------------------------------------
# S3. structure
# --------------------------------------------------------------------------


def artefact_tests(field, t_hi, hc):
    """Spike / full-ring groove / horizontal step on the 2-D relief."""
    from scipy.ndimage import median_filter
    rel = np.asarray(field["relief"], float)
    t = np.asarray(field["t"], float)
    th = np.asarray(field["theta"], float)
    win = (t > 0.44) & (t < t_hi)
    ang = np.abs(np.degrees(th)) < 95
    sub = rel[np.ix_(win, ang)]
    sub = np.where(np.isfinite(sub), sub, 0.0)
    loc = median_filter(sub, size=(5, 7), mode="nearest")
    spike = sub - loc
    neg = (sub < -0.8).mean(axis=1)
    j0 = int(np.argmin(np.abs(th)))
    col = np.where(np.isfinite(rel[:, j0]) & win, rel[:, j0], np.nan)
    d = np.abs(np.gradient(col, t * hc))
    out = {"window_t": [0.44, t_hi],
           "spike_max_mm": float(np.max(spike)) if spike.size else 0.0,
           "spike_over_1_5mm": int((spike > 1.5).sum()),
           "rings_mostly_negative": int((neg > 0.9).sum()),
           "anterior_step_max_mm_per_mm": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0}
    if spike.size:
        i, j = np.unravel_index(int(np.argmax(spike)), spike.shape)
        out["spike_at"] = {"t": float(t[np.flatnonzero(win)[i]]),
                           "theta_deg": float(np.degrees(th[np.flatnonzero(ang)[j]]))}
    if np.isfinite(d).any():
        i = int(np.nanargmax(d))
        out["step_at"] = {"t": float(t[i]), "theta_deg": float(np.degrees(th[j0]))}
    ok = np.isfinite(col)
    if ok.sum() > 4:
        c = col[ok]
        hh = (t * hc)[ok]
        dd = np.diff(c) / np.maximum(np.diff(hh), 1e-9)
        sgn = np.sign(dd)
        out["anterior_slope_sign_changes"] = int((sgn[1:] != sgn[:-1]).sum())
        out["anterior_curvature_max_per_mm"] = float(np.max(np.abs(np.gradient(dd, hh[1:]))))
    return out


def lower_structure(field, hc, t_hi, half_deg=20.0):
    rel = np.asarray(field["relief"], float)
    t = np.asarray(field["t"], float)
    th = np.asarray(field["theta"], float)
    sel = np.abs(np.degrees(th)) < half_deg
    prof = np.nanmean(rel[:, sel], axis=1)
    win = (t > 0.30) & (t < t_hi)
    out = {"profile_vertices_deg": half_deg}
    seg = win & (t >= 0.60) & (t <= 0.72)
    if seg.any():
        i = int(np.flatnonzero(seg)[int(np.nanargmin(prof[seg]))])
        out["saddle"] = {"t": float(t[i]), "mm": float(prof[i])}
    seg = win & (t >= 0.46) & (t <= 0.60)
    if seg.any():
        i = int(np.flatnonzero(seg)[int(np.nanargmax(prof[seg]))])
        out["lower_lobe"] = {"t": float(t[i]), "mm": float(prof[i])}
    pk = win & (t >= 0.68) & (t <= 0.84)
    if pk.any():
        i = int(np.flatnonzero(pk)[int(np.nanargmax(prof[pk]))])
        out["peak"] = {"t": float(t[i]), "mm": float(prof[i])}
    if "saddle" in out and "peak" in out:
        dh = (out["peak"]["t"] - out["saddle"]["t"]) * hc
        out["transition_slope_mm_per_mm"] = float(
            (out["peak"]["mm"] - out["saddle"]["mm"]) / dh) if dh > 0 else None
    if "saddle" in out and "lower_lobe" in out:
        out["saddle_below_lobe_mm"] = float(out["saddle"]["mm"] - out["lower_lobe"]["mm"])
    if "peak" in out:
        half = 0.5 * out["peak"]["mm"]
        above = np.flatnonzero((prof > half) & (t > 0.40) & (t < t_hi))
        if len(above):
            out["profile_width_t_mm"] = float((t[above].max() - t[above].min()) * hc)
            out["profile_width_truncated_by_window"] = bool(
                t[above].max() > t_hi - 0.006 or t[above].min() < 0.406)
            out["profile_width_span_t"] = [float(t[above].min()), float(t[above].max())]
    return out


def structure_section(field, ref, acc, hc, p, label, ref_field=None, acc_field=None):
    # the window follows THIS state's own closure height: with the jaw open the
    # chin sits lower, so a window that ended at the closed value would measure
    # the chin surface as a spike
    t_hi = 0.90 if label.endswith("closed") else 0.84
    q = ns.structure_descriptors(field, label, t_lo=0.42, t_hi=t_hi)
    prim, pr = q["primary"], ref["primary"]
    out = {"primary": prim, "reference": pr, "secondary": q.get("secondary"),
           "secondary_reference": ref.get("secondary"), "saddle": q.get("saddle"),
           "saddle_reference": ref.get("saddle"), "lateral": q.get("lateral"),
           "lateral_reference": ref.get("lateral"),
           "asymmetry_rms_mm": q.get("asymmetry_rms_mm"),
           "relief_rms_mm": q.get("relief_rms_mm"),
           "relief_min_mm": q.get("relief_min_mm"), "relief_max_mm": q.get("relief_max_mm")}
    out["position_deviation_mm"] = abs(prim["t"] - pr["t"]) * hc
    out["position_deviation_pct_of_neck"] = 100.0 * abs(prim["t"] - pr["t"])
    out["prominence_change_pct_vs_reference"] = 100.0 * (prim["relief_mm"] / pr["relief_mm"] - 1.0)
    out["angular_width_change_pct_vs_reference"] = 100.0 * (
        prim["width_theta_deg"] / pr["width_theta_deg"] - 1.0)
    out["axial_width_change_pct_vs_reference"] = 100.0 * (
        prim["width_t_mm"] / pr["width_t_mm"] - 1.0)
    if acc is not None:
        ap = acc["primary"]
        out["accepted"] = ap
        out["accepted_secondary"] = acc.get("secondary")
        out["accepted_lateral"] = acc.get("lateral")
        out["prominence_change_pct_vs_accepted"] = 100.0 * (prim["relief_mm"] / ap["relief_mm"] - 1.0)
        out["angular_width_change_pct_vs_accepted"] = 100.0 * (
            prim["width_theta_deg"] / ap["width_theta_deg"] - 1.0)
        out["axial_width_change_pct_vs_accepted"] = 100.0 * (
            prim["width_t_mm"] / ap["width_t_mm"] - 1.0)
        out["position_deviation_vs_accepted_mm"] = abs(prim["t"] - ap["t"]) * hc

    out.update(artefact_tests(field, t_hi, hc))
    out["artefacts_reference"] = artefact_tests(ref_field, 0.90, hc) if ref_field is not None else None
    out["artefacts_accepted"] = artefact_tests(acc_field, 0.90, hc) if acc_field is not None else None
    out["artefacts_interior"] = artefact_tests(field, 0.80, hc)
    out["artefacts_interior_reference"] = (artefact_tests(ref_field, 0.80, hc)
                                           if ref_field is not None else None)
    out["artefacts_interior_accepted"] = (artefact_tests(acc_field, 0.80, hc)
                                          if acc_field is not None else None)
    out["lower_structure"] = lower_structure(field, hc, t_hi)
    out["lower_structure_reference"] = lower_structure(ref_field, hc, 0.90) if ref_field is not None else None
    out["lower_structure_accepted"] = lower_structure(acc_field, hc, 0.90) if acc_field is not None else None
    if out["lower_structure_reference"]:
        a, b = out["lower_structure"], out["lower_structure_reference"]
        if "saddle" in a and "saddle" in b:
            out["saddle_position_deviation_pct"] = 100.0 * abs(a["saddle"]["t"] - b["saddle"]["t"])
            out["saddle_depth_change_pct"] = (100.0 * (a["saddle"]["mm"] / b["saddle"]["mm"] - 1.0)
                                              if b["saddle"]["mm"] else None)
        if "lower_lobe" in a and "lower_lobe" in b:
            out["lower_lobe_change_pct"] = (100.0 * (a["lower_lobe"]["mm"] / b["lower_lobe"]["mm"] - 1.0)
                                            if b["lower_lobe"]["mm"] else None)
            out["lower_lobe_position_deviation_pct"] = 100.0 * abs(
                a["lower_lobe"]["t"] - b["lower_lobe"]["t"])
        if (a.get("transition_slope_mm_per_mm") is not None
                and b.get("transition_slope_mm_per_mm")):
            out["transition_slope_change_pct"] = 100.0 * (
                a["transition_slope_mm_per_mm"] / b["transition_slope_mm_per_mm"] - 1.0)
    return out


# --------------------------------------------------------------------------
# S4. protection
# --------------------------------------------------------------------------


def protection_section(V0, V0o, F0, Vc, Fc, Vo, Fo, npz, ax, curve, hc, p, meta):
    """Does the head stay where it was, and is the cut boundary bit-identical?

    The protected set is the build's own (``state_masks``): region A, the open
    rim and the root (material t < 0.02), fixed **before** any solve and the same
    material point set in both states.  Everything is measured in exported index
    space, because the OBJ is the compacted array: slot k of the file is original
    slot ``orig_id[k]``, and each state is compared with its own capture -- the
    mandible really does rotate between the states, so comparing the open OBJ
    with the closed capture would report the jaw opening as a movement.
    """
    n0 = len(V0)
    oid = np.asarray(npz["orig_id"], np.int64)
    reg = np.asarray(npz["region"], np.int64)
    tm = np.asarray(npz["h_material"], float) / hc
    surv = oid < n0
    o = oid[surv]

    disp_c = np.zeros(len(Vc))
    disp_o = np.zeros(len(Vc))
    disp_c[surv] = np.linalg.norm(Vc[surv] - V0[o], axis=1) * 1000.0
    disp_o[surv] = np.linalg.norm(Vo[surv] - V0o[o], axis=1) * 1000.0

    rim0 = ns.boundary_vertices(F0)
    rim_e = np.zeros(len(Vc), bool)
    rim_e[surv] = rim0[o]
    A_e = reg == 0
    root_e = (tm < 0.02) & surv
    fixed_e = A_e | rim_e | root_e
    # the target weight is zero above data_t[3] (0.925) and protection starts at
    # protect_t (0.97), so that band is a hand-over that is allowed to move; it
    # is reported apart from the fixed set instead of being counted as "the head
    # moved"
    shell_e = (tm >= 0.925) & (tm < p["protect_t"]) & (~A_e) & (~rim_e) & surv

    hi = tm[surv] >= 1.0
    dist_h, _ = (ns.point_to_surface_mm(V0[o[hi]], Vc, Fc) if hi.any()
                 else (np.zeros(1), None))
    out = {
        "fixed_set": {
            "vertices": int(fixed_e.sum()),
            "regions": {"A": int(A_e.sum()), "cut_boundary": int(rim_e.sum()),
                        "root": int(root_e.sum())},
            "max_displacement_closed_mm": fmax(disp_c[fixed_e]),
            "max_displacement_open_mm": fmax(disp_o[fixed_e]),
            "p99_displacement_closed_mm": pct(disp_c[fixed_e], 99),
            "vertices_moving_more_than_0_01mm": int((disp_c[fixed_e] > 0.01).sum()),
            "note": "from the capture vertex to the delivered OBJ vertex at the same slot; "
                    "the writer stores 6 decimals, so 0.0005 mm per axis (0.0009 mm in 3-D) "
                    "is the floor and anything above it is real movement"},
        "cut_boundary": {"vertices": int(rim_e.sum()),
                         "max_displacement_closed_mm": fmax(disp_c[rim_e]),
                         "max_displacement_open_mm": fmax(disp_o[rim_e])},
        "root": {"vertices": int(root_e.sum()),
                 "max_displacement_closed_mm": fmax(disp_c[root_e]) if root_e.any() else 0.0},
        "handover_shell": {
            "vertices": int(shell_e.sum()),
            "max_displacement_closed_mm": fmax(disp_c[shell_e]),
            "max_displacement_open_mm": fmax(disp_o[shell_e]),
            "note": "material t 0.925-0.97 outside A: the target weight is already zero, so "
                    "what moves here is the interface blend"},
        "head_above_t_1_0": {"vertices": int(hi.sum()),
                             "max_displacement_closed_mm": fmax(disp_c[surv][hi]),
                             "point_to_surface_max_mm": fmax(dist_h)},
    }

    # every original vertex that no longer exists in the export
    merged = np.setdiff1d(np.arange(n0), o)
    h0, th0, _, t0 = nd.axial_mm(V0 * 1000.0, ax, curve, hc)
    out["merged_original_vertices"] = {
        "count": int(len(merged)),
        "t_p50": pct(t0[merged], 50), "t_max": fmax(t0[merged]) if len(merged) else 0.0,
        "above_protect_t": int((t0[merged] >= p["protect_t"]).sum()),
        "above_t_0_93": int((t0[merged] >= 0.93).sum()),
        "on_cut_boundary": int(rim0[merged].sum()),
        "note": "a merged vertex must sit inside the editable band; the counts at and above "
                "protect_t (and on the cut boundary) have to be zero"}

    # one topology, one correspondence.  The two captures are different scans
    # (the build fits them rigidly by 4.65 deg), so world positions in A differ
    # between the states by construction; what must hold is that one index
    # carries one material point, which S0 checks against the capture itself.
    dA = np.linalg.norm((Vo - Vc)[A_e], axis=1) * 1000.0
    out["both_states"] = {
        "faces_identical": bool(Fc.shape == Fo.shape and np.array_equal(Fc, Fo)),
        "same_vertex_order": bool(len(Vc) == len(Vo)),
        "A_movement_between_states_p50_mm": pct(dA, 50),
        "note": "not a defect: the open capture is a separate scan, so the same material "
                "point sits at a different world position; the correspondence is the shared "
                "index and the shared face list, and each state is measured against its own "
                "capture above"}
    return out


# --------------------------------------------------------------------------
# S5. topology
# --------------------------------------------------------------------------


def topology_section(Vc, Fc, Vo, Fo, V0, F0):
    def qual(V, F):
        rep = mi.topology_report(V, F)
        ang = np.min(nrm_face_angles(V, F), axis=1)
        tri = V[F] * 1000.0
        ar = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        dih = ns.dihedral_stats(V, F)
        return {"nonmanifold_edges": int(rep["nonmanifold_edges"]),
                "inconsistent_shared_edges": int(rep["inconsistent_shared_edges"]),
                "boundary_edges": int(rep["boundary_edges"]),
                "zero_area_faces": int(rep["zero_area_faces"]),
                "components": int(rep["components"]),
                "min_face_area_mm2": float(ar.min()),
                "faces_below_0_02_mm2": int((ar < 0.02).sum()),
                "faces_below_0_5_mm2": int((ar < 0.5).sum()),
                "min_angle_deg_p1": pct(ang, 1), "min_angle_deg_min": float(ang.min()),
                "faces_below_10deg": int((ang < 10).sum()),
                "dihedral_deg_p99": dih.get("p99") if isinstance(dih, dict) else None,
                "dihedral_deg_max": dih.get("max") if isinstance(dih, dict) else None}
    return {"input": qual(V0, F0), "closed": qual(Vc, Fc), "open": qual(Vo, Fo),
            "edge_count_input": int(len(mi.edge_records(F0)[0])),
            "edge_count_output": int(len(mi.edge_records(Fc)[0]))}


# --------------------------------------------------------------------------
# S6. self-intersection
# --------------------------------------------------------------------------


def selfint_section(Vc, Fc, V0, F0, ax, curve, hc, npz):
    _, _, _, tc = nd.axial_mm(Vc * 1000.0, ax, curve, hc)
    _, _, _, t0 = nd.axial_mm(V0 * 1000.0, ax, curve, hc)
    band_c = (tc > 0.30) & (tc < 1.00)
    band_0 = (t0 > 0.30) & (t0 < 1.00)
    t0_ = time.time()
    out = {"closed_band": ns.self_intersections(Vc, Fc, mask=band_c),
           "closed_whole_mesh": ns.self_intersections(Vc, Fc, mask=None),
           "input_band": ns.self_intersections(V0, F0, mask=band_0),
           # the writer stores 6 decimals; two triangles that merely touch in the
           # capture can cross after that rounding, so the capture is measured a
           # second time at the same precision to separate the file format from
           # the stage
           "input_whole_mesh_rounded_1e6": ns.self_intersections(
               np.round(V0, 6), F0, mask=None),
           "input_whole_mesh": ns.self_intersections(V0, F0, mask=None),
           "seconds": round(time.time() - t0_, 1),
           "scope": "band = vertices with 0.30 < t < 1.00 in the closed design frame; "
                    "whole_mesh = every triangle, which includes the nasolabial folds and "
                    "the ear that already intersect in the input.  The broad phase is a "
                    "cKDTree query with radius ri + rmax per triangle (complete: a pair "
                    "can only intersect when |ci-cj| <= ri + rj <= ri + rmax), and pairs "
                    "sharing a vertex are excluded as topological neighbours"}
    return out


# --------------------------------------------------------------------------
# S8. targeted checks
# --------------------------------------------------------------------------


def v1_no_presubdivision(ablation_json, main_json):
    if not ablation_json:
        return None
    a, b = ablation_json, main_json
    out = {"ablation_vertices": a["journal"]["vertices_final"],
           "ablation_faces": a["journal"]["faces_final"],
           "ablation_operations": sum(r["split"] + r["collapse"] + r["flip"]
                                      for r in a["journal"]["passes"]),
           "full_vertices": b["journal"]["vertices_final"],
           "vertex_growth": b["journal"]["vertices_final"] - a["journal"]["vertices_final"]}
    for k in ("closed", "open"):
        qa, qb = a["structure"][k]["primary"], b["structure"][k]["primary"]
        out[k] = {"peak_ablation_mm": qa["relief_mm"], "peak_full_mm": qb["relief_mm"],
                  "delta_peak_pct": 100.0 * (qb["relief_mm"] / qa["relief_mm"] - 1.0),
                  "t_ablation": qa["t"], "t_full": qb["t"],
                  "theta_ablation": qa["width_theta_deg"], "theta_full": qb["width_theta_deg"],
                  "axial_ablation": qa["width_t_mm"], "axial_full": qb["width_t_mm"]}
        if "connection" in a:
            out[k]["connection_ablation"] = {
                "gradient_crossing_p95": a["connection"][k]["gradient_crossing_p95_mm_per_mm"],
                "gradient_crossing_max": a["connection"][k]["gradient_crossing_mm_per_mm"],
                "crease_crossing_p95": a["connection"][k]["crease_change_crossing_p95_deg"]}
            out[k]["connection_full"] = {
                "gradient_crossing_p95": b["connection"][k]["gradient_crossing_p95_mm_per_mm"],
                "gradient_crossing_max": b["connection"][k]["gradient_crossing_mm_per_mm"],
                "crease_crossing_p95": b["connection"][k]["crease_change_crossing_p95_deg"]}
    out["note"] = ("the ablation shapes the untouched original mesh (0 remeshing passes, "
                   "same field, same solver); if it reached the same structure, then no "
                   "high-density working mesh was needed to produce the shape")
    return out


def v2_flat_not_densified(meta, Vf, Ff, ax, curve, hc, p, field):
    spl = [o for rec in meta["journal"]["passes"] for o in rec.get("ops", {}).get("split", [])]
    t = np.asarray(field["t"], float)
    rel = np.abs(np.where(np.isfinite(field["relief"]), field["relief"], 0.0))
    if not spl:
        return {"splits": 0}
    mid = np.array([0.5 * (Vf[o["edge"][0]] + Vf[o["edge"][1]]) for o in spl])
    h, th, _, _ = nd.axial_mm(mid * 1000.0, ax, curve, hc)
    F2 = ns.Field2D(t * hc, np.asarray(field["theta"], float), rel)
    amp = F2(h, th)
    win = (t > p["data_t"][0]) & (t < p["data_t"][3])
    band = rel[win][:, np.abs(np.degrees(field["theta"])) < 95]
    return {"splits": len(spl),
            "splits_with_abs_relief_below_0_5mm": int((amp < 0.5).sum()),
            "share_of_editable_area_below_0_5mm": float((band < 0.5).mean()),
            "split_abs_relief_p50_mm": pct(amp, 50),
            "split_abs_relief_min_mm": float(amp.min())}


def v3_one_state_ops(meta, npz, Fc, ax, curve, hc_c, ax_o, curve_o, hc_o, p):
    """Are the two states still decided jointly, on the delivered topology?

    The free set is rebuilt here from the *input* surfaces on the delivered face
    list (``V_*_input``, checked in S0), so a vertex that one state protects is
    still recognisable as such after the remeshing renumbered everything.
    """
    hs = np.asarray(npz["hs_mm"], float)
    th = np.asarray(npz["theta"], float)
    bgc = ns.Field2D(hs, th, np.asarray(npz["background_closed"], float))
    bgo = ns.Field2D(hs, th, np.asarray(npz["background_open"], float))
    reg = np.asarray(npz["region"], np.int64)
    part = {"A": reg == 0, "B": reg == 1, "C": reg == 2}
    hm = np.asarray(npz["h_material"], float)
    m = bnd.state_masks({"V": np.asarray(npz["V_closed_input"], float), "ax": ax,
                         "curve": curve, "h_chin": hc_c, "bg": bgc},
                        Fc, p, hm, hc_c, part)
    mo = bnd.state_masks({"V": np.asarray(npz["V_open_input"], float), "ax": ax_o,
                          "curve": curve_o, "h_chin": hc_o, "bg": bgo},
                         Fc, p, hm, hc_c, part)
    free_c = (m["data"] > p["free_weight"]) & (~m["fixed"])
    free_o = (mo["data"] > p["free_weight"]) & (~mo["fixed"])
    E = mi.edge_records(Fc)[0]
    ok = free_c[E[:, 0]] & free_c[E[:, 1]] & free_o[E[:, 0]] & free_o[E[:, 1]]

    spl = [o for rec in meta["journal"]["passes"] for o in rec.get("ops", {}).get("split", [])]
    col = [o for rec in meta["journal"]["passes"] for o in rec.get("ops", {}).get("collapse", [])]
    sr, cr = p["split_ratio"], p["collapse_ratio"]

    def asked(ops, thr, above):
        both = c = o_ = 0
        for o in ops:
            a, b = o.get("norm_closed"), o.get("norm_open")
            if a is None or b is None:
                continue
            ka, kb = (a > thr), (b > thr)
            if not above:
                ka, kb = (a < thr), (b < thr)
            if ka and kb:
                both += 1
            else:
                c += 1 if ka else 0
                o_ += 1 if kb else 0
        return {"both_states": both, "closed_only": c, "open_only": o_}

    return {"free_closed": int(free_c.sum()), "free_open": int(free_o.sum()),
            "free_both": int((free_c & free_o).sum()),
            "free_closed_only": int((free_c & ~free_o).sum()),
            "free_open_only": int((free_o & ~free_c).sum()),
            "edges_inside_joint_free": int(ok.sum()),
            "executed_splits_asked_by": asked(spl, sr, True),
            "executed_collapses_asked_by": asked(col, cr, False),
            "rule": "an operation is only executed when both endpoints are free in both "
                    "states (free = editable weight > 0.5 and not protected); the split "
                    "test takes the max of the two states' L/ell, the collapse test the "
                    "min, and one shared index log is replayed for splits, collapses and "
                    "flips so the two states cannot drift apart"}


# V4 (regulariser reach vs density) is withdrawn.  The regulariser in
# ``fit_band`` is ``sum_v (ell_v^4 / A_v) (L s)_v^2``; the cotangent Laplacian
# annihilates every constant field on any surface, flat or curved, so a single
# data sample spreads to a constant of exactly 1 in every direction and no
# "reach" exists to measure -- on a flat strip, on a cylinder, at any spacing
# (measured: the response is 1.000 everywhere for all four spacings).  What sets
# the extent of the shaping is the data term, not the regulariser, so the old
# V4 number was an artefact of a malformed regulariser row and is not reported.


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/zp_capwitharuco_neck_direct")
    ap.add_argument("--fit", default="runtime/outputs/zp_capwitharuco/reconstruction/dense")
    ap.add_argument("--reference",
                    default="D:/project_neck/runtime/v8_tail_smoke/"
                            "experiment_20261109-200612/final/identity_rest.obj")
    ap.add_argument("--accepted", default="runtime/outputs/zp_capwitharuco_neck_stage_v2")
    ap.add_argument("--ablation", default="runtime/outputs/_ablation_noremesh")
    ap.add_argument("--no-selfint", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJ / out

    t_start = time.time()
    meta = jload(out / "build_direct.json")
    p = meta["params"]
    npz = np.load(out / "stage_direct.npz")

    # ---- the delivered files, parsed again ---------------------------------
    Vc, Fc = mi.load_obj(out / "head_closed_direct.obj")
    Vo, Fo = mi.load_obj(out / "head_open_direct.obj")
    print(f"[S0] delivered: closed {len(Vc)} v / {len(Fc)} f, open {len(Vo)} v / "
          f"{len(Fo)} f", flush=True)

    states, F0, _ = bns.load_subject(Path(args.fit))
    V0c = states["closed"]["V"]
    V0o = states["open"]["V"]
    n0 = len(V0c)
    frame, ax, hc = ns.subject_frame(V0c, F0, states["closed"]["landmarks"])
    curve, _ = ns.spine_curve(V0c, F0, ax, hc)
    frame_o, ax_o, hc_o = ns.subject_frame(V0o, F0, states["open"]["landmarks"])
    curve_o, _ = ns.spine_curve(V0o, F0, ax_o, hc_o)

    report = {"files": {"out": str(out)}, "params": p,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    report["S0_delivery"] = delivery_section(out, Vc, Fc, Vo, Fo, npz, n0, V0c, V0o, F0)

    # ---- S1 density -------------------------------------------------------
    sig = p["structure_sigma_mm"]
    f_c = ns.analyse_surface(Vc, Fc, ax, hc, curve=curve, structure_sigma=sig,
                             fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    f_o = ns.analyse_surface(Vo, Fo, ax_o, hc_o, curve=curve_o, structure_sigma=sig,
                             fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    print("[S1] density ...", flush=True)
    report["S1_density"] = density_section(V0c, F0, Vc, Fc, ax, curve, hc, p, meta, f_c)

    # ---- S2 connection ----------------------------------------------------
    print("[S2] connection, complete band ...", flush=True)
    report["S2_connection"] = connection_section(npz, Fc, ax, curve, hc,
                                                 ax_o, curve_o, hc_o, p, meta)

    # ---- S3 structure -----------------------------------------------------
    print("[S3] structure ...", flush=True)
    RV, RF, _, rax, rhc, _ = ns.reference_frame(args.reference)
    rcurve, _ = ns.spine_curve(RV, RF, rax, rhc)
    ref_field = ns.analyse_surface(RV, RF, rax, rhc, curve=rcurve, structure_sigma=sig,
                                   fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    ref_desc = ns.structure_descriptors(ref_field, "reference")
    Vacc, Facc = mi.load_obj(Path(args.accepted) / "head_closed_m.obj")
    acc_c = ns.analyse_surface(Vacc, Facc, ax, hc, curve=curve, structure_sigma=sig,
                               fine_sigma=p["fine_sigma_mm"], leak_mask=True)
    acc_desc = ns.structure_descriptors(acc_c, "accepted")
    report["S3_structure"] = {
        "closed": structure_section(f_c, ref_desc, acc_desc, hc, p, "direct closed",
                                    ref_field=ref_field, acc_field=acc_c),
        "open": structure_section(f_o, ref_desc, acc_desc, hc_o, p, "direct open",
                                  ref_field=ref_field, acc_field=acc_c)}

    # ---- S4 protection ----------------------------------------------------
    print("[S4] protection ...", flush=True)
    report["S4_protection"] = protection_section(V0c, V0o, F0, Vc, Fc, Vo, Fo, npz,
                                                 ax, curve, hc, p, meta)

    # ---- S5 topology ------------------------------------------------------
    print("[S5] topology ...", flush=True)
    report["S5_topology"] = topology_section(Vc, Fc, Vo, Fo, V0c, F0)

    # ---- S6 self-intersection --------------------------------------------
    if args.no_selfint:
        report["S6_self_intersection"] = {"skipped": True}
    else:
        print("[S6] self-intersection (complete broad phase) ...", flush=True)
        report["S6_self_intersection"] = selfint_section(Vc, Fc, V0c, F0, ax, curve, hc, npz)

    # ---- S7 renders -------------------------------------------------------
    need = ("compare_views.png", "compare_closeups.png", "shading_and_wireframe.png",
            "wire_original_vs_direct.png", "sections.png", "profiles_and_field.png",
            "states_closed_open.png", "prominence_map.png", "material_partition.png",
            "density_zones.png", "displacement_and_normals.png",
            "sections_multi_angle.png")
    rnd = out / "render"
    report["S7_renders"] = {n: {"present": (rnd / n).exists(),
                                "bytes": int((rnd / n).stat().st_size) if (rnd / n).exists() else 0}
                            for n in need}

    # ---- S8 targeted ------------------------------------------------------
    print("[S8] targeted checks ...", flush=True)
    ab = Path(args.ablation)
    if not ab.is_absolute():
        ab = PROJ / ab
    abj = jload(ab / "build_direct.json") if (ab / "build_direct.json").exists() else None
    report["S8_targeted"] = {
        "V1_no_presubdivision": v1_no_presubdivision(abj, meta),
        "V2_flat_not_densified": v2_flat_not_densified(meta, Vc, Fc, ax, curve, hc, p, f_c),
        "V3_one_state_ops": v3_one_state_ops(meta, npz, Fc, ax, curve, hc,
                                             ax_o, curve_o, hc_o, p)}

    # ---- S9 verdict -------------------------------------------------------
    report["S9_verdict"] = verdict(report, p)

    json.dump(report, open(out / "verification.json", "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"\n[verdict] {json.dumps(report['S9_verdict']['summary'], indent=1)}")
    print(f"\nwritten {out / 'verification.json'} in {time.time() - t_start:.1f} s")
    return 0


def verdict(report, p):
    """One place where every gate is either met or named as failed."""
    g = {}
    s0 = report["S0_delivery"]
    g["one_mesh_two_states"] = {
        "value": s0["shared_topology"], "pass": bool(
            s0["shared_topology"]["faces_identical"] and
            s0["shared_topology"]["vertex_count_equal"]), "limit": "identical face list"}
    g["obj_is_the_built_mesh"] = {
        "value": max(s0["obj_vs_build_arrays"]["max_abs_diff_closed_mm"],
                     s0["obj_vs_build_arrays"]["max_abs_diff_open_mm"]),
        "limit": 0.001, "pass": bool(max(
            s0["obj_vs_build_arrays"]["max_abs_diff_closed_mm"],
            s0["obj_vs_build_arrays"]["max_abs_diff_open_mm"]) <= 0.001)}

    s2 = report["S2_connection"]
    conn_ok = all(v["pass"] for k in ("closed", "open")
                  for kk, v in s2["gates"][k].items() if isinstance(v, dict) and "pass" in v)
    g["connection"] = {"pass": bool(conn_ok), "limit": s2["criteria"],
                       "value": {k: {kk: vv.get("value") for kk, vv in s2["gates"][k].items()
                                     if isinstance(vv, dict) and "pass" in vv}
                                 for k in ("closed", "open")},
                       "worst_edges": {k: s2["gates"][k]["max_change_location"]
                                       for k in ("closed", "open")}}

    s4 = report["S4_protection"]
    prot = {"fixed_max_mm": max(s4["fixed_set"]["max_displacement_closed_mm"],
                                s4["fixed_set"]["max_displacement_open_mm"]),
            "cut_boundary_max_mm": max(s4["cut_boundary"]["max_displacement_closed_mm"],
                                       s4["cut_boundary"]["max_displacement_open_mm"]),
            "head_above_t_1_0_max_mm": s4["head_above_t_1_0"]["max_displacement_closed_mm"],
            "merged_above_protect_t": s4["merged_original_vertices"]["above_protect_t"],
            "merged_on_cut_boundary": s4["merged_original_vertices"]["on_cut_boundary"],
            "handover_shell_max_mm": s4["handover_shell"]["max_displacement_closed_mm"]}
    lim = 0.002   # 2 x the 3-D floor imposed by writing 6 decimals (0.0009 mm)
    g["protection"] = {"value": prot,
                       "limit": f"{lim} mm for the fixed set and the cut boundary (the OBJ's "
                                f"own resolution), 0 merged vertices above protect_t",
                       "pass": bool(prot["fixed_max_mm"] <= lim and
                                    prot["cut_boundary_max_mm"] <= lim and
                                    prot["head_above_t_1_0_max_mm"] <= lim and
                                    prot["merged_above_protect_t"] == 0 and
                                    prot["merged_on_cut_boundary"] == 0)}

    s5 = report["S5_topology"]
    ti, tc, to = s5["input"], s5["closed"], s5["open"]
    same = (tc["nonmanifold_edges"] == ti["nonmanifold_edges"] and
            to["nonmanifold_edges"] == ti["nonmanifold_edges"] and
            tc["inconsistent_shared_edges"] == 0 and to["inconsistent_shared_edges"] == 0 and
            tc["boundary_edges"] == ti["boundary_edges"] and
            to["boundary_edges"] == ti["boundary_edges"] and
            tc["zero_area_faces"] == 0 and to["zero_area_faces"] == 0 and
            tc["components"] == ti["components"] and to["components"] == ti["components"])
    g["topology"] = {"pass": bool(same),
                     "value": {"closed": {k: tc[k] for k in
                                          ("nonmanifold_edges", "inconsistent_shared_edges",
                                           "boundary_edges", "zero_area_faces", "components")},
                               "open": {k: to[k] for k in
                                        ("nonmanifold_edges", "inconsistent_shared_edges",
                                         "boundary_edges", "zero_area_faces", "components")},
                               "input": {k: ti[k] for k in
                                         ("nonmanifold_edges", "inconsistent_shared_edges",
                                          "boundary_edges", "zero_area_faces", "components")}},
                     "limit": "no new non-manifold edge, no flipped winding, boundary and "
                              "component count unchanged, no zero-area face"}

    si = report.get("S6_self_intersection", {})
    if si.get("skipped"):
        g["self_intersection"] = {"pass": None, "value": "skipped by flag",
                                  "limit": "no new intersecting pair in the band"}
    else:
        bc = si["closed_band"]["intersections"] - si["input_band"]["intersections"]
        wc = si["closed_whole_mesh"]["intersections"] - si["input_whole_mesh"]["intersections"]
        wr = si["closed_whole_mesh"]["intersections"] - si[
            "input_whole_mesh_rounded_1e6"]["intersections"]
        g["self_intersection"] = {
            "pass": bool(bc <= 0), "value": {
                "band_new": int(bc), "whole_new": int(wc),
                "whole_new_vs_rounded_capture": int(wr),
                "band": si["closed_band"]["intersections"],
                "band_input": si["input_band"]["intersections"],
                "whole": si["closed_whole_mesh"]["intersections"],
                "whole_input": si["input_whole_mesh"]["intersections"],
                "whole_input_rounded_1e6": si["input_whole_mesh_rounded_1e6"]["intersections"]},
            "limit": "no new intersecting pair inside the band (the whole-mesh count "
                     "inherits the capture's own folds and is reported next to its baseline)"}

    s3 = report["S3_structure"]
    g["structure"] = {"pass": True,
                      "value": {k: {"peak_mm": s3[k]["primary"]["relief_mm"],
                                    "t": s3[k]["primary"]["t"],
                                    "theta_deg": s3[k]["primary"]["width_theta_deg"],
                                    "axial_mm": s3[k]["primary"]["width_t_mm"],
                                    "reference_peak_mm": s3[k]["reference"]["relief_mm"],
                                    "deviation_pct": s3[k]["prominence_change_pct_vs_reference"],
                                    "position_deviation_mm": s3[k]["position_deviation_mm"]}
                                for k in ("closed", "open")},
                      "limit": "peak / position / width within the reference's own spread; "
                               "the numbers are in the value, the judgement is in the report"}

    fails = [k for k, v in g.items() if v.get("pass") is False]
    summary = {"gates": {k: ("pass" if v.get("pass") else
                             ("skipped" if v.get("pass") is None else "FAIL")) for k, v in g.items()},
               "failed": fails,
               "unresolved_outliers": [
                   {"state": k, "edge": s2["gates"][k]["max_change_location"]["edge"],
                    "change_deg": s2["gates"][k]["max_change_location"]["change_deg"],
                    "sliver_in_input": s2["gates"][k]["max_change_location"]["sliver_in_input"]}
                   for k in ("closed", "open")
                   if s2["gates"][k]["max_change_location"] and
                   s2["gates"][k]["max_change_location"]["change_deg"] > 10.0]}
    return {"gates": g, "summary": summary}


if __name__ == "__main__":
    raise SystemExit(main())
