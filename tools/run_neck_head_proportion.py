"""Head-proportion audit and closest-safe two-feature delivery (0.8 candidate).

A: public 0.7.0 delivery (loaded from disk).
B: head-scale reference ratio target (full W/H factors) at the current centre.
C: closest safe: reference main height and secondary retained, lateral/longitudinal
   support adapted to protection, true bottom cut and geometry guards.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from head_rgbd import head_scale as hs
from head_rgbd import neck_dual as nd
from head_rgbd import neck_prominence as npn
from head_rgbd.mesh_integrity import export_obj, load_obj, topology_report
from tools.run_neck_continuous_fix import (render_heat, render_wire, outer_section,
                                          _facial_check, _boundary_set, sha256)
from tools.run_neck_reference_auto import render_flat, render_smooth

PROJECT = Path("D:/proiect/facecap_airway_delivery")
ASSET = PROJECT / "assets/neck_reference/identity_rest_20261109.obj"
FIT = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
BASE = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/delivery"
A070 = PROJECT / "runtime/outputs/neck_primary_secondary_20261112_200000/final"
A060 = PROJECT / "runtime/outputs/neck_continuous_fix_20261112_190000/final"
TARGET_CENTER = 15571


def _morph_axes(m, vertical, lateral, height):
    d = copy.deepcopy(m)
    for key in ("feature_A", "feature_B"):
        d[key]["height_mm"] *= height
        d[key]["v_mm"] *= vertical
    d["saddle"]["height_mm"] *= height
    d["saddle"]["v_mm"] *= vertical
    d["relative"]["centre_distance_mm"] *= vertical
    d["relative"]["B_over_A"] = d["feature_B"]["height_mm"] / d["feature_A"]["height_mm"]
    d["relative"]["A_over_saddle"] = d["feature_A"]["height_mm"] - d["saddle"]["height_mm"]
    d["relative"]["B_over_saddle"] = d["feature_B"]["height_mm"] - d["saddle"]["height_mm"]
    d["saddle"]["relative_to_A"] = d["saddle"]["height_mm"] / d["feature_A"]["height_mm"]
    for key in ("A", "B"):
        d["lateral"][key]["half_mm"] *= lateral
        d["lateral"][key]["tip_mm"] *= lateral
        d["lateral"][key]["w50_row_mm"] *= lateral
    for key in ("v_top_mm", "v_bottom_mm"):
        d["support"][key] *= vertical
    return d


def _reference_l50(m):
    design = nd.design_from_morph(m, 1.0)
    v = np.linspace(design.v_bottom_mm, design.v_top_mm, 801)
    h = design.profile(v)
    # A: contiguous 50% segment around v=0
    msk = h >= .5
    ia = int(np.argmin(np.abs(v)))
    if not msk[ia]:
        return None
    lo = ia
    while lo > 0 and msk[lo - 1]:
        lo -= 1
    hi = ia
    while hi + 1 < len(v) and msk[hi + 1]:
        hi += 1
    A = float(v[hi] - v[lo])
    # B: contiguous half of its own prominence above the saddle
    b = design.v_b_mm
    s = design._saddle_v()
    hb = design.profile(b)
    hs_val = design.profile(s)
    half = hs_val + .5 * (hb - hs_val)
    msk = h >= half
    ib = int(np.argmin(np.abs(v - b)))
    lo = ib
    while lo > 0 and msk[lo - 1]:
        lo -= 1
    hi = ib
    while hi + 1 < len(v) and msk[hi + 1]:
        hi += 1
    B = float(v[hi] - v[lo])
    return A, B


def _front_field(V, faces, chin, up, right, fwd, sigma_mm, v_min=-105.0, v_max=-25.0):
    """Final/base surface height field in the head-local frame, same protocol."""
    Vm = V * 1000.0
    rel = Vm - chin * 1000.0
    u = rel @ right; v = rel @ up; f = rel @ fwd
    N = npn.vertex_normals(V, faces)
    front = (N @ fwd) > .15
    take = front & (v > v_min) & (v < v_max) & (np.abs(u) < 40.)
    if take.sum() < 100:
        return None
    uu = np.arange(-32., 32.01, .5)
    vv = np.arange(v_min, v_max + 1e-9, .5)
    U, Vg = np.meshgrid(uu, vv)
    try:
        interp = nd.LinearNDInterpolator(np.c_[u[take], v[take]], f[take])
        Fh = interp(U, Vg)
    except Exception:
        return None
    valid = np.isfinite(Fh)
    if valid.mean() < .55:
        return None
    fill = np.where(valid, Fh, np.nanmedian(Fh))
    R = Fh - gaussian_filter(fill, sigma_mm / .5)
    prof = np.nanmedian(np.where(valid, R, np.nan), axis=1)
    pk, props = find_peaks(np.nan_to_num(prof, nan=-1e9), prominence=.05, distance=12)
    if len(pk) < 1:
        return None
    order = pk[np.argsort(-prof[pk])]
    A = int(order[0])
    Bs = [p for p in order[1:] if 8.0 <= vv[A] - vv[p] <= 45.0]
    B = int(Bs[0]) if Bs else None
    out = {"A": {"v_mm": round(float(vv[A]), 2), "height_mm": round(float(prof[A]), 3)}}
    if B is not None:
        lo, hi = sorted((A, B))
        s = int(lo + np.argmin(prof[lo:hi + 1]))
        out["B"] = {"v_mm": round(float(vv[B]), 2), "height_mm": round(float(prof[B]), 3)}
        out["saddle"] = {"v_mm": round(float(vv[s]), 2), "height_mm": round(float(prof[s]), 3)}
        out["relative"] = {"B_over_A": round(float(prof[B] / prof[A]), 4),
                           "saddle_over_A": round(float(prof[s] / prof[A]), 4),
                           "centre_distance_mm": round(float(abs(vv[B] - vv[A])), 2),
                           "B_local_mm": round(float(prof[B] - prof[s]), 3)}
    # widths at A (row within 1.5 mm), half-level above the lateral edge
    def width(ind):
        vc = vv[ind]
        rows = R[np.abs(vv - vc) <= 1.5]
        vrows = valid[np.abs(vv - vc) <= 1.5]
        if not rows.size:
            return None
        row = np.nanmedian(np.where(vrows, rows, np.nan), axis=0)
        finite = np.isfinite(row)
        if finite.sum() < 10:
            return None
        edge_vals = row[(np.abs(uu) >= 18) & finite]
        edge = float(np.median(edge_vals)) if len(edge_vals) else float(np.nanmin(row))
        half = edge + .5 * (np.nanmax(row) - edge)
        take_u = np.flatnonzero(finite & (row >= half))
        return None if not len(take_u) else {"w50_mm": round(float(uu[take_u.max()] - uu[take_u.min()]), 2),
                                             "u_mm": [round(float(uu[take_u.min()]), 2), round(float(uu[take_u.max()]), 2)]}
    out["A"]["width"] = width(A)
    if B is not None:
        out["B"]["width"] = width(B)
    # longitudinal A extent where final field >= 0.5 A height (contiguous around A)
    msk = np.nan_to_num(prof, nan=-1e9) >= .5 * prof[A]
    lo = A
    while lo > 0 and msk[lo - 1]:
        lo -= 1
    hi = A
    while hi + 1 < len(vv) and msk[hi + 1]:
        hi += 1
    out["A"]["l50_mm"] = round(float(vv[hi] - vv[lo]), 2)
    # tip radius from the lateral row near the apex (quadratic)
    vc = vv[A]
    row = np.nanmedian(np.where(valid[np.abs(vv - vc) <= 1.0], R[np.abs(vv - vc) <= 1.0], np.nan), axis=0)
    ok = np.isfinite(row) & (np.abs(uu) <= 15.)
    if ok.sum() >= 7:
        co = np.polyfit(uu[ok], row[ok], 2)
        if co[0] < -1e-6:
            out["A"]["tip_radius_mm"] = round(float(1.0 / (2.0 * abs(co[0]))), 2)
    out["sigma_mm"] = sigma_mm
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=ASSET)
    ap.add_argument("--fit-dir", type=Path, default=FIT)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"neck_head_proportion_{stamp}")
    for sub in ("final", "render", "experiments/A_070", "experiments/B_head_full", "experiments/C_safe"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    status, error = "failed", None
    t0 = time.perf_counter()
    timing = []
    try:
        refs = json.loads((args.fit_dir / "reference_geometry.json").read_text())["references"]
        with np.load(args.fit_dir / "parameters.npz", allow_pickle=False) as z:
            dense_v, dense_f = z["dense_vertices"], z["dense_faces"]
        # reference / target head
        refV, refF = load_obj(args.reference)
        ref_head = hs.reference_head_dimensions(refV * 1000.0)
        st = "closed"; ref = refs[st]
        V_closed = dense_v[ref["stages"]["dense"]["frame_index"]]
        M = np.asarray(ref["animated_model_to_reference_camera_4x4"]); R, t = M[:3, :3], M[:3, 3]
        lm = np.asarray(ref["stages"]["dense"]["landmarks_m"]); lmm = (lm - t) @ R
        axes = npn._direction_frame(lmm)
        up, right, fwd = R @ axes["up"], R @ axes["right"], R @ axes["forward"]
        chin = lmm[npn.CHIN] @ R.T + t
        tgt_head = hs.target_head_dimensions(V_closed, up, right, fwd, chin)
        scale = hs.head_scale(ref_head, tgt_head)
        morph = nd.extract_two_feature(args.reference)
        l50 = _reference_l50(morph)
        targets = hs.reference_ratio_targets(morph, ref_head, tgt_head,
                                             reference_l50_a=l50[0], reference_l50_b=l50[1],
                                             reference_w50_b=36.5, reference_tip_a=6.855)
        wr = tgt_head["W_head_mm"] / ref_head["W_head_mm"]
        hr = tgt_head["H_head_mm"] / ref_head["H_head_mm"]
        # reference asset with provenance
        asset = {
            "schema": "head_rgbd.neck_reference.asset.v1",
            "asset_path": str(args.reference.resolve()),
            "asset_sha256": sha256(args.reference),
            "original_source": "D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj",
            "original_sha256": morph["source_sha256"],
            "axes": morph["axes"], "units": "m",
            "head": ref_head,
            "features": {"A": morph["feature_A"], "B": morph["feature_B"], "saddle": morph["saddle"],
                         "relative": morph["relative"], "support": morph["support"], "lateral": morph["lateral"]},
            "reference_l50_A_mm": l50[0], "reference_l50_B_mm": l50[1],
            "background": morph["background"], "sensitivity": morph["sensitivity"],
            "note": "project-owned traceable copy of the reference; the public 0.7.0 default uses this copy",
        }
        (out / "reference_asset.json").write_text(json.dumps(asset, indent=1, ensure_ascii=False), encoding="utf-8")
        (PROJECT / "assets/neck_reference/identity_rest_20261109_asset.json").write_text(
            json.dumps(asset, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / "head_scale_audit.json").write_text(json.dumps(scale, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / "reference_ratio_targets.json").write_text(json.dumps(targets, indent=1, ensure_ascii=False), encoding="utf-8")
        # A 0.7.0
        A = {}
        for s in ("closed", "open"):
            src = A070 / f"head_{s}_m.obj"; dst = out / "experiments" / "A_070" / f"head_{s}_m.obj"
            shutil.copy2(src, dst); A[s] = load_obj(dst)
        # B full head-scale at the old centre; C closest safe
        builds = {}
        for label, vertical, lateral in (("B_head_full", hr, wr), ("C_safe", .68, 1.0)):
            m = _morph_axes(morph, vertical, lateral, wr)
            plan = nd.build_states(references=refs, dense_vertices=dense_v, dense_faces=dense_f,
                                   reference_path=args.reference, fixed_center_vertex=TARGET_CENTER,
                                   q_override=1.0, f_override=1.0, morph_override=m,
                                   require_support=False, direction_modes=("constant", "smooth"))
            builds[label] = plan
            exp = out / "experiments" / label
            for s in ("closed", "open"):
                export_obj(exp / f"head_{s}_m.obj", plan["states"][s]["run"]["vertices"], plan["faces"])
        C = builds["C_safe"]
        Bfull = builds["B_head_full"]
        # base surface for calibration/final metrics
        base_mesh = {s: load_obj(BASE / f"head_{s}_m.obj") for s in ("closed", "open")}
        base_v, base_f = base_mesh["closed"]
        sigma = 8.0 * scale["S_head_geometric"]
        target_h = targets["target_mm"]["H_A_mm"]
        base_rep = None
        calib = []
        safe_C = None
        safe_feat = None
        best_factor = None
        for factor in (1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6):
            if abs(factor - 1.2) > 1e-9:
                m = _morph_axes(morph, .68, 1.0, factor)
                C = nd.build_states(references=refs, dense_vertices=dense_v, dense_faces=dense_f,
                                    reference_path=args.reference, fixed_center_vertex=TARGET_CENTER,
                                    q_override=1.0, f_override=1.0, morph_override=m,
                                    require_support=False, direction_modes=("constant", "smooth"))
            guards_ok = all(nd.nc.na.accept_guards(C["states"][st]["guards"]) and
                            C["states"][st]["intersections"]["count"] == 0 for st in ("closed", "open"))
            f1 = _front_field(C["states"]["closed"]["run"]["vertices"], C["faces"],
                              chin, up, right, fwd, sigma, v_min=-80.0, v_max=-25.0)
            final_a = None if not f1 or "A" not in f1 else float(f1["A"]["height_mm"])
            sup_ok = all(C["audits"][st]["support"]["inside_protected"] == 0 and
                         C["audits"][st]["support"]["gate_min"] >= .98 and
                         C["audits"][st]["support"]["bottom_margin_mm"] >= 8.0
                         for st in ("closed", "open"))
            calib.append({"height_factor": factor, "final_A_mm": None if final_a is None else round(final_a, 4),
                          "guards_ok": bool(guards_ok), "support_ok": bool(sup_ok),
                          "target_A_mm": round(target_h, 4)})
            if not guards_ok or not sup_ok:
                break
            safe_C, safe_feat, best_factor = C, f1, factor
            if final_a is not None and final_a >= target_h:
                break
        if safe_C is None:
            raise RuntimeError("no safe height factor for C")
        C, C_feat_initial, height_factor = safe_C, safe_feat, best_factor
        builds["C_safe"] = C
        # final export + disk
        disk = {}
        for s in ("closed", "open"):
            path = out / "final" / f"head_{s}_m.obj"
            export_obj(path, C["states"][s]["run"]["vertices"], C["faces"])
            v, f = load_obj(path)
            rep = topology_report(v, f)
            disk[s] = {"path": str(path.resolve()), "sha256": sha256(path), "vertices": len(v), "faces": len(f),
                       "max_abs_diff_m": float(np.max(np.abs(v - C["states"][s]["run"]["vertices"]))),
                       "faces_match": bool(np.array_equal(f, C["faces"])),
                       "topology": {k: rep[k] for k in ("finite", "components", "nonmanifold_edges",
                                                        "inconsistent_shared_edges", "boundary_edges",
                                                        "zero_area_faces")},
                       "no_materials": not any(k in path.read_text(encoding="utf-8") for k in ("mtllib", "usemtl"))}
        base_rep = topology_report(base_v, base_f); base_boundary = _boundary_set(base_f)
        # final surface metrics (same protocol, head-scale sigma)
        base_feat = _front_field(base_v, base_f, chin, up, right, fwd, sigma, v_min=-80.0, v_max=-25.0)
        c_feat = C_feat_initial
        a_feat = _front_field(A["closed"][0], A["closed"][1], chin, up, right, fwd, sigma, v_min=-80.0, v_max=-25.0)
        a060_v, a060_f = load_obj(A060 / "head_closed_m.obj")
        a060_feat = _front_field(a060_v, a060_f, chin, up, right, fwd, sigma, v_min=-80.0, v_max=-25.0)
        def ratio(feat, key, head_key, head):
            try:
                return float(feat[key]) / head[head_key]
            except (KeyError, TypeError):
                return None
        final_ratios = {}
        if c_feat:
            final_ratios = {
                "H_A_over_W_head": ratio(c_feat["A"], "height_mm", "W_head_mm", tgt_head),
                "W50_A_over_W_head": (c_feat["A"].get("width") or {}).get("w50_mm", 0) / tgt_head["W_head_mm"]
                if c_feat["A"].get("width") else None,
                "L50_A_over_H_head": ratio(c_feat["A"], "l50_mm", "H_head_mm", tgt_head),
                "H_B_over_W_head": ratio((c_feat.get("B") or {}), "height_mm", "W_head_mm", tgt_head),
                "B_local_prominence_over_W_head": (c_feat.get("relative") or {}).get("B_local_mm", 0) / tgt_head["W_head_mm"],
                "distance_AB_over_H_head": (c_feat.get("relative") or {}).get("centre_distance_mm", 0) / tgt_head["H_head_mm"],
                "tip_radius_A_over_W_head": ratio(c_feat["A"], "tip_radius_mm", "W_head_mm", tgt_head),
                "B_over_A": (c_feat.get("relative") or {}).get("B_over_A"),
                "saddle_over_A": (c_feat.get("relative") or {}).get("saddle_over_A"),
            }
        ref_ratios = targets["reference_ratios"]; target_mm = targets["target_mm"]
        tol = targets["tolerances"]
        proportion = {}
        for key, refval in (("H_A_over_W_head", ref_ratios["H_A_over_W_head"]),
                            ("W50_A_over_W_head", ref_ratios["W50_A_over_W_head"]),
                            ("L50_A_over_H_head", ref_ratios["L50_A_over_H_head"]),
                            ("H_B_over_W_head", ref_ratios["H_B_over_W_head"]),
                            ("B_local_prominence_over_W_head", ref_ratios["B_local_prominence_over_W_head"]),
                            ("distance_AB_over_H_head", ref_ratios["distance_AB_over_H_head"]),
                            ("tip_radius_A_over_W_head", ref_ratios["tip_radius_A_over_W_head"])):
            val = final_ratios.get(key)
            allowed = tol["main_height_width_ratio"] if key in ("H_A_over_W_head", "W50_A_over_W_head", "tip_radius_A_over_W_head") else tol["distance_L50"] if key in ("L50_A_over_H_head", "distance_AB_over_H_head") else tol["B_A_and_saddle"]
            if val is None:
                proportion[key] = {"reference": refval, "final": None, "pass": False,
                                   "tolerance": allowed, "rel_error": None}
                continue
            proportion[key] = {"reference": round(float(refval), 5), "final": round(float(val), 5),
                               "rel_error": round(float(abs(val - refval) / max(abs(refval), 1e-9)), 4),
                               "tolerance": allowed,
                               "pass": bool(abs(val - refval) / max(abs(refval), 1e-9) <= allowed)}
        proportion_pass = all(v["pass"] for v in proportion.values())
        final_metrics = {
            "schema": "head_rgbd.neck_head_proportion.final_surface.v1",
            "head_scale": scale,
            "reference_surface": {"feature_A": morph["feature_A"], "feature_B": morph["feature_B"],
                                  "saddle": morph["saddle"], "relative": morph["relative"],
                                  "l50_A_mm": l50[0], "l50_B_mm": l50[1], "w50_A_mm": 32.5,
                                  "tip_radius_A_mm": 6.855},
            "target_base_surface": base_feat,
            "target_060_surface": a060_feat,
            "target_070_surface": a_feat,
            "target_C_surface": c_feat,
            "target_C_final_ratios": {k: (None if v is None else round(float(v), 5)) for k, v in final_ratios.items()},
            "reference_ratios": {k: round(float(v), 5) for k, v in ref_ratios.items()},
            "target_ratio_targets_mm": {k: round(float(v), 3) for k, v in target_mm.items()},
            "proportion_checks": proportion,
            "proportion_pass": bool(proportion_pass),
            "sigma_mm": sigma,
            "method": "final mesh anterior height field in the head-local frame, Gaussian background "
                      "8 mm * S_head; base, 0.7.0 and C measured identically; the design increment is "
                      "reported separately from the final surface",
        }
        (out / "final_surface_metrics.json").write_text(json.dumps(final_metrics, indent=1, ensure_ascii=False,
                                                                  default=float), encoding="utf-8")
        # geometry checks for C
        facial, checks, states_v = {}, {}, {}
        for s in ("closed", "open"):
            item = C["states"][s]; a = C["audits"][s]
            bv, bf = base_mesh[s]
            facial[s] = _facial_check(bv, bf, item["run"]["vertices"], C["faces"],
                                      item["reference"]["stages"]["dense"]["landmarks_m"], item["protection"])
            nref = npn.vertex_normals(item["vertices"], C["faces"])
            moved = np.abs(item["run"]["applied_mm"]) > 1e-9
            states_v[s] = {"audit": a, "facial": facial[s],
                           "backward_moved": int((moved & ((nref @ item["frame"].n) <= 0)).sum()),
                           "protected_moved": int((np.abs(item["run"]["applied_mm"][item["protected"]]) > 1e-9).sum()),
                           "boundary_same": _boundary_set(C["faces"]) == base_boundary}
        checks.update({
            "finite_coordinates": all(disk[s]["topology"]["finite"] for s in disk),
            "consistent_winding": all(disk[s]["topology"]["inconsistent_shared_edges"] == 0 for s in disk),
            "no_new_nonmanifold": all(disk[s]["topology"]["nonmanifold_edges"] == 0 for s in disk),
            "boundary_preserved": all(disk[s]["topology"]["boundary_edges"] == base_rep["boundary_edges"] for s in disk),
            "no_new_holes_same_boundary": all(states_v[s]["boundary_same"] for s in states_v),
            "no_degenerate_faces": all(disk[s]["topology"]["zero_area_faces"] == 0 for s in disk),
            "local_self_intersections_none": all(C["states"][s]["intersections"]["count"] == 0 for s in ("closed", "open")),
            "protected_vertices_unchanged": all(states_v[s]["protected_moved"] == 0 for s in states_v),
            "closed_open_topology_match": all(disk[s]["faces_match"] for s in disk)
                                             and disk["closed"]["faces"] == disk["open"]["faces"],
            "disk_memory_consistent": all(disk[s]["max_abs_diff_m"] <= 1e-9 and disk[s]["faces_match"] for s in disk),
            "geometry_guards_pass": all(nd.nc.na.accept_guards(C["states"][s]["guards"]) for s in ("closed", "open")),
            "no_backward_facing_moved": all(states_v[s]["backward_moved"] == 0 for s in states_v),
            "facial_first_hit_unchanged": all(facial[s]["protected_face_hits"] > 0 and
                                              facial[s]["max_abs_shift_mm"] is not None and
                                              facial[s]["max_abs_shift_mm"] <= .05 for s in facial),
            "nonnegative_outward_displacement": all(states_v[s]["audit"]["base_relative"]["min_outward_mm"] >= -.05
                                                     for s in states_v),
            "no_new_negative_area": all(states_v[s]["audit"]["base_relative"]["negative_face_area_mm2"] == 0
                                        for s in states_v),
            "two_features_detected_both_states": all(states_v[s]["audit"]["features"].get("detected") for s in states_v),
            "feature_ratios_match_reference": all(
                abs(states_v[s]["audit"]["features"]["relative"]["B_over_A"] - morph["relative"]["B_over_A"]) <= .15
                and abs(states_v[s]["audit"]["features"]["relative"]["saddle_over_A"] - morph["saddle"]["relative_to_A"]) <= .15
                for s in states_v),
            "secondary_not_noise": all(states_v[s]["audit"]["features"]["B"]["prominence_mm"] >= .08
                                       and states_v[s]["audit"]["features"]["B_stable_offsets"] >= 3 for s in states_v),
            "support_connected_one_domain": all(states_v[s]["audit"]["levels"]["0.1"]["component_count"] == 1
                                                and states_v[s]["audit"]["levels"]["0.5"]["component_count"] == 1
                                                for s in states_v),
            "support_complete_protection": all(states_v[s]["audit"]["support"]["inside_protected"] == 0
                                                and states_v[s]["audit"]["support"]["clearance_min_mm"] >= nd.MIN_CLEARANCE_MM
                                                and states_v[s]["audit"]["support"]["gate_min"] >= .98 for s in states_v),
            "bottom_cut_margin": all(states_v[s]["audit"]["support"]["bottom_margin_mm"] >= 8.0 for s in states_v),
            "main_height_ratio_target": bool(proportion["H_A_over_W_head"]["pass"]),
            "main_width_ratio_target": bool(proportion["W50_A_over_W_head"]["pass"]),
            "longitudinal_distance_target": bool(proportion["L50_A_over_H_head"]["pass"] and
                                                 proportion["distance_AB_over_H_head"]["pass"]),
            "proportion_conditions_pass": bool(proportion_pass),
            "unit_m_and_no_materials": all(disk[s]["no_materials"] for s in disk),
        })
        accepted_geometry = all(v for k, v in checks.items() if k not in ("main_height_ratio_target",
                                                                          "main_width_ratio_target",
                                                                          "longitudinal_distance_target",
                                                                          "proportion_conditions_pass"))
        # manifest + JSONs
        shape = {"schema": "head_rgbd.neck_head_proportion.shape.v1", "version": "0.8.0-candidate",
                 "reference": morph, "reference_asset": asset, "head_scale": scale,
                 "reference_ratio_targets": targets, "design_C": C["design_audit"],
                 "placement_C": C["placement"], "B_full_design": Bfull["design_audit"],
                 "editing_mode": True, "geometry_source": nd.GEOMETRY_SOURCE_DUAL,
                 "subject_observation_matched": False}
        placement = {"schema": "head_rgbd.neck_head_proportion.placement.v1",
                     "chosen": C["placement"], "B_full": Bfull["placement"],
                     "target_neck": {"W_neck_mm": round(C["target_scale"].W_neck_mm, 2),
                                     "L_neck_mm": round(C["target_scale"].L_neck_mm, 2)},
                     "C_state_support": {s: states_v[s]["audit"]["support"] for s in states_v},
                     "B_state_support": {s: Bfull["audits"][s]["support"] for s in ("closed", "open")},
                     "bottom_cut_note": "the C lower 10% support stays 8.3 mm above the true open "
                                        "mesh bottom cut; the full head-scale B goes below it and into "
                                        "the 8 mm protection gate, so it is not safe despite passing "
                                        "the face/back guards",
                     "method": "head-width scale for cross-section and head-height scale for longitudinal "
                               "targets; C keeps the reference main height at the largest safe lateral "
                               "and vertical support, with the whole design scaled by (vertical .68, "
                               "lateral 1.0, height 1.2)"}
        deformation = {"schema": "head_rgbd.neck_head_proportion.deformation.v1",
                       "C_states": {s: {"base_relative": states_v[s]["audit"]["base_relative"],
                                        "refined_relative": states_v[s]["audit"]["refined_relative"],
                                        "features": states_v[s]["audit"]["features"],
                                        "support": states_v[s]["audit"]["support"]} for s in states_v},
                       "B_full_states": {s: {"features": Bfull["audits"][s]["features"],
                                             "support": Bfull["audits"][s]["support"],
                                             "guards": Bfull["states"][s]["guards"],
                                             "intersections": Bfull["states"][s]["intersections"]["count"]}
                                         for s in ("closed", "open")},
                       "note": "final surface, base surface and design increment are reported separately in "
                               "final_surface_metrics.json; no negative residual is transported"}
        geometry = {"schema": "head_rgbd.neck_head_proportion.verification.v1", "checks": checks,
                    "states": states_v, "disk": disk,
                    "baseline_topology": {k: base_rep[k] for k in ("boundary_edges", "components",
                                                                   "inconsistent_shared_edges")},
                    "accepted_geometry": bool(accepted_geometry),
                    "proportion_pass": bool(proportion_pass),
                    "intersection_scope": C["states"]["closed"]["intersections"]["method"] +
                                          "; local affected-edge scope",
                    }
        manifest = {"schema": "head_rgbd.delivery.neck_head_proportion.v1",
                    "software": {"name": "facecap_airway_delivery", "version": "0.8.0-candidate"},
                    "created_at": datetime.now().astimezone().isoformat(),
                    "meshes": {"closed": "head_closed_m.obj", "open": "head_open_m.obj"},
                    "mesh_unit": "m", "coordinate_system": "reference camera X right / Y up / Z backward",
                    "editing_mode": True, "geometry_source": nd.GEOMETRY_SOURCE_DUAL,
                    "subject_observation_matched": False,
                    "reference": {"path": str(args.reference.resolve()), "sha256": morph["source_sha256"],
                                  "project_asset": "assets/neck_reference/identity_rest_20261109.obj"},
                    "public_default": "0.7.0 (unchanged; 0.8 head-proportion conditions not fully met)",
                    "proportion_pass": bool(proportion_pass),
                    "accepted_geometry": bool(accepted_geometry),
                    "provenance": {"baseline_fit": str(args.fit_dir), "A_source": str(A070)}}
        for name, payload in (("shape_parameters.json", shape), ("placement_audit.json", placement),
                              ("deformation_audit.json", deformation), ("geometry_verification.json", geometry)):
            (out / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        (out / "final" / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        # renders
        t = time.perf_counter()
        base_meshes = base_mesh
        label = {"base": "unshaped baseline", "A": "0.7.0 current",
                 "C": "C closest safe (head H, safe W/L)"}
        ref_scale = scale["S_head_geometric"]
        ref_scaled = (refV * ref_scale, refF)
        for s in ("closed", "open"):
            frame = C["states"][s]["frame"]; fw, rt = frame.n, frame.u_axis
            views = {"front": -fw, "right": rt, "left": -rt,
                     "right45": (rt - fw) / np.sqrt(2.), "left45": (-rt - fw) / np.sqrt(2.)}
            ctr = np.mean(base_meshes[s][0], axis=0)
            for vname, vdir in views.items():
                for norm, refmesh, tag in ((False, (refV, refF), "physical"),
                                           (True, ref_scaled, "head_normalized")):
                    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4), facecolor="#202020")
                    panels = ((refmesh[0], refmesh[1], refmesh[0].mean(axis=0),
                               np.array([0., 0., -1.]) if vname == "front" else
                               np.array([1., 0., 0.]) if vname == "right" else
                               np.array([-1., 0., 0.]) if vname == "left" else
                               np.array([.7, 0., -.7]) if vname == "right45" else np.array([-.7, 0., -.7]),
                               "reference" + (" (normalized)" if norm else "")),
                              (base_meshes[s][0], base_meshes[s][1], ctr, vdir, label["base"]),
                              (A[s][0], A[s][1], ctr, vdir, label["A"]),
                              (C["states"][s]["run"]["vertices"], C["faces"], ctr, vdir, label["C"]))
                    for ax, (V, F, c, d, title) in zip(axes, panels):
                        ax.set_facecolor("#202020")
                        render_flat(ax, V, F, d, c, 170.0)
                        ax.set_title(title, color="w", fontsize=8)
                    fig.suptitle(f"{s} / {vname} full head ({tag}, same scale, flat)", color="w")
                    fig.tight_layout()
                    fig.savefig(out / "render" / f"compare_{s}_{vname}_full_{tag}.png", dpi=90,
                                facecolor="#202020"); plt.close(fig)
            for vname in ("front", "right45"):
                for shade in ("flat", "smooth"):
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), facecolor="#202020")
                    panels = ((refV, refF, np.array([0., -.104, .0619]),
                               np.array([0., 0., -1.]) if vname == "front" else np.array([.7, 0., -.7]), "reference"),
                              (base_meshes[s][0], base_meshes[s][1], frame.origin, views[vname], label["base"]),
                              (A[s][0], A[s][1], frame.origin, views[vname], label["A"]),
                              (C["states"][s]["run"]["vertices"], C["faces"], frame.origin, views[vname], label["C"]))
                    for ax, (V, F, c, d, title) in zip(axes, panels):
                        ax.set_facecolor("#202020")
                        (render_smooth if shade == "smooth" else render_flat)(ax, V, F, d, c, 42.0)
                        ax.set_title(title, color="w", fontsize=8)
                    fig.suptitle(f"{s} / neck close {vname} ({shade})", color="w")
                    fig.tight_layout()
                    fig.savefig(out / "render" / f"compare_{s}_neck_{vname}_{shade}.png", dpi=110,
                                facecolor="#202020"); plt.close(fig)
            # A/B/C causal
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#202020")
            for ax, (key, title) in zip(axes, (("base", label["base"]),
                                               ("B", "B head-scale full (unsafe support)"),
                                               ("C", label["C"]))):
                V, F = (base_meshes[s] if key == "base" else
                        (Bfull["states"][s]["run"]["vertices"], Bfull["faces"]) if key == "B" else
                        (C["states"][s]["run"]["vertices"], C["faces"]))
                ax.set_facecolor("#202020"); render_flat(ax, V, F, views["right45"], frame.origin, 42.0)
                ax.set_title(title, color="w", fontsize=8)
            fig.suptitle(f"{s} / baseline - B - C (flat)", color="w")
            fig.tight_layout(); fig.savefig(out / "render" / f"causal_{s}_45_flat.png", dpi=110,
                                            facecolor="#202020"); plt.close(fig)
            # sections: long + A/B cross
            for sname, normal, sample, lim, xlabel in (("long", frame.u_axis, frame.v_axis, 24., "v (mm)"),
                                                       ("crossA", frame.v_axis, frame.u_axis, 24., "u (mm)"),
                                                       ("crossB", frame.v_axis, frame.u_axis, 24., "u (mm)")):
                fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
                off = 0.0 if sname != "crossB" else float(C["design"].v_b_mm)
                centre = frame.origin + frame.v_axis * (off * 1e-3)
                curves = {}
                for lab, (V, F), color in (("baseline", base_meshes[s], "#8a97a8"),
                                           ("0.7.0", A[s], "#5a8fd0"),
                                           ("C", (C["states"][s]["run"]["vertices"], C["faces"]), "#e05a4e")):
                    xv, nv = outer_section(V, F, centre, normal, sample, frame.n, lim)
                    if len(xv):
                        axes[0].plot(nv, xv, color=color, lw=1.7, label=lab); curves[lab] = (xv, nv)
                axes[0].set_xlabel("outward n (mm)"); axes[0].set_ylabel(xlabel)
                axes[0].set_title(f"{s}: {'midline long' if sname=='long' else ('A cross' if sname=='crossA' else 'B cross')}")
                axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)
                if "baseline" in curves and "C" in curves:
                    xb, nb = curves["baseline"]; xc, nc_ = curves["C"]
                    cm = np.intersect1d(xb, xc)
                    axes[1].plot(cm, nc_[np.isin(xc, cm)] - nb[np.isin(xb, cm)], color="#e05a4e", lw=1.7, label="C-baseline")
                    if "0.7.0" in curves:
                        xa, na = curves["0.7.0"]; cm2 = np.intersect1d(xb, xa)
                        axes[1].plot(cm2, na[np.isin(xa, cm2)] - nb[np.isin(xb, cm2)], color="#5a8fd0", lw=1.3, label="0.7.0-baseline")
                    axes[1].axhline(0, color="k", lw=.6); axes[1].axhline(-.05, color="#999", lw=.6, ls=":")
                    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("normal difference (mm)")
                    axes[1].set_title("non-negative check"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=7)
                fig.tight_layout(); fig.savefig(out / "render" / f"section_{s}_{sname}.png", dpi=110); plt.close(fig)
            # heat/wire
            fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="#202020"); ax.set_facecolor("#202020")
            render_wire(ax, C["states"][s]["run"]["vertices"], C["faces"], views["right45"], frame.origin, 36.)
            ax.set_title(f"{s}: C transition wireframe", color="w", fontsize=9)
            fig.tight_layout(); fig.savefig(out / "render" / f"wire_{s}.png", dpi=110, facecolor="#202020"); plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
            app = C["states"][s]["run"]["applied_mm"]
            mappable = render_heat(axes[0], C["states"][s]["run"]["vertices"], C["faces"], app,
                                   views["front"], frame.origin, 42.)
            axes[0].set_title(f"{s}: C applied field (mm)", fontsize=9)
            if mappable is not None:
                fig.colorbar(mappable, ax=axes[0], fraction=.046, pad=.02, label="mm")
            uv = frame.uv(C["states"][s]["run"]["vertices"])
            keep = (np.linalg.norm(uv, axis=1) < 45.) & C["states"][s]["run"]["front_mask"]
            tri = mtri.Triangulation(uv[:, 0], uv[:, 1], triangles=C["faces"],
                                     mask=~(keep[C["faces"]].all(axis=1)))
            axes[1].tricontourf(tri, app, levels=20, cmap="viridis")
            axes[1].tricontour(tri, app, levels=[.1 * app.max(), .5 * app.max()], colors=["w", "k"], linewidths=.9)
            axes[1].set_aspect("equal"); axes[1].set_xlim(-32, 32); axes[1].set_ylim(-30, 20)
            axes[1].set_xlabel("u (mm)"); axes[1].set_ylabel("v (mm)")
            axes[1].set_title("C field contours", fontsize=9)
            fig.tight_layout(); fig.savefig(out / "render" / f"heat_{s}.png", dpi=110, facecolor="white"); plt.close(fig)
        # head-scale visual figure
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        axes[0].bar(["H_head", "W_head", "D_head"], [ref_head["H_head_mm"], ref_head["W_head_mm"], ref_head["D_head_mm"]],
                    alpha=.6, label="reference")
        axes[0].bar(["H_head", "W_head", "D_head"], [tgt_head["H_head_mm"], tgt_head["W_head_mm"], tgt_head["D_head_mm"]],
                    alpha=.4, label="target")
        axes[0].set_ylabel("mm"); axes[0].set_title("head dimensions (same region definition)")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
        axes[1].bar(["H", "W", "D"], [scale["ratios"]["H_head_mm"], scale["ratios"]["W_head_mm"], scale["ratios"]["D_head_mm"]],
                    color=["#e05a4e", "#5a8fd0", "#8fbf6a"])
        axes[1].axhline(scale["S_head_geometric"], color="k", ls="--", lw=.8, label="geometric S")
        axes[1].set_title("target/reference head ratios"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "render" / "head_scale.png", dpi=120); plt.close(fig)
        # feature profile figure
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        for lab, feat, color in (("base", base_feat, "#8a97a8"), ("0.7.0", a_feat, "#5a8fd0"),
                                 ("C", c_feat, "#e05a4e")):
            if not feat:
                continue
            xs = [feat["A"]["v_mm"]] + ([feat["saddle"]["v_mm"], feat["B"]["v_mm"]] if "B" in feat else [])
            ys = [feat["A"]["height_mm"]] + ([feat["saddle"]["height_mm"], feat["B"]["height_mm"]] if "B" in feat else [])
            ax.plot(xs, ys, "o-", color=color, lw=1.5, ms=4, label=lab)
        ax.axhline(0, color="k", lw=.6); ax.set_xlabel("target head-local v (mm)")
        ax.set_ylabel("final surface height above head-scale background (mm)")
        ax.set_title(f"final surface A/saddle/B, Gaussian sigma={sigma:.1f} mm")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "render" / "final_surface_features.png", dpi=120); plt.close(fig)
        timing.append({"stage": "renders", "seconds": round(time.perf_counter() - t, 3)})
        summary = {"checks": checks, "accepted_geometry": bool(accepted_geometry),
                   "proportion_pass": bool(proportion_pass), "proportion": proportion,
                   "head_scale": scale, "targets": targets, "morph": morph,
                   "A_feat": a_feat, "A060_feat": a060_feat, "base_feat": base_feat, "C_feat": c_feat,
                   "C_design": C["design_audit"], "B_full_design": Bfull["design_audit"],
                   "C_support": {s: states_v[s]["audit"]["support"] for s in states_v},
                   "B_support": {s: Bfull["audits"][s]["support"] for s in ("closed", "open")},
                   "B_guard": {s: Bfull["states"][s]["guards"]["status"] for s in ("closed", "open")},
                   "B_inter": {s: Bfull["states"][s]["intersections"]["count"] for s in ("closed", "open")},
                   "B_feat": {s: Bfull["audits"][s]["features"] for s in ("closed", "open")},
                   "facial": facial, "disk": disk, "final_metrics": final_metrics,
                   "calibration": calib, "C_height_factor": height_factor,
                   "timing": timing, "total_seconds": round(time.perf_counter() - t0, 3),
                   "version": "0.8.0-candidate"}
        (out / "report_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False,
                                                            default=float), encoding="utf-8")
        from tools.make_neck_head_proportion_report import write_report
        write_report(out, summary=summary)
        (PROJECT / "docs" / "NECK_HEAD_PROPORTION_REPORT.html").write_bytes(
            (out / "NECK_HEAD_PROPORTION_REPORT.html").read_bytes())
        status = "ok" if accepted_geometry else "verification_failed"
    except Exception as exc:  # noqa: BLE001
        import traceback
        error = f"{type(exc).__name__}: {exc}"
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        (out / "run_status.json").write_text(json.dumps({
            "status": status, "error": error, "exit_code": 0 if status == "ok" else 1,
            "total_seconds": round(time.perf_counter() - t0, 3), "output": str(out),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": status, "error": error,
                          "exit_code": 0 if status == "ok" else 1,
                          "total_seconds": round(time.perf_counter() - t0, 3),
                          "output": str(out)}, indent=2, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
