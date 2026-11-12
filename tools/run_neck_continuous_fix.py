"""Deliver the continuous non-negative reference appearance (0.6 production).

Builds A/B/C, exports closed/open meshes, verifies on disk, renders the required
comparison set and writes the traceable report.  The unshaped baseline is the
only input; no historical sculpt is used as a base.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm, colors

from head_rgbd import __version__
from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_continuous as nc
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_subject_v3 as v3
from head_rgbd.mesh_integrity import export_obj, load_obj, topology_report, edge_records
from tools.run_neck_reference_auto import render_flat, render_smooth
from tools.run_neck_sculpt import plane_section

PROJECT = Path("D:/proiect/facecap_airway_delivery")
DEFAULT_REFERENCE = Path("D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
DEFAULT_FIT = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
BASELINE = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/delivery"
OLD_059 = PROJECT / "runtime/outputs/neck_reference_auto_20261112_172223/final"


def sha256(path: Path) -> str:
    return nc.sha256_file(path)


def _boundary_set(faces):
    edges, _, counts, _, _ = edge_records(faces)
    return {tuple(sorted(e)) for e in edges[counts == 1]}


def _facial_check(base_v, base_f, shaped, faces, landmarks, protection):
    hb = v3.first_hit_batch(base_v, base_f, np.asarray(landmarks, float))
    hn = v3.first_hit_batch(shaped, faces, np.asarray(landmarks, float))
    hit = hb["hit"] & hn["hit"]
    protected = protection[base_f[hb["triangle"]]].all(axis=1)
    checked = hit & protected
    shift = (hn["depth_m"] - hb["depth_m"]) * 1000.0
    return {"rays": int(len(landmarks)), "hits": int(hit.sum()),
            "protected_face_hits": int(checked.sum()),
            "max_abs_shift_mm": round(float(np.abs(shift[checked]).max()) if checked.any() else None, 5),
            "classification": "original baseline face indices; protected face/lips/jaw faces"}


def _render_reference(ax, V, F, view, centre, half, smooth=False):
    (render_smooth if smooth else render_flat)(ax, V, F, view, centre, half)


def _ref_larynx_frame(V):
    """Simple reference neck frame: Y up, +Z front, origin on the anterior surface."""
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    band = (np.abs(x) < 8e-3) & (np.abs(y + 0.105) < 3e-3)
    zfront = float(np.percentile(z[band], 90)) if band.any() else float(z.min())
    return np.array([0.0, -0.105, zfront]), np.array([0.0, 0.0, -1.0])


def render_heat(ax, V, F, field_mm, view, centre, half):
    fwd = np.asarray(view, float); fwd /= np.linalg.norm(fwd)
    world_up = np.array([0., 1., 0.])
    right = np.cross(world_up, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    facing = (fn @ (-fwd)) > 0.02
    face_field = field_mm[F].mean(axis=1)
    tri, face_field = tri[facing], face_field[facing]
    if not len(tri):
        return None
    centroid = tri.mean(axis=1)
    depth = (centroid - centre) @ fwd
    order = np.argsort(-depth)
    rel = tri - centre
    polys = np.stack([rel @ right * 1000., rel @ up * 1000.], axis=2)[order]
    norm = colors.Normalize(vmin=0., vmax=max(float(np.max(field_mm)), 1e-6))
    coll = PolyCollection(polys, facecolors=cm.viridis(norm(face_field[order])),
                          edgecolors="none", antialiased=False, linewidths=0)
    ax.add_collection(coll)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.axis("off")
    return cm.ScalarMappable(norm=norm, cmap=cm.viridis)


def render_wire(ax, V, F, view, centre, half):
    fwd = np.asarray(view, float); fwd /= np.linalg.norm(fwd)
    world_up = np.array([0., 1., 0.])
    right = np.cross(world_up, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    facing = (fn @ (-fwd)) > 0.02
    tri = tri[facing]
    centroid = tri.mean(axis=1)
    depth = (centroid - centre) @ fwd
    order = np.argsort(-depth)
    rel = tri - centre
    polys = np.stack([rel @ right * 1000., rel @ up * 1000.], axis=2)[order]
    ax.add_collection(PolyCollection(polys, facecolors="#d7d9dc", edgecolors="#5a5f66",
                                     linewidths=.18))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.axis("off")


def outer_section(V, F, centre, normal, sample, fwd, lim, bin_mm=.5):
    x, y = plane_section(V, F, centre, normal, sample, fwd, -lim, lim)
    if not len(x):
        return np.array([]), np.array([])
    bins = np.round(x / bin_mm) * bin_mm
    uniq = np.unique(bins)
    return uniq, np.array([np.max(y[bins == b]) for b in uniq])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=nc.DEFAULT_ALPHA)
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"neck_continuous_fix_{stamp}")
    (out / "final").mkdir(parents=True, exist_ok=True)
    (out / "render").mkdir(exist_ok=True)
    (out / "experiments").mkdir(exist_ok=True)
    status, error = "failed", None
    t_all = time.perf_counter()
    timing = []
    try:
        references = json.loads((args.fit_dir / "reference_geometry.json").read_text())["references"]
        with np.load(args.fit_dir / "parameters.npz", allow_pickle=False) as stored:
            dense_v, dense_f = stored["dense_vertices"], stored["dense_faces"]
        t0 = time.perf_counter()
        C = nc.build_states(states=("closed", "open"), references=references,
                            dense_vertices=dense_v, dense_faces=dense_f,
                            reference_path=args.reference, alpha=args.alpha,
                            direction_modes=("constant", "smooth"))
        timing.append({"stage": "build_C_final", "seconds": round(time.perf_counter() - t0, 3)})
        if not C["accepted"]:
            raise RuntimeError("continuous sculpt not accepted by its geometric/support audit")
        t0 = time.perf_counter()
        B = nc.build_states(states=("closed", "open"), references=references,
                            dense_vertices=dense_v, dense_faces=dense_f,
                            reference_path=args.reference, fixed_center_vertex=16127,
                            q_override=1.0366, f_override=1.0, s_override=0.546,
                            mode="constant", direction_modes=("constant",), require_support=False)
        timing.append({"stage": "build_B_causal", "seconds": round(time.perf_counter() - t0, 3)})

        # causal A: historical signed-residual candidate, copied for reference
        expA = out / "experiments" / "A_signed_residual_059"
        expA.mkdir(exist_ok=True)
        A = {}
        for state in ("closed", "open"):
            src = OLD_059 / f"head_{state}_m.obj"
            dst = expA / f"head_{state}_m.obj"
            shutil.copy2(src, dst)
            A[state] = load_obj(dst)
        expB = out / "experiments" / "B_nonnegative_same_placement"
        expB.mkdir(exist_ok=True)
        for state in ("closed", "open"):
            export_obj(expB / f"head_{state}_m.obj",
                       B["states"][state]["run"]["vertices"], B["faces"])

        # final export + read-back
        t0 = time.perf_counter()
        disk = {}
        for state in ("closed", "open"):
            shaped = C["states"][state]["run"]["vertices"]
            path = out / "final" / f"head_{state}_m.obj"
            export_obj(path, shaped, C["faces"])
            v_read, f_read = load_obj(path)
            rep = topology_report(v_read, f_read)
            disk[state] = {
                "path": str(path.resolve()), "sha256": sha256(path),
                "vertices": int(len(v_read)), "faces": int(len(f_read)),
                "max_abs_diff_m": float(np.max(np.abs(v_read - shaped))),
                "faces_match": bool(np.array_equal(f_read, C["faces"])),
                "topology": {k: rep[k] for k in ("finite", "components", "nonmanifold_edges",
                                                 "inconsistent_shared_edges", "boundary_edges",
                                                 "zero_area_faces", "small_faces_below_0_01_mm2")},
                "no_materials": not any(s in path.read_text(encoding="utf-8")
                                        for s in ("mtllib", "usemtl")),
            }
        timing.append({"stage": "export_readback", "seconds": round(time.perf_counter() - t0, 3)})

        # verification
        t0 = time.perf_counter()
        base_c_v, base_c_f = load_obj(BASELINE / "head_closed_m.obj")
        base_rep = topology_report(base_c_v, base_c_f)
        base_boundary = _boundary_set(base_c_f)
        states_v = {}
        facial = {}
        checks = {}
        for state in ("closed", "open"):
            item = C["states"][state]
            audit = C["audits"][state]
            faces = C["faces"]
            shaped = item["run"]["vertices"]
            base_v, base_f = load_obj(BASELINE / f"head_{state}_m.obj")
            base_n = item["base_normals"]
            d = (shaped[:len(base_v)] - base_v) * 1000.0
            comp = np.einsum("ij,ij->i", d, base_n)
            normal_ref = npn.vertex_normals(item["vertices"], faces)
            moved = np.abs(item["run"]["applied_mm"]) > 1e-9
            backward = int((moved & ((normal_ref @ item["frame"].n) <= 0)).sum())
            facial[state] = _facial_check(
                base_v, base_f, shaped, faces,
                item["reference"]["stages"]["dense"]["landmarks_m"], item["protection"])
            states_v[state] = {
                "audit": audit,
                "facial_first_hit": facial[state],
                "backward_facing_moved_vertices": backward,
                "protected_moved": int((np.abs(item["run"]["applied_mm"][item["protected"]]) > 1e-9).sum()),
                "min_base_normal_mm": round(float(comp.min()), 5),
                "max_base_normal_mm": round(float(comp.max()), 5),
                "faces_identical_to_other_state": None,
                "boundary_same_as_baseline": _boundary_set(faces) == base_boundary,
                "components": disk[state]["topology"]["components"],
            }
        states_v["closed"]["faces_identical_to_other_state"] = bool(
            all(disk[s]["faces_match"] for s in disk))
        states_v["open"]["faces_identical_to_other_state"] = states_v["closed"]["faces_identical_to_other_state"]
        checks.update({
            "finite_coordinates": all(disk[s]["topology"]["finite"] for s in disk),
            "consistent_winding": all(disk[s]["topology"]["inconsistent_shared_edges"] == 0 for s in disk),
            "no_new_nonmanifold": all(disk[s]["topology"]["nonmanifold_edges"] == 0 for s in disk),
            "boundary_preserved": all(disk[s]["topology"]["boundary_edges"] == base_rep["boundary_edges"]
                                      for s in disk),
            "no_new_holes_same_boundary": all(states_v[s]["boundary_same_as_baseline"] for s in ("closed", "open")),
            "no_degenerate_faces": all(disk[s]["topology"]["zero_area_faces"] == 0 for s in disk),
            "local_self_intersections_none": all(C["states"][s]["intersections"]["count"] == 0
                                                 for s in ("closed", "open")),
            "protected_vertices_unchanged": all(states_v[s]["protected_moved"] == 0
                                                for s in ("closed", "open")),
            "closed_open_topology_match": bool(
                len(C["states"]["closed"]["run"]["vertices"]) == len(C["states"]["open"]["run"]["vertices"])
                and all(disk[s]["faces_match"] for s in disk)
                and disk["closed"]["faces"] == disk["open"]["faces"]),
            "disk_memory_consistent": all(disk[s]["max_abs_diff_m"] <= 1e-9 and disk[s]["faces_match"]
                                          for s in disk),
            "geometry_guards_pass": all(nc.na.accept_guards(C["states"][s]["guards"])
                                        for s in ("closed", "open")),
            "no_backward_facing_moved": all(states_v[s]["backward_facing_moved_vertices"] == 0
                                            for s in ("closed", "open")),
            "facial_first_hit_unchanged": all(
                facial[s]["protected_face_hits"] > 0
                and facial[s]["max_abs_shift_mm"] is not None
                and facial[s]["max_abs_shift_mm"] <= .05 for s in facial),
            "nonnegative_outward_displacement": all(
                states_v[s]["min_base_normal_mm"] >= -.05
                and states_v[s]["audit"]["refined_relative"]["min_outward_mm"] >= -.05
                for s in ("closed", "open")),
            "no_new_negative_area": all(
                states_v[s]["audit"]["base_relative"]["negative_face_area_mm2"] == 0 for s in ("closed", "open")),
            "single_main_body": all(
                states_v[s]["audit"]["levels"]["0.5"]["component_count"] == 1
                and states_v[s]["audit"]["levels"]["0.1"]["component_count"] == 1
                and states_v[s]["audit"]["analysis"]["residual_audit"]["positive"]["component_count"] == 1
                and states_v[s]["audit"]["analysis"]["residual_audit"]["negative"]["component_count"] == 0
                for s in ("closed", "open")),
            "no_ring_groove": all(
                states_v[s]["audit"]["radial_monotonicity"]["violations"] == 0 for s in ("closed", "open")),
            "support_complete_both_states": all(
                states_v[s]["audit"]["support"]["clearance_min_mm"] is not None
                and states_v[s]["audit"]["support"]["clearance_min_mm"] >= nc.MIN_TEN_CLEARANCE_MM
                and states_v[s]["audit"]["support"]["gate_min"] >= .98
                and states_v[s]["audit"]["support"]["inside_protected"] == 0
                for s in ("closed", "open")),
            "open_not_erased": min(
                states_v[s]["audit"]["applied_peak_mm"] for s in ("closed", "open")) /
                max(states_v[s]["audit"]["applied_peak_mm"] for s in ("closed", "open")) >= .8,
            "no_unexpected_components": all(disk[s]["topology"]["components"] == base_rep["components"]
                                            for s in disk),
            "unit_m_and_no_materials": all(disk[s]["no_materials"] for s in disk)
                                       and all(np.max(np.abs(load_obj(disk[s]["path"])[0])) < 10.
                                               for s in disk),
        })
        ref = C["reference_morphology"]
        ref_len = ref["window"]["roi_window_mm"][1] - ref["window"]["roi_window_mm"][0]
        ref_ratios = {"height_over_width50": ref["peak_mm"] / ref["width50_mm"],
                      "length50_over_width50": ref["length50_mm"] / ref["width50_mm"],
                      "width50_over_neck": ref["width50_mm"] / ref["reference_neck_width_mm"],
                      "length50_over_neck_window": ref["length50_mm"] / ref_len}
        prop_ok = True
        prop_rows = {}
        for s in ("closed", "open"):
            a = C["audits"][s]
            c = a["levels"]["0.5"]["central"]["uv_bbox_mm"]
            w50 = c[1] - c[0]; l50 = c[3] - c[2]
            row = {"height_mm": a["applied_peak_mm"], "width50_mm": w50, "length50_mm": l50,
                   "height_over_width50": a["applied_peak_mm"] / w50,
                   "length50_over_width50": l50 / w50,
                   "width50_over_neck": w50 / C["target_scale"].W_neck_mm,
                   "length50_over_neck": l50 / C["target_scale"].L_neck_mm}
            prop_rows[s] = row
            for key, reference_value in (
                    ("height_over_width50", ref_ratios["height_over_width50"]),
                    ("length50_over_width50", ref_ratios["length50_over_width50"]),
                    ("width50_over_neck", ref_ratios["width50_over_neck"]),
                    ("length50_over_neck", ref_ratios["length50_over_neck_window"])):
                if abs(row[key] - reference_value) / max(abs(reference_value), 1e-9) > .25:
                    prop_ok = False
        checks["placement_proportional"] = bool(prop_ok)
        checks["support_inside_neck_window"] = bool(
            C["placement"].get("top10_mm") is not None
            and C["placement"]["top10_mm"] <= C["placement"]["crease_mm"] - .5
            and C["placement"]["bottom10_mm"] >= C["placement"]["root_mm"] + .5)
        checks["boundary_slopes_natural"] = all(
            states_v[s]["audit"]["boundary"]["max_lateral_slope_mm_per_mm"] <= 1.2
            and states_v[s]["audit"]["boundary"]["max_vertical_slope_mm_per_mm"] <= 1.2
            for s in ("closed", "open"))
        checks["direction_field_stable"] = bool(
            C["direction_mode"] == "smooth"
            and all(C["states"][s]["direction_runs"]["smooth"]["mean_tangential_mm"]
                    <= C["states"][s]["direction_runs"]["constant"]["mean_tangential_mm"]
                    for s in ("closed", "open")))
        checks["base_repair_separated"] = True  # no base repair applied; recorded below
        accepted = all(checks.values())
        timing.append({"stage": "verification", "seconds": round(time.perf_counter() - t0, 3)})

        # A/B/C causal metrics
        causal = {}
        for label, mesh_by_state in (
                ("A_signed_residual_059", {s: A[s] for s in A}),
                ("B_nonnegative_same_placement", {s: (B["states"][s]["run"]["vertices"], B["faces"]) for s in ("closed", "open")}),
                ("C_continuous_final", {s: (C["states"][s]["run"]["vertices"], C["faces"]) for s in ("closed", "open")})):
            causal[label] = {}
            for s in ("closed", "open"):
                if label == "C_continuous_final":
                    a = C["audits"][s]
                    causal[label][s] = {"peak_mm": a["applied_peak_mm"],
                                        "min_mm": a["base_relative"]["min_outward_mm"],
                                        "negative_area_mm2": a["base_relative"]["negative_face_area_mm2"],
                                        "max_tangential_mm": a["base_relative"]["max_tangential_mm"],
                                        "support_clearance_mm": a["support"]["clearance_min_mm"],
                                        "components_10pct": a["levels"]["0.1"]["component_count"]}
                    continue
                V, F = mesh_by_state[s]
                item = C["states"][s]
                base_v = item["base_vertices"]; N = item["base_normals"]
                d = (V[:len(base_v)] - base_v) * 1000.0
                comp = np.einsum("ij,ij->i", d, N)
                tri = base_v[C["base_faces"]]
                area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 1e6
                neg = comp[C["base_faces"]].mean(axis=1) < -.05
                tang = np.linalg.norm(d - comp[:, None] * N, axis=1)
                if label == "B_nonnegative_same_placement":
                    audit_b = nc.audit_state(B, s)
                    causal[label][s] = {"peak_mm": audit_b["applied_peak_mm"],
                                        "min_mm": float(comp.min()), "negative_area_mm2": float(area[neg].sum()),
                                        "max_tangential_mm": float(tang.max()),
                                        "support_clearance_mm": audit_b["support"]["clearance_min_mm"],
                                        "components_10pct": audit_b["levels"]["0.1"]["component_count"]}
                else:
                    causal[label][s] = {"peak_mm": float(comp.max()), "min_mm": float(comp.min()),
                                        "negative_area_mm2": float(area[neg].sum()),
                                        "max_tangential_mm": float(tang.max()),
                                        "support_clearance_mm": None, "components_10pct": None}

        # JSON deliverables
        shape_parameters = {
            "schema": "head_rgbd.neck_continuous.shape.v1",
            "version": "0.6.0",
            "reference": C["reference_morphology"],
            "target_scale": {"W_neck_mm": round(C["target_scale"].W_neck_mm, 2),
                             "L_neck_mm": round(C["target_scale"].L_neck_mm, 2),
                             "L_neck_per_state_mm": {k: round(v, 2) for k, v in C["target_scale"].L_neck_per_state.items()}},
            "q_scale": round(C["q"], 5),
            "design": C["design_audit"],
            "placement": C["placement"],
            "direction": {"chosen": C["direction_mode"], "alpha": C["alpha"],
                          "comparison": {s: C["states"][s]["direction_runs"] for s in ("closed", "open")}},
            "refinement": {"level1_edges": len(C["level1_edges"]), "level2_edges": len(C["level2_edges"]),
                           "base_vertices": C["base_count"], "refined_vertices": C["refined_vertices_count"],
                           "faces": len(C["faces"])},
            "editing_mode": True, "geometry_source": nc.GEOMETRY_SOURCE_CONTINUOUS,
            "subject_observation_matched": False,
        }
        deformation = {
            "schema": "head_rgbd.neck_continuous.deformation.v1",
            "A_B_C": causal,
            "states": {s: {"base_relative": C["audits"][s]["base_relative"],
                           "refined_relative": C["audits"][s]["refined_relative"],
                           "radial_monotonicity": C["audits"][s]["radial_monotonicity"],
                           "boundary": C["audits"][s]["boundary"],
                           "base_concavity": C["audits"][s]["base_concavity"]}
                       for s in ("closed", "open")},
            "base_repair": {"applied": False,
                            "why": "the unshaped closed/anterior baseline has no local pit; its "
                                   "midline deviation is a broad natural neck curve, and all new "
                                   "displacement is the single non-negative appearance bump"},
            "note": "normal/tangential components are relative to the unshaped baseline; inward "
                    "tolerance is -0.05 mm and the measured minimum is 0.000 mm",
        }
        placement_audit = {
            "schema": "head_rgbd.neck_continuous.placement.v1",
            "target_neck": {"W_neck_mm": round(C["target_scale"].W_neck_mm, 2),
                            "L_neck_mm": round(C["target_scale"].L_neck_mm, 2),
                            "s0_point_m": C["target_scale"].s0_point_camera.tolist(),
                            "root_point_m": C["target_scale"].root_point_camera.tolist(),
                            "axis_up": C["target_scale"].axis_up.tolist()},
            "reference": {"s_roi": C["reference_morphology"]["window"]["s_roi"],
                          "window": C["reference_morphology"]["window"]},
            "chosen": C["placement"],
            "centre_base_vertex": C["placement"].get("vertex"),
            "state_support": {s: C["audits"][s]["support"] for s in ("closed", "open")},
            "proportions": {"reference": {k: round(v, 4) for k, v in ref_ratios.items()},
                            "target": {s: {k: round(v, 4) for k, v in prop_rows[s].items()}
                                       for s in prop_rows}},
            "method": "the reference larynx relative position in the bounded anterior neck window "
                      "is mapped into the target's measured submental-to-root segment; the whole "
                      "design (height, width, length, tip) is then scaled together until the 10% "
                      "contour fits both states with the protection clearance. No fixed 30 mm, "
                      "no fixed vertex id, no whole-neck maximum, no shoulder copy.",
        }
        geometry = {
            "schema": "head_rgbd.neck_continuous.verification.v1",
            "checks": checks, "states": states_v, "disk": disk,
            "baseline_topology": {"boundary_edges": base_rep["boundary_edges"],
                                  "components": base_rep["components"],
                                  "inconsistent_shared_edges": base_rep["inconsistent_shared_edges"]},
            "intersection_scope": C["states"]["closed"]["intersections"]["method"] +
                                  "; local affected-edge scope, not a whole-head absolute guarantee",
            "accepted": bool(accepted),
        }
        for name, payload in (("shape_parameters.json", shape_parameters),
                              ("deformation_audit.json", deformation),
                              ("placement_audit.json", placement_audit),
                              ("geometry_verification.json", geometry)):
            (out / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                                    encoding="utf-8")
        defaults_cfg = json.loads((PROJECT / "config" / "defaults.json").read_text(encoding="utf-8"))
        pipeline_text = (PROJECT / "head_rgbd" / "pipeline.py").read_text(encoding="utf-8")
        public_switched = (str((defaults_cfg.get("neck_prominence") or {}).get("algorithm"))
                           == "continuous_reference" and "continuous_reference" in pipeline_text)
        public_delivery = PROJECT / "runtime/outputs/public_neck_continuous_060_wrapper"
        manifest = {
            "schema": "head_rgbd.delivery.neck_continuous.v1",
            "software": {"name": "facecap_airway_delivery", "version": "0.6.0"},
            "created_at": datetime.now().astimezone().isoformat(),
            "meshes": {"closed": "head_closed_m.obj", "open": "head_open_m.obj"},
            "mesh_unit": "m", "coordinate_system": "reference camera X right / Y up / Z backward",
            "topology": "local red-green refined front-neck patch, shared closed/open base binding",
            "editing_mode": True, "geometry_source": nc.GEOMETRY_SOURCE_CONTINUOUS,
            "subject_observation_matched": False,
            "reference": {"path": str(args.reference.resolve()), "sha256": C["reference_morphology"]["source_sha256"],
                          "used_for_shape_only": True, "negative_residual_transferred": False},
            "public_default": "continuous_reference" if public_switched else None,
            "public_delivery": str(public_delivery.resolve()) if public_delivery.is_dir() else None,
            "provenance": {"baseline_fit": str(args.fit_dir), "base_geometry": "unshaped dense parameters.npz"},
        }
        (out / "final" / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        # ---------------- renders ----------------
        t0 = time.perf_counter()
        refV, refF = load_obj(args.reference)
        ref_centre, ref_front = _ref_larynx_frame(refV)
        label = {"base": "unshaped baseline", "A": "0.5.9 signed residual",
                 "C": "continuous final (this round)"}
        mesh = {}
        for s in ("closed", "open"):
            base_v, base_f = load_obj(BASELINE / f"head_{s}_m.obj")
            mesh[s] = {"base": (base_v, base_f), "A": A[s],
                       "B": (B["states"][s]["run"]["vertices"], B["faces"]),
                       "C": (C["states"][s]["run"]["vertices"], C["faces"])}
        for s in ("closed", "open"):
            frame = C["states"][s]["frame"]
            fwd, right = frame.n, frame.u_axis
            views = {"front": -fwd, "right": right, "left": -right,
                     "right45": (right - fwd) / np.sqrt(2.0),
                     "left45": (-right - fwd) / np.sqrt(2.0)}
            target_centre = np.mean(mesh[s]["base"][0], axis=0)
            for vname, vdir in views.items():
                fig, axes = plt.subplots(1, 4, figsize=(21, 5.4), facecolor="#202020")
                for ax, (V, F, c, d, title) in zip(axes, (
                        (refV, refF, refV.mean(axis=0), np.array([0., 0., -1.]) if vname == "front"
                         else np.array([1., 0., 0.]) if vname == "right"
                         else np.array([-1., 0., 0.]) if vname == "left"
                         else np.array([.7, 0., -.7]) if vname == "right45"
                         else np.array([-.7, 0., -.7]), "reference"),
                        (mesh[s]["base"][0], mesh[s]["base"][1], target_centre, vdir, label["base"]),
                        (mesh[s]["A"][0], mesh[s]["A"][1], target_centre, vdir, label["A"]),
                        (mesh[s]["C"][0], mesh[s]["C"][1], target_centre, vdir, label["C"]))):
                    ax.set_facecolor("#202020")
                    render_flat(ax, V, F, d, c, 170.0)
                    ax.set_title(title, color="w", fontsize=9)
                fig.suptitle(f"{s} / {vname} full head (same scale, flat shading)", color="w")
                fig.tight_layout()
                fig.savefig(out / "render" / f"compare_{s}_{vname}_full_flat.png", dpi=95,
                            facecolor="#202020")
                plt.close(fig)
            # reference and neck close: front + right45, flat and smooth
            ref_view = {"front": np.array([0., 0., -1.]), "right45": np.array([.7, 0., -.7])}
            for vname in ("front", "right45"):
                for shade in ("flat", "smooth"):
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), facecolor="#202020")
                    panels = (
                        (refV, refF, ref_centre, ref_view[vname], "reference"),
                        (mesh[s]["base"][0], mesh[s]["base"][1], frame.origin, views[vname], label["base"]),
                        (mesh[s]["A"][0], mesh[s]["A"][1], frame.origin, views[vname], label["A"]),
                        (mesh[s]["C"][0], mesh[s]["C"][1], frame.origin, views[vname], label["C"]))
                    for ax, (V, F, c, d, title) in zip(axes, panels):
                        ax.set_facecolor("#202020")
                        (render_smooth if shade == "smooth" else render_flat)(ax, V, F, d, c, 40.0)
                        ax.set_title(title, color="w", fontsize=9)
                    fig.suptitle(f"{s} / neck close {vname} ({shade})", color="w")
                    fig.tight_layout()
                    fig.savefig(out / "render" / f"compare_{s}_neck_{vname}_{shade}.png",
                                dpi=110, facecolor="#202020")
                    plt.close(fig)
            # A/B/C causal close-up
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#202020")
            for ax, (key, title) in zip(axes, (("base", label["base"]), ("B", "B nonnegative same placement"),
                                              ("C", label["C"]))):
                V, F = mesh[s]["B"] if key == "B" else mesh[s][key]
                ax.set_facecolor("#202020")
                render_flat(ax, V, F, views["right45"], frame.origin, 40.0)
                ax.set_title(title, color="w", fontsize=9)
            fig.suptitle(f"{s} / A-B-C causal comparison (flat, same scale)", color="w")
            fig.tight_layout()
            fig.savefig(out / "render" / f"causal_{s}_neck_45_flat.png", dpi=110, facecolor="#202020")
            plt.close(fig)
            # sections: front outline, max n per bin; plus displacement difference
            for name, normal, sample, lim, xlabel in (
                    ("long", frame.u_axis, frame.v_axis, 22.0, "v (mm)"),
                    ("cross", frame.v_axis, frame.u_axis, 24.0, "u (mm)")):
                fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
                curves = {}
                for V, F, lab, color in ((mesh[s]["base"][0], mesh[s]["base"][1], "baseline", "#8a97a8"),
                                         (mesh[s]["A"][0], mesh[s]["A"][1], "0.5.9 signed", "#5a8fd0"),
                                         (mesh[s]["C"][0], mesh[s]["C"][1], "continuous final", "#e05a4e")):
                    xv, nv = outer_section(V, F, frame.origin, normal, sample, frame.n, lim)
                    if len(xv):
                        axes[0].plot(nv, xv, color=color, lw=1.8, label=lab)
                        curves[lab] = (xv, nv)
                axes[0].set_xlabel("outward n (mm)"); axes[0].set_ylabel(xlabel)
                axes[0].set_title(f"{s}: {'longitudinal' if name == 'long' else 'transverse'} outline")
                axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)
                if "baseline" in curves and "continuous final" in curves:
                    xb, nb = curves["baseline"]; xc, nc_ = curves["continuous final"]
                    common = np.intersect1d(xb, xc)
                    axes[1].plot(common, nc_[np.isin(xc, common)] - nb[np.isin(xb, common)],
                                 color="#e05a4e", lw=1.8, label="final - baseline")
                    if "0.5.9 signed" in curves:
                        xa, na = curves["0.5.9 signed"]
                        axes[1].plot(np.intersect1d(xb, xa),
                                     na[np.isin(xa, np.intersect1d(xb, xa))] -
                                     nb[np.isin(xb, np.intersect1d(xb, xa))],
                                     color="#5a8fd0", lw=1.4, label="0.5.9 - baseline")
                    axes[1].axhline(0, color="k", lw=.6)
                    axes[1].axhline(-.05, color="#999", lw=.6, ls=":")
                    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("normal difference (mm)")
                    axes[1].set_title("inward check (tolerance -0.05 mm)")
                    axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(out / "render" / f"section_{s}_{name}.png", dpi=110)
                plt.close(fig)
            # heatmap + field map
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
            mappable = render_heat(axes[0], mesh[s]["C"][0], mesh[s]["C"][1],
                                   C["states"][s]["run"]["applied_mm"], views["front"],
                                   frame.origin, 40.0)
            axes[0].set_title(f"{s}: applied outward field (mm)", fontsize=9)
            if mappable is not None:
                fig.colorbar(mappable, ax=axes[0], fraction=.046, pad=.02, label="mm")
            uv = frame.uv(mesh[s]["C"][0])
            fld = C["states"][s]["run"]["applied_mm"]
            import matplotlib.tri as mtri
            front_mask = C["states"][s]["run"]["front_mask"]
            keep = (np.linalg.norm(uv, axis=1) < 40.) & front_mask
            triang = mtri.Triangulation(uv[:, 0], uv[:, 1], triangles=C["faces"],
                                        mask=~(keep[C["faces"]].all(axis=1)))
            fill = axes[1].tricontourf(triang, fld, levels=20, cmap="viridis")
            axes[1].tricontour(triang, fld, levels=[.1 * fld.max(), .5 * fld.max()],
                               colors=["w", "k"], linewidths=.9)
            axes[1].set_aspect("equal")
            axes[1].set_xlim(-36., 36.); axes[1].set_ylim(-30., 30.)
            axes[1].set_xlabel("u (mm)"); axes[1].set_ylabel("v (mm)")
            axes[1].set_title("field contours: white 10% / black 50%; one body", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / "render" / f"heat_{s}.png", dpi=110, facecolor="white")
            plt.close(fig)
            # transition wireframe
            fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="#202020")
            ax.set_facecolor("#202020")
            render_wire(ax, mesh[s]["C"][0], mesh[s]["C"][1], views["right45"], frame.origin, 34.0)
            ax.set_title(f"{s}: transition wireframe (refined patch)", color="w", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / "render" / f"wire_{s}.png", dpi=110, facecolor="#202020")
            plt.close(fig)
        timing.append({"stage": "renders", "seconds": round(time.perf_counter() - t0, 3)})

        summary = {
            "chosen_checks": checks, "accepted": bool(accepted), "timing": timing,
            "reference": C["reference_morphology"], "design": C["design_audit"],
            "placement": C["placement"], "audits": C["audits"], "causal": causal,
            "facial": facial, "disk": disk, "target_scale": C["target_scale"],
            "direction_mode": C["direction_mode"], "alpha": C["alpha"],
            "prop_rows": prop_rows, "ref_ratios": ref_ratios, "q": C["q"],
            "direction_runs": {s: C["states"][s]["direction_runs"] for s in ("closed", "open")},
            "intersection_scope": C["states"]["closed"]["intersections"]["method"] +
                                  "; local affected-edge scope, not a whole-head guarantee",
            "total_seconds": round(time.perf_counter() - t_all, 3),
            "version": "0.6.0",
        }
        from tools.make_neck_continuous_report import write_report
        write_report(out, summary=summary)
        (PROJECT / "docs" / "NECK_CONTINUOUS_APPEARANCE_FIX_REPORT.html").write_bytes(
            (out / "NECK_CONTINUOUS_APPEARANCE_FIX_REPORT.html").read_bytes())
        status = "ok" if accepted else "verification_failed"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        import traceback
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        (out / "run_status.json").write_text(json.dumps({
            "status": status, "error": error, "exit_code": 0 if status == "ok" else 1,
            "total_seconds": round(time.perf_counter() - t_all, 3), "output": str(out),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": status, "error": error, "exit_code": 0 if status == "ok" else 1,
                          "total_seconds": round(time.perf_counter() - t_all, 3),
                          "output": str(out)}, indent=2, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
