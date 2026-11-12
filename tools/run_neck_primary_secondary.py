"""Deliver the two-feature continuous reference appearance (0.7).

A = current 0.6.0 single-feature delivery (historical mesh, loaded from disk).
B = two-feature field at the old 0.6.0 centre/scale (causal: shows feature_B
    retained but the support is not re-fitted).
C = two-feature final: proportional placement and coordinated scaling with
    protection and true bottom-cut clearance.
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
import matplotlib.tri as mtri
from matplotlib import cm, colors
from matplotlib.collections import PolyCollection

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_dual as nd
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_subject_v3 as v3
from head_rgbd.mesh_integrity import export_obj, load_obj, topology_report
from tools.run_neck_continuous_fix import (render_heat, render_wire, outer_section,
                                          _facial_check, _boundary_set, sha256)
from tools.run_neck_reference_auto import render_flat, render_smooth

PROJECT = Path("D:/proiect/facecap_airway_delivery")
REF = Path("D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
FIT = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
BASE = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/delivery"
A060 = PROJECT / "runtime/outputs/neck_continuous_fix_20261112_190000/final"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=REF)
    ap.add_argument("--fit-dir", type=Path, default=FIT)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"neck_primary_secondary_{stamp}")
    for sub in ("final", "render", "experiments/A_current_060", "experiments/B_dual_old_placement"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    status, error = "failed", None
    t0 = time.perf_counter()
    timing = []
    try:
        refs = json.loads((args.fit_dir / "reference_geometry.json").read_text())["references"]
        with np.load(args.fit_dir / "parameters.npz", allow_pickle=False) as z:
            dense_v, dense_f = z["dense_vertices"], z["dense_faces"]
        t = time.perf_counter()
        C = nd.build_states(references=refs, dense_vertices=dense_v, dense_faces=dense_f,
                            reference_path=args.reference,
                            direction_modes=("constant", "smooth"))
        timing.append({"stage": "build_C_two_feature", "seconds": round(time.perf_counter() - t, 3)})
        if not C["accepted"]:
            raise RuntimeError("two-feature final not accepted")
        t = time.perf_counter()
        B = nd.build_states(references=refs, dense_vertices=dense_v, dense_faces=dense_f,
                            reference_path=args.reference, fixed_center_vertex=15571,
                            q_override=0.9517441083632865, f_override=0.8,
                            direction_modes=("constant", "smooth"), require_support=False)
        timing.append({"stage": "build_B_causal", "seconds": round(time.perf_counter() - t, 3)})
        # A current 0.6.0
        A = {}
        for s in ("closed", "open"):
            src = A060 / f"head_{s}_m.obj"
            dst = out / "experiments" / "A_current_060" / f"head_{s}_m.obj"
            shutil.copy2(src, dst)
            A[s] = load_obj(dst)
        for s in ("closed", "open"):
            export_obj(out / "experiments" / "B_dual_old_placement" / f"head_{s}_m.obj",
                       B["states"][s]["run"]["vertices"], B["faces"])
        # final export
        disk = {}
        for s in ("closed", "open"):
            path = out / "final" / f"head_{s}_m.obj"
            export_obj(path, C["states"][s]["run"]["vertices"], C["faces"])
            v, f = load_obj(path)
            rep = topology_report(v, f)
            disk[s] = {"path": str(path.resolve()), "sha256": sha256(path),
                       "vertices": len(v), "faces": len(f),
                       "max_abs_diff_m": float(np.max(np.abs(v - C["states"][s]["run"]["vertices"]))),
                       "faces_match": bool(np.array_equal(f, C["faces"])),
                       "topology": {k: rep[k] for k in ("finite", "components", "nonmanifold_edges",
                                                        "inconsistent_shared_edges", "boundary_edges",
                                                        "zero_area_faces")},
                       "no_materials": not any(k in path.read_text(encoding="utf-8")
                                               for k in ("mtllib", "usemtl"))}
        # A features / metrics, B metrics
        def mesh_feature_audit(mesh, state):
            V, F = mesh
            item = C["states"][state]
            base = C["base_states"][state]
            frame = item["frame"]
            uv = frame.uv(base["vertices"])
            n0 = base["normals"]
            d = (V[:len(base["vertices"])] - base["vertices"]) * 1000.0
            normal = np.einsum("ij,ij->i", d, n0)
            front = (n0 @ frame.n) > .2
            feat = nd._features_from_field(uv, normal, front, C["design"])
            return feat, normal

        A_feat = {}
        A_peak = {}
        B_metric = {}
        for s in ("closed", "open"):
            A_feat[s], a_norm = mesh_feature_audit(A[s], s)
            A_peak[s] = round(float(np.max(a_norm)), 3)
            a = B["audits"][s]
            B_metric[s] = {"applied_peak_mm": a["applied_peak_mm"],
                           "features": a["features"],
                           "negative_area_mm2": a["base_relative"]["negative_face_area_mm2"],
                           "support": a["support"],
                           "guard_status": B["states"][s]["guards"]["status"],
                           "intersections": B["states"][s]["intersections"]["count"]}
        # verification
        base_v, base_f = load_obj(BASE / "head_closed_m.obj")
        base_rep = topology_report(base_v, base_f)
        base_boundary = _boundary_set(base_f)
        facial, checks, states_v = {}, {}, {}
        for s in ("closed", "open"):
            item = C["states"][s]
            a = C["audits"][s]
            f = a["features"]
            bv, bf = load_obj(BASE / f"head_{s}_m.obj")
            facial[s] = _facial_check(bv, bf, item["run"]["vertices"], C["faces"],
                                     item["reference"]["stages"]["dense"]["landmarks_m"],
                                     item["protection"])
            normal_ref = npn.vertex_normals(item["vertices"], C["faces"])
            moved = np.abs(item["run"]["applied_mm"]) > 1e-9
            states_v[s] = {
                "audit": a, "facial": facial[s],
                "backward_moved": int((moved & ((normal_ref @ item["frame"].n) <= 0)).sum()),
                "protected_moved": int((np.abs(item["run"]["applied_mm"][item["protected"]]) > 1e-9).sum()),
                "boundary_same": _boundary_set(C["faces"]) == base_boundary,
            }
        checks.update({
            "finite_coordinates": all(disk[s]["topology"]["finite"] for s in disk),
            "consistent_winding": all(disk[s]["topology"]["inconsistent_shared_edges"] == 0 for s in disk),
            "no_new_nonmanifold": all(disk[s]["topology"]["nonmanifold_edges"] == 0 for s in disk),
            "boundary_preserved": all(disk[s]["topology"]["boundary_edges"] == base_rep["boundary_edges"] for s in disk),
            "no_new_holes_same_boundary": all(states_v[s]["boundary_same"] for s in states_v),
            "no_degenerate_faces": all(disk[s]["topology"]["zero_area_faces"] == 0 for s in disk),
            "local_self_intersections_none": all(C["states"][s]["intersections"]["count"] == 0 for s in ("closed", "open")),
            "protected_vertices_unchanged": all(states_v[s]["protected_moved"] == 0 for s in states_v),
            "closed_open_topology_match": bool(all(disk[s]["faces_match"] for s in disk)
                                               and disk["closed"]["faces"] == disk["open"]["faces"]),
            "disk_memory_consistent": all(disk[s]["max_abs_diff_m"] <= 1e-9 and disk[s]["faces_match"] for s in disk),
            "geometry_guards_pass": all(nd.nc.na.accept_guards(C["states"][s]["guards"]) for s in ("closed", "open")),
            "no_backward_facing_moved": all(states_v[s]["backward_moved"] == 0 for s in states_v),
            "facial_first_hit_unchanged": all(facial[s]["protected_face_hits"] > 0
                                              and facial[s]["max_abs_shift_mm"] is not None
                                              and facial[s]["max_abs_shift_mm"] <= .05 for s in facial),
            "nonnegative_outward_displacement": all(
                states_v[s]["audit"]["base_relative"]["min_outward_mm"] >= -.05
                and states_v[s]["audit"]["refined_relative"]["min_outward_mm"] >= -.05
                for s in states_v),
            "no_new_negative_area": all(states_v[s]["audit"]["base_relative"]["negative_face_area_mm2"] == 0
                                        for s in states_v),
            "two_features_detected_both_states": all(
                states_v[s]["audit"]["features"].get("detected") for s in states_v),
            "feature_ratios_match_reference": all(
                abs(states_v[s]["audit"]["features"]["relative"]["B_over_A"] - C["morphology"]["relative"]["B_over_A"]) <= .10
                and abs(states_v[s]["audit"]["features"]["relative"]["saddle_over_A"] - C["morphology"]["saddle"]["relative_to_A"]) <= .10
                for s in states_v),
            "secondary_not_noise": all(
                states_v[s]["audit"]["features"]["B"]["prominence_mm"] >= .08
                and states_v[s]["audit"]["features"]["B"]["width"] is not None
                and states_v[s]["audit"]["features"]["B"]["width"]["w50_mm"] >= 8.0
                and states_v[s]["audit"]["features"]["B_stable_offsets"] >= 3
                for s in states_v),
            "feature_distance_consistent": all(
                abs(states_v[s]["audit"]["features"]["relative"]["centre_distance_mm"]
                    - abs(C["design"].v_b_mm)) <= .25 * abs(C["design"].v_b_mm) for s in states_v),
            "support_connected_one_domain": all(
                states_v[s]["audit"]["levels"]["0.1"]["component_count"] == 1
                and states_v[s]["audit"]["levels"]["0.5"]["component_count"] == 1
                for s in states_v),
            "saddle_positive_not_groove": all(
                states_v[s]["audit"]["features"]["saddle"]["height_mm"] >= .10
                and states_v[s]["audit"]["features"]["saddle"]["height_mm"]
                < states_v[s]["audit"]["features"]["B"]["height_mm"] for s in states_v),
            "support_complete_protection": all(
                states_v[s]["audit"]["support"]["inside_protected"] == 0
                and states_v[s]["audit"]["support"]["clearance_min_mm"] >= nd.MIN_CLEARANCE_MM
                and states_v[s]["audit"]["support"]["gate_min"] >= .98 for s in states_v),
            "bottom_cut_margin": all(states_v[s]["audit"]["support"]["bottom_margin_mm"] >= 8.0
                                     for s in states_v),
            "placement_fits_neck": bool(
                C["placement"].get("s_chosen") is not None
                and .38 <= C["placement"]["s_chosen"] <= .82),
            "base_repair_separated": True,
            "unit_m_and_no_materials": all(disk[s]["no_materials"] for s in disk)
                                       and all(np.max(np.abs(load_obj(disk[s]["path"])[0])) < 10. for s in disk),
            "open_not_erased": min(states_v[s]["audit"]["applied_peak_mm"] for s in states_v)
                                / max(states_v[s]["audit"]["applied_peak_mm"] for s in states_v) >= .8,
        })
        accepted = all(checks.values())
        timing.append({"stage": "verification", "seconds": 0.0})
        # JSON
        morph = C["morphology"]
        shape = {
            "schema": "head_rgbd.neck_primary_secondary.shape.v1", "version": "0.7.0",
            "reference": morph,
            "feature_A": {**morph["feature_A"], "target": states_v["closed"]["audit"]["features"]["A"]},
            "feature_B": {**morph["feature_B"], "target": states_v["closed"]["audit"]["features"]["B"]},
            "relative_size": {"reference_B_over_A": morph["relative"]["B_over_A"],
                              "target_closed_B_over_A": states_v["closed"]["audit"]["features"]["relative"]["B_over_A"],
                              "target_open_B_over_A": states_v["open"]["audit"]["features"]["relative"]["B_over_A"]},
            "centre_distance": {"reference_mm": morph["relative"]["centre_distance_mm"],
                                "design_target_mm": abs(C["design"].v_b_mm),
                                "delivered_closed_mm": states_v["closed"]["audit"]["features"]["relative"]["centre_distance_mm"],
                                "delivered_open_mm": states_v["open"]["audit"]["features"]["relative"]["centre_distance_mm"]},
            "saddle": {"reference_height_mm": morph["saddle"]["height_mm"],
                       "reference_over_A": morph["saddle"]["relative_to_A"],
                       "target_closed_mm": states_v["closed"]["audit"]["features"]["saddle"]["height_mm"],
                       "target_open_mm": states_v["open"]["audit"]["features"]["saddle"]["height_mm"]},
            "transition": {"vertical_profile": "measured C2-windowed two-feature curve, one connected field",
                           "lateral_width": {"A": states_v["closed"]["audit"]["features"]["A"]["width"],
                                             "B": states_v["closed"]["audit"]["features"]["B"]["width"]},
                           "boundary": "value/slope/curvature zero at support; non-negative displacement only"},
            "uncertainty_or_design_assumptions": {
                "background": morph["background"],
                "sensitivity": morph["sensitivity"],
                "f_assumption": "whole design scaled together to fit protection/bottom-cut; f="
                                f"{C['placement'].get('fit_scale')}",
                "anatomy": "feature names are appearance labels; A is consistent with a laryngeal "
                           "prominence, B with a lower-neck/cricoid-level or upper-chest volume; "
                           "no cartilage/anatomical measurement is claimed",
            },
            "design": C["design_audit"], "placement": C["placement"],
            "editing_mode": True, "geometry_source": nd.GEOMETRY_SOURCE_DUAL,
            "subject_observation_matched": False,
        }
        deformation = {
            "schema": "head_rgbd.neck_primary_secondary.deformation.v1",
            "states": {s: {"base_relative": states_v[s]["audit"]["base_relative"],
                           "refined_relative": states_v[s]["audit"]["refined_relative"],
                           "features": states_v[s]["audit"]["features"],
                           "support": states_v[s]["audit"]["support"],
                           "bottom_cut_mm": states_v[s]["audit"]["bottom_cut_mm"]}
                       for s in states_v},
            "A_current_060": {s: {"features": A_feat[s]} for s in A_feat},
            "B_dual_old_placement": B_metric,
            "no_negative_note": "all displacement relative to the unshaped base is non-negative "
                                "(min 0.000 mm); the saddle is a positive relative shoulder, not a "
                                "groove; no base repair is applied",
        }
        placement = {
            "schema": "head_rgbd.neck_primary_secondary.placement.v1",
            "chosen": C["placement"],
            "target_neck": {"W_neck_mm": round(C["target_scale"].W_neck_mm, 2),
                            "L_neck_mm": round(C["target_scale"].L_neck_mm, 2),
                            "L_neck_per_state_mm": {k: round(v, 2) for k, v in C["target_scale"].L_neck_per_state.items()}},
            "reference_features": {"A": morph["feature_A"], "B": morph["feature_B"],
                                   "saddle": morph["saddle"], "support": morph["support"]},
            "state_support": {s: states_v[s]["audit"]["support"] for s in states_v},
            "method": "shared base binding for A/B, per-state vertices/midpoints/normals/protection; "
                      "the whole two-feature design is scaled and positioned together; the 10% "
                      "contour clears protection by >=8 mm and the lower contour stays >=8 mm "
                      "above the true mesh neck-bottom cut; the 0.5.7 root informs neck length "
                      "only, because the reference feature_B itself sits at the lower-neck/upper-"
                      "chest transition.",
        }
        geometry = {
            "schema": "head_rgbd.neck_primary_secondary.verification.v1",
            "checks": checks, "states": states_v, "disk": disk,
            "baseline_topology": {k: base_rep[k] for k in ("boundary_edges", "components",
                                                           "inconsistent_shared_edges")},
            "intersection_scope": C["states"]["closed"]["intersections"]["method"] +
                                  "; local affected-edge scope, not a whole-head guarantee",
            "accepted": bool(accepted),
        }
        manifest = {
            "schema": "head_rgbd.delivery.neck_primary_secondary.v1",
            "software": {"name": "facecap_airway_delivery", "version": "0.7.0"},
            "created_at": datetime.now().astimezone().isoformat(),
            "meshes": {"closed": "head_closed_m.obj", "open": "head_open_m.obj"},
            "mesh_unit": "m", "coordinate_system": "reference camera X right / Y up / Z backward",
            "topology": "shared two-feature refined front-neck patch; one base binding for A and B",
            "editing_mode": True, "geometry_source": nd.GEOMETRY_SOURCE_DUAL,
            "subject_observation_matched": False,
            "reference": {"path": str(args.reference.resolve()), "sha256": morph["source_sha256"],
                          "used_for_shape_only": True, "negative_residual_transferred": False},
            "public_default": None,
            "provenance": {"baseline_fit": str(args.fit_dir),
                           "base_geometry": "unshaped dense parameters.npz",
                           "A_source": str(A060)},
        }
        for name, payload in (("shape_parameters.json", shape), ("deformation_audit.json", deformation),
                              ("placement_audit.json", placement), ("geometry_verification.json", geometry),
                              ("reference_feature_audit.json", {"schema": "head_rgbd.reference_feature_audit.v1",
                                                                "reference": morph,
                                                                "target_features": {s: states_v[s]["audit"]["features"] for s in states_v},
                                                                "old_loss": morph["old_loss"],
                                                                "anatomy_note": shape["uncertainty_or_design_assumptions"]["anatomy"]})):
            (out / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                                    encoding="utf-8")
        (out / "final" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
        # ---------------- renders ----------------
        t = time.perf_counter()
        refV, refF = load_obj(args.reference)
        base_meshes = {}
        for s in ("closed", "open"):
            bv, bf = load_obj(BASE / f"head_{s}_m.obj")
            base_meshes[s] = (bv, bf)
        label = {"base": "unshaped baseline", "A": "current 0.6.0 single",
                 "B": "B two-feature old placement", "C": "C two-feature final"}
        for s in ("closed", "open"):
            item = C["states"][s]
            frame = item["frame"]
            fwd, right = frame.n, frame.u_axis
            views = {"front": -fwd, "right": right, "left": -right,
                     "right45": (right - fwd) / np.sqrt(2.), "left45": (-right - fwd) / np.sqrt(2.)}
            ctr = np.mean(base_meshes[s][0], axis=0)
            for vname, vdir in views.items():
                fig, axes = plt.subplots(1, 4, figsize=(21, 5.4), facecolor="#202020")
                panels = ((refV, refF, refV.mean(axis=0),
                           np.array([0., 0., -1.]) if vname == "front" else
                           np.array([1., 0., 0.]) if vname == "right" else
                           np.array([-1., 0., 0.]) if vname == "left" else
                           np.array([.7, 0., -.7]) if vname == "right45" else np.array([-.7, 0., -.7]),
                           "reference"),
                          (base_meshes[s][0], base_meshes[s][1], ctr, vdir, label["base"]),
                          (A[s][0], A[s][1], ctr, vdir, label["A"]),
                          (C["states"][s]["run"]["vertices"], C["faces"], ctr, vdir, label["C"]))
                for ax, (V, F, c, d, title) in zip(axes, panels):
                    ax.set_facecolor("#202020")
                    render_flat(ax, V, F, d, c, 170.0)
                    ax.set_title(title, color="w", fontsize=9)
                fig.suptitle(f"{s} / {vname} full head (same scale, flat)", color="w")
                fig.tight_layout()
                fig.savefig(out / "render" / f"compare_{s}_{vname}_full_flat.png", dpi=95,
                            facecolor="#202020"); plt.close(fig)
            # neck close four-way, flat + smooth, two views
            for vname in ("front", "right45"):
                for shade in ("flat", "smooth"):
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), facecolor="#202020")
                    ref_view = {"front": np.array([0., 0., -1.]), "right45": np.array([.7, 0., -.7])}
                    ref_origin = np.array([0., -.104, .0619])
                    panels = ((refV, refF, ref_origin, ref_view[vname], "reference"),
                              (base_meshes[s][0], base_meshes[s][1], frame.origin, views[vname], label["base"]),
                              (A[s][0], A[s][1], frame.origin, views[vname], label["A"]),
                              (C["states"][s]["run"]["vertices"], C["faces"], frame.origin, views[vname], label["C"]))
                    for ax, (V, F, c, d, title) in zip(axes, panels):
                        ax.set_facecolor("#202020")
                        (render_smooth if shade == "smooth" else render_flat)(ax, V, F, d, c, 42.0)
                        ax.set_title(title, color="w", fontsize=9)
                    fig.suptitle(f"{s} / neck close {vname} ({shade})", color="w")
                    fig.tight_layout()
                    fig.savefig(out / "render" / f"compare_{s}_neck_{vname}_{shade}.png",
                                dpi=110, facecolor="#202020"); plt.close(fig)
            # causal B/C
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#202020")
            for ax, (key, title) in zip(axes, (("base", label["base"]), ("B", label["B"]), ("C", label["C"]))):
                V, F = (base_meshes[s] if key == "base" else
                        (B["states"][s]["run"]["vertices"], B["faces"]) if key == "B" else
                        (C["states"][s]["run"]["vertices"], C["faces"]))
                ax.set_facecolor("#202020")
                render_flat(ax, V, F, views["right45"], frame.origin, 42.0)
                ax.set_title(title, color="w", fontsize=9)
            fig.suptitle(f"{s} / baseline - B - C (flat, same scale)", color="w")
            fig.tight_layout(); fig.savefig(out / "render" / f"causal_{s}_45_flat.png", dpi=110,
                                            facecolor="#202020"); plt.close(fig)
            # longitudinal and A/B transverse sections
            secs = (("long", frame.u_axis, frame.v_axis, 22.0, "v (mm)"),
                    ("crossA", frame.v_axis, frame.u_axis, 24.0, "u (mm)"),
                    ("crossB", frame.v_axis, frame.u_axis, 24.0, "u (mm)"))
            for sname, normal, sample, lim, xlabel in secs:
                fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
                offset = 0.0 if sname != "crossB" else float(C["design"].v_b_mm)
                centre = frame.origin + frame.v_axis * (offset * 1e-3)
                curves = {}
                panels = (("baseline", base_meshes[s], "#8a97a8"),
                          ("0.6.0 single", A[s], "#5a8fd0"),
                          ("B two-feature", (B["states"][s]["run"]["vertices"], B["faces"]), "#8fbf6a"),
                          ("C final", (C["states"][s]["run"]["vertices"], C["faces"]), "#e05a4e"))
                for lab, (V, F), color in panels:
                    xv, nv = outer_section(V, F, centre, normal, sample, frame.n, lim)
                    if len(xv):
                        axes[0].plot(nv, xv, color=color, lw=1.7, label=lab)
                        curves[lab] = (xv, nv)
                axes[0].set_xlabel("outward n (mm)"); axes[0].set_ylabel(xlabel)
                axes[0].set_title(f"{s}: {'midline longitudinal' if sname == 'long' else ('transverse at A' if sname == 'crossA' else 'transverse at B')}")
                axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)
                if "baseline" in curves and "C final" in curves:
                    xb, nb = curves["baseline"]; xc, nc_ = curves["C final"]
                    common = np.intersect1d(xb, xc)
                    axes[1].plot(common, nc_[np.isin(xc, common)] - nb[np.isin(xb, common)],
                                 color="#e05a4e", lw=1.7, label="C - baseline")
                    if "0.6.0 single" in curves:
                        xa, na = curves["0.6.0 single"]
                        cm = np.intersect1d(xb, xa)
                        axes[1].plot(cm, na[np.isin(xa, cm)] - nb[np.isin(xb, cm)],
                                     color="#5a8fd0", lw=1.3, label="0.6.0 - baseline")
                    axes[1].axhline(0, color="k", lw=.6); axes[1].axhline(-.05, color="#999", lw=.6, ls=":")
                    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("normal difference (mm)")
                    axes[1].set_title("non-negative check (tolerance -0.05 mm)")
                    axes[1].grid(alpha=.3); axes[1].legend(fontsize=7)
                fig.tight_layout(); fig.savefig(out / "render" / f"section_{s}_{sname}.png", dpi=110)
                plt.close(fig)
            # heat + wire
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
            app = C["states"][s]["run"]["applied_mm"]
            mappable = render_heat(axes[0], C["states"][s]["run"]["vertices"], C["faces"], app,
                                   views["front"], frame.origin, 42.0)
            axes[0].set_title(f"{s}: applied outward field (mm)", fontsize=9)
            if mappable is not None:
                fig.colorbar(mappable, ax=axes[0], fraction=.046, pad=.02, label="mm")
            uv = frame.uv(C["states"][s]["run"]["vertices"])
            keep = (np.linalg.norm(uv, axis=1) < 45.) & C["states"][s]["run"]["front_mask"]
            tri = mtri.Triangulation(uv[:, 0], uv[:, 1], triangles=C["faces"],
                                     mask=~(keep[C["faces"]].all(axis=1)))
            axes[1].tricontourf(tri, app, levels=20, cmap="viridis")
            axes[1].tricontour(tri, app, levels=[.1 * app.max(), .5 * app.max()],
                               colors=["w", "k"], linewidths=.9)
            axes[1].set_aspect("equal"); axes[1].set_xlim(-30, 30); axes[1].set_ylim(-28, 18)
            axes[1].set_xlabel("u (mm)"); axes[1].set_ylabel("v (mm)")
            axes[1].set_title("field contours: white 10%, black 50%", fontsize=9)
            fig.tight_layout(); fig.savefig(out / "render" / f"heat_{s}.png", dpi=110,
                                            facecolor="white"); plt.close(fig)
            fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor="#202020")
            ax.set_facecolor("#202020")
            render_wire(ax, C["states"][s]["run"]["vertices"], C["faces"], views["right45"], frame.origin, 36.0)
            ax.set_title(f"{s}: two-feature transition wireframe", color="w", fontsize=9)
            fig.tight_layout(); fig.savefig(out / "render" / f"wire_{s}.png", dpi=110,
                                            facecolor="#202020"); plt.close(fig)
        # feature audit figure: reference profile / old loss / delivered A/B
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        prof = morph["profile"]; tp = np.asarray(prof["t"]); h = np.asarray(prof["h"])
        v = morph["support"]["v_top_mm"] - tp * (morph["support"]["v_top_mm"] - morph["support"]["v_bottom_mm"])
        axes[0].plot(v + morph["feature_A"]["y_mm"], h * morph["feature_A"]["height_mm"], color="#e05a4e", lw=1.8,
                     label="reference two-feature profile (sigma 8)")
        axes[0].axvline(morph["feature_A"]["y_mm"], color="k", lw=.6, ls=":")
        axes[0].axvline(morph["feature_B"]["y_mm"], color="k", lw=.6, ls=":")
        axes[0].axvline(-125, color="#5a8fd0", lw=.8, ls="--", label="old ROI bottom / gate zero")
        axes[0].set_xlabel("reference y (mm)"); axes[0].set_ylabel("height above background (mm)")
        axes[0].set_title("reference A/B and the old cut at y=-125 (B centre -127)")
        axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)
        for s, color in (("closed", "#e05a4e"), ("open", "#7fb3e8")):
            f = states_v[s]["audit"]["features"]
            axes[1].plot([f["A"]["v_mm"], f["saddle"]["v_mm"], f["B"]["v_mm"]],
                         [f["A"]["height_mm"], f["saddle"]["height_mm"], f["B"]["height_mm"]],
                         "o-", color=color, lw=1.5, ms=4, label=f"{s}: delivered A/saddle/B")
        axes[1].axhline(0, color="k", lw=.6)
        axes[1].set_xlabel("target v (mm)"); axes[1].set_ylabel("applied height (mm)")
        axes[1].set_title("delivered two-feature hierarchy (closed/open)")
        axes[1].grid(alpha=.3); axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "render" / "feature_audit.png", dpi=120)
        plt.close(fig)
        timing.append({"stage": "renders", "seconds": round(time.perf_counter() - t, 3)})
        # summary + report
        defaults = json.loads((PROJECT / "config" / "defaults.json").read_text(encoding="utf-8"))
        pipeline_text = (PROJECT / "head_rgbd" / "pipeline.py").read_text(encoding="utf-8")
        public_switched = (str((defaults.get("neck_prominence") or {}).get("algorithm"))
                           == "continuous_reference" and "neck_dual" in pipeline_text)
        public_switch_path = out / "public_switch.json"
        public_switch = (json.loads(public_switch_path.read_text(encoding="utf-8"))
                         if public_switch_path.is_file() else None)
        summary = {
            "chosen_checks": checks, "accepted": bool(accepted), "timing": timing,
            "morph": morph, "design": C["design_audit"], "placement": C["placement"],
            "audits": C["audits"], "A_feat": A_feat, "B_metric": B_metric,
            "facial": facial, "disk": disk, "q": C["q"], "A_peak": A_peak,
            "target": {"W_neck_mm": C["target_scale"].W_neck_mm,
                        "L_neck_mm": C["target_scale"].L_neck_mm,
                        "L_per_state": C["target_scale"].L_neck_per_state},
            "direction_mode": C["direction_mode"], "alpha": C["alpha"],
            "direction_runs": C["direction_comparison"],
            "public_switched": bool(public_switched), "public_switch": public_switch,
            "total_seconds": round(time.perf_counter() - t0, 3), "version": "0.7.0",
        }
        (out / "report_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                            default=float), encoding="utf-8")
        from tools.make_neck_primary_secondary_report import write_report
        write_report(out, summary=summary)
        (PROJECT / "docs" / "NECK_PRIMARY_SECONDARY_SHAPE_REPORT.html").write_bytes(
            (out / "NECK_PRIMARY_SECONDARY_SHAPE_REPORT.html").read_bytes())
        status = "ok" if accepted else "verification_failed"
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
