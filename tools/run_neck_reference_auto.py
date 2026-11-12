"""Formal reference-driven auto sculpt: extract, bind, build, export, verify, render, report.

Default parameters work for the supplied reference; no manual tuning round.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_prominence as prominence
from head_rgbd import neck_reference_auto as nra
from head_rgbd import neck_subject_v3 as v3
from head_rgbd.mesh_integrity import export_obj, load_obj, orient_faces_consistently, topology_report
from tools.run_neck_subject_v3 import sha256
from tools.run_neck_sculpt import plane_section

PROJECT = Path("D:/proiect/facecap_airway_delivery")
DEFAULT_REFERENCE = Path("D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
DEFAULT_FIT = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
OLD_DELIVERY = PROJECT / "runtime/outputs/neck_broad_volume_20261112_152709/final"
BASELINE_DELIVERY = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/delivery"


def render_flat(ax, V, F, view_dir, centre, half):
    import matplotlib.colors
    fwd = np.asarray(view_dir, float); fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    facing = (n @ (-fwd)) > 0.02
    tri = tri[facing]; n = n[facing]
    if not len(tri):
        return
    light = -fwd + 0.35 * up - 0.20 * right
    light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.0, 1.0) * 0.8 + 0.2
    centroid = tri.mean(axis=1)
    depth = (centroid - centre) @ fwd
    order = np.argsort(-depth)
    rel = tri - centre
    polys = np.stack([rel @ right * 1000.0, rel @ up * 1000.0], axis=2)[order]
    colors = np.repeat(np.clip(shade[order][:, None], 0, 1), 3, axis=1)
    from matplotlib.collections import PolyCollection
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="none"))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.axis("off")


def render_smooth(ax, V, F, view_dir, centre, half):
    """Smooth (area-weighted vertex normal) shading: proves geometry, not flat-facet artifacts."""
    fwd = np.asarray(view_dir, float); fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    from head_rgbd import neck_prominence as npn
    vn = npn.vertex_normals(V, F)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    facing = (fn @ (-fwd)) > 0.02
    tri = tri[facing]
    bary_n = vn[F][facing].mean(axis=1)
    bary_n /= np.maximum(np.linalg.norm(bary_n, axis=1, keepdims=True), 1e-12)
    light = -fwd + 0.35 * up - 0.20 * right
    light /= np.linalg.norm(light)
    shade = np.clip(bary_n @ light, 0.0, 1.0) * 0.8 + 0.2
    centroid = tri.mean(axis=1)
    depth = (centroid - centre) @ fwd
    order = np.argsort(-depth)
    rel = tri - centre
    polys = np.stack([rel @ right * 1000.0, rel @ up * 1000.0], axis=2)[order]
    colors = np.repeat(np.clip(shade[order][:, None], 0, 1), 3, axis=1)
    from matplotlib.collections import PolyCollection
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="none"))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.axis("off")


def render_field(out, state_name, item, analysis, faces):
    """Contour map and midline profiles of the realised field (auditable, not cropped)."""
    import matplotlib.tri as mtri
    uv = np.asarray(item["uv_mm"], float)
    applied = np.asarray(item["applied_mm"], float)
    front = prominence.vertex_normals(item["refined_vertices"], faces) @ item["frame"].n > .2
    near = np.linalg.norm(uv, axis=1) < 60.
    keep_face = near[faces].all(axis=1) & front[faces].all(axis=1)
    peak = float(analysis["peak_outward_mm"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    triang = mtri.Triangulation(uv[:, 0], uv[:, 1], triangles=faces, mask=~keep_face)
    lim = max(abs(applied.min()), applied.max())
    fill = axes[0].tricontourf(triang, applied, levels=np.linspace(-lim, lim, 41), cmap="coolwarm")
    axes[0].tricontour(triang, applied, levels=[.1*peak, .5*peak], colors=["#ffffff", "#ffd400"],
                       linewidths=1.0)
    apex = analysis["apex_uv_mm"]
    axes[0].plot(apex[0], apex[1], "+", color="k", ms=9)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("u (mm)"); axes[0].set_ylabel("v (mm)")
    axes[0].set_title(f"{state_name}: applied field, white=10% / yellow=50%")
    fig.colorbar(fill, ax=axes[0], label="mm")
    prof = analysis["profiles"]
    for key, style, label in (("longitudinal_u0", "-", "u=0 longitudinal"),
                              ("transverse_apex", "--", "v=apex transverse")):
        if key not in prof:
            continue
        values = np.array([np.nan if v is None else v for v in prof[key]["disp_mm"]], float)
        axes[1].plot(prof[key]["axis_mm"], values, style, lw=1.6, label=label)
    axes[1].axhline(0, color="k", lw=.6)
    axes[1].axhline(.1*peak, color="#999", lw=.6, ls=":")
    axes[1].axhline(.5*peak, color="#999", lw=.6, ls=":")
    axes[1].set_xlabel("axis (mm)"); axes[1].set_ylabel("outward displacement (mm)")
    axes[1].set_title(f"{state_name}: midline profiles (dotted 10% / 50%)")
    axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "render" / f"field_{state_name}.png", dpi=110, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"neck_reference_auto_{stamp}")
    (out / "render").mkdir(parents=True, exist_ok=True)
    (out / "final").mkdir(exist_ok=True)
    status, error = "failed", None
    t_all = time.perf_counter()
    timing = []
    try:
        # ---------- 1. extract the reference field ----------
        t0 = time.perf_counter()
        asset, ref_V, ref_F = nra.extract_reference(args.reference)
        asset["background_sensitivity"] = nra.background_sensitivity(args.reference)
        timing.append({"stage": "reference_extract", "seconds": round(time.perf_counter() - t0, 3)})
        (out / "reference_shape.json").write_text(
            json.dumps(asset, indent=1, ensure_ascii=False), encoding="utf-8")

        # ---------- 2. build both states ----------
        references = json.loads((args.fit_dir / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
        with np.load(args.fit_dir / "parameters.npz", allow_pickle=False) as stored:
            dense_vertices, dense_faces = stored["dense_vertices"], stored["dense_faces"]
        t0 = time.perf_counter()
        plan = nra.build_states(references, dense_vertices, dense_faces, asset, strength=args.strength)
        timing.append({"stage": "build_states", "seconds": round(time.perf_counter() - t0, 3)})
        if not plan["accepted"]:
            print("build not accepted; guards:", {s: plan["states"][s]["guards"] for s in plan["states"]})
        faces = plan["faces"]
        t0 = time.perf_counter()
        analysis = {s: nra.analyse_applied(plan["states"][s], faces) for s in ("closed", "open")}
        timing.append({"stage": "shape_analysis", "seconds": round(time.perf_counter() - t0, 3)})

        # ---------- 3. orientation-consistent export + disk read-back ----------
        t0 = time.perf_counter()
        disk = {}
        for state in ("closed", "open"):
            item = plan["states"][state]
            shaped = item["shaped_vertices"]
            path = out / "final" / f"head_{state}_m.obj"
            export_obj(path, shaped, faces)
            v_read, f_read = load_obj(path)
            rep = topology_report(v_read, f_read)
            mesh = trimesh.load(str(path), process=False, maintain_order=True)
            disk[state] = {
                "path": str(path), "sha256": sha256(path),
                "vertices": int(len(v_read)), "faces": int(len(f_read)),
                "max_abs_diff_m": float(np.max(np.abs(v_read - shaped))) if len(v_read) == len(shaped) else None,
                "faces_match": bool(np.array_equal(f_read, faces)),
                "topology": {k: rep[k] for k in ("finite", "components", "nonmanifold_edges",
                                                 "inconsistent_shared_edges", "boundary_edges",
                                                 "zero_area_faces", "small_faces_below_0_01_mm2")},
                "vertex_normals_recomputed": True,
                "obj_normals_finite": bool(np.isfinite(np.asarray(mesh.vertex_normals)).all()),
            }
        timing.append({"stage": "export_readback", "seconds": round(time.perf_counter() - t0, 3)})

        # ---------- 4. orientation-only repair of the old delivery ----------
        orient_fix_dir = out / "old_delivery_orient_fix"
        orient_fix_dir.mkdir(exist_ok=True)
        old_fix = {}
        for state in ("closed", "open"):
            V_old, F_old = load_obj(OLD_DELIVERY / f"head_{state}_m.obj")
            F_fixed, info = orient_faces_consistently(F_old)
            export_obj(orient_fix_dir / f"head_{state}_m.obj", V_old, F_fixed)
            rep = topology_report(V_old, F_fixed)
            old_fix[state] = {
                "reversed_faces": info["reversed_faces"],
                "reversed_face_ids": info["face_ids"],
                "vertex_movement_mm": info["vertex_movement_mm"],
                "inconsistent_after": rep["inconsistent_shared_edges"],
                "boundary_edges": rep["boundary_edges"],
                "path": str(orient_fix_dir / f"head_{state}_m.obj"),
            }
        (orient_fix_dir / "repair_report.json").write_text(json.dumps({
            "schema": "head_rgbd.old_delivery_orient_fix.v1",
            "method": "face reordering only (reverse the minority winding parity per component); "
                      "vertices untouched; normals recomputed area-weighted on export",
            "states": old_fix,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # ---------- 5. verification ----------
        t0 = time.perf_counter()
        verification = {}
        for state in ("closed", "open"):
            item = plan["states"][state]
            guards = item["guards"]
            refined_normals = prominence.vertex_normals(item["shaped_vertices"], faces)
            moved = np.abs(item["applied_mm"]) > 1e-9
            backward_moved = int((moved & ((refined_normals @ item["frame"].n) <= 0.)).sum())
            # Facial first hits are classified by ORIGINAL baseline face indices.
            base_v, base_f = load_obj(BASELINE_DELIVERY / f"head_{state}_m.obj")
            lm = np.asarray(item["landmarks_camera"], float)
            hit_base = v3.first_hit_batch(base_v, base_f, lm)
            hit_new = v3.first_hit_batch(item["shaped_vertices"], faces, lm)
            hit = hit_base["hit"] & hit_new["hit"]
            protected_face = item["protection"][base_f[hit_base["triangle"]]].all(axis=1)
            checked = hit & protected_face
            shift_mm = (hit_new["depth_m"] - hit_base["depth_m"]) * 1000.
            facial = {
                "rays": int(len(lm)), "hits": int(hit.sum()),
                "protected_face_hits": int(checked.sum()),
                "max_abs_shift_mm": round(float(np.max(np.abs(shift_mm[checked])))
                                          if checked.any() else None, 5),
                "classification": "original baseline face indices; protected face/lips/jaw/"
                                  "ears/back/bottom faces only",
            }
            verification[state] = {
                "guards": guards,
                "intersections": item["intersections"],
                "metrics": item["metrics"],
                "protected_displacement_mm": item["metrics"]["protected_displacement_mm"],
                "backward_facing_moved_vertices": backward_moved,
                "moved_vertices": int(moved.sum()),
                "facial_first_hit": facial,
            }
        # closed/open topology identity
        topology_match = (
            disk["closed"]["vertices"] == disk["open"]["vertices"]
            and disk["closed"]["faces"] == disk["open"]["faces"])
        # baseline boundary comparison (mouth interior + neck bottom preserved)
        V_base, F_base = load_obj(BASELINE_DELIVERY / "head_closed_m.obj")
        base_rep = topology_report(V_base, F_base)
        checks = {
            "finite_coordinates": all(disk[s]["topology"]["finite"] for s in disk),
            "consistent_winding": all(disk[s]["topology"]["inconsistent_shared_edges"] == 0 for s in disk),
            "no_new_nonmanifold": all(disk[s]["topology"]["nonmanifold_edges"] == 0 for s in disk),
            "boundary_preserved": all(
                disk[s]["topology"]["boundary_edges"] == base_rep["boundary_edges"] for s in disk),
            "no_degenerate_faces": all(disk[s]["topology"]["zero_area_faces"] == 0 for s in disk),
            "local_self_intersections_none": all(
                verification[s]["intersections"]["count"] == 0 for s in verification),
            "protected_vertices_unchanged": all(
                verification[s]["protected_displacement_mm"] == 0 for s in verification),
            "closed_open_topology_match": topology_match,
            "disk_memory_consistent": all(
                disk[s]["max_abs_diff_m"] is not None and disk[s]["max_abs_diff_m"] <= 1e-9
                and disk[s]["faces_match"] for s in disk),
            "geometry_guards_pass": all(
                nra.appearance.accept_guards(verification[s]["guards"]) for s in verification),
            "no_backward_facing_moved": all(
                verification[s]["backward_facing_moved_vertices"] == 0 for s in verification),
            "facial_first_hit_unchanged": all(
                verification[s]["facial_first_hit"]["protected_face_hits"] > 0
                and verification[s]["facial_first_hit"]["max_abs_shift_mm"] is not None
                and verification[s]["facial_first_hit"]["max_abs_shift_mm"] <= .05
                for s in verification),
        }
        (out / "geometry_verification.json").write_text(json.dumps({
            "schema": "head_rgbd.neck_reference_auto.verification.v1",
            "checks": checks,
            "states": verification,
            "disk": disk,
            "old_delivery_orient_fix": old_fix,
            "baseline_topology": {k: base_rep[k] for k in ("boundary_edges", "components")},
            "intersection_scope": "local: affected edges vs nearby faces (AABB broad phase); "
                                  "not an exhaustive whole-head guarantee",
        }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        (out / "shape_metrics.json").write_text(json.dumps({
            "schema": "head_rgbd.neck_reference_auto.metrics.v1",
            "reference": {
                "source_path": asset["source_path"], "source_sha256": asset["source_sha256"],
                "source_axes": asset["source_axes"], "source_units": asset["source_units"],
                "core_peak_mm": asset["core_peak_mm"],
                "core_peak_xy_mm": asset["core_peak_xy_mm"],
                "reference_neck_width_mm": asset["reference_neck_width_mm"],
                "negative_extent_mm": asset["negative_extent_mm"],
                "background_sensitivity": asset["background_sensitivity"],
                "method": asset["method"], "excluded": asset["excluded"],
            },
            "transfer": {
                "scale": plan["scale"],
                "target_neck_width_mm": plan["target_neck_width_mm"],
                "strength": plan["strength"],
                "centre_base_vertex": plan["centre_vertex"],
                "centre_selection": "automatic: anterior midline ~30 mm below the closed chin, "
                                    "unprotected and front-facing in both states; not hardcoded",
            },
            "states": {s: plan["states"][s]["metrics"] for s in ("closed", "open")},
            "analysis": analysis,
            "analysis_note": "contours, components, profiles and apex curvature of the applied "
                             "appearance field in the shared (u,v) frame; outer regions included",
            "editing_mode": True,
            "geometry_source": "reference_driven_appearance",
            "subject_observation_matched": False,
        }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        timing.append({"stage": "verification", "seconds": round(time.perf_counter() - t0, 3)})

        # ---------- 6. four-way renders ----------
        t0 = time.perf_counter()
        centre_closed = plan["states"]["closed"]["frame"].origin
        for state in ("closed", "open"):
            frame = plan["states"][state]["frame"]
            centre = frame.origin
            fwd, right = frame.n, frame.u_axis
            shaped = plan["states"][state]["shaped_vertices"]
            old_V, old_F = load_obj(OLD_DELIVERY / f"head_{state}_m.obj")
            fix_V, fix_F = load_obj(orient_fix_dir / f"head_{state}_m.obj")
            for view_name, view_dir in (("front", -fwd), ("side", right),
                                        ("45deg", (right - fwd) / np.sqrt(2.0))):
                fig, axes = plt.subplots(1, 4, figsize=(21, 5.2), facecolor="#202020")
                ref_dir = np.array([1.0, 0, 0]) if view_name == "side" else (
                    np.array([0, 0, -1.0]) if view_name == "front" else np.array([0.7, 0, -0.7]))
                ref_centre = ref_V.mean(axis=0)
                for ax, (V, F, c, d, label) in zip(
                    axes,
                    ((ref_V, ref_F, ref_centre, ref_dir, "reference (rest frame)"),
                     (old_V, old_F, centre, view_dir, "old delivery (black faces)"),
                     (fix_V, fix_F, centre, view_dir, "old + winding fix"),
                     (shaped, faces, centre, view_dir, "reference auto sculpt"))):
                    ax.set_facecolor("#202020")
                    render_flat(ax, V, F, d, c, 55.0)
                    ax.set_title(label, color="w", fontsize=9)
                fig.suptitle(f"{state} / {view_name} (same scale, flat shading)", color="w")
                fig.tight_layout()
                fig.savefig(out / "render" / f"compare_{state}_{view_name}_flat.png", dpi=95, facecolor="#202020")
                plt.close(fig)
            # smooth-shaded + neck close for the new result vs winding fix
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#202020")
            for ax, (V, F, c, d, label) in zip(
                axes, ((fix_V, fix_F, centre, right, "old + winding fix"),
                       (shaped, faces, centre, right, "auto sculpt (smooth)"),
                       (shaped, faces, centre, right, "auto sculpt (flat)"))):
                ax.set_facecolor("#202020")
                (render_smooth if "smooth" in label else render_flat)(ax, V, F, d, c, 30.0)
                ax.set_title(label, color="w", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / "render" / f"compare_{state}_neck_close.png", dpi=110, facecolor="#202020")
            plt.close(fig)
            render_field(out, state, plan["states"][state], analysis[state], faces)
            # sections from the delivered disk OBJ
            v_disk, f_disk = load_obj(disk[state]["path"])
            base_v, base_f = load_obj(BASELINE_DELIVERY / f"head_{state}_m.obj")
            for sname, normal_axis, sample_axis, limit in (
                    ("long", frame.u_axis, frame.v_axis, 26.0),
                    ("cross", frame.v_axis, frame.u_axis, 24.0)):
                fig, ax = plt.subplots(figsize=(6.2, 5.0))
                for V, F, label, color in ((base_v, base_f, "baseline", "#8a97a8"),
                                           (v_disk, f_disk, "auto sculpt", "#e05a4e")):
                    x, y = plane_section(V, F, centre, normal_axis, sample_axis, frame.n,
                                         -limit, limit)
                    if sname == "cross":
                        # Raw intersection pairs are not an ordered path; keep the outer
                        # envelope per 1-mm lateral bin so the section is a single curve.
                        bins = np.round(x)
                        uniq = np.unique(bins)
                        x, y = uniq, np.array([y[bins == b].max() for b in uniq])
                    ax.plot(y, x, color=color, lw=1.8, label=label)
                ax.set_xlabel("outward n (mm)")
                ax.set_ylabel("v (mm)" if sname == "long" else "u (mm)")
                ax.set_title(f"{state}: {'longitudinal' if sname == 'long' else 'transverse'} section (disk OBJ)")
                ax.grid(alpha=0.3); ax.legend()
                fig.savefig(out / "render" / f"{state}_section_{sname}.png", dpi=110)
                plt.close(fig)
        timing.append({"stage": "renders", "seconds": round(time.perf_counter() - t0, 3)})

        # ---------- 7. manifest + report ----------
        (out / "final" / "manifest.json").write_text(json.dumps({
            "schema": "head_rgbd.delivery.neck_reference_auto.v1",
            "software": {"name": "facecap_airway_delivery", "version": "0.5.9-candidate"},
            "created_at": datetime.now().astimezone().isoformat(),
            "meshes": {"closed": "head_closed_m.obj", "open": "head_open_m.obj"},
            "mesh_unit": "m",
            "topology": "local red-green refined front-neck patch, shared closed/open",
            "editing_mode": True,
            "geometry_source": "reference_driven_appearance",
            "subject_observation_matched": False,
            "reference": {"path": str(args.reference), "sha256": asset["source_sha256"],
                          "used_for_transfer": True},
            "provenance": {"baseline_run": str(args.fit_dir)},
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        summary = {
            "status_pending": True,
            "chosen_checks": checks,
            "timing": timing,
            "asset_peak_mm": asset["core_peak_mm"],
            "scale": plan["scale"],
            "centre_vertex": plan["centre_vertex"],
            "reference": {
                "path": asset["source_path"], "sha256": asset["source_sha256"],
                "axes": asset["source_axes"], "units": asset["source_units"],
                "peak_xy_mm": asset["core_peak_xy_mm"],
                "negative_extent_mm": asset["negative_extent_mm"],
                "reference_neck_width_mm": asset["reference_neck_width_mm"],
                "background_sensitivity": asset["background_sensitivity"],
            },
            "analysis": analysis,
            "metrics": {s: plan["states"][s]["metrics"] for s in ("closed", "open")},
            "disk": disk,
            "old_fix": old_fix,
            "strength": args.strength,
            "total_seconds": round(time.perf_counter() - t_all, 3),
            "verification_passed": all(checks.values()),
        }
        from tools.make_reference_auto_report import write_report
        write_report(out, summary=summary, timing=timing)
        (PROJECT / "docs" / "NECK_REFERENCE_AUTO_SCULPT_REPORT.html").write_bytes(
            (out / "NECK_REFERENCE_AUTO_SCULPT_REPORT.html").read_bytes())
        status = "ok" if all(checks.values()) else "verification_failed"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        summary = {"status": status, "error": error, "exit_code": 0 if status == "ok" else 1,
                   "total_seconds": round(time.perf_counter() - t_all, 3), "output": str(out)}
        (out / "run_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
