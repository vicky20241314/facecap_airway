"""One low/mid-frequency displacement field, then one smooth output surface.

Exactly one call to :func:`head_rgbd.geometry_d.solve_field` and exactly one Loop
blend decision. No iteration between shape and surface, no per-iteration export and
no dense residual solve.

Two things are worth reading in the code because they are the contract:

``rows_field``
    The angle group replaces ``segment`` before the solve. ``burst_weights`` groups
    frames by ``segment``, so without this substitution the whole turn would be one
    view and the oblique angles would share a single vote. With ``view_balance`` the
    frontal view also stops receiving its historical 3x boost, so no label decides
    how much a surface region may move.

``loop blend``
    The blend starts at ``loop_target_blend`` (0.5) and is only reduced when the
    geometric safety check rejects it. It is never chosen by comparing a handful of
    depth samples, so a few noisy pixels cannot switch smoothing off.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import trimesh

from head_rgbd.audit import sha
from head_rgbd.fit_b import normals
from head_rgbd.fit_d import load_npz
from head_rgbd.geometry_c import ray_surface, sample_depth
from head_rgbd.geometry_d import Settings, anatomical_exclusion, edges_of, solve_field, subdivide, surface_state
from head_rgbd.reporting import write_json


def run(run_dir: Path, config, device: str = "cpu") -> dict:
    run_dir = Path(run_dir).resolve()
    field_dir = run_dir / "field"
    field_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    correction = run_dir / "pose_correction"
    report = json.loads((correction / "report.json").read_text(encoding="utf-8"))
    saved = load_npz(correction / "parameters.npz")
    sequence_path = run_dir / "prepared" / "sequence.json"
    if report.get("sequence_sha256") != sha(sequence_path):
        raise ValueError("the pose correction does not bind prepared/sequence.json")
    info = json.loads(sequence_path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in report["frames"]]
    for row in rows:
        row.setdefault("degraded", [])
    data = [load_npz(sequence_path.parent / next(item["file"] for item in info["frames"]
                                                 if item["id"] == row["id"])) for row in rows]

    base = np.asarray(saved["posed_vertices"], dtype=np.float64)
    faces = np.asarray(saved["faces"])
    vertex_count = int(base.shape[1])
    n = len(rows)
    if base.shape[0] != n:
        raise ValueError(f"pose correction has {base.shape[0]} frames but the report lists {n}")
    direction = np.array([normals(vertices, faces) for vertices in base])
    identity_ids = [int(i) for i in np.asarray(saved.get("identity_selected", np.arange(n)))]
    front = [i for i, row in enumerate(rows) if row.get("view") == "front"]
    mouth = np.asarray(saved.get("mouth_identity_excluded"))
    if mouth.shape != (vertex_count,):
        raise ValueError("mouth exclusion mask does not match the posed topology")
    excluded = anatomical_exclusion(base, data, front, mouth)

    rows_field = [dict(row, segment=str(row.get("angle_group") or "one_group")) for row in rows]
    settings = _settings(config.field)
    field, support, log = solve_field(base, faces, direction, data, rows_field, identity_ids,
                                      excluded, settings, stage="turn_low_mid_field")
    control_vertices = surface_state(base, direction, field)
    field_elapsed = time.perf_counter() - started
    print(json.dumps({"stage": "field", "free_frames": len(identity_ids),
                      "maximum_field_mm": float(np.abs(field).max()),
                      "elapsed_s": round(field_elapsed, 3)}), flush=True)

    dense_faces, mapping, midpoint_edges, parent_faces = subdivide(faces, vertex_count)
    dense_base = np.array([mapping @ vertices for vertices in base])
    dense_direction = np.array([mapping @ normal for normal in direction])
    dense_direction /= np.maximum(np.linalg.norm(dense_direction, axis=2, keepdims=True), 1e-12)
    dense_excluded = anatomical_exclusion(dense_base, data, front, (mapping @ excluded.astype(float)) > 0)
    interpolation = mapping @ field
    interpolation_vertices = np.array([mapping @ vertices for vertices in control_vertices])

    from head_rgbd.surface_v2 import loop_mapping, loop_support_exclusion, safe_loop_base

    stencil_excluded, minimum_loop_support = loop_support_exclusion(excluded, support, settings)
    _, _, stencil, _, _, loop_fixed = loop_mapping(faces, vertex_count, stencil_excluded)
    stencil = stencil.tolil()
    for vertex in np.flatnonzero(dense_excluded):
        row = mapping.getrow(vertex)
        stencil.rows[vertex] = row.indices.tolist()
        stencil.data[vertex] = row.data.tolist()
    stencil = stencil.tocsr()
    smooth_mapping, final_vertices, selection = safe_loop_base(
        mapping, stencil, control_vertices, dense_base, dense_direction, dense_faces,
        interpolation, settings)
    loop_elapsed = time.perf_counter() - started - field_elapsed

    reference_frame = _reference_frame(report, rows)
    reference_i = next(i for i, row in enumerate(rows) if row["id"] == reference_frame)
    np.savez_compressed(
        field_dir / "final_vertices.npz",
        vertices=final_vertices, faces=dense_faces, dense_base_vertices=dense_base,
        control_vertices=control_vertices, interpolation_vertices=interpolation_vertices,
        dense_field_mm=interpolation, anatomical_excluded=excluded,
        dense_anatomical_excluded=dense_excluded, support_frame_mask=support,
        reference_frame_id=np.asarray(reference_frame),
        prolongation_data=mapping.data, prolongation_indices=mapping.indices,
        prolongation_indptr=mapping.indptr, prolongation_shape=np.asarray(mapping.shape),
        smooth_data=smooth_mapping.data, smooth_indices=smooth_mapping.indices,
        smooth_indptr=smooth_mapping.indptr, smooth_shape=np.asarray(smooth_mapping.shape),
        midpoint_edges=midpoint_edges, parent_face_indices=parent_faces,
    )
    final_mesh = field_dir / "final_mesh.obj"
    trimesh.Trimesh(final_vertices[reference_i], dense_faces, process=False).export(
        final_mesh, include_normals=True)

    depth_consistency, geometry_change, smoothness = _metrics(
        final_vertices, dense_faces, data, rows, dense_base, interpolation_vertices,
        interpolation, config, reference_i)
    elapsed = time.perf_counter() - started

    payload = {
        "schema": "head_rgbd.turn.field.v1",
        "algorithm": "one low/mid-frequency shared displacement field plus one geometry-safe Loop blend",
        "research_only": True,
        "sequence": str(sequence_path),
        "sequence_sha256": sha(sequence_path),
        "pose_correction": {"path": str(correction),
                            "parameters_sha256": sha(correction / "parameters.npz"),
                            "motion_correction": report.get("motion_correction")},
        "solver_calls": 1,
        "field": {
            "base": "posed FLAME vertices after the single pose correction",
            "unknown": "one shared normal displacement in millimetres on the FLAME topology",
            "settings": log["settings"],
            "free_set_rule": log["free_set_rule"],
            "frames": len(identity_ids),
            "angle_groups": sorted({row["segment"] for row in rows_field}),
            "grouping_rule": ("rows[segment] was set to the angle group, so burst_weights gives every "
                              "angle the same total vote; view_balance removes the frontal boost"),
            "maximum_absolute_displacement_mm": log["maximum_absolute_displacement_mm"],
            "stop_reason": log["stop_reason"],
            "final_safety": log["final_safety"],
            "information_total": log["information_total"],
            "history": log["history"],
            "sampling": log["sampling"],
        },
        "subdivision": {
            "source": "deterministic linear midpoint subdivision, one level",
            "topology": {"vertices": int(dense_base.shape[1]), "faces": int(len(dense_faces))},
            "loop_blend": {
                "target": float(config.field.loop_target_blend),
                "blend": float(selection["blend"]),
                "alpha_safe": float(selection.get("alpha_safe", selection["blend"])),
                "policy": "target 0.5, reduced only by the geometric safety check",
                "fallback_reason": selection.get("fallback_reason"),
                "nonlinear_trials": selection.get("nonlinear_trials"),
                "minimum_support_frames": minimum_loop_support,
                "anatomical_fixed_controls": int(np.asarray(loop_fixed).sum()),
                "data_selection": None,
                "note": "no depth-sample comparison decides the blend, so a few noisy pixels cannot switch smoothing off",
            },
            "maximum_move_from_linear_mm": float(
                np.linalg.norm(final_vertices - interpolation_vertices, axis=2).max() * 1000.),
        },
        "reference_frame": reference_frame,
        "metrics": {
            "per_view_depth_consistency": depth_consistency,
            "side_geometry_change_from_joint": geometry_change,
            "surface_smoothness": smoothness,
            "observation": ("these numbers are measured against the same recorded depth that drove the fit; "
                            "they are a diagnostic, not independent accuracy"),
        },
        "frames": [{key: row.get(key) for key in
                    ("id", "view", "angle_group", "tracking_group", "tracking_quality", "frame_weight",
                     "fitted_yaw_deg", "depth_correspondences", "depth_residual_median_mm", "degraded")}
                   for row in rows],
        "elapsed_s": elapsed,
        "timing_sections": [{"section": "low_mid_field_s", "elapsed_s": round(field_elapsed, 3)},
                            {"section": "subdivide_and_loop_s", "elapsed_s": round(max(loop_elapsed, 0.), 3)},
                            {"section": "metrics_and_export_s",
                             "elapsed_s": round(max(elapsed - field_elapsed - loop_elapsed, 0.), 3)}],
        "statuses": {"implementation": "implemented", "geometry_quality": "not_evaluated",
                     "independent_accuracy": "unverified"},
        "limitations": [
            "the delivered surface is the geometry-safe Loop base of one control field; it is not a measured dense scan",
            "unobserved regions keep the parametric prior and are labelled by the support mask",
            "depth consistency is measured against the same capture that drove the fit",
            "no independent scan, manual contour or calibrated external pose is used",
        ],
    }
    write_json(field_dir / "report.json", payload)
    print(json.dumps({"output": str(field_dir), "elapsed_s": elapsed,
                      "vertices": int(dense_base.shape[1]), "faces": int(len(dense_faces)),
                      "loop_blend": selection["blend"]}), flush=True)
    return {"field": str(field_dir), "elapsed_s": elapsed, "vertices": int(dense_base.shape[1]),
            "faces": int(len(dense_faces)), "loop_blend": float(selection["blend"]),
            "maximum_field_mm": log["maximum_absolute_displacement_mm"],
            "reference_frame": reference_frame, "metrics": payload["metrics"]}


# ------------------------------------------------------------------------ detail
def _settings(section) -> Settings:
    return Settings(
        solver=section.solver, points=section.points_per_frame, outer_steps=section.outer_steps,
        gradient_length_mm=section.gradient_length_mm, bending_length_mm=section.bending_length_mm,
        low_bending_length_mm=section.low_bending_length_mm, noise_floor_mm=section.noise_floor_mm,
        maximum_slope=section.maximum_slope,
        maximum_normal_change_deg=section.maximum_normal_change_deg,
        displacement_bound_mm=section.displacement_bound_mm,
        minimum_signed_area_ratio=section.minimum_signed_area_ratio,
        research_v2=section.research_v2, fixed_ray_policy=section.fixed_ray_policy,
        support_min_information=section.support_min_information,
        coverage_allowance_ratio=section.coverage_allowance_ratio,
        information_measure=section.information_measure,
        information_damping_cap=section.information_damping_cap,
        view_balance=section.view_balance, loop_target_blend=section.loop_target_blend,
    )


def _reference_frame(report: dict, rows) -> str:
    reference = (report.get("references") or {}).get("closed", {})
    frame_id = reference.get("frame_id")
    if frame_id and any(row["id"] == frame_id for row in rows):
        return frame_id
    best = min(rows, key=lambda row: abs(float(row.get("fitted_yaw_deg") or 0.)))
    return best["id"]


def _metrics(final_vertices, dense_faces, data, rows, shared_vertices, control_vertices,
             interpolation, config, reference_i):
    rng = np.random.default_rng(int(config.experiment.seed) + 17)
    per_frame = []
    for i, row in enumerate(rows):
        points = sample_depth(data[i], config.field.probe_points_per_frame, rng, stride=3)[0]
        entry = {"frame_id": row["id"], "view": row.get("view"), "angle_group": row.get("angle_group"),
                 "yaw_deg": row.get("fitted_yaw_deg"), "probe_points": int(len(points))}
        # All three arrays are in the same frame, coordinate system and topology:
        # dense shared shape -> dense control-field surface -> dense Loop result.
        entry["geometry_change"] = _geometry_change(
            shared_vertices[i], control_vertices[i], final_vertices[i])
        # Backward-compatible names now use the true shared-shape baseline.
        entry.update(change_from_joint_median_mm=entry["geometry_change"]["shared_shape_to_final_total"]["median_mm"],
                     change_from_joint_p90_mm=entry["geometry_change"]["shared_shape_to_final_total"]["p90_mm"],
                     total_from_flame_median_mm=entry["geometry_change"]["shared_shape_to_final_total"]["median_mm"],
                     total_from_flame_max_mm=entry["geometry_change"]["shared_shape_to_final_total"]["maximum_mm"])
        if not len(points):
            entry.update(coverage=0.0, median_mm=None, p90_mm=None, note="no reliable depth to probe")
            per_frame.append(entry)
            continue
        correspondence, stats = ray_surface(final_vertices[i], dense_faces, points, 0.030, None)
        residual = np.full(len(points), np.nan)
        if correspondence is not None:
            residual[correspondence["ray"]] = np.abs(
                correspondence["hit"][:, 2] - correspondence["target"][:, 2]) * 1000.
        measured = residual[np.isfinite(residual)]
        entry.update(coverage=round(float(stats.get("accepted", 0)) / max(len(points), 1), 4),
                     median_mm=None if not len(measured) else round(float(np.median(measured)), 3),
                     p90_mm=None if not len(measured) else round(float(np.percentile(measured, 90)), 3),
                     missing_hits=int(sum(1 for value in residual if not np.isfinite(value))))
        per_frame.append(entry)

    values = [entry for entry in per_frame if entry.get("median_mm") is not None]
    summary = {
        "frames_probed": len(values),
        "median_mm_across_frames": None if not values else round(
            float(np.median([entry["median_mm"] for entry in values])), 3),
        "worst_median_mm": None if not values else max(entry["median_mm"] for entry in values),
        "mean_coverage": None if not per_frame else round(
            float(np.mean([entry["coverage"] for entry in per_frame])), 4),
        "by_view": _group_summary(per_frame, "view"),
        "by_angle_group": _group_summary(per_frame, "angle_group"),
        "scope": "nearest measured depth along each probe ray on the delivered dense surface",
    }
    sides = [entry for entry in per_frame if (entry.get("view") or "front") != "front"]
    geometry = {
        "reference_frame": rows[reference_i]["id"],
        "all_frames": _geometry_summary(per_frame, None),
        "side_frames": len(sides),
        "sides": _geometry_summary(sides, None),
        "by_view": _geometry_by_view(per_frame),
        "all_frames_median_mm": _nested_median(per_frame, "shared_shape_to_final_total", "median_mm"),
        "side_median_mm": _nested_median(sides, "shared_shape_to_final_total", "median_mm"),
        "maximum_mm": _nested_max(per_frame, "shared_shape_to_final_total", "p90_mm"),
        "scope": ("per-vertex displacement in metres converted to millimetres, comparing matched dense "
                  "vertices in each frame without a pose change"),
        "definitions": {
            "shared_shape_to_control": "field contribution relative to the shared joint-shape surface",
            "control_to_loop": "additional Loop-subdivision movement relative to the control-field surface",
            "shared_shape_to_final_total": "total geometric movement relative to the shared joint-shape surface",
            "accuracy_note": "geometric movement is not an accuracy metric",
        },
    }
    edges = edges_of(dense_faces)
    lengths = np.linalg.norm(
        (shared_vertices[reference_i][edges[:, 0]] - shared_vertices[reference_i][edges[:, 1]]) * 1000.,
        axis=1)
    slope = np.abs(interpolation[edges[:, 0]] - interpolation[edges[:, 1]]) / np.maximum(lengths, 0.05)
    smoothness = {
        "measure": "|d_a - d_b| / edge_length on the dense control field, dimensionless",
        "median": round(float(np.median(slope)), 4),
        "p90": round(float(np.percentile(slope, 90)), 4),
        "p99": round(float(np.percentile(slope, 99)), 4),
        "maximum": round(float(slope.max()), 4),
        "limit": float(config.field.maximum_slope),
        "edge_count": int(len(edges)),
    }
    return per_frame, geometry, smoothness


def _distance_stats(distance_mm) -> dict:
    values = np.asarray(distance_mm, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median_mm": None, "p90_mm": None, "maximum_mm": None}
    return {
        "median_mm": round(float(np.median(values)), 3),
        "p90_mm": round(float(np.percentile(values, 90)), 3),
        "maximum_mm": round(float(values.max()), 3),
    }


def _geometry_change(shared_vertices, control_vertices, final_vertices) -> dict:
    """Compare matched vertices in one coordinate system and identical topology."""
    shared = np.asarray(shared_vertices, dtype=np.float64)
    control = np.asarray(control_vertices, dtype=np.float64)
    final = np.asarray(final_vertices, dtype=np.float64)
    if not (shared.shape == control.shape == final.shape):
        raise ValueError("geometry comparisons require matching vertex arrays")
    return {
        "shared_shape_to_control": _distance_stats(
            np.linalg.norm(control - shared, axis=1) * 1000.),
        "control_to_loop": _distance_stats(
            np.linalg.norm(final - control, axis=1) * 1000.),
        "shared_shape_to_final_total": _distance_stats(
            np.linalg.norm(final - shared, axis=1) * 1000.),
    }


def _geometry_summary(entries, selector) -> dict:
    members = entries if selector is None else [entry for entry in entries if selector(entry)]
    result = {}
    for key in ("shared_shape_to_control", "control_to_loop", "shared_shape_to_final_total"):
        fields = {name: [entry["geometry_change"][key][name] for entry in members
                         if entry.get("geometry_change", {}).get(key, {}).get(name) is not None]
                  for name in ("median_mm", "p90_mm", "maximum_mm")}
        result[key] = {
            name: None if not fields[name] else {
                "median_of_frames_mm": round(float(np.median(fields[name])), 3),
                "maximum_of_frames_mm": round(float(max(fields[name])), 3),
            }
            for name in fields
        }
    return result


def _geometry_by_view(per_frame) -> dict:
    views = sorted({str(entry.get("view")) for entry in per_frame if entry.get("geometry_change")})
    return {view: _geometry_summary(per_frame, lambda entry, view=view: str(entry.get("view")) == view)
            for view in views}


def _nested_median(entries, component: str, field: str):
    values = [entry["geometry_change"][component][field] for entry in entries
              if entry.get("geometry_change", {}).get(component, {}).get(field) is not None]
    return None if not values else round(float(np.median(values)), 3)


def _nested_max(entries, component: str, field: str):
    values = [entry["geometry_change"][component][field] for entry in entries
              if entry.get("geometry_change", {}).get(component, {}).get(field) is not None]
    return None if not values else round(float(max(values)), 3)


def _group_summary(per_frame, key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for entry in per_frame:
        groups.setdefault(str(entry.get(key)), []).append(entry)
    summary = {}
    for label, entries in groups.items():
        measured = [entry for entry in entries if entry.get("median_mm") is not None]
        summary[label] = {
            "frames": len(entries),
            "median_mm": None if not measured else round(
                float(np.median([entry["median_mm"] for entry in measured])), 3),
            "coverage": round(float(np.mean([entry["coverage"] for entry in entries])), 4),
        }
    return summary
