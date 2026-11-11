"""Shared public pipeline: prepare -> front initialization -> joint identity ->
one fixed-identity motion correction -> low/mid field -> safe Loop delivery.

Defaults live in config/defaults.json. The explicit full chain remains available
for reproducible controls; fixed-ray missing penalties and geometric budgets are
shared by both paths.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from head_rgbd.audit import sha
from head_rgbd.paths import (
    PROJECT_ROOT,
    asset_paths,
    ensure_runtime_layout,
    resolve_runtime_dir,
    verify_assets,
    write_json_exclusive,
)
from head_rgbd.reporting import (
    StageRunner,
    inspect_sequence,
    require_stage_success,
)

ALGORITHM = "facecap-inspired-v2-strict"
ALGORITHM_SOURCE = (
    "smooth-fast v0.5.2: joint shared identity, reliable open lower-face expression depth, "
    "batched fixed-ray motion acceptance, lean fast chain, barycentric support, "
    "evidence-consistent Loop stencils and geometry-checked target blend; "
    "validation: docs/LOSS_GUIDED_IMPROVEMENT_REPORT.html"
)
MESH_METERS_PER_UNIT = 1.0


def load_defaults() -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / "config" / "defaults.json").read_text(encoding="utf-8"))


def asset_gate(runtime: str | Path | None = None) -> dict[str, Any]:
    checks = verify_assets(asset_paths(runtime))
    bad = [row for row in checks if row["status"] != "ok"]
    if bad:
        details = ", ".join(f"{row['name']}={row['status']}" for row in bad)
        raise RuntimeError(f"runtime assets failed verification: {details}")
    return {"status": "verified", "count": len(checks), "assets": checks}


def _fit_arguments(args, *, steps: int | None = None) -> list[str | Path]:
    return [
        "--device",
        args.device,
        "--points",
        str(args.points),
        "--steps",
        str(args.steps if steps is None else steps),
        "--keyframes-per-segment",
        str(args.keyframes_per_segment),
    ]


def prepare_stage(
    runner: StageRunner,
    *,
    capture: Path | None,
    frame_manifest: Path | None,
    output: Path,
    landmark_device: str,
    allow_unverified_alignment: bool = False,
) -> Path:
    capture_path = Path(capture) if capture is not None else None
    manifest_path = frame_manifest
    if capture_path is not None and not (capture_path / "session.json").is_file():
        # A folder holding a raw-acquisition frames.json is imported first, in the
        # same run directory, so the source archive stays untouched and traceable.
        frames = capture_path / "frames.json"
        if not frames.is_file():
            raise ValueError(
                f"{capture_path} contains neither session.json nor frames.json; pass a capture folder, "
                "a raw-acquisition folder or --sequence"
            )
        raw_manifest = json.loads(frames.read_text(encoding="utf-8"))
        chunks = raw_manifest.get("chunks", [])
        if not chunks:
            raise ValueError(f"{frames} contains no chunks")
        imported = output.parent / "raw_input"
        code = runner.run(
            "import_raw_acquisition",
            "head_rgbd.prepare",
            ["--raw-acquisition", frames, "--output", imported],
        )
        require_stage_success(runner, code)
        frame_manifest = imported / "frames.json"
        allow_unverified_alignment = True
        capture_path = None
    source_args = ["--capture", capture_path] if capture_path is not None else ["--frame-manifest", frame_manifest]
    arguments: list[str | Path] = [
        *source_args,
        "--output",
        output,
        "--runtime",
        runner.runtime,
        "--landmark-device",
        landmark_device,
    ]
    if allow_unverified_alignment:
        arguments.append("--allow-unverified-alignment")
    code = runner.run("prepare_observations", "head_rgbd.prepare", arguments)
    require_stage_success(runner, code)
    return output / "sequence.json"


def reconstruct_locked(
    runner: StageRunner,
    *,
    sequence: Path,
    destination: Path,
    args,
) -> Path:
    """Construct shared stages once, with an independent frontal fit budget."""
    common = _fit_arguments(args)
    joint_mode = getattr(args, "shape_fusion", "guarded")
    joint = joint_mode == "joint"
    anchor = destination / "front_anchor"
    require_stage_success(
        runner,
        runner.run(
            "identity_front_anchor",
            "head_rgbd.fit_c",
            [
                "--sequence",
                sequence,
                "--output",
                anchor,
                *_fit_arguments(args, steps=getattr(args, 'front_steps', args.steps)),
                "--front-only",
                "--expression-components",
                str(args.expression_components),
                "--runtime",
                runner.runtime,
                "--postfit",
                getattr(args, "postfit", "full"),
            ],
        ),
    )
    guarded = destination / "guarded"
    require_stage_success(
        runner,
        runner.run(
            "shared_identity_joint" if joint else "shared_identity_guarded",
            "head_rgbd.fit_c",
            [
                "--sequence",
                sequence,
                "--output",
                guarded,
                *common,
                "--front-anchor",
                anchor,
                "--expression-components",
                str(args.expression_components),
                "--runtime",
                runner.runtime,
                "--postfit",
                getattr(args, "postfit", "full"),
                "--shape-fusion",
                joint_mode,
                *(["--identity-views", args.identity_views]),
                *(["--side-depth-weight", str(args.side_depth_weight)] if joint else []),
            ],
        ),
    )
    motion_only = destination / "motion_only"
    require_stage_success(
        runner,
        runner.run(
            "fixed_identity_motion",
            "head_rgbd.motion_v2",
            [
                "--sequence",
                sequence,
                "--runtime",
                runner.runtime,
                "--device",
                args.device,
                "--initial-fit",
                guarded,
                "--output",
                motion_only,
                "--lowerface-weight",
                str(getattr(args, 'motion_lowerface_weight', .5)),
            ],
        ),
    )
    lean_chain = getattr(args, "chain", "full")=="lean"
    low_frequency = destination / "low_frequency"
    alternated = destination / "alternated_motion"
    if lean_chain:
        # The joint fit ends with its own motion phase, and fit_d rebuilds the
        # low/mid field from the fitted pose. The standalone low-frequency field and
        # the second motion pass re-solve what those stages already settled.
        alternated = motion_only
    else:
        require_stage_success(
        runner,
        runner.run(
            "low_frequency_shared_identity",
            "head_rgbd.identity_v2",
            [
                "--sequence",
                sequence,
                "--initial-fit",
                motion_only,
                "--output",
                low_frequency,
                "--shape-fusion",
                joint_mode,
                *(["--joint-front-confidence", str(args.joint_front_confidence)] if joint else []),
                *(["--identity-views", args.identity_views]),
                *(["--reuse-anchor"] if getattr(args, "reuse_anchor", False) else []),
                *(["--support-min-information", str(args.support_min_information)]),
                *(["--coverage-allowance-ratio", str(args.coverage_allowance_ratio)]),
                *(["--anchor-prior-weight", str(args.anchor_prior_weight)]),
                *(["--information-measure", args.information_measure]),
                *(["--information-damping-cap", str(args.information_damping_cap)]),
                *(["--view-balance"] if getattr(args, "view_balance", False) else []),
            ],
        ),
        )
        require_stage_success(
        runner,
        runner.run(
            "motion_reestimate_after_identity",
            "head_rgbd.motion_v2",
            [
                "--sequence",
                sequence,
                "--runtime",
                runner.runtime,
                "--device",
                args.device,
                "--initial-fit",
                motion_only,
                "--identity-field",
                low_frequency / "identity_field.npz",
                "--output",
                alternated,
                "--lowerface-weight",
                str(getattr(args, 'motion_lowerface_weight', .5)),
            ],
        ),
    )
    final = destination / "dense"
    require_stage_success(
        runner,
        runner.run(
            "guarded_measured_dense_surface",
            "head_rgbd.fit_d",
            [
                "--sequence",
                sequence,
                "--runtime",
                runner.runtime,
                "--initial-fit",
                alternated,
                "--output",
                final,
                "--mode",
                "guarded",
                "--research-only",
                "--solver",
                "qp",
                "--dense-mode",
                "loop-research",
                "--research-v2",
                "--points",
                str(args.dense_points),
                '--bending-length-mm', str(getattr(args, 'bending_length_mm', 4.)),
                '--low-bending-length-mm', str(getattr(args, 'low_bending_length_mm', 8.)),
                '--loop-target-blend', str(getattr(args, 'loop_target_blend', .5)),
                *(["--fast"] if getattr(args, "fast", False) else []),
                "--shape-fusion",
                joint_mode,
                *(["--joint-front-confidence", str(args.joint_front_confidence)] if joint else []),
                *(["--mid-rounds", str(args.mid_rounds)] if getattr(args, "fast", False) else []),
                *(["--loop-data-selection"] if getattr(args, "loop_data_selection", False) else []),
                *(["--reuse-field"] if getattr(args, "reuse_field", False) else []),
                *(["--identity-views", args.identity_views]),
                *(["--support-min-information", str(args.support_min_information)]),
                *(["--coverage-allowance-ratio", str(args.coverage_allowance_ratio)]),
                *(["--anchor-prior-weight", str(args.anchor_prior_weight)]),
                *(["--information-measure", args.information_measure]),
                *(["--information-damping-cap", str(args.information_damping_cap)]),
                *(["--view-balance"] if getattr(args, "view_balance", False) else []),
            ],
        ),
    )
    return final


def mesh_checks(path: Path, reference: dict[str, Any] | None = None) -> dict[str, Any]:
    """Independent read-back of the exported mesh: finite, indexed, metre scale."""
    import trimesh

    mesh = trimesh.load(str(path), process=False, maintain_order=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if not np.isfinite(vertices).all():
        raise ValueError(f"mesh has non-finite vertices: {path}")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError(f"mesh has out-of-range face indices: {path}")
    extents = vertices.max(0) - vertices.min(0) if len(vertices) else np.zeros(3)
    extent = float(extents.max()) if len(vertices) else 0.0
    if not 0.10 < extent < 0.60:
        raise ValueError(
            f"mesh extent {extent:.4f} is not metre scale; the exported unit contract is metres"
        )
    if reference is not None:
        indices = np.asarray(reference['landmark_triangle_indices'], dtype=int)
        barycentric = np.asarray(reference['landmark_barycentric'], dtype=float)
        if np.any(indices < 0) or np.any(indices >= len(faces)):
            raise ValueError('reference landmarks use out-of-range face indices')
        measured = (vertices[faces[indices]] * barycentric[:, :, None]).sum(axis=1)
        np.testing.assert_allclose(measured, reference['landmarks_m'], atol=2e-7)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extents_m": [float(value) for value in extents],
        "finite": True,
        "indices_valid": True,
        "unit": "m",
    }


SUPPORT_LABELS = [
    "unobserved_completion",
    "single_or_correlated_burst_plus_prior",
    "distinct_closed_bursts_and_angle_supported",
]


def _observation_prominence(
    *, state, base_vertices, faces, landmark_record, transform, model_to_camera,
    landmarks_m, reference, config, prepared_dir,
):
    """Subject-observed V3 neck correction for the public export entry."""
    from head_rgbd import neck_subject_v3 as n3
    from head_rgbd import neck_prominence as npn
    from head_rgbd import neck_landmark as nl
    import numpy as np
    import time as _time

    started = _time.perf_counter()
    report = {
        "state": state,
        "neck_method": n3.NECK_METHOD_V3,
        "geometry_source": n3.GEOMETRY_SOURCE_V3,
        "editing_mode": False,
        "subject_sex": config.subject_sex,
        "mode": config.mode,
    }
    observed = landmark_record.get("status") == "observed" and landmark_record.get("point_observed_m")
    enabled, reason = npn.effective_enable(config, bool(observed))
    report["effective_enabled"] = bool(enabled)
    report["landmark_source"] = landmark_record.get("source")
    if not enabled:
        report.update(status="skipped", reason=reason,
                      elapsed_ms=(_time.perf_counter() - started) * 1000.0)
        return report
    if transform.get("status") != "observed":
        report.update(status="skipped", reason=transform.get("reason") or "transform_unavailable",
                      elapsed_ms=(_time.perf_counter() - started) * 1000.0)
        return report
    point = nl.apply_transform(landmark_record["point_observed_m"], transform)
    if point is None:
        report.update(status="skipped", reason="transform_application_failed",
                      elapsed_ms=(_time.perf_counter() - started) * 1000.0)
        return report
    frame_id = reference["frame_id"]
    annotation_frame = transform.get("annotation_frame")
    if annotation_frame not in (None, frame_id):
        report.update(status="skipped", reason="cross_frame_observation_unsupported",
                      elapsed_ms=_time.perf_counter() * 1000.0)
        return report
    segment = landmark_record.get("segment_file") or f"{frame_id}.npz"
    path = Path(prepared_dir) / segment
    if not path.is_file():
        report.update(status="skipped", reason="prepared_segment_missing",
                      elapsed_ms=(_time.perf_counter() - started) * 1000.0)
        return report
    data = np.load(path, allow_pickle=False)
    lm_cam = np.asarray(landmarks_m, dtype=float)
    head_up = lm_cam[npn.NOSE_BRIDGE_TOP] - lm_cam[npn.CHIN]
    head_up /= np.linalg.norm(head_up)
    vertices = np.asarray(base_vertices, dtype=float)
    try:
        obs = n3.collect_observations(
            state=state, frame_id=frame_id, depth=data["depth"], neck_mask=data["neck_mask"],
            intrinsics=data["intrinsics"],
            label_pixel=tuple(landmark_record.get("pixel_uv") or [0.0, 0.0]),
            label_point=np.asarray(point, dtype=float),
            head_up_camera=head_up, baseline_vertices=vertices, faces=faces,
        )
        M = np.asarray(model_to_camera, dtype=float)
        rot, trans = M[:3, :3], M[:3, 3]
        lm_model = (lm_cam - trans) @ rot
        axes = npn._direction_frame(lm_model)
        normals_model = npn.vertex_normals((vertices - trans) @ rot, faces)
        protection = npn._protected_mask((vertices - trans) @ rot, lm_model, normals_model, axes)
        normals_cam = normals_model @ rot.T
        fit = n3.fit_state(obs, vertices=vertices, faces=faces, frame=obs.frame,
                           protected=protection["mask"], vertex_normals=normals_cam)
        applied = n3.apply_field(
            vertices=vertices, faces=faces, frame=obs.frame, background=fit.background,
            theta=fit.theta, protected=protection["mask"], vertex_normals=normals_cam,
            residual_coefficients=fit.residual_coefficients,
        )
    except Exception as error:  # keep the baseline rather than block the export
        report.update(status="skipped", reason=f"observation_fit_failed:{type(error).__name__}",
                      elapsed_ms=(_time.perf_counter() - started) * 1000.0)
        return report
    disp = np.abs(applied["applied_mm"])
    report.update(
        status="applied",
        reason=None,
        moved_vertices=int((disp > 1e-6).sum()),
        max_displacement_mm=round(float(disp.max()), 3),
        ridge_parameters=fit.diagnostics.get("parameters"),
        background=fit.diagnostics.get("background"),
        depth_observation_rms_mm=fit.diagnostics.get("depth_rms_mm"),
        residual_max_mm=fit.diagnostics.get("residual_max_mm"),
        observation_supported=None,
        elapsed_ms=(_time.perf_counter() - started) * 1000.0,
    )
    report["shaped_vertices"] = applied["shaped_vertices"]
    return report


def export_delivery(
    *,
    fit: Path,
    sequence: Path,
    output: Path,
    runtime: Path,
    capture: Path | None,
    runner: StageRunner,
    defaults: dict[str, Any],
    args: Any | None = None,
) -> dict[str, Any]:
    """Write one closed and one open final mesh, one manifest and the sidecars.

    Exactly one mesh per expression state: ``head_closed_m.obj`` and
    ``head_open_m.obj``. Both come from the same dense stage, the same shared
    identity and the same topology; only the per-frame pose/expression/jaw of the
    reference frame differs. No coarse, anchor, smooth-base or comparison mesh is
    exported, and no debug switch exists in the user flow.

    After the baseline dense surface is read and before either mesh is written,
    the optional marker-guided neck prominence stage may displace a small
    connected patch of front-neck vertices around the valid laryngeal landmark.
    It is label-localised template shaping, never a measured neck
    reconstruction; with the stage off, absent or unsafe the baseline geometry is
    delivered unchanged.
    """
    from head_rgbd import __version__
    from head_rgbd import neck_landmark as nl
    from head_rgbd import neck_prominence as npn
    import shutil

    output.mkdir(parents=True, exist_ok=True)
    fit_report = json.loads((fit / "report.json").read_text(encoding="utf-8"))
    references = json.loads((fit / "reference_geometry.json").read_text(encoding="utf-8"))
    sequence_info = inspect_sequence(sequence)
    sequence_rows = json.loads(Path(sequence).read_text(encoding="utf-8")).get("frames", [])

    prominence_config = npn.resolve_configuration(defaults, args)
    landmarks_path = getattr(args, "neck_landmarks", None)
    if landmarks_path is None:
        fallback = output.parent / nl.LANDMARKS_FILE
        landmarks_path = fallback if fallback.is_file() else None
    landmarks_payload, landmarks_reason = nl.load_neck_landmarks(landmarks_path)

    stage = defaults["export"]["mesh_stage"]
    states = list(defaults["export"]["mesh_states"])
    if states != ["closed", "open"]:
        raise ValueError("delivery exports exactly the closed and open reference states")
    filenames = dict(defaults["export"]["filenames"])
    geometries: dict[str, dict[str, Any]] = {}
    exports: dict[str, dict[str, Any]] = {}
    prominence_reports: dict[str, dict[str, Any]] = {}
    saved_dense = None
    for state in states:
        reference = references["references"][state]
        if stage not in reference.get("stages", {}):
            raise ValueError(f"reference stage {stage!r} missing from {fit} for state {state!r}")
        if not reference["frame_id"]:
            raise ValueError(f"state {state!r} has no real reference frame; nothing is invented")
        stage_reference = reference['stages'][stage]
        final_mesh = output / filenames[state]
        if final_mesh.exists():
            raise FileExistsError(f"refusing to overwrite {final_mesh}")
        if stage_reference.get('storage') == 'parameters.npz':
            import trimesh

            if saved_dense is None:
                with np.load(fit / 'parameters.npz', allow_pickle=False) as stored:
                    saved_dense = stored['dense_vertices'], stored['dense_faces']
            frame = stage_reference['frame_index']
            if not isinstance(frame, int) or not 0 <= frame < len(saved_dense[0]):
                raise ValueError('invalid final mesh reference frame index')
            if fit_report['frames'][frame]['id'] != reference['frame_id']:
                raise ValueError('final mesh frame index does not match the declared reference frame')
            base_vertices = saved_dense[0][frame]

            # Marker-guided neck prominence: one application point, after the
            # baseline dense surface exists and before anything is written.
            landmark_record = (
                (landmarks_payload or {}).get("states", {}).get(state)
                or nl.unavailable_record(state, landmarks_reason or "no_neck_landmarks_file")
            )
            transform = nl.resolve_reference_transform(
                landmark_record, sequence_rows, reference["frame_id"]
            )
            algorithm = str(getattr(args, "neck_algorithm", None)
                            or (defaults.get("neck_prominence") or {}).get("algorithm", "observation"))
            if algorithm == "observation":
                prominence_report = _observation_prominence(
                    state=state,
                    base_vertices=base_vertices,
                    faces=saved_dense[1],
                    landmark_record=landmark_record,
                    transform=transform,
                    model_to_camera=reference["animated_model_to_reference_camera_4x4"],
                    landmarks_m=stage_reference["landmarks_m"],
                    reference=reference,
                    config=prominence_config,
                    prepared_dir=Path(sequence).parent,
                )
            else:
                prominence_report = npn.apply_prominence(
                    state=state,
                    base_vertices=base_vertices,
                    faces=saved_dense[1],
                    landmark_record=landmark_record,
                    transform=transform,
                    model_to_camera=reference["animated_model_to_reference_camera_4x4"],
                    landmarks_m=stage_reference["landmarks_m"],
                    config=prominence_config,
                    prepared_dir=Path(sequence).parent,
                )
                prominence_report["neck_method"] = "manual_edit_two_field_prominence"
                prominence_report["editing_mode"] = True
            shaped_vertices = prominence_report.pop("shaped_vertices", None)
            prominence_reports[state] = prominence_report
            export_vertices = (
                shaped_vertices if shaped_vertices is not None else base_vertices
            )
            trimesh.Trimesh(export_vertices, saved_dense[1], process=False).export(
                final_mesh, include_normals=True)
        else:
            # Historical runs already contain the final dense OBJ files.
            prominence_reports[state] = {
                "state": state,
                "status": npn.STATUS_SKIPPED,
                "reason": "legacy_mesh_storage_shaping_unsupported",
                "effective_enabled": False,
                "neck_method": npn.NECK_METHOD,
                "geometry_source": npn.GEOMETRY_SOURCE,
            }
            shutil.copy2(fit / stage_reference['mesh'], final_mesh)
        geometries[state] = mesh_checks(final_mesh, stage_reference)
        exports[state] = {"reference": reference, "mesh": final_mesh, "geometry": geometries[state]}
    vertices = {state: item["geometry"]["vertices"] for state, item in exports.items()}
    if len(set(vertices.values())) != 1:
        raise ValueError(f"closed and open meshes must share one dense topology: {vertices}")

    with np.load(fit / "support_dense.npz", allow_pickle=False) as stored:
        support = {key: stored[key] for key in stored.files}
    label = support["region_label"]
    vertex_count = geometries[states[0]]["vertices"]
    if len(label) != vertex_count:
        raise ValueError("support labels do not match the exported dense topology")
    counts = np.bincount(label, minlength=3).tolist()
    evidence = {
        "schema": "head_rgbd.delivery.support.v1",
        "labels": SUPPORT_LABELS,
        "counts": counts,
        "vertex_label": "np.int8 per vertex, same order as both exported meshes (shared topology)",
        "applies_to": [filenames[state] for state in states],
        "confidence": support["confidence"].tolist(),
        "confidence_is_calibrated": False,
        "observed_frames": support["observed_frames"].tolist(),
        "closed_burst_count": support["closed_burst_count"].tolist(),
        "observed_closed_yaw_span_deg": support["observed_closed_yaw_span_deg"].tolist(),
        "scope": (
            "heuristic real-observation support; model prior and interpolation are labelled, "
            "never counted as new real observations"
        ),
        "support_is_shared": (
            "support is accumulated over all agreed identity frames; it is not a per-state rescore, "
            "so sections observed only while open or only while closed are not separable here"
        ),
    }
    write_json_exclusive(output / "support.json", evidence)

    landmark_payload = {
        "schema": "head_rgbd.delivery.reference_landmarks.v2",
        "states": {},
        "landmark_indices": "FLAME dynamic contour + static 68, zero based",
        "observation_is_manual_truth": False,
        "transform_translation_unit": "m",
        "note": (
            "each state carries its own reference frame and rigid transform; the two transforms are "
            "not comparable, there is no cross-state registration here"
        ),
    }
    for state in states:
        reference = exports[state]["reference"]
        stage_item = reference["stages"][stage]
        landmark_payload["states"][state] = {
            "frame_id": reference["frame_id"],
            "source_frame": reference.get("source_frame"),
            "timestamp_s": reference.get("timestamp_s"),
            "coordinate_system": reference["coordinate_system"],
            "mesh_unit": reference["mesh_unit"],
            "mesh": exports[state]["mesh"].name,
            "landmark_count": len(stage_item["landmarks_m"]),
            "landmark_triangle_indices": stage_item["landmark_triangle_indices"],
            "landmark_barycentric": stage_item["landmark_barycentric"],
            "landmarks_m": stage_item["landmarks_m"],
            "observed_landmarks_m": reference["observed_landmarks_m"],
            "observed_valid": reference["observed_valid"],
            "observed_inner_lip_gap_mm": reference["observed_inner_lip_gap_mm"],
            "model_inner_lip_gap_mm": stage_item["inner_lip_gap_mm"],
            "animated_model_to_reference_camera_4x4": reference["animated_model_to_reference_camera_4x4"],
            "reference_camera_to_animated_model_4x4": reference["reference_camera_to_animated_model_4x4"],
            "transform_scope": reference["transform_scope"],
        }
    write_json_exclusive(output / "reference_landmarks.json", landmark_payload)

    neck = fit_report.get("neck", [])

    any_applied = any(
        report.get("status") == npn.STATUS_APPLIED for report in prominence_reports.values()
    )
    neck_prominence_payload = {
        "schema": "head_rgbd.delivery.neck_prominence.v2",
        "base_algorithm_version": "0.5.2",
        "neck_method": next(
            (r.get("neck_method") for r in prominence_reports.values() if r.get("neck_method")),
            npn.NECK_METHOD,
        ),
        "editing_mode": bool(any(r.get("editing_mode") for r in prominence_reports.values())),
        "neck_algorithm": str(getattr(args, "neck_algorithm", None)
                              or (defaults.get("neck_prominence") or {}).get("algorithm", "observation")),
        "geometry_source": next(
            (r.get("geometry_source") for r in prominence_reports.values() if r.get("geometry_source")),
            npn.GEOMETRY_SOURCE,
        ),
        "effective_enabled": bool(any_applied),
        "subject_sex": prominence_config.subject_sex,
        "mode": prominence_config.mode,
        "landmarks_file": str(landmarks_path) if landmarks_path else None,
        "landmarks_sha256": landmarks_payload.get("sha256") if landmarks_payload else None,
        "landmark_source": {
            state: report.get("landmark_source") for state, report in prominence_reports.items()
        },
        "requested_final_prominence_mm": prominence_config.amplitude_mm,
        "candidate_final_prominence_mm": [float(v) for v in prominence_config.candidate_amplitudes_mm],
        "requested_radius_x_mm": prominence_config.radius_x_mm,
        "requested_radius_y_mm": prominence_config.radius_y_mm,
        "requested_base_radius_x_mm": prominence_config.base_radius_x_mm,
        "requested_base_radius_y_mm": prominence_config.base_radius_y_mm,
        "requested_base_share": prominence_config.base_share,
        "states": {
            state: {
                key: report.get(key)
                for key in (
                    "status", "reason", "effective_enabled", "landmark_source",
                    "editing_mode", "neck_method", "geometry_source",
                    "moved_vertices", "max_displacement_mm", "ridge_parameters",
                    "background", "depth_observation_rms_mm", "residual_max_mm",
                    "requested_final_prominence_mm", "baseline_prominence_mm",
                    "final_prominence_mm", "target_error_mm",
                    "binding_forward_displacement_mm", "max_vertex_displacement_mm",
                    "calibrated_amplitude_mm", "candidate_amplitudes_mm",
                    "shape", "extent", "trend", "calibration", "centre_snap",
                    "depth_conditioning", "candidate_attempts",
                    "binding", "direction", "support", "protection", "safety",
                    "transform", "point_reference_camera_m", "elapsed_ms",
                )
            }
            for state, report in prominence_reports.items()
        },
        "statement": (
            "appearance shaping of the template neck only: the validated laryngeal landmark and the "
            "template neck ridge locate a depth-conditioned, target-height-calibrated two-field "
            "procedural displacement of front-neck template vertices; it is not a measured neck "
            "reconstruction, performs no per-point neck fitting and does not restore measured neck geometry"
        ),
    }
    write_json_exclusive(output / "neck_prominence.json", neck_prominence_payload)

    warnings = _quality_warnings(fit_report, sequence_info, support)
    warnings += _state_warnings(exports, stage)
    frames = [
        {key: row.get(key) for key in ("id", "segment", "timestamp_s", "view", "state", "role", "degraded")}
        for row in fit_report.get("frames", [])
    ]
    mesh_payload = {
        "unit": "m",
        "meters_per_unit": MESH_METERS_PER_UNIT,
        "stage": stage,
        "topology_shared": True,
        "mesh_files": [filenames[state] for state in states],
        "states": {},
    }
    for state in states:
        geometry = exports[state]["geometry"]
        reference = exports[state]["reference"]
        stage_item = reference["stages"][stage]
        mesh_payload["states"][state] = {
            "path": filenames[state],
            "reference_frame": reference["frame_id"],
            "coordinate_system": reference["coordinate_system"],
            "vertices": geometry["vertices"],
            "faces": geometry["faces"],
            "extents_m": geometry["extents_m"],
            "dense": geometry["vertices"] == fit_report["dense_topology"]["vertices"],
            "model_inner_lip_gap_mm": stage_item["inner_lip_gap_mm"],
            "observed_inner_lip_gap_mm": reference["observed_inner_lip_gap_mm"],
        }
    manifest = {
        "schema": "head_rgbd.delivery.manifest.v2",
        "software": {
            "name": "facecap_airway_delivery",
            "version": __version__,
            "algorithm": ALGORITHM,
            "algorithm_source": ALGORITHM_SOURCE,
            "full_facecap_reproduction": False,
        },
        "created_at": fit_report.get("created_at"),
        "status": "degraded" if warnings else "success",
        "mesh": mesh_payload,
        "configuration": actual_configuration(fit, runner),
        "input": {
            "capture": str(capture) if capture is not None else None,
            "capture_kind": (
                "recorded_session"
                if capture is not None and (capture / "session.json").is_file()
                else "imported_raw_acquisition"
                if capture is not None
                else None
            ),
            "capture_manifest": str(capture / "session.json") if capture is not None and (capture / "session.json").is_file() else None,
            "capture_manifest_sha256": sha(capture / "session.json") if capture is not None and (capture / "session.json").is_file() else None,
            "sequence": str(sequence),
            "sequence_sha256": sequence_info["sha256"],
            "frames": frames,
            "reference_frames": {state: exports[state]["reference"]["frame_id"] for state in states},
            "closed_reference_frame": exports["closed"]["reference"]["frame_id"],
            "open_reference_frame": exports["open"]["reference"]["frame_id"],
            "continuous_full_turn": False,
        },
        "sidecars": {
            "support": "support.json",
            "reference_landmarks": "reference_landmarks.json",
            "neck_prominence": "neck_prominence.json",
            "run_report": "run_report.json",
        },
        "neck": neck,
        "neck_prominence": neck_prominence_payload,
        "support_counts": dict(zip(SUPPORT_LABELS, counts)),
        "warnings": warnings,
        "limitations": fit_report.get("limitations", [])
        + [
            "geometry accuracy is not independently verified: no scan registration, no manual contour",
            "support labels separate real observations, correlated bursts plus prior and unobserved completion",
            "closed and open meshes are two states of one shared identity; they are not registered to each other by any external measurement",
            "this is a FaceCap-inspired offline FLAME prototype, not a FaceCap reproduction",
            "neck front geometry is template-based: any prominence is a depth-conditioned procedural appearance deformation with a calibrated target height, not a measured reconstruction",
        ],
        "environment": {
            "runtime": str(runtime),
            "python": sys.version.split()[0],
            "assets_verified": True,
        },
    }
    write_json_exclusive(output / "manifest.json", manifest)
    return manifest


def actual_configuration(fit: Path, runner=None) -> dict[str, Any]:
    """Report measured stage settings, including re-export of a historical run."""
    report = json.loads((fit / 'report.json').read_text(encoding='utf-8'))
    config = dict(runner.report.get('configuration', {})) if runner is not None else {}
    chain = fit.parent
    for folder, key in (('front_anchor', 'front_steps'), ('guarded', 'steps')):
        path = chain / folder / 'report.json'
        if path.exists():
            stage = json.loads(path.read_text(encoding='utf-8'))
            config[key] = stage['steps']
            if folder == 'guarded':
                config.update(device=stage['device'], points_per_frame=stage['points_per_frame'],
                              expression_components=stage['expression_components'],
                              postfit=stage.get('postfit', config.get('postfit', 'unknown_legacy')))
    logs = report.get('field_logs', {})
    low = logs.get('front_anchor_regularized', {})
    mid = low.get('mid_frequency', {})
    settings = mid.get('settings', low.get('settings', {}))
    fusion = report.get('shape_fusion', {})
    loop = report.get('loop_base') or {}
    motion_path=chain/'motion_only/report.json'
    motion=json.loads(motion_path.read_text(encoding='utf-8')).get('motion_correction',{}) if motion_path.exists() else {}
    output_policy = report.get('output_policy', {})
    config.update(shape_fusion=fusion.get('mode', 'guarded'),
                  fast=report.get('fast_path', {}).get('enabled', False),
                  chain='full' if (chain / 'low_frequency/report.json').exists() else 'lean',
                  mid_rounds=report.get('fast_path', {}).get('mid_frequency_rounds'),
                  reuse_field=logs.get('regularized', {}).get('reused', False),
                  dense_points=settings.get('points'), bending_length_mm=settings.get('bending_length_mm'),
                  low_bending_length_mm=low.get('settings', {}).get('bending_length_mm'),
                  information_measure=settings.get('information_measure', 'barycentric'),
                  support_min_information=settings.get('support_min_information', 0.),
                  coverage_allowance_ratio=settings.get('coverage_allowance_ratio', 0.),
                  loop_data_selection=loop.get('data_selection') is not None,
                  loop_target_blend=loop.get('target_blend'), loop_actual_blend=loop.get('blend'),
                  loop_fallback_reason=loop.get('fallback_reason'),
                  motion_steps=motion.get('steps') if motion_path.exists() else None,
                  motion_lowerface_weight=motion.get('lowerface_weight',0.) if motion_path.exists() else None,
                  motion_lowerface_frames=motion.get('lowerface_frames',[]),
                  motion_acceptance_strategy=motion.get('acceptance_strategy','per_frame_legacy') if motion_path.exists() else None,
                  motion_acceptance_forward_calls=motion.get('acceptance_forward_calls'),
                  mesh_output=output_policy.get('meshes', 'legacy_intermediate_meshes'),
                  parameter_archive=output_policy.get('parameter_archive', 'deflated_npz'),
                  fixed_ray_policy=settings.get('fixed_ray_policy', 'strict-coverage'))
    if config['chain'] == 'lean':
        config.pop('reuse_anchor', None)
    return config


def _state_warnings(exports: dict[str, dict[str, Any]], stage: str = "dense") -> list[str]:
    """Flag a state whose reference frame has no usable open-mouth evidence."""
    warnings: list[str] = []
    for state, item in exports.items():
        reference = item["reference"]
        observed = reference.get("observed_inner_lip_gap_mm")
        model = reference["stages"][stage]["inner_lip_gap_mm"]
        if observed is None:
            warnings.append(
                f"{state} reference {reference['frame_id']}: inner lip depth invalid; "
                f"gap is model prior ({model:.2f} mm), not a measured opening"
            )
        elif state == "open" and observed < 10.0:
            warnings.append(
                f"open reference {reference['frame_id']}: measured inner lip gap {observed:.2f} mm is small; "
                "the open state may be weakly observed"
            )
    return warnings


def _quality_warnings(fit_report: dict[str, Any], sequence_info: dict[str, Any], support: dict) -> list[str]:
    warnings: list[str] = []
    degraded: dict[str, list[str]] = {}
    for row in fit_report.get("frames", []):
        for item in row.get("degraded", []) or []:
            degraded.setdefault(str(item), []).append(str(row["id"]))
    for item, frames in degraded.items():
        warnings.append(f"{item} ({len(frames)} frame(s): {', '.join(frames)})")
    if not sequence_info["has_real_closed_oblique"]:
        warnings.append("no real closed oblique observation")
    if sequence_info["excluded_frames"]:
        warnings.append(f"{sequence_info['excluded_frames']} prepared frame(s) excluded")
    maximum = float(np.asarray(support["deviation_m"]).__abs__().max() * 1000) if "deviation_m" in support else 0.0
    if maximum > 3.00001:
        warnings.append(f"dense deviation {maximum:.3f} mm exceeds the 3 mm total budget")
    if fit_report.get("statuses", {}).get("geometry_quality") not in (None, "not_evaluated"):
        warnings.append(f"geometry quality status: {fit_report['statuses'].get('geometry_quality')}")
    return warnings


def run_pipeline(
    *,
    runtime: str | Path | None,
    output: Path,
    capture: Path | None,
    sequence: Path | None,
    args,
    purpose: str,
    capture_is_recorded_session: bool = True,
) -> tuple[int, Path]:
    """Prepare (optional), reconstruct with the locked chain, export one mesh."""
    defaults = load_defaults()
    runtime_root = resolve_runtime_dir(runtime)
    ensure_runtime_layout(runtime_root)
    asset_gate(runtime_root)
    runner = StageRunner(output, runtime=runtime_root, purpose=purpose)
    runner.report['source_hashes'] = {
        str(p.relative_to(PROJECT_ROOT)): sha(p)
        for p in [*Path(__file__).parent.glob('*.py'), PROJECT_ROOT / 'config/defaults.json', PROJECT_ROOT / 'VERSION']
    }
    runner.report['configuration'] = {
        key: getattr(args, key) for key in defaults['reconstruction']
        if hasattr(args, key) and key != 'mode'
    }
    runner.report['configuration']['postfit'] = getattr(args, 'postfit', 'full')
    runner.flush()
    try:
        prepared = None
        if capture is not None or sequence is None:
            prepared = prepare_stage(
                runner,
                capture=capture,
                frame_manifest=None,
                output=runner.output / "prepared",
                landmark_device=args.landmark_device,
                allow_unverified_alignment=bool(getattr(args, "allow_unverified_alignment", False)),
            )
        sequence_path = Path(prepared or sequence).resolve()
        sequence_info = inspect_sequence(sequence_path)
        if not sequence_info["has_real_closed_oblique"]:
            raise ValueError(
                "reconstruction requires a real closed left/right observation; "
                "no frontal frame is relabelled as oblique"
            )
        fit = reconstruct_locked(
            runner,
            sequence=sequence_path,
            destination=runner.output / "reconstruction",
            args=args,
        )
        delivery = runner.output / "delivery"
        manifest = export_delivery(
            fit=fit,
            sequence=sequence_path,
            output=delivery,
            runtime=runtime_root,
            # Inside the run, an imported raw acquisition is not a recorded session.
            capture=capture if capture_is_recorded_session else None,
            runner=runner,
            defaults=defaults,
            args=args,
        )
        runner.report['configuration'] = manifest['configuration']
        runner.report['neck_prominence'] = manifest['neck_prominence']
        code = runner.finish(
            statuses={
                "pipeline_execution": "passed",
                "mesh_export": "passed",
                "real_capture_pipeline": "passed" if capture is not None else "not_applicable",
                "geometry_quality": "not_evaluated",
                "independent_accuracy": "unverified",
            },
            summary={
                "algorithm": ALGORITHM,
                "meshes": {state: str(delivery / item["path"]) for state, item in manifest["mesh"]["states"].items()},
                "manifest": str(delivery / "manifest.json"),
                "vertices": {state: item["vertices"] for state, item in manifest["mesh"]["states"].items()},
                "faces": {state: item["faces"] for state, item in manifest["mesh"]["states"].items()},
                "status": manifest["status"],
            },
        )
        return code, delivery
    except BaseException as exc:
        runner.report["error"] = f"{type(exc).__name__}: {exc}"
        runner.finish()
        raise


def device_report(runtime: str | Path | None, samples: int = 3, device: str = "cpu") -> dict[str, Any]:
    """Open the camera and report RGB/depth streams, alignment and intrinsics."""
    from core.capture import probe_camera
    from head_rgbd.prepare import models_self_check

    root = resolve_runtime_dir(runtime)
    result = probe_camera(1000, samples)
    result["models"] = models_self_check(root, device=device)
    result["assets"] = verify_assets(asset_paths(root))
    return result
