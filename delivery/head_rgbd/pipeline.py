"""The single delivery pipeline: real capture -> offline reconstruction -> one head mesh.

One algorithm, one configuration and one export contract. The default path is the
frozen ``candidate_frozen`` chain:

    fit_c --front-only            coarse frontal anchor
    fit_c --front-anchor          guarded shared identity (fixed ray policy: strict)
    motion_v2                     per-frame pose, expression and jaw before any depth fusion
    identity_v2                   low-frequency shared identity (forward protected)
    motion_v2 --identity-field    motion re-estimated with identity held fixed
    fit_d loop-research           measured Loop base + dense depth residual

``accounted-research`` and the B/C/D research entries are intentionally absent: this
module cannot switch policies at runtime. ``tests/test_contracts.py`` pins the
strict-coverage acceptance rule so the default cannot drift silently.
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
    "frozen candidate chain verified 2026-10-31 at "
    "runtime/zhou_cap_10310944_refinement_v2/candidate_frozen (source commit 9ef3837); "
    "strict-coverage fixed-ray policy, no accounted-research option"
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


def _fit_arguments(args) -> list[str | Path]:
    return [
        "--device",
        args.device,
        "--points",
        str(args.points),
        "--steps",
        str(args.steps),
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
    """Run the frozen chain. No mode switch, no alternative solver, no policy flag."""
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
                *common,
                "--front-only",
                "--expression-components",
                str(args.expression_components),
                "--runtime",
                runner.runtime,
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
                "--shape-fusion",
                joint_mode,
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
            ],
        ),
    )
    low_frequency = destination / "low_frequency"
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
            ],
        ),
    )
    alternated = destination / "alternated_motion"
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
                *(["--fast"] if getattr(args, "fast", False) else []),
                "--shape-fusion",
                joint_mode,
                *(["--joint-front-confidence", str(args.joint_front_confidence)] if joint else []),
                *(["--mid-rounds", str(args.mid_rounds)] if getattr(args, "fast", False) else []),
            ],
        ),
    )
    return final


def mesh_checks(path: Path) -> dict[str, Any]:
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


def export_delivery(
    *,
    fit: Path,
    sequence: Path,
    output: Path,
    runtime: Path,
    capture: Path | None,
    runner: StageRunner,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Write one closed and one open final mesh, one manifest and the sidecars.

    Exactly one mesh per expression state: ``head_closed_m.obj`` and
    ``head_open_m.obj``. Both come from the same dense stage, the same shared
    identity and the same topology; only the per-frame pose/expression/jaw of the
    reference frame differs. No coarse, anchor, smooth-base or comparison mesh is
    exported, and no debug switch exists in the user flow.
    """
    from head_rgbd import __version__
    import shutil

    output.mkdir(parents=True, exist_ok=True)
    fit_report = json.loads((fit / "report.json").read_text(encoding="utf-8"))
    references = json.loads((fit / "reference_geometry.json").read_text(encoding="utf-8"))
    sequence_info = inspect_sequence(sequence)

    stage = defaults["export"]["mesh_stage"]
    states = list(defaults["export"]["mesh_states"])
    if states != ["closed", "open"]:
        raise ValueError("delivery exports exactly the closed and open reference states")
    filenames = dict(defaults["export"]["filenames"])
    geometries: dict[str, dict[str, Any]] = {}
    exports: dict[str, dict[str, Any]] = {}
    for state in states:
        reference = references["references"][state]
        if stage not in reference.get("stages", {}):
            raise ValueError(f"reference stage {stage!r} missing from {fit} for state {state!r}")
        if not reference["frame_id"]:
            raise ValueError(f"state {state!r} has no real reference frame; nothing is invented")
        source_mesh = fit / reference["stages"][stage]["mesh"]
        final_mesh = output / filenames[state]
        if final_mesh.exists():
            raise FileExistsError(f"refusing to overwrite {final_mesh}")
        shutil.copy2(source_mesh, final_mesh)
        geometries[state] = mesh_checks(final_mesh)
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
    defaults_config = defaults
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
        "configuration": {
            "steps": defaults_config["reconstruction"]["steps"],
            "points_per_frame": defaults_config["reconstruction"]["points_per_frame"],
            "keyframes_per_segment": defaults_config["reconstruction"]["keyframes_per_segment"],
            "expression_components": defaults_config["reconstruction"]["expression_components"],
            "dense_points": defaults_config["reconstruction"]["dense_points"],
            "device": defaults_config["reconstruction"]["device"],
            "fixed_ray_policy": "strict-coverage",
        },
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
            "run_report": "run_report.json",
        },
        "neck": neck,
        "support_counts": dict(zip(SUPPORT_LABELS, counts)),
        "warnings": warnings,
        "limitations": fit_report.get("limitations", [])
        + [
            "geometry accuracy is not independently verified: no scan registration, no manual contour",
            "support labels separate real observations, correlated bursts plus prior and unobserved completion",
            "closed and open meshes are two states of one shared identity; they are not registered to each other by any external measurement",
            "this is a FaceCap-inspired offline FLAME prototype, not a FaceCap reproduction",
        ],
        "environment": {
            "runtime": str(runtime),
            "python": sys.version.split()[0],
            "assets_verified": True,
        },
    }
    write_json_exclusive(output / "manifest.json", manifest)
    return manifest


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
        )
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
