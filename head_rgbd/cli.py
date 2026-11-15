"""Unified delivery entry point: one algorithm, one output contract.

Commands
--------
run         capture a guided session, then reconstruct and export one head mesh
capture     capture only (raw segments + session.json)
reconstruct rebuild an existing capture or prepared sequence, then export one mesh
device      open the camera and report RGB/depth alignment and intrinsics
assets      verify model files against the checked-in SHA256 manifest
--version   print the software version
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from head_rgbd import __version__
from head_rgbd import rebuild_timing as rebuild_timing
from head_rgbd.paths import (
    ensure_runtime_layout,
    resolve_runtime_dir,
    write_json_exclusive,
)
from head_rgbd.pipeline import (
    ALGORITHM,
    asset_gate,
    device_report,
    load_defaults,
    run_pipeline,
)
from head_rgbd.reporting import timestamp_id, write_json

DEFAULT_VIEWS = ["closed:front", "closed:left", "closed:right", "open:front"]


def _mesh_paths(delivery: Path) -> dict[str, str]:
    """Final meshes of one delivery folder, keyed by expression state."""
    defaults = load_defaults()
    return {
        state: str(delivery / name)
        for state, name in defaults["export"]["filenames"].items()
    }


def _add_capture_options(parser: argparse.ArgumentParser, defaults: dict) -> None:
    capture = defaults["capture"]
    parser.add_argument("--session-name", default="head", help="label stored in the capture folder name")
    parser.add_argument("--frames-per-view", type=int, default=capture["frames_per_view"])
    parser.add_argument("--interval-s", type=float, default=capture["interval_s"])
    parser.add_argument("--warmup-s", type=float, default=capture["warmup_s"])
    parser.add_argument("--countdown-s", type=int, default=capture["countdown_s"],
                        help="countdown for --auto only; manual click/SPACE/Enter starts immediately")
    parser.add_argument("--views", action="append", help="state:view, default closed:front closed:left closed:right open:front")
    parser.add_argument("--auto", action="store_true", help="start each pose after --countdown-s without click/SPACE/Enter")
    parser.add_argument("--no-preview", action="store_true", help="hide the OpenCV preview window")
    parser.add_argument(
        "--allow-marker-fallback", action="store_true",
        help="live: continue when a state's front ArUco marker is invalid; "
             "the capture is marked not_usable and no pair-able delivery is written",
    )


def _add_neck_prominence_options(parser: argparse.ArgumentParser, defaults: dict) -> None:
    """Marker-guided neck-front prominence; defaults come only from config."""
    prominence = defaults.get("neck_prominence") or {}
    parser.add_argument(
        "--neck-landmarks",
        type=Path,
        default=None,
        help="neck_landmarks.json sidecar (one laryngeal landmark record per state); "
             "falls back to <run>/neck_landmarks.json",
    )
    parser.add_argument(
        "--subject-sex",
        choices=("male", "female", "unspecified"),
        default=prominence.get("subject_sex", "unspecified"),
        help="user-declared subject sex for --neck-prominence auto; never inferred",
    )
    parser.add_argument(
        "--neck-prominence",
        choices=("auto", "on", "off"),
        default=prominence.get("mode", "auto"),
        help="auto: male + valid landmark enables shaping, female/unspecified keep the template neck",
    )
    parser.add_argument(
        "--neck-algorithm",
        choices=("reference_driven_neck_volume", "observation", "manual_edit",
                 "continuous_reference"),
        default=prominence.get("algorithm", "reference_driven_neck_volume"),
        help="reference_driven_neck_volume (0.9 default): one unified cross-section-driven "
             "neck volume and head/neck/shoulder transition; continuous_reference: historical "
             "two-feature non-negative appearance bump; manual_edit: older designed ridge; "
             "observation: subject-observed background+ridge fit",
    )
    parser.add_argument(
        "--neck-reference",
        default=prominence.get("reference_path"),
        help="reference OBJ for the reference-driven neck stage (shape only; default from config)",
    )
    parser.add_argument(
        "--neck-alpha",
        type=float,
        default=prominence.get("alpha", 0.55),
        help="smooth direction field blend: 1 = frame normal, 0 = smoothed surface normal",
    )
    parser.add_argument(
        "--neck-preset",
        choices=("light", "natural", "pronounced"),
        default=prominence.get("preset", "natural"),
        help="appearance preset: light 3 mm / natural 5 mm / pronounced 7 mm forward ridge",
    )
    parser.add_argument(
        "--neck-prominence-mm",
        type=float,
        default=None,
        help="explicit target FINAL prominence height in mm (default: candidate targets from config)",
    )
    parser.add_argument(
        "--neck-prominence-radius-x-mm",
        type=float,
        default=prominence.get("radius_x_mm", 11.0),
        help="central ridge lateral radius in mm",
    )
    parser.add_argument(
        "--neck-prominence-radius-y-mm",
        type=float,
        default=prominence.get("radius_y_mm", 12.0),
        help="central ridge vertical radius in mm",
    )
    parser.add_argument(
        "--neck-prominence-base-radius-x-mm",
        type=float,
        default=prominence.get("base_radius_x_mm", 20.0),
        help="wide base field lateral radius in mm",
    )
    parser.add_argument(
        "--neck-prominence-base-radius-y-mm",
        type=float,
        default=prominence.get("base_radius_y_mm", 16.0),
        help="wide base field vertical radius in mm",
    )
    parser.add_argument(
        "--neck-prominence-base-share",
        type=float,
        default=prominence.get("base_share", 0.35),
        help="share of the prominence height carried by the wide base field",
    )


def _add_neck_direct_options(parser: argparse.ArgumentParser, defaults: dict) -> None:
    """Neck direct-reshaping options shared by run / reconstruct / rebuild."""
    parser.add_argument(
        "--neck-landmarks", type=Path, default=None,
        help="neck_landmarks.json from neck_landmark_aruco; when absent the "
             "rebuild scans <output>/prepared front frames and the dense reference",
    )
    parser.add_argument(
        "--neck-variant", choices=("auto", "prior", "no-prior"), default="prior",
        help="default prior (C): anatomy-localized V larynx prior; "
             "auto: try prior then fall back to no-prior above the crease limit; "
             "no-prior: conservative direct solve",
    )
    parser.add_argument(
        "--crease-limit", type=float, default=20.0,
        help="full-edge / B positive crease maximum used to choose the neck variant",
    )
    parser.add_argument(
        "--larynx-prior-ratio", type=float, default=0.35,
        help="larynx prior amplitude = ratio x jaw reference",
    )
    parser.add_argument(
        "--neck-gain", type=float, default=None,
        help="structure gain on the reference relief (default: builder 1.10); "
             "does not change the marker-derived centre or amplitude semantics",
    )
    parser.add_argument(
        "--neck-bridge-width", type=float, default=None,
        help="submental bridge geodesic width in mm (default: builder 12.0)",
    )
    parser.add_argument(
        "--neck-seam-taper-ratio", type=float, default=None,
        help="data-weight fade width from the protected head, in local edges "
             "(default: builder 2.5)",
    )
    parser.add_argument(
        "--neck-mu-t-bridge", type=float, default=None,
        help="tangential prior in the submental bridge (default: builder 0.15)",
    )
    parser.add_argument(
        "--neck-mu-t-body", type=float, default=None,
        help="tangential prior in the neck body (default: builder 0.50)",
    )
    parser.add_argument(
        "--neck-anchor-near-a", type=float, default=None,
        help="extra hand-over stiffness near the fixed head "
             "(default: builder 2.0)",
    )
    parser.add_argument("--base-only", action="store_true",
                        help="stop after the base reconstruction (no neck reshape)")
    parser.add_argument("--full-verify", action="store_true",
                        help="build zero-topology/zero-new-vertex ablations and write "
                             "the complete S0-S9 verification.json")
    rebuild_defaults = defaults.get("rebuild") or {}
    parser.add_argument(
        "--minimal-output", action=argparse.BooleanOptionalAction,
        default=bool(rebuild_defaults.get("minimal_output", True)),
        help="fast path: no renders and no display-only duplicate meshes under "
             "result/ (neck submeshes, base-head copies); final/, measured/, "
             "quick_check.json and run_report.json are always written. "
             "--full-verify still builds its explicit ablations.")
    parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction,
        default=bool(rebuild_defaults.get("render", False)),
        help="write the standard neck and larynx renders (default off in the "
             "fast path; --no-render is the explicit alias)")


def _add_fit_options(parser: argparse.ArgumentParser, defaults: dict) -> None:
    reconstruction = defaults["reconstruction"]
    parser.add_argument("--device", choices=("cuda", "cpu"), default=reconstruction["device"])
    parser.add_argument("--points", type=int, default=reconstruction["points_per_frame"])
    parser.add_argument("--steps", type=int, default=reconstruction["steps"])
    parser.add_argument("--front-steps", type=int, default=reconstruction.get("front_steps", 240),
                        help="frontal initialization budget; --steps controls the joint fit")
    parser.add_argument(
        "--phase-cuts", default=reconstruction.get("phase_cuts"),
        help="comma-separated 4 interior fit_c phase fractions (default "
             "0.167,0.458,0.625,0.833); both fit_c stages use them")
    parser.add_argument(
        "--correspondence-interval", type=int,
        default=reconstruction.get("correspondence_interval", 10),
        help="steps between fit_c surface correspondence refreshes; "
             "phase boundaries refresh")
    parser.add_argument(
        "--motion-correspondence-interval", type=int,
        default=reconstruction.get("motion_correspondence_interval", 10),
        help="steps between motion_v2 surface correspondence refreshes; "
             "phase boundaries refresh")
    parser.add_argument(
        "--motion-steps", type=int,
        default=reconstruction.get("motion_steps", 80),
        help="motion_v2 correction budget (fit_c budgets stay --steps/--front-steps)")
    parser.add_argument(
        "--motion-expression-start", type=int, default=None,
        help="motion_v2 first expression/jaw step (default motion_steps//2)")
    parser.add_argument(
        "--motion-expression-end", type=int, default=None,
        help="motion_v2 exclusive expression/jaw end step (default 3*motion_steps//4)")
    parser.add_argument(
        "--loss-mode", choices=("legacy", "batch"),
        default=reconstruction.get("loss_mode", "legacy"),
        help="fit_c/motion_v2 loss evaluation: legacy per-frame loop or "
             "frame-dimension batch with identical weights")
    parser.add_argument("--motion-lowerface-weight", type=float,
                        default=reconstruction.get("motion_lowerface_weight", .5),
                        help="reliable open-front lower-face depth during expression correction; 0 disables it")
    parser.add_argument("--bending-length-mm", type=float,
                        default=reconstruction.get("bending_length_mm", 4.))
    parser.add_argument("--low-bending-length-mm", type=float,
                        default=reconstruction.get("low_bending_length_mm", 8.))
    parser.add_argument("--loop-target-blend", type=float,
                        default=reconstruction.get("loop_target_blend", .5))
    parser.add_argument(
        "--keyframes-per-segment",
        type=int,
        default=reconstruction["keyframes_per_segment"],
    )
    parser.add_argument(
        "--expression-components",
        type=int,
        default=reconstruction["expression_components"],
    )
    parser.add_argument(
        "--dense-points",
        type=int,
        default=reconstruction.get("dense_points", 5000),
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=reconstruction.get("fast", False),
        help="fast path: deliver the Loop base without a dense residual solve",
    )
    parser.add_argument(
        "--mid-rounds",
        type=int,
        default=reconstruction.get("mid_rounds", 2),
        help="mid-frequency correction rounds used by the fast path",
    )
    parser.add_argument(
        "--shape-fusion",
        choices=("guarded", "joint"),
        default=reconstruction.get("shape_fusion", "guarded"),
        help="joint: front and side views jointly determine one shared face shape; "
             "guarded: the reproducible front-anchored control",
    )
    parser.add_argument(
        "--joint-front-confidence",
        type=float,
        default=reconstruction.get("joint_front_confidence", 0.15),
        help="joint mode: data weight kept by a front-covered vertex instead of a hard lock",
    )
    parser.add_argument(
        "--chain",
        choices=("full", "lean"),
        default=reconstruction.get("chain", "full"),
        help="lean drops the standalone low-frequency stage and the second motion pass",
    )
    parser.add_argument(
        "--postfit",
        choices=("full", "minimal"),
        default=reconstruction.get("postfit", "full"),
        help="minimal skips fit_c post-fit work the joint fast chain never consumes",
    )
    parser.add_argument(
        "--reuse-anchor", action="store_true",
        help="full chain only: skip the second standalone low-frequency solve",
    )
    parser.add_argument(
        "--reuse-field", action=argparse.BooleanOptionalAction,
        default=reconstruction.get("reuse_field", False),
        help="reuse the low/mid field instead of a regularized re-solve",
    )
    parser.add_argument(
        "--identity-views",
        choices=("all", "front"),
        default=reconstruction.get("identity_views", "all"),
        help="which closed views update the shared identity; front is the side-data ablation",
    )
    parser.add_argument(
        "--support-min-information",
        type=float,
        default=reconstruction.get("support_min_information", 0.0),
        help="free set from accumulated barycentric information; 0 keeps seen-by-two-frames",
    )
    parser.add_argument(
        "--coverage-allowance-ratio",
        type=float,
        default=reconstruction.get("coverage_allowance_ratio", 0.0),
        help="allowed missing fixed-ray growth as a fraction of the fixed ray set, vs stage start",
    )
    parser.add_argument(
        "--view-balance",
        action="store_true",
        default=reconstruction.get("view_balance", False),
        help="equalise each view's total data weight so equation count cannot dominate",
    )
    parser.add_argument(
        "--information-measure",
        choices=("barycentric", "curvature"),
        default=reconstruction.get("information_measure", "barycentric"),
        help="how much a vertex constrains the solve: hit-weight count, or diag(A^T W A)",
    )
    parser.add_argument(
        "--information-damping-cap",
        type=float,
        default=reconstruction.get("information_damping_cap", 10.0),
        help="cap on the evidence-driven damping multiplier (curvature measure)",
    )
    parser.add_argument(
        "--anchor-prior-weight",
        type=float,
        default=reconstruction.get("anchor_prior_weight", 0.0),
        help="explicit pull of the deformation field toward the previous stage field, mm^-2",
    )
    parser.add_argument(
        "--loop-data-selection",
        action=argparse.BooleanOptionalAction,
        default=reconstruction.get("loop_data_selection", False),
        help="choose the Loop subdivision blend by data cost inside the geometric-safe range",
    )
    parser.add_argument(
        "--side-depth-weight",
        type=float,
        default=reconstruction.get("side_depth_weight", 1.0),
        help="joint mode: side-view depth term weight relative to a front view "
             "(set 0 for the side-ablation control)",
    )
    parser.add_argument(
        "--landmark-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="TUFA landmark network device; UniFace parsing always uses CPU",
    )


def command_capture(args: argparse.Namespace) -> int:
    from head_rgbd.capture_session import (
        capture_session, CaptureStalled, MarkerValidationError)

    try:
        folder = capture_session(
            runtime=args.runtime,
            session_name=args.session_name,
            views=args.views,
            frames_per_view=args.frames_per_view,
            interval_s=args.interval_s,
            warmup_s=args.warmup_s,
            countdown_s=args.countdown_s,
            preview=not args.no_preview,
            auto=args.auto,
            allow_marker_fallback=args.allow_marker_fallback,
        )
    except MarkerValidationError as exc:
        print(json.dumps({"error": "marker_validation_failed",
                          "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except CaptureStalled as exc:
        print(json.dumps({"error": "capture_stalled",
                          "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    marker = (json.loads((folder / "marker_validation.json").read_text(encoding="utf-8"))
              if (folder / "marker_validation.json").is_file() else {})
    print(json.dumps({"capture": str(folder), "manifest": str(folder / "session.json"),
                      "marker_status": marker.get("status"),
                      "marker_states": {k: v.get("status") for k, v in
                                        (marker.get("states") or {}).items()}},
                     ensure_ascii=False, indent=2))
    return 0


def _neck_after_base(args: argparse.Namespace, output, delivery):
    """Default: direct neck reshaping on the just-built dense fit."""
    if getattr(args, "base_only", False):
        args.neck_return_code = 0
        return None
    from head_rgbd.rebuild import run_rebuild

    output = Path(output)
    rc = run_rebuild(
        runtime=resolve_runtime_dir(args.runtime),
        output=output,
        dense=output / "reconstruction" / "dense",
        base_delivery=Path(delivery),
        args=args,
    )
    args.neck_return_code = int(rc)
    return str(output / "result") if rc == 0 else None


def command_run(args: argparse.Namespace) -> int:
    from head_rgbd.capture_session import (
        capture_session, CaptureStalled, MarkerValidationError)

    runtime = resolve_runtime_dir(args.runtime)
    layout = ensure_runtime_layout(runtime)
    try:
        capture = capture_session(
            runtime=runtime,
            session_name=args.session_name,
            views=args.views,
            frames_per_view=args.frames_per_view,
            interval_s=args.interval_s,
            warmup_s=args.warmup_s,
            countdown_s=args.countdown_s,
            preview=not args.no_preview,
            auto=args.auto,
            allow_marker_fallback=args.allow_marker_fallback,
        )
    except MarkerValidationError as exc:
        print(json.dumps({"error": "marker_validation_failed",
                          "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except CaptureStalled as exc:
        print(json.dumps({"error": "capture_stalled",
                          "reason": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    timer = rebuild_timing.RebuildTimer()   # capture has just returned
    capture_manifest = json.loads((capture / "session.json").read_text(encoding="utf-8")) \
        if (capture / "session.json").is_file() else {}
    marker_status = capture_manifest.get("marker_status")
    args.marker_status = marker_status
    args.capture = str(capture)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ensure_runtime_layout(runtime)["outputs"] / f"{capture.name}_reconstruction"
    )
    timing_log = layout["outputs"] / "rebuild_timings.jsonl"
    code = None
    delivery = None
    neck_result = None
    neck_code = 0
    error = None
    try:
        # The old appearance stage must not compete with the direct neck reshaping.
        args.neck_prominence = "off"
        code, delivery = run_pipeline(
            runtime=runtime,
            output=output,
            capture=capture,
            sequence=None,
            args=args,
            purpose="live_capture_to_offline_reconstruction",
        )
        _attach_capture(output / "run_report.json", capture, "passed")
        neck_result = _neck_after_base(args, output, delivery) if code == 0 else None
        neck_code = int(getattr(args, "neck_return_code", 0))
        elapsed = timer.stop()
        final_dir = str(Path(neck_result) / "final") if neck_result else None
        measured_dir = str(Path(neck_result) / "measured") if neck_result else None
        payload = {
            "meshes": _mesh_paths(delivery),
            "manifest": str(delivery / "manifest.json"),
            "marker_status": getattr(args, "marker_status", marker_status),
            "neck_result": neck_result,
            "final": final_dir, "measured": measured_dir,
            "final_sidecar": (str(Path(neck_result) / "final" / "anatomical_landmarks.json")
                              if neck_result else None),
            "measured_sidecar": (str(Path(neck_result) / "measured" / "anatomical_landmarks.json")
                                 if neck_result else None),
            "rebuild_elapsed_s": round(elapsed, 3),
            "rebuild_elapsed_hms": rebuild_timing.format_elapsed(elapsed),
            "timing_log": str(timing_log),
        }
        if neck_code != 0:
            payload["error"] = "neck_stage_failed"
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return neck_code
        print(json.dumps(payload, indent=2))
        return code
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status = ("exception" if error is not None
                  else "fail" if (code not in (None, 0) or neck_code not in (None, 0))
                  else "pass_base_only" if getattr(args, "base_only", False)
                  else "pass")
        record = rebuild_timing.build_record(
            timer=timer, mode="live_run", capture=str(capture),
            output=str(output), delivery=str(delivery) if delivery else None,
            neck_result=str(neck_result) if neck_result else None,
            marker_status=getattr(args, "marker_status", marker_status),
            pipeline_code=code, neck_code=neck_code, status=status,
            error=error, base_only=bool(getattr(args, "base_only", False)))
        try:
            written = rebuild_timing.write_timing(
                log_path=timing_log, output_dir=output, record=record)
            print(json.dumps({"rebuild_timing": written,
                              "rebuild_elapsed_s": record["rebuild_elapsed_s"],
                              "rebuild_elapsed_hms": record["rebuild_elapsed_hms"]},
                             ensure_ascii=False), flush=True)
        except Exception as timing_exc:
            print(f"rebuild timing write failed: {type(timing_exc).__name__}: {timing_exc}",
                  flush=True)


def command_reconstruct(args: argparse.Namespace) -> int:
    runtime = resolve_runtime_dir(args.runtime)
    layout = ensure_runtime_layout(runtime)
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        stem = Path(args.capture or args.sequence).resolve().name
        output = layout["outputs"] / f"{stem}_reconstruction"
    # Only a capture with session.json is a recorded session; a raw-acquisition
    # folder is imported inside the run and its provenance lands in the delivery.
    capture_dir = Path(args.capture).expanduser().resolve() if args.capture else None
    recorded = capture_dir is not None and (capture_dir / "session.json").is_file()
    # The default is base reconstruction + direct neck reshaping; the old
    # reference-driven neck volume is not part of this entry.
    args.neck_prominence = "off"
    code, delivery = run_pipeline(
        runtime=runtime,
        output=output,
        capture=capture_dir,
        capture_is_recorded_session=recorded,
        sequence=Path(args.sequence).expanduser().resolve() if args.sequence else None,
        args=args,
        purpose="offline_reconstruction",
    )
    if recorded:
        _attach_capture(output / "run_report.json", capture_dir, "passed")
    neck_result = _neck_after_base(args, output, delivery) if code == 0 else None
    if getattr(args, "neck_return_code", 0) != 0:
        print(json.dumps({"error": "neck_stage_failed"}, indent=2))
        return int(args.neck_return_code)
    final_dir = str(Path(neck_result) / "final") if neck_result else None
    measured_dir = str(Path(neck_result) / "measured") if neck_result else None
    print(json.dumps({"meshes": _mesh_paths(delivery),
                      "manifest": str(delivery / "manifest.json"),
                      "marker_status": getattr(args, "marker_status", None),
                      "neck_result": neck_result,
                      "final": final_dir, "measured": measured_dir},
                     indent=2))
    return code


def command_rebuild(args: argparse.Namespace) -> int:
    """One-click: locked head reconstruction + direct neck reshaping."""
    from head_rgbd.rebuild import run_base_and_rebuild, run_rebuild

    runtime = resolve_runtime_dir(args.runtime)
    layout = ensure_runtime_layout(runtime)
    if args.dense:
        dense = Path(args.dense).expanduser().resolve()
        if not (dense / "parameters.npz").exists():
            raise FileNotFoundError(f"dense reconstruction not found: {dense}")
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else dense.parents[1] / "rebuild"
        )
        base = dense.parents[1] / "delivery"
        base_delivery = base if (base / "manifest.json").is_file() else None
        rc = run_rebuild(runtime=runtime, output=output, dense=dense,
                         base_delivery=base_delivery, args=args)
        if rc == 0:
            print(json.dumps({"result": str(output / "result"),
                              "final": str(output / "result" / "final"),
                              "measured": str(output / "result" / "measured"),
                              "marker_status": getattr(args, "marker_status", None)},
                             indent=2))
        return rc
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        stem = Path(args.capture or args.sequence).resolve().name
        output = layout["outputs"] / f"{stem}_rebuild"
    capture_dir = Path(args.capture).expanduser().resolve() if args.capture else None
    recorded = capture_dir is not None and (capture_dir / "session.json").is_file()
    sequence = Path(args.sequence).expanduser().resolve() if args.sequence else None
    rc = run_base_and_rebuild(
        runtime=runtime,
        output=output,
        capture=capture_dir,
        sequence=sequence,
        args=args,
        capture_is_recorded_session=recorded,
    )
    if rc == 0:
        print(json.dumps({"result": str(output / "result"),
                          "final": str(output / "result" / "final"),
                          "measured": str(output / "result" / "measured"),
                          "marker_status": getattr(args, "marker_status", None)},
                         indent=2))
    return rc


def command_export(args: argparse.Namespace) -> int:
    """Re-export one mesh from a completed run (no refitting)."""
    from head_rgbd.pipeline import export_delivery

    runtime = resolve_runtime_dir(args.runtime)
    run = Path(args.run).expanduser().resolve()
    fit = run / "reconstruction" / "dense"
    sequence = run / "prepared" / "sequence.json"
    if not (fit / "report.json").is_file():
        raise FileNotFoundError(f"completed dense reconstruction not found under {run}")
    if not sequence.is_file():
        raise FileNotFoundError(f"prepared sequence not found: {sequence}")
    delivery = (
        Path(args.delivery_output).expanduser().resolve()
        if getattr(args, "delivery_output", None)
        else run / "delivery"
    )
    manifest = export_delivery(
        fit=fit,
        sequence=sequence,
        output=delivery,
        runtime=runtime,
        capture=None,
        runner=None,
        defaults=load_defaults(),
        args=args,
    )
    print(
        json.dumps(
            {
                "meshes": _mesh_paths(delivery),
                "vertices": {state: item["vertices"] for state, item in manifest["mesh"]["states"].items()},
                "faces": {state: item["faces"] for state, item in manifest["mesh"]["states"].items()},
                "status": manifest["status"],
            },
            indent=2,
        )
    )
    return 0


def command_device(args: argparse.Namespace) -> int:
    runtime = resolve_runtime_dir(args.runtime)
    layout = ensure_runtime_layout(runtime)
    result = device_report(runtime, args.samples)
    result["created_at"] = datetime.now().astimezone().isoformat()
    output = Path(args.output).expanduser().resolve() if args.output else (
        layout["logs"] / f"device_{timestamp_id()}.json"
    )
    write_json_exclusive(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    ok = bool(result.get("opened")) and bool(result.get("frames"))
    return 0 if ok else 2


def command_assets(args: argparse.Namespace) -> int:
    runtime = resolve_runtime_dir(args.runtime)
    layout = ensure_runtime_layout(runtime)
    try:
        result = asset_gate(runtime)
        status = "passed"
        code = 0
    except RuntimeError as exc:
        result = {"status": "failed", "error": str(exc)}
        status = "failed"
        code = 1
    payload = {"schema": "head_rgbd.delivery.assets.v1", "status": status, **result, "runtime": str(runtime)}
    output = Path(args.output).expanduser().resolve() if args.output else (
        layout["logs"] / f"assets_{timestamp_id()}.json"
    )
    write_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return code


def _attach_capture(report_path: Path, capture: Path, status: str) -> None:
    """Keep the raw-session audit trail next to the reconstruction report."""
    if not report_path.is_file():
        return
    from head_rgbd.audit import sha

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["capture"] = {
        "directory": str(capture),
        "manifest": str(capture / "session.json"),
        "manifest_sha256": sha(capture / "session.json"),
    }
    report.setdefault("statuses", {})["real_capture_pipeline"] = status
    write_json(report_path, report)


def build_parser() -> argparse.ArgumentParser:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(
        prog="headrgbd",
        description=(
            "Orbbec Gemini 215 RGB-D head reconstruction: guided capture, offline "
            "reconstruction and two metric meshes, closed and open (algorithm: " + ALGORITHM + ")."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples\n"
            "  headrgbd.cmd run\n"
            "  headrgbd.cmd capture --session-name subject1\n"
            "  headrgbd.cmd rebuild --capture runtime/captures/<session> --output runtime/outputs/<name>\n"
            "  headrgbd.cmd rebuild --dense runtime/outputs/<name>/reconstruction/dense\n"
            "  headrgbd.cmd reconstruct --capture runtime/captures/<session>   # default C direct; --base-only for base\n"
            "  headrgbd.cmd device\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"headrgbd {__version__} ({ALGORITHM})")
    parser.add_argument("--runtime", type=Path, help="runtime root (default: <project>/runtime)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="capture a guided session, reconstruct, direct-neck reshape and export",
    )
    _add_capture_options(run, defaults)
    _add_fit_options(run, defaults)
    _add_neck_direct_options(run, defaults)
    run.add_argument("--output", type=Path, help="run directory (default: runtime/outputs/<capture>_reconstruction)")
    run.set_defaults(func=command_run)

    capture = sub.add_parser("capture", help="capture raw RGB-D segments only")
    _add_capture_options(capture, defaults)
    capture.set_defaults(func=command_capture)

    reconstruct = sub.add_parser(
        "reconstruct",
        help="rebuild an existing capture/sequence, then direct-neck reshape "
             "(default C prior; --base-only for the base head)",
    )
    inputs = reconstruct.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--capture", type=Path, help="capture folder containing session.json")
    inputs.add_argument("--sequence", type=Path, help="prepared sequence.json")
    reconstruct.add_argument("--output", type=Path, help="run directory (never reused)")
    reconstruct.add_argument(
        "--allow-unverified-alignment",
        action="store_true",
        help="accept archives without a per-session calibration record and mark them unverified",
    )
    _add_fit_options(reconstruct, defaults)
    _add_neck_direct_options(reconstruct, defaults)
    reconstruct.set_defaults(func=command_reconstruct)

    rebuild = sub.add_parser(
        "rebuild",
        help="one-click: locked head reconstruction + direct neck reshaping (recommended)",
    )
    rebuild_inputs = rebuild.add_mutually_exclusive_group(required=True)
    rebuild_inputs.add_argument("--capture", type=Path,
                                 help="capture folder (session.json or raw acquisition)")
    rebuild_inputs.add_argument("--sequence", type=Path, help="prepared sequence.json")
    rebuild_inputs.add_argument(
        "--dense", type=Path,
        help="existing .../reconstruction/dense; skips the base fit and only rebuilds the neck",
    )
    rebuild.add_argument("--output", type=Path,
                         help="run directory (default: runtime/outputs/<stem>_rebuild)")
    rebuild.add_argument(
        "--allow-unverified-alignment",
        action="store_true",
        help="accept archives without a per-session calibration record and mark them unverified",
    )
    _add_fit_options(rebuild, defaults)
    _add_neck_direct_options(rebuild, defaults)
    rebuild.set_defaults(func=command_rebuild)

    export = sub.add_parser(
        "export",
        help="re-export closed + open meshes from a completed run (no refitting)",
    )
    export.add_argument("--run", type=Path, required=True, help="run directory holding reconstruction/dense")
    export.add_argument(
        "--delivery-output",
        type=Path,
        default=None,
        help="alternative delivery directory (default: <run>/delivery; never reused)",
    )
    _add_neck_prominence_options(export, defaults)
    export.set_defaults(func=command_export)

    device = sub.add_parser("device", help="open the camera and report alignment and intrinsics")
    device.add_argument("--samples", type=int, default=3)
    device.add_argument("--output", type=Path)
    device.set_defaults(func=command_device)

    assets = sub.add_parser("assets", help="verify model files against config/assets.json")
    assets.add_argument("--output", type=Path)
    assets.set_defaults(func=command_assets)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ('run', 'reconstruct', 'rebuild'):
        if min(args.steps, args.front_steps) < 40 or args.mid_rounds < 0:
            parser.error('fitting steps must be >= 40 and mid rounds must be nonnegative')
        if not 0 <= args.motion_lowerface_weight < float('inf'):
            parser.error('motion lowerface weight must be finite and nonnegative')
        if not 0 <= args.loop_target_blend <= 1 or min(args.bending_length_mm, args.low_bending_length_mm) <= 0:
            parser.error('Loop blend must be in [0, 1] and bending length must be positive')
    if getattr(args, "runtime", None) is not None:
        os.environ["WITHB_RUNTIME_ROOT"] = str(Path(args.runtime).expanduser().resolve())
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("ERROR: interrupted; the partial capture was saved as session.partial.json", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
