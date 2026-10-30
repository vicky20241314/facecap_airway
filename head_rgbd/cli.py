"""Unified capture, preparation, reconstruction and verification entry point."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

from head_rgbd.audit import sha
from head_rgbd.paths import (
    PROJECT_ROOT,
    asset_paths,
    configure_environment,
    ensure_runtime_layout,
    resolve_runtime_dir,
    verify_assets,
    write_json_exclusive,
)
from head_rgbd.reporting import (
    StageRunner,
    collect_environment,
    inspect_sequence,
    require_stage_success,
    timestamp_id,
    write_json,
)


def load_defaults() -> dict[str, Any]:
    return json.loads(
        (PROJECT_ROOT / "config" / "defaults.json").read_text(encoding="utf-8")
    )


def _asset_gate(runtime: Path) -> dict[str, Any]:
    checks = verify_assets(asset_paths(runtime))
    bad = [row for row in checks if row["status"] != "ok"]
    if bad:
        details = ", ".join(f"{row['name']}={row['status']}" for row in bad)
        raise RuntimeError(f"runtime assets failed verification: {details}")
    return {
        "status": "verified",
        "count": len(checks),
        "assets": checks,
    }


def _attach_capture_report(
    report_path: Path,
    capture: Path,
    *,
    validation_status: str,
    reason: str | None = None,
) -> None:
    """Keep the raw-session audit trail even when offline validation fails."""
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = capture / "session.json"
    report["capture"] = {
        "directory": str(capture),
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
    }
    report.setdefault("statuses", {})["real_capture_pipeline"] = validation_status
    if validation_status != "passed":
        report["capture_validation"] = {
            "status": validation_status,
            "reason": reason or "raw capture was saved but offline validation did not complete",
            "manifest_schema": json.loads(
                manifest.read_text(encoding="utf-8")
            ).get("schema"),
        }
    write_json(report_path, report)


def _fit_arguments(args: argparse.Namespace) -> list[str | Path]:
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


def _add_fit_options(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    reconstruction = defaults["reconstruction"]
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=reconstruction["device"],
    )
    parser.add_argument("--points", type=int, default=reconstruction["points_per_frame"])
    parser.add_argument("--steps", type=int, default=reconstruction["steps"])
    parser.add_argument(
        "--keyframes-per-segment",
        type=int,
        default=reconstruction["keyframes_per_segment"],
    )


def _validate_identity_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.free_identity and not args.research_only:
        parser.error("--free-identity is research-only and requires --research-only")
    if args.research_only and not args.free_identity:
        parser.error("--research-only is only valid together with --free-identity")


def _validate_reconstruction_input(
    sequence_info: dict[str, Any], *, free_identity: bool
) -> None:
    if not free_identity and not sequence_info["has_real_closed_oblique"]:
        raise ValueError(
            "guarded reconstruction requires a real closed left/right observation"
        )


def _run_reconstruction(
    runner: StageRunner,
    *,
    sequence: Path,
    destination: Path,
    args: argparse.Namespace,
) -> Path:
    common = _fit_arguments(args)
    if args.free_identity:
        final = destination / "free_identity_research"
        code = runner.run(
            "fit_c_free_identity_research",
            "head_rgbd.fit_c",
            [
                "--sequence",
                sequence,
                "--output",
                final,
                *common,
                "--expression-components",
                "30",
                "--runtime",
                runner.runtime,
            ],
        )
        require_stage_success(runner, code)
        return final

    anchor = destination / "front_anchor"
    code = runner.run(
        "fit_c_front_anchor",
        "head_rgbd.fit_c",
        [
            "--sequence",
            sequence,
            "--output",
            anchor,
            *common,
            "--front-only",
            "--expression-components",
            "30",
            "--runtime",
            runner.runtime,
        ],
    )
    require_stage_success(runner, code)
    final = destination / "guarded"
    code = runner.run(
        "fit_c_guarded",
        "head_rgbd.fit_c",
        [
            "--sequence",
            sequence,
            "--output",
            final,
            *common,
            "--front-anchor",
            anchor,
            "--expression-components",
            "30",
            "--runtime",
            runner.runtime,
        ],
    )
    require_stage_success(runner, code)
    return final


def _run_verification(
    runner: StageRunner,
    *,
    fit: Path,
    sequence: Path,
    destination: Path,
) -> dict[str, str]:
    verification = destination / "verification.json"
    code = runner.run(
        "verify_geometry_contracts",
        "head_rgbd.verify_c",
        ["--fit", fit, "--sequence", sequence, "--output", verification],
    )
    require_stage_success(runner, code)
    reference = destination / "reference_geometry.json"
    code = runner.run(
        "reference_landmarks",
        "head_rgbd.reference_metrics",
        [
            "--fit",
            fit,
            "--sequence",
            sequence,
            "--output",
            reference,
            "--runtime",
            runner.runtime,
        ],
    )
    require_stage_success(runner, code)
    return {"verification": str(verification), "reference_geometry": str(reference)}


def _fit_summary(path: Path) -> dict[str, Any]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    frames = report.get("frames", [])
    identity = [row["id"] for row in frames if row.get("identity_keyframe")]
    return {
        "path": str(path),
        "algorithm": report.get("algorithm"),
        "status": report.get("status"),
        "device": report.get("device"),
        "elapsed_s": report.get("elapsed_s"),
        "fitting_s": report.get("fitting_s"),
        "cuda_peak_allocated_mib": report.get("cuda_peak_allocated_mib"),
        "cuda_peak_reserved_mib": report.get("cuda_peak_reserved_mib"),
        "identity_keyframes": identity,
        "identity_detail_frames": report.get("identity_detail_frames", []),
        "motion_only_frames": [
            row["id"] for row in frames if not row.get("identity_keyframe")
        ],
        "excluded_frames": [
            {"id": row["id"], "reason": row.get("reason")}
            for row in report.get("prepared_excluded_frames", [])
        ],
        "degraded_frames": [
            {"id": row["id"], "degraded": row.get("degraded", [])}
            for row in frames
            if row.get("degraded")
        ],
        "fitted_yaw_deg": [
            {"id": row["id"], "yaw_deg": row.get("fitted_yaw_deg")}
            for row in frames
            if row.get("fitted_yaw_deg") is not None
        ],
        "acceptance": report.get("acceptance", {}),
        "limitations": report.get("limitations", []),
    }


def _prepare_stage(
    runner: StageRunner,
    *,
    capture: Path | None,
    frame_manifest: Path | None,
    output: Path,
    landmark_device: str,
) -> Path:
    source_args = (
        ["--capture", capture]
        if capture is not None
        else ["--frame-manifest", frame_manifest]
    )
    code = runner.run(
        "prepare_observations",
        "head_rgbd.prepare",
        [
            *source_args,
            "--output",
            output,
            "--runtime",
            runner.runtime,
            "--landmark-device",
            landmark_device,
        ],
    )
    require_stage_success(runner, code)
    return output / "sequence.json"


def command_status(args: argparse.Namespace) -> int:
    paths = configure_environment(args.runtime)
    layout = ensure_runtime_layout(paths.runtime)
    assets = _asset_gate(paths.runtime)
    result = {
        "schema": "head_rgbd.status.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "runtime_layout": {name: str(path) for name, path in layout.items()},
        "assets": assets,
        "environment": collect_environment(paths.runtime),
        "statuses": {
            "migration_implementation": "implemented",
            "real_capture_pipeline": "unverified",
            "geometry_effect_acceptance": "not_passed",
            "independent_accuracy": "unverified",
        },
    }
    output = args.output or layout["logs"] / f"status_{timestamp_id()}.json"
    output = Path(output).expanduser().resolve()
    write_json_exclusive(output, result)
    print(json.dumps({"report": str(output), "assets": "verified"}, indent=2))
    return 0


def command_assets(args: argparse.Namespace) -> int:
    runtime = resolve_runtime_dir(args.runtime)
    result = {
        "schema": "head_rgbd.asset_verification.v1",
        "runtime": str(runtime),
        "created_at": datetime.now().astimezone().isoformat(),
        "assets": verify_assets(asset_paths(runtime)),
    }
    result["status"] = (
        "passed"
        if all(row["status"] == "ok" for row in result["assets"])
        else "failed"
    )
    output = args.output
    if output is None:
        layout = ensure_runtime_layout(runtime)
        output = layout["logs"] / f"assets_{timestamp_id()}.json"
    output = Path(output).expanduser().resolve()
    write_json_exclusive(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


def command_probe(args: argparse.Namespace) -> int:
    from core.capture import probe_camera

    runtime = resolve_runtime_dir(args.runtime)
    layout = ensure_runtime_layout(runtime)
    result = probe_camera(args.timeout_ms, args.samples)
    result["created_at"] = datetime.now().astimezone().isoformat()
    output = args.output or layout["logs"] / "probes" / f"probe_{timestamp_id()}.json"
    output = Path(output).expanduser().resolve()
    write_json_exclusive(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("opened") and result.get("frames") else 2


def command_capture(args: argparse.Namespace) -> int:
    from head_rgbd.capture_session import capture_session

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
    )
    print(json.dumps({"capture": str(folder), "manifest": str(folder / "session.json")}))
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    _asset_gate(resolve_runtime_dir(args.runtime))
    runner = StageRunner(args.output, runtime=args.runtime, purpose="prepare")
    try:
        sequence = _prepare_stage(
            runner,
            capture=args.capture,
            frame_manifest=args.frame_manifest,
            output=runner.output / "prepared",
            landmark_device=args.landmark_device,
        )
        summary = inspect_sequence(sequence)
        code = runner.finish(summary={"sequence": summary})
    except Exception as exc:
        runner.report["error"] = f"{type(exc).__name__}: {exc}"
        code = runner.finish()
        raise
    print(json.dumps({"run_report": str(runner.report_path), "sequence": str(sequence)}))
    return code


def command_reconstruct(args: argparse.Namespace) -> int:
    _asset_gate(resolve_runtime_dir(args.runtime))
    sequence_info = inspect_sequence(args.sequence)
    _validate_reconstruction_input(sequence_info, free_identity=args.free_identity)
    runner = StageRunner(args.output, runtime=args.runtime, purpose="reconstruct")
    try:
        final = _run_reconstruction(
            runner,
            sequence=Path(args.sequence).resolve(),
            destination=runner.output,
            args=args,
        )
        summary = {
            "sequence": sequence_info,
            "reconstruction": _fit_summary(final),
            "identity_mode": "free_research" if args.free_identity else "guarded_default",
        }
        code = runner.finish(
            statuses={
                "geometry_effect_acceptance": "not_evaluated",
                "independent_accuracy": "unverified",
            },
            summary=summary,
        )
    except Exception as exc:
        runner.report["error"] = f"{type(exc).__name__}: {exc}"
        code = runner.finish()
        raise
    print(json.dumps({"run_report": str(runner.report_path), "fit": str(final)}))
    return code


def command_verify(args: argparse.Namespace) -> int:
    runner = StageRunner(args.output, runtime=args.runtime, purpose="verify")
    try:
        outputs = _run_verification(
            runner,
            fit=Path(args.fit).resolve(),
            sequence=Path(args.sequence).resolve(),
            destination=runner.output,
        )
        code = runner.finish(
            statuses={
                "geometry_effect_acceptance": "not_evaluated",
                "independent_accuracy": "unverified",
            },
            summary=outputs,
        )
    except Exception as exc:
        runner.report["error"] = f"{type(exc).__name__}: {exc}"
        code = runner.finish()
        raise
    print(json.dumps({"run_report": str(runner.report_path), **outputs}))
    return code


def _run_offline_pipeline(
    *,
    output: Path,
    runtime: Path,
    sequence: Path,
    args: argparse.Namespace,
    capture_source: Path | None = None,
    purpose: str,
    real_capture: bool = False,
) -> int:
    runner = StageRunner(output, runtime=runtime, purpose=purpose)
    try:
        if capture_source is not None:
            sequence = _prepare_stage(
                runner,
                capture=capture_source,
                frame_manifest=None,
                output=runner.output / "prepared",
                landmark_device=args.landmark_device,
            )
        sequence_info = inspect_sequence(sequence)
        _validate_reconstruction_input(sequence_info, free_identity=args.free_identity)
        final = _run_reconstruction(
            runner,
            sequence=sequence,
            destination=runner.output / "reconstruction",
            args=args,
        )
        verification = _run_verification(
            runner,
            fit=final,
            sequence=sequence,
            destination=runner.output / "validation",
        )
        summary = {
            "sequence": sequence_info,
            "reconstruction": _fit_summary(final),
            "verification": verification,
            "identity_mode": "free_research" if args.free_identity else "guarded_default",
        }
        code = runner.finish(
            statuses={
                "real_capture_pipeline": "passed" if real_capture else "not_applicable",
                "geometry_effect_acceptance": "not_evaluated",
                "independent_accuracy": "unverified",
            },
            summary=summary,
        )
    except Exception as exc:
        runner.report["error"] = f"{type(exc).__name__}: {exc}"
        code = runner.finish()
        raise
    print(json.dumps({"run_report": str(runner.report_path), "fit": str(final)}))
    return code


def command_run(args: argparse.Namespace) -> int:
    _asset_gate(resolve_runtime_dir(args.runtime))
    sequence = Path(args.sequence).resolve() if args.sequence else None
    capture = Path(args.capture).resolve() if args.capture else None
    output = Path(args.output).resolve()
    try:
        code = _run_offline_pipeline(
            output=output,
            runtime=resolve_runtime_dir(args.runtime),
            sequence=sequence,
            args=args,
            capture_source=capture,
            purpose=(
                "capture_to_offline_reconstruction"
                if capture
                else "offline_reconstruction"
            ),
            real_capture=capture is not None,
        )
    except BaseException as exc:
        if capture is not None:
            _attach_capture_report(
                output / "run_report.json",
                capture,
                validation_status="failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
        raise
    if capture is not None:
        _attach_capture_report(
            output / "run_report.json",
            capture,
            validation_status="passed",
        )
    return code


def command_capture_run(args: argparse.Namespace) -> int:
    from head_rgbd.capture_session import capture_session

    runtime = resolve_runtime_dir(args.runtime)
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
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ensure_runtime_layout(runtime)["outputs"]
        / f"{capture.name}_reconstruction"
    )
    report_path = output / "run_report.json"
    try:
        code = _run_offline_pipeline(
            output=output,
            runtime=runtime,
            sequence=None,
            args=args,
            capture_source=capture,
            purpose="live_capture_to_offline_reconstruction",
            real_capture=True,
        )
    except BaseException as exc:
        _attach_capture_report(
            report_path,
            capture,
            validation_status="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise
    _attach_capture_report(report_path, capture, validation_status="passed")
    print(json.dumps({"capture": str(capture), "run_report": str(report_path)}))
    return code


def _geometry_acceptance(comparison: dict[str, Any]) -> dict[str, Any]:
    frames = comparison["frames"]
    side_failures = []
    for row in frames:
        if row.get("view") == "front":
            continue
        b = row["methods"].get("B", {}).get("full", {})
        c = row["methods"].get("C", {}).get("full", {})
        if (
            b.get("p95_mm") is None
            or c.get("p95_mm") is None
            or c["p95_mm"] > b["p95_mm"]
            or c.get("median_mm", float("inf")) > b.get("median_mm", float("-inf"))
        ):
            side_failures.append(row["frame_id"])
    open_front_p95_failures = []
    front_regressions = []
    for row in frames:
        c_full = row["methods"].get("C", {}).get("full", {})
        c_front = row["methods"].get("C_front", {}).get("full", {})
        if row.get("state") == "open" and c_full.get("p95_mm") is not None:
            if c_front.get("p95_mm") is None or c_full["p95_mm"] > c_front["p95_mm"]:
                open_front_p95_failures.append(row["frame_id"])
        if row.get("view") == "front":
            change = row.get("front_regression", {}).get("C", {}).get("full", {}).get(
                "p95_mm"
            )
            if change is not None and change > 0:
                front_regressions.append({"frame_id": row["frame_id"], "p95_change_mm": change})
    passed = not side_failures and not open_front_p95_failures and not front_regressions
    return {
        "status": "passed" if passed else "failed",
        "side_p95_or_median_not_better_than_B": side_failures,
        "open_front_p95_regressions": open_front_p95_failures,
        "guarded_front_p95_regressions": front_regressions,
        "scope": "training-data camera-axis diagnostics, not independent accuracy",
    }


def command_regression(args: argparse.Namespace) -> int:
    _asset_gate(resolve_runtime_dir(args.runtime))
    sequence = Path(args.sequence).resolve()
    original = Path(args.original_comparison).resolve()
    sequence_info = inspect_sequence(sequence)
    if not sequence_info["has_real_closed_oblique"]:
        raise ValueError("regression requires real closed oblique frames")
    runner = StageRunner(args.output, runtime=args.runtime, purpose="real_data_regression")
    try:
        b = runner.output / "b"
        b_front = runner.output / "b_front"
        c_front = runner.output / "c_front"
        c_guarded = runner.output / "c_guarded"
        common = _fit_arguments(args)
        stages = [
            (
                "fit_b_all",
                "head_rgbd.fit_b",
                ["--sequence", sequence, "--output", b, *common, "--runtime", runner.runtime],
            ),
            (
                "fit_b_front",
                "head_rgbd.fit_b",
                [
                    "--sequence",
                    sequence,
                    "--output",
                    b_front,
                    *common,
                    "--front-only",
                    "--runtime",
                    runner.runtime,
                ],
            ),
            (
                "fit_c_front",
                "head_rgbd.fit_c",
                [
                    "--sequence",
                    sequence,
                    "--output",
                    c_front,
                    *common,
                    "--front-only",
                    "--expression-components",
                    "30",
                    "--runtime",
                    runner.runtime,
                ],
            ),
            (
                "fit_c_guarded",
                "head_rgbd.fit_c",
                [
                    "--sequence",
                    sequence,
                    "--output",
                    c_guarded,
                    *common,
                    "--front-anchor",
                    c_front,
                    "--expression-components",
                    "30",
                    "--runtime",
                    runner.runtime,
                ],
            ),
        ]
        for name, module, stage_args in stages:
            require_stage_success(runner, runner.run(name, module, stage_args))
        verification = runner.output / "verification.json"
        reference = runner.output / "reference_geometry.json"
        require_stage_success(
            runner,
            runner.run(
                "verify_guarded",
                "head_rgbd.verify_c",
                ["--fit", c_guarded, "--sequence", sequence, "--output", verification],
            ),
        )
        require_stage_success(
            runner,
            runner.run(
                "reference_metrics",
                "head_rgbd.reference_metrics",
                [
                    "--fit",
                    c_guarded,
                    "--sequence",
                    sequence,
                    "--output",
                    reference,
                    "--runtime",
                    runner.runtime,
                ],
            ),
        )
        comparison_dir = runner.output / "comparison"
        require_stage_success(
            runner,
            runner.run(
                "compare_all_methods",
                "head_rgbd.evaluate_c",
                [
                    "--sequence",
                    sequence,
                    "--original-comparison",
                    original,
                    "--b",
                    b,
                    "--b-front",
                    b_front,
                    "--c",
                    c_guarded,
                    "--c-front",
                    c_front,
                    "--output",
                    comparison_dir,
                ],
            ),
        )
        comparison_path = comparison_dir / "comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        acceptance = _geometry_acceptance(comparison)
        summary = {
            "sequence": sequence_info,
            "original_comparison": {
                "path": str(original),
                "sha256": sha(original),
            },
            "methods": {
                name: _fit_summary(path)
                for name, path in (
                    ("B", b),
                    ("B_front", b_front),
                    ("C_front", c_front),
                    ("C_guarded", c_guarded),
                )
            },
            "comparison": str(comparison_path),
            "geometry_effect_acceptance": acceptance,
            "independent_accuracy": {
                "status": "unverified",
                "reason": "no independent extrinsics, manual contours or reference scan",
            },
        }
        code = runner.finish(
            statuses={
                "geometry_effect_acceptance": acceptance["status"],
                "independent_accuracy": "unverified",
            },
            summary=summary,
        )
    except Exception as exc:
        runner.report["error"] = f"{type(exc).__name__}: {exc}"
        code = runner.finish()
        raise
    print(json.dumps({"run_report": str(runner.report_path), "comparison": str(comparison_path)}))
    return code


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, Any]]:
    defaults = load_defaults()
    capture_defaults = defaults["capture"]
    parser = argparse.ArgumentParser(
        prog="headrgbd",
        description="Standalone Gemini RGB-D head reconstruction prototype",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        help="runtime root containing env, assets, captures, outputs and logs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="verify runtime and report environment")
    status.add_argument("--output", type=Path)
    status.set_defaults(func=command_status)

    assets = sub.add_parser("assets", help="verify runtime asset existence and SHA256")
    assets.add_argument("--output", type=Path)
    assets.set_defaults(func=command_assets)

    probe = sub.add_parser("probe", help="open the Orbbec camera and save aligned frames")
    probe.add_argument("--timeout-ms", type=int, default=1000)
    probe.add_argument("--samples", type=int, default=3)
    probe.add_argument("--output", type=Path)
    probe.set_defaults(func=command_probe)

    capture = sub.add_parser("capture", help="guided multi-segment RGB-D capture")
    capture.add_argument("--session-name", default="head")
    capture.add_argument("--frames-per-view", type=int, default=capture_defaults["frames_per_view"])
    capture.add_argument("--interval-s", type=float, default=capture_defaults["interval_s"])
    capture.add_argument("--warmup-s", type=float, default=capture_defaults["warmup_s"])
    capture.add_argument("--countdown-s", type=int, default=capture_defaults["countdown_s"])
    capture.add_argument("--views", action="append")
    capture.add_argument("--auto", action="store_true")
    capture.add_argument("--no-preview", action="store_true")
    capture.set_defaults(func=command_capture)

    prepare = sub.add_parser("prepare", help="prepare a capture for offline fitting")
    inputs = prepare.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--capture", type=Path)
    inputs.add_argument("--frame-manifest", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--landmark-device", choices=("cpu", "cuda", "auto"), default="cpu")
    prepare.set_defaults(func=command_prepare)

    reconstruct = sub.add_parser(
        "reconstruct",
        help="fit shared identity and per-frame motion; guarded mode is default",
    )
    reconstruct.add_argument("--sequence", type=Path, required=True)
    reconstruct.add_argument("--output", type=Path, required=True)
    reconstruct.add_argument("--free-identity", action="store_true")
    reconstruct.add_argument("--research-only", action="store_true")
    _add_fit_options(reconstruct, defaults)
    reconstruct.set_defaults(func=command_reconstruct)

    verify = sub.add_parser("verify", help="validate geometry, landmarks and support exports")
    verify.add_argument("--fit", type=Path, required=True)
    verify.add_argument("--sequence", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(func=command_verify)

    run = sub.add_parser("run", help="prepare optional raw capture, reconstruct and verify")
    run_inputs = run.add_mutually_exclusive_group(required=True)
    run_inputs.add_argument("--capture", type=Path)
    run_inputs.add_argument("--sequence", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--landmark-device", choices=("cpu", "cuda", "auto"), default="cpu")
    run.add_argument("--free-identity", action="store_true")
    run.add_argument("--research-only", action="store_true")
    _add_fit_options(run, defaults)
    run.set_defaults(func=command_run)

    capture_run = sub.add_parser(
        "capture-run",
        help="capture a guided session, then prepare, reconstruct and verify offline",
    )
    capture_run.add_argument("--session-name", default="head")
    capture_run.add_argument(
        "--frames-per-view", type=int, default=capture_defaults["frames_per_view"]
    )
    capture_run.add_argument("--interval-s", type=float, default=capture_defaults["interval_s"])
    capture_run.add_argument("--warmup-s", type=float, default=capture_defaults["warmup_s"])
    capture_run.add_argument("--countdown-s", type=int, default=capture_defaults["countdown_s"])
    capture_run.add_argument("--views", action="append")
    capture_run.add_argument("--auto", action="store_true")
    capture_run.add_argument("--no-preview", action="store_true")
    capture_run.add_argument("--output", type=Path)
    capture_run.add_argument("--landmark-device", choices=("cpu", "cuda", "auto"), default="cpu")
    capture_run.add_argument("--free-identity", action="store_true")
    capture_run.add_argument("--research-only", action="store_true")
    _add_fit_options(capture_run, defaults)
    capture_run.set_defaults(func=command_capture_run)

    regression = sub.add_parser(
        "regression",
        help="run B, B-front, C-front, guarded C and all-frame comparison serially",
    )
    regression.add_argument("--sequence", type=Path, required=True)
    regression.add_argument("--original-comparison", type=Path, required=True)
    regression.add_argument("--output", type=Path, required=True)
    _add_fit_options(regression, defaults)
    regression.set_defaults(func=command_regression)
    return parser, defaults


def main(argv: list[str] | None = None) -> int:
    parser, _ = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "free_identity"):
        _validate_identity_mode(parser, args)
    if hasattr(args, "runtime") and args.runtime is not None:
        os.environ["HEADRGBD_RUNTIME_DIR"] = str(Path(args.runtime).expanduser().resolve())
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
