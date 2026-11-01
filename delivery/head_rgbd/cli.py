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
    parser.add_argument("--countdown-s", type=int, default=capture["countdown_s"])
    parser.add_argument("--views", action="append", help="state:view, default closed:front closed:left closed:right open:front")
    parser.add_argument("--auto", action="store_true", help="no Enter prompt between segments")
    parser.add_argument("--no-preview", action="store_true", help="hide the OpenCV preview window")


def _add_fit_options(parser: argparse.ArgumentParser, defaults: dict) -> None:
    reconstruction = defaults["reconstruction"]
    parser.add_argument("--device", choices=("cuda", "cpu"), default=reconstruction["device"])
    parser.add_argument("--points", type=int, default=reconstruction["points_per_frame"])
    parser.add_argument("--steps", type=int, default=reconstruction["steps"])
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
        action="store_true",
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
        "--landmark-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="TUFA landmark network device; UniFace parsing always uses CPU",
    )


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
    print(json.dumps({"capture": str(folder), "manifest": str(folder / "session.json")}, indent=2))
    return 0


def command_run(args: argparse.Namespace) -> int:
    from head_rgbd.capture_session import capture_session

    runtime = resolve_runtime_dir(args.runtime)
    ensure_runtime_layout(runtime)
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
        else ensure_runtime_layout(runtime)["outputs"] / f"{capture.name}_reconstruction"
    )
    code, delivery = run_pipeline(
        runtime=runtime,
        output=output,
        capture=capture,
        sequence=None,
        args=args,
        purpose="live_capture_to_offline_reconstruction",
    )
    _attach_capture(output / "run_report.json", capture, "passed")
    print(json.dumps({"meshes": _mesh_paths(delivery), "manifest": str(delivery / "manifest.json")}, indent=2))
    return code


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
    print(json.dumps({"meshes": _mesh_paths(delivery), "manifest": str(delivery / "manifest.json")}, indent=2))
    return code


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
    manifest = export_delivery(
        fit=fit,
        sequence=sequence,
        output=run / "delivery",
        runtime=runtime,
        capture=None,
        runner=None,
        defaults=load_defaults(),
    )
    print(
        json.dumps(
            {
                "meshes": _mesh_paths(run / "delivery"),
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
            "  headrgbd.cmd reconstruct --capture runtime/captures/<session>\n"
            "  headrgbd.cmd device\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"headrgbd {__version__} ({ALGORITHM})")
    parser.add_argument("--runtime", type=Path, help="runtime root (default: <project>/runtime)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="capture a guided session, reconstruct and export closed + open meshes",
    )
    _add_capture_options(run, defaults)
    _add_fit_options(run, defaults)
    run.add_argument("--output", type=Path, help="run directory (default: runtime/outputs/<capture>_reconstruction)")
    run.set_defaults(func=command_run)

    capture = sub.add_parser("capture", help="capture raw RGB-D segments only")
    _add_capture_options(capture, defaults)
    capture.set_defaults(func=command_capture)

    reconstruct = sub.add_parser(
        "reconstruct",
        help="rebuild an existing capture (session.json) or prepared sequence.json",
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
    reconstruct.set_defaults(func=command_reconstruct)

    export = sub.add_parser(
        "export",
        help="re-export closed + open meshes from a completed run (no refitting)",
    )
    export.add_argument("--run", type=Path, required=True, help="run directory holding reconstruction/dense")
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
