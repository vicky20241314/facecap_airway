"""Command line entry point for the front-anchor turn experiment.

    python -m head_rgbd.experiment.turn_run --anchor <frontal RGB-D> --turn <turn RGB-D>

This module owns the orchestration only. Each stage runs in its own process through
:class:`head_rgbd.reporting.StageRunner`, so a stage sees the exact interpreter,
environment and log file that produced its output.

Stage order, one pass, no iteration:

1. ``tracking``            lightweight continuous yaw/pitch/roll and continuity
2. ``keyframe_selection``  angle bins plus quality, hard budget, no padding
3. ``prepared_keyframes``  observations for the selected keyframes only
4. ``front_anchor``        ``head_rgbd.fit_c --front-only`` on the frontal subset
5. ``joint_shared_identity`` one shared shape over all keyframes
6. ``pose_correction``     ``head_rgbd.motion_v2``, one pass
7. ``control_field``       one low/mid-frequency field plus one Loop blend
8. ``delivery``            six output files and nothing else

``--mode`` runs a single stage against an existing run directory; it exists so the
stages can be re-run in isolation without changing the orchestrator.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from head_rgbd import __version__
from head_rgbd.audit import sha
from head_rgbd.experiment import EXPERIMENT, EXPERIMENT_VERSION
from head_rgbd.experiment.turn_config import TurnConfig
from head_rgbd.experiment.turn_inputs import ROLE_ANCHOR, ROLE_TURN, open_source
from head_rgbd.experiment.turn_keyframes import select
from head_rgbd.paths import configure_environment, create_exclusive_dir, resolve_runtime_dir
from head_rgbd.pipeline import asset_gate
from head_rgbd.reporting import StageRunner, require_stage_success, write_json

EXPERIMENT_SCHEMA = "head_rgbd.turn_experiment.v1"
STAGE_ORDER = ("tracking", "selection", "prepared", "front_anchor", "joint_shape",
               "pose_correction", "field", "delivery")
MODES = STAGE_ORDER


def _configure_stdio():
    """Match StageRunner's UTF-8 log decoder on Windows as well as Unix."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


# -------------------------------------------------------------------------- main
def main(argv=None) -> int:
    _configure_stdio()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode:
        if not args.run:
            parser.error("--mode requires --run <existing run directory>")
        return _run_stage(args)
    if not args.anchor or not args.turn:
        parser.error("--anchor and --turn are both required")
    return _run_experiment(args, parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchor", type=Path,
                        help="frontal closed-mouth RGB-D anchor (session/frames/sequence/npz)")
    parser.add_argument("--turn", type=Path,
                        help="continuous closed-mouth left-to-right head-turn RGB-D sequence")
    parser.add_argument("--output", type=Path, default=None,
                        help="new run directory; defaults to <runtime>/turn_experiment/<name>_<timestamp>")
    parser.add_argument("--runtime", type=Path, default=None, help="runtime root containing assets")
    parser.add_argument("--config", type=Path, default=None,
                        help="experiment configuration; defaults to config/turn_experiment.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--detect-stride", type=int, default=None,
                        help="override tracking.detect_stride")
    parser.add_argument("--max-keyframes", type=int, default=None,
                        help="override keyframes.max_keyframes")
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default=None,
                        help="stop after this stage; the run directory keeps what was produced")
    parser.add_argument("--mode", choices=MODES, default=None, help="run one stage against --run")
    parser.add_argument("--run", type=Path, default=None, help="existing run directory for --mode")
    return parser


def _run_experiment(args, parser) -> int:
    config = TurnConfig.load(args.config)
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.detect_stride is not None:
        overrides["tracking"] = {"detect_stride": args.detect_stride}
    if args.max_keyframes is not None:
        overrides["keyframes"] = {"max_keyframes": args.max_keyframes}
    if overrides:
        config = config.replace(**overrides)

    runtime = resolve_runtime_dir(args.runtime)
    configure_environment(runtime)
    assets = asset_gate(runtime)

    # Strict input validation happens before any directory is created, so a colour
    # video or an unaligned source fails without leaving a half-built run behind.
    sources = {
        ROLE_ANCHOR: open_source(args.anchor, ROLE_ANCHOR, config),
        ROLE_TURN: open_source(args.turn, ROLE_TURN, config),
    }

    if args.output is not None:
        run_dir = Path(args.output).expanduser().resolve()
        create_exclusive_dir(run_dir)
    else:
        stem = Path(args.turn).resolve().name or "turn"
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = create_exclusive_dir(
            runtime / config.experiment.output_root_name / f"{_slug(stem)}_{config.experiment.output_suffix}_{stamp}")

    experiment = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment": EXPERIMENT,
        "version": EXPERIMENT_VERSION,
        "project_version": __version__,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": [str(item) for item in sys.argv],
        "python": sys.executable,
        "runtime": str(runtime),
        "device": config.experiment.device,
        "output": str(run_dir),
        "config": config.to_dict(),
        "config_path": config.source_path,
        "config_sha256": sha(config.source_path) if config.source_path else None,
        "assets": {"status": assets["status"], "count": assets["count"]},
        "inputs": {role: sources[role].provenance() for role in sources},
        "stage_order": list(STAGE_ORDER),
        "deliverables": ["keyframe_selection.json", "init_mesh.obj", "joint_mesh.obj",
                         "final_mesh.obj", "experiment_config.json", "stages.json"],
    }
    write_json(run_dir / "experiment.json", experiment, exclusive=True)

    runner = StageRunner(run_dir / "audit", runtime=runtime,
                         purpose=f"{EXPERIMENT}: frontal RGB-D anchor plus continuous closed head turn")
    runner.report["experiment"] = {"name": EXPERIMENT, "version": EXPERIMENT_VERSION,
                                   "run": str(run_dir), "config": str(config.source_path)}
    runner.report["source_hashes"] = _source_hashes(config.source_path)
    runner.flush()

    plan = [
        ("tracking", "tracking", "head_rgbd.experiment.turn_run",
         ["--mode", "tracking", "--run", run_dir]),
        ("keyframe_selection", "selection", "head_rgbd.experiment.turn_run",
         ["--mode", "selection", "--run", run_dir]),
        ("prepared_keyframes", "prepared", "head_rgbd.experiment.turn_run",
         ["--mode", "prepared", "--run", run_dir]),
        ("front_anchor", "front_anchor", "head_rgbd.fit_c",
         _front_anchor_arguments(run_dir, config, runtime)),
        ("joint_shared_identity", "joint_shape", "head_rgbd.experiment.turn_run",
         ["--mode", "joint_shape", "--run", run_dir]),
        ("pose_correction", "pose_correction", "head_rgbd.motion_v2",
         _pose_correction_arguments(run_dir, config, runtime)),
        ("control_field", "field", "head_rgbd.experiment.turn_run",
         ["--mode", "field", "--run", run_dir]),
        ("delivery", "delivery", "head_rgbd.experiment.turn_run",
         ["--mode", "delivery", "--run", run_dir]),
    ]
    try:
        for name, mode, module, arguments in plan:
            require_stage_success(runner, runner.run(name, module, arguments))
            if args.stop_after and mode == args.stop_after:
                runner.report["stopped_after"] = name
                break
        _finalize_delivery_statistics(run_dir, runner.report)
        summary = _read_stage_summary(run_dir)
        runner.finish(statuses={"pipeline_execution": "passed", "geometry_quality": "not_evaluated",
                                "independent_accuracy": "unverified",
                                "real_capture_pipeline": "passed"},
                      summary=summary)
    except BaseException as exc:
        runner.report["error"] = f"{type(exc).__name__}: {exc}"
        runner.finish()
        raise
    print(json.dumps({"run": str(run_dir), "delivery": str(run_dir / "delivery"),
                      "elapsed_s": round(runner.report.get("elapsed_s") or 0., 2)}, indent=2), flush=True)
    return 0


def _front_anchor_arguments(run_dir: Path, config, runtime: Path) -> list:
    return [
        "--sequence", run_dir / "prepared" / "anchor_sequence.json",
        "--output", run_dir / "front_anchor",
        "--device", config.experiment.device,
        "--points", str(config.front_init.points_per_frame),
        "--steps", str(config.front_init.steps),
        "--keyframes-per-segment", str(config.keyframes.max_keyframes),
        "--expression-components", str(config.flame.expression_components),
        "--postfit", "minimal",
        "--front-only",
        "--runtime", runtime,
    ]


def _pose_correction_arguments(run_dir: Path, config, runtime: Path) -> list:
    return [
        "--sequence", run_dir / "prepared" / "sequence.json",
        "--initial-fit", run_dir / "joint_shape",
        "--output", run_dir / "pose_correction",
        "--runtime", runtime,
        "--device", config.experiment.device,
        "--steps", str(config.pose_correction.steps),
    ]


# ------------------------------------------------------------------- stage modes
def _run_stage(args) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    experiment = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    config = TurnConfig.from_dict(experiment["config"], source_path=experiment.get("config_path"))
    configure_environment(experiment.get("runtime"))
    role = args.mode
    if role == "tracking":
        _stage_tracking(run_dir, config, experiment)
    elif role == "selection":
        _stage_selection(run_dir, config, experiment)
    elif role == "prepared":
        from head_rgbd.experiment.turn_prepare import materialise

        materialise(run_dir, config, experiment)
    elif role == "joint_shape":
        from head_rgbd.experiment.turn_shape import run as shape_run

        shape_run(run_dir, config, experiment.get("device", config.experiment.device))
    elif role == "field":
        from head_rgbd.experiment.turn_field import run as field_run

        field_run(run_dir, config)
    elif role == "delivery":
        _stage_delivery(run_dir, config, experiment)
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(f"{role} is handled by its own module, not by --mode")
    return 0


def _stage_tracking(run_dir: Path, config, experiment: dict) -> None:
    from head_rgbd.experiment import turn_tracking

    output = run_dir / "tracking"
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for role in (ROLE_ANCHOR, ROLE_TURN):
        sequence = open_source(experiment["inputs"][role]["source"]["path"], role, config)
        result = turn_tracking.analyze(sequence, config, role)
        write_json(output / f"{role}.json", {
            "schema": "head_rgbd.turn.tracking.v1",
            "role": role,
            "input": sequence.metadata.as_dict(),
            "summary": result["summary"],
            "frames": result["records"],
        })
        np.savez_compressed(output / f"{role}.npz", **result["arrays"])
        summary[role] = result["summary"]
        print(json.dumps({"role": role, "frames": len(result["records"]),
                          "usable": result["summary"]["usable_frames"],
                          "tracking_groups": result["summary"]["tracking_group_count"]}), flush=True)
    write_json(output / "summary.json", {
        "schema": "head_rgbd.turn.tracking.summary.v1",
        "roles": summary,
        "policy": ("detector every detect_stride frames or after a loss; forward-backward KLT in between; "
                   "yaw is a landmark-rigid angle, not a goniometer"),
    })


def _stage_selection(run_dir: Path, config, experiment: dict) -> None:
    output = run_dir / "selection"
    output.mkdir(parents=True, exist_ok=False)
    anchor = json.loads((run_dir / "tracking" / "anchor.json").read_text(encoding="utf-8"))
    turn = json.loads((run_dir / "tracking" / "turn.json").read_text(encoding="utf-8"))
    entries = {}
    for role, payload in ((ROLE_ANCHOR, anchor), (ROLE_TURN, turn)):
        arrays = np.load(run_dir / "tracking" / f"{role}.npz", allow_pickle=False)
        visible = arrays["visible_landmarks"]
        merged = []
        for record in payload["frames"]:
            item = dict(record)
            index = int(record["source_index"])
            item["visible_landmarks"] = visible[index] if index < len(visible) else None
            merged.append(item)
        entries[role] = merged
    result = select(entries[ROLE_ANCHOR], entries[ROLE_TURN], config)
    result.update({
        "schema": "head_rgbd.turn.keyframes.v1",
        "experiment": EXPERIMENT,
        "version": EXPERIMENT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            "max_keyframes": config.keyframes.max_keyframes,
            "anchor_budget": config.keyframes.anchor_budget,
            "repeat_observations_per_angle": config.keyframes.repeat_observations_per_angle,
            "repeat_max_angle_delta_deg": config.keyframes.repeat_max_angle_delta_deg,
            "angle_bin_deg": config.keyframes.angle_bin_deg,
            "min_yaw_separation_deg": config.keyframes.min_yaw_separation_deg,
            "max_yaw_deg": config.keyframes.max_yaw_deg,
            "min_quality": config.keyframes.min_quality,
            "front_yaw_deg": config.keyframes.front_yaw_deg,
            "anchor_max_yaw_deg": config.keyframes.anchor_max_yaw_deg,
        },
        "tracking_summary": {"anchor": anchor["summary"], "turn": turn["summary"]},
    })
    write_json(output / "keyframes.json", result)
    print(json.dumps({"selected": len(result["keyframes"]),
                      "support_observations": len(result["support_frames"]),
                      "angle_groups": len(result["angle_groups"]),
                      "tracking_groups": len(result["tracking_groups"]),
                      "stop_reason": result["budget"]["stop_reason"],
                      "yaw_span_deg": result["coverage"]["yaw_span_deg"]}, indent=2), flush=True)


def _stage_delivery(run_dir: Path, config, experiment: dict) -> None:
    output = run_dir / "delivery"
    output.mkdir(parents=True, exist_ok=False)
    field_report = json.loads((run_dir / "field" / "report.json").read_text(encoding="utf-8"))
    joint_report = json.loads((run_dir / "joint_shape" / "report.json").read_text(encoding="utf-8"))
    anchor_report = json.loads((run_dir / "front_anchor" / "report.json").read_text(encoding="utf-8"))
    reference = field_report["reference_frame"]

    stored = np.load(run_dir / "field" / "final_vertices.npz", allow_pickle=False)
    final_vertices = stored["vertices"]
    dense_faces = stored["faces"]
    reference_index = _index_of(joint_report["frames"], reference)

    import trimesh

    _export(output / "final_mesh.obj", final_vertices[reference_index], dense_faces)
    joint = np.load(run_dir / "joint_shape" / "parameters.npz", allow_pickle=False)
    _export(output / "joint_mesh.obj", joint["posed_vertices"][reference_index], joint["faces"])
    anchor = np.load(run_dir / "front_anchor" / "parameters.npz", allow_pickle=False)
    anchor_index = _index_of(anchor_report["frames"], reference, allow_fallback=True)
    _export(output / "init_mesh.obj", anchor["posed_vertices"][anchor_index], anchor["faces"])

    shutil.copy2(run_dir / "selection" / "keyframes.json", output / "keyframe_selection.json")

    write_json(output / "experiment_config.json", {
        "schema": "head_rgbd.turn_experiment.delivery_config.v1",
        "experiment": EXPERIMENT,
        "version": EXPERIMENT_VERSION,
        "project_version": __version__,
        "config": config.to_dict(),
        "config_path": experiment.get("config_path"),
        "config_sha256": experiment.get("config_sha256"),
        "device": experiment.get("device"),
        "runtime": experiment.get("runtime"),
        "inputs": {role: {"source": item["source"], "metadata": item["metadata"],
                          "frames": item["frames"]} for role, item in experiment["inputs"].items()},
        "effective_overrides": _overrides(config, experiment),
    })

    run_report = json.loads((run_dir / "audit" / "run_report.json").read_text(encoding="utf-8"))
    tracking_summary = json.loads((run_dir / "tracking" / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection" / "keyframes.json").read_text(encoding="utf-8"))
    stages = [{"name": row["name"], "module": row["module"], "status": row["status"],
               "exit_code": row.get("exit_code"), "elapsed_s": None if row.get("elapsed_s") is None
               else round(row["elapsed_s"], 3)} for row in run_report.get("stages", [])]
    write_json(output / "stages.json", {
        "schema": "head_rgbd.turn_experiment.stages.v1",
        "experiment": EXPERIMENT,
        "version": EXPERIMENT_VERSION,
        "run": str(run_dir),
        "reference_frame": reference,
        "coordinate_note": ("each mesh is written in the reference frame's camera coordinates with that "
                            "stage's own model pose; the three meshes are comparable but not registered"),
        "stages": stages,
        "stage_elapsed_s": {row["name"]: row["elapsed_s"] for row in stages},
        "total_elapsed_s": round(float(run_report.get("elapsed_s") or 0.), 3),
        "observation_stats": {
            "tracking": {role: {"frames": item["frames"], "usable_frames": item["usable_frames"],
                                "tracking_groups": item["tracking_group_count"],
                                "detector_calls": item.get("detector_calls"),
                                "tracked_by_klt": item.get("tracked_by_klt"),
                                "unusable_reasons": item["unusable_reasons"]}
                         for role, item in tracking_summary["roles"].items()},
            "keyframes": {
                "selected": len(selection["keyframes"]),
                "support_observations": len(selection.get("support_frames", [])),
                "total_observations": len(selection["keyframes"]) + len(selection.get("support_frames", [])),
                "budget": selection["budget"]["max_keyframes"],
                "padded": selection["budget"]["padded"],
                "stop_reason": selection["budget"]["stop_reason"],
                "angle_groups": len(selection["angle_groups"]),
                "tracking_groups": len(selection["tracking_groups"]),
                "coverage": selection["coverage"],
                "rejected": selection["rejected"],
            },
            "shared_shape": {
                "frames": len(joint_report["frames"]),
                "angle_weighting": joint_report["angle_weighting"],
                "shape_delta_from_anchor_absmax": joint_report["shape_delta_from_anchor_absmax"],
                "pose_change_from_initial": joint_report["pose_change_from_initial"],
                "positive_correspondences": sum(
                    int(row.get("depth_correspondences") or 0) for row in joint_report["frames"]),
            },
            "control_field": {
                "maximum_absolute_displacement_mm": field_report["field"]["maximum_absolute_displacement_mm"],
                "free_set_rule": field_report["field"]["free_set_rule"],
                "loop_blend": field_report["subdivision"]["loop_blend"]["blend"],
                "loop_target": field_report["subdivision"]["loop_blend"]["target"],
                "loop_fallback_reason": field_report["subdivision"]["loop_blend"]["fallback_reason"],
                "dense_topology": field_report["subdivision"]["topology"],
                "final_safety": field_report["field"]["final_safety"],
            },
            "metrics": field_report["metrics"],
        },
        "metrics_to_watch": list(config.delivery.metrics_to_watch),
        "statuses": {"implementation": "implemented", "geometry_quality": "not_evaluated",
                     "independent_accuracy": "unverified", "speed_improvement": "not_claimed",
                     "accuracy_improvement": "not_claimed"},
    })
    print(json.dumps({"delivery": str(output), "files": sorted(item.name for item in output.iterdir())},
                     indent=2), flush=True)


def _export(path: Path, vertices, faces) -> None:
    import trimesh

    mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(faces), process=False)
    mesh.export(path, include_normals=True)
    reloaded = trimesh.load(path, process=False, maintain_order=True)
    if len(reloaded.vertices) != len(mesh.vertices) or len(reloaded.faces) != len(mesh.faces):
        raise ValueError(f"exported mesh {path} did not round-trip")


def _index_of(rows, frame_id: str, *, allow_fallback: bool = False) -> int:
    for index, row in enumerate(rows):
        if row.get("id") == frame_id:
            return index
    if allow_fallback:
        for index, row in enumerate(rows):
            if row.get("view") == "front":
                return index
        return 0
    raise ValueError(f"reference frame {frame_id!r} is missing from the stage report")


# ------------------------------------------------------------------------ helpers
def _finalize_delivery_statistics(run_dir: Path, run_report: dict) -> None:
    """The delivery child cannot know its own final exit status or duration."""
    path = run_dir / "delivery" / "stages.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = [{key: row.get(key) for key in ("name", "module", "status", "exit_code", "elapsed_s")}
              for row in run_report["stages"]]
    payload["stages"] = stages
    payload["stage_elapsed_s"] = {row["name"]: row["elapsed_s"] for row in stages}
    payload["total_elapsed_s"] = round(float(run_report.get("elapsed_s") or 0.), 3)
    write_json(path, payload)


def _read_stage_summary(run_dir: Path) -> dict:
    stages_path = run_dir / "delivery" / "stages.json"
    if not stages_path.is_file():
        return {"delivery": "not_reached"}
    payload = json.loads(stages_path.read_text(encoding="utf-8"))
    return {"delivery": str(run_dir / "delivery"),
            "reference_frame": payload.get("reference_frame"),
            "total_elapsed_s": payload.get("total_elapsed_s"),
            "stage_elapsed_s": payload.get("stage_elapsed_s"),
            "keyframes": payload["observation_stats"]["keyframes"]["selected"]}


def _overrides(config, experiment: dict) -> dict:
    original = TurnConfig.load(experiment.get("config_path"))
    effective = config.to_dict()
    base = original.to_dict()
    return {section: {key: value for key, value in effective[section].items() if base[section].get(key) != value}
            for section in effective if isinstance(effective[section], dict)
            and any(base[section].get(key) != value for key, value in effective[section].items())}


def _source_hashes(config_path) -> dict:
    from head_rgbd.paths import PROJECT_ROOT

    root = PROJECT_ROOT
    targets = [root / "VERSION"]
    targets += sorted((root / "head_rgbd" / "experiment").glob("*.py"))
    if config_path:
        targets.append(Path(config_path))
    return {str(path.relative_to(root)): sha(path) for path in targets if path.is_file()}


def _slug(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_"
                      for character in value.lower())
    return (cleaned.strip("_-") or "turn")[:60]


if __name__ == "__main__":
    raise SystemExit(main())
