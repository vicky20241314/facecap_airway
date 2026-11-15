"""Standalone CLI for local loss analysis. Never part of the reconstruction path.

Every command requires an explicit path. There is no default capture, no default
run directory and no automatic discovery: without input the tool prints help (or
fails with a clear message) instead of analysing something arbitrary.

Commands
--------
stages    print the stage/term catalogue transcribed from the delivery source
inspect   describe what a run directory contains (read-only)
analyze   read a completed run, record real objectives, write CSV/JSON/plots/report.md/report.html
converge  per-stage convergence plateau, phase-local gap and amortised time ledger
gap       per-stage distance to the layered termination target, plus targeted candidates
optimize  offline whitelisted parameter search against tiered terminal targets

The analyser never launches a reconstruction, never touches the GPU and never
writes inside the run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent           # .../loss_analysis/loss_analysis
ANALYSIS_ROOT = PACKAGE_ROOT.parent                       # .../loss_analysis
PROJECT_ROOT = ANALYSIS_ROOT.parent                       # .../facecap_airway_delivery
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime" / "loss_analysis"


def _catalogue_rows() -> list[dict]:
    from .objectives import ALL_STAGES

    return [
        {
            "key": stage.key,
            "label_cn": stage.label_cn,
            "module": stage.module,
            "parameters": list(stage.parameters),
            "solver": stage.solver,
            "objective_scope": stage.objective_scope,
            "notes": list(stage.notes),
            "terms": [
                {
                    "key": term.key,
                    "label_cn": term.label_cn,
                    "kind": term.kind,
                    "source": term.source,
                    "normaliser": term.normaliser,
                    "normaliser_unit": term.normaliser_unit,
                    "weight": term.weight,
                    "weight_unit": term.weight_unit,
                    "reduction": term.reduction,
                    "denominator": term.denominator,
                    "notes": term.notes,
                }
                for term in stage.terms
            ],
        }
        for stage in ALL_STAGES
    ]


def _variant_rows() -> list[dict]:
    """v0.4.0 mode variants: one stage key, several objectives."""
    from .objectives import MODE_VARIANTS

    return [
        {
            "variant_key": variant.variant_key,
            "stage_key": variant.stage_key,
            "label_cn": variant.label_cn,
            "activated_by": variant.activated_by,
            "data_frames": variant.data_frames,
            "term_overrides": variant.term_overrides,
            "evidence": variant.evidence,
        }
        for variant in MODE_VARIANTS
    ]


def command_stages(args: argparse.Namespace) -> int:
    rows = _catalogue_rows()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for stage in rows:
        print(f"{stage['key']}  —  {stage['label_cn']}")
        print(f"  module : {stage['module']}")
        print(f"  solver : {stage['solver']}")
        print(f"  scope  : {stage['objective_scope']}")
        for term in stage["terms"]:
            print(f"    - {term['key']:<24} {term['kind']:<8} w={term['weight']:<40} {term['source']}")
        print()
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    from .adapter import AdapterError, describe_inputs

    try:
        summary = describe_inputs(args.run)
    except AdapterError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _new_output_dir(base: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = Path(base) / f"{stamp}_{label}"
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def command_analyze(args: argparse.Namespace) -> int:
    from .adapter import AdapterError, describe_inputs, ingest_skills, ingest_solver_logs, run_manifest
    from .diagnostics import depth_surface_error_mm, load_frame, support_label_summary
    from .html_report import build_analysis_html
    from . import neck_adapter
    from .objectives import ALL_STAGES
    from .plots import build_report, plot_gap_panel, plot_metrics, plot_stage_series, write_json, write_report
    from .recorder import IterationRecord, Recorder

    run_dir = Path(args.run).expanduser().resolve()
    try:
        summary = describe_inputs(run_dir)
    except AdapterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest = run_manifest(run_dir) or {}
    software = manifest.get("software", {})
    label = args.label or run_dir.name
    recorder = Recorder(
        run_label=label,
        algorithm=software.get("algorithm", "unknown"),
        software_version=software.get("version", "unknown"),
    )
    counts = ingest_solver_logs(recorder, run_dir)
    chain_mode = summary.get("chain_shape_fusion")
    skill_status = ingest_skills(recorder, run_dir, chain_mode=chain_mode)

    metrics: dict[str, dict] = {}
    diagnostics_note: list[str] = []
    if args.sequence:
        sequence_path = Path(args.sequence).expanduser().resolve()
        if not sequence_path.is_file():
            print(f"ERROR: sequence not found: {sequence_path}", file=sys.stderr)
            return 2
        mesh_dir = Path(args.mesh_dir).expanduser().resolve() if args.mesh_dir else run_dir / "delivery"
        for state, name in (("closed", "head_closed_m.obj"), ("open", "head_open_m.obj")):
            mesh_path = mesh_dir / name
            if not mesh_path.is_file():
                recorder.gap(
                    f"diagnostics.{state}",
                    f"mesh not found: {mesh_path}",
                    expected_tier="diagnostic",
                    source_hint=str(mesh_path),
                )
                continue
            try:
                import trimesh

                mesh = trimesh.load(str(mesh_path), process=False, maintain_order=True)
                vertices = mesh.vertices
                faces = mesh.faces
            except Exception as exc:  # noqa: BLE001 - report, do not crash the report
                recorder.gap(f"diagnostics.{state}", f"mesh unreadable: {type(exc).__name__}: {exc}",
                             expected_tier="diagnostic", source_hint=str(mesh_path))
                continue
            sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
            reference = next(
                (row for row in sequence.get("frames", []) if row.get("status") == "candidate"
                 and row.get("state") == state),
                None,
            )
            if reference is None:
                recorder.gap(f"diagnostics.{state}", "no candidate frame of this state in sequence",
                             expected_tier="diagnostic")
                continue
            frame = load_frame(sequence_path.parent / reference["file"])
            result = depth_surface_error_mm(
                vertices, faces, frame["xyz"], frame["valid"],
                max_depth_m=args.max_depth_mm / 1000.0,
            )
            # frame ids already start with their state (closed_front_2), so no prefix
            metrics[reference["id"]] = result
            metric_values = result.get("metrics", {})
            coverage_block = result.get("coverage", {})
            for key in ("mae_mm", "rmse_mm", "p95_mm"):
                value = metric_values.get(key)
                recorder.add(
                    IterationRecord(
                        stage=f"diagnostics.{state}",
                        iteration=0,
                        tier="diagnostic",
                        value=value,
                        decision=key,
                        source="recomputed",
                        notes=[
                            f"{key} on fixed real rays of {reference['id']}; diagnostic only",
                            f"sample size {metric_values.get('n_valid')} of {coverage_block.get('n_required')} "
                            f"prepared observations hit the surface inside the depth gate: the metric is over "
                            f"that subset, not over the whole face",
                        ],
                        extras={
                            "coverage": coverage_block.get("coverage"),
                            "n_covered": coverage_block.get("n_covered"),
                            "n_required": coverage_block.get("n_required"),
                            "n_valid": metric_values.get("n_valid"),
                            "median_mm": metric_values.get("median_mm"),
                            "max_mm": metric_values.get("max_mm"),
                            "reference_frame": reference["id"],
                        },
                    )
                )
        diagnostics_note.append(
            "深度误差基于真实 prepared 帧的固定射线，与算法内部目标不可比较"
        )
    else:
        recorder.gap(
            "diagnostics",
            "no --sequence supplied: prepared frames were not read, so no depth diagnostic exists",
            expected_tier="diagnostic",
        )

    support = support_label_summary(run_dir / "delivery" / "support.json")

    neck_result = None
    if args.neck_result:
        neck_result = Path(args.neck_result).expanduser().resolve()
        if not neck_result.is_dir():
            print(f"ERROR: neck result not found: {neck_result}", file=sys.stderr)
            return 2
        neck_dir = neck_adapter.find_neck_dir(neck_result)
        neck_adapter.ingest_neck_direct(recorder, neck_dir)
        gates = neck_adapter.neck_gates(neck_result, neck_dir)
        protection = gates.get("protection") or {}
        upper = gates.get("upper_band") or {}
        recorder.declare_stage("neck_direct.gates", has_iterations=False, notes=[
            f"quick_check pass={gates.get('quick_pass')}; "
            f"shared_topology={gates.get('shared_topology')}",
            f"connection max positive crease={gates.get('connection_max_deg')} deg "
            f"(the delivery records the value even when it exceeds the 20 deg gate)",
            f"protection pass={protection.get('pass')} "
            f"fixed_max={protection.get('fixed_max_mm')} mm",
            f"upper band no_increase={upper.get('no_increase')}",
            "acceptance gates are booleans, not a numeric loss; they are not "
            "converted to 1/0",
        ])
        diagnostics_note.append(
            f"颈部轨迹来自 {neck_dir}；验收门为布尔值，不参与 loss 求和"
        )
    else:
        recorder.gap(
            "neck_direct",
            "no --neck-result supplied: the direct neck trace and its acceptance "
            "gates were not read; the tool does not infer them from the final mesh",
            expected_tier="solver_objective",
        )

    out_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _new_output_dir(DEFAULT_OUTPUT, label)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = recorder.write_json(out_dir / "records.json")
    csv_path = recorder.write_csv(out_dir / "records.csv")
    payload = recorder.payload()

    figures: list[str] = []
    if not args.no_plots:
        # Plot every stage that actually has records, plus every catalogue stage
        # (which may only contribute a "no data" placeholder via the gap panel).
        recorded = sorted({row["stage"] for row in payload.get("iterations", [])})
        catalogue = sorted({stage.key for stage in ALL_STAGES})
        for stage_key in sorted(set(recorded) | set(catalogue)):
            name = stage_key.replace(".", "_")
            figure = plot_stage_series(payload, out_dir / "figures" / f"{name}.png", stage=stage_key)
            if figure:
                figures.append(str(figure))
        figures.append(str(plot_gap_panel(payload, out_dir / "figures" / "_gaps.png")))
        metric_figure = plot_metrics(metrics, out_dir / "figures" / "_diagnostics.png")
        if metric_figure:
            figures.append(str(metric_figure))

    notes = diagnostics_note + [
        f"per-stage solver records: {counts}",
        f"field stages read: {skill_status}",
        f"chain shape_fusion (from artefacts): {chain_mode}",
        f"version stamps inside the run: {summary.get('versions', {}).get('stated')} "
        f"— consistent={summary.get('versions', {}).get('consistent')}",
    ]
    report = build_report(
        payload,
        stage_catalogue=_catalogue_rows(),
        mode_variants=_variant_rows(),
        input_summary={**summary, "support": support},
        metrics=metrics or None,
        extra_notes=notes,
    )
    report_path = write_report(out_dir / "report.md", report)
    html_path = write_report(out_dir / "report.html", build_analysis_html(
        payload,
        stage_catalogue=_catalogue_rows(),
        mode_variants=_variant_rows(),
        input_summary={**summary, "support": support},
        metrics=metrics or None,
        extra_notes=notes,
        figures=figures,
    ))
    write_json(out_dir / "summary.json", {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "records": str(json_path),
        "csv": str(csv_path),
        "report": str(report_path),
        "html": str(html_path),
        "figures": figures,
        "neck_result": str(neck_result) if neck_result else None,
        "gaps": len(payload.get("gaps", [])),
        "iterations": len(payload.get("iterations", [])),
        "support": support,
        "inputs": summary,
        "note": "analysis only: no reconstruction was run and the run directory was not modified",
    })
    projected = sum(1 for value in counts.values() if value)
    print(json.dumps({
        "output_dir": str(out_dir),
        "report": str(report_path),
        "html": str(html_path),
        "records": str(json_path),
        "csv": str(csv_path),
        "figures": figures,
        "stages_with_solver_records": projected,
        "gaps": [
            {"stage": gap.stage, "reason": gap.reason}
            for gap in recorder.gaps
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def command_gap(args: argparse.Namespace) -> int:
    from .gap_report import compute_gaps, write_outputs

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        return 2
    dense_dir = (Path(args.dense).expanduser().resolve()
                 if args.dense else run_dir / "reconstruction" / "dense")
    if not dense_dir.is_dir():
        print(f"ERROR: dense directory not found: {dense_dir}", file=sys.stderr)
        return 2
    result_dir = Path(args.neck_result).expanduser().resolve() if args.neck_result else None
    if result_dir is not None and not result_dir.is_dir():
        print(f"ERROR: result directory not found: {result_dir}", file=sys.stderr)
        return 2
    trace_dir = Path(args.trace).expanduser().resolve() if args.trace else None
    if trace_dir is not None and not trace_dir.is_dir():
        print(f"ERROR: trace directory not found: {trace_dir}", file=sys.stderr)
        return 2
    landmarks = Path(args.landmarks).expanduser().resolve() if args.landmarks else None
    out_dir = (Path(args.output).expanduser().resolve()
               if args.output else _new_output_dir(DEFAULT_OUTPUT, "gaps"))

    report = compute_gaps(
        run_dir=run_dir,
        dense_dir=dense_dir,
        result_dir=result_dir,
        landmarks=landmarks,
        trace_dir=trace_dir,
    )
    paths = write_outputs(report, out_dir)
    print(json.dumps({
        "output_dir": paths["output_dir"],
        "report": paths["report"],
        "gaps_json": paths["gaps_json"],
        "csv": paths["csv"],
        "dominant": (report.get("dominant") or {}).get("key"),
        "dominant_status": (report.get("dominant") or {}).get("status"),
        "targeted": [item["key"] for item in report.get("targeted") or []],
        "gaps": report.get("gaps"),
    }, ensure_ascii=False, indent=2))
    return 0


def command_optimize(args: argparse.Namespace) -> int:
    from .optimizer import run_optimizer

    return run_optimizer(args)


def command_converge(args: argparse.Namespace) -> int:
    from .converge import run_converge

    return run_converge(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lossana",
        description=(
            "Local loss analysis for facecap_airway_delivery. Reads a finished run, "
            "records real objectives by provenance tier and writes diagnostics. "
            "Never runs reconstruction and never writes inside the run directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples\n"
            "  python -m loss_analysis stages\n"
            "  python -m loss_analysis inspect --run runtime\\outputs\\<run>\n"
            "  python -m loss_analysis analyze --run runtime\\outputs\\<run> --sequence "
            "runtime\\outputs\\<run>\\prepared\\sequence.json\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"lossana {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    stages = sub.add_parser("stages", help="print the stage/term catalogue from the source")
    stages.add_argument("--json", action="store_true", help="machine-readable output")
    stages.set_defaults(func=command_stages)

    inspect = sub.add_parser("inspect", help="describe a run directory (read-only)")
    inspect.add_argument("--run", type=Path, required=True, help="run directory to inspect")
    inspect.set_defaults(func=command_inspect)

    analyze = sub.add_parser("analyze", help="record real objectives and write diagnostics")
    analyze.add_argument("--run", type=Path, required=True, help="completed run directory")
    analyze.add_argument("--sequence", type=Path,
                         help="prepared sequence.json (enables depth diagnostics; optional)")
    analyze.add_argument("--mesh-dir", type=Path,
                         help="directory holding head_closed_m.obj / head_open_m.obj (default: <run>/delivery)")
    analyze.add_argument("--output", type=Path, help="new output directory (default: runtime/loss_analysis/<stamp>_<label>)")
    analyze.add_argument("--label", help="run label used in the output directory name")
    analyze.add_argument("--max-depth-mm", type=float, default=30.0,
                         help="ray hit depth gate in mm (default 30, same order as the algorithm's 30 mm query)")
    analyze.add_argument("--neck-result", type=Path,
                         help="rebuild result directory (quick_check.json + neck dir); enables the "
                              "direct neck trace and its acceptance gates")
    analyze.add_argument("--no-plots", action="store_true", help="write CSV/JSON/report only")
    analyze.set_defaults(func=command_analyze)

    optimize = sub.add_parser(
        "optimize",
        help="offline whitelisted parameter search (plan by default; --execute to run)",
    )
    optimize.add_argument("--baseline-result", type=Path, required=True,
                          help="delivered result directory used as the baseline (read-only)")
    optimize.add_argument("--baseline-dense", type=Path, required=True,
                          help="base dense directory of the baseline (fixed-ray tier)")
    optimize.add_argument("--baseline-run", type=Path,
                          help="base run directory holding logs/ (default: the run that owns "
                               "--baseline-dense, else --baseline-result)")
    optimize.add_argument("--baseline-neck", type=Path,
                          help="neck directory of the baseline (default: resolved from quick_check)")
    optimize.add_argument("--baseline-trace", type=Path,
                          help="optional neck directory holding loss_direct.json for the baseline trace")
    optimize.add_argument("--targeted", action="store_true",
                          help="select at most two candidates from the dominant gap instead of the "
                               "full one-move plan")
    optimize.add_argument("--capture", type=Path,
                          help="capture folder, required when a candidate changes a base parameter")
    optimize.add_argument("--output", type=Path,
                          help="new output directory (default: runtime/loss_analysis/<stamp>_neck_optimizer)")
    optimize.add_argument("--candidate", action="append", metavar="KEY=VALUE",
                          help="explicit candidate override (whitelisted key); "
                               "when omitted a one-move plan is generated")
    optimize.add_argument("--limit", type=int, default=12,
                          help="maximum generated candidates (hard cap 12)")
    optimize.add_argument("--execute", action="store_true",
                          help="run the selected candidates; without it only the plan is written")
    optimize.add_argument("--max-candidates", type=int, default=1,
                          help="how many candidates to execute (default 1)")
    optimize.add_argument("--freeze-landmarks", action="store_true",
                          help="forward the baseline neck_landmarks.json to candidates; "
                               "off by default so the live prepared recheck runs")
    optimize.add_argument("--neck-only-dense", action="store_true",
                          help="run neck-only candidates off the baseline dense "
                               "(no prepared/, so T3 is skipped and the delivery is "
                               "not_usable); default re-runs base from --capture")
    optimize.add_argument("--landmarks", type=Path,
                          help="baseline neck_landmarks.json (default: <baseline>/../neck_landmarks.json)")
    optimize.add_argument("--timeout-s", type=int, default=3600,
                          help="per-candidate rebuild timeout (default 3600 s)")
    optimize.add_argument("--no-figures", action="store_true",
                          help="do not draw solver-trace figures")
    optimize.set_defaults(func=command_optimize)

    gap = sub.add_parser(
        "gap",
        help="per-stage distance to the layered termination target (read-only)",
    )
    gap.add_argument("--run", type=Path, required=True,
                     help="base run directory holding run_report.json and logs/")
    gap.add_argument("--dense", type=Path,
                     help="base dense directory (default: <run>/reconstruction/dense)")
    gap.add_argument("--neck-result", type=Path,
                     help="delivered result directory (quick_check.json + neck dir); "
                          "without it the acceptance layer is recorded as a gap")
    gap.add_argument("--trace", type=Path,
                     help="neck directory holding loss_direct.json (default: resolved neck dir)")
    gap.add_argument("--landmarks", type=Path,
                     help="neck_landmarks.json used by the delivery (marker semantics)")
    gap.add_argument("--output", type=Path,
                     help="new output directory (default: runtime/loss_analysis/<stamp>_gaps)")
    gap.set_defaults(func=command_gap)

    converge = sub.add_parser(
        "converge",
        help="per-stage convergence plateau, phase-local gap and amortised time ledger (read-only)",
    )
    converge.add_argument("--run", type=Path, action="append", default=[],
                          help="completed run directory; repeatable, aggregated")
    converge.add_argument("--reference-run", type=Path, action="append", default=[],
                          help="completed run shown separately, excluded from the aggregate")
    converge.add_argument("--epsilon", type=float, default=0.002,
                          help="remaining-potential threshold for a plateau marker "
                               "(default 0.002 = 0.2%%)")
    converge.add_argument("--persist-steps", type=int, default=60,
                          help="a plateau must persist at least this many steps "
                               "(default 60; the logs print every 20 steps)")
    converge.add_argument("--output", type=Path,
                          help="new output directory (default: runtime/loss_analysis/<stamp>_converge)")
    converge.add_argument("--no-figures", action="store_true",
                          help="do not draw per-stage objective figures")
    converge.set_defaults(func=command_converge)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        print("\n(no input supplied: pass a subcommand, e.g. 'stages' or 'inspect --run <dir>')")
        return 0
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - the CLI must not print a raw traceback
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
