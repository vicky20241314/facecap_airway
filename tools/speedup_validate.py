"""Run one speed-up validation phase and collect comparable metrics.

Protocol (per phase): one full chain on the ``zp`` capture plus the three 0.9.0
base-only captures.  Every run gets a new directory; nothing is overwritten.
The tool only reads the produced reports/meshes after each run and writes
``metrics.json`` (plus ``runs.jsonl``) under the phase root.

The algorithm itself is untouched: this is a runner/collector.

Example:
    python -m tools.speedup_validate --phase p0 --label p0_...
        --profile tools/speedup_profiles/p0.json --refresh-downstream
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
RUNTIME = PROJ / "runtime"
PYTHON = RUNTIME / "env" / "python.exe"
DOWNSTREAM_SRC = Path(r"D:\proiect\head_windpipe_regis\src")

RUNS = (
    dict(key="zp_full", kind="full", capture="20261108_164853_124297_head"),
    dict(key="jzh_base", kind="base", capture="20261103_160315_882457_head"),
    dict(key="s100624_base", kind="base", capture="20261115_100624_353630_head"),
    dict(key="s193720_base", kind="base", capture="20261115_193720_747162_head"),
)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["HEADRGBD_RUNTIME_DIR"] = str(RUNTIME)
    env["WITHB_RUNTIME_ROOT"] = str(RUNTIME)
    return env


def _cmd(profile: dict, kind: str, capture: Path, output: Path) -> list[str]:
    args = [str(PYTHON), "-m", "head_rgbd", "rebuild",
            "--capture", str(capture), "--output", str(output)]
    args += [str(a) for a in profile.get("common", [])]
    if kind == "base":
        args.append("--base-only")
    args += [str(a) for a in profile.get("base", [])]
    if kind == "full":
        args += [str(a) for a in profile.get("full", [])]
    return args


def _history_phase_final(history: list) -> dict:
    out: dict = {}
    for row in history or []:
        phase = row.get("phase")
        if phase is not None:
            out[phase] = {"step": row.get("step"), "objective": row.get("objective")}
    return out


def _fit_metrics(stage_dir: Path) -> dict:
    report = _json(stage_dir / "report.json", {}) or {}
    timing = report.get("timing") or {}
    return {
        "elapsed_s": report.get("elapsed_s"),
        "fitting_s": report.get("fitting_s"),
        "steps": report.get("steps"),
        "timing": timing,
        "history": report.get("history") or [],
        "phase_final": _history_phase_final(report.get("history") or []),
        "anchor_joint_update_mm": report.get("joint_anchor_update_mm"),
        "local_field_max_mm": report.get("local_field_max_mm"),
        "expression_components": report.get("expression_components"),
        "front_anchor": (report.get("front_anchor") or {}).get("maximum_side_detail_change_mm"),
    }


def _motion_metrics(stage_dir: Path) -> dict:
    report = _json(stage_dir / "report.json", {}) or {}
    motion = report.get("motion_correction") or {}
    acceptance = motion.get("fixed_ray_acceptance") or []
    rows = []
    for row in acceptance:
        if row.get("original_fixed_ray_mse_mm2") is None:
            continue
        rows.append(row)
    accepted = [r for r in rows if r.get("accepted")]
    orig = [r["original_fixed_ray_mse_mm2"] for r in rows]
    final = [r["final_fixed_ray_mse_mm2"] for r in rows]
    return {
        "elapsed_s": report.get("elapsed_s"),
        "steps": motion.get("steps"),
        "timing": motion.get("timing") or {},
        "history": motion.get("history") or [],
        "phase_final": _history_phase_final(motion.get("history") or []),
        "frame_acceptance": rows,
        "fixed_ray": {
            "frames": len(rows),
            "accepted": len(accepted),
            "original_mse_mean_mm2": (sum(orig) / len(orig)) if orig else None,
            "final_mse_mean_mm2": (sum(final) / len(final)) if final else None,
            "final_mse_max_mm2": max(final) if final else None,
            "all_covered": bool(rows) and all(
                int(r.get("fixed_ray_count") or 0) > 0 for r in rows),
        },
        "lowerface_frames": motion.get("lowerface_frames"),
    }


def _dense_metrics(stage_dir: Path) -> dict:
    report = _json(stage_dir / "report.json", {}) or {}
    fields = {}
    for name, log in (report.get("field_logs") or {}).items():
        entries = log.get("history") or []
        last = entries[-1] if entries else {}
        diag = last.get("fixed_ray_diagnostic") or {}
        fields[name] = {
            "outer_steps": len(entries),
            "final_objective_after": last.get("objective_after"),
            "final_objective_before": last.get("objective_before"),
            "fixed_ray_after_mse_mm2": diag.get("after_mse_mm2"),
            "fixed_ray_before_mse_mm2": diag.get("before_mse_mm2"),
            "missing_after": diag.get("missing_after"),
            "missing_allowance": diag.get("missing_allowance"),
            "ray_count": diag.get("ray_count"),
            "actual_update_mm": last.get("actual_update_mm"),
        }
    loop = report.get("loop_base") or {}
    return {
        "elapsed_s": report.get("elapsed_s"),
        "timing_sections": {row.get("section"): row.get("elapsed_s")
                            for row in report.get("timing_sections") or []},
        "fields": fields,
        "loop_actual_blend": loop.get("blend"),
        "loop_fallback_reason": loop.get("fallback_reason"),
        "dense_change_from_interpolated_mm": report.get("dense_change_from_interpolated_regularized_mm"),
        "maximum_dense_total_displacement_mm": report.get("maximum_dense_total_displacement_mm"),
        "statuses": report.get("statuses"),
        "dense_topology": report.get("dense_topology"),
    }


def _acceptance(run_dir: Path) -> dict:
    result = run_dir / "result"
    quick = _json(result / "quick_check.json", {}) or {}
    q = quick.get("quick") or {}
    out = {
        "quick_present": bool(quick),
        "variant": quick.get("variant"),
        "chosen": quick.get("chosen"),
        "variants": {k: {"metric": (v or {}).get("metric")}
                     for k, v in (quick.get("variants") or {}).items()},
    }
    if q:
        s = q.get("signed_crease") or {}
        out.update({
            "pass": q.get("pass"),
            "protection": q.get("protection"),
            "topology_pass": (q.get("topology") or {}).get("pass"),
            "shared_topology": q.get("shared_topology"),
            "upper_band": q.get("upper_band_0.925_0.97"),
            "structure": q.get("structure"),
            "crease": s,
            "vertices": q.get("vertices"),
            "faces": q.get("faces"),
        })
    for role in ("final", "measured"):
        manifest = _json(result / role / "manifest.json", {}) or {}
        out[role] = {
            "present": bool(manifest),
            "status": manifest.get("status"),
            "delivery_role": manifest.get("delivery_role"),
            "topology_shared": (manifest.get("mesh") or {}).get("topology_shared"),
            "sidecar_written": ((manifest.get("provenance") or {}).get("sidecar_written")),
            "sidecar_file": (result / role / "anatomical_landmarks.json").is_file(),
            "mesh": {state: ((manifest.get("mesh") or {}).get("states") or {}).get(state)
                     for state in ("closed", "open")},
        }
    return out


def _downstream(run_dir: Path) -> dict:
    if not DOWNSTREAM_SRC.is_dir():
        return {"status": "skipped", "reason": str(DOWNSTREAM_SRC)}
    result = run_dir / "result"
    if not (result / "final" / "manifest.json").is_file():
        return {"status": "skipped", "reason": "no result/final"}
    if str(DOWNSTREAM_SRC) not in sys.path:
        sys.path.insert(0, str(DOWNSTREAM_SRC))
    from input_resolution import resolve_inputs  # noqa: WPS433
    out = {}
    for role in ("final", "measured"):
        for state in ("closed", "open"):
            head = result / role / f"head_{state}.obj"
            try:
                rec = resolve_inputs(head)
                out[f"{role}/{state}"] = {
                    "status": rec.get("status"),
                    "problems": rec.get("problems"),
                    "state": (rec.get("discovery") or {}).get("state"),
                    "point_field": (rec.get("landmark") or {}).get("point_field"),
                    "mode": rec.get("mode"),
                }
            except Exception as exc:  # keep the phase report honest
                out[f"{role}/{state}"] = {"status": "error",
                                          "problems": [f"{type(exc).__name__}: {exc}"]}
    out["all_pass"] = all(row.get("status") == "pass" for key, row in out.items()
                          if "/" in key)
    return out


def collect(run_dir: Path) -> dict:
    report = _json(run_dir / "run_report.json", {}) or {}
    stages = {}
    for row in report.get("stages") or []:
        stages[row.get("name")] = {
            "elapsed_s": row.get("elapsed_s"),
            "exit_code": row.get("exit_code"),
            "status": row.get("status"),
        }
    recon = run_dir / "reconstruction"
    neck = _json(run_dir / "neck_direct_v2" / "build_direct.json", {}) or {}
    record = {
        "run_dir": str(run_dir),
        "capture": (report.get("capture") or {}).get("directory")
        if isinstance(report.get("capture"), dict) else report.get("capture"),
        "purpose": report.get("purpose"),
        "status": report.get("status"),
        "elapsed_s": report.get("elapsed_s"),
        "configuration": report.get("configuration"),
        "stages": stages,
        "neck_seconds": neck.get("seconds"),
        "budget": {"total": neck.get("budget"),
                   "region": neck.get("budget_by_region"),
                   "zone": neck.get("budget_by_zone")},
        "fit_c": {name: _fit_metrics(recon / name)
                  for name in ("front_anchor", "guarded") if (recon / name / "report.json").is_file()},
        "motion": _motion_metrics(recon / "motion_only")
        if (recon / "motion_only" / "report.json").is_file() else None,
        "dense": _dense_metrics(recon / "dense")
        if (recon / "dense" / "report.json").is_file() else None,
        "acceptance": _acceptance(run_dir),
        "delivery_manifest": _json(run_dir / "delivery" / "manifest.json", {}),
        "rebuild_timing": _json(run_dir / "rebuild_timing.json", {}),
    }
    return record


def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--profile", type=Path, required=True,
                        help="JSON {common:[], base:[], full:[]} extra CLI args")
    parser.add_argument("--runs", default="zp_full,jzh_base,s100624_base,s193720_base")
    parser.add_argument("--refresh-downstream", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.label or args.phase
    root = (args.output_root or (RUNTIME / "outputs" / f"_speedup_{label}_{stamp}")).resolve()
    if root.exists():
        parser.error(f"output root already exists, refusing to overwrite: {root}")
    profile = _json(args.profile, None)
    if not isinstance(profile, dict):
        parser.error(f"profile not readable: {args.profile}")
    wanted = {row.strip() for row in args.runs.split(",") if row.strip()}
    runs = [row for row in RUNS if row["key"] in wanted]

    plan = []
    for row in runs:
        capture = RUNTIME / "captures" / row["capture"]
        output = root / row["key"]
        plan.append((row, capture, output, _cmd(profile, row["kind"], capture, output)))
    for row, capture, output, cmd in plan:
        print(json.dumps({"run": row["key"], "capture": str(capture),
                          "output": str(output), "cmd": cmd}, ensure_ascii=False))
    if args.dry_run:
        return 0

    root.mkdir(parents=True, exist_ok=False)
    started = _now()
    metrics = {
        "schema": "speedup.validation.v1",
        "phase": args.phase,
        "label": label,
        "created_at": started,
        "profile": profile,
        "runs": {},
    }
    lines = []
    for row, capture, output, cmd in plan:
        # The pipeline creates its run directory itself (exclusive); creating it
        # here would make StageRunner fail with FileExistsError.
        t0 = time.perf_counter()
        completed = subprocess.run(cmd, cwd=str(PROJ), env=_env())
        wall = time.perf_counter() - t0
        record = collect(output)
        record.update({"key": row["key"], "kind": row["kind"],
                       "exit_code": completed.returncode, "wall_s": wall,
                       "capture_name": row["capture"]})
        if row["kind"] == "full" and args.refresh_downstream:
            record["downstream"] = _downstream(output)
        else:
            record["downstream"] = None
        metrics["runs"][row["key"]] = record
        lines.append({key: record.get(key) for key in
                      ("key", "kind", "exit_code", "wall_s", "elapsed_s",
                       "neck_seconds", "stages")})
        print(f"[speedup] {row['key']}: exit={completed.returncode} "
              f"wall={wall:.1f}s", flush=True)
    metrics["finished_at"] = _now()
    (root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
        encoding="utf-8")
    with (root / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for line in lines:
            stream.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"root": str(root), "metrics": str(root / "metrics.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
