"""Per-stage convergence plateau and the amortised time ledger.

Read-only: the module never launches a reconstruction and never writes inside a
run directory.

Reference rule
--------------
A gap is measured against the SAME run's final recorded value inside the SAME
phase.  Objectives are not comparable across phases or stages: ``fit_c``
changes its active term set between the five phases and every stage defines its
own denominator, so a cross-phase or cross-stage difference is recorded as a
gap, never subtracted.

The plateau marker is a *candidate cut point*, not a source convergence
threshold.  ``fit_c`` / ``motion_v2`` stop on a fixed step budget and define no
loss tolerance, so the report never claims that a stage "converged".  Time
savings are amortised (stage ``elapsed_s`` / recorded steps) and are labelled
as estimates: correspondence refreshes make the real per-step cost non-uniform.
"""
from __future__ import annotations

import ast
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from .objectives import LINEARISED, SOLVER_OBJECTIVE
from .plots import _matplotlib, write_json, write_report

SCHEMA = "loss_analysis.convergence.v1"
DEFAULT_EPSILON = 0.002
DEFAULT_PERSIST_STEPS = 60
MIN_POINTS_FOR_PLATEAU = 3

FIT_C = "fit_c"
MOTION = "motion_v2"
DENSE = "dense"
PREPARE = "prepare"
OTHER = "other"


def _is_number(value: Any) -> bool:
    return (not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _num(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _load(path: Path | None) -> dict:
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _stage_kind(name: str) -> str:
    lowered = name.lower()
    if "dense" in lowered:
        return DENSE
    if "motion" in lowered:
        return MOTION
    if "identity" in lowered:
        return FIT_C
    if lowered.startswith("prepare"):
        return PREPARE
    return OTHER


def parse_objective_line(line: str) -> dict | None:
    """Accept both the JSON and the Python-dict lines the stage logs contain."""
    text = line.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(data, dict):
        return None
    step, value = data.get("step"), data.get("objective")
    if not _is_number(step) or not _is_number(value):
        return None
    phase = data.get("phase")
    return {
        "step": int(step),
        "value": float(value),
        "phase": str(phase) if phase not in (None, "") else "unlabelled",
    }


def read_objective_points(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = parse_objective_line(line)
        if record is not None:
            records[record["step"]] = record
    return [records[step] for step in sorted(records)]


def _phase_groups(points: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: list[tuple[str, list[dict]]] = []
    for point in points:
        if not groups or groups[-1][0] != point["phase"]:
            groups.append((point["phase"], []))
        groups[-1][1].append(point)
    return groups


def _phase_analysis(name: str, index: int, records: list[dict],
                    epsilon: float, persist_steps: int) -> dict:
    values = [row["value"] for row in records]
    final_value = values[-1]
    final_best = min(values)
    denominator = max(abs(final_best), 1e-12)
    best = math.inf
    previous: dict | None = None
    rows: list[dict] = []
    for record, value in zip(records, values):
        best = min(best, value)
        improvement = previous["best_so_far"] - best if previous else None
        step_delta = record["step"] - previous["step"] if previous else None
        rows.append({
            "phase": name,
            "index": index,
            "step": record["step"],
            "value": value,
            "best_so_far": best,
            "gap_to_phase_final": value - final_value,
            "relative_gap": (value - final_value) / denominator,
            "remaining_potential": best - final_best,
            "remaining_potential_relative": (best - final_best) / denominator,
            "improvement_from_prev": improvement,
            "improvement_per_step": (
                improvement / step_delta
                if improvement is not None and step_delta not in (None, 0) else None
            ),
        })
        previous = rows[-1]

    candidate = None
    status = "insufficient_points"
    if len(rows) >= MIN_POINTS_FOR_PLATEAU:
        last_step = rows[-1]["step"]
        status = "no_plateau"
        for offset, row in enumerate(rows):
            if last_step - row["step"] < persist_steps:
                continue
            if all(later["remaining_potential_relative"] <= epsilon
                   for later in rows[offset:]):
                candidate = {
                    "cut_step": row["step"],
                    "saved_steps": last_step - row["step"],
                    "lost_potential_relative": row["remaining_potential_relative"],
                    "raw_gap_relative": row["relative_gap"],
                    "raw_gap": row["gap_to_phase_final"],
                }
                status = "plateau"
                break

    window_start = rows[0]
    if len(rows) >= 2:
        target = rows[-1]["step"] - min(persist_steps, rows[-1]["step"] - rows[0]["step"])
        window_start = next((row for row in rows if row["step"] >= target), rows[0])
    last_window_improvement = window_start["best_so_far"] - final_best
    return {
        "phase": name,
        "index": index,
        "first_step": records[0]["step"],
        "last_step": records[-1]["step"],
        "n_points": len(records),
        "final_value": final_value,
        "final_best_value": final_best,
        "last_window_start_step": window_start["step"],
        "last_window_steps": rows[-1]["step"] - window_start["step"],
        "last_window_improvement": last_window_improvement,
        "last_window_improvement_relative": last_window_improvement / denominator,
        "candidate": candidate,
        "status": status,
        "points": rows,
    }


def _amortised_saving(stage_elapsed: float | None, saved_steps: int,
                      stage_steps: int | None) -> float | None:
    if stage_elapsed is None or not stage_steps:
        return None
    return float(stage_elapsed) * saved_steps / float(stage_steps)


def _analyse_solver_stage(record: dict, points: list[dict],
                          kind: str, epsilon: float,
                          persist_steps: int) -> dict:
    elapsed = _num(record.get("elapsed_s"))
    last_step = points[-1]["step"] if points else None
    per_step = (elapsed / last_step
                if elapsed is not None and last_step not in (None, 0) else None)
    phases = [
        _phase_analysis(name, index, rows, epsilon, persist_steps)
        for index, (name, rows) in enumerate(_phase_groups(points))
    ]
    for index, phase in enumerate(phases):
        candidate = phase.get("candidate")
        if candidate is None:
            continue
        candidate["saved_s_estimate"] = _amortised_saving(
            elapsed, candidate["saved_steps"], last_step)
        candidate["application"] = (
            "total_budget" if index == len(phases) - 1 else "phase_reallocation")
    last = phases[-1] if phases else None
    stage_candidate = None
    if last and last.get("candidate"):
        stage_candidate = dict(last["candidate"])
        stage_candidate["phase"] = last["phase"]
    return {
        "name": record.get("name"),
        "kind": kind,
        "tier": SOLVER_OBJECTIVE,
        "elapsed_s": elapsed,
        "per_step_s_estimate": per_step,
        "steps_first": points[0]["step"] if points else None,
        "steps_last": last_step,
        "n_points": len(points),
        "cut_scope": "candidate",
        "phases": phases,
        "stage_candidate": stage_candidate,
        "reason": None,
    }


def _dense_points(run_dir: Path) -> tuple[list[dict], dict | None, str | None]:
    report_path = run_dir / "reconstruction" / "dense" / "report.json"
    if not report_path.is_file():
        return [], None, f"dense report not found: {report_path}"
    report = _load(report_path)
    field_logs = report.get("field_logs") or {}
    points: list[dict] = []
    for key, block in field_logs.items():
        if not isinstance(block, dict):
            continue
        for entry in block.get("history") or []:
            if not isinstance(entry, dict):
                continue
            value = _num(entry.get("objective_after"))
            outer = entry.get("outer")
            if value is None or not _is_number(outer):
                continue
            points.append({"step": int(outer), "value": value,
                           "phase": key, "stage": entry.get("stage")})
    points.sort(key=lambda row: (row["phase"], row["step"]))
    return points, report, None


def _analyse_dense_stage(run_dir: Path, record: dict,
                         epsilon: float, persist_steps: int) -> tuple[dict, list[str]]:
    gaps: list[str] = []
    points, report, error = _dense_points(run_dir)
    elapsed = _num(record.get("elapsed_s"))
    if error:
        gaps.append(error)
    if not points:
        return {
            "name": record.get("name"),
            "kind": DENSE,
            "tier": LINEARISED,
            "elapsed_s": elapsed,
            "per_step_s_estimate": None,
            "steps_first": None,
            "steps_last": None,
            "n_points": 0,
            "cut_scope": "ledger_only",
            "phases": [],
            "stage_candidate": None,
            "reason": error or "dense report contains no field_logs history",
        }, gaps
    phases = [
        _phase_analysis(name, index, rows, epsilon, persist_steps)
        for index, (name, rows) in enumerate(_phase_groups(points))
    ]
    for phase in phases:
        phase["candidate"] = None
        phase["status"] = "ledger_only"
    return {
        "name": record.get("name"),
        "kind": DENSE,
        "tier": LINEARISED,
        "elapsed_s": elapsed,
        "per_step_s_estimate": None,
        "steps_first": None,
        "steps_last": None,
        "n_points": len(points),
        "cut_scope": "ledger_only",
        "phases": phases,
        "stage_candidate": None,
        "reason": ("dense field solves have only a few outer steps; the stage "
                   "time is dominated by correspondences, ray casting and export, "
                   "so amortised per-step savings would be misleading"),
    }, gaps


def _find_stage_log(run_dir: Path, record: dict) -> Path | None:
    raw = record.get("log")
    if raw:
        path = Path(raw)
        if path.is_file():
            return path
        fallback = run_dir / "logs" / path.name
        if fallback.is_file():
            return fallback
    return None


def _rebuild_total(run_dir: Path) -> tuple[float | None, str | None]:
    path = run_dir.parent / "rebuild_timings.jsonl"
    if not path.is_file():
        return None, f"rebuild timing ledger not found: {path}"
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        output = record.get("output")
        try:
            matches = output and Path(output).resolve() == run_dir
        except OSError:
            matches = False
        if matches:
            return _num(record.get("rebuild_elapsed_s")), None
    return None, f"no rebuild timing record for {run_dir}"


def analyse_run(run_dir: Path, *, label: str | None = None,
                epsilon: float = DEFAULT_EPSILON,
                persist_steps: int = DEFAULT_PERSIST_STEPS) -> dict:
    run_dir = Path(run_dir).resolve()
    report_path = run_dir / "run_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"run_report.json not found: {report_path}")
    report = _load(report_path)
    base_elapsed = _num(report.get("elapsed_s"))
    stages: list[dict] = []
    ledger: list[dict] = []
    gaps: list[str] = []
    for record in report.get("stages") or []:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "unknown")
        kind = _stage_kind(name)
        elapsed = _num(record.get("elapsed_s"))
        if kind in (FIT_C, MOTION):
            log_path = _find_stage_log(run_dir, record)
            points = read_objective_points(log_path)
            if points is None:
                gaps.append(f"{name}: objective log not found "
                            f"(recorded {record.get('log')!r})")
                stages.append({
                    "name": name, "kind": kind, "tier": SOLVER_OBJECTIVE,
                    "elapsed_s": elapsed, "per_step_s_estimate": None,
                    "steps_first": None, "steps_last": None, "n_points": 0,
                    "cut_scope": "ledger_only", "phases": [],
                    "stage_candidate": None,
                    "reason": "stage log missing; no curve and no cut candidate",
                })
            elif not points:
                gaps.append(f"{name}: log contains no objective records")
                stages.append({
                    "name": name, "kind": kind, "tier": SOLVER_OBJECTIVE,
                    "elapsed_s": elapsed, "per_step_s_estimate": None,
                    "steps_first": None, "steps_last": None, "n_points": 0,
                    "cut_scope": "ledger_only", "phases": [],
                    "stage_candidate": None,
                    "reason": "no objective records in the stage log",
                })
            else:
                stages.append(_analyse_solver_stage(
                    record, points, kind, epsilon, persist_steps))
        elif kind == DENSE:
            stage, stage_gaps = _analyse_dense_stage(
                run_dir, record, epsilon, persist_steps)
            stages.append(stage)
            gaps.extend(f"{name}: {gap}" for gap in stage_gaps)
        else:
            stages.append({
                "name": name,
                "kind": kind,
                "tier": None,
                "elapsed_s": elapsed,
                "per_step_s_estimate": None,
                "steps_first": None,
                "steps_last": None,
                "n_points": 0,
                "cut_scope": "ledger_only",
                "phases": [],
                "stage_candidate": None,
                "reason": ("no solver objective in this stage; it stays in the "
                           "time ledger only"),
            })
        ledger.append({
            "run": str(run_dir),
            "label": label or run_dir.name,
            "stage": name,
            "kind": kind,
            "elapsed_s": elapsed,
            "share_of_base": (elapsed / base_elapsed
                              if elapsed is not None and base_elapsed else None),
            "per_step_s_estimate": stages[-1].get("per_step_s_estimate"),
            "steps_last": stages[-1].get("steps_last"),
            "cut_scope": stages[-1]["cut_scope"],
            "reason": stages[-1]["reason"],
        })

    neck_seconds = _num((_load(run_dir / "result" / "build_direct.json")
                         or {}).get("seconds"))
    if neck_seconds is None:
        gaps.append(f"neck build seconds not found: "
                    f"{run_dir / 'result' / 'build_direct.json'}")
    ledger.append({
        "run": str(run_dir),
        "label": label or run_dir.name,
        "stage": "neck_direct_v2 (build)",
        "kind": "neck",
        "elapsed_s": neck_seconds,
        "share_of_base": (neck_seconds / base_elapsed
                          if neck_seconds is not None and base_elapsed else None),
        "cut_scope": "ledger_only",
        "reason": "neck fit has its own step/energy/gradient tolerances; not a "
                  "loss-budget cut target in this round",
    })
    total_elapsed, timing_note = _rebuild_total(run_dir)
    if timing_note:
        gaps.append(timing_note)
    if total_elapsed is not None:
        rest = (total_elapsed - (base_elapsed or 0.0) - (neck_seconds or 0.0))
        ledger.append({
            "run": str(run_dir),
            "label": label or run_dir.name,
            "stage": "delivery / checks / renders",
            "kind": "rest",
            "elapsed_s": rest if rest >= 0 else None,
            "share_of_base": (rest / base_elapsed
                              if rest >= 0 and base_elapsed else None),
            "cut_scope": "ledger_only",
            "reason": "derived from rebuild_timings.jsonl minus base and neck",
        })

    loss_elapsed = sum(stage["elapsed_s"] or 0.0 for stage in stages
                       if stage["kind"] in (FIT_C, MOTION))
    stage_saving = sum(
        (stage["stage_candidate"] or {}).get("saved_s_estimate") or 0.0
        for stage in stages if stage["cut_scope"] == "candidate")
    return {
        "run_dir": str(run_dir),
        "label": label or run_dir.name,
        "project_version": report.get("project_version"),
        "base_elapsed_s": base_elapsed,
        "neck_seconds": neck_seconds,
        "total_rebuild_elapsed_s": total_elapsed,
        "loss_stage_elapsed_s": loss_elapsed,
        "stage_candidate_saving_s": stage_saving,
        "stages": stages,
        "ledger": ledger,
        "gaps": gaps,
    }


def _run_candidates(run: dict) -> list[dict]:
    rows: list[dict] = []
    for stage in run["stages"]:
        for phase in stage.get("phases") or []:
            candidate = phase.get("candidate")
            if not candidate:
                continue
            rows.append({
                "run": run["run_dir"],
                "label": run["label"],
                "stage": stage["name"],
                "phase": phase["phase"],
                "application": candidate.get("application"),
                "cut_step": candidate["cut_step"],
                "saved_steps": candidate["saved_steps"],
                "lost_potential_relative": candidate["lost_potential_relative"],
                "raw_gap_relative": candidate["raw_gap_relative"],
                "saved_s_estimate": candidate.get("saved_s_estimate"),
            })
    return rows


def aggregate_candidates(aggregate_runs: list[dict]) -> list[dict]:
    by_run: dict[tuple[str, str], dict[str, dict]] = {}
    for run in aggregate_runs:
        for row in _run_candidates(run):
            key = (row["stage"], row["phase"])
            current = by_run.setdefault(key, {}).get(row["label"])
            if current is None or row["cut_step"] > current["cut_step"]:
                by_run[key][row["label"]] = row
    rows: list[dict] = []
    for (stage, phase), per_label in sorted(by_run.items()):
        items = list(per_label.values())
        status = ("all_runs_plateau"
                  if len(items) == len(aggregate_runs) else "partial_runs")
        cut = max(item["cut_step"] for item in items)
        rows.append({
            "stage": stage,
            "phase": phase,
            "application": items[0]["application"],
            "status": status,
            "runs_with_plateau": len(items),
            "runs_required": len(aggregate_runs),
            "conservative_cut_step": cut,
            "per_run": [
                {key: item[key] for key in (
                    "label", "cut_step", "saved_steps",
                    "lost_potential_relative", "raw_gap_relative",
                    "saved_s_estimate")}
                for item in sorted(items, key=lambda row: row["label"])
            ],
        })
    return rows


def _phase_diagnostics(report: dict) -> tuple[dict, list[dict]]:
    counts = {"plateau": 0, "no_plateau": 0,
              "insufficient_points": 0, "ledger_only": 0}
    closest: list[dict] = []
    for run in report["runs"]:
        for stage in run["stages"]:
            for phase in stage.get("phases") or []:
                counts[phase["status"]] = counts.get(phase["status"], 0) + 1
                if (phase["status"] in ("no_plateau", "insufficient_points")
                        and phase["n_points"] >= 2
                        and phase["last_window_steps"] > 0):
                    closest.append({
                        "run": run["label"],
                        "stage": stage["name"],
                        "phase": phase["phase"],
                        "n_points": phase["n_points"],
                        "last_window_steps": phase["last_window_steps"],
                        "last_window_improvement_relative":
                            phase["last_window_improvement_relative"],
                        "status": phase["status"],
                    })
    closest.sort(key=lambda row: row["last_window_improvement_relative"])
    return counts, closest[:10]


def _near_plateau_budget(run: dict, limit: float = 0.01) -> float:
    """Observed near-plateau windows (not candidates): amortised seconds."""
    total = 0.0
    for stage in run["stages"]:
        if stage["kind"] not in (FIT_C, MOTION):
            continue
        for phase in stage.get("phases") or []:
            if (phase["status"] == "insufficient_points"
                    and phase["n_points"] >= 2
                    and phase["last_window_steps"] > 0
                    and phase["last_window_improvement_relative"] <= limit):
                total += ((stage.get("per_step_s_estimate") or 0.0)
                          * phase["last_window_steps"])
    return total


def _summary_facts(report: dict) -> dict:
    diagnostics = report.get("diagnostics") or {}
    counts = diagnostics.get("phase_status_counts") or {}
    loss = [value for value in (diagnostics.get("loss_stage_elapsed_s") or {}).values()
            if value is not None]
    near = [value for value in (diagnostics.get("near_plateau_budget_s") or {}).values()
            if value is not None]
    fit = diagnostics.get("per_step_s", {}).get(FIT_C) or []
    motion = diagnostics.get("per_step_s", {}).get(MOTION) or []
    shares = [run["loss_stage_elapsed_s"] / run["base_elapsed_s"]
              for run in report["runs"]
              if run["loss_stage_elapsed_s"] and run["base_elapsed_s"]]
    return {
        "counts": counts,
        "loss_min": min(loss) if loss else None,
        "loss_max": max(loss) if loss else None,
        "share_min": min(shares) if shares else None,
        "share_max": max(shares) if shares else None,
        "near_min": min(near) if near else None,
        "near_max": max(near) if near else None,
        "fit_min": min(fit) if fit else None,
        "fit_max": max(fit) if fit else None,
        "motion_min": min(motion) if motion else None,
        "motion_max": max(motion) if motion else None,
    }


def _summary_html(report: dict) -> str:
    from .html_report import kv_table

    facts = _summary_facts(report)
    counts = facts["counts"]
    return kv_table([
        ("平台候选", counts.get("plateau", 0)),
        ("no_plateau / insufficient_points",
         f"{counts.get('no_plateau', 0)} / {counts.get('insufficient_points', 0)}"),
        ("迭代型 loss 时间",
         f"{_fmt(facts['loss_min'], 1)}–{_fmt(facts['loss_max'], 1)} s "
         f"（占基重建 {_pct(facts['share_min'])}–{_pct(facts['share_max'])}）"),
        ("每步均摊 fit_c", f"{_fmt(facts['fit_min'], 3)}–{_fmt(facts['fit_max'], 3)} s"),
        ("每步均摊 motion_v2",
         f"{_fmt(facts['motion_min'], 3)}–{_fmt(facts['motion_max'], 3)} s"),
        ("近平台窗口的步切上界（非候选）",
         f"{_fmt(facts['near_min'], 1)}–{_fmt(facts['near_max'], 1)} s/run；"
         f"窗口只有 1–2 个采样点，低于 K 判据"),
        ("判读", "按本轮判据没有可砍切点；近平台窗口的省时上界很小，"
                 "说明砍步不是主要提速杠杆。fit_c / motion_v2 每 10 步刷新对应关系，"
                 "每步成本才是下一轮应先量化的对象。"),
    ])


def _summary_md(report: dict) -> list[str]:
    facts = _summary_facts(report)
    counts = facts["counts"]
    return [
        "## 结论摘要",
        f"- 平台候选: {counts.get('plateau', 0)}；no_plateau "
        f"{counts.get('no_plateau', 0)}；insufficient_points "
        f"{counts.get('insufficient_points', 0)}",
        f"- 迭代型 loss 时间: {_fmt(facts['loss_min'], 1)}–{_fmt(facts['loss_max'], 1)} s，"
        f"占基重建 {_pct(facts['share_min'])}–{_pct(facts['share_max'])}",
        f"- 每步均摊: fit_c {_fmt(facts['fit_min'], 3)}–{_fmt(facts['fit_max'], 3)} s；"
        f"motion_v2 {_fmt(facts['motion_min'], 3)}–{_fmt(facts['motion_max'], 3)} s",
        f"- 近平台窗口的步切上界（非候选）: "
        f"{_fmt(facts['near_min'], 1)}–{_fmt(facts['near_max'], 1)} s/run",
        "- 判读: 按本轮判据没有可砍切点；近平台窗口的省时上界很小，"
        "砍步不是主要提速杠杆。fit_c / motion_v2 每 10 步刷新对应关系，"
        "每步成本才是下一轮应先量化的对象。",
        "",
    ]


def _point_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    for run in report["runs"]:
        for stage in run["stages"]:
            for phase in stage.get("phases") or []:
                for point in phase["points"]:
                    rows.append({
                        "run": run["label"],
                        "stage": stage["name"],
                        "kind": stage["kind"],
                        "tier": stage["tier"],
                        "phase": phase["phase"],
                        "phase_index": phase["index"],
                        **{key: point[key] for key in (
                            "step", "value", "best_so_far",
                            "gap_to_phase_final", "relative_gap",
                            "remaining_potential", "remaining_potential_relative",
                            "improvement_from_prev", "improvement_per_step")},
                    })
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _plot_stage(path: Path, run_label: str, stage: dict) -> Path | None:
    phases = [phase for phase in stage.get("phases") or [] if phase["n_points"] >= 2]
    if not phases:
        return None
    plt = _matplotlib()
    figure, axes = plt.subplots(figsize=(8.0, 3.6))
    for phase in phases:
        steps = [point["step"] for point in phase["points"]]
        values = [point["value"] for point in phase["points"]]
        line, = axes.plot(steps, values, marker="o", markersize=3,
                          linewidth=1.2, label=phase["phase"])
        candidate = phase.get("candidate")
        if candidate:
            axes.axvline(candidate["cut_step"], color=line.get_color(),
                         linestyle="--", linewidth=1.0, alpha=0.7)
            axes.annotate(f"cut {candidate['cut_step']}",
                          xy=(candidate["cut_step"], values[0]),
                          xytext=(4, 6), textcoords="offset points", fontsize=8,
                          color=line.get_color())
    axes.set_title(f"{run_label} / {stage['name']} — solver objective")
    axes.set_xlabel("step")
    axes.set_ylabel("objective (phase-local)")
    axes.grid(True, alpha=0.25)
    axes.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _ledger_table(run: dict) -> str:
    from .html_report import table

    rows = []
    for row in run["ledger"]:
        rows.append([
            row["stage"], row["kind"],
            _fmt(row["elapsed_s"], 2),
            _fmt(row["share_of_base"] * 100, 1) if row["share_of_base"] is not None else "null",
            row.get("steps_last"),
            _fmt(row.get("per_step_s_estimate"), 3),
            row["cut_scope"], row["reason"] or "",
        ])
    return table(["阶段", "类型", "elapsed_s", "占基重建 %", "记录步",
                  "每步估计 s", "本轮范围", "说明"],
                 rows, numeric={2, 3, 4, 5})


def _stage_table(run: dict) -> str:
    from .html_report import table

    rows = []
    for stage in run["stages"]:
        if not stage["phases"]:
            continue
        for phase in stage["phases"]:
            candidate = phase.get("candidate") or {}
            rows.append([
                stage["name"], phase["phase"],
                f"{phase['first_step']}–{phase['last_step']}",
                phase["n_points"],
                _fmt(phase["final_value"], 6),
                _fmt(phase["final_best_value"], 6),
                candidate.get("cut_step"),
                candidate.get("saved_steps"),
                _pct(phase.get("last_window_improvement_relative")),
                _pct(candidate.get("lost_potential_relative")),
                _pct(candidate.get("raw_gap_relative")),
                _fmt(candidate.get("saved_s_estimate"), 2),
                candidate.get("application", ""),
                phase["status"],
            ])
    return table(["阶段", "phase", "记录步", "点数", "最终原始值", "最终最优值",
                  "候选切点", "可省步", "末窗改善", "剩余潜力",
                  "切点相对最终", "省时估计 s", "应用方式", "状态"],
                 rows, numeric={3, 4, 5, 6, 7, 11})


def _aggregate_table(aggregate: list[dict]) -> str:
    from .html_report import table

    rows = []
    for row in aggregate:
        per_run = "; ".join(
            f"{item['label']}: cut {item['cut_step']}"
            for item in row["per_run"])
        savings = "; ".join(
            f"{item['label']}: {_fmt(item['saved_s_estimate'], 2)} s"
            for item in row["per_run"])
        rows.append([
            row["stage"], row["phase"], row["application"], row["status"],
            f"{row['runs_with_plateau']}/{row['runs_required']}",
            str(row["conservative_cut_step"]), per_run, savings,
        ])
    if not rows:
        return '<p class="note">没有可合并的候选切点。</p>'
    return table(["阶段", "phase", "应用方式", "状态", "覆盖", "保守切点",
                  "各 run 切点", "各 run 省时估计"],
                 rows, numeric={5})


def _html_report(report: dict, figures: list[Path]) -> str:
    from .html_report import _figure_gallery, kv_table, render_page, table

    criteria = kv_table([
        ("schema", SCHEMA),
        ("参考", "同一次运行、同一 phase 内的最终记录值"),
        ("平台判据", f"从切点到该 phase 结束至少 {report['parameters']['persist_steps']} 步，"
                    f"且此间相对剩余潜力 ≤ {report['parameters']['epsilon']}；"
                    f"百分比列以 % 显示"),
        ("采样", f"日志每 20 步一条 objective；fit_c 的 phase 只有 40–70 步，"
                 f"即 2–4 点/phase；要求剩余 ≥ {report['parameters']['persist_steps']} 步时，"
                 f"多数 phase 只能判 insufficient_points"),
        ("耗时", "阶段 elapsed_s / 记录步数（均摊估计，非实测每步）"),
        ("跨阶段", "目标不可比；跨阶段共同几何诊断不在本轮（无每阶段网格）"),
        ("候选性质", "仅候选切点，不是源码收敛阈值；砍前必须实跑验证"),
    ])
    evidence = kv_table([
        ("聚合 run", "；".join(str(path) for path in report["evidence"]["aggregate"])),
        ("参考 run", "；".join(str(path) for path in report["evidence"]["reference"]) or "无"),
        ("输出", report["output_dir"]),
    ])
    sections = [
        {"id": "criteria", "heading": "口径与边界", "html": criteria + evidence},
        {"id": "summary", "heading": "结论摘要", "html": _summary_html(report)},
    ]
    for run in report["runs"]:
        sections.append({
            "id": f"run-{len(sections)}",
            "heading": f"运行 {run['label']}",
            "html": (
                kv_table([
                    ("run", run["run_dir"]),
                    ("project_version", run["project_version"]),
                    ("基重建 elapsed_s", _fmt(run["base_elapsed_s"], 2)),
                    ("总重建 elapsed_s", _fmt(run["total_rebuild_elapsed_s"], 2)),
                    ("迭代型 loss 阶段 elapsed_s", _fmt(run["loss_stage_elapsed_s"], 2)),
                    ("候选切点均摊省时估计 s", _fmt(run["stage_candidate_saving_s"], 2)),
                ])
                + "<h3>省时账本</h3>" + _ledger_table(run)
                + "<h3>阶段收敛明细</h3>" + _stage_table(run)
            ),
        })
    sections.append({
        "id": "aggregate",
        "heading": "跨 run 合并候选（保守）",
        "html": _aggregate_table(report["aggregate"]) + (
            '<p class="note">合并规则：同一 stage/phase 取各聚合 run 中最晚的候选切点；'
            '参考 run 不参与合并。</p>'),
    })
    diagnostics = report.get("diagnostics") or {}
    counts = diagnostics.get("phase_status_counts") or {}
    closest = diagnostics.get("closest_to_plateau") or []
    closest_rows = [[
        row["run"], row["stage"], row["phase"], row["n_points"],
        row["last_window_steps"],
        _pct(row["last_window_improvement_relative"]),
        row["status"],
    ] for row in closest]
    sections.append({
        "id": "closest",
        "heading": "判定统计与最接近平台的阶段（非候选）",
        "html": (
            kv_table([
                ("plateau", counts.get("plateau", 0)),
                ("no_plateau", counts.get("no_plateau", 0)),
                ("insufficient_points", counts.get("insufficient_points", 0)),
                ("ledger_only", counts.get("ledger_only", 0)),
            ])
            + ("<p class=\"note\">下表按末窗相对改善升序排列；"
               "只有 plateau 才是候选切点，其余仅供判断是否值得改用更细日志。</p>")
            + (table(["run", "阶段", "phase", "点数", "末窗步数",
                      "末窗改善", "状态"],
                     closest_rows, numeric={3, 4})
               if closest_rows else '<p class="note">没有可比较的 phase。</p>')
        ),
    })
    gap_rows = [[gap] for gap in report["gaps"]]
    sections.append({
        "id": "gaps",
        "heading": "缺口与不解释项",
        "html": (table(["缺口"], gap_rows) if gap_rows
                 else '<p class="note">没有记录缺口。</p>'),
    })
    sections.append({
        "id": "figures",
        "heading": "曲线",
        "html": _figure_gallery(figures),
    })
    return render_page(
        title="逐阶段收敛与省时账本",
        subtitle=(f"schema {SCHEMA} · 只读分析 · "
                  f"epsilon={report['parameters']['epsilon']} · "
                  f"persist={report['parameters']['persist_steps']} steps"),
        sections=sections,
        footer=("只读分析：未启动重建、未修改任何 run 目录；候选切点必须经过缩短预算实跑验证。"),
    )


def _markdown(report: dict) -> str:
    lines = [
        "# 逐阶段收敛与省时账本",
        "",
        f"- schema: `{SCHEMA}`",
        f"- 参考: 同一次运行、同一 phase 内的最终记录值",
        f"- 平台判据: 从切点到该 phase 结束至少 {report['parameters']['persist_steps']} 步，"
        f"且此间相对剩余潜力 ≤ {report['parameters']['epsilon']}；百分比列以 % 显示",
        "- 采样: 日志每 20 步一条 objective；fit_c 的 phase 只有 40–70 步，"
        "即 2–4 点/phase；多数 phase 只能判 insufficient_points",
        "- 耗时: 阶段 elapsed_s / 记录步数（均摊估计，非实测每步）",
        "- 候选性质: 仅候选切点，不是源码收敛阈值；砍前必须实跑验证",
        "",
    ] + _summary_md(report) + [
        "## 证据集",
        f"- 聚合: {', '.join(str(p) for p in report['evidence']['aggregate'])}",
        f"- 参考(不合并): {', '.join(str(p) for p in report['evidence']['reference']) or '无'}",
        "",
    ]
    for run in report["runs"]:
        lines += [
            f"## 运行 {run['label']}",
            f"- run: `{run['run_dir']}`",
            f"- 基重建 elapsed_s: {_fmt(run['base_elapsed_s'], 2)}",
            f"- 总重建 elapsed_s: {_fmt(run['total_rebuild_elapsed_s'], 2)}",
            f"- 迭代型 loss 阶段 elapsed_s: {_fmt(run['loss_stage_elapsed_s'], 2)}",
            f"- 候选切点均摊省时估计 s: {_fmt(run['stage_candidate_saving_s'], 2)}",
            "",
            "| 阶段 | 类型 | elapsed_s | 占基重建 | 记录步 | 每步估计 s | 本轮范围 | 说明 |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
        for row in run["ledger"]:
            share = (f"{row['share_of_base'] * 100:.1f}%"
                     if row["share_of_base"] is not None else "null")
            lines.append(
                f"| {row['stage']} | {row['kind']} | {_fmt(row['elapsed_s'], 2)} | "
                f"{share} | {_show(row.get('steps_last'))} | "
                f"{_fmt(row.get('per_step_s_estimate'), 3)} | "
                f"{row['cut_scope']} | {row['reason'] or ''} |")
        lines += ["", "| 阶段 | phase | 记录步 | 点数 | 最终原始值 | 候选切点 | 可省步 | "
                      "末窗改善 | 剩余潜力 | 省时估计 s | 应用 | 状态 |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for stage in run["stages"]:
            for phase in stage.get("phases") or []:
                candidate = phase.get("candidate") or {}
                lines.append(
                    f"| {stage['name']} | {phase['phase']} | "
                    f"{phase['first_step']}–{phase['last_step']} | "
                    f"{phase['n_points']} | "
                    f"{_fmt(phase['final_value'], 6)} | "
                    f"{_show(candidate.get('cut_step'))} | "
                    f"{_show(candidate.get('saved_steps'))} | "
                    f"{_pct(phase.get('last_window_improvement_relative'))} | "
                    f"{_pct(candidate.get('lost_potential_relative'))} | "
                    f"{_fmt(candidate.get('saved_s_estimate'), 2)} | "
                    f"{candidate.get('application', '')} | {phase['status']} |")
        lines += ["", "缺口:"]
        lines += [f"- {gap}" for gap in run["gaps"]] or ["- 无"]
        lines.append("")
    lines += ["## 跨 run 合并候选（保守）", "",
              "| 阶段 | phase | 状态 | 覆盖 | 保守切点 | 各 run 切点 | 各 run 省时估计 |",
              "|---|---|---|---:|---:|---|---|"]
    for row in report["aggregate"]:
        cuts = "; ".join(f"{item['label']}: {item['cut_step']}"
                         for item in row["per_run"])
        savings = "; ".join(
            f"{item['label']}: {_fmt(item['saved_s_estimate'], 2)} s"
            for item in row["per_run"])
        lines.append(
            f"| {row['stage']} | {row['phase']} | {row['status']} | "
            f"{row['runs_with_plateau']}/{row['runs_required']} | "
            f"{row['conservative_cut_step']} | {cuts} | {savings} |")
    diagnostics = report.get("diagnostics") or {}
    counts = diagnostics.get("phase_status_counts") or {}
    closest = diagnostics.get("closest_to_plateau") or []
    lines += ["", "## 判定统计", "",
              f"- plateau: {counts.get('plateau', 0)}",
              f"- no_plateau: {counts.get('no_plateau', 0)}",
              f"- insufficient_points: {counts.get('insufficient_points', 0)}",
              f"- ledger_only: {counts.get('ledger_only', 0)}",
              "", "## 最接近平台的阶段（非候选）", "",
              "| run | 阶段 | phase | 点数 | 末窗步数 | 末窗改善 | 状态 |",
              "|---|---|---|---:|---:|---:|---|"]
    for row in closest:
        lines.append(
            f"| {row['run']} | {row['stage']} | {row['phase']} | "
            f"{row['n_points']} | {row['last_window_steps']} | "
            f"{_pct(row['last_window_improvement_relative'])} | {row['status']} |")
    lines += ["", "## 全局缺口", ""]
    lines += [f"- {gap}" for gap in report["gaps"]] or ["- 无"]
    lines.append("")
    return "\n".join(lines)


def run_converge(args) -> int:
    from .cli import DEFAULT_OUTPUT, _new_output_dir

    aggregate_paths = [Path(path).expanduser().resolve() for path in (args.run or [])]
    reference_paths = [Path(path).expanduser().resolve()
                       for path in (args.reference_run or [])]
    if not aggregate_paths and not reference_paths:
        print("ERROR: pass at least one --run or --reference-run", flush=True)
        return 2
    if not (0.0 < args.epsilon < 1.0):
        print("ERROR: --epsilon must be inside (0, 1)", flush=True)
        return 2
    if args.persist_steps <= 0:
        print("ERROR: --persist-steps must be positive", flush=True)
        return 2
    for path in aggregate_paths + reference_paths:
        if not path.is_dir():
            print(f"ERROR: run directory not found: {path}", flush=True)
            return 2
        if not (path / "run_report.json").is_file():
            print(f"ERROR: run_report.json not found: {path}", flush=True)
            return 2

    runs = [analyse_run(path, label=path.name, epsilon=args.epsilon,
                        persist_steps=args.persist_steps)
            for path in aggregate_paths]
    references = [analyse_run(path, label=path.name, epsilon=args.epsilon,
                              persist_steps=args.persist_steps)
                  for path in reference_paths]
    aggregate = aggregate_candidates(runs)
    all_runs = runs + references
    gaps = [f"{run['label']}: {gap}" for run in all_runs for gap in run["gaps"]]

    out_dir = (Path(args.output).expanduser().resolve()
               if args.output else _new_output_dir(DEFAULT_OUTPUT, "converge"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "parameters": {
            "epsilon": args.epsilon,
            "persist_steps": args.persist_steps,
            "sampling_note": "stage logs print objective every 20 steps; no interpolation",
            "time_note": ("saved_s_estimate = stage elapsed_s * saved_steps / "
                          "recorded steps; amortised, not measured per step"),
        },
        "evidence": {
            "aggregate": [str(path) for path in aggregate_paths],
            "reference": [str(path) for path in reference_paths],
        },
        "runs": all_runs,
        "aggregate": aggregate,
        "gaps": gaps,
        "output_dir": str(out_dir),
    }
    counts, closest = _phase_diagnostics(report)
    report["diagnostics"] = {
        "phase_status_counts": counts,
        "closest_to_plateau": closest,
        "near_plateau_budget_s": {
            run["label"]: _near_plateau_budget(run) for run in runs},
        "loss_stage_elapsed_s": {
            run["label"]: run["loss_stage_elapsed_s"] for run in runs},
        "per_step_s": {
            kind: sorted({stage["per_step_s_estimate"]
                          for run in runs for stage in run["stages"]
                          if stage["kind"] == kind
                          and stage["per_step_s_estimate"] is not None})
            for kind in (FIT_C, MOTION)
        },
    }

    figures: list[Path] = []
    if not getattr(args, "no_figures", False):
        for run in all_runs:
            for stage in run["stages"]:
                if stage["kind"] not in (FIT_C, MOTION):
                    continue
                name = f"{run['label']}_{stage['name']}".replace(" ", "_")
                figure = _plot_stage(out_dir / "figures" / f"{name}.png",
                                     run["label"], stage)
                if figure:
                    figures.append(figure)

    point_rows = _point_rows(report)
    point_fields = ["run", "stage", "kind", "tier", "phase", "phase_index",
                    "step", "value", "best_so_far", "gap_to_phase_final",
                    "relative_gap", "remaining_potential",
                    "remaining_potential_relative", "improvement_from_prev",
                    "improvement_per_step"]
    candidate_rows = [row for run in all_runs for row in _run_candidates(run)]
    candidate_fields = ["run", "stage", "phase", "application", "cut_step",
                        "saved_steps", "lost_potential_relative",
                        "raw_gap_relative", "saved_s_estimate"]
    csv_path = _write_csv(out_dir / "convergence.csv", point_fields, point_rows)
    candidates_csv = _write_csv(out_dir / "candidates.csv", candidate_fields,
                                candidate_rows)
    json_path = write_json(out_dir / "convergence.json", report)
    md_path = write_report(out_dir / "report.md", _markdown(report))
    html_path = write_report(out_dir / "report.html",
                             _html_report(report, figures))
    summary = {
        "output_dir": str(out_dir),
        "report": str(md_path),
        "html": str(html_path),
        "json": str(json_path),
        "csv": str(csv_path),
        "candidates_csv": str(candidates_csv),
        "figures": [str(path) for path in figures],
        "runs": [
            {
                "label": run["label"],
                "base_elapsed_s": run["base_elapsed_s"],
                "total_rebuild_elapsed_s": run["total_rebuild_elapsed_s"],
                "loss_stage_elapsed_s": run["loss_stage_elapsed_s"],
                "stage_candidate_saving_s": run["stage_candidate_saving_s"],
                "stage_candidates": [
                    {"stage": stage["name"],
                     **{key: (stage["stage_candidate"] or {}).get(key)
                        for key in ("phase", "cut_step", "saved_steps",
                                    "lost_potential_relative", "saved_s_estimate")}}
                    for stage in run["stages"] if stage["stage_candidate"]
                ],
            }
            for run in all_runs
        ],
        "aggregate": aggregate,
        "diagnostics": report["diagnostics"],
        "gaps": gaps,
        "note": "read-only analysis; no reconstruction was run and no run directory was modified",
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _fmt(value: Any, digits: int = 6) -> str:
    value = _num(value)
    return "null" if value is None else f"{value:.{digits}f}"


def _show(value: Any) -> str:
    return "null" if value is None else str(value)


def _pct(value: Any) -> str:
    value = _num(value)
    return "null" if value is None else f"{value * 100.0:.2f}%"
