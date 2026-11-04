"""Read-only adapter: turn an existing run directory into loss records.

Hard rules
----------
* never writes into the run directory, never re-runs any fitting stage;
* only reads files that a completed ``headrgbd reconstruct`` already produced;
* every number it reports carries its provenance tier;
* when the artefacts needed for a tier are absent, it records a *gap* instead of
  reconstructing a plausible-looking value.

What is genuinely available offline
-----------------------------------
``fit_c`` / ``motion_v2``  the per-step ``objective`` values are printed to the
stage logs (``logs/*.log`` lines ``{'step': .., 'objective': ..}``). These are the
real solver objectives; the per-term split is NOT in the artefacts and is
recorded as a gap.
``geometry_d.solve_field`` the ``report.json`` of each stage keeps ``field_logs``
with ``objective_before`` / ``objective_after``, ``free_vertices``,
``step_accepted``, ``shared_step_scale`` and (research_v2) the fixed-ray
diagnostics and line-search trials. Those are real recorded linearised objectives
and nonlinear fixed-ray energies.
``loop blend`` has no objective at all — feasibility trials only.
Diagnostics (MAE/RMSE/P95/coverage on real depth) require the prepared frames,
which only exist if the user passes ``--prepared``; otherwise diagnostics are
recorded as gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .objectives import LINEARISED, NONLINEAR_ENERGY, SOLVER_OBJECTIVE
from .recorder import IterationRecord, Recorder

FEATURE_PATTERN = re.compile(
    r"^\s*\{\s*'step'\s*:\s*(?P<step>-?\d+)\s*,\s*'phase'\s*:\s*'(?P<phase>[^']+)'\s*,"
    r"\s*'objective'\s*:\s*(?P<objective>[-+0-9.eE]+)\s*\}\s*$"
)


class AdapterError(RuntimeError):
    """Raised when the supplied directory cannot be analysed."""


@dataclass
class StageLog:
    name: str
    path: Path
    iterations: list[dict[str, Any]]

    @property
    def complete(self) -> bool:
        return bool(self.iterations)


def parse_feature_log(path: Path) -> list[dict[str, Any]]:
    """Extract the ``{'step':..,'objective':..}`` records printed by fit_c/motion_v2."""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = FEATURE_PATTERN.match(line)
        if not match:
            continue
        rows.append({
            "step": int(match.group("step")),
            "phase": match.group("phase"),
            "objective": float(match.group("objective")),
        })
    return rows


def discover_stage_logs(run_dir: Path) -> dict[str, StageLog]:
    """Map stage names to their log files without executing anything."""
    logs = Path(run_dir) / "logs"
    found: dict[str, StageLog] = {}
    if not logs.is_dir():
        return found
    for path in sorted(logs.glob("*.log")):
        stem = path.stem
        name = re.sub(r"^\d+_", "", stem)
        found[name] = StageLog(name=name, path=path, iterations=parse_feature_log(path))
    return found


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _field_log_entry(report: dict[str, Any], key: str) -> dict[str, Any] | None:
    logs = report.get("field_logs") or {}
    entry = logs.get(key)
    return entry if isinstance(entry, dict) else None


def _fusion_mode(value: Any) -> tuple[str | None, dict[str, Any]]:
    """Normalise the ``shape_fusion`` field, which v0.4.0 writes in two shapes.

    ``fit_c`` / ``identity_v2`` write a plain string (``'joint'`` / ``'guarded'``);
    ``fit_d`` writes a block (``{'mode': 'joint', 'front_hard_locked_vertices': 0,
    'front_downweighted_vertices': 2522, ...}``). A reader that assumes a string
    either crashes or -- worse -- stringifies the dict into a stage label, so both
    shapes go through this function and nothing else.
    """
    if isinstance(value, str):
        mode = value.strip().lower() or None
        return (mode if mode in ("joint", "guarded") else None), {}
    if isinstance(value, dict):
        raw = value.get("mode")
        mode = raw.strip().lower() if isinstance(raw, str) else None
        details = {key: value[key] for key in (
            "joint_front_confidence", "identity_views", "shape_frames",
            "front_hard_locked_vertices", "front_downweighted_vertices",
        ) if key in value}
        return (mode if mode in ("joint", "guarded") else None), details
    return None, {}


def chain_fusion(run_dir: Path) -> str | None:
    """Which shape-fusion mode the CHAIN asked for, read from artefacts only.

    ``run_report.json`` does not record the flag, so the mode is taken from the
    shared-identity stage, which is the first stage whose report carries it, and
    cross-checked against fit_d's block. A disagreement is returned as ``None``:
    the tool must not pick a winner between two contradictory artefacts.
    """
    run_dir = Path(run_dir)
    seen: dict[str, str] = {}
    for name in ("guarded", "dense"):
        report = _load_json(run_dir / "reconstruction" / name / "report.json")
        if not report:
            continue
        mode, _ = _fusion_mode(report.get("shape_fusion"))
        if mode:
            seen[name] = mode
    if not seen:
        return None
    modes = set(seen.values())
    return modes.pop() if len(modes) == 1 else None


def _missing_objective_reason(path: Path) -> str:
    """Say why a stage log has no objective, instead of one generic sentence.

    Three different situations hide behind "no per-step objective line", and they
    mean different things for the loss table:

    * the stage is not an optimiser at all (``prepare_observations``);
    * the stage is a ``solve_field`` run: its log prints one acceptance line per
      outer (``outer=.. free=.. alpha=.. update=..``) and the objective lives in
      ``field_logs`` instead, so the solver_objective tier is absent by design and
      the value is reported under the linearised tier;
    * a genuine loss of coverage that the run failed to persist.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "no per-step objective line in the stage log (file unreadable)"
    if "outer=" in text:
        return ("this stage log prints one acceptance line per outer (free vertices / alpha / update) but "
                "no objective value: a solve_field stage keeps its objectives in field_logs, so the "
                "solver_objective tier does not exist for it by design and its values are reported under "
                "the linearised tier")
    if "candidate" in text or "excluded_frames" in text:
        return ("this stage is not an optimiser (no objective is defined for it): it only records frame "
                "candidacy and preparation counts")
    return "no per-step objective line in the stage log"


def ingest_solver_logs(recorder: Recorder, run_dir: Path) -> dict[str, int]:
    """Add fit_c / motion_v2 per-step solver objectives found in the run logs."""
    counts: dict[str, int] = {}
    for name, log in discover_stage_logs(run_dir).items():
        if not log.complete:
            recorder.gap(
                name,
                _missing_objective_reason(log.path),
                expected_tier=SOLVER_OBJECTIVE,
                source_hint=str(log.path),
            )
            counts[name] = 0
            continue
        for row in log.iterations:
            recorder.add(
                IterationRecord(
                    stage=name,
                    iteration=row["step"],
                    tier=SOLVER_OBJECTIVE,
                    phase=row["phase"],
                    value=row["objective"],
                    source=SOLVER_OBJECTIVE,
                    notes=[
                        "solver objective recorded by the delivery run; per-term split is unavailable offline",
                    ],
                )
            )
        recorder.declare_stage(
            name,
            has_iterations=True,
            notes=[
                f"real per-step solver objective from {log.path.name} ({len(log.iterations)} records)",
                "term decomposition not persisted by the run: recorded as gap, not invented",
            ],
        )
        counts[name] = len(log.iterations)
    return counts


INACTIVE_STATUSES = ("not_executed", "skipped_no_free_vertices")


def _log_stage_name(log: dict[str, Any], fallback: str) -> str:
    """Prefer the stage name the delivery run itself wrote.

    v0.4.0 reuses the same ``field_logs`` key for different stages (``regularized``
    holds either ``guarded_regularized`` or ``joint_confidence_weighted_regularized``),
    so the key is not a reliable stage identifier; the name inside the log is.
    """
    name = log.get("stage")
    if isinstance(name, str) and name:
        return name
    history = log.get("history")
    if isinstance(history, list):
        for record in history:
            candidate = record.get("stage") if isinstance(record, dict) else None
            if isinstance(candidate, str) and candidate:
                return candidate
    return fallback


def _outer_groups(entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Split one field log into its own rounds plus any nested sub-round.

    v0.4.0 stores the mid-frequency round INSIDE the low-frequency field log
    (``field_logs['front_anchor_regularized']['mid_frequency']``). A reader that
    only walks ``history`` silently drops that whole round, which is exactly the
    round ``--mid-rounds`` controls.
    """
    groups: list[tuple[str, dict[str, Any]]] = [("", entry)]
    for key, value in entry.items():
        if key == "history":
            continue
        if isinstance(value, dict) and isinstance(value.get("history"), list):
            groups.append((key, value))
    return groups


def _inactive_status(log: dict[str, Any]) -> str | None:
    for field in ("status", "stop_reason"):
        value = log.get(field)
        if isinstance(value, str) and value in INACTIVE_STATUSES:
            return value
    return None


def ingest_field_stage(recorder: Recorder, stage_path: Path, stage_name: str,
                       log_key: str, *, prefix: str = "dense.",
                       extra_notes: Iterable[str] = ()) -> bool:
    """Add linearised objectives + fixed-ray nonlinear energies of one solve_field run.

    Handles the v0.4.0 layout: nested mid-frequency round, stages that were
    bypassed (``--fast``) or that found no free vertex, and the per-trial
    line-search candidates that the acceptance gate rejected.
    """
    report = _load_json(Path(stage_path) / "report.json")
    label = f"{prefix}{stage_name}"
    if report is None:
        recorder.gap(
            label,
            "stage report.json missing; no iteration state to read",
            expected_tier=LINEARISED,
            source_hint=str(Path(stage_path) / "report.json"),
        )
        return False
    entry = _field_log_entry(report, log_key)
    if entry is None:
        recorder.gap(
            label,
            f"field_logs[{log_key!r}] missing from report.json",
            expected_tier=LINEARISED,
            source_hint=str(Path(stage_path) / "report.json"),
        )
        return False
    if entry.get("reused"):
        # v0.5.0 regression default (--reuse-field): the second solve is skipped and the
        # front-anchor low/mid field is carried over. The entry still exists, so a reader
        # that only checks for emptiness would call this stage "solved".
        recorder.declare_stage(
            label,
            has_iterations=False,
            notes=[
                f"field_logs[{log_key!r}] reports stage={entry.get('stage')!r} with reused=true: "
                f"--reuse-field carried the front-anchor low/mid field over instead of solving again",
                *extra_notes,
            ],
        )
        recorder.gap(
            label,
            f"field reuse, not a solve: field_logs[{log_key!r}] has stage={entry.get('stage')!r}, "
            f"reused=true and an empty history. Its maximum_absolute_displacement_mm="
            f"{entry.get('maximum_absolute_displacement_mm')} is inherited from the field it reused, "
            f"so this stage has no objective of its own and no curve to compare with the stage it replaced",
            expected_tier=LINEARISED,
            source_hint=str(Path(stage_path) / "report.json"),
        )
        return False
    produced = False
    for suffix, log in _outer_groups(entry):
        fallback = stage_name if not suffix else f"{prefix}{suffix}"
        name = prefix + _log_stage_name(log, fallback.removeprefix(prefix))
        produced |= _ingest_one_field_series(recorder, name, log, Path(stage_path), suffix,
                                             extra_notes=extra_notes)
    return produced


def _ingest_one_field_series(recorder: Recorder, stage_name: str, log: dict[str, Any],
                             stage_path: Path, suffix: str,
                             *, extra_notes: Iterable[str] = ()) -> bool:
    inactive = _inactive_status(log)
    history = log.get("history")
    if not isinstance(history, list):
        history = []
    if not history and "objective_before" in log:
        # Older runs stored a single outer step directly on the field log.
        history = [log]
    # v0.2.0 recorded a starved solve as `free_vertices=0` with no objective and no
    # status field. That is not a measurement gap: the stage ran and could not move.
    if history and all(record.get("free_vertices") == 0 for record in history) and all(
            record.get("objective_before") is None for record in history):
        recorder.declare_stage(stage_name, has_iterations=False,
                               notes=["every outer step had free_vertices=0", *extra_notes])
        recorder.gap(
            stage_name,
            "every outer step recorded free_vertices=0 with no objective: the free set was empty, so the "
            "stage inherited the previous field instead of solving one (this run predates the explicit "
            "'skipped_no_free_vertices' status)",
            expected_tier=LINEARISED,
            source_hint=str(stage_path / "report.json"),
        )
        return False
    if not history:
        if inactive:
            recorder.gap(
                stage_name,
                f"stage recorded status={inactive!r}: it produced no objective because it solved nothing, "
                f"not because it converged (stop_reason={log.get('stop_reason')!r})",
                expected_tier=LINEARISED,
                source_hint=str(stage_path / "report.json"),
            )
        else:
            recorder.gap(
                stage_name,
                "field log has neither history nor an inactive status",
                expected_tier=LINEARISED,
                source_hint=str(stage_path / "report.json"),
            )
        recorder.declare_stage(
            stage_name,
            has_iterations=False,
            notes=[f"status={inactive or 'unknown'} from field_logs (not a zero-valued curve)",
                   *extra_notes],
        )
        return False
    recorder.declare_stage(
        stage_name,
        has_iterations=True,
        notes=[
            f"outer-step records transcribed from field_logs[{suffix or 'history'!r}] of {stage_path.name}",
            "linearised objective is per-outer and re-linearised between outers, so it is NOT monotone "
            "across outer indices even when every accepted step improves its own linearisation",
            *extra_notes,
        ],
    )
    for index, record in enumerate(history):
        outer = record.get("outer", index)
        before = record.get("objective_before")
        after = record.get("objective_after")
        recorder.add(
            IterationRecord(
                stage=stage_name,
                iteration=int(outer),
                tier=LINEARISED,
                phase=record.get("stage"),
                value=before if before is not None else None,
                decision="initial",
                source="recorded",
                notes=["linearised quadratic under frozen correspondences (before the shared line search)"],
                extras={"solver_status": record.get("solver_status"),
                        "free_vertices": record.get("free_vertices")},
            )
        )
        if after is not None:
            recorder.add(
                IterationRecord(
                    stage=stage_name,
                    iteration=int(outer),
                    tier=LINEARISED,
                    value=after,
                    decision="accepted" if record.get("step_accepted") else "rejected",
                    step_scale=record.get("shared_step_scale"),
                    source="recorded",
                    notes=["candidate objective after the shared halving line search"],
                    extras={"objective_before": before},
                )
            )
        # Candidates the acceptance gate threw away. Without these the report can
        # only show that a step was small, never that a BETTER step was refused.
        for trial in record.get("line_search_trials") or []:
            if not isinstance(trial, dict) or trial.get("reason") is None:
                continue
            objective = trial.get("objective")
            if objective is not None:
                recorder.add(
                    IterationRecord(
                        stage=stage_name,
                        iteration=int(outer),
                        tier=LINEARISED,
                        value=float(objective),
                        decision="rejected_trial",
                        step_scale=trial.get("alpha"),
                        source="recorded",
                        notes=[f"line-search candidate alpha={trial.get('alpha')} rejected by "
                               f"gate {trial.get('reason')!r}"],
                        extras={"alpha": trial.get("alpha"), "gate_reason": trial.get("reason"),
                                "geometry_valid": trial.get("geometry_valid"),
                                "fixed_ray_mse_mm2": trial.get("fixed_ray_mse_mm2"),
                                "missing_rays": trial.get("missing_rays")},
                    )
                )
            mse = trial.get("fixed_ray_mse_mm2")
            if mse is not None:
                recorder.add(
                    IterationRecord(
                        stage=stage_name,
                        iteration=int(outer),
                        tier=NONLINEAR_ENERGY,
                        value=float(mse),
                        decision="rejected_trial",
                        step_scale=trial.get("alpha"),
                        source="recorded",
                        notes=[f"fixed-ray MSE of the rejected alpha={trial.get('alpha')} candidate"],
                        extras={"alpha": trial.get("alpha"), "gate_reason": trial.get("reason"),
                                "missing_rays": trial.get("missing_rays")},
                    )
                )
        diagnostic = record.get("fixed_ray_diagnostic")
        if diagnostic:
            allowance = diagnostic.get("missing_allowance")
            cap = diagnostic.get("missing_cap")
            for tag, value_key, missing_key in (
                ("before", "before_mse_mm2", "missing_before"),
                ("after", "after_mse_mm2", "missing_after"),
            ):
                value = diagnostic.get(value_key)
                if value is None:
                    continue
                missing = diagnostic.get(missing_key)
                note = ("fixed-ray penalised MSE (strict-coverage) recomputed by ray casting, not by the "
                        "solver")
                if tag == "after" and isinstance(allowance, (int, float)) and allowance > 0:
                    # v0.5.0 replaced the zero-tolerance rule with a per-stage allowance
                    # (--coverage-allowance-ratio). Whether a step is accepted therefore
                    # depends on this number, so it must travel with the value.
                    within = isinstance(missing, (int, float)) and missing <= allowance
                    note += (f"; coverage {missing}/{diagnostic.get('ray_count')} missing vs an allowance "
                             f"of {allowance} ({'within' if within else 'over'} it, cap {cap}): with the "
                             f"pre-0.5.0 zero-tolerance rule this step would have been refused")
                recorder.add(
                    IterationRecord(
                        stage=stage_name,
                        iteration=int(outer),
                        tier=NONLINEAR_ENERGY,
                        value=float(value),
                        coverage=None,
                        decision=tag,
                        source="recorded",
                        notes=[note],
                        extras={
                            "missing_rays": missing,
                            "ray_count": diagnostic.get("ray_count"),
                            "missing_allowance": allowance,
                            "missing_cap": cap,
                            "policy": diagnostic.get("policy", "strict-coverage"),
                            "missing_penalty_mm": diagnostic.get("missing_penalty_mm", 8.0),
                        },
                    )
                )
        verdict = record.get("safety") or {}
        if verdict:
            recorder.add(
                IterationRecord(
                    stage=stage_name,
                    iteration=int(outer),
                    tier="diagnostic",
                    value=1.0 if verdict.get("reason") is None else 0.0,
                    decision="accepted" if record.get("step_accepted") else "rejected",
                    source="recorded",
                    notes=["1.0 = geometric safety passed, 0.0 = rejected; not a loss value"],
                    extras={"safety": verdict},
                )
            )
    return True


def ingest_identity_stage(recorder: Recorder, stage_path: Path, stage_name: str,
                          *, chain_mode: str | None = None) -> bool:
    """Read identity_v2's report, which stores a ``stages`` LIST rather than field_logs.

    Two things this must not lose.

    1. A solve that reports ``skipped_no_free_vertices`` did run, and produced a
       field of exactly zero. That is a finding, not a missing measurement.
    2. v0.4.0 gives this module two branches (hard ``locked=`` vs ``soft_lock``
       confidence) and the pipeline only forwards the flag to ``fit_d``. The branch
       is therefore read from the report, and a branch that disagrees with the rest
       of the chain is recorded as an explicit inconsistency instead of being
       silently averaged into the same stage.
    """
    stage_path = Path(stage_path)
    report = _load_json(stage_path / "report.json")
    if report is None:
        recorder.gap(stage_name, "stage report.json missing",
                     expected_tier=LINEARISED, source_hint=str(stage_path / "report.json"))
        return False
    stages = report.get("stages")
    if not isinstance(stages, list) or not stages:
        recorder.gap(stage_name, "report.json has no `stages` list",
                     expected_tier=LINEARISED, source_hint=str(stage_path / "report.json"))
        return False
    mode, _ = _fusion_mode(report.get("shape_fusion"))
    confidence = report.get("joint_front_confidence")
    locked = report.get("front_locked_vertices")
    downweighted = report.get("front_downweighted_vertices")
    soft = isinstance(downweighted, (int, float)) and downweighted > 0
    if chain_mode and mode and chain_mode != mode:
        # The chain asked for one mechanism and this stage ran the other. Every
        # number below then belongs to the OTHER objective, so it gets its own gap.
        recorder.gap(
            stage_name,
            f"mode mismatch: the chain reports shape_fusion={chain_mode!r} but this stage ran "
            f"shape_fusion={mode!r} (front_locked_vertices={locked}, "
            f"front_downweighted_vertices={downweighted}). Its term set is the "
            f"{'hard-locked' if mode == 'guarded' else 'confidence-weighted'} one, so its objective is "
            f"not the {'joint' if chain_mode == 'joint' else 'guarded'} low-frequency objective. Causes "
            f"are one-sided: either the stage flag was not forwarded (pipeline.py:202-217 does not pass "
            f"--shape-fusion to identity_v2) or the stage fell back on purpose -- the report alone cannot "
            f"tell them apart",
            expected_tier=LINEARISED,
            source_hint=str(stage_path / "report.json"),
        )
    # One figure per solve: the two solves have different data frames and different
    # lock semantics, so joining them on one axis would draw a fake zig-zag.
    # The solve name is NOT at the top of the entry in v0.4.0 (only `stop_reason`
    # and `settings` are); it is inside history[0]['stage'], which is why
    # _log_stage_name is used instead of a direct get.
    solves = [
        _log_stage_name(entry, f"solve{index}") if isinstance(entry, dict) else f"solve{index}"
        for index, entry in enumerate(stages)
    ]
    split = len(set(solves)) > 1
    produced = False
    per_solve_gaps = 0
    for entry, solve in zip(stages, solves):
        if not isinstance(entry, dict):
            continue
        label = f"{stage_name}.{solve}" if split and solve else stage_name
        record = (entry.get("history") or [{}])[0] if entry.get("history") else {}
        status = (entry.get("status") or entry.get("stop_reason")
                  or record.get("status") or record.get("stop_reason"))
        history = [item for item in (entry.get("history") or []) if isinstance(item, dict)]
        inferred = bool(history) and status is None and all(
            item.get("free_vertices") == 0 and item.get("objective_before") is None for item in history)
        if inferred:
            # v0.2.0 has no status field: an empty free set is only visible as
            # free_vertices=0 with no objective. That is still "the stage solved
            # nothing", not "the stage converged".
            status = "skipped_no_free_vertices (inferred: free_vertices=0, no objective)"
        if isinstance(status, str) and (status in INACTIVE_STATUSES or inferred):
            per_solve_gaps += 1
            details = [f"maximum_absolute_displacement_mm={entry.get('maximum_absolute_displacement_mm')}"]
            if record.get("reason"):
                details.insert(0, f"reason={record['reason']!r}")
            if soft:
                details.append(f"front vertices downweighted={downweighted} (they stay in the free set)")
            elif locked is not None:
                details.append(f"front vertices hard-locked={locked}, all removed from the free set")
            recorder.gap(
                label,
                f"this solve ({solve!r}) reported status={status!r}: the free set was empty, so it "
                f"returned the initial field and no objective exists ({'; '.join(details)})",
                expected_tier=LINEARISED,
                source_hint=str(stage_path / "report.json"),
            )
            recorder.declare_stage(label, has_iterations=False,
                                   notes=[f"status={status!r} recorded by the run itself"
                                          if not inferred else
                                          "no status field in this run; an all-zero free set with no "
                                          "objective is the same finding"])
            continue
        count = 0
        for index, record in enumerate(entry.get("history") or []):
            outer = record.get("outer", index)
            for tag, key in (("initial", "objective_before"), ("accepted", "objective_after")):
                value = record.get(key)
                if value is None:
                    continue
                count += 1
                recorder.add(
                    IterationRecord(
                        stage=label,
                        iteration=int(outer),
                        tier=LINEARISED,
                        phase=record.get("stage"),
                        value=float(value),
                        decision=tag,
                        step_scale=record.get("shared_step_scale"),
                        source="recorded",
                        notes=[f"linearised quadratic recorded in identity_v2's `stages` list "
                               f"(solve {solve!r}, fusion={mode or 'unknown'})"],
                        extras={
                            "free_vertices": record.get("free_vertices"),
                            "solve_stage": solve,
                            "shape_fusion": mode,
                            "soft_lock": soft,
                            "joint_front_confidence": confidence,
                            "front_locked_vertices": locked,
                            "front_downweighted_vertices": downweighted,
                        },
                    )
                )
        if count:
            produced = True
            recorder.declare_stage(
                label,
                has_iterations=True,
                notes=[
                    f"solve {solve!r} of identity_v2, fusion={mode or 'unknown'}",
                    (f"soft_lock: front-protected vertices keep {confidence} of their depth weight but "
                     f"stay in the free set ({downweighted} vertices)"
                     if soft else
                     f"hard lock: {locked} front-protected vertices are removed from the free set, so "
                     f"free_vertices already excludes them"),
                    "the two solves of this stage are different objectives and are never summed",
                ],
            )
    if produced:
        return True
    if per_solve_gaps:
        # Each solve already carries its own, individually named gap. A second
        # aggregate gap would make the same absence look like two findings.
        moved = max((float(entry.get("maximum_absolute_displacement_mm") or 0.0) for entry in stages
                     if isinstance(entry, dict)), default=0.0)
        recorder.declare_stage(stage_name, has_iterations=False,
                               notes=[f"every solve reported an inactive status; "
                                      f"maximum_absolute_displacement_mm={moved:g}"])
        return False
    return True


def _fit_c_notes(report: dict[str, Any]) -> list[str]:
    """v0.4.0 joint fit_c records two displacement figures; neither is a loss.

    They are kept as stage notes (not records) so the four tiers stay clean: the
    tool has no per-term split for an Adam run and must not fabricate one.
    """
    mode, _ = _fusion_mode(report.get("shape_fusion"))
    notes = [f"shape_fusion={mode or 'unknown'}"]
    for key in ("side_depth_weight", "joint_shape_prior", "joint_front_confidence"):
        if key in report and report[key] is not None:
            notes.append(f"{key}={report[key]}")
    update = report.get("joint_anchor_update_mm")
    delta = report.get("joint_shape_delta_from_anchor")
    if update is not None:
        notes.append(f"joint_anchor_update_mm={update:g} (contains pose and expression freedom, not a "
                     f"shape figure)")
    if delta is not None:
        notes.append(f"joint_shape_delta_from_anchor={delta:g} (raw FLAME coefficient distance moved "
                     f"away from the front anchor; held only by --joint-shape-prior)")
    return notes


def ingest_skills(recorder: Recorder, run_dir: Path, *, chain_mode: str | None = None) -> dict[str, bool]:
    """Read every stage directory of the delivery chain, tolerating absences."""
    run_dir = Path(run_dir)
    reconstruction = run_dir / "reconstruction"
    configuration = (run_report(run_dir) or {}).get("configuration") or {}
    chain = configuration.get("chain")
    dense_report = _load_json(reconstruction / "dense" / "report.json") or {}
    dense_mode, dense_details = _fusion_mode(dense_report.get("shape_fusion"))
    dense_notes: list[str] = []
    fast = dense_report.get("fast_path")
    if isinstance(fast, dict) and fast:
        dense_notes.append(
            f"fast_path.enabled={fast.get('enabled')}: dense_residual={fast.get('dense_residual')!r}, "
            f"mid_frequency_rounds={fast.get('mid_frequency_rounds')}, "
            f"regularized_evidence={fast.get('regularized_evidence')!r}"
        )
    loop = dense_report.get("loop_base")
    change = dense_report.get("dense_change_from_interpolated_regularized_mm")
    move = loop.get("maximum_move_from_linear_mm") if isinstance(loop, dict) else None
    if isinstance(change, (int, float)) and isinstance(move, (int, float)):
        same = math.isclose(float(change), float(move), rel_tol=0.0, abs_tol=1e-12)
        dense_notes.append(
            f"delivered surface moved {float(change):.6f} mm away from the interpolated regularized "
            f"surface; the Loop base alone accounts for {float(move):.6f} mm "
            f"({'numerically identical to the delivered change' if same else 'a distinct amount'})"
            + (" — the dense residual therefore contributed nothing, so the delivered surface IS the "
               "Loop base" if same else "")
        )
        if isinstance(loop, dict):
            dense_notes.append(
                f"loop blend: target={loop.get('target_blend')}, adopted={loop.get('blend')}, "
                f"fallback_reason={loop.get('fallback_reason')!r}, trials={len(loop.get('nonlinear_trials') or [])}"
            )
    if dense_mode:
        locked = dense_details.get("front_hard_locked_vertices")
        down = dense_details.get("front_downweighted_vertices")
        frames = dense_details.get("shape_frames")
        dense_notes.append(
            f"this stage directory ran shape_fusion={dense_mode!r}"
            + (f" over shape_frames={frames}" if frames else "")
            + (f"; front_hard_locked_vertices={locked}" if locked is not None else "")
            + (f"; front_downweighted_vertices={down}" if down is not None else "")
        )
        if isinstance(down, (int, float)) and down:
            dense_notes.append(
                "soft_lock in force: the front-protected vertices stay in the free set, so free_vertices "
                "counts them even though they keep only --joint-front-confidence of their depth weight"
            )
        if chain_mode and chain_mode != dense_mode:
            recorder.gap(
                "dense",
                f"mode mismatch: chain shape_fusion={chain_mode!r} but the dense stage ran "
                f"{dense_mode!r}",
                expected_tier=LINEARISED,
                source_hint=str(reconstruction / "dense" / "report.json"),
            )
    field_stages = [
        ("front_anchor_low_frequency", reconstruction / "dense", "front_anchor_regularized"),
        ("guarded_regularized", reconstruction / "dense", "regularized"),
        ("front_anchor_dense", reconstruction / "dense", "front_anchor_dense"),
        ("guarded_dense", reconstruction / "dense", "dense"),
    ]
    status: dict[str, bool] = {}
    # identity_v2 is handled separately: it has real solver logs in a different shape.
    identity_dir = reconstruction / "low_frequency"
    if identity_dir.is_dir():
        status["low_frequency_shared_identity"] = ingest_identity_stage(
            recorder, identity_dir, "low_frequency_shared_identity", chain_mode=chain_mode)
    else:
        lean = chain == "lean"
        recorder.gap(
            "low_frequency_shared_identity",
            ("stage intentionally not run: chain='lean' (the 0.5.0 default) drops the standalone "
             "low-frequency field because the joint fit already ends with its own motion phase and "
             "fit_d rebuilds the low/mid field from the fitted pose. This is a chain decision, not "
             "missing data" if lean else "stage directory not present in this run"),
            expected_tier=LINEARISED,
            source_hint=str(reconstruction / "low_frequency"),
        )
        recorder.declare_stage(
            "low_frequency_shared_identity",
            has_iterations=False,
            notes=[f"chain={chain!r} from run_report.configuration; this stage is outside that chain"],
        )
        status["low_frequency_shared_identity"] = False
    # fit_c writes no field_logs at all: it is an Adam run whose only loss record is
    # the per-step objective already ingested from logs/.
    for name, path in (("identity_front_anchor", reconstruction / "front_anchor"),
                       ("shared_identity", reconstruction / "guarded")):
        if not path.is_dir():
            recorder.gap(name, "stage directory not present in this run",
                         expected_tier=LINEARISED, source_hint=str(path))
            continue
        report = _load_json(path / "report.json") or {}
        mode, _ = _fusion_mode(report.get("shape_fusion"))
        label = f"{name}_{mode}" if mode else name
        recorder.gap(
            label,
            "fit_c is an Adam run: its only persisted loss is the per-step objective read from logs/, "
            "and its report.json carries no per-term split, so no linearised curve exists for this stage",
            expected_tier=LINEARISED,
            source_hint=str(path / "report.json"),
        )
        recorder.stage_meta.setdefault(label, {"has_iterations": False, "notes": []})["notes"].extend(
            _fit_c_notes(report))
    for name, path, key in field_stages:
        label = f"dense.{name}"
        if not path.is_dir():
            recorder.gap(label, "stage directory not present in this run",
                         expected_tier=LINEARISED, source_hint=str(path))
            status[label] = False
            continue
        status[label] = ingest_field_stage(recorder, path, name, key, prefix="dense.",
                                           extra_notes=dense_notes)
    if chain == "lean":
        # The second motion pass is dropped by the lean chain, so this log is the only
        # motion solve in the run and fit_d consumes its output directly.
        meta = recorder.stage_meta.get("fixed_identity_motion")
        if meta is not None:
            meta["notes"].append(
                "chain='lean': this is the run's ONLY motion pass; the standalone "
                "motion_reestimate_after_identity stage is not executed, so its motion is not "
                "re-optimised after the low/mid field is rebuilt"
            )
    return status


def run_manifest(run_dir: Path) -> dict[str, Any] | None:
    return _load_json(Path(run_dir) / "delivery" / "manifest.json")


def run_report(run_dir: Path) -> dict[str, Any] | None:
    return _load_json(Path(run_dir) / "run_report.json")


def version_skew(run_dir: Path) -> dict[str, Any]:
    """Compare the versions the run directory states about itself.

    A run can be produced by code newer than the version stamped into its own
    manifest (the manifests are written from a template, the reports from the code).
    Reading a stage's behaviour off the stamp instead of off the stage report is how
    a v0.4.0 stage gets described with v0.3.0 semantics, so the skew is surfaced --
    and it is detected from FIELDS the reports actually contain, never from a guess.
    """
    run_dir = Path(run_dir)
    stated: dict[str, str] = {}
    manifest = run_manifest(run_dir) or {}
    version = (manifest.get("software") or {}).get("version")
    if version:
        stated["delivery/manifest.json"] = str(version)
    report = run_report(run_dir) or {}
    if report.get("project_version"):
        stated["run_report.json"] = str(report["project_version"])
    dense = _load_json(run_dir / "reconstruction" / "dense" / "report.json") or {}
    if dense.get("project_version"):
        stated["reconstruction/dense/report.json"] = str(dense["project_version"])
    # v0.4.0 behaviour markers: fields that only exist after the joint-fusion release.
    markers: dict[str, str] = {}
    for name in ("front_anchor", "guarded", "low_frequency", "dense"):
        stage = _load_json(run_dir / "reconstruction" / name / "report.json") or {}
        for key in ("shape_fusion", "joint_shape_prior", "joint_anchor_update_mm",
                    "joint_shape_delta_from_anchor", "joint_front_confidence",
                    "front_downweighted_vertices"):
            if key in stage and key not in markers:
                markers[key] = name
    values = set(stated.values())
    consistent = len(values) <= 1
    note = "run directory states one version" if consistent else (
        "the run directory states several versions; take each stage's semantics from the stage "
        "report itself, not from the stamp")
    if markers and consistent and values and all(value.startswith("0.3") for value in values):
        note = (f"every stamp says {sorted(values)[0]} but the stage reports contain v0.4.0-only fields "
                f"({sorted(markers)}): the stamp understates the code that produced this run, so the "
                f"stage reports are the authority")
    return {
        "stated": stated,
        "consistent": consistent,
        "v0_4_markers": markers,
        "note": note,
    }


def describe_inputs(run_dir: Path) -> dict[str, Any]:
    """What the tool found, without reading anything outside the run directory."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise AdapterError(f"run directory does not exist: {run_dir}")
    manifest = run_manifest(run_dir)
    report = run_report(run_dir)
    logs = discover_stage_logs(run_dir)
    reconstruction = run_dir / "reconstruction"
    configuration = (report or {}).get("configuration") or {}
    return {
        "run_dir": str(run_dir),
        "run_report": report is not None,
        "delivery_manifest": manifest is not None,
        "software_version": (manifest or {}).get("software", {}).get("version"),
        "algorithm": (manifest or {}).get("software", {}).get("algorithm"),
        "chain": configuration.get("chain"),
        "configuration": {
            key: configuration.get(key)
            for key in ("chain", "shape_fusion", "fast", "postfit", "mid_rounds", "reuse_field",
                        "information_measure", "support_min_information", "coverage_allowance_ratio",
                        "loop_target_blend", "loop_actual_blend", "bending_length_mm",
                        "low_bending_length_mm", "steps", "front_steps", "motion_steps",
                        "fixed_ray_policy")
            if key in configuration
        },
        "chain_shape_fusion": chain_fusion(run_dir),
        "versions": version_skew(run_dir),
        "stage_logs": {name: len(log.iterations) for name, log in logs.items()},
        "stage_dirs": sorted(p.name for p in reconstruction.iterdir()) if reconstruction.is_dir() else [],
        "prepared": (run_dir / "prepared" / "sequence.json").is_file(),
        "note": "read-only: nothing in the run directory is modified or re-executed",
    }
