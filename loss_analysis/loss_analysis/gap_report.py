"""Per-stage distance to the layered termination target, and the next move.

Two layers are reported separately and never merged into one gap:

* **solver layer** -- the termination rules the delivery source really defines.
  ``fit_c`` / ``motion_v2`` stop on a fixed step budget and define no
  convergence tolerance, so the report says exactly that; ``solve_field`` uses
  the recorded fixed-ray acceptance band; ``fit_unified`` uses the step /
  energy / gradient tolerances in ``head_rgbd/neck_fit.py``.
* **acceptance layer** -- the delivery gates with numeric margins (connection
  20 deg, protection 0.002 mm, upper-band density, topology, marker semantics,
  split budget, self-contained sidecars).

A gap is signed: positive means margin still available, negative means the
value is over the target.  Booleans stay booleans.  Missing values stay null.

The dominant gap (first hard failure, otherwise the smallest remaining margin)
is mapped to at most two whitelisted parameters of the offline optimiser.  No
parameter is proposed when the failing gate is not controlled by the whitelist
(for example a protection failure or a density increase).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import adapter, neck_adapter
from .html_report import render_page, table, _fmt, _text
from .plots import write_json, write_report

GAP_SCHEMA = "loss_analysis.gaps.v1"

CONNECTION_LIMIT_DEG = 20.0
PROTECTION_LIMIT_MM = 0.002
DENSE_MSE_FACTOR = 1.02
DENSE_MSE_ADD_MM2 = 0.002
NECK_TOL_STEP_MM = 1e-3
NECK_TOL_ENERGY_REL = 1e-8
NECK_TOL_GRAD_REL = 1e-9


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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


@dataclass
class GapEntry:
    key: str
    stage: str
    group: str                 # solver | acceptance | diagnostic
    tier: str
    metric: str
    current: float | bool | None
    target: float | None
    kind: str                  # max | tol | budget | boolean | none
    gap: float | None
    status: str
    source: str
    unit: str = ""
    hard: bool = False
    priority: int = 9
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "group": self.group,
            "tier": self.tier,
            "metric": self.metric,
            "current": self.current,
            "target": self.target,
            "kind": self.kind,
            "gap": self.gap,
            "status": self.status,
            "unit": self.unit,
            "hard": self.hard,
            "priority": self.priority,
            "source": self.source,
            "notes": self.notes,
        }


def _max_gap(current: float | None, limit: float | None) -> float | None:
    if current is None or limit is None:
        return None
    return float(limit - current)


def _tol_gap(current: float | None, tol: float | None) -> float | None:
    if current is None or tol is None:
        return None
    return float(current - tol)


def _tol_status(value: float | None, tol: float, *, converged: bool,
                stop_reason: str | None) -> str:
    if value is None:
        return "unknown"
    if value <= tol:
        return "pass"
    if stop_reason == "energy_stationary":
        return "stationary"
    if stop_reason == "max_iter":
        return "budget_limited"
    if converged:
        return "converged_elsewhere"
    return "stopped"


def _stop_status(converged: bool, stop_reason: str | None) -> str:
    if stop_reason == "max_iter":
        return "budget_limited"
    if stop_reason == "energy_stationary":
        return "stationary"
    return "converged" if converged else "stopped"


def solver_log_entries(run_dir: Path | None) -> tuple[list[GapEntry], list[dict[str, str]]]:
    """Step-budget terminal state of the Adam stages that persist a log."""
    if run_dir is None:
        return [], [{
            "stage": "fit_c/motion_v2",
            "reason": "no base run directory supplied: the per-step solver logs were not read",
            "source": "",
        }]
    run_dir = Path(run_dir)
    try:
        summary = adapter.describe_inputs(run_dir)
    except adapter.AdapterError as exc:
        return [], [{"stage": "fit_c/motion_v2", "reason": str(exc), "source": str(run_dir)}]

    from .recorder import Recorder

    recorder = Recorder(
        run_label=run_dir.name,
        algorithm=str(summary.get("algorithm") or "unknown"),
        software_version=str(summary.get("software_version") or "unknown"),
    )
    adapter.ingest_solver_logs(recorder, run_dir)
    config = summary.get("configuration") or {}
    budgets = {
        "shared_identity_joint": _num(config.get("steps")),
        "fixed_identity_motion": _num(config.get("motion_steps")),
        "identity_front_anchor": _num(config.get("front_steps")),
    }
    by_stage: dict[str, list] = {}
    for record in recorder.iterations:
        if record.value is not None and record.iteration is not None:
            by_stage.setdefault(record.stage, []).append(record)

    entries: list[GapEntry] = []
    gaps: list[dict[str, str]] = []
    for stage, records in sorted(by_stage.items()):
        values = sorted(((int(r.iteration), float(r.value)) for r in records),
                        key=lambda item: item[0])
        if not values:
            continue
        first_iter, first_value = values[0]
        last_iter, last_value = values[-1]
        budget = budgets.get(stage)
        used = last_iter + 1
        source = str(Path(run_dir) / "logs")
        drop = ((first_value - last_value) / abs(first_value)
                if first_value not in (0.0, None) else None)
        entries.append(GapEntry(
            key=f"{stage}.objective",
            stage=stage,
            group="solver",
            tier="solver_objective",
            metric="last recorded Adam objective",
            current=last_value,
            target=None,
            kind="none",
            gap=None,
            status="recorded",
            source=source,
            unit="",
            priority=7,
            notes=[
                f"first={first_value:g} at step {first_iter}; last={last_value:g} at step {last_iter}; "
                f"relative drop={drop if drop is None else round(drop, 6)}",
                "the delivery source defines no convergence tolerance for this stage: it stops "
                "on a fixed step budget, so this is a recorded value, not a target",
                "the log samples roughly every 20 steps; the run persists no per-term split, "
                "so no term decomposition is invented",
            ],
        ))
        if budget:
            remaining = float(budget - used)
            entries.append(GapEntry(
                key=f"{stage}.step_budget",
                stage=stage,
                group="solver",
                tier="solver_objective",
                metric="steps used / configured budget",
                current=float(used),
                target=float(budget),
                kind="budget",
                gap=remaining,
                status="budget_limited" if remaining <= 0 else "within_budget",
                source=source,
                unit="steps",
                priority=7,
                notes=[
                    "termination is the configured budget (steps / motion_steps / front_steps in "
                    "run_report.configuration); the source has no loss threshold to converge to",
                    "remaining=0 means the budget is exhausted, not that the loss converged",
                ],
            ))
    if not entries:
        gaps.append({
            "stage": "fit_c/motion_v2",
            "reason": "no per-step objective record was found in the run logs",
            "source": str(run_dir / "logs"),
        })
    return entries, gaps


def dense_entries(dense_dir: Path) -> tuple[list[GapEntry], list[dict[str, str]]]:
    """Terminal fixed-ray acceptance band of the last real base outer step."""
    base = neck_adapter.base_terminal(Path(dense_dir))
    source = str(base.get("source") or (Path(dense_dir) / "report.json"))
    if not base.get("present"):
        return [], [{
            "stage": "dense.solve_field",
            "reason": str(base.get("reason") or "no field_logs entry carries a fixed_ray_diagnostic"),
            "source": source,
        }]

    entries: list[GapEntry] = []
    before = _num(base.get("fixed_ray_before_mse_mm2"))
    after = _num(base.get("fixed_ray_after_mse_mm2"))
    context = (f"field={base.get('field')} outer={base.get('outer')} stage={base.get('stage')}; "
               f"accepted={base.get('step_accepted')}")
    if before is not None and after is not None:
        limit = before * DENSE_MSE_FACTOR + DENSE_MSE_ADD_MM2
        entries.append(GapEntry(
            key="dense.fixed_ray_mse",
            stage="dense.solve_field",
            group="solver",
            tier="nonlinear_energy",
            metric="fixed-ray MSE after / acceptance limit",
            current=after,
            target=limit,
            kind="max",
            gap=_max_gap(after, limit),
            status="pass" if after <= limit else "fail",
            source=source,
            unit="mm^2",
            hard=True,
            priority=1,
            notes=[
                f"before={before:g}; limit=before*1.02+0.002 from the recorded strict-coverage "
                f"allowance; {context}",
                "this is the re-cast fixed-ray energy, not the quadratic the solver minimised",
            ],
        ))
    else:
        entries.append(GapEntry(
            key="dense.fixed_ray_mse",
            stage="dense.solve_field",
            group="solver",
            tier="nonlinear_energy",
            metric="fixed-ray MSE after",
            current=after,
            target=None,
            kind="none",
            gap=None,
            status="unknown",
            source=source,
            unit="mm^2",
            hard=True,
            priority=1,
            notes=["the terminal diagnostic does not carry both before and after MSE", context],
        ))

    accepted = base.get("step_accepted")
    entries.append(GapEntry(
        key="dense.step_accepted",
        stage="dense.solve_field",
        group="acceptance",
        tier="acceptance_gate",
        metric="last outer step accepted",
        current=bool(accepted) if accepted is not None else None,
        target=None,
        kind="boolean",
        gap=None,
        status="pass" if accepted else ("fail" if accepted is False else "unknown"),
        source=source,
        unit="",
        hard=True,
        priority=1,
        notes=["the solver's own acceptance verdict for the terminal outer step", context],
    ))

    missing_after = _num(base.get("missing_after"))
    missing_cap = _num(base.get("missing_cap"))
    entries.append(GapEntry(
        key="dense.missing_rays",
        stage="dense.solve_field",
        group="acceptance",
        tier="acceptance_gate",
        metric="missing diagnostic rays after / cap",
        current=missing_after,
        target=missing_cap,
        kind="max",
        gap=_max_gap(missing_after, missing_cap),
        status=("pass" if missing_after is not None and missing_cap is not None
                and missing_after <= missing_cap else
                ("fail" if missing_after is not None and missing_cap is not None else "unknown")),
        source=source,
        unit="rays",
        hard=True,
        priority=1,
        notes=[
            f"missing before={base.get('missing_before')} after={missing_after} "
            f"cap={missing_cap} stage_baseline={base.get('missing_stage_baseline')}",
            "the cap is the stage allowance recorded by the run; a per-step increase below that "
            "cap is not re-labelled as a coverage loss",
        ],
    ))

    objective_before = _num(base.get("objective_before"))
    objective_after = _num(base.get("objective_after"))
    if objective_before is not None and objective_after is not None:
        entries.append(GapEntry(
            key="dense.linearised_objective",
            stage="dense.solve_field",
            group="solver",
            tier="linearised_objective",
            metric="frozen-correspondence objective before -> after",
            current=objective_after,
            target=None,
            kind="none",
            gap=None,
            status="recorded",
            source=source,
            unit="",
            priority=8,
            notes=[
                f"before={objective_before:g} after={objective_after:g}; the system is re-linearised "
                "between outers, so this value is not monotone across outers and is not comparable "
                "with the fixed-ray MSE",
            ],
        ))
    return entries, []


def neck_solver_entries(loss: dict | None, source: str) -> tuple[list[GapEntry], list[dict[str, str]]]:
    """Terminal tolerances of ``neck_fit.fit_unified`` for both states."""
    if not loss:
        return [], [{
            "stage": "neck_direct.fit_unified",
            "reason": "loss_direct.json not found: the neck build predates record-only persistence, "
                      "so the per-iteration solver trace and its terminal tolerance state are unknown",
            "source": source,
        }]
    params = loss.get("params") or {}
    max_iter = _num(params.get("fit_max_iter")) or 40.0
    entries: list[GapEntry] = []
    gaps: list[dict[str, str]] = []
    for state in ("closed", "open"):
        block = (loss.get("states") or {}).get(state) or {}
        rows = [row for row in block.get("iterations") or []
                if _num(row.get("energy")) is not None]
        if not rows:
            gaps.append({
                "stage": f"neck_direct.fit_unified.{state}",
                "reason": f"no iteration recorded (stop_reason={block.get('stop_reason')!r})",
                "source": source,
            })
            continue
        last = rows[-1]
        first = rows[0]
        converged = bool(block.get("converged"))
        stop_reason = block.get("stop_reason")
        stage = f"neck_direct.fit_unified.{state}"
        base_note = (f"converged={converged} stop_reason={stop_reason} "
                     f"n_iter={block.get('n_iter')}")
        entries.append(GapEntry(
            key=f"{stage}.energy",
            stage=stage,
            group="solver",
            tier="solver_objective",
            metric="fit_unified energy (first -> last)",
            current=_num(block.get("energy")),
            target=None,
            kind="none",
            gap=None,
            status="recorded",
            source=source,
            unit="",
            priority=8,
            notes=[
                f"first={_num(first.get('energy'))} last={_num(block.get('energy'))}; {base_note}",
                "energy = data + bending + tangential + anchor + small prior inside one solve; "
                "the parts may be summed only within an iteration",
            ],
        ))
        for metric, value_key, tol, unit in (
            ("step_max_mm", "step_max_mm", NECK_TOL_STEP_MM, "mm"),
            ("energy_rel", "energy_rel", NECK_TOL_ENERGY_REL, ""),
            ("grad_rel", "grad_rel", NECK_TOL_GRAD_REL, ""),
        ):
            value = _num(last.get(value_key))
            entries.append(GapEntry(
                key=f"{stage}.{metric}",
                stage=stage,
                group="solver",
                tier="solver_objective",
                metric=f"last {metric} / tolerance",
                current=value,
                target=tol,
                kind="tol",
                gap=_tol_gap(value, tol),
                status=_tol_status(value, tol, converged=converged, stop_reason=stop_reason),
                source=source,
                unit=unit,
                priority=7,
                notes=[
                    f"tolerance is the default in head_rgbd/neck_fit.py:682 "
                    f"(tol_step_mm=1e-3, tol_rel=1e-8; gradient 1e-9 relative at line 767); "
                    f"max_iter={max_iter:g} from build params",
                    (f"{base_note}; energy_stationary means the line search stalled at "
                     f"alpha<=1e-4, so the gradient is not zero"
                     if stop_reason == "energy_stationary" else base_note),
                    "positive gap = still above tolerance, negative = inside it",
                ],
            ))
        n_iter = _num(block.get("n_iter"))
        entries.append(GapEntry(
            key=f"{stage}.iterations",
            stage=stage,
            group="solver",
            tier="solver_objective",
            metric="iterations used / max_iter",
            current=n_iter,
            target=max_iter,
            kind="budget",
            gap=(max_iter - n_iter) if n_iter is not None else None,
            status=_stop_status(converged, stop_reason),
            source=source,
            unit="iterations",
            priority=7,
            notes=[base_note],
        ))
        residual = block.get("residual_after") or {}
        for metric in ("vertex_rms_mm", "quad_rms_mm"):
            value = _num(residual.get(metric))
            entries.append(GapEntry(
                key=f"{stage}.residual.{metric}",
                stage=stage,
                group="diagnostic",
                tier="diagnostic",
                metric=f"target-field {metric} after solve",
                current=value,
                target=None,
                kind="none",
                gap=None,
                status="recorded",
                source=source,
                unit="mm",
                priority=9,
                notes=[
                    "recomputed at the end of the same solve; diagnostic only, NOT added to the energy",
                    "vertex samples and in-face quadrature samples are separate",
                ],
            ))
        parts = (rows[-1].get("energy_parts") or {}) if rows else {}
        for name, value in parts.items():
            number = _num(value)
            if number is None:
                continue
            entries.append(GapEntry(
                key=f"{stage}.part.{name}",
                stage=stage,
                group="solver",
                tier="solver_objective",
                metric=f"energy part: {name}",
                current=number,
                target=None,
                kind="none",
                gap=None,
                status="recorded",
                source=source,
                unit="",
                priority=8,
                notes=["term split of the solver objective inside the last iteration only"],
            ))
    return entries, gaps


def gate_entries(*, result_dir: Path | None, neck_dir: Path | None,
                 landmarks: Path | None, trace_source: str) -> tuple[list[GapEntry], dict, list[dict[str, str]]]:
    """Delivery acceptance gates with numeric margins."""
    if result_dir is None:
        return [], {}, [{
            "stage": "neck.acceptance",
            "reason": "no delivered result directory supplied: gates were not read",
            "source": "",
        }]
    result_dir = Path(result_dir)
    neck_dir = Path(neck_dir) if neck_dir else neck_adapter.find_neck_dir(result_dir)
    gates = neck_adapter.neck_gates(result_dir, neck_dir)
    quick = _load(result_dir / "quick_check.json")
    build = _load(Path(neck_dir) / "build_direct.json") if neck_dir else {}
    entries: list[GapEntry] = []
    regions: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, str]] = []

    signed = ((quick.get("quick") or {}).get("signed_crease")
              or gates.get("signed_crease") or {})
    for state in ("closed", "open"):
        block = signed.get(state) or {}
        full = _num(block.get("full_positive_max_deg"))
        bridge = _num(block.get("B_positive_max_deg"))
        if full is None:
            continue
        regions[state] = {
            "full_positive_max_deg": full,
            "B_positive_max_deg": bridge,
            "worst": ("B" if bridge is not None and bridge >= full - 1e-9 else "outside_B"),
            "positive_gt20": block.get("positive_gt20"),
        }
    connection = _num(gates.get("connection_max_deg"))
    if regions:
        worst_detail = "; ".join(
            f"{state}: full={regions[state]['full_positive_max_deg']:g}, "
            f"B={regions[state]['B_positive_max_deg']}, region={regions[state]['worst']}"
            for state in regions
        )
    else:
        worst_detail = "no per-state signed crease block was recorded"
    entries.append(GapEntry(
        key="neck.connection",
        stage="neck.connection",
        group="acceptance",
        tier="acceptance_gate",
        metric="maximum positive crease on the full edge set",
        current=connection,
        target=CONNECTION_LIMIT_DEG,
        kind="max",
        gap=_max_gap(connection, CONNECTION_LIMIT_DEG),
        status=("pass" if connection is not None and connection <= CONNECTION_LIMIT_DEG
                else ("fail" if connection is not None else "unknown")),
        source=str(result_dir / "quick_check.json"),
        unit="deg",
        hard=True,
        priority=2,
        notes=[
            worst_detail,
            "the gate is the full edge set; the B-only value does not replace it",
        ],
    ))

    protection = gates.get("protection") or {}
    fixed_max = _num(protection.get("fixed_max_mm"))
    entries.append(GapEntry(
        key="neck.protection",
        stage="neck.protection",
        group="acceptance",
        tier="acceptance_gate",
        metric="protected-set maximum displacement",
        current=fixed_max,
        target=PROTECTION_LIMIT_MM,
        kind="max",
        gap=_max_gap(fixed_max, PROTECTION_LIMIT_MM),
        status=("pass" if protection.get("pass") and fixed_max is not None
                and fixed_max <= PROTECTION_LIMIT_MM else
                ("fail" if fixed_max is not None else "unknown")),
        source=str(result_dir / "quick_check.json"),
        unit="mm",
        hard=True,
        priority=2,
        notes=[
            f"pass={protection.get('pass')} fixed_vertices={protection.get('fixed_vertices')} "
            f"above_0.01mm={protection.get('vertices_above_0_01mm')}",
            "a protection failure is a hard invariant, not a tuning knob",
        ],
    ))

    upper = gates.get("upper_band") or {}
    input_block = upper.get("input") or {}
    full_block = upper.get("full") or {}
    delta = None
    if _num(input_block.get("vertices")) is not None and _num(full_block.get("vertices")) is not None:
        delta = float(full_block["vertices"]) - float(input_block["vertices"])
    entries.append(GapEntry(
        key="neck.upper_density",
        stage="neck.density",
        group="acceptance",
        tier="acceptance_gate",
        metric="upper-band vertex increase",
        current=delta,
        target=0.0,
        kind="max",
        gap=_max_gap(delta, 0.0),
        status=("pass" if upper.get("no_increase") and (delta is None or delta <= 0)
                else ("fail" if upper.get("no_increase") is False or (delta is not None and delta > 0)
                      else "unknown")),
        source=str(result_dir / "quick_check.json"),
        unit="vertices",
        hard=True,
        priority=3,
        notes=[
            f"no_increase={upper.get('no_increase')}; input={input_block.get('vertices')} "
            f"full={full_block.get('vertices')}",
            "the 2 percent split budget below is a separate, cumulative topology check",
        ],
    ))

    topology = gates.get("topology") or {}
    entries.append(GapEntry(
        key="neck.topology",
        stage="neck.topology",
        group="acceptance",
        tier="acceptance_gate",
        metric="topology pass + shared two-state face table",
        current=(bool(topology.get("pass")) if topology else None),
        target=None,
        kind="boolean",
        gap=None,
        status=("pass" if topology.get("pass") and gates.get("shared_topology") else
                ("fail" if topology else "unknown")),
        source=str(result_dir / "quick_check.json"),
        unit="",
        hard=True,
        priority=3,
        notes=[f"quick topology pass={topology.get('pass')}; shared_topology={gates.get('shared_topology')}"],
    ))

    journal = build.get("journal") or {}
    split_total = _num(journal.get("split_new_total"))
    split_cap = _num(journal.get("split_new_cap"))
    entries.append(GapEntry(
        key="neck.split_budget",
        stage="neck.density",
        group="acceptance",
        tier="acceptance_gate",
        metric="cumulative new vertices / 2 percent budget",
        current=split_total,
        target=split_cap,
        kind="max",
        gap=_max_gap(split_total, split_cap),
        status=("pass" if split_total is not None and split_cap is not None
                and split_total <= split_cap else
                ("fail" if split_total is not None and split_cap is not None else "unknown")),
        source=str(Path(neck_dir) / "build_direct.json") if neck_dir else str(trace_source),
        unit="vertices",
        hard=True,
        priority=3,
        notes=[
            f"vertices_original={journal.get('vertices_original')} "
            f"vertices_final={journal.get('vertices_final')} "
            f"merged_slots={journal.get('merged_slots')}",
            "the cap is the budget recorded by the builder; it is not re-derived from vertex counts",
        ],
    ))

    probe = _delivery_probe(result_dir)
    entries.append(GapEntry(
        key="delivery.self_contained",
        stage="delivery",
        group="acceptance",
        tier="acceptance_gate",
        metric="final/measured sidecar + manifest probe",
        current=probe.get("present"),
        target=None,
        kind="boolean",
        gap=None,
        status="pass" if probe.get("status") == "pass" else "fail",
        source=str(result_dir),
        unit="",
        hard=True,
        priority=0,
        notes=probe.get("problems") or ["final/measured are self-contained"],
    ))

    markers = neck_adapter.marker_semantics(landmarks)
    present = bool(markers.get("present") and {"closed", "open"} <= set((markers.get("states") or {})))
    marker_detail = "; ".join(
        f"{state}: status={block.get('status')} source={block.get('source')} "
        f"t={block.get('t_center_used')} theta={block.get('theta_used_deg')} "
        f"size={block.get('black_square_mm')}"
        for state, block in (markers.get("states") or {}).items()
    ) or str(markers.get("reason") or "no marker sidecar")
    entries.append(GapEntry(
        key="marker.semantics",
        stage="marker",
        group="acceptance",
        tier="acceptance_gate",
        metric="marker-derived centre and amplitude recorded per state",
        current=present,
        target=None,
        kind="boolean",
        gap=None,
        status="pass" if present else "unknown",
        source=str(landmarks) if landmarks else "",
        unit="",
        hard=True,
        priority=0,
        notes=[marker_detail, "the marker is a measurement, not a candidate parameter"],
    ))

    reference_peak = _num((((build.get("structure") or {}).get("reference") or {})
                           .get("primary") or {}).get("relief_mm"))
    structure = ((quick.get("quick") or {}).get("structure")
                 or gates.get("structure") or {})
    for state in ("closed", "open"):
        delivered = _num((((structure.get(state) or {}).get("larynx_local") or {})
                          .get("relief_mm")))
        entries.append(GapEntry(
            key=f"neck.structure.{state}",
            stage="neck.structure",
            group="diagnostic",
            tier="diagnostic",
            metric="laryngeal relief vs reference peak",
            current=delivered,
            target=reference_peak,
            kind="none",
            gap=(delivered - reference_peak
                 if delivered is not None and reference_peak is not None else None),
            status="recorded",
            source=str(result_dir / "quick_check.json"),
            unit="mm",
            priority=5,
            notes=[
                "structure deviation is reported, not gated; the reference is shape-only",
                "a negative gap means the delivered relief is below the reference peak",
            ],
        ))
    if not gates.get("present"):
        gaps.append({
            "stage": "neck.acceptance",
            "reason": "quick_check.json / verification.json not found for this delivery",
            "source": str(result_dir),
        })
    return entries, regions, gaps


def _delivery_probe(result_dir: Path) -> dict:
    from .optimizer import _delivery_probe as probe

    return probe(result_dir)


def dominant_entry(entries: list[GapEntry]) -> GapEntry | None:
    hard = [entry for entry in entries if entry.hard]
    failed = sorted((entry for entry in hard if entry.status == "fail"),
                    key=lambda entry: entry.priority)
    if failed:
        return failed[0]
    numeric = [
        entry for entry in hard
        if entry.gap is not None and entry.target not in (None, 0.0)
        and isinstance(entry.gap, (int, float))
    ]
    if numeric:
        return min(
            numeric,
            key=lambda entry: float(entry.gap) / max(abs(float(entry.target)), 1e-12),
        )
    return None


def targeted_keys(dominant: GapEntry | None,
                  regions: dict[str, dict[str, Any]],
                  entries: tuple[GapEntry, ...] | list[GapEntry] = ()) -> list[dict[str, str]]:
    """At most two whitelisted parameters that directly act on the dominant gap."""
    if dominant is None:
        return []
    key = dominant.key
    if key == "neck.connection":
        if regions:
            max_full = max(region["full_positive_max_deg"] for region in regions.values())
            worst = [region for region in regions.values()
                     if abs(region["full_positive_max_deg"] - max_full) <= 1e-9]
            outside = any(region.get("worst") == "outside_B" for region in worst)
        else:
            outside = True
        second = ({
            "key": "neck.mu_t_bridge" if not outside else "neck.anchor_near_A",
            "why": ("the worst positive crease is inside the bridge ring; add tangential stiffness "
                    "at the B/C crossing" if not outside else
                    "the full-edge maximum sits outside the bridge ring; stiffen the hand-over "
                    "next to the protected head"),
        })
        relief = [
            entry.gap / abs(entry.target)
            for entry in entries
            if entry.key.startswith("neck.structure.")
            and entry.gap is not None and entry.target not in (None, 0)
        ]
        if relief and sum(relief) / len(relief) > 0:
            return [{
                "key": "neck.structure_gain",
                "why": "the delivered laryngeal relief is above the shape-only reference peak; "
                       "lowering the structural gain is the direct global control, before changing "
                       "the seam or bridge locally",
            }, second]
        if outside:
            return [
                {"key": "neck.anchor_near_A",
                 "why": "the full-edge maximum sits outside the bridge ring; stiffen the hand-over "
                        "next to the protected head before changing the bridge prior"},
                {"key": "neck.seam_taper_ratio",
                 "why": "the worst positive crease is outside B, i.e. at the seam / A side; widen the "
                        "metric fade there"},
            ]
        return [
            {"key": "neck.mu_t_bridge",
             "why": "the worst positive crease is inside the bridge ring; add tangential stiffness "
                    "at the B/C crossing"},
            {"key": "neck.anchor_near_A",
             "why": "second move in the same region: stiffen the hand-over next to the head"},
        ]
    if key in ("dense.fixed_ray_mse", "dense.missing_rays"):
        return [
            {"key": "base.bending_length_mm",
             "why": "the terminal fixed-ray energy is the base tier; a stiffer dense bending prior "
                    "smooths the surface without touching the neck"},
            {"key": "base.low_bending_length_mm",
             "why": "the low-frequency field is the second, separately identifiable base scale"},
        ]
    if key == "neck.structure":
        return [{
            "key": "neck.structure_gain",
            "why": "the delivered laryngeal relief is the gap; gain is the direct structural control",
        }]
    if key.startswith("neck_direct.fit_unified") or key.startswith("neck.target_residual"):
        return [
            {"key": "neck.mu_t_body",
             "why": "the target-field residual is still visible after the solve; raise the body "
                    "tangential prior before touching the anchor"},
            {"key": "neck.anchor_near_A",
             "why": "if the residual concentrates near the fixed head, the hand-over stiffness is "
                    "the second move"},
        ]
    return []


def compute_gaps(*, run_dir: Path | None, dense_dir: Path,
                 result_dir: Path | None = None, neck_dir: Path | None = None,
                 landmarks: Path | None = None,
                 trace_dir: Path | None = None) -> dict[str, Any]:
    dense_dir = Path(dense_dir)
    run_dir = Path(run_dir) if run_dir else None
    result_dir = Path(result_dir) if result_dir else None
    neck_dir = (Path(neck_dir) if neck_dir else
                (neck_adapter.find_neck_dir(result_dir) if result_dir else None))
    trace_dir = Path(trace_dir) if trace_dir else neck_dir
    loss = neck_adapter.read_loss_direct(trace_dir)
    loss_source = str(Path(trace_dir) / "loss_direct.json") if trace_dir else ""

    log_entries, gaps = solver_log_entries(run_dir)
    dense_items, dense_gaps = dense_entries(dense_dir)
    solver_items, neck_gaps = neck_solver_entries(loss, loss_source)
    gate_items, regions, gate_gaps = gate_entries(
        result_dir=result_dir, neck_dir=neck_dir, landmarks=landmarks,
        trace_source=loss_source,
    )
    gaps.extend(dense_gaps + neck_gaps + gate_gaps)
    entries = log_entries + dense_items + solver_items + gate_items
    dominant = dominant_entry(entries)

    notes = [
        "the solver layer and the acceptance layer are reported separately; there is no total gap",
        "a positive gap is remaining margin, a negative gap is the amount over the target",
        "booleans are not converted into 1/0",
        "missing values are null and appear in the gap list, never as 0",
    ]
    if loss is None:
        notes.append("no loss_direct.json: the neck solver trace is a gap; the delivered mesh is "
                     "never differentiated to invent a curve")
    if not run_dir:
        notes.append("no base run directory: Adam step budgets were not read")

    return {
        "schema": GAP_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "run": str(run_dir) if run_dir else None,
            "dense": str(dense_dir),
            "result": str(result_dir) if result_dir else None,
            "neck": str(neck_dir) if neck_dir else None,
            "trace": str(trace_dir) if trace_dir else None,
            "landmarks": str(landmarks) if landmarks else None,
        },
        "targets": {
            "connection_deg": CONNECTION_LIMIT_DEG,
            "protection_mm": PROTECTION_LIMIT_MM,
            "dense_mse": "before * 1.02 + 0.002 mm^2 (recorded strict-coverage allowance)",
            "neck_fit_unified": {
                "tol_step_mm": NECK_TOL_STEP_MM,
                "tol_rel": NECK_TOL_ENERGY_REL,
                "grad_rel": NECK_TOL_GRAD_REL,
                "max_iter": (loss or {}).get("params", {}).get("fit_max_iter"),
            },
        },
        "entries": [entry.as_dict() for entry in entries],
        "solver": [entry.as_dict() for entry in entries if entry.group == "solver"],
        "acceptance": [entry.as_dict() for entry in entries if entry.group == "acceptance"],
        "diagnostic": [entry.as_dict() for entry in entries if entry.group == "diagnostic"],
        "connection_regions": regions,
        "dominant": dominant.as_dict() if dominant else None,
        "targeted": targeted_keys(dominant, regions, entries),
        "gaps": gaps,
        "notes": notes,
    }


def _csv_rows(report: dict) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for entry in report.get("entries") or []:
        rows.append([
            entry.get("group"), entry.get("key"), entry.get("stage"), entry.get("tier"),
            entry.get("metric"), entry.get("current"), entry.get("target"),
            entry.get("kind"), entry.get("gap"), entry.get("unit"),
            entry.get("status"), entry.get("hard"), entry.get("priority"),
            entry.get("source"), " | ".join(entry.get("notes") or []),
        ])
    return rows


def _html(report: dict) -> str:
    sections: list[dict[str, str]] = []
    rules = (
        "<p>两层终止目标分别报告，不合并、不相加：</p>"
        "<ul class='tight'>"
        "<li><b>求解器层</b>：用源码真实终止条件。"
        "<code>fit_c</code> / <code>motion_v2</code> 只按固定步数预算停止，源码没有收敛阈值，"
        "因此只报告预算消耗与最后记录值；<code>solve_field</code> 用 fixed-ray 接受带；"
        "<code>neck_fit.fit_unified</code> 用步长 / 能量 / 梯度阈值。</li>"
        "<li><b>验收层</b>：连接 20°（全边集）、保护 0.002 mm、上颈密度、拓扑与两态面表、"
        "拆分预算、标记语义、交付 sidecar/manifest 自包含。</li>"
        "</ul>"
        "<div class='callout'>gap 符号：正数=还有余量，负数=已超出目标；布尔值保持布尔，"
        "缺失写 null。四档位不可相加，两层 gap 也不可相加。</div>"
    )
    sections.append({"id": "rules", "heading": "0. 终止目标与 gap 约定", "html": rules})

    dominant = report.get("dominant")
    if dominant:
        rows = [[
            _text(dominant.get("key")), _text(dominant.get("status")),
            _fmt(dominant.get("current")), _fmt(dominant.get("target")),
            _fmt(dominant.get("gap")), _text(dominant.get("unit")),
            "<br>".join(_text(note) for note in dominant.get("notes") or []),
        ]]
        dominant_html = table(["key", "状态", "当前", "目标", "gap", "单位", "说明"], rows)
    else:
        dominant_html = "<p>没有硬性失败，也没有可比较的数值余量。</p>"
    targeted_rows = [
        [_text(item["key"]), _text(item["why"])]
        for item in report.get("targeted") or []
    ]
    if targeted_rows:
        targeted_html = table(["白名单参数", "为什么是它"], targeted_rows)
    else:
        targeted_html = ("<p class='note'>本轮不提出参数候选：主导 gap 不由白名单参数控制，"
                         "或没有任何失败/临界项。</p>")
    sections.append({
        "id": "dominant",
        "heading": "1. 主导 gap 与定向候选（≤2）",
        "html": dominant_html + targeted_html,
    })

    def entry_rows(items: list[dict]) -> list[list[str]]:
        return [[
            _text(item.get("stage")), _text(item.get("metric")),
            _fmt(item.get("current")), _fmt(item.get("target")),
            _fmt(item.get("gap")), _text(item.get("unit")),
            _text(item.get("status")), _text(item.get("tier")),
            "<br>".join(_text(note) for note in item.get("notes") or []),
        ] for item in items]

    sections.append({
        "id": "solver",
        "heading": "2. 求解器层：距终止条件",
        "html": table(["阶段", "指标", "当前", "目标/阈值", "gap", "单位", "状态", "档位", "说明"],
                      entry_rows(report.get("solver") or []), numeric={2, 3, 4})
        if report.get("solver") else "<p>没有求解器记录。</p>",
    })
    sections.append({
        "id": "acceptance",
        "heading": "3. 验收层：余量",
        "html": table(["阶段", "指标", "当前", "目标/阈值", "gap", "单位", "状态", "档位", "说明"],
                      entry_rows(report.get("acceptance") or []), numeric={2, 3, 4})
        if report.get("acceptance") else "<p>没有验收记录。</p>",
    })
    sections.append({
        "id": "diagnostic",
        "heading": "4. 诊断层（非算法 loss）",
        "html": table(["阶段", "指标", "当前", "参考", "差值", "单位", "状态", "档位", "说明"],
                      entry_rows(report.get("diagnostic") or []), numeric={2, 3, 4})
        if report.get("diagnostic") else "<p>没有诊断记录。</p>",
    })
    gap_rows = [[_text(gap.get("stage")), _text(gap.get("reason")), _text(gap.get("source"))]
                for gap in report.get("gaps") or []]
    sections.append({
        "id": "gaps",
        "heading": "5. 缺口：没有真实状态的地方",
        "html": table(["阶段", "原因", "来源"], gap_rows) if gap_rows else "<p>没有缺口。</p>",
    })
    conclusion = (
        "<ul class='tight'>"
        "<li>本报告只读已完成产物；没有启动任何重建，也没有修改任何参数。</li>"
        "<li>“预算耗尽”不等于“收敛”；“目标残差”不等于“算法 loss”；“下游 pass/fail”不在本报告内。</li>"
        "<li>定向候选只是下一步实验建议，是否执行由调用方显式选择；候选写入新目录。</li>"
        "</ul>"
    )
    sections.append({"id": "boundaries", "heading": "6. 结论边界", "html": conclusion})
    subtitle = (f"run={_text((report.get('inputs') or {}).get('run'))} · "
                f"neck={_text((report.get('inputs') or {}).get('neck'))} · "
                f"生成于 {_text(report.get('created_at'))}")
    return render_page(title="逐阶段终止 gap 报告", subtitle=subtitle, sections=sections,
                       footer="由 loss_analysis 离线生成 · 只读产物 · 两层 gap 不相加")


def write_outputs(report: dict, out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out_dir / "gaps.json", report)
    csv_path = out_dir / "gaps.csv"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "key", "stage", "tier", "metric", "current", "target",
                         "kind", "gap", "unit", "status", "hard", "priority", "source", "notes"])
        writer.writerows(_csv_rows(report))
    html_path = write_report(out_dir / "report.html", _html(report))
    summary = {
        "schema": "loss_analysis.gaps.summary.v1",
        "output_dir": str(out_dir),
        "report": str(html_path),
        "gaps_json": str(json_path),
        "csv": str(csv_path),
        "dominant": (report.get("dominant") or {}).get("key"),
        "dominant_status": (report.get("dominant") or {}).get("status"),
        "targeted": [item["key"] for item in report.get("targeted") or []],
        "hard_failures": [entry["key"] for entry in report.get("entries") or []
                          if entry.get("hard") and entry.get("status") == "fail"],
        "gap_count": len(report.get("gaps") or []),
        "note": "analysis only: no reconstruction was run and no parameter was changed",
    }
    write_json(out_dir / "summary.json", summary)
    return {
        "output_dir": str(out_dir),
        "report": str(html_path),
        "gaps_json": str(json_path),
        "csv": str(csv_path),
        "summary": summary,
    }
