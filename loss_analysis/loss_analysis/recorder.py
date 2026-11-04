"""Iteration recorder with explicit provenance tiers.

The recorder refuses to fabricate progress. Four tiers are stored separately and
are never merged into one "loss":

``solver_objective``   the scalar the optimiser actually minimised at that step
``linearised_objective`` value of the frozen-correspondence linear system
``nonlinear_energy``   re-evaluated energy after correspondences are refreshed
``diagnostic``         our own MAE/RMSE/P95/coverage; never the algorithm's loss

If a stage produced no iteration-level record (for example the Loop blend search,
which has no objective), the recorder writes an explicit *gap* entry and the
plotter refuses to draw a line through it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .objectives import (
    DIAGNOSTIC,
    LINEARISED,
    NONLINEAR_ENERGY,
    NOT_USED,
    NULL,
    RECORDED,
    RECOMPUTED,
    SOLVER_OBJECTIVE,
)

RECORD_SCHEMA = "loss_analysis.record.v1"
TIERS = (SOLVER_OBJECTIVE, LINEARISED, NONLINEAR_ENERGY, DIAGNOSTIC)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class IterationRecord:
    """One observation of one stage at one iteration index."""

    stage: str
    iteration: int | None
    tier: str
    phase: str | None = None
    value: float | None = None
    terms: dict[str, float | None] = field(default_factory=dict)
    weights: dict[str, float | None] = field(default_factory=dict)
    raw: dict[str, float | None] = field(default_factory=dict)
    n_valid: dict[str, int | None] = field(default_factory=dict)
    denominators: dict[str, float | None] = field(default_factory=dict)
    decision: str | None = None            # "initial" | "candidate" | "accepted" | "rejected" | None
    step_scale: float | None = None
    coverage: float | None = None
    source: str = RECOMPUTED
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"unknown provenance tier {self.tier!r}; allowed {TIERS}")
        for key, value in list(self.terms.items()):
            if value is None:
                self.notes.append(f"term {key} is null (unavailable), not zero")
            elif not isinstance(value, (int, float)):
                raise TypeError(f"term {key} must be a number or None")

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": RECORD_SCHEMA,
            "stage": self.stage,
            "iteration": self.iteration,
            "tier": self.tier,
            "phase": self.phase,
            "value": self.value,
            "terms": self.terms,
            "weights": self.weights,
            "raw": self.raw,
            "n_valid": self.n_valid,
            "denominators": self.denominators,
            "decision": self.decision,
            "step_scale": self.step_scale,
            "coverage": self.coverage,
            "source": self.source,
            "notes": self.notes,
            "extras": self.extras,
        }


@dataclass
class GapRecord:
    """A stage/iteration that has no real recorded state — explicitly a hole."""

    stage: str
    reason: str
    expected_tier: str = SOLVER_OBJECTIVE
    source_hint: str = ""

    def as_record(self) -> dict[str, Any]:
        # Accept both a string tier and an explicit-but-optional attribute.
        tier = getattr(self, "expected_tier", None) or SOLVER_OBJECTIVE
        return {
            "schema": RECORD_SCHEMA,
            "kind": "gap",
            "expected_tier": tier,
            "stage": self.stage,
            "iteration": None,
            "tier": tier,
            "value": None,
            "status": NULL,
            "reason": self.reason,
            "source_hint": self.source_hint,
            "note": "missing iteration state is recorded as null; the tool never interpolates a curve",
        }


class Recorder:
    """Append-only in-memory log plus atomic JSON/CSV writers."""

    def __init__(self, *, run_label: str, algorithm: str, software_version: str) -> None:
        self.run_label = run_label
        self.algorithm = algorithm
        self.software_version = software_version
        self.created_at = _now()
        self.iterations: list[IterationRecord] = []
        self.gaps: list[GapRecord] = []
        self.stage_meta: dict[str, dict[str, Any]] = {}

    # -- writing ----------------------------------------------------------
    def add(self, record: IterationRecord) -> None:
        self.iterations.append(record)

    def gap(self, stage: str, reason: str, *, expected_tier: str = SOLVER_OBJECTIVE,
            source_hint: str = "") -> None:
        self.gaps.append(GapRecord(stage, reason, expected_tier, source_hint))

    def declare_stage(self, stage: str, *, has_iterations: bool, notes: Iterable[str] = ()) -> None:
        self.stage_meta[stage] = {
            "has_iterations": bool(has_iterations),
            "notes": list(notes),
        }

    # -- queries ----------------------------------------------------------
    def has_tier(self, stage: str, tier: str) -> bool:
        return any(r.stage == stage and r.tier == tier and r.value is not None
                   for r in self.iterations)

    def series(self, stage: str, tier: str) -> list[tuple[int, float]]:
        """Points with a real iteration index AND a real value, ascending."""
        rows = [
            (r.iteration, r.value)
            for r in self.iterations
            if r.stage == stage and r.tier == tier and r.iteration is not None and r.value is not None
        ]
        return sorted(rows, key=lambda item: item[0])

    def tiers_present(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.iterations:
            out.setdefault(r.stage, [])
            if r.tier not in out[r.stage] and r.value is not None:
                out[r.stage].append(r.tier)
        return out

    # -- persistence ------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        return {
            "schema": "loss_analysis.run.v1",
            "run_label": self.run_label,
            "algorithm": self.algorithm,
            "software_version": self.software_version,
            "created_at": self.created_at,
            "tier_definitions": {
                SOLVER_OBJECTIVE: "scalar the optimiser minimised at that step",
                LINEARISED: "frozen-correspondence linear system value (solver space)",
                NONLINEAR_ENERGY: "re-evaluated energy after refreshing correspondences",
                DIAGNOSTIC: "this tool's MAE/RMSE/P95/coverage; NOT the algorithm loss",
            },
            "stages": self.stage_meta,
            "iterations": [r.as_record() for r in self.iterations],
            "gaps": [g.as_record() for g in self.gaps],
            "totals_policy": (
                "stages have no common total loss; no cross-stage total is produced"
            ),
        }

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.payload(), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return path

    def write_csv(self, path: Path) -> Path:
        """Long format: one row per (stage, iteration, tier, term)."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        fields = [
            "stage", "iteration", "tier", "phase", "decision", "step_scale", "coverage",
            "term", "value", "weight", "raw", "n_valid", "denominator", "source", "notes",
        ]
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r in self.iterations:
                base = {
                    "stage": r.stage,
                    "iteration": r.iteration,
                    "tier": r.tier,
                    "phase": r.phase,
                    "decision": r.decision,
                    "step_scale": r.step_scale,
                    "coverage": r.coverage,
                    "source": r.source,
                    "notes": " | ".join(r.notes),
                }
                if r.terms:
                    for key, value in r.terms.items():
                        writer.writerow({
                            **base,
                            "term": key,
                            "value": value,
                            "weight": r.weights.get(key),
                            "raw": r.raw.get(key),
                            "n_valid": r.n_valid.get(key),
                            "denominator": r.denominators.get(key),
                        })
                else:
                    writer.writerow({**base, "term": "__total__", "value": r.value})
            for g in self.gaps:
                writer.writerow({
                    "stage": g.stage,
                    "iteration": "",
                    "tier": g.expected_tier,
                    "term": "__gap__",
                    "value": "",
                    "notes": g.reason,
                    "source": "gap",
                })
        return path


def load_run_payload(path: Path) -> dict[str, Any]:
    """Read a previously written run JSON (read-only)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stages_with_gaps(payload: dict[str, Any]) -> list[str]:
    return sorted({g["stage"] for g in payload.get("gaps", [])})


def available_series(payload: dict[str, Any], stage: str, tier: str) -> list[tuple[int, float]]:
    """Extract a plottable series, skipping nulls instead of coercing them to 0."""
    rows = []
    for r in payload.get("iterations", []):
        if r.get("stage") != stage or r.get("tier") != tier:
            continue
        if r.get("iteration") is None or r.get("value") is None:
            continue
        rows.append((int(r["iteration"]), float(r["value"])))
    return sorted(rows)
