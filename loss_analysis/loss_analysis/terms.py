"""Term-level loss computation, transcribed one-to-one from the delivery source.

Design rules that make this an *analysis* tool rather than a second solver:

* every function is pure: it takes arrays, returns a :class:`TermResult`;
* the denominator is always reported next to the value, and a zero denominator
  yields ``value=None`` with ``status="no_valid_samples"`` — never ``0.0``;
* invalid landmarks / invalid depth are excluded by an explicit mask, so a
  missing observation can never contribute a fake zero residual;
* nothing here mutates the caller's arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .objectives import (
    DIAGNOSTIC,
    NONLINEAR_ENERGY,
    NOT_USED,
    NULL,
    RECOMPUTED,
    SOLVER_OBJECTIVE,
)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def smooth_l1_beta03(values: np.ndarray) -> np.ndarray:
    """Element-wise PyTorch ``smooth_l1_loss(x, 0, beta=0.3)`` without the mean.

    Matches ``torch.nn.functional.smooth_l1_loss(x, zeros, beta=b, reduction='none')``:
    ``0.5*x^2/b`` for ``|x| < b`` and ``|x| - 0.5*b`` otherwise.
    """
    x = np.asarray(values, dtype=float)
    b = 0.3
    small = 0.5 * x * x / b
    large = np.abs(x) - 0.5 * b
    return np.where(np.abs(x) < b, small, large)


def smooth_l1_beta1(values: np.ndarray) -> np.ndarray:
    """Element-wise ``smooth_l1_loss(x, 0, beta=1.0)`` used by motion_v2."""
    x = np.asarray(values, dtype=float)
    return np.where(np.abs(x) < 1.0, 0.5 * x * x, np.abs(x) - 0.5)


def hinge_relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _finite(values: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    return values, keep


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class TermResult:
    """One term of one stage at one point in time.

    ``value`` is the term's contribution **as the source adds it** (already
    multiplied by its constant weight and divided by its denominator), i.e. it is
    directly comparable with the recorded ``objective`` scale.
    """

    key: str
    stage: str
    provenance: str
    raw: float | None = None            # pre-weight statistic (e.g. mean residual)
    weight: float | None = None         # effective constant multiplier
    value: float | None = None          # contribution added to the stage objective
    unit: str = ""
    n_valid: int | None = None
    n_total: int | None = None
    denominator: float | None = None
    status: str = RECOMPUTED
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "stage": self.stage,
            "provenance": self.provenance,
            "raw": self.raw,
            "weight": self.weight,
            "value": self.value,
            "unit": self.unit,
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "denominator": self.denominator,
            "status": self.status,
            "notes": self.notes,
            **self.extras,
        }


@dataclass
class StageResult:
    """All terms of a stage at one point, plus how to read the total."""

    stage: str
    provenance: str
    terms: list[TermResult]
    total: float | None
    total_kind: str
    n_terms_used: int
    n_terms_missing: int
    notes: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provenance": self.provenance,
            "total": self.total,
            "total_kind": self.total_kind,
            "n_terms_used": self.n_terms_used,
            "n_terms_missing": self.n_terms_missing,
            "terms": [term.as_record() for term in self.terms],
            "notes": self.notes,
        }


def _missing(key: str, stage: str, provenance: str, reason: str, *, status: str = NOT_USED) -> TermResult:
    return TermResult(
        key=key,
        stage=stage,
        provenance=provenance,
        status=status,
        notes=reason,
    )


def summarise(
    stage: str,
    terms: list[TermResult],
    expected: tuple[str, ...],
    *,
    provenance: str = RECOMPUTED,
    total_kind: str = "sum_of_available_terms",
    notes: list[str] | None = None,
) -> StageResult:
    """Combine terms, keeping missing/zero-sample terms out of the total.

    ``total`` is ``None`` when **no** term could be computed, and it is flagged as
    partial whenever some expected term is ``not_used`` or ``null``. The tool never
    invents a zero to make the total look complete.
    """
    present = {term.key for term in terms if term.value is not None}
    missing = [key for key in expected if key not in present]
    usable = [term.value for term in terms if term.value is not None]
    total = float(sum(usable)) if usable else None
    all_notes = list(notes or [])
    if missing:
        all_notes.append(
            "partial total: " + ", ".join(missing) + " unavailable, not set to zero"
        )
    return StageResult(
        stage=stage,
        provenance=provenance,
        terms=terms,
        total=total,
        total_kind=total_kind if not missing else total_kind + "_partial",
        n_terms_used=len(usable),
        n_terms_missing=len(missing),
        notes=all_notes,
    )


# ---------------------------------------------------------------------------
# fit_c / motion_v2 style terms (torch Adam objectives, per-step scalars)
# ---------------------------------------------------------------------------
def landmark_term(
    landmark_xyz: np.ndarray,
    landmark_target: np.ndarray,
    valid: np.ndarray,
    *,
    indices: slice,
    scale_m: float,
    weight: float,
    stage: str,
    key: str,
    beta: str = "0.3",
    provenance: str = SOLVER_OBJECTIVE,
) -> TermResult:
    """Robust landmark residual normalized by ``scale_m`` (metres -> dimensionless)."""
    xyz = np.asarray(landmark_xyz, dtype=float)
    target = np.asarray(landmark_target, dtype=float)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if xyz.shape != target.shape or xyz.shape[-1] != 3:
        raise ValueError("landmark arrays must both be (68,3)")
    mask = np.zeros(len(valid), bool)
    mask[indices] = True
    mask &= valid
    delta = xyz[mask] - target[mask]
    delta, keep = _finite(delta.reshape(-1), None)
    n_total = int(mask.sum()) * 3
    if not np.any(keep):
        return TermResult(
            key=key,
            stage=stage,
            provenance=provenance,
            weight=weight,
            unit="dimensionless",
            n_valid=0,
            n_total=int(xyz.shape[0]) * 3,
            denominator=0.0,
            status="no_valid_samples",
            notes="all candidate landmarks invalid or non-finite; term omitted, NOT zero",
        )
    residual = delta[keep] / scale_m
    per = smooth_l1_beta03(residual) if beta == "0.3" else smooth_l1_beta1(residual)
    raw = float(np.mean(per))
    raw_no_robust = float(np.mean(residual))
    return TermResult(
        key=key,
        stage=stage,
        provenance=provenance,
        raw=raw,
        weight=weight,
        value=weight * raw,
        unit="dimensionless",
        n_valid=int(keep.sum()),
        n_total=n_total,
        denominator=float(keep.sum()),
        notes=f"mean over {int(keep.sum())} finite components of index set [{indices.start}:{indices.stop}]",
        extras={"mean_normalised_residual": raw_no_robust},
    )


def uv_term(
    uv: np.ndarray,
    uv_target: np.ndarray,
    valid: np.ndarray,
    *,
    indices: slice,
    scale_px: float,
    weight: float,
    stage: str,
    key: str,
    provenance: str = SOLVER_OBJECTIVE,
) -> TermResult:
    """2D reprojection residual; needs no depth so it survives missing depth."""
    a = np.asarray(uv, dtype=float)
    b = np.asarray(uv_target, dtype=float)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if a.shape != b.shape or a.shape[-1] != 2:
        raise ValueError("uv arrays must both be (68,2)")
    mask = np.zeros(len(valid), bool)
    mask[indices] = True
    mask &= valid
    delta = (a[mask] - b[mask]).reshape(-1)
    delta, keep = _finite(delta, None)
    if not np.any(keep):
        return TermResult(
            key=key,
            stage=stage,
            provenance=provenance,
            weight=weight,
            unit="dimensionless",
            n_valid=0,
            n_total=int(a.shape[0]) * 2,
            denominator=0.0,
            status="no_valid_samples",
            notes="no finite 2D landmark; term omitted, NOT zero",
        )
    residual = delta[keep] / scale_px
    per = smooth_l1_beta03(residual)
    raw = float(np.mean(per))
    return TermResult(
        key=key,
        stage=stage,
        provenance=provenance,
        raw=raw,
        weight=weight,
        value=weight * raw,
        unit="dimensionless",
        n_valid=int(keep.sum()),
        n_total=int(a.shape[0]) * 2,
        denominator=float(keep.sum()),
        extras={"mean_normalised_residual": float(np.mean(residual))},
    )


def mouth_gap_term(
    landmark_xyz: np.ndarray,
    pairs: np.ndarray,
    *,
    free_mm: float = 2.0,
    band_mm: float = 3.0,
    weight: float,
    stage: str,
    key: str = "mouth_gap_closed",
    provenance: str = SOLVER_OBJECTIVE,
) -> TermResult:
    """``relu(gap - 2mm)/3mm`` over the three inner-lip pairs (closed frames only)."""
    xyz = np.asarray(landmark_xyz, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be (K,2)")
    a, b = pairs[:, 0], pairs[:, 1]
    if a.max() >= len(xyz) or b.max() >= len(xyz):
        raise ValueError("lip pair index out of range")
    gap_mm = np.linalg.norm(xyz[a] - xyz[b], axis=1) * 1000.0
    gap_mm, keep = _finite(gap_mm, None)
    if not np.any(keep):
        return TermResult(
            key=key,
            stage=stage,
            provenance=provenance,
            weight=weight,
            unit="dimensionless",
            n_valid=0,
            n_total=len(pairs),
            denominator=0.0,
            status="no_valid_samples",
            notes="inner lip depth invalid; hinge omitted, NOT zero",
        )
    normalised = hinge_relu(gap_mm[keep] - free_mm) / band_mm
    raw = float(np.mean(normalised**2))
    return TermResult(
        key=key,
        stage=stage,
        provenance=provenance,
        raw=raw,
        weight=weight,
        value=weight * raw,
        unit="dimensionless",
        n_valid=int(keep.sum()),
        n_total=len(pairs),
        denominator=float(keep.sum()),
        extras={"mean_gap_mm": float(np.mean(gap_mm[keep]))},
    )


def quadratic_prior(values: np.ndarray, *, weight: float, stage: str, key: str,
                    unit: str = "coefficient^2", denominator: str = "element count") -> TermResult:
    """``weight * mean(values^2)`` over finite entries."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x, keep = _finite(x, None)
    if not np.any(keep):
        return TermResult(key=key, stage=stage, provenance=SOLVER_OBJECTIVE, weight=weight,
                          unit=unit, n_valid=0, n_total=int(x.size), denominator=0.0,
                          status="no_valid_samples", notes="empty prior vector; omitted, NOT zero")
    raw = float(np.mean(x[keep] ** 2))
    return TermResult(
        key=key,
        stage=stage,
        provenance=SOLVER_OBJECTIVE,
        raw=raw,
        weight=weight,
        value=weight * raw,
        unit=unit,
        n_valid=int(keep.sum()),
        n_total=int(x.size),
        denominator=float(keep.sum()),
        notes=denominator,
    )


def expression_prior_fit_c(expression: np.ndarray, frames: int, *, weight: float = 0.002,
                           stage: str = "fit_c") -> TermResult:
    """fit_c's asymmetric expression prior: ``sum(expr^2) / (n_frames * 10)``."""
    x = np.asarray(expression, dtype=float)
    x, keep = _finite(x, None)
    denom = float(max(frames, 1) * 10)
    if not np.any(keep):
        return TermResult(key="expression_prior", stage=stage, provenance=SOLVER_OBJECTIVE,
                          weight=weight, unit="coefficient^2", n_valid=0, n_total=int(x.size),
                          denominator=denom, status="no_valid_samples",
                          notes="no finite coefficient; omitted, NOT zero")
    raw = float(np.sum(x[keep] ** 2) / denom)
    return TermResult(
        key="expression_prior",
        stage=stage,
        provenance=SOLVER_OBJECTIVE,
        raw=raw,
        weight=weight,
        value=weight * raw,
        unit="coefficient^2",
        n_valid=int(keep.sum()),
        n_total=int(x.size),
        denominator=denom,
        notes=f"sum(expr^2) / (n_frames*10) with n_frames={frames} — NOT the component count",
    )


def sum_squared_regularizer(residual_rows: np.ndarray, *, stage: str,
                            key: str = "fem_total_regularizer",
                            weight: float = 1.0) -> TermResult:
    """``||R f + bias||^2`` for the stacked regulariser (linear, not robust)."""
    r = np.asarray(residual_rows, dtype=float).reshape(-1)
    r, keep = _finite(r, None)
    if not np.any(keep):
        return TermResult(key=key, stage=stage, provenance=LINEARISED_FALLBACK, weight=weight,
                          unit="mm^2", n_valid=0, n_total=int(r.size), denominator=0.0,
                          status="no_valid_samples", notes="no regulariser row; omitted, NOT zero")
    raw = float(np.sum(r[keep] ** 2))
    return TermResult(
        key=key,
        stage=stage,
        provenance=LINEARISED_FALLBACK,
        raw=raw,
        weight=weight,
        value=weight * raw,
        unit="mm^2",
        n_valid=int(keep.sum()),
        n_total=int(r.size),
        denominator=float(keep.sum()),
    )


LINEARISED_FALLBACK = "linearised_objective"


def field_equation_term(
    A_rows: np.ndarray,
    b_mm: np.ndarray,
    rw: np.ndarray,
    *,
    field_mm: np.ndarray | None = None,
    stage: str,
    key: str = "depth_field_equation",
) -> TermResult:
    """Weighted depth equations of ``solve_field`` at a given field.

    ``value = ||rw * (A f - b)||^2``. When ``field_mm`` is ``None`` the *target*
    inconsistency ``||rw*b||^2`` is reported instead, and the result is flagged.
    """
    A = np.asarray(A_rows, dtype=float)
    b = np.asarray(b_mm, dtype=float).reshape(-1)
    w = np.asarray(rw, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != b.size or b.size != w.size:
        raise ValueError("A must be (m,k), b and rw must be (m,)")
    if field_mm is None:
        residual = b
        note = "no field supplied: reports target inconsistency ||rw*b||^2, not the solved residual"
    else:
        f = np.asarray(field_mm, dtype=float).reshape(-1)
        if f.size != A.shape[1]:
            raise ValueError("field length must equal A.shape[1]")
        residual = (A @ f) - b
        note = "||rw*(A f - b)||^2 at the supplied field"
    scaled = residual * w
    scaled, keep = _finite(scaled, None)
    if not np.any(keep):
        return TermResult(key=key, stage=stage, provenance=LINEARISED_FALLBACK, unit="mm^2",
                          n_valid=0, n_total=int(b.size), denominator=0.0,
                          status="no_valid_samples", notes="no finite equation; NOT zero")
    raw = float(np.sum(scaled[keep] ** 2))
    return TermResult(
        key=key,
        stage=stage,
        provenance=LINEARISED_FALLBACK,
        raw=raw,
        weight=None,
        value=raw,
        unit="mm^2",
        n_valid=int(keep.sum()),
        n_total=int(b.size),
        denominator=float(keep.sum()),
        notes=note,
        extras={"mean_abs_scaled_residual_mm": float(np.mean(np.abs(scaled[keep])))},
    )


# ---------------------------------------------------------------------------
# fixed-ray nonlinear energy (motion_v2 acceptance + geometry_d diagnostics)
# ---------------------------------------------------------------------------
def fixed_ray_energy_mm2(
    depth_residual_mm: np.ndarray,
    covered: np.ndarray,
    *,
    policy: str = "strict-coverage",
    stage: str,
    aggregation_weights: np.ndarray | None = None,
) -> float | None:
    """Per-frame fixed-ray MSE, mirroring ``geometry_d.fixed_ray_energy``.

    Missing rays are charged a penalty (8 mm strict / 32 mm accounted), never
    dropped: that is precisely why a shrinking ray set cannot look "better".
    Returns ``None`` when there is no ray at all.
    """
    residual = np.asarray(depth_residual_mm, dtype=float).reshape(-1)
    covered = np.asarray(covered, dtype=bool).reshape(-1)
    if residual.size != covered.size:
        raise ValueError("residual and coverage must have the same length")
    if residual.size == 0:
        return None
    penalty = 32.0 if policy == "accounted-research" else 8.0
    scored = np.where(covered, residual, penalty)
    scored, keep = _finite(scored, None)
    if not np.any(keep):
        return None
    if aggregation_weights is not None:
        weights = np.asarray(aggregation_weights, dtype=float).reshape(-1)
        if weights.size != scored.size:
            raise ValueError("aggregation weights must match ray count")
        return float(np.average(scored[keep] ** 2, weights=weights[keep]))
    return float(np.mean(scored[keep] ** 2))


def aggregate_fixed_ray(frame_scores: list[float], ray_counts: list[int],
                        policy: str = "strict-coverage") -> float | None:
    """``geometry_d.aggregate_fixed_ray_energy``: equal weight per frame (strict)."""
    pairs = [(s, c) for s, c in zip(frame_scores, ray_counts) if s is not None and c > 0]
    if not pairs:
        return None
    scores = np.asarray([p[0] for p in pairs], dtype=float)
    counts = np.asarray([p[1] for p in pairs], dtype=float)
    if policy == "accounted-research":
        return float(np.average(scores, weights=counts))
    return float(np.mean(scores))


def strict_acceptance(
    *,
    mse_before: float,
    mse_after: float,
    missing_before: int,
    missing_after: int,
) -> dict[str, Any]:
    """The exact strict-coverage acceptance predicate from geometry_d."""
    coverage_ok = missing_after <= missing_before
    mse_ok = mse_after <= mse_before * 1.02 + 0.002
    return {
        "coverage_ok": bool(coverage_ok),
        "mse_ok": bool(mse_ok),
        "accepted": bool(coverage_ok and mse_ok),
        "rule": "missing_after <= missing_before AND mse_after <= mse_before*1.02 + 0.002",
    }


def motion_frame_acceptance(
    *,
    mse_before: float,
    mse_after: float,
    required_rays: int,
    covered_required: int,
    minimum_rays: int = 30,
) -> dict[str, Any]:
    """motion_v2's own rule: ``mse_after <= mse_before + 1e-8`` AND full coverage."""
    if required_rays < minimum_rays:
        return {
            "accepted": False,
            "reason": "insufficient_fixed_rays",
            "rule": f"required_rays < {minimum_rays}: original motion retained (alpha=0)",
        }
    full = covered_required >= required_rays
    ok = mse_after <= mse_before + 1e-8
    return {
        "accepted": bool(ok and full),
        "reason": None if (ok and full) else ("mse_increased" if not ok else "ray_coverage_lost"),
        "rule": "mse_after <= mse_before + 1e-8 AND every required ray is covered",
    }


# ---------------------------------------------------------------------------
# diagnostics (ours, NOT the algorithm's loss)
# ---------------------------------------------------------------------------
def metric_summary(values_mm: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """MAE / RMSE / P95 / max / count on *valid* samples only."""
    v = np.asarray(values_mm, dtype=float).reshape(-1)
    v, keep = _finite(v, mask)
    if not np.any(keep):
        return {
            "n_valid": 0,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_mm": None,
            "median_mm": None,
            "max_mm": None,
            "status": "no_valid_samples",
        }
    a = np.abs(v[keep])
    return {
        "n_valid": int(keep.sum()),
        "mae_mm": float(np.mean(a)),
        "rmse_mm": float(np.sqrt(np.mean(a**2))),
        "p95_mm": float(np.percentile(a, 95)),
        "median_mm": float(np.median(a)),
        "max_mm": float(a.max()),
        "status": "ok",
    }


def coverage_summary(covered: np.ndarray, required: np.ndarray | None = None) -> dict[str, Any]:
    """Coverage fraction plus the absolute ray counts that make it meaningful."""
    c = np.asarray(covered, dtype=bool).reshape(-1)
    if required is not None:
        r = np.asarray(required, dtype=bool).reshape(-1)
        if r.size != c.size:
            raise ValueError("required mask must match coverage length")
        subset = c & r
        total = int(r.sum())
        hit = int(subset.sum())
    else:
        total = int(c.size)
        hit = int(c.sum())
    if total == 0:
        return {"n_required": 0, "n_covered": 0, "coverage": None, "status": "no_valid_samples"}
    return {
        "n_required": total,
        "n_covered": hit,
        "n_missing": total - hit,
        "coverage": hit / total,
        "status": "ok",
    }


def version_guard(manifest_version: str | None, expected: str) -> dict[str, Any]:
    """Never claim loss numbers for a mesh produced by a different build."""
    return {
        "manifest_version": manifest_version,
        "expected_version": expected,
        "matches": manifest_version == expected,
        "note": None if manifest_version == expected else
        "run artefact version differs from this tool's expectation; treat recomputed losses as indicative",
    }
