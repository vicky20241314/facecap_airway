"""Read-only adapter for the direct neck stage and the tiered terminal targets.

The loss analyser never runs a reconstruction.  For the neck it reads what the
builder now persists (``loss_direct.json``), plus the existing acceptance
artefacts (``quick_check.json`` / ``verification.json``) and the base dense
report that already feeds the main adapter.

Tiers follow the same rules as everywhere else:

* the per-iteration ``energy`` is the solver objective (sum of its parts);
* the target-field residual after the solve is a **diagnostic**, not a second
  objective, and is never added to the energy;
* acceptance gates are booleans, so they are reported as a structured
  assessment and never converted into a fake numeric loss (no 1/0 "loss").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .objectives import DIAGNOSTIC, SOLVER_OBJECTIVE
from .recorder import IterationRecord, Recorder

LOSS_SCHEMA = "neck_direct.loss.v1"


def _load(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def find_neck_dir(result_dir: Path) -> Path | None:
    """Locate the neck directory that produced the delivery.

    ``quick_check.json.chosen`` is authoritative when present; the fallbacks
    cover older runs without a quick check.
    """
    result_dir = Path(result_dir)
    quick = _load(result_dir / "quick_check.json") or {}
    chosen = quick.get("chosen")
    if chosen:
        path = Path(chosen)
        if path.is_dir():
            return path
    for name in ("neck_direct_v2_prior", "neck_direct_v2", "neck_direct"):
        path = result_dir.parent / name
        if path.is_dir():
            return path
    path = result_dir / "neck_direct"
    return path if path.is_dir() else None


def read_loss_direct(neck_dir: Path | None) -> dict | None:
    if neck_dir is None:
        return None
    loss = _load(Path(neck_dir) / "loss_direct.json")
    if not loss or loss.get("schema") != LOSS_SCHEMA:
        return None
    return loss


def ingest_neck_direct(recorder: Recorder, neck_dir: Path | None) -> dict:
    """Record the real fit_unified trace and the terminal target residual.

    Returns a small summary for the caller; missing files become explicit gaps.
    """
    loss = read_loss_direct(neck_dir)
    if loss is None:
        recorder.gap(
            "neck_direct",
            "loss_direct.json not found: the neck build predates record-only "
            "persistence, so no per-iteration solver objective exists. Rebuild "
            "the neck with the current tools.build_neck_direct_v2 to obtain it; "
            "the final value is not back-filled from the delivered mesh",
            expected_tier=SOLVER_OBJECTIVE,
            source_hint=str(Path(neck_dir) / "loss_direct.json") if neck_dir else "",
        )
        return {"present": False, "states": {}}

    summary: dict[str, Any] = {"present": True, "states": {}, "path": str(Path(neck_dir) / "loss_direct.json")}
    params = loss.get("params") or {}
    for state in ("closed", "open"):
        block = (loss.get("states") or {}).get(state) or {}
        stage = f"neck_direct.fit_unified.{state}"
        iterations = block.get("iterations") or []
        if not iterations:
            recorder.gap(
                stage,
                f"no iteration was recorded for the {state} solve "
                f"(stop_reason={block.get('stop_reason')!r}); nothing is invented",
                expected_tier=SOLVER_OBJECTIVE,
            )
        for row in iterations:
            terms = {
                key: _num(value) for key, value in (row.get("energy_parts") or {}).items()
                if value is not None
            }
            recorder.add(IterationRecord(
                stage=stage,
                iteration=int(row.get("iter", len(iterations))),
                tier=SOLVER_OBJECTIVE,
                phase="fit_unified",
                value=_num(row.get("energy")),
                terms=terms,
                decision="accepted" if row.get("accepted") else "rejected",
                step_scale=_num(row.get("line_search_alpha")),
                source="recorded",
                notes=[
                    "solver objective = sum of the five energy parts; all parts split the "
                    "sum, so they may be added inside this iteration only",
                    "Gauss-Newton step under radial/tangential trust-region bounds",
                ],
                extras={
                    "state": state,
                    "samples": row.get("samples"),
                    "samples_vertex": row.get("samples_vertex"),
                    "samples_face": row.get("samples_face"),
                    "step_max_mm": row.get("step_max_mm"),
                    "grad_max": row.get("grad_max"),
                    "grad_rel": row.get("grad_rel"),
                    "energy_rel": row.get("energy_rel"),
                    "max_d_mm": row.get("max_d_mm"),
                },
            ))

        residual = block.get("residual_after") or {}
        if residual:
            recorder.add(IterationRecord(
                stage=f"neck_direct.target_residual.{state}",
                iteration=int(block.get("n_iter") or 0),
                tier=DIAGNOSTIC,
                phase="after_solve",
                value=_num(residual.get("vertex_rms_mm")),
                terms={key: _num(value) for key, value in residual.items()
                       if _num(value) is not None},
                decision="after_solve",
                source="recorded",
                notes=[
                    "target-field residual recomputed at the end of the same solve; "
                    "diagnostic only, NOT added to the energy",
                    "vertex and in-face (quadrature) residual are separate samples",
                ],
                extras={"state": state, "n_iter": block.get("n_iter")},
            ))

        recorder.declare_stage(
            stage,
            has_iterations=bool(iterations),
            notes=[
                f"stop_reason={block.get('stop_reason')}; converged={block.get('converged')}; "
                f"n_iter={block.get('n_iter')}",
                f"free={block.get('n_free')} fixed={block.get('n_fixed')} "
                f"data_samples={block.get('data_samples')}",
                "remesh passes after this solve are geometry proposals with a crease "
                "rollback; they are not optimisation steps and do not appear as a loss curve",
            ],
        )
        summary["states"][state] = {
            "n_iter": block.get("n_iter"),
            "stop_reason": block.get("stop_reason"),
            "converged": block.get("converged"),
            "energy": _num(block.get("energy")),
            "energy_parts": block.get("energy_parts"),
            "residual_after": residual,
            "max_d_mm": block.get("max_d_mm"),
            "n_free": block.get("n_free"),
            "n_fixed": block.get("n_fixed"),
        }
    recorder.declare_stage(
        "neck_direct",
        has_iterations=any(bool((loss.get("states") or {}).get(s, {}).get("iterations"))
                           for s in ("closed", "open")),
        notes=[
            f"whitelisted parameters recorded by the build: "
            f"structure_gain={params.get('structure_gain')}, "
            f"bridge_width_mm={params.get('bridge_width_mm')}, "
            f"seam_taper_ratio={params.get('seam_taper_ratio')}, "
            f"mu_t_bridge={params.get('mu_t_bridge')}, "
            f"mu_t_body={params.get('mu_t_body')}, "
            f"anchor_near_A={params.get('anchor_near_A')}",
            "marker-derived t_center / theta / amplitude are inputs and are not "
            "candidate parameters",
            str(loss.get("tier_note") or ""),
        ],
    )
    return summary


def base_terminal(dense_dir: Path) -> dict:
    """Last real fixed-ray diagnostic of the base dense report.

    The base dense stage has no mean over its five Adam terms persisted; what it
    does persist per outer step is the frozen linear objective, the fixed-ray
    MSE recomputed by ray casting and the acceptance verdict.  Those are the
    terminal targets used for ranking a base-parameter candidate.
    """
    report = _load(Path(dense_dir) / "report.json") or {}
    field_logs = report.get("field_logs") or {}
    terminal: dict[str, Any] = {"present": False, "source": str(Path(dense_dir) / "report.json")}
    for name, entry in field_logs.items():
        history = entry.get("history") if isinstance(entry, dict) else None
        rows = history if isinstance(history, list) else [entry]
        for row in rows:
            if not isinstance(row, dict):
                continue
            diagnostic = row.get("fixed_ray_diagnostic")
            if not diagnostic:
                continue
            terminal = {
                "present": True,
                "field": name,
                "outer": row.get("outer"),
                "stage": row.get("stage"),
                "objective_before": _num(row.get("objective_before")),
                "objective_after": _num(row.get("objective_after")),
                "step_accepted": row.get("step_accepted"),
                "fixed_ray_after_mse_mm2": _num(diagnostic.get("after_mse_mm2")),
                "fixed_ray_before_mse_mm2": _num(diagnostic.get("before_mse_mm2")),
                "missing_after": diagnostic.get("missing_after"),
                "missing_before": diagnostic.get("missing_before"),
                "missing_cap": diagnostic.get("missing_cap"),
                "missing_allowance": diagnostic.get("missing_allowance"),
                "missing_stage_baseline": diagnostic.get("missing_stage_baseline"),
                "ray_count": diagnostic.get("ray_count"),
                "policy": diagnostic.get("policy"),
                "allowance": diagnostic.get("allowance"),
                "safety_reason": (row.get("safety") or {}).get("reason"),
                "source": str(Path(dense_dir) / "report.json"),
            }
    if not terminal["present"]:
        terminal["reason"] = ("no field_logs entry carries a fixed_ray_diagnostic: "
                              "the dense stage either did not run or reused a field")
    return terminal


def neck_gates(result_dir: Path | None, neck_dir: Path | None) -> dict:
    """Acceptance gates exactly as the delivery recorded them (booleans)."""
    result_dir = Path(result_dir) if result_dir else None
    quick_payload = _load(result_dir / "quick_check.json") if result_dir else None
    quick = (quick_payload or {}).get("quick") or {}
    verification = _load(Path(neck_dir) / "verification.json") if neck_dir else None
    gates: dict[str, Any] = {
        "present": bool(quick or verification),
        "quick_pass": quick.get("pass"),
        "protection": quick.get("protection") or {},
        "topology": quick.get("topology") or {},
        "shared_topology": quick.get("shared_topology"),
        "signed_crease": quick.get("signed_crease") or {},
        "upper_band": quick.get("upper_band_0.925_0.97") or {},
        "structure": quick.get("structure") or {},
        "vertices_faces": {"vertices": quick.get("vertices"), "faces": quick.get("faces")},
    }
    if verification:
        gates["verification"] = {
            "failed": verification.get("failed"),
            "gates": verification.get("gates"),
        }
    crease = gates["signed_crease"]
    values = [
        _num((crease.get(state) or {}).get("full_positive_max_deg"))
        for state in ("closed", "open")
    ]
    values = [value for value in values if value is not None]
    gates["connection_max_deg"] = max(values) if values else None
    return gates


def marker_semantics(landmarks_path: Path | None) -> dict:
    """Marker-derived centre and amplitude semantics, per state.

    A candidate must keep these identical: the marker is the measurement, not a
    knob.  Any difference is a hard rejection, not a ranking penalty.
    """
    payload = _load(Path(landmarks_path)) if landmarks_path else None
    if not payload:
        return {"present": False, "reason": "neck_landmarks.json not supplied"}
    out: dict[str, Any] = {"present": True, "states": {}}
    for state, block in (payload.get("states") or {}).items():
        centre = block.get("larynx_center") or {}
        marker = block.get("marker") or {}
        print_block = marker.get("print") or {}
        out["states"][state] = {
            "status": block.get("status"),
            "source": block.get("source"),
            "frame_id": block.get("frame_id"),
            "target_id": print_block.get("target_id"),
            "black_square_mm": _num(print_block.get("black_square_mm")),
            "size_source": print_block.get("size_source"),
            "t_center_used": _num(centre.get("t_center_used")),
            "theta_used_deg": _num(centre.get("theta_used_deg")),
            "t_gate": _num(centre.get("t_gate")),
        }
    return out


def downstream_input_probe(batch_dir: Path | None) -> dict:
    """Read the downstream input-validation probe, never the registration result.

    A candidate is only allowed to reach the downstream tool if this passes; its
    geometry verdict is out of scope for the optimiser (the downstream algorithm
    is not modified and its pass/fail is not a tuning target).
    """
    payload = _load(Path(batch_dir) / "input_validation.json") if batch_dir else None
    if not payload:
        return {"present": False}
    problems = payload.get("problems")
    return {
        "present": True,
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "problems": problems or [],
        "source": str(Path(batch_dir) / "input_validation.json"),
    }


def assess(*, dense_dir: Path, result_dir: Path | None, neck_dir: Path | None,
           landmarks_path: Path | None = None, downstream_batch: Path | None = None) -> dict:
    """Compose the terminal assessment used by the offline optimiser."""
    loss = read_loss_direct(neck_dir)
    residual: dict[str, Any] = {}
    if loss:
        for state in ("closed", "open"):
            block = (loss.get("states") or {}).get(state) or {}
            residual[state] = block.get("residual_after") or {}
    gates = neck_gates(result_dir, neck_dir)
    markers = marker_semantics(landmarks_path)
    build = _load(Path(neck_dir) / "build_direct.json") if neck_dir else {}
    journal = (build or {}).get("journal") or {}
    remesh = {
        "split_new_total": journal.get("split_new_total"),
        "split_new_cap": journal.get("split_new_cap"),
        "vertices_original": journal.get("vertices_original"),
        "vertices_final": journal.get("vertices_final"),
        "merged_slots": journal.get("merged_slots"),
    }
    vertex_rms = [
        _num(residual.get(state, {}).get("vertex_rms_mm"))
        for state in ("closed", "open")
    ]
    vertex_rms = [value for value in vertex_rms if value is not None]
    quad_rms = [
        _num(residual.get(state, {}).get("quad_rms_mm"))
        for state in ("closed", "open")
    ]
    quad_rms = [value for value in quad_rms if value is not None]
    return {
        "base": base_terminal(dense_dir),
        "neck": {
            "present": bool(loss or gates.get("present")),
            "loss_present": bool(loss),
            "gates": gates,
            "target_residual": {
                "present": bool(residual),
                "states": residual,
                "mean_vertex_rms_mm": (sum(vertex_rms) / len(vertex_rms)) if vertex_rms else None,
                "mean_quad_rms_mm": (sum(quad_rms) / len(quad_rms)) if quad_rms else None,
            },
            "markers": markers,
            "remesh": remesh,
        },
        "downstream": downstream_input_probe(downstream_batch),
    }
