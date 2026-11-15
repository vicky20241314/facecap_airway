"""Offline loss-guided parameter search for the base reconstruction + neck reshaping.

This is an analysis-side tool.  It never runs inside the delivery path and it
never changes a default parameter: it proposes a small number of whitelisted
candidates, optionally runs them through the normal ``headrgbd rebuild`` entry,
and ranks the resulting deliveries against a fixed, tiered terminal target.

Ranking rules (fixed in this file, not tuned per run)
-----------------------------------------------------
Hard constraints must pass before a candidate is ranked at all:

* marker-derived centre/amplitude semantics unchanged (the marker is a
  measurement, not a knob);
* protection, topology, shared two-state face table, upper-band density;
* the delivery probe (``final/`` sidecar + manifest are self-contained).

The surviving candidates are compared lexicographically:

1. base dense terminal fixed-ray MSE and missing-ray count (acceptance included);
2. neck connection: maximum positive crease of the delivered states;
3. neck target residual: mean vertex, then quadrature RMS after the solve;
4. structure/density deviation: laryngeal relief vs the reference peak and the
   upper-band vertex delta.

The tiers are never summed: a lower target residual cannot buy a worse
connection, and a better connection cannot buy a worse base fixed-ray energy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import Any

from . import gap_report, neck_adapter
from .html_report import render_page, table, _figure_gallery, _fmt, _text
from .plots import write_json, write_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_CANDIDATES = 12


@dataclass(frozen=True)
class Param:
    key: str
    scope: str          # "base" | "neck"
    flag: str           # headrgbd rebuild flag
    baseline: float
    step: float
    lo: float
    hi: float
    reason: str

    def move(self, baseline: float) -> float:
        value = baseline + self.step
        return float(min(self.hi, max(self.lo, value)))


# One direction per parameter, chosen from the failure the baseline actually
# shows (connection 21.45 deg > 20 limit; delivered relief above the reference
# peak).  Both directions are not explored in round one: the budget is 12.
PARAMS: dict[str, Param] = {
    "neck.structure_gain": Param(
        "neck.structure_gain", "neck", "--neck-gain", 1.10, -0.05, 1.00, 1.20,
        "delivered laryngeal relief is above the reference peak (5.10/4.69 vs "
        "4.43/4.29 mm); a slightly lower gain tests whether the connection "
        "crease follows the over-relief",
    ),
    "neck.mu_t_bridge": Param(
        "neck.mu_t_bridge", "neck", "--neck-mu-t-bridge", 0.15, 0.05, 0.05, 0.40,
        "stronger tangential prior in the submental bridge reduces shear at the "
        "B/C crossing that carries the largest positive crease",
    ),
    "neck.anchor_near_A": Param(
        "neck.anchor_near_A", "neck", "--neck-anchor-near-a", 2.00, 0.50, 1.00, 4.00,
        "stiffer hand-over in the first 2.5 edges from the fixed head; the worst "
        "crease sits next to A",
    ),
    "neck.seam_taper_ratio": Param(
        "neck.seam_taper_ratio", "neck", "--neck-seam-taper-ratio", 2.50, 0.50, 1.50, 4.00,
        "wider metric fade from the protected head softens the seam transition",
    ),
    "neck.mu_t_body": Param(
        "neck.mu_t_body", "neck", "--neck-mu-t-body", 0.50, 0.10, 0.30, 0.90,
        "more tangential body prior suppresses lateral folding",
    ),
    "neck.bridge_width_mm": Param(
        "neck.bridge_width_mm", "neck", "--neck-bridge-width", 12.0, 1.0, 10.0, 16.0,
        "a wider bridge spreads the deformation over more edges",
    ),
    "base.motion_lowerface_weight": Param(
        "base.motion_lowerface_weight", "base", "--motion-lowerface-weight", 0.50, 0.10, 0.20, 0.90,
        "reliable open-front lower-face depth during expression correction; it "
        "changes the fitted open state that the neck solve starts from",
    ),
    "base.bending_length_mm": Param(
        "base.bending_length_mm", "base", "--bending-length-mm", 8.0, 1.0, 3.0, 12.0,
        "stiffer dense bending prior; a smoother base surface can lower the "
        "terminal fixed-ray error without touching the neck",
    ),
    "base.low_bending_length_mm": Param(
        "base.low_bending_length_mm", "base", "--low-bending-length-mm", 8.0, 1.0, 6.0, 14.0,
        "stiffer low-frequency field; tested separately from the dense bending "
        "length so the two scales stay identifiable",
    ),
}

PARAM_ORDER = (
    "neck.anchor_near_A",
    "neck.structure_gain",
    "neck.mu_t_bridge",
    "neck.seam_taper_ratio",
    "neck.mu_t_body",
    "neck.bridge_width_mm",
    "base.motion_lowerface_weight",
    "base.bending_length_mm",
    "base.low_bending_length_mm",
)


@dataclass
class Candidate:
    key: str
    overrides: dict[str, float]
    params: dict[str, float]
    note: str = ""
    run_dir: Path | None = None
    assessment: dict[str, Any] | None = None
    hard: list[dict[str, str]] = field(default_factory=list)
    rank: tuple[float, ...] | None = None
    error: str | None = None


def _read_json(path: Path | None) -> dict:
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def baseline_values(*, baseline_run: Path, baseline_neck: Path | None,
                    baseline_dense: Path | None = None) -> dict[str, float]:
    """Read the baseline values from artefacts, not from this file's defaults."""
    run_report = _read_json(Path(baseline_run) / "run_report.json")
    if not run_report and baseline_dense is not None:
        # dense = <run>/reconstruction/dense
        run_report = _read_json(Path(baseline_dense).parents[1] / "run_report.json")
    config = run_report.get("configuration") or {}
    build = _read_json(Path(baseline_neck) / "build_direct.json") if baseline_neck else {}
    neck_params = build.get("params") or {}
    values: dict[str, float] = {}
    for key, param in PARAMS.items():
        if param.scope == "base":
            raw = config.get(key.split(".", 1)[1])
        else:
            raw = neck_params.get(key.split(".", 1)[1])
        values[key] = float(raw) if isinstance(raw, (int, float)) else float(param.baseline)
    return values


def plan_candidates(values: dict[str, float], limit: int = MAX_CANDIDATES) -> list[Candidate]:
    """One-parameter coordinate moves, highest-leverage first, capped at 12."""
    limit = max(1, min(int(limit), MAX_CANDIDATES))
    out: list[Candidate] = []
    for key in PARAM_ORDER:
        if len(out) >= limit:
            break
        param = PARAMS[key]
        baseline = float(values.get(key, param.baseline))
        value = param.move(baseline)
        if math.isclose(value, baseline):
            continue
        label = f"{param.scope}_{key.split('.', 1)[1]}"
        out.append(Candidate(
            key=f"{label}_{value:g}",
            overrides={key: value},
            params={**values, key: value},
            note=param.reason,
        ))
    return out


def targeted_candidates(values: dict[str, float], gaps: dict,
                        limit: int = 2) -> list[Candidate]:
    """Candidates named by the dominant gap, at most two, one parameter each."""
    out: list[Candidate] = []
    seen: set[str] = set()
    for item in (gaps.get("targeted") or [])[:max(1, int(limit))]:
        key = item.get("key")
        if key in seen or key not in PARAMS:
            continue
        seen.add(key)
        param = PARAMS[key]
        baseline = float(values.get(key, param.baseline))
        value = param.move(baseline)
        if math.isclose(value, baseline):
            continue
        out.append(Candidate(
            key=f"{param.scope}_{key.split('.', 1)[1]}_{value:g}",
            overrides={key: value},
            params={**values, key: value},
            note=item.get("why") or param.reason,
        ))
    return out


def _delivery_probe(result_dir: Path) -> dict:
    """Structural probe of the downstream input contract (no registration)."""
    final = Path(result_dir) / "final"
    measured = Path(result_dir) / "measured"
    problems: list[str] = []
    items: dict[str, Any] = {"present": False}
    for name, folder in (("final", final), ("measured", measured)):
        sidecar = _read_json(folder / "anatomical_landmarks.json")
        manifest = _read_json(folder / "manifest.json")
        mesh_closed = folder / "head_closed.obj"
        mesh_open = folder / "head_open.obj"
        sidecar_states = (sidecar.get("states") or {}) if sidecar else {}
        state_files = {
            state: (sidecar_states.get(state) or {}).get("main_delivery_mesh_file")
            for state in ("closed", "open")
        }
        main_file = (state_files.get("closed")
                     or (sidecar.get("main_delivery_mesh_file") if sidecar else None))
        entry = {
            "sidecar": bool(sidecar),
            "manifest": bool(manifest),
            "meshes": [mesh_closed.is_file(), mesh_open.is_file()],
            "status": (sidecar.get("status") if sidecar else None),
            "main_delivery_mesh_file": main_file,
            "state_mesh_files": state_files,
            "manifest_states": sorted(((manifest.get("mesh") or {}).get("states") or {}).keys()),
        }
        if not (entry["sidecar"] and entry["manifest"] and all(entry["meshes"])):
            problems.append(f"{name}: required files missing")
        if main_file != "head_closed.obj":
            problems.append(f"{name}: main_delivery_mesh_file is not head_closed.obj")
        if state_files.get("open") not in (None, "head_open.obj"):
            problems.append(f"{name}: open state mesh file is {state_files.get('open')}")
        if entry["manifest_states"] and set(entry["manifest_states"]) != {"closed", "open"}:
            problems.append(f"{name}: manifest states are {entry['manifest_states']}")
        items[name] = entry
    items["present"] = all(items.get(k, {}).get("sidecar") for k in ("final", "measured"))
    items["problems"] = problems
    items["status"] = "pass" if items["present"] and not problems else "fail"
    return items


def _marker_equal(baseline: dict, candidate: dict) -> tuple[bool, str]:
    b_states = (baseline or {}).get("states") or {}
    c_states = (candidate or {}).get("states") or {}
    if not b_states or not c_states:
        return False, "marker semantics unavailable on one side"
    for state in ("closed", "open"):
        b = b_states.get(state) or {}
        c = c_states.get(state) or {}
        for field_name in ("status", "source", "frame_id", "target_id",
                           "black_square_mm", "t_center_used", "theta_used_deg"):
            if b.get(field_name) != c.get(field_name):
                return False, (f"{state}.{field_name}: baseline {b.get(field_name)!r} "
                               f"!= candidate {c.get(field_name)!r}")
    return True, "identical marker centre, size and state per side"


def _split_within_budget(remesh: dict) -> bool:
    total = remesh.get("split_new_total")
    cap = remesh.get("split_new_cap")
    return (isinstance(total, (int, float)) and isinstance(cap, (int, float))
            and total <= cap)


def _split_detail(remesh: dict) -> str:
    return (f"split_new_total={remesh.get('split_new_total')} "
            f"cap={remesh.get('split_new_cap')} "
            f"vertices={remesh.get('vertices_original')}->{remesh.get('vertices_final')}")


def hard_constraints(baseline: dict, candidate: dict, probe: dict) -> list[dict[str, str]]:
    gates = (candidate.get("neck") or {}).get("gates") or {}
    protection = gates.get("protection") or {}
    topology = gates.get("topology") or {}
    upper = gates.get("upper_band") or {}
    equal, why = _marker_equal((baseline.get("neck") or {}).get("markers") or {},
                               (candidate.get("neck") or {}).get("markers") or {})
    rows = [
        {
            "name": "marker semantics",
            "status": "pass" if equal else "fail",
            "detail": why,
        },
        {
            "name": "protection",
            "status": ("pass" if protection.get("pass") and
                       (protection.get("fixed_max_mm") or 9e9) <= 0.002 else "fail"),
            "detail": f"fixed_max={protection.get('fixed_max_mm')} mm",
        },
        {
            "name": "topology",
            "status": "pass" if topology.get("pass") and gates.get("shared_topology") else "fail",
            "detail": f"shared_topology={gates.get('shared_topology')}",
        },
        {
            "name": "upper-band density",
            "status": "pass" if upper.get("no_increase") else "fail",
            "detail": f"no_increase={upper.get('no_increase')}",
        },
        {
            "name": "split budget",
            "status": ("pass" if _split_within_budget(
                (candidate.get("neck") or {}).get("remesh") or {}) else "fail"),
            "detail": _split_detail((candidate.get("neck") or {}).get("remesh") or {}),
        },
        {
            "name": "delivery probe",
            "status": "pass" if probe.get("status") == "pass" else "fail",
            "detail": "; ".join(probe.get("problems") or []) or "final/measured self-contained",
        },
    ]
    return rows


def ranking_key(assessment: dict, reference_peak: float | None = None) -> tuple[float, ...]:
    base = assessment.get("base") or {}
    neck = assessment.get("neck") or {}
    gates = neck.get("gates") or {}
    residual = neck.get("target_residual") or {}
    structure = gates.get("structure") or {}
    upper = gates.get("upper_band") or {}
    relief_error = []
    for state in ("closed", "open"):
        delivered = ((structure.get(state) or {}).get("larynx_local") or {}).get("relief_mm")
        if isinstance(delivered, (int, float)) and isinstance(reference_peak, (int, float)):
            relief_error.append(abs(float(delivered) - float(reference_peak)))
    try:
        density_delta = float((upper.get("full") or {}).get("vertices", 0)) - float(
            (upper.get("input") or {}).get("vertices", 0))
    except (TypeError, ValueError):
        density_delta = None

    def num(value: Any) -> float:
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.inf

    return (
        num(base.get("fixed_ray_after_mse_mm2")),
        num(base.get("missing_after")),
        num(gates.get("connection_max_deg")),
        num(residual.get("mean_vertex_rms_mm")),
        num(residual.get("mean_quad_rms_mm")),
        (sum(relief_error) / len(relief_error)) if relief_error else math.inf,
        num(density_delta),
    )


def _reference_peak_from_build(neck_dir: Path | None) -> float | None:
    build = _read_json(Path(neck_dir) / "build_direct.json") if neck_dir else {}
    primary = (((build.get("structure") or {}).get("reference") or {}).get("primary") or {})
    value = primary.get("relief_mm")
    return float(value) if isinstance(value, (int, float)) else None


def evaluate(*, result_dir: Path, dense_dir: Path, landmarks: Path | None,
             baseline_assessment: dict) -> tuple[dict, list[dict[str, str]], dict]:
    neck_dir = neck_adapter.find_neck_dir(result_dir)
    assessment = neck_adapter.assess(
        dense_dir=dense_dir,
        result_dir=result_dir,
        neck_dir=neck_dir,
        landmarks_path=landmarks,
    )
    probe = _delivery_probe(result_dir)
    hard = hard_constraints(baseline_assessment, assessment, probe)
    return assessment, hard, probe


def _rank_key(assessment: dict) -> tuple[float, ...]:
    return ranking_key(assessment, (assessment.get("neck") or {}).get("reference_peak_mm"))


def _slug(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)[:80]


def _candidate_command(candidate: Candidate, *, capture: Path | None, dense: Path | None,
                       out_dir: Path, landmarks: Path | None,
                       freeze_landmarks: bool, neck_only_dense: bool) -> list[str]:
    base_change = any(key.startswith("base.") for key in candidate.overrides)
    if neck_only_dense and not base_change:
        if dense is None:
            raise ValueError("--neck-only-dense needs --baseline-dense")
        base_args = ["--dense", str(dense)]
    elif capture is not None:
        base_args = ["--capture", str(capture)]
    elif dense is not None:
        # No capture supplied: a neck-only run off the baseline dense.  This
        # has no prepared/ folder, so T3 cannot re-check the marker and the
        # delivery will be not_usable; the report says so instead of pretending.
        base_args = ["--dense", str(dense)]
    else:
        raise ValueError("a candidate needs --capture (or --baseline-dense with --neck-only-dense)")
    command = [sys.executable, "-m", "head_rgbd", "rebuild", *base_args,
               "--output", str(out_dir), "--neck-variant", "prior", "--no-render"]
    if freeze_landmarks and landmarks is not None:
        # Deliberately not the default: forwarding the baseline sidecar would
        # make the "marker semantics unchanged" check vacuous.  Candidates
        # re-run the live prepared recheck unless the caller freezes it.
        command += ["--neck-landmarks", str(landmarks)]
    for key, value in candidate.overrides.items():
        command += [PARAMS[key].flag, f"{value:g}"]
    return command


def _run_candidate(candidate: Candidate, *, capture: Path | None, dense: Path | None,
                   out_dir: Path, landmarks: Path | None, freeze_landmarks: bool,
                   neck_only_dense: bool, timeout_s: int,
                   env: dict[str, str]) -> tuple[int, str]:
    command = _candidate_command(candidate, capture=capture, dense=dense,
                                 out_dir=out_dir, landmarks=landmarks,
                                 freeze_landmarks=freeze_landmarks,
                                 neck_only_dense=neck_only_dense)
    # The rebuild entry refuses an existing output directory (that is how it
    # avoids overwriting a previous run), so the candidate directory must not
    # be created here.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command, cwd=str(PROJECT_ROOT), env=env, capture_output=True,
        text=True, encoding="utf-8", timeout=timeout_s,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.log").write_text(
        "$ " + " ".join(command) + "\n\n"
        + (completed.stdout or "") + "\n--- stderr ---\n" + (completed.stderr or ""),
        encoding="utf-8",
    )
    return completed.returncode, " ".join(command)


def _trace_figure(loss: dict, out_path: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - figures are optional
        return None
    states = [s for s in ("closed", "open") if (loss.get("states") or {}).get(s)]
    if not states:
        return None
    fig, axes = plt.subplots(1, len(states), figsize=(5.4 * len(states), 3.4), dpi=140)
    if len(states) == 1:
        axes = [axes]
    for ax, state in zip(axes, states):
        rows = [row for row in loss["states"][state].get("iterations", [])
                if isinstance(row.get("energy"), (int, float))]
        accepted = [(row["iter"], row["energy"]) for row in rows if row.get("accepted")]
        rejected = [(row["iter"], row["energy"]) for row in rows if not row.get("accepted")]
        if accepted:
            ax.plot([x for x, _ in accepted], [y for _, y in accepted],
                    marker="o", markersize=3, label="accepted")
        if rejected:
            ax.plot([x for x, _ in rejected], [y for _, y in rejected],
                    marker="x", linestyle="none", label="rejected")
        ax.set_title(f"neck_direct.fit_unified.{state}", fontsize=9)
        ax.set_xlabel("iteration")
        ax.set_ylabel("solver objective (sum of parts)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _jsonable_rank(rank: tuple[float, ...] | None) -> list[float | None] | None:
    if rank is None:
        return None
    return [None if math.isinf(value) else value for value in rank]


def _html(*, baseline_assessment: dict, baseline_label: str, candidates: list[Candidate],
          plan_only: bool, figures: list[str], execute: bool,
          winner: Candidate | None, notes: list[str], gap: dict | None = None) -> str:
    sections: list[dict[str, str]] = []

    rules = (
        "<p>分层终止目标，按顺序比较；任何一层都不能用下一层来交换：</p>"
        "<ol class='tight'>"
        "<li><b>基重建</b>：dense report 里最后一个真实 outer 的固定射线 MSE 与缺失射线数，"
        "以及该步是否被接受；<b>fit_c/motion_v2 没有持久化分项，不反推</b>。</li>"
        "<li><b>颈部连接</b>：交付两态固定射线外的有符号折痕（正向最大，闭/张取大）。</li>"
        "<li><b>颈部目标残差</b>：solve 结束时目标场残差 vertex RMS，再比 quadrature RMS。</li>"
        "<li><b>结构与密度</b>：喉结局部 relief 与参考峰值的偏差；上颈带是否加密。</li>"
        "</ol>"
        "<div class='callout'>硬约束先于排名：标记语义（t/θ/尺寸/状态）必须逐态不变、"
        "保护集、拓扑与两态面表、上颈密度、交付 sidecar/manifest 自包含。任何一项 "
        "fail 或 unverified 的候选不参与排名。四档数值不可相加。</div>"
        "<p class='note'>本工具只读分析 + 可选地重跑候选；不改变默认参数、不接受条件、"
        "不修改下游算法。候选输出写入新目录，不覆盖已有运行。</p>"
    )
    sections.append({"id": "rules", "heading": "0. 排序规则与硬约束", "html": rules})

    def base_rows(assessment: dict) -> list[list[str]]:
        base = assessment.get("base") or {}
        neck = assessment.get("neck") or {}
        gates = neck.get("gates") or {}
        residual = neck.get("target_residual") or {}
        markers = neck.get("markers") or {}
        quick = gates.get("quick_pass")
        if isinstance(quick, dict):
            quick = ", ".join(f"{key}={value}" for key, value in quick.items())
        rows = [
            ["base fixed-ray after MSE (mm²)", _fmt(base.get("fixed_ray_after_mse_mm2")),
             f"missing after {base.get('missing_after')}/{base.get('ray_count')}; "
             f"accepted={base.get('step_accepted')}; stage={base.get('stage')}"],
            ["base linearised objective before → after",
             f"{_fmt(base.get('objective_before'))} → {_fmt(base.get('objective_after'))}",
             "frozen-correspondence linear value; not comparable with the fixed-ray MSE"],
            ["neck connection max positive crease (deg)", _fmt(gates.get("connection_max_deg")),
             "closed/open maximum of full and B positive dihedral change"],
            ["neck target residual mean vertex RMS (mm)", _fmt(residual.get("mean_vertex_rms_mm")),
             "diagnostic recomputed at the end of the solve; not added to the energy"],
            ["neck target residual mean quad RMS (mm)", _fmt(residual.get("mean_quad_rms_mm")),
             "in-face quadrature samples; separate from vertex samples"],
            ["neck quick_check pass", _fmt(gates.get("quick_pass")),
             f"shared_topology={gates.get('shared_topology')}"],
        ]
        if markers.get("states"):
            for state, block in markers["states"].items():
                rows.append([
                    f"marker {state}", _fmt(block.get("t_center_used")),
                    f"θ={block.get('theta_used_deg')}°, id={block.get('target_id')}, "
                    f"black={block.get('black_square_mm')} mm, source={block.get('source')}",
                ])
        return rows

    sections.append({
        "id": "baseline",
        "heading": "1. 基线状态",
        "html": f"<p>基线：{_text(baseline_label)}</p>"
                + table(["项", "值", "说明"], base_rows(baseline_assessment)),
    })

    if gap:
        gap_rows = []
        for entry in (gap.get("solver") or []) + (gap.get("acceptance") or []):
            gap_rows.append([
                _text(entry.get("key")), _text(entry.get("status")),
                _fmt(entry.get("current")), _fmt(entry.get("target")),
                _fmt(entry.get("gap")), _text(entry.get("unit")),
            ])
        dominant = gap.get("dominant") or {}
        targeted_rows = [
            [_text(item.get("key")), _text(item.get("why"))]
            for item in gap.get("targeted") or []
        ]
        gap_html = (
            ("<p>主导 gap：<code>" + escape(str(dominant.get("key"))) + "</code>，状态 "
             + escape(str(dominant.get("status"))) + "，当前 "
             + _fmt(dominant.get("current")) + "，目标 " + _fmt(dominant.get("target"))
             + "，gap " + _fmt(dominant.get("gap")) + "。</p>"
             if dominant else "<p>没有主导 gap。</p>")
            + (table(["白名单参数", "为什么是它"], targeted_rows)
               if targeted_rows else
               "<p class='note'>主导 gap 不由白名单参数控制，本轮不提出定向候选。</p>")
            + table(["key", "状态", "当前", "目标", "gap", "单位"], gap_rows,
                    numeric={2, 3, 4})
            + "<p class='note'>求解器层与验收层分别报告，不相加；正 gap 是余量，负 gap 是超出量。</p>"
        )
        sections.append({"id": "gap", "heading": "2. 逐阶段终止 gap 与定向选择",
                         "html": gap_html})

    plan_rows = []
    for index, candidate in enumerate(candidates, 1):
        overrides = ", ".join(f"<code>{escape(key)}={value:g}</code>"
                              for key, value in candidate.overrides.items())
        plan_rows.append([str(index), escape(candidate.key), overrides, escape(candidate.note)])
    sections.append({
        "id": "plan",
        "heading": "3. 候选计划（白名单，≤12）",
        "html": table(["#", "候选", "改动", "为什么"], plan_rows)
        + f"<p class='note'>只有执行了 {len([c for c in candidates if c.assessment])} 个候选；"
          f"{'本次仅生成计划，未运行（--execute 关闭）。' if plan_only else ''}</p>",
    })

    result_rows = []
    for index, candidate in enumerate(candidates, 1):
        assessment = candidate.assessment
        if candidate.error:
            result_rows.append([str(index), escape(candidate.key), "error", escape(candidate.error),
                                "—", "—", "—", "—"])
            continue
        if not assessment:
            result_rows.append([str(index), escape(candidate.key), "planned", "—", "—", "—", "—", "—"])
            continue
        base = assessment.get("base") or {}
        neck = assessment.get("neck") or {}
        hard_fail = [row["name"] for row in candidate.hard if row["status"] != "pass"]
        result_rows.append([
            str(index), escape(candidate.key),
            "rejected" if hard_fail else "ranked",
            escape(", ".join(hard_fail) or "all pass"),
            _fmt(base.get("fixed_ray_after_mse_mm2")),
            _fmt((neck.get("gates") or {}).get("connection_max_deg")),
            _fmt((neck.get("target_residual") or {}).get("mean_vertex_rms_mm")),
            _fmt((neck.get("target_residual") or {}).get("mean_quad_rms_mm")),
        ])
    sections.append({
        "id": "results",
        "heading": "4. 候选结果（硬约束通过后才参与排名）",
        "html": table(["#", "候选", "状态", "硬约束失败项", "base fixed-ray MSE (mm²)",
                       "连接折痕 (deg)", "vertex RMS (mm)", "quad RMS (mm)"], result_rows,
                      numeric={4, 5, 6, 7}),
    })

    hard_rows = []
    for candidate in candidates:
        if not candidate.assessment:
            continue
        for row in candidate.hard:
            hard_rows.append([escape(candidate.key), escape(row["name"]),
                              escape(row["status"]), escape(row["detail"])])
    sections.append({
        "id": "hard",
        "heading": "5. 硬约束矩阵",
        "html": table(["候选", "约束", "状态", "细节"], hard_rows)
        if hard_rows else "<p>没有已评估的候选。</p>",
    })

    figure_html = _figure_gallery(figures) if figures else "<p class='note'>没有已执行候选的轨迹图。</p>"
    sections.append({"id": "traces", "heading": "6. 已执行候选的求解器轨迹", "html": figure_html})

    conclusion = "<ul class='tight'>"
    if winner is not None:
        conclusion += (f"<li>推荐：<code>{escape(winner.key)}</code>（字典序最优且硬约束全部通过）。</li>")
    elif execute:
        conclusion += "<li>没有候选同时通过全部硬约束；不要用牺牲标记/保护/拓扑来换取指标。</li>"
    else:
        conclusion += "<li>本次只生成计划；运行候选需要 <code>--execute</code>。</li>"
    for note in notes:
        conclusion += f"<li>{escape(note)}</li>"
    conclusion += (
        "<li>目标残差、连接折痕、固定射线 MSE 属于不同档位，不能相加或互相抵消。</li>"
        "<li>下游几何 pass/fail 不在本优化器范围内：只对交付做输入接口探针，"
        "不修改气管、不缩小气管、不改下游算法。</li>"
        "</ul>"
    )
    sections.append({"id": "conclusion", "heading": "7. 结论与边界", "html": conclusion})

    subtitle = (f"基线 <code>{escape(baseline_label)}</code> · "
                f"候选 {len(candidates)} 个 · "
                f"{'已执行 ' + str(len([c for c in candidates if c.assessment])) + ' 个' if execute else '仅计划'} · "
                f"生成于 {escape(datetime.now().astimezone().isoformat(timespec='seconds'))}")
    return render_page(title="重建 / 颈部重塑 loss 优化报告", subtitle=subtitle,
                       sections=sections, footer="由 loss_analysis 离线优化器生成 · 只读基线、候选写入新目录 · 单文件报告")


def run_optimizer(args: argparse.Namespace) -> int:
    from . import __version__

    baseline_result = Path(args.baseline_result).expanduser().resolve()
    if not baseline_result.is_dir():
        raise FileNotFoundError(f"baseline result directory not found: {baseline_result}")
    baseline_dense = (Path(args.baseline_dense).expanduser().resolve()
                      if args.baseline_dense else None)
    if baseline_dense is None:
        raise ValueError("--baseline-dense is required: the base dense report is a ranking tier")
    baseline_neck = (Path(args.baseline_neck).expanduser().resolve()
                     if args.baseline_neck else neck_adapter.find_neck_dir(baseline_result))
    baseline_trace = (Path(args.baseline_trace).expanduser().resolve()
                      if args.baseline_trace else None)
    landmarks = Path(args.landmarks).expanduser().resolve() if args.landmarks else None
    if landmarks is None:
        candidate_landmarks = baseline_result.parent / "neck_landmarks.json"
        if candidate_landmarks.is_file():
            landmarks = candidate_landmarks

    capture = Path(args.capture).expanduser().resolve() if args.capture else None
    dense_for_neck_only = baseline_dense

    out_dir = Path(args.output).expanduser().resolve() if args.output else None
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "runtime" / "loss_analysis" / f"{stamp}_neck_optimizer"
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    values = baseline_values(baseline_run=baseline_result, baseline_neck=baseline_neck,
                             baseline_dense=baseline_dense)
    explicit: list[Candidate] = []
    for item in args.candidate or []:
        if "=" not in item:
            raise ValueError(f"--candidate must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in PARAMS:
            raise ValueError(f"parameter {key!r} is not in the whitelist: {sorted(PARAMS)}")
        value = float(raw)
        param = PARAMS[key]
        if not (param.lo <= value <= param.hi):
            raise ValueError(f"{key}={value} outside the allowed range [{param.lo}, {param.hi}]")
        explicit.append(Candidate(key=f"{key.split('.', 1)[1]}_{value:g}",
                                  overrides={key: value},
                                  params={**values, key: value},
                                  note=param.reason))

    baseline_markers = landmarks
    baseline_assessment = neck_adapter.assess(
        dense_dir=baseline_dense,
        result_dir=baseline_result,
        neck_dir=baseline_neck,
        landmarks_path=baseline_markers,
    )
    if baseline_trace is not None:
        trace_loss = neck_adapter.read_loss_direct(baseline_trace)
        if trace_loss:
            states: dict[str, dict] = {}
            for state in ("closed", "open"):
                block = (trace_loss.get("states") or {}).get(state) or {}
                if block.get("residual_after"):
                    states[state] = block["residual_after"]
            vertex = [float(states[s]["vertex_rms_mm"]) for s in states
                      if isinstance(states[s].get("vertex_rms_mm"), (int, float))]
            quad = [float(states[s]["quad_rms_mm"]) for s in states
                    if isinstance(states[s].get("quad_rms_mm"), (int, float))]
            baseline_assessment["neck"]["target_residual"] = {
                "present": bool(states),
                "states": states,
                "mean_vertex_rms_mm": (sum(vertex) / len(vertex)) if vertex else None,
                "mean_quad_rms_mm": (sum(quad) / len(quad)) if quad else None,
            }
            baseline_assessment["neck"]["loss_present"] = True
            baseline_assessment["neck"]["trace_source"] = str(baseline_trace)
    baseline_probe = _delivery_probe(baseline_result)
    reference_peak = _reference_peak_from_build(baseline_neck or baseline_trace)
    if reference_peak is not None:
        baseline_assessment["neck"]["reference_peak_mm"] = reference_peak

    baseline_run = (Path(args.baseline_run).expanduser().resolve()
                    if getattr(args, "baseline_run", None) else None)
    if baseline_run is None:
        owner = baseline_dense.parents[1]
        baseline_run = owner if (owner / "logs").is_dir() else baseline_result
    gaps = gap_report.compute_gaps(
        run_dir=baseline_run if (baseline_run / "logs").is_dir() else None,
        dense_dir=baseline_dense,
        result_dir=baseline_result,
        neck_dir=baseline_neck,
        landmarks=landmarks,
        trace_dir=baseline_trace,
    )

    notes: list[str] = []
    if explicit:
        candidates = explicit
    elif getattr(args, "targeted", False):
        candidates = targeted_candidates(values, gaps)
        if not candidates:
            notes.append("--targeted: the dominant gap is not controlled by a whitelisted "
                         "parameter; no candidate was generated")
        else:
            notes.append("--targeted: candidates come from the dominant-gap mapping; at most two "
                         "single-parameter moves, no full whitelist sweep")
    else:
        candidates = plan_candidates(values, limit=args.limit)
    if baseline_trace is not None:
        notes.append("基线求解器轨迹来自同参数重建的 loss_direct.json；基线交付本身未重跑。")
    if baseline_assessment["neck"].get("target_residual", {}).get("mean_vertex_rms_mm") is None:
        notes.append("基线没有目标残差数值（旧构建未持久化），候选残差只与候选之间比较。")

    execute = bool(args.execute)
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONIOENCODING": "utf-8"}
    figures: list[str] = []
    max_candidates = min(len(candidates), max(0, int(args.max_candidates)))
    if execute and max_candidates == 0:
        notes.append("--max-candidates 为 0：没有执行任何候选。")

    for index, candidate in enumerate(candidates, 1):
        should_run = execute and index <= max_candidates
        if not should_run:
            continue
        cand_dir = out_dir / "candidates" / f"c{index:02d}_{_slug(candidate.key)}"
        try:
            code, command = _run_candidate(
                candidate, capture=capture, dense=dense_for_neck_only,
                out_dir=cand_dir, landmarks=landmarks,
                freeze_landmarks=bool(args.freeze_landmarks),
                neck_only_dense=bool(getattr(args, "neck_only_dense", False)),
                timeout_s=int(args.timeout_s), env=env,
            )
        except Exception as exc:  # noqa: BLE001 - a failed candidate is reported, not fatal
            candidate.error = f"{type(exc).__name__}: {exc}"
            notes.append(f"候选 {candidate.key} 执行失败：{candidate.error}")
            continue
        candidate.run_dir = cand_dir
        if code != 0:
            candidate.error = f"rebuild exited with code {code}; see {cand_dir / 'run.log'}"
            notes.append(f"候选 {candidate.key} 未完成：{candidate.error}")
            continue
        result_dir = cand_dir / "result"
        dense_dir = cand_dir / "reconstruction" / "dense"
        cand_landmarks = cand_dir / "neck_landmarks.json"
        if not cand_landmarks.is_file():
            cand_landmarks = landmarks
        assessment, hard, probe = evaluate(
            result_dir=result_dir, dense_dir=dense_dir, landmarks=cand_landmarks,
            baseline_assessment=baseline_assessment,
        )
        reference_peak = (baseline_assessment.get("neck") or {}).get("reference_peak_mm")
        if reference_peak is not None:
            assessment["neck"]["reference_peak_mm"] = reference_peak
        assessment["downstream_probe"] = probe
        candidate.assessment = assessment
        candidate.hard = hard
        candidate.rank = _rank_key(assessment)
        if not args.no_figures:
            loss = neck_adapter.read_loss_direct(neck_adapter.find_neck_dir(result_dir))
            if loss:
                png = _trace_figure(loss, out_dir / "figures" / f"c{index:02d}_{_slug(candidate.key)}.png")
                if png:
                    figures.append(str(png))

    ranked = [c for c in candidates if c.rank is not None and all(r["status"] == "pass" for r in c.hard)]
    ranked.sort(key=lambda c: c.rank)
    winner = ranked[0] if ranked else None

    write_json(out_dir / "plan.json", {
        "schema": "loss_analysis.optimizer.plan.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lossana_version": __version__,
        "baseline": {
            "result": str(baseline_result),
            "dense": str(baseline_dense),
            "neck": str(baseline_neck) if baseline_neck else None,
            "trace": str(baseline_trace) if baseline_trace else None,
            "values": values,
            "assessment": baseline_assessment,
            "probe": baseline_probe,
        },
        "candidates": [
            {"key": c.key, "overrides": c.overrides, "note": c.note,
             "params": {k: c.params[k] for k in c.overrides}}
            for c in candidates
        ],
        "gap": gaps,
        "execute": execute,
    })
    write_json(out_dir / "candidates.json", {
        "schema": "loss_analysis.optimizer.candidates.v1",
        "candidates": [
            {
                "key": c.key,
                "overrides": c.overrides,
                "run_dir": str(c.run_dir) if c.run_dir else None,
                "error": c.error,
                "hard": c.hard,
                "rank": _jsonable_rank(c.rank),
                "assessment": c.assessment,
            }
            for c in candidates
        ],
        "winner": winner.key if winner else None,
    })
    csv_path = out_dir / "candidates.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate", "status", "hard_failures", "base_fixed_ray_mse_mm2",
                         "connection_max_deg", "target_vertex_rms_mm", "target_quad_rms_mm"])
        for c in candidates:
            if not c.assessment:
                writer.writerow([c.key, "error" if c.error else "planned", "", "", "", "", ""])
                continue
            base = (c.assessment.get("base") or {})
            neck = (c.assessment.get("neck") or {})
            fails = [r["name"] for r in c.hard if r["status"] != "pass"]
            writer.writerow([
                c.key, "rejected" if fails else "ranked", ";".join(fails),
                base.get("fixed_ray_after_mse_mm2"),
                (neck.get("gates") or {}).get("connection_max_deg"),
                (neck.get("target_residual") or {}).get("mean_vertex_rms_mm"),
                (neck.get("target_residual") or {}).get("mean_quad_rms_mm"),
            ])

    html_path = write_report(out_dir / "report.html", _html(
        baseline_assessment=baseline_assessment,
        baseline_label=str(baseline_result),
        candidates=candidates,
        plan_only=not execute,
        figures=figures,
        execute=execute,
        winner=winner,
        notes=notes,
        gap=gaps,
    ))
    write_json(out_dir / "summary.json", {
        "output_dir": str(out_dir),
        "report": str(html_path),
        "plan": str(out_dir / "plan.json"),
        "candidates": str(out_dir / "candidates.json"),
        "csv": str(csv_path),
        "figures": figures,
        "executed": len([c for c in candidates if c.assessment]),
        "ranked": len(ranked),
        "winner": winner.key if winner else None,
        "baseline_assessment": baseline_assessment,
        "dominant_gap": (gaps.get("dominant") or {}).get("key"),
        "targeted": [item["key"] for item in gaps.get("targeted") or []],
        "notes": notes,
        "note": ("analysis only; the baseline was read, candidates were written to new "
                 "directories, defaults and acceptance rules were not changed"),
    })
    print(json.dumps({
        "output_dir": str(out_dir),
        "report": str(html_path),
        "winner": winner.key if winner else None,
        "executed": len([c for c in candidates if c.assessment]),
        "candidates": [c.key for c in candidates],
        "dominant_gap": (gaps.get("dominant") or {}).get("key"),
        "targeted": [item["key"] for item in gaps.get("targeted") or []],
        "notes": notes,
    }, ensure_ascii=False, indent=2))
    return 0
