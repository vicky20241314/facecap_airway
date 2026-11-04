"""Figures and the Chinese report.

Plotting is deliberately conservative: a line is drawn only between real
``(iteration, value)`` points of one stage and one provenance tier. Stages with a
gap, or with fewer than two real points, get a placeholder panel that says why —
the tool never interpolates a curve from a final mesh.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .objectives import DIAGNOSTIC, LINEARISED, NONLINEAR_ENERGY, SOLVER_OBJECTIVE
from .recorder import available_series, stages_with_gaps

TIER_LABEL_CN = {
    SOLVER_OBJECTIVE: "求解器目标（每步）",
    LINEARISED: "冻结对应线性化目标",
    NONLINEAR_ENERGY: "刷新对应后的实际能量（固定射线 MSE）",
    DIAGNOSTIC: "独立几何诊断（非算法 loss）",
}


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Prefer a CJK font if one is installed; fall back silently to the default.
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if any(name.lower() == f.name.lower() for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
    return plt


def series_by_decision(payload: dict[str, Any], stage: str, tier: str) -> dict[str, list[tuple[int, float]]]:
    """Split one tier into before/after (or solver) sub-series by decision label.

    One iteration can legitimately carry two values (objective before and after the
    line search). Joining them into a single polyline would draw a fake oscillation,
    so they are kept as separate named series.
    """
    out: dict[str, list[tuple[int, float]]] = {}
    for row in payload.get("iterations", []):
        if row.get("stage") != stage or row.get("tier") != tier:
            continue
        if row.get("iteration") is None or row.get("value") is None:
            continue
        label = row.get("decision") or "value"
        out.setdefault(label, []).append((int(row["iteration"]), float(row["value"])))
    return {label: sorted(rows) for label, rows in out.items()}


def _solver_phase_runs(payload: dict[str, Any], stage: str) -> list[tuple[str, list[tuple[int, float]]]]:
    """Split a per-step solver objective into runs of consecutive identical phase.

    The objective's active term set changes at every phase cut, so joining across a
    cut draws a saw-tooth that reads as "the optimiser got worse". It did not: only
    the terms being summed changed. A run must also break when a phase is revisited
    (motion_v2 alternates rigid / expression_jaw / rigid), which a plain group-by
    would wrongly stitch back together.
    """
    rows = sorted(
        (int(row["iteration"]), float(row["value"]), row.get("phase") or "phase")
        for row in payload.get("iterations", [])
        if row.get("stage") == stage and row.get("tier") == SOLVER_OBJECTIVE
        and row.get("iteration") is not None and row.get("value") is not None
    )
    runs: list[tuple[str, list[tuple[int, float]]]] = []
    for iteration, value, phase in rows:
        if runs and runs[-1][0] == phase:
            runs[-1][1].append((iteration, value))
        else:
            runs.append((phase, [(iteration, value)]))
    return runs


def plot_stage_series(payload: dict[str, Any], out_path: Path, *, stage: str,
                      tiers: tuple[str, ...] = (SOLVER_OBJECTIVE, LINEARISED, NONLINEAR_ENERGY)) -> Path | None:
    """One figure per stage. Returns ``None`` when nothing real can be plotted."""
    series: list[tuple[str, list[tuple[int, float]]]] = []
    boundaries: list[tuple[int, str]] = []
    for tier in tiers:
        if tier == SOLVER_OBJECTIVE:
            for phase, rows in _solver_phase_runs(payload, stage):
                series.append((f"{phase}", rows))
                boundaries.append((rows[0][0], phase))
            continue
        for decision, rows in sorted(series_by_decision(payload, stage, tier).items()):
            if len(rows) >= 2:
                series.append((f"{TIER_LABEL_CN[tier]} [{decision}]", rows))
            elif rows:
                series.append((f"{TIER_LABEL_CN[tier]} [{decision}] (单点)", rows))
    if not series:
        return None
    plt = _matplotlib()
    # A stage that ran a single outer step has no curve to draw: points on a linear
    # iteration axis collapse onto one x and the axis even pads to negative integers.
    # Those stages get a grouped bar chart instead, so the tiers stay readable.
    distinct_iterations = {x for _, rows in series for x, _ in rows}
    if len(distinct_iterations) <= 1:
        return _plot_single_iteration_bars(plt, series, out_path, stage, payload)
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=140)
    for label, rows in series:
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        if len(rows) >= 2:
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=label)
        else:
            ax.plot(xs, ys, marker="o", markersize=5, linestyle="none", label=label)
    # Mark each phase cut: the break between runs is a term-set change, not a regression.
    if boundaries:
        top = max(y for _, rows in series for _, y in rows)
        for index, (position, phase) in enumerate(sorted(boundaries)):
            if index == 0:
                continue
            ax.axvline(position, color="0.55", linewidth=0.9, linestyle="--")
            ax.annotate(phase, (position, top), xytext=(2, -2), textcoords="offset points",
                        fontsize=6, color="0.35", rotation=90, va="top")
    gap_note = "；虚线=相位切换（项集合改变，非退化）" if boundaries else ""
    ax.set_title(f"{stage} — 分级目标曲线{gap_note}".strip(), fontsize=10)
    ax.set_xlabel("迭代编号（真实记录）")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("目标值（各档不可直接相加）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_single_iteration_bars(plt, series, out_path: Path, stage: str,
                                payload: dict[str, Any]) -> Path:
    """Grouped bars for a stage that recorded only one outer iteration.

    Values of different tiers never share a unit, so the bars are an inventory of
    what was recorded at that one step, not a comparison of magnitudes.
    """
    labels = [label for label, _ in series]
    values = [rows[0][1] for _, rows in series]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=140)
    positions = range(len(values))
    bars = ax.bar(positions, values, width=0.6)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.6g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([label.replace(" ", "\n", 1) for label in labels],
                       fontsize=7, rotation=18, ha="right")
    # The gap note is per STAGE: saying "this run has a gap somewhere" on a figure
    # whose own stage is complete makes a reader look for a hole that is not there.
    gap_note = "（本阶段另有缺口，见缺口表）" if stage in stages_with_gaps(payload) else ""
    ax.set_title(f"{stage} — 仅 1 个 outer，无曲线可画 {gap_note}".strip(), fontsize=10)
    ax.set_ylabel("记录值（各档不可直接相加）")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_gap_panel(payload: dict[str, Any], out_path: Path) -> Path:
    """Explicit panel: which stages have no real iteration state, and why.

    Reasons are wrapped: a single unwrapped line runs off the canvas and the most
    important part of the sentence (the cause) is exactly the part that gets cut.
    """
    import textwrap

    plt = _matplotlib()
    gaps = payload.get("gaps", [])
    wrapped: list[tuple[str, list[str]]] = [
        (gap["stage"], textwrap.wrap(gap["reason"], width=104) or [""]) for gap in gaps
    ]
    line_count = sum(len(lines) for _, lines in wrapped) + len(wrapped)
    fig, ax = plt.subplots(figsize=(11, 1.4 + 0.19 * max(line_count, 1)), dpi=140)
    ax.axis("off")
    ax.set_title("缺口阶段（无真实迭代状态，不绘制曲线）", fontsize=10, loc="left")
    if not gaps:
        ax.text(0.01, 0.5, "无缺口：每个阶段都有真实记录或已标记 not_used。", fontsize=9)
    else:
        step = 1.0 / max(line_count + 1, 2)
        cursor = 0.98
        for stage, lines in wrapped:
            ax.text(0.01, cursor, f"• {stage}", fontsize=8, transform=ax.transAxes,
                    va="top", fontweight="bold")
            cursor -= step
            for line in lines:
                ax.text(0.03, cursor, line, fontsize=7.5, transform=ax.transAxes, va="top")
                cursor -= step
            cursor -= step * 0.4
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_metrics(metrics_by_frame: dict[str, dict[str, Any]], out_path: Path) -> Path | None:
    """Diagnostic MAE/RMSE/P95 per frame; clearly labelled as diagnostics."""
    usable = {k: v for k, v in metrics_by_frame.items() if v.get("metrics", {}).get("status") == "ok"}
    if not usable:
        return None
    plt = _matplotlib()
    names = list(usable)
    mae = [usable[n]["metrics"]["mae_mm"] for n in names]
    rmse = [usable[n]["metrics"]["rmse_mm"] for n in names]
    p95 = [usable[n]["metrics"]["p95_mm"] for n in names]
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(names)), 3.4), dpi=140)
    xs = range(len(names))
    ax.plot(xs, mae, marker="o", label="MAE (mm)")
    ax.plot(xs, rmse, marker="s", label="RMSE (mm)")
    ax.plot(xs, p95, marker="^", label="P95 (mm)")
    for x, name in zip(xs, names):
        block = usable[name]
        n_valid = block.get("metrics", {}).get("n_valid")
        total = block.get("coverage", {}).get("n_required")
        if n_valid is not None and total:
            ax.annotate(f"n={n_valid}/{total}", (x, block["metrics"]["mae_mm"]),
                        textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("毫米")
    ax.set_title("独立诊断：固定观测射线深度误差（非算法 loss；n=命中射线条数/总射线条数）", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def build_report(
    payload: dict[str, Any],
    *,
    stage_catalogue: list[dict[str, Any]],
    input_summary: dict[str, Any],
    metrics: dict[str, dict[str, Any]] | None = None,
    extra_notes: list[str] | None = None,
    mode_variants: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# loss 分析报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(f"- 运行标签：{payload.get('run_label')}")
    lines.append(f"- 算法：{payload.get('algorithm')}")
    lines.append(f"- 软件版本：{payload.get('software_version')}")
    if input_summary.get("chain_shape_fusion"):
        lines.append(f"- 链路 shape_fusion（由产物判定）：{input_summary['chain_shape_fusion']}")
    if input_summary.get("versions"):
        stamp = input_summary["versions"]
        lines.append(f"- 版本戳：{stamp.get('stated')}；一致={stamp.get('consistent')}")
        if not stamp.get("consistent") or stamp.get("v0_4_markers"):
            lines.append(f"  - {stamp.get('note')}")
    lines.append(f"- 输入：{input_summary.get('run_dir')}")
    config = input_summary.get("configuration") or {}
    if config:
        lines.append("- 运行配置（由 `run_report.configuration` 自报，非本工具推断）：")
        for key, value in config.items():
            lines.append(f"  - `{key}` = {value!r}")
    lines.append("")
    lines.append("## 0. 必读：本报告中的数字是什么")
    lines.append("")
    for tier, definition in payload["tier_definitions"].items():
        lines.append(f"- **{TIER_LABEL_CN[tier]}**（`{tier}`）：{definition}")
    lines.append("")
    lines.append("各阶段没有共同总目标，本工具**不产出跨阶段 total loss**；报告中的 total 仅为同一阶段可得分项之和，"
                 "缺失项记为 null 而不是 0。")
    lines.append("")
    lines.append("## 1. 各阶段优化目标（源码依据）")
    lines.append("")
    if mode_variants:
        lines.append("### 1a. 模式变体：同一个阶段键，不同的目标")
        lines.append("")
        lines.append("v0.4.0 让同一个阶段键在开关不同时优化**不同**的目标，而且其中一个阶段"
                     "（`identity_v2`）的开关并不会被 pipeline 转发；0.5.0 又增加了"
                     "「直接复用上一个阶段的场」与「默认不执行密集残差」两种变体。因此"
                     "“这一阶段这次到底优化了什么”只从运行产物判定，不从命令行判定。"
                     "变体中 `stage_key` 与运行自报名一致时，仍以产物为准。")
        lines.append("")
        lines.append("| 变体（运行自报的名字） | 所属阶段 | 触发方式 | 参与数据的帧 | 分项差异 | 判定依据 |")
        lines.append("|---|---|---|---|---|---|")
        for variant in mode_variants:
            lines.append(
                f"| `{variant['variant_key']}` | `{variant['stage_key']}` | {variant['activated_by']} | "
                f"{variant['data_frames']} | {variant['term_overrides']} | {variant['evidence']} |"
            )
        lines.append("")
        lines.append("### 1b. 各阶段分项")
        lines.append("")
    for stage in stage_catalogue:
        lines.append(f"### {stage['key']} — {stage['label_cn']}")
        lines.append("")
        lines.append(f"- 模块：`{stage['module']}`")
        lines.append(f"- 优化参数：{', '.join(stage['parameters'])}")
        lines.append(f"- 求解器：{stage['solver']}")
        lines.append(f"- 目标口径：{stage['objective_scope']}")
        lines.append("")
        lines.append("| 分项 | 类别 | 归一化分母 | 权重 | 来源 |")
        lines.append("|---|---|---|---|---|")
        for term in stage["terms"]:
            lines.append(
                f"| {term['label_cn']} (`{term['key']}`) | {term['kind']} | "
                f"{term['normaliser']} | {term['weight']} | `{term['source']}` |"
            )
        for note in stage.get("notes", []):
            lines.append(f"- 备注：{note}")
        lines.append("")
    lines.append("## 2. 记录到的真实迭代（按档位分列）")
    lines.append("")
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("iterations", []):
        by_stage.setdefault(row["stage"], []).append(row)
    if not by_stage:
        lines.append("本次没有记录到任何真实迭代状态。")
    for stage, rows in by_stage.items():
        tiers = sorted({r["tier"] for r in rows})
        lines.append(f"### {stage}")
        lines.append("")
        lines.append(f"- 记录条数：{len(rows)}；档位：{', '.join(f'`{t}`' for t in tiers)}")
        iters = [r["iteration"] for r in rows if r["iteration"] is not None]
        if iters:
            lines.append(f"- 迭代范围：{min(iters)} .. {max(iters)}")
        # Each (tier, decision) is its own series. Summarising a stage across series
        # would let the `initial` value of the LAST outer pose as the stage outcome
        # (e.g. 3.487 instead of the actually accepted 3.133), so never merge them.
        per_series: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for row in rows:
            if row["iteration"] is None or row["value"] is None:
                continue
            key = (row["tier"], row.get("decision") or "value")
            per_series.setdefault(key, []).append((int(row["iteration"]), float(row["value"])))
        for (tier, decision), points in sorted(per_series.items()):
            points.sort()
            if len(points) == 1:
                lines.append(f"- `{tier}` [{decision}]：仅 1 点 {_fmt(points[0][1])}（iter {points[0][0]}），无法判断趋势")
            else:
                lines.append(
                    f"- `{tier}` [{decision}]：{len(points)} 点，首 {_fmt(points[0][1])} → 末 "
                    f"{_fmt(points[-1][1])}（同一序列内比较，不与其它序列混算）"
                )
        # Stage-level notes carry the conditions the numbers were produced under
        # (fast path, reused field, phase term-set switching). They were collected
        # from the artefacts, so dropping them from the report hides the reason a
        # curve looks the way it does.
        meta_notes = (payload.get("stages", {}).get(stage) or {}).get("notes") or []
        if meta_notes:
            lines.append("- 阶段条件：")
            for note in meta_notes:
                lines.append(f"  - {note}")
        lines.append("")
    # Stages that produced no iteration row still carry artefact-derived conditions
    # (a reused field, a stage outside the chain). Those belong next to the gaps.
    declared = {
        name: meta for name, meta in (payload.get("stages") or {}).items()
        if name not in by_stage and (meta.get("notes") or [])
    }
    if declared:
        lines.append("### 无迭代记录、但产物已说明条件的阶段")
        lines.append("")
        for name, meta in sorted(declared.items()):
            lines.append(f"- **{name}**")
            for note in meta.get("notes") or []:
                lines.append(f"  - {note}")
        lines.append("")
    lines.append("## 3. 缺口阶段（无真实状态）")
    lines.append("")
    gaps = payload.get("gaps", [])
    if not gaps:
        lines.append("无。")
    else:
        lines.append("| 阶段 | 期望档位 | 原因 | 线索 |")
        lines.append("|---|---|---|---|")
        for gap in gaps:
            lines.append(f"| {gap['stage']} | `{gap['expected_tier']}` | {gap['reason']} | {gap.get('source_hint', '')} |")
        lines.append("")
        lines.append("> 这些阶段不会出现在曲线中；工具不会对最终网格做反向拟合来补出曲线。")
    lines.append("")
    if metrics:
        lines.append("## 4. 独立几何诊断（固定观测，非算法 loss）")
        lines.append("")
        lines.append("计算方式：取该帧 prepared 数据里逐点记录的观测位置（当前流程为 68 个 landmark 点），"
                     "对每个点从相机原点发一条射线打到交付网格上，取**沿光轴的深度差绝对值**（mm）；"
                     "深度差超过 30 mm 或没有打到网格的射线记为**未覆盖**，不计为 0 误差。"
                     "MAE/RMSE/P95 只在命中的子集上统计。**这不是稠密表面误差**：样本量就是每帧的观测点数，"
                     "因此只能作为指示性检查，且交付网格位于参考帧（每状态的第一帧，正面）的相机坐标系下，"
                     "其他视角的帧需要该帧的估计位姿才能比较，未做位姿变换的跨视角数字没有意义。")
        lines.append("")
        lines.append("| 帧 | 观测点 | 命中 | MAE (mm) | RMSE (mm) | P95 (mm) | 中位数 (mm) | 覆盖 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, item in metrics.items():
            m = item.get("metrics", {})
            cov = item.get("coverage", {})
            lines.append(
                f"| {name} | {cov.get('n_required')} | {m.get('n_valid')} | {_fmt(m.get('mae_mm'), 4)} | "
                f"{_fmt(m.get('rmse_mm'), 4)} | {_fmt(m.get('p95_mm'), 4)} | {_fmt(m.get('median_mm'), 4)} | "
                f"{_fmt(cov.get('coverage'), 4)} |"
            )
        lines.append("")
    for note in extra_notes or []:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 5. 结论边界")
    lines.append("")
    lines.append("- 本工具只做观测与统计，不改变优化目标、参数更新、采样、随机种子、迭代次数、接受条件或几何保护。")
    lines.append("- 求解器目标、线性化目标、刷新后的实际能量、独立诊断四类数值不可互相比较或相加。")
    lines.append("- 构造数据上的测试只证明计算与接口正确，不证明真实重建效果。")
    return "\n".join(lines) + "\n"


def write_report(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path
