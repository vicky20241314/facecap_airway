"""Single-file HTML report for one analysis run.

The page opens on a machine without network access and without the output
directory: styles are inline and every figure is embedded as a base64 PNG. The
report is a *presentation* of the same payload the Markdown report uses, so it
inherits the same rules: four provenance tiers stay separate, a missing value is
``null`` (never 0), and no cross-stage total is printed.

No JavaScript is used; ``<details>`` gives collapsibility with plain HTML.
"""
from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Iterable

from .objectives import DIAGNOSTIC, LINEARISED, NONLINEAR_ENERGY, SOLVER_OBJECTIVE
from .plots import TIER_LABEL_CN

TIER_CLASS = {
    SOLVER_OBJECTIVE: "tier-solver",
    LINEARISED: "tier-linear",
    NONLINEAR_ENERGY: "tier-energy",
    DIAGNOSTIC: "tier-diag",
}

MAX_EMBED_BYTES = 8 * 1024 * 1024

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f4f6f9; color: #1b2430;
       font: 14px/1.6 "Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }
header { background: #15222f; color: #eef4fa; padding: 26px 36px; }
header h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; }
header .meta { color: #b9c9d8; font-size: 12.5px; }
header .meta code { background: #24384b; color: #dce8f3; }
main { max-width: 1200px; margin: 0 auto; padding: 20px 36px 70px; }
nav.toc { background: #fff; border: 1px solid #e1e6ed; border-radius: 10px;
          padding: 12px 18px; margin: 18px 0; font-size: 13px; }
nav.toc a { color: #1f5fa8; text-decoration: none; margin-right: 14px; white-space: nowrap; }
nav.toc a:hover { text-decoration: underline; }
section.card { background: #fff; border: 1px solid #e1e6ed; border-radius: 10px;
               padding: 18px 24px 22px; margin: 18px 0; }
section.card > h2 { margin: 0 0 12px; font-size: 17px; color: #14212e;
                    border-bottom: 1px solid #e8edf3; padding-bottom: 8px; }
h3 { font-size: 14.5px; margin: 20px 0 8px; color: #22384d; }
p, li { font-size: 13px; }
p.note { color: #5a6b7d; font-size: 12.5px; }
code { background: #eef2f7; border-radius: 4px; padding: 0 4px;
       font: 12px/1.5 Consolas, "Courier New", monospace; }
.null { color: #b23b3b; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 12.5px; }
th, td { border: 1px solid #e1e6ed; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 600; white-space: nowrap; }
tbody tr:nth-child(even) td { background: #fafbfc; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.badge { display: inline-block; border-radius: 10px; padding: 0 8px; font-size: 11.5px;
         font-weight: 600; white-space: nowrap; }
.tier-solver { background: #e3effb; color: #1f5fa8; }
.tier-linear { background: #f1e8fb; color: #6435a8; }
.tier-energy { background: #e6f5ea; color: #1d6f37; }
.tier-diag { background: #fdf0e0; color: #8a5a12; }
.tier-none { background: #f1f3f5; color: #55606c; }
ul.tight { margin: 4px 0 10px 20px; padding: 0; }
ul.tight li { margin: 2px 0; }
details { border: 1px dashed #d4dbe4; border-radius: 8px; padding: 8px 12px; margin: 10px 0; }
details > summary { cursor: pointer; font-weight: 600; color: #22384d; }
figure { margin: 14px 0; }
figure img { max-width: 100%; border: 1px solid #dde3ea; border-radius: 8px; background: #fff; }
figcaption { color: #5a6b7d; font-size: 12px; margin-top: 4px; }
.callout { border-left: 3px solid #c98b1e; background: #fdf9ef; padding: 8px 12px;
           margin: 10px 0; font-size: 12.5px; }
footer { max-width: 1200px; margin: 0 auto; padding: 0 36px 40px;
         color: #6a7683; font-size: 12px; }
@media print {
  body { background: #fff; }
  header { background: #fff; color: #14212e; border-bottom: 2px solid #14212e; }
  header .meta { color: #44515f; }
  section.card { border-color: #cfd6de; break-inside: avoid-page; }
  nav.toc { display: none; }
}
"""


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return '<span class="null">null</span>'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return escape(f"{value:.{digits}g}")
    return escape(str(value))


def _text(value: Any) -> str:
    if value is None:
        return '<span class="null">null</span>'
    return escape(str(value))


def _join_lines(lines: Iterable[str]) -> str:
    items = [escape(line) for line in lines if line]
    return "<br>".join(items) if items else "—"


def badge(tier: str | None) -> str:
    css = TIER_CLASS.get(tier or "", "tier-none")
    label = TIER_LABEL_CN.get(tier or "", tier or "unknown")
    return f'<span class="badge {css}">{escape(label)}</span>'


def table(headers: list[str], rows: list[list[str]], *, numeric: set[int] | None = None) -> str:
    """Render a table. Cell strings are already-safe HTML (use _fmt/_text)."""
    numeric = numeric or set()
    head = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            css = ' class="num"' if index in numeric else ""
            cells.append(f"<td{css}>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def kv_table(pairs: list[tuple[str, Any]]) -> str:
    rows = [[f"<b>{escape(key)}</b>", _fmt(value)] for key, value in pairs]
    return table(["项", "值"], rows)


def render_page(*, title: str, subtitle: str, sections: list[dict[str, str]],
                footer: str = "") -> str:
    """Generic page shell shared by the analysis and the optimizer reports."""
    nav = "".join(
        f'<a href="#{escape(section["id"])}">{escape(section["heading"])}</a>'
        for section in sections
    )
    cards = "".join(
        f'<section class="card" id="{escape(section["id"])}">'
        f'<h2>{escape(section["heading"])}</h2>{section["html"]}</section>'
        for section in sections
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        f"<header><h1>{escape(title)}</h1><div class=\"meta\">{subtitle}</div></header>\n"
        f'<main>\n<nav class="toc">{nav}</nav>\n{cards}\n</main>\n'
        f"<footer>{footer}</footer>\n"
        "</body>\n</html>\n"
    )


def _figure_gallery(figures: Iterable[str | Path]) -> str:
    blocks: list[str] = []
    for raw in figures:
        path = Path(raw)
        caption = path.stem
        if not path.is_file():
            blocks.append(f'<p class="note">图缺失：{_text(str(path))}</p>')
            continue
        size = path.stat().st_size
        if size > MAX_EMBED_BYTES:
            blocks.append(
                f'<p class="note">图过大未内嵌（{size} 字节）：{_text(str(path))}</p>'
            )
            continue
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append(
            "<figure>"
            f'<img alt="{escape(caption)}" loading="lazy" '
            f'src="data:image/png;base64,{data}">'
            f"<figcaption>{escape(caption)}</figcaption>"
            "</figure>"
        )
    return "".join(blocks) if blocks else '<p class="note">本次没有生成图。</p>'


def _series_summary(rows: list[dict[str, Any]]) -> list[list[str]]:
    """One row per (tier, phase-or-decision) series. Never merges series."""
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        if row.get("iteration") is None or row.get("value") is None:
            continue
        tier = row.get("tier") or "unknown"
        label = (row.get("phase") if tier == SOLVER_OBJECTIVE else None) \
            or row.get("decision") or "value"
        series.setdefault((tier, label), []).append(
            (int(row["iteration"]), float(row["value"]))
        )
    out: list[list[str]] = []
    for (tier, label), points in sorted(series.items()):
        points.sort()
        values = [value for _, value in points]
        trend = (
            _fmt(points[0][1]) + " → " + _fmt(points[-1][1])
            if len(points) > 1 else _fmt(points[0][1]) + "（仅 1 点）"
        )
        out.append([
            badge(tier),
            escape(label),
            _fmt(len(points)),
            f"iter {points[0][0]} .. {points[-1][0]}",
            trend,
            _fmt(min(values)),
            _fmt(max(values)),
        ])
    return out


def _record_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("iteration") if row.get("iteration") is not None else -1,
            row.get("tier") or "",
            row.get("decision") or "",
        ),
    )
    out: list[list[str]] = []
    for row in ordered:
        terms = row.get("terms") or {}
        term_html = "<br>".join(
            f"<code>{escape(str(key))}={_fmt(value)}</code>"
            for key, value in sorted(terms.items())
        ) or "—"
        out.append([
            _fmt(row.get("iteration")),
            badge(row.get("tier")),
            _text(row.get("phase")),
            _text(row.get("decision")),
            _fmt(row.get("value")),
            _text(row.get("step_scale")),
            _text(row.get("coverage")),
            term_html,
            _join_lines(row.get("notes") or []),
            _text(row.get("source")),
        ])
    return out


def build_analysis_html(
    payload: dict[str, Any],
    *,
    stage_catalogue: list[dict[str, Any]],
    input_summary: dict[str, Any],
    metrics: dict[str, dict[str, Any]] | None = None,
    extra_notes: list[str] | None = None,
    mode_variants: list[dict[str, Any]] | None = None,
    figures: Iterable[str | Path] = (),
    generated_at: str | None = None,
) -> str:
    """Compose the analysis page. Same payload and rules as ``plots.build_report``."""
    stamp = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    sections: list[dict[str, str]] = []

    facts: list[tuple[str, Any]] = [
        ("运行标签", payload.get("run_label")),
        ("算法", payload.get("algorithm")),
        ("软件版本", payload.get("software_version")),
        ("输入目录", input_summary.get("run_dir")),
    ]
    versions = input_summary.get("versions") or {}
    if versions:
        facts.append(("版本戳", versions.get("stated")))
        facts.append(("版本一致", versions.get("consistent")))
        if versions.get("note"):
            facts.append(("版本说明", versions.get("note")))
    if input_summary.get("chain_shape_fusion"):
        facts.append(("链路 shape_fusion（由产物判定）", input_summary["chain_shape_fusion"]))
    facts.append(("生成时间", stamp))
    config = input_summary.get("configuration") or {}
    body = kv_table(facts)
    if config:
        body += "<h3>运行配置（run_report 自报，非本工具推断）</h3>" + table(
            ["键", "值"], [[f"<code>{escape(str(k))}</code>", _fmt(v)] for k, v in config.items()]
        )
    sections.append({"id": "overview", "heading": "0. 本次运行", "html": body})

    tier_rows = [
        [badge(tier), escape(str(definition))]
        for tier, definition in payload.get("tier_definitions", {}).items()
    ]
    sections.append({
        "id": "tiers",
        "heading": "1. 数字档位（不可互相比较，不相加）",
        "html": table(["档位", "含义"], tier_rows)
        + '<p class="note">各阶段没有共同总目标，本报告不产出跨阶段 total loss；'
          '同一阶段内也只对可得分项求和，缺失项记为 null 而不是 0。</p>'
          '<div class="callout">读图规则：solve_field 的固定射线 MSE 是接受判据，'
          '不属于求解器二次型；诊断 MAE/RMSE/P95 是本工具自己的度量，不是算法 loss。</div>',
    })

    catalogue_html = ""
    if mode_variants:
        catalogue_html += "<h3>模式变体：同一阶段键，不同目标</h3>" + table(
            ["变体", "阶段", "触发", "参与数据", "分项差异"],
            [[f"<code>{escape(str(v['variant_key']))}</code>", f"<code>{escape(str(v['stage_key']))}</code>",
              _text(v.get("activated_by")), _text(v.get("data_frames")), _text(v.get("term_overrides"))]
             for v in mode_variants],
        )
    for stage in stage_catalogue:
        rows = [
            [f"{escape(str(term['label_cn']))} <code>{escape(str(term['key']))}</code>",
             _text(term.get("kind")), _text(term.get("normaliser")),
             _text(term.get("weight")), f"<code>{escape(str(term.get('source')))}</code>"]
            for term in stage["terms"]
        ]
        inner = (
            f"<p class='note'>模块 <code>{escape(str(stage['module']))}</code>；"
            f"参数 {escape(', '.join(stage['parameters']))}；求解器 {escape(str(stage['solver']))}</p>"
            + table(["分项", "类别", "归一化", "权重", "源码"], rows)
        )
        for note in stage.get("notes") or []:
            inner += f"<p class='note'>备注：{escape(str(note))}</p>"
        catalogue_html += (
            f"<details><summary>{escape(str(stage['key']))} — {escape(str(stage['label_cn']))}</summary>"
            f"{inner}</details>"
        )
    sections.append({
        "id": "catalogue",
        "heading": "2. 各阶段优化目标（源码依据）",
        "html": catalogue_html,
    })

    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("iterations", []):
        by_stage.setdefault(row["stage"], []).append(row)
    if by_stage:
        records_html = ""
        for stage, rows in by_stage.items():
            summary_rows = _series_summary(rows)
            meta_notes = (payload.get("stages", {}).get(stage) or {}).get("notes") or []
            records_html += f"<h3>{escape(stage)}（{len(rows)} 条记录）</h3>"
            records_html += (
                table(["档位", "序列", "点数", "迭代范围", "首 → 末（同序列内）", "最小", "最大"],
                      summary_rows, numeric={2, 5, 6})
                if summary_rows else '<p class="note">该阶段没有可画成序列的数值记录。</p>'
            )
            if meta_notes:
                records_html += "<ul class='tight'>" + "".join(
                    f"<li>{escape(str(note))}</li>" for note in meta_notes
                ) + "</ul>"
            records_html += (
                "<details><summary>逐条记录（含 null 与决策）</summary>"
                + table(["iter", "档位", "相位", "决策", "值", "步长", "覆盖", "分项", "备注", "来源"],
                        _record_rows(rows), numeric={0, 4, 5, 6})
                + "</details>"
            )
    else:
        records_html = "<p>本次没有记录到任何真实迭代状态。</p>"
    declared = {
        name: meta for name, meta in (payload.get("stages") or {}).items()
        if name not in by_stage and (meta.get("notes") or [])
    }
    if declared:
        records_html += "<h3>无迭代记录、但产物已说明条件的阶段</h3><ul class='tight'>"
        for name, meta in sorted(declared.items()):
            records_html += f"<li><b>{escape(name)}</b><ul class='tight'>" + "".join(
                f"<li>{escape(str(note))}</li>" for note in meta.get("notes") or []
            ) + "</ul></li>"
        records_html += "</ul>"
    sections.append({
        "id": "records",
        "heading": "3. 真实迭代记录（按档位与决策分列）",
        "html": records_html + "<h3>图</h3>" + _figure_gallery(figures),
    })

    gaps = payload.get("gaps", [])
    gaps_html = (
        table(["阶段", "期望档位", "原因", "线索"],
              [[f"<code>{escape(str(gap['stage']))}</code>", badge(gap.get("expected_tier")),
                _text(gap.get("reason")), _text(gap.get("source_hint"))] for gap in gaps])
        + '<p class="note">这些阶段不会出现在曲线中；工具不会对最终网格做反向拟合来补出曲线。</p>'
        if gaps else "<p>无缺口。</p>"
    )
    sections.append({"id": "gaps", "heading": "4. 缺口阶段（无真实状态）", "html": gaps_html})

    if metrics:
        metric_rows = []
        for name, item in metrics.items():
            m = item.get("metrics", {})
            cov = item.get("coverage", {})
            metric_rows.append([
                escape(str(name)), _fmt(cov.get("n_required")), _fmt(m.get("n_valid")),
                _fmt(m.get("mae_mm"), 4), _fmt(m.get("rmse_mm"), 4), _fmt(m.get("p95_mm"), 4),
                _fmt(m.get("median_mm"), 4), _fmt(m.get("max_mm"), 4), _fmt(cov.get("coverage"), 4),
            ])
        sections.append({
            "id": "diagnostics",
            "heading": "5. 独立几何诊断（固定观测，非算法 loss）",
            "html": '<p class="note">每个观测点从相机原点发一条射线打到交付网格，取沿光轴的深度差绝对值；'
                    '超过 30 mm 或未命中记为未覆盖，不计为 0 误差。MAE/RMSE/P95 只在命中子集上统计；'
                    '交付网格位于参考帧相机坐标系，跨视角比较需要该帧估计位姿。</p>'
                    + table(["帧", "观测点", "命中", "MAE (mm)", "RMSE (mm)", "P95 (mm)", "中位数 (mm)",
                             "最大 (mm)", "覆盖"], metric_rows, numeric={1, 2, 3, 4, 5, 6, 7, 8}),
        })

    support = input_summary.get("support") or {}
    notes_html = "<ul class='tight'>" + "".join(
        f"<li>{escape(str(note))}</li>" for note in (extra_notes or [])
    ) + "</ul>"
    if support:
        notes_html += "<h3>支持度标签（整次运行累计）</h3>" + kv_table(
            [(str(key), value) for key, value in support.items()]
        )
    notes_html += (
        "<h3>结论边界</h3><ul class='tight'>"
        "<li>本工具只做观测与统计，不改变优化目标、参数更新、采样、随机种子、迭代次数、接受条件或几何保护。</li>"
        "<li>求解器目标、线性化目标、刷新后的实际能量、独立诊断四类数值不可互相比较或相加。</li>"
        "<li>构造数据上的测试只证明计算与接口正确，不证明真实重建效果。</li>"
        "<li>“loss 下降”不等于“几何更准”；精度仍需独立扫描或人工轮廓验证。</li>"
        "</ul>"
    )
    sections.append({"id": "notes", "heading": "6. 备注与结论边界", "html": notes_html})

    subtitle = (
        f"运行 <code>{escape(str(payload.get('run_label')))}</code> · "
        f"算法 <code>{escape(str(payload.get('algorithm')))}</code> · "
        f"版本 <code>{escape(str(payload.get('software_version')))}</code> · "
        f"生成于 {escape(stamp)}"
    )
    footer = (
        "由 loss_analysis 生成 · 只读分析：不运行重建、不写入运行目录 · "
        "单文件报告（样式与图均内嵌，无外部依赖）"
    )
    return render_page(title=f"loss 分析报告 — {payload.get('run_label')}",
                       subtitle=subtitle, sections=sections, footer=footer)
