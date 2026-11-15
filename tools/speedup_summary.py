"""Aggregate P0/P1/P2 phase reports into one summary (JSON + single-file HTML).

Each phase root must contain ``report.json`` from ``tools.speedup_report`` and
``metrics.json`` from ``tools.speedup_validate``.  The summary never merges
different phases' objectives; it only compares each phase with the same
baseline and states the measured wall-clock effect, including the run-to-run
variance control from the base-only captures.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
RUN_ORDER = ("zp_full", "jzh_base", "s100624_base", "s193720_base")
ITEM_ORDER = ("prepare_observations", "identity_front_anchor",
              "shared_identity_joint", "fixed_identity_motion",
              "guarded_measured_dense_surface", "neck_build",
              "base_stages_sum", "delivery_and_render_overhead", "total_wall")
PHASES = ("p0", "p1", "p2")


def _json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _num(value, digits=3):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _pct(base, cand):
    if base in (None, 0) or cand is None:
        return None
    return (cand - base) / abs(base) * 100.0


def _timing_map(report: dict, run: str) -> dict:
    rows = ((report.get("runs") or {}).get(run) or {}).get("timing") or []
    return {row["item"]: row for row in rows}


def collect(phase_dir: Path) -> dict:
    report = _json(phase_dir / "report.json", {}) or {}
    metrics = _json(phase_dir / "metrics.json", {}) or {}
    runs = {}
    for run in RUN_ORDER:
        row = (report.get("runs") or {}).get(run)
        if not row:
            continue
        timing = _timing_map(report, run)
        runs[run] = {
            "kind": row.get("kind"),
            "timing": {item: timing.get(item) for item in ITEM_ORDER if item in timing},
            "objective_max_diff_pct": row.get("objective_max_diff_pct"),
            "objective_min_diff_pct": row.get("objective_min_diff_pct"),
            "objective_pass": row.get("objective_pass"),
            "fixed_ray_pass": row.get("fixed_ray_pass"),
            "acceptance_pass": all(item.get("pass") is not False
                                   for item in row.get("acceptance") or []),
            "downstream_pass": all(item.get("pass") is not False
                                   for item in row.get("downstream") or []),
        }
    return {
        "label": report.get("candidate_label") or metrics.get("label"),
        "phase": report.get("phase"),
        "pass": bool(report.get("pass")),
        "failed": report.get("failed") or [],
        "runs": runs,
    }


def build(baseline_dir: Path, phase_dirs: dict, render_probe: Path | None) -> dict:
    baseline_metrics = _json(baseline_dir / "metrics.json", {}) or {}
    base_map = {}
    for run in RUN_ORDER:
        rec = (baseline_metrics.get("runs") or {}).get(run)
        if not rec:
            continue
        stages = {name: (row or {}).get("elapsed_s")
                  for name, row in (rec.get("stages") or {}).items()}
        total = sum(v or 0.0 for v in stages.values())
        neck = rec.get("neck_seconds")
        wall = rec.get("wall_s")
        base_map[run] = {
            "kind": rec.get("kind"),
            "timing": {
                **{name: {"baseline_s": value, "candidate_s": None}
                   for name, value in stages.items()},
                "base_stages_sum": {"baseline_s": total, "candidate_s": None},
                "neck_build": {"baseline_s": neck, "candidate_s": None},
                "total_wall": {"baseline_s": wall, "candidate_s": None},
                "delivery_and_render_overhead": {
                    "baseline_s": (wall - total - (neck or 0.0))
                    if wall is not None else None,
                    "candidate_s": None},
            },
        }
    phases = {}
    for name, path in phase_dirs.items():
        if path is None:
            continue
        phases[name] = collect(path)
    probe = _json(render_probe, None) if render_probe else None
    return {
        "schema": "speedup.summary.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline": {"root": str(baseline_dir), "runs": base_map},
        "phases": phases,
        "render_probe": probe,
        "baseline_label": baseline_metrics.get("label"),
    }


def _cell(value):
    if value is None:
        return "<td class='na'>null</td>"
    if isinstance(value, bool):
        return f"<td class='{'pass' if value else 'fail'}'>{'pass' if value else 'fail'}</td>"
    if isinstance(value, float):
        return f"<td>{value:.6g}</td>"
    return f"<td>{html.escape(str(value))}</td>"


def _table(headers, rows, keys):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key)
            if key.endswith("pass") or key == "pass":
                cells.append(_cell(bool(value)) if value is not None
                             else "<td class='na'>null</td>")
            else:
                cells.append(_cell(value))
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(summary: dict) -> str:
    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>主算法提速 P0/P1/P2 汇总</title>",
        """<style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;color:#1d2330;background:#f7f8fa}
h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:26px 0 8px}
h3{font-size:14px;margin:16px 0 4px}p,li{font-size:13px;line-height:1.55}
table{border-collapse:collapse;width:100%;margin:8px 0 16px;background:#fff;
box-shadow:0 1px 2px rgba(0,0,0,.08);font-size:12px}
th,td{border:1px solid #e3e6ec;padding:5px 7px;text-align:left}
th{background:#eef1f6}td.pass{color:#0b6b3a;font-weight:600}td.fail{color:#a8071a;font-weight:600}
td.na{color:#8a8f99}.meta{color:#5b6272}
code{background:#eef1f6;padding:1px 4px;border-radius:3px}
</style></head><body>""",
        "<h1>主算法提速 P0/P1/P2 汇总</h1>",
        "<p class='meta'>生成 " + html.escape(summary["created_at"]) +
        "；基线目录 <code>" + html.escape(summary["baseline"]["root"]) + "</code>。"
        "每阶段只与同一基线比较；跨阶段 objective 不相减。</p>",
    ]
    rows = []
    for name in PHASES:
        phase = summary["phases"].get(name)
        if not phase:
            continue
        rows.append({
            "phase": name.upper(),
            "pass": phase["pass"],
            "failed": len(phase["failed"]),
            "obj": max((run.get("objective_max_diff_pct") or 0)
                       for run in phase["runs"].values() if run) if phase["runs"] else None,
            "fixed": all((run.get("fixed_ray_pass") is not False)
                         for run in phase["runs"].values() if run),
            "gates": all((run.get("acceptance_pass") is not False)
                         for run in phase["runs"].values() if run),
            "down": all((run.get("downstream_pass") is not False)
                        for run in phase["runs"].values() if run),
        })
    parts.append("<h2>阶段判定</h2>")
    parts.append(_table(["阶段", "总判定", "失败项数", "objective 最大恶化 %",
                         "fixed-ray", "验收门", "下游配对"],
                        rows, ["phase", "pass", "failed", "obj", "fixed",
                               "gates", "down"]))
    if summary.get("render_probe"):
        probe = summary["render_probe"]
        parts.append("<p class='meta'>直接渲染探针（在同一 session 的输出上重跑两个渲染工具）："
                     + html.escape(json.dumps(probe, ensure_ascii=False)) + "</p>")
    parts.append("<h2>时间账本（相对同一基线）</h2>")
    for name in PHASES:
        phase = summary["phases"].get(name)
        if not phase:
            continue
        parts.append(f"<h3>{name.upper()}（{html.escape(str(phase['label']))}）</h3>")
        rows = []
        for run, rec in phase["runs"].items():
            timing = rec.get("timing") or {}
            for item in ("base_stages_sum", "neck_build",
                         "delivery_and_render_overhead", "total_wall"):
                row = timing.get(item) or {}
                b = row.get("baseline_s")
                c = row.get("candidate_s")
                rows.append({"run": run, "item": item, "base_s": _num(b),
                             "cand_s": _num(c), "delta_s": _num(c - b)
                             if b is not None and c is not None else None,
                             "delta_pct": _num(_pct(b, c), 2)})
        parts.append(_table(["运行", "项", "基线 s", "候选 s", "Δ s", "Δ %"],
                            rows, ["run", "item", "base_s", "cand_s",
                                   "delta_s", "delta_pct"]))
    parts.append("<h2>全链各阶段（zp 全链）</h2>")
    for name in PHASES:
        phase = summary["phases"].get(name)
        if not phase:
            continue
        run = phase["runs"].get("zp_full") or {}
        rows = []
        for item in ITEM_ORDER:
            row = (run.get("timing") or {}).get(item)
            if not row:
                continue
            b = row.get("baseline_s")
            c = row.get("candidate_s")
            rows.append({"item": item, "base_s": _num(b), "cand_s": _num(c),
                         "delta_s": _num(c - b) if b is not None and c is not None else None,
                         "delta_pct": _num(_pct(b, c), 2)})
        parts.append(f"<h3>{name.upper()}</h3>")
        parts.append(_table(["阶段/项", "基线 s", "候选 s", "Δ s", "Δ %"],
                            rows, ["item", "base_s", "cand_s", "delta_s", "delta_pct"]))
    parts.append("<h2>失败项明细</h2>")
    any_fail = False
    for name in PHASES:
        phase = summary["phases"].get(name)
        if not phase or not phase["failed"]:
            continue
        any_fail = True
        parts.append(f"<h3>{name.upper()}</h3><ul>")
        parts += [f"<li class='fail'>{html.escape(item)}</li>" for item in phase["failed"]]
        parts.append("</ul>")
    if not any_fail:
        parts.append("<p>三阶段均无失败项。</p>")
    parts.append("""<h2>读法</h2><ul>
<li>objective 只在该 phase 内比较；固定射线门要求 final MSE 不高于基线 3%，
终值 max 允许 3%，覆盖要求全部原始射线覆盖；交付 dense 的 final MSE 允许 3%。</li>
<li>总 wall 含 prepare/颈部/交付，单次运行有 10–20% 的机器方差；三条 base-only 是
“P0/P1/P2 不应改变它们”的对照，其 wall 差值用于估计方差，不作为省时。</li>
<li>P0 的直接效果用两个渲染工具的独立计时（render_probe）与
delivery_and_render_overhead 行衡量。</li>
</ul></body></html>""")
    return "".join(parts)


def render_markdown(summary: dict) -> str:
    lines = ["# 主算法提速 P0/P1/P2 汇总", "",
             f"- 基线：`{summary['baseline']['root']}`",
             f"- 生成：{summary['created_at']}", ""]
    for name in PHASES:
        phase = summary["phases"].get(name)
        if not phase:
            continue
        lines.append(f"## {name.upper()}：{'PASS' if phase['pass'] else 'FAIL'}")
        lines.append("")
        if phase["failed"]:
            lines += [f"- 失败：`{item}`" for item in phase["failed"]]
        else:
            lines.append("- 失败：无")
        for run, rec in phase["runs"].items():
            timing = rec.get("timing") or {}
            bw = (timing.get("total_wall") or {}).get("baseline_s")
            cw = (timing.get("total_wall") or {}).get("candidate_s")
            ss = (timing.get("base_stages_sum") or {})
            over = (timing.get("delivery_and_render_overhead") or {})
            lines.append(f"- {run}: wall {bw} → {cw} s；stage sum "
                         f"{ss.get('baseline_s')} → {ss.get('candidate_s')} s；"
                         f"overhead {over.get('baseline_s')} → {over.get('candidate_s')} s；"
                         f"objective max diff {rec.get('objective_max_diff_pct')}%")
        lines.append("")
    if summary.get("render_probe"):
        lines += ["## 渲染探针", "",
                  f"- {json.dumps(summary['render_probe'], ensure_ascii=False)}", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--p0", type=Path)
    parser.add_argument("--p1", type=Path)
    parser.add_argument("--p2", type=Path)
    parser.add_argument("--render-probe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    dirs = {"p0": args.p0, "p1": args.p1, "p2": args.p2}
    summary = build(args.baseline, dirs, args.render_probe)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
        encoding="utf-8")
    (out / "report.html").write_text(render_html(summary), encoding="utf-8")
    (out / "REBUILD_REPORT.md").write_text(render_markdown(summary),
                                           encoding="utf-8")
    failed = [name for name, phase in summary["phases"].items() if not phase["pass"]]
    print(json.dumps({"ok": not failed, "failed": failed,
                      "out": str(out / "report.html")}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
