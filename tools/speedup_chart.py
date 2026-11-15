"""Render the P0/P2 wall-clock and loss-time changes as a PNG + single-file HTML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

PROJ = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = {
    "baseline": PROJ / "runtime/outputs/_speedup_baseline_20261115_214949",
    "p0": PROJ / "runtime/outputs/_speedup_p0_20261115_221350",
    "p2": PROJ / "runtime/outputs/_speedup_p2_20261115_233622",
}
STAGE_ORDER = (
    ("prepare_observations", "预处理 prepare", "#8c9bab"),
    ("identity_front_anchor", "fit_c 正面锚", "#4e79a7"),
    ("shared_identity_joint", "fit_c 联合身份", "#2f5d8a"),
    ("fixed_identity_motion", "motion_v2 运动校正", "#59a14f"),
    ("guarded_measured_dense_surface", "dense 表面", "#b07aa1"),
)
LABELS = {
    "baseline": "0.9.0 基线（渲染开、legacy loss）",
    "p0": "P0：无渲染、legacy loss",
    "p2": "当前默认：P0 + P2(batch loss)",
}


def _run(root: Path, key: str = "zp_full") -> dict:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    return metrics["runs"][key]


def _stage_values(rec: dict) -> dict:
    values = {}
    for name, _, _ in STAGE_ORDER:
        values[name] = float((rec.get("stages", {}).get(name) or {}).get("elapsed_s") or 0.0)
    values["neck"] = float(rec.get("neck_seconds") or 0.0)
    stages_sum = sum(float((row or {}).get("elapsed_s") or 0.0)
                     for row in (rec.get("stages") or {}).values())
    values["delivery"] = float(rec.get("wall_s") or 0.0) - stages_sum - values["neck"]
    values["wall"] = float(rec.get("wall_s") or 0.0)
    return values


def _loss_pair(rec: dict) -> dict:
    pairs = {}
    for key, label in (("front_anchor", "fit_c 正面"),
                       ("guarded", "fit_c 联合"),
                       ("motion_only", "motion_v2")):
        if key == "motion_only":
            timing = rec["motion"]["timing"]
        else:
            timing = rec["fit_c"][key]["timing"]
        pairs[label] = float(timing["loss_s"]) + float(timing["backward_s"])
    return pairs


def _savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_png(data: dict, out: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16.5, 11.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0],
                            height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.42)

    # --- panel 1: stacked wall bars -------------------------------------
    ax = fig.add_subplot(grid[:, 0])
    configs = ["baseline", "p0", "p2"]
    stage_keys = [name for name, _, _ in STAGE_ORDER] + ["neck", "delivery"]
    stage_labels = [label for _, label, _ in STAGE_ORDER] + ["颈部 C", "交付+渲染"]
    colors = [color for _, _, color in STAGE_ORDER] + ["#e15759", "#f28e2b"]
    y = list(range(len(configs)))[::-1]
    max_total = max(data[name]["wall"] for name in configs)
    for yi, name in zip(y, configs):
        left = 0.0
        values = data[name]
        for key, label, color in zip(stage_keys, stage_labels, colors):
            value = values[key]
            ax.barh(yi, value, left=left, color=color, height=0.58,
                    edgecolor="white", linewidth=0.8)
            if value >= 12:
                ax.text(left + value / 2, yi, f"{value:.0f}",
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
            left += value
        total = values["wall"]
        ax.text(total + 3, yi, f"{total:.1f} s", va="center", fontsize=11,
                fontweight="bold", color="#1d2330")
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[name] for name in configs], fontsize=10.5)
    ax.set_xlim(0, max_total * 1.14)
    ax.set_xlabel("全链 wall time（秒；含子进程启动）", fontsize=11)
    ax.set_title("zp 全链时间分解：0.9.0 → P0 → 当前默认", fontsize=14, fontweight="bold",
                 loc="left", pad=36)
    ax.grid(axis="x", color="#dfe3ea", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l)
                       for c, l in zip(colors, stage_labels)],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9,
              frameon=False, ncol=4)
    ax.text(0.01, 1.008, f"P0 移除渲染 ≈28 s；P2 loss+backward −{data['loss_saving']:.1f} s",
            transform=ax.transAxes, fontsize=9.5, color="#5b6272")
    baseline_wall = data["baseline"]["wall"]
    current_wall = data["p2"]["wall"]
    ax.text(0.99, 1.008, f"总时间 {baseline_wall:.1f} → {current_wall:.1f} s"
                         f"（{current_wall - baseline_wall:+.1f} s，"
                         f"{(current_wall / baseline_wall - 1) * 100:+.1f}%）",
            transform=ax.transAxes, ha="right", fontsize=9.5,
            color="#0b6b3a", fontweight="bold")

    # --- panel 2: loss + backward --------------------------------------
    ax2 = fig.add_subplot(grid[0, 1])
    stages = list(data["baseline_loss"].keys())
    x = range(len(stages))
    width = 0.36
    base_vals = [data["baseline_loss"][s] for s in stages]
    p2_vals = [data["p2_loss"][s] for s in stages]
    ax2.bar([i - width / 2 for i in x], base_vals, width, color="#c9d2e0",
            label="0.9.0 基线")
    ax2.bar([i + width / 2 for i in x], p2_vals, width, color="#2f5d8a",
            label="当前默认 (batch)")
    for i, (b, c) in enumerate(zip(base_vals, p2_vals)):
        ax2.text(i - width / 2, b + 0.4, f"{b:.1f}", ha="center", fontsize=8.5, color="#4b5563")
        ax2.text(i + width / 2, c + 0.4, f"{c:.1f}", ha="center", fontsize=8.5, color="#1d2330")
        ax2.text(i, max(b, c) + 1.8, f"−{(1 - c / b) * 100:.0f}%", ha="center",
                 fontsize=10, fontweight="bold", color="#0b6b3a")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(stages, fontsize=9.5)
    ax2.set_ylabel("loss + backward（秒）", fontsize=10)
    ax2.set_title("每阶段 loss+backward：逐帧循环 → 帧维批量", fontsize=12.5,
                  fontweight="bold", loc="left")
    ax2.grid(axis="y", color="#dfe3ea", linewidth=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.legend(fontsize=9, frameon=False)

    # --- panel 3: base-only captures -----------------------------------
    ax3 = fig.add_subplot(grid[1, 1])
    names = ["jzh", "100624", "193720"]
    base_wall = [data["base_only"]["baseline"][n] for n in names]
    p2_wall = [data["base_only"]["p2"][n] for n in names]
    x = range(len(names))
    ax3.bar([i - width / 2 for i in x], base_wall, width, color="#c9d2e0",
            label="0.9.0 基线")
    ax3.bar([i + width / 2 for i in x], p2_wall, width, color="#59a14f",
            label="当前默认 (base-only)")
    for i, (b, c) in enumerate(zip(base_wall, p2_wall)):
        ax3.text(i - width / 2, b + 2, f"{b:.0f}", ha="center", fontsize=8.5, color="#4b5563")
        ax3.text(i + width / 2, c + 2, f"{c:.0f}", ha="center", fontsize=8.5, color="#1d2330")
        ax3.text(i, max(b, c) + 12, f"−{(1 - c / b) * 100:.1f}%", ha="center",
                 fontsize=10, fontweight="bold", color="#0b6b3a")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels([f"{n}（base-only）" for n in names], fontsize=9.5)
    ax3.set_ylabel("wall time（秒）", fontsize=10)
    ax3.set_ylim(0, max(base_wall + p2_wall) * 1.18)
    ax3.set_title("三条 0.9.0 base-only 对照（不含颈部/渲染）", fontsize=12.5,
                  fontweight="bold", loc="left", pad=14)
    ax3.grid(axis="y", color="#dfe3ea", linewidth=0.8)
    ax3.set_axisbelow(True)
    for side in ("top", "right"):
        ax3.spines[side].set_visible(False)
    ax3.legend(fontsize=9, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=2)

    fig.text(0.01, -0.22,
             "注：颈部 build 未改动，dense 的微小变化会让 remesh 路径在 37–143 s 间跳变；"
             "因此总 wall 会波动，稳定的省时证据是渲染探针与 loss+backward。base-only 的差值为 "
             "P2 效果 + 单次运行方差。",
             fontsize=8.8, color="#5b6272", ha="left", va="top")
    _savefig(fig, out)


def _bar_row(name: str, values: dict, max_total: float, colors: list[str],
             labels: list[str]) -> str:
    segments = []
    for key, color, label in zip(["prepare_observations", "identity_front_anchor",
                                  "shared_identity_joint", "fixed_identity_motion",
                                  "guarded_measured_dense_surface", "neck", "delivery"],
                                 colors, labels):
        value = values[key]
        width = value / max_total * 100.0
        text = f"{value:.0f}" if value >= 12 else ""
        segments.append(
            f"<div class='seg' title='{label}: {value:.1f} s' "
            f"style='width:{width:.3f}%;background:{color}'>{text}</div>")
    return (f"<div class='row'><div class='rowlabel'>{name}</div>"
            f"<div class='track'>{''.join(segments)}</div>"
            f"<div class='total'>{values['wall']:.1f} s</div></div>")


def make_html(data: dict, out: Path) -> None:
    colors = ["#8c9bab", "#4e79a7", "#2f5d8a", "#59a14f", "#b07aa1", "#e15759", "#f28e2b"]
    labels = ["预处理", "fit_c 正面", "fit_c 联合", "motion_v2", "dense", "颈部 C", "交付+渲染"]
    max_total = max(data[name]["wall"] for name in ("baseline", "p0", "p2"))
    rows = "".join(_bar_row(LABELS[name], data[name], max_total, colors, labels)
                   for name in ("baseline", "p0", "p2"))
    loss_rows = []
    for stage in data["baseline_loss"]:
        b = data["baseline_loss"][stage]
        c = data["p2_loss"][stage]
        loss_rows.append(
            f"<tr><td>{stage}</td><td>{b:.3f}</td><td>{c:.3f}</td>"
            f"<td class='good'>−{(1 - c / b) * 100:.1f}%</td></tr>")
    base_rows = []
    for n in ("jzh", "100624", "193720"):
        b = data["base_only"]["baseline"][n]
        c = data["base_only"]["p2"][n]
        base_rows.append(f"<tr><td>{n}（base-only）</td><td>{b:.1f}</td><td>{c:.1f}</td>"
                         f"<td class='good'>−{(1 - c / b) * 100:.1f}%</td></tr>")
    stage_table = []
    for key, label, color in STAGE_ORDER:
        b = data["baseline"][key]
        c = data["p2"][key]
        stage_table.append(f"<tr><td><span class='dot' style='background:{color}'></span>"
                           f"{label}</td><td>{b:.1f}</td><td>{c:.1f}</td>"
                           f"<td>{c - b:+.1f}</td></tr>")
    for key, label, color in (("neck", "颈部 C", "#e15759"),
                              ("delivery", "交付+渲染", "#f28e2b")):
        b = data["baseline"][key]
        c = data["p2"][key]
        stage_table.append(f"<tr><td><span class='dot' style='background:{color}'></span>"
                           f"{label}</td><td>{b:.1f}</td><td>{c:.1f}</td>"
                           f"<td>{c - b:+.1f}</td></tr>")
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>重建时间变化：0.9.0 → P0 → 当前默认</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;background:#f7f8fa;color:#1d2330}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:26px 0 10px}}
p,li{{font-size:13px;line-height:1.55;color:#3b4252}}.meta{{color:#5b6272}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:12px 16px;min-width:170px}}
.card .k{{font-size:12px;color:#5b6272}}.card .v{{font-size:24px;font-weight:700}}
.good{{color:#0b6b3a}}.bad{{color:#a8071a}}
.row{{display:flex;align-items:center;gap:10px;margin:6px 0}}
.rowlabel{{width:230px;font-size:12.5px;text-align:right;color:#3b4252}}
.track{{flex:1;display:flex;height:30px;background:#eef1f6;border-radius:4px;overflow:hidden}}
.seg{{display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:600}}
.total{{width:76px;text-align:right;font-weight:700;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:12.5px;box-shadow:0 1px 2px rgba(0,0,0,.06)}}
th,td{{border:1px solid #e3e6ec;padding:6px 8px;text-align:left}}th{{background:#eef1f6}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:7px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#3b4252;margin:8px 0 4px}}
.legend span{{display:inline-flex;align-items:center;gap:5px}}
.note{{background:#fff;border-left:4px solid #f28e2b;padding:10px 14px;font-size:12.5px;color:#3b4252}}
</style></head><body>
<h1>重建时间变化：0.9.0 → P0 → 当前默认（P0+P2）</h1>
<p class='meta'>数据来源：同一 session 的 zp 参考采集全链 330.30 s 基线、P0 282.66 s、当前默认 268.01 s；
base-only 三条为 jzh / 100624 / 193720。详细逐项 pass/fail 见 <code>report.html</code>。</p>
<div class='cards'>
<div class='card'><div class='k'>全链总时间</div><div class='v'>330.3 → 268.0 s</div><div class='good'>−62.3 s（−18.9%）</div></div>
<div class='card'><div class='k'>渲染移除（直接探针）</div><div class='v'>≈28.0–29.0 s</div><div class='good'>P0</div></div>
<div class='card'><div class='k'>loss + backward</div><div class='v'>45.5 → 11.9 s</div><div class='good'>−73.8%（P2）</div></div>
<div class='card'><div class='k'>base-only</div><div class='v'>122.9–144.2 s</div><div class='good'>−8.4…−24.4%</div></div>
</div>

<h2>全链 wall 分解（含子进程启动）</h2>
<div class='legend'>{''.join(f"<span><i class='dot' style='background:{c}'></i>{l}</span>" for c,l in zip(colors,labels))}</div>
{rows}
<p class='meta'>基线条形 330.3 s；P0 去掉渲染与展示副本后 282.7 s；当前默认在 P0 基础上把 loss+backward 从 45.5 s 降到 11.9 s，但同 run 颈部 remesh 路径变长（108.5→132.4 s），所以总时间为 268.0 s。颈部不是本轮优化项。</p>

<h2>每阶段 loss + backward</h2>
<table><thead><tr><th>阶段</th><th>0.9.0 基线 (s)</th><th>当前默认 (s)</th><th>变化</th></tr></thead>
<tbody>{''.join(loss_rows)}</tbody></table>

<h2>逐阶段 wall 明细（zp 全链）</h2>
<table><thead><tr><th>阶段</th><th>基线 (s)</th><th>当前默认 (s)</th><th>Δ (s)</th></tr></thead>
<tbody>{''.join(stage_table)}</tbody></table>

<h2>三条 0.9.0 base-only 对照</h2>
<table><thead><tr><th>采集</th><th>基线 wall (s)</th><th>当前默认 wall (s)</th><th>变化</th></tr></thead>
<tbody>{''.join(base_rows)}</tbody></table>

<div class='note'><b>读法/限制：</b>总 wall 含 prepare 与子进程启动，单次运行有 10–20% 机器方差；三条 base-only 的差值是 P2 效果加方差。
颈部 build 未改动，dense 的微小变化会让 remesh 路径在 37–143 s 间跳变，因此总 wall 会波动；稳定的省时证据是渲染直接探针与
loss+backward。P1 的步数/刷新缩减因下游 fixed-ray 与颈部 crease 退化未采用，故当前默认仍是 240/240/80/刷新 10。</div>
</body></html>"""
    out.write_text(html, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_ROOTS["baseline"])
    parser.add_argument("--p0", type=Path, default=DEFAULT_ROOTS["p0"])
    parser.add_argument("--p2", type=Path, default=DEFAULT_ROOTS["p2"])
    parser.add_argument("--out", type=Path,
                        default=PROJ / "runtime/outputs/_speedup_final_20261115_234941")
    args = parser.parse_args(argv)
    roots = {"baseline": args.baseline, "p0": args.p0, "p2": args.p2}
    data = {}
    for name, root in roots.items():
        rec = _run(root)
        data[name] = _stage_values(rec)
    data["baseline_loss"] = _loss_pair(_run(args.baseline))
    data["p2_loss"] = _loss_pair(_run(args.p2))
    data["loss_saving"] = (sum(data["baseline_loss"].values())
                           - sum(data["p2_loss"].values()))
    base_keys = {"jzh": "jzh_base", "100624": "s100624_base",
                 "193720": "s193720_base"}
    data["base_only"] = {
        "baseline": {name: float(_run(args.baseline, key)["wall_s"])
                     for name, key in base_keys.items()},
        "p2": {name: float(_run(args.p2, key)["wall_s"])
               for name, key in base_keys.items()},
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    make_png(data, out / "timing_breakdown.png")
    make_html(data, out / "timing_breakdown.html")
    print(json.dumps({"png": str(out / "timing_breakdown.png"),
                      "html": str(out / "timing_breakdown.html")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
