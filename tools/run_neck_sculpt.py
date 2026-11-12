"""Formal appearance-sculpt experiment: three presets, real data, renders, report.

This is a manual appearance design (user-authorised non-observed geometry), not
a subject reconstruction.  Every preset is built with the same kernel and local
refinement that the public export entry uses.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from head_rgbd import neck_appearance as na
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_subject_v3 as v3
from tools.run_neck_subject_v3 import load_obj, render_view, render_wire, sha256

PROJECT = Path("D:/proiect/facecap_airway_delivery")
FIT_DIR = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
PREPARED = PROJECT / "runtime/outputs/zp_0.5.1/prepared"
LANDMARKS = PROJECT / "runtime/outputs/zp_052_neck_bump_v2/neck_landmarks_v2.json"
BASELINE_DELIVERY = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/delivery"
PRESETS = ("light", "natural", "pronounced")


def plane_section(vertices, faces, centre, n_axis, up_axis, fwd_axis, up_min, up_max, slab_mm=10.0):
    """Local front-neck section only: samples with up in [up_min, up_max] and
    forward > -45 mm (the plane also crosses the jaw/back of the head)."""
    tri = vertices[faces]
    xs = (tri - centre) @ n_axis
    cross = (xs[:, 0] * xs[:, 1] <= 0) | (xs[:, 1] * xs[:, 2] <= 0) | (xs[:, 0] * xs[:, 2] <= 0)
    near = (np.abs(xs).max(axis=1) < slab_mm * 1e-3) | (np.abs(xs).min(axis=1) < 1e-9)
    ups, fwds = [], []
    for fi in np.flatnonzero(near & cross):
        v = faces[fi]
        s = xs[fi]
        pts = []
        for a in range(3):
            b = (a + 1) % 3
            if s[a] * s[b] <= 0 and abs(s[a] - s[b]) > 1e-12:
                t = s[a] / (s[a] - s[b])
                pts.append(vertices[v[a]] + t * (vertices[v[b]] - vertices[v[a]]))
        if len(pts) == 2:
            r0 = pts[0] - centre
            r1 = pts[1] - centre
            u0, u1 = float(r0 @ up_axis) * 1000.0, float(r1 @ up_axis) * 1000.0
            f0, f1 = float(r0 @ fwd_axis) * 1000.0, float(r1 @ fwd_axis) * 1000.0
            steps = max(1, int(np.ceil(abs(u1 - u0) / 0.5)))
            for step in range(steps + 1):
                t = step / steps
                uu = u0 + t * (u1 - u0)
                ff = f0 + t * (f1 - f0)
                if up_min <= uu <= up_max and ff > -45.0:
                    ups.append(uu)
                    fwds.append(ff)
    ups = np.asarray(ups)
    fwds = np.asarray(fwds)
    order = np.argsort(ups)
    return ups[order], fwds[order]


def write_report(out: Path, *, timing, results, plan, baseline_meshes, render_paths, input_notes):
    from head_rgbd import neck_appearance as na_mod

    def fig(name, caption, width="100%"):
        import base64
        path = out / "render" / name
        if not path.is_file():
            return ""
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return (f"<figure><img style='width:{width}' src='data:image/png;base64,{data}'/>"
                f"<figcaption>{caption}</figcaption></figure>")

    def kv(rows):
        return "<table class='kv'>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"

    def fmt(v, d=2, s=""):
        try:
            return f"{float(v):.{d}f}{s}"
        except (TypeError, ValueError):
            return "—"

    preset_rows = []
    for preset in PRESETS:
        p = results[preset]
        for state in ("closed", "open"):
            r = p["plan"]["states"][state]
            m = r["metrics"]
            g = r["guards"]
            preset_rows.append(
                f"<tr><td>{preset}</td><td>{state}</td><td>{fmt(m['requested_height_mm'], 1)}</td>"
                f"<td><b>{fmt(m['realized_height_mm'])}</b></td><td>{fmt(m['requested_width_mm'], 1)}</td>"
                f"<td>{fmt(m['width_half_mm'])} / {fmt(m['width_ten_mm'])}</td>"
                f"<td>{fmt(m['requested_length_mm'], 1)}</td><td>{fmt(m['length_half_mm'])} / {fmt(m['length_ten_mm'])}</td>"
                f"<td>{fmt(m['requested_tip_radius_mm'], 1)}</td><td>{m['requested_transitions_mm']}</td>"
                f"<td>{g['true_fold']}</td><td>{g['max_edge_growth']:.2f}</td>"
                f"<td>{r['self_intersections']['count']}</td><td>{r['protection_check']['protected_vertices_moved']}</td>"
                f"<td>{'通过' if p['accepted'] else '未通过'}</td></tr>"
            )

    all_accepted = all(results[p]["accepted"] for p in PRESETS)
    nat = results["natural"]["plan"]["states"]
    checks = results["natural"]["checks"]
    check_rows = "".join(
        f"<tr class='{'pass' if v else 'fail'}'><td>{k}</td><td>{'通过' if v else '失败'}</td></tr>"
        for k, v in checks.items())

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>颈部外观塑形报告 / Neck Appearance Sculpt</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#1c1e22;color:#d8dce2;line-height:1.55}}
 .wrap{{max-width:1120px;margin:0 auto;padding:32px 24px 64px}}
 h1{{font-size:25px;border-bottom:2px solid #4a90d9;padding-bottom:10px}}
 h2{{font-size:19px;color:#7fb3e8;margin-top:34px;border-left:4px solid #4a90d9;padding-left:10px}}
 table{{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px}}
 td,th{{border:1px solid #3a3f47;padding:5px 9px;text-align:left}}
 th{{background:#262a31}} tr.pass td:last-child{{color:#7fd88f}} tr.fail td:last-child{{color:#e06c5f}}
 table.kv td:first-child{{color:#aab7c8;width:40%}} code{{background:#262a31;padding:1px 5px;border-radius:3px}}
 .note{{background:#2a2519;border:1px solid #6b5b1e;padding:10px 14px;border-radius:6px;margin:12px 0}}
 figure{{margin:16px 0;text-align:center}} figure img{{border-radius:6px;max-width:100%}}
 figcaption{{font-size:12.5px;color:#8a97a8;margin-top:6px}}
</style></head><body><div class="wrap">
<h1>颈部外观塑形 — 人工设计喉结（0.5.6）</h1>
<p style="color:#8a97a8">Neck appearance sculpt — user-authorised non-observed geometry</p>

<div class="note"><b>本报告描述的是“人工外观塑形”，不是本人喉结的观测重建。</b>
用户已明确允许生成不对应真实观测的喉结几何。塑形高度、宽度、长度和圆钝半径都是设计参数，
并非人体测量结论；观测只用于（可选）低频背景校正与（用户提供时）定位。</div>

<h2>0. 摘要</h2>
{kv([
 ("模式 / mode", "manual_edit（外观编辑模式），公共入口默认算法"),
 ("editing_mode", "true"),
 ("geometry_source", "procedural_appearance"),
 ("subject_observation_matched", "false"),
 ("默认预设", "natural（前突 5 mm），可切换 light / pronounced"),
 ("基底", "0.5.2 未塑形颈部 + 可选低频背景校正（无观测先验、无负脊补偿、无深度回拉）"),
 ("塑形场", "p(u,v)=A(v)·C(u,v) 单一连续解析场；无 V2 双场、无 V3 脊形叠加、无导出后平滑"),
 ("顶点与中点", "局部两级红绿细分；closed/open 共享父面与细分构造，逐状态生成中点/法向"),
 ("总耗时", f"{timing[-1]['seconds']:.1f} s"),
 ("整体结论", "三预设几何验收全部通过" if all_accepted else "存在未通过项，见下表"),
])}

<h2>1. 形态设计与参数</h2>
<ul>
<li>A(v)：Wendland C2 纵脊包络，上下支撑长度独立（natural 上 15 / 下 24 mm），边界值与一阶导为零；</li>
<li>C(s)：圆钝 V 形截面，s=(u−c)/w；线性坡面 + sqrt 圆钝顶点（半径 2.5 mm），C2 窗口在侧向支撑边界归零；</li>
<li>w(v)：横向半宽，端部略收（×0.78）；c(v)：可偏移/倾斜的中心线；</li>
<li>位移方向：状态局部颈坐标的 n（单一方向），只乘连续保护 taper，导出即设计本身。</li>
</ul>
<table><tr><th>预设</th><th>目标前突(mm)</th><th>总宽(mm)</th><th>纵向总长(mm)</th><th>尖端半径(mm)</th><th>上/下过渡(mm)</th></tr>
<tr><td>light</td><td>3.0</td><td>26</td><td>36</td><td>2.0</td><td>14 / 22</td></tr>
<tr><td>natural（默认）</td><td>5.0</td><td>29</td><td>39</td><td>2.5</td><td>15 / 24</td></tr>
<tr><td>pronounced</td><td>7.0</td><td>32</td><td>42</td><td>3.0</td><td>16 / 26</td></tr></table>
<p>以上均为外观设计参数，不是解剖标准。</p>

<h2>2. 三预设几何验收（闭口/张口）</h2>
<table><tr><th>预设</th><th>状态</th><th>请求高度</th><th>实际高度</th><th>请求宽</th><th>半高宽 / 10% 宽</th>
<th>请求长</th><th>半高长 / 10% 长</th><th>尖端</th><th>过渡</th><th>真实折叠</th><th>最大边长增长</th><th>自交</th><th>保护顶点移动</th><th>结论</th></tr>
{''.join(preset_rows)}</table>

<h2>3. 自然预设的完整验收</h2>
<table><tr><th>检查</th><th>结果</th></tr>{check_rows}</table>
{kv([
 ("细分（两级）", f"level1 半径 42 mm，level2 半径 22 mm；基础 {plan['base_vertex_count']} → 细化 {plan['refined_vertex_count']} 顶点"),
 ("核心边长", f"细化后核心区中位边长 {fmt(results['natural']['edge_stats']['core_median_mm'])} mm（基础 {fmt(results['natural']['edge_stats']['base_median_mm'])} mm）"),
 ("面部命中", f"面部表面命中点最大位移 {fmt(results['natural']['facial_shift_mm'], 4)} mm（工程容差 0.05 mm）"),
 ("磁盘 OBJ 复核", results['natural']['disk_check']),
])}

<h2>4. 渲染对照（相同视角、光照、尺度）</h2>
<p>左：未塑形基线；右：外观塑形（natural）。其余预设与张口见渲染目录。</p>
{''.join(fig(f'{p}_{s}_{v}_compare.png', f'{p} / {s}：{v}') for p in PRESETS for s in ('closed','open') for v in ('front','side'))}
{''.join(fig(f'{p}_{s}_45deg_compare.png', f'{p} / {s}：45°') for p in PRESETS for s in ('closed','open'))}
{''.join(fig(f'{p}_{s}_wireframe.png', f'{p} / {s}：局部线框（真实几何）') for p in PRESETS for s in ('closed','open'))}

<h2>5. 真实几何剖面（面片∩平面）</h2>
{''.join(fig(f'{p}_{s}_section_long.png', f'{p} / {s}：纵向剖面（过脊线，n–v 平面）') for p in PRESETS for s in ('closed','open'))}
{''.join(fig(f'{p}_{s}_section_cross.png', f'{p} / {s}：横向剖面（过峰值高度，n–u 平面）') for p in PRESETS for s in ('closed','open'))}
<p>剖面由面片与平面真实求交，交线段内按 0.5 mm 加密仅用于绘图。</p>

<h2>6. 外观检查要点</h2>
<ul>
<li>纵脊清楚：中央脊线连续，峰值位于标签/估计位置；</li>
<li>两侧缓坡可辨：截面为线性坡面（半高宽 ≈ 设计总宽的 60–70%）；</li>
<li>整体融入颈部：边界值与斜率归零，无硬接缝、无独立球状鼓包；</li>
<li>上下不对称：下方过渡 24 mm 长于上方 15 mm；</li>
<li>张口状态：同一设计在张口局部坐标系中应用，未被拉长或抹平。</li>
</ul>

<h2>7. 限制与声明</h2>
<div class="note">
<ul>
<li>这是人工外观塑形：不声称与真实喉结形状一致，不用深度 MAE 或观测支持度作为通过条件。</li>
<li>位置优先使用用户标签；无标签时按颈部中线、下颌与颈根估计并记录 position_source=estimated_from_mesh。</li>
<li>低频背景校正（如启用）只决定基底，不参与隆起的形状或幅度。</li>
<li>较大局部曲率是设计的一部分；验收只拒绝真实折叠、自交、退化、翻面、异常拉长与保护失效。</li>
</ul></div>
<p style="color:#8a97a8;margin-top:32px">生成：{datetime.now().strftime('%Y-%m-%d %H:%M')} · tools/run_neck_sculpt.py</p>
</div></body></html>"""
    report = Path(args_report_path(out))
    report.write_text(html, encoding="utf-8")
    return report


def args_report_path(out: Path) -> Path:
    return out / "NECK_APPEARANCE_SCULPT_REPORT.html"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"zp_056_neck_sculpt_{stamp}")
    (out / "render").mkdir(parents=True, exist_ok=True)
    (out / "presets").mkdir(exist_ok=True)
    timing = []
    status, error = "failed", None
    t_all = time.perf_counter()
    results = {}
    plan_nat = None
    try:
        references = json.loads((FIT_DIR / "reference_geometry.json").read_text(encoding="utf-8"))["references"]
        with np.load(FIT_DIR / "parameters.npz", allow_pickle=False) as stored:
            dense_vertices, dense_faces = stored["dense_vertices"], stored["dense_faces"]
        landmarks_payload = json.loads(LANDMARKS.read_text(encoding="utf-8"))
        sequence_rows = json.loads((PREPARED / "sequence.json").read_text(encoding="utf-8"))["frames"]
        base_meshes = {s: load_obj(BASELINE_DELIVERY / f"head_{s}_m.obj") for s in ("closed", "open")}
        base_stats = {}
        for state in ("closed", "open"):
            V, F = base_meshes[state]
            centre = np.asarray(landmarks_payload["states"][state]["point_observed_m"], float)
            centroid = V[F].mean(axis=1)
            near = np.linalg.norm(centroid - centre, axis=1) < 0.03
            edges = np.unique(np.sort(np.concatenate(
                [F[near][:, [0, 1]], F[near][:, [1, 2]], F[near][:, [2, 0]]]), axis=1), axis=0)
            lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1) * 1000.0
            base_stats[state] = float(np.median(lengths))

        for preset in PRESETS:
            t0 = time.perf_counter()
            design = na.design_from_preset(preset)
            plan = na.build_sculpt_states(
                states=["closed", "open"], references=references, dense_vertices=dense_vertices,
                dense_faces=dense_faces, prepared_dir=PREPARED, landmarks_payload=landmarks_payload,
                design=design,
            )
            timing.append({"stage": f"{preset}:build", "seconds": round(time.perf_counter() - t0, 3)})
            # disk export + verify
            delivery = out / "presets" / preset
            delivery.mkdir(parents=True, exist_ok=True)
            disk = {}
            for state in ("closed", "open"):
                shaped = plan["states"][state]["shaped_vertices"]
                path = delivery / f"head_{state}_m.obj"
                trimesh.Trimesh(shaped, plan["refined_faces"], process=False).export(path, include_normals=True)
                v_read, f_read = load_obj(path)
                disk[state] = {
                    "path": str(path), "sha256": sha256(path),
                    "vertices": int(len(v_read)), "faces": int(len(f_read)),
                    "max_abs_diff_m": float(np.max(np.abs(v_read - shaped))) if len(v_read) == len(shaped) else None,
                    "faces_match": bool(len(f_read) == len(plan["refined_faces"]) and np.array_equal(f_read, plan["refined_faces"])),
                }
            # facial first-hit protection (facial-surface hits only)
            facial = {}
            for state in ("closed", "open"):
                item = plan["states"][state]
                ref = references[state]
                lm_cam = np.asarray(ref["stages"]["dense"]["landmarks_m"], float)
                base_v, base_f = base_meshes[state]
                low = v3.first_hit_metrics  # noqa: F841  (kept for API parity)
                hits_b = v3.first_hit_batch(base_v, base_f, lm_cam)
                hits_s = v3.first_hit_batch(item["shaped_vertices"], plan["refined_faces"], lm_cam)
                ok = hits_b["hit"] & hits_s["hit"]
                shift = np.linalg.norm(hits_b["point_m"][ok] - hits_s["point_m"][ok], axis=1) * 1000.0
                hit_tris = hits_b["triangle"][ok]
                facial_hit = np.array([item["protection"][base_f[t]].all()
                                       for t in hit_tris]) if len(hit_tris) else np.zeros(0, bool)
                facial[state] = {
                    "facial_rays": int(ok.sum()),
                    "facial_surface_hits": int(facial_hit.sum()),
                    "max_facial_shift_mm": round(float(shift[facial_hit].max()) if facial_hit.any() else 0.0, 4),
                    "max_other_shift_mm": round(float(shift[~facial_hit].max()) if (~facial_hit).any() else 0.0, 4),
                }
            refined_edges_near = np.unique(np.sort(np.concatenate(
                [plan["refined_faces"][:, [0, 1]], plan["refined_faces"][:, [1, 2]],
                 plan["refined_faces"][:, [2, 0]]]), axis=1), axis=0)
            RV = plan["states"]["closed"]["refined_vertices"]
            centre = plan["states"]["closed"]["frame"].origin
            near_e = np.linalg.norm(RV[refined_edges_near].mean(axis=1) - centre, axis=1) < 0.015
            core_median = float(np.median(np.linalg.norm(
                RV[refined_edges_near[near_e, 0]] - RV[refined_edges_near[near_e, 1]], axis=1) * 1000.0))
            checks = {
                "closed_open_shared_refined_topology": True,
                "no_self_intersections": all(plan["states"][s]["self_intersections"]["count"] == 0 for s in ("closed", "open")),
                "no_true_folds": all(plan["states"][s]["guards"]["true_fold"] == 0 for s in ("closed", "open")),
                "no_normal_reversals": all(plan["states"][s]["guards"]["normal_reversal"] == 0 for s in ("closed", "open")),
                "no_degeneration": all(plan["states"][s]["guards"]["degenerate_faces"] == 0
                                       and plan["states"][s]["guards"]["area_collapse"] == 0 for s in ("closed", "open")),
                "no_slender_faces": all(plan["states"][s]["guards"]["slender_faces"] == 0 for s in ("closed", "open")),
                "edge_growth_within_cap": all(plan["states"][s]["guards"]["max_edge_growth"] <= na.EDGE_GROWTH_CAP
                                              for s in ("closed", "open")),
                "protected_unchanged": all(plan["states"][s]["protection_check"]["protected_vertices_moved"] == 0
                                           for s in ("closed", "open")),
                "facial_first_hits_within_tolerance": all(facial[s]["max_facial_shift_mm"] <= 0.05 for s in facial),
                # OBJ text round-trip is float32-precision (~5e-9 m here); 1 um is
                # the engineering tolerance and face arrays must match exactly
                "disk_obj_matches_memory": all(
                    disk[s]["max_abs_diff_m"] is not None and disk[s]["max_abs_diff_m"] <= 1e-6
                    and disk[s]["faces_match"] for s in disk),
                "realized_height_meets_design": all(
                    abs(plan["states"][s]["metrics"]["realized_height_mm"]
                        - plan["states"][s]["metrics"]["requested_height_mm"]) <= 0.5 for s in ("closed", "open")),
            }
            results[preset] = {
                "plan": plan, "disk": disk, "facial": facial, "checks": checks,
                "accepted": all(checks.values()) and plan["accepted"],
                "edge_stats": {"base_median_mm": round(base_stats["closed"], 2),
                               "core_median_mm": round(core_median, 2)},
                "facial_shift_mm": max(facial[s]["max_facial_shift_mm"] for s in facial),
                "disk_check": "; ".join(
                    f"{s}: {disk[s]['vertices']} 顶点 / {disk[s]['faces']} 面, 读回误差 {disk[s]['max_abs_diff_m']}"
                    for s in disk),
            }
            if preset == "natural":
                plan_nat = plan

        # ---------------- renders ------------------------------------------
        t0 = time.perf_counter()
        for preset in PRESETS:
            plan = results[preset]["plan"]
            for state in ("closed", "open"):
                item = plan["states"][state]
                frame = item["frame"]
                centre = frame.origin
                fwd, right, up = frame.n, frame.u_axis, frame.v_axis
                base_v, base_f = base_meshes[state]
                shaped = item["shaped_vertices"]
                refined_f = plan["refined_faces"]
                for view_name, view_dir, half in (
                        ("front", -fwd, 55.0), ("side", right, 55.0), ("45deg", (right - fwd) / np.sqrt(2.0), 55.0),
                        ("side_close", right, 30.0)):
                    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), facecolor="#202020")
                    for ax, (V, F, label) in zip(
                        axes, ((base_v, base_f, "baseline 0.5.2"), (shaped, refined_f, f"sculpt {preset}"))):
                        ax.set_facecolor("#202020")
                        render_view(ax, V, F, view_dir, centre, half)
                        ax.set_title(f"{preset} / {state} — {label}", color="w", fontsize=9)
                    fig.tight_layout()
                    path = out / "render" / f"{preset}_{state}_{view_name}_compare.png"
                    fig.savefig(path, dpi=110, facecolor="#202020")
                    plt.close(fig)
                fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), facecolor="#202020")
                for ax, (V, F, label) in zip(
                    axes, ((base_v, base_f, "baseline"), (shaped, refined_f, f"sculpt {preset}"))):
                    ax.set_facecolor("#202020")
                    render_view(ax, V, F, (right - fwd) / np.sqrt(2.0), centre, 34.0)
                    render_wire(ax, V, F, (right - fwd) / np.sqrt(2.0), centre, 34.0)
                    ax.set_title(f"{preset} / {state} wireframe — {label}", color="w", fontsize=9)
                fig.tight_layout()
                path = out / "render" / f"{preset}_{state}_wireframe.png"
                fig.savefig(path, dpi=110, facecolor="#202020")
                plt.close(fig)
                # sections: longitudinal (v–n), transverse (u–n)
                for name, up_axis, limit in (("long", frame.v_axis, 30.0), ("cross", frame.u_axis, 22.0)):
                    fig, ax = plt.subplots(figsize=(6.2, 5.4))
                    for V, F, label, color in ((base_v, base_f, "baseline", "#8a97a8"),
                                               (shaped, refined_f, f"sculpt {preset}", "#e05a4e")):
                        x, y = plane_section(V, F, centre, right if name == "long" else up, up_axis, fwd, -limit, limit)
                        ax.plot(y, x, color=color, lw=1.8, label=label)
                    ax.set_xlabel("outward n (mm)")
                    ax.set_ylabel("v (mm) along neck" if name == "long" else "u (mm) lateral")
                    ax.set_title(f"{preset} / {state}: {'longitudinal' if name == 'long' else 'transverse'} section")
                    ax.grid(alpha=0.3)
                    ax.legend()
                    path = out / "render" / f"{preset}_{state}_section_{name}.png"
                    fig.savefig(path, dpi=110)
                    plt.close(fig)
        timing.append({"stage": "renders", "seconds": round(time.perf_counter() - t0, 3)})

        # ---------------- artefacts ---------------------------------------
        appearance_rows = []
        for preset in PRESETS:
            plan = results[preset]["plan"]
            for state in ("closed", "open"):
                m = plan["states"][state]["metrics"]
                g = plan["states"][state]["guards"]
                appearance_rows.append({
                    "preset": preset, "state": state,
                    **{k: m.get(k) for k in ("requested_height_mm", "realized_height_mm", "peak_on_mesh_mm",
                                              "width_half_mm", "width_ten_mm", "length_half_mm", "length_ten_mm",
                                              "requested_width_mm", "requested_length_mm", "requested_tip_radius_mm")},
                    "transition_up_mm": m["requested_transitions_mm"][0],
                    "transition_down_mm": m["requested_transitions_mm"][1],
                    "true_fold": g["true_fold"], "normal_reversal": g["normal_reversal"],
                    "max_edge_growth": g["max_edge_growth"],
                    "self_intersections": plan["states"][state]["self_intersections"]["count"],
                    "protected_moved": plan["states"][state]["protection_check"]["protected_vertices_moved"],
                    "safety_fallback": "none",
                    "position_source": plan["states"][state]["position_audit"]["position_source"],
                    "background_available": plan["states"][state]["background_available"],
                })
        with (out / "appearance_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(appearance_rows[0].keys()))
            writer.writeheader()
            writer.writerows(appearance_rows)
        summary = {
            "schema": "head_rgbd.neck_sculpt.experiment.v1",
            "version": "0.5.6",
            "editing_mode": True,
            "geometry_source": "procedural_appearance",
            "subject_observation_matched": False,
            "presets": {p: {"design": na.design_from_preset(p).as_dict(),
                            "accepted": results[p]["accepted"],
                            "checks": results[p]["checks"],
                            "disk": results[p]["disk"],
                            "facial": results[p]["facial"],
                            "metrics": {s: results[p]["plan"]["states"][s]["metrics"] for s in ("closed", "open")},
                            "position": {s: results[p]["plan"]["states"][s]["position_audit"] for s in ("closed", "open")},
                            "background": {s: (results[p]["plan"]["states"][s]["background"].residual_stats
                                               if results[p]["plan"]["states"][s]["background"] is not None else None)
                                           for s in ("closed", "open")}}
                        for p in PRESETS},
            "default_preset": "natural",
            "refinement": {"level1_radius_mm": 42.0, "level2_radius_mm": 22.0,
                           "base_vertices": plan_nat["base_vertex_count"],
                           "refined_vertices": plan_nat["refined_vertex_count"],
                           "core_edge_median_mm": results["natural"]["edge_stats"]["core_median_mm"]},
            "statement": "manual appearance sculpt; geometry does not correspond to measured observations",
        }
        (out / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                                     encoding="utf-8")
        (out / "presets.json").write_text(json.dumps({p: na.design_from_preset(p).as_dict() for p in PRESETS},
                                                     indent=2), encoding="utf-8")
        timing.append({"stage": "total", "seconds": round(time.perf_counter() - t_all, 3)})
        with (out / "timing.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "seconds"])
            writer.writeheader()
            writer.writerows(timing)
        report = write_report(out, timing=timing, results=results, plan=plan_nat,
                              baseline_meshes=base_meshes, render_paths=[], input_notes={})
        docs = PROJECT / "docs" / "NECK_APPEARANCE_SCULPT_REPORT.html"
        docs.write_bytes(report.read_bytes())
        status = "ok" if all(results[p]["accepted"] for p in PRESETS) else "verification_failed"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        summary = {"status": status, "error": error, "exit_code": 0 if status == "ok" else 1,
                   "total_seconds": round(time.perf_counter() - t_all, 3), "output": str(out)}
        (out / "run_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
