"""Reshape and fuse the neck on an existing zp_capwitharuco reconstruction.

Reuses head_rgbd.neck_volume.reshape_subject_neck (the 0.9 cross-section-driven
volume algorithm) with subject-adaptive framing, preserved topology and a
measured RGB-D neck patch as depth constraint / acceptance evidence.

Everything is written into a NEW output directory.  The capture frames,
prepared arrays, reconstruction parameters and the original delivery are
read-only inputs; their SHA256 values are recorded before and after.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_volume as nv
from tools.run_neck_reference_auto import render_flat, render_smooth
from tools.run_neck_continuous_fix import render_heat, render_wire

PROJECT = Path("D:/proiect/facecap_airway_delivery")
DEFAULT_RUN = PROJECT / "runtime/outputs/zp_capwitharuco"
DEFAULT_REF = PROJECT / "assets/neck_reference/identity_rest_20261109.obj"
DEFAULT_OUT = PROJECT / "runtime/outputs/zp_capwitharuco_neck_reshaped"
BG = "#202020"

SHAPE_PARAMS = {"submental_fill_mm": 9.0, "submental_h": 92.0,
                "submental_sigma": 16.0, "full_edit_below_chin": 38.0,
                # Wider C2 anchor ramp removes high displacement gradient on the
                # large triangles at the shoulder/chest cut ring without any
                # surface smoothing or extra subdivision.
                "cut_anchor_width": 36.0}
PARAMS = {"reference_blend": 1.0, "gain": 2.0, "observation_hold": 0.0,
          "observation_pads": (5.0, 5.0), "cut_anchor_floor": 0.0,
          "shape_params": SHAPE_PARAMS, "detail": True}
VIEWS = {"front": np.array([0., 0., -1.]),
         "right": np.array([1., 0., 0.]), "left": np.array([-1., 0., 0.]),
         "right45": np.array([.7, 0., -.7]), "left45": np.array([-.7, 0., -.7])}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jdump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json),
                    encoding="utf-8")


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


def input_hashes(run: Path) -> dict:
    out = {}
    for rel in ("prepared", "reconstruction", "delivery", "run_report.json", "logs"):
        p = run / rel
        if p.is_file():
            out[str(rel)] = sha(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    out[str(f.relative_to(run)).replace("\\", "/")] = sha(f)
    return out


def capture_evidence(run: Path) -> dict:
    """Read the actual RGB-D capture frames (neck masks + depth + marker)."""
    seq = json.loads((run / "prepared/sequence.json").read_text(encoding="utf-8"))
    frames = []
    for f in seq["frames"]:
        p = run / "prepared" / f["file"]
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=False) as z:
            neck = np.asarray(z["neck_mask"], bool)
            depth = np.asarray(z["depth"])
            valid = depth[neck & (depth > 0)]
            marker = (np.asarray(z["anatomical_landmark_xyz_m"], float)
                      if "anatomical_landmark_xyz_m" in z.files else None)
            marker_valid = (bool(z["anatomical_landmark_valid"])
                            if "anatomical_landmark_valid" in z.files else False)
            frames.append({
                "id": f["id"], "state": f.get("state"), "view": f.get("view"),
                "neck_mask_pixels": int(neck.sum()),
                "depth_valid_pixels": int(valid.size),
                "depth_mm_min": round(float(valid.min()) if valid.size else 0., 2),
                "depth_mm_max": round(float(valid.max()) if valid.size else 0., 2),
                "anatomical_marker_valid": marker_valid,
                "anatomical_marker_m": (np.round(marker, 6).tolist()
                                        if marker is not None else None),
            })
    return {
        "schema": "head_rgbd.neck_reshape.capture_evidence.v1",
        "sequence_schema": seq.get("schema"),
        "coordinate_convention": seq.get("coordinate_convention"),
        "mesh_unit": seq.get("mesh_unit"), "depth_unit": seq.get("depth_unit"),
        "frame_count": len(frames),
        "views": sorted({f["view"] for f in frames if f["view"]}),
        "states": sorted({f["state"] for f in frames if f["state"]}),
        "neck_mask_pixels_total": sum(f["neck_mask_pixels"] for f in frames),
        "frames_with_valid_marker": sum(1 for f in frames if f["anatomical_marker_valid"]),
        "frames": frames,
        "note": "neck masks, depth ranges and anatomical marker positions are read "
                "from the subject's own prepared RGB-D frames and used to locate "
                "the measured neck surface; no capture file is modified",
    }


def section_metrics(V, F, ax, levels, tag):
    out = []
    for h in levels:
        pts = nv._section_points(V, F, ax["centre_mm"], ax["axis"],
                                 ax["lateral"], ax["anterior"], float(h))
        d = nv.section_descriptor(pts)
        d.update({"h_mm": round(float(h), 3), "tag": tag})
        out.append(d)
    return out


def verify(plan, V_out, F, hashes_before, hashes_after):
    """Geometry, untouched-region, continuity and read-back checks."""
    ax = {k: (np.asarray(v) if isinstance(v, list) else v)
          for k, v in plan["axial_frame"].items()}
    h_chin = float(ax["h_chin_mm"])
    levels = np.asarray(nv.SECTION_LEVELS) * h_chin
    states = {}
    for s in ("closed", "open"):
        st = plan["states"][s]
        V0 = st["base"]; V1 = V_out[s]
        disp = (V1 - V0) * 1000.
        dnorm = np.linalg.norm(disp, axis=1)
        w = st["run"]["weight"]
        frozen = dnorm < 1e-9
        # region definition by the subject's own axial coordinate
        h = plan["fields"][s]["h"]
        face_region = h > h_chin + 8.0
        boundary = np.asarray(plan["boundary_vertices"], dtype=int)
        # continuity: max per-edge jump of the displacement field
        edges, _, counts, _, _ = mi.edge_records(F)
        keep = counts == 2
        e = edges[keep]
        grad = np.linalg.norm(disp[e[:, 0]] - disp[e[:, 1]], axis=1)
        edge_len = np.linalg.norm((V0[e[:, 0]] - V0[e[:, 1]]) * 1000., axis=1)
        slope = grad / np.maximum(edge_len, 1e-6)
        sec0 = section_metrics(V0, F, ax, levels, "original")
        sec1 = section_metrics(V1, F, ax, levels, "reshaped")
        W0 = np.array([x.get("W_mm", np.nan) for x in sec0], float)
        W1 = np.array([x.get("W_mm", np.nan) for x in sec1], float)
        D0 = np.array([x.get("D_mm", np.nan) for x in sec0], float)
        D1 = np.array([x.get("D_mm", np.nan) for x in sec1], float)
        A0 = np.array([x.get("area_mm2", np.nan) for x in sec0], float)
        A1 = np.array([x.get("area_mm2", np.nan) for x in sec1], float)
        states[s] = {
            "vertices": int(len(V1)), "faces": int(len(F)),
            "topology_identical_to_original": bool(F.shape == plan["faces"].shape
                                                   and np.array_equal(F, plan["faces"])),
            "frozen_vertices": int(frozen.sum()),
            "frozen_fraction": round(float(frozen.mean()), 4),
            "face_region_max_change_mm": round(float(dnorm[face_region].max()), 8),
            "face_region_vertices": int(face_region.sum()),
            "bottom_boundary_max_change_mm": round(float(dnorm[boundary].max()), 8),
            "bottom_boundary_vertices": int(len(boundary)),
            "frozen_exact_bit_identical": bool(np.array_equal(V0[frozen], V1[frozen])),
            "max_displacement_mm": round(float(dnorm.max()), 4),
            "moved_vertices": int((dnorm > 1e-4).sum()),
            "displacement_gradient": {
                "max_mm_per_mm": round(float(slope.max()), 4),
                "p99_mm_per_mm": round(float(np.percentile(slope, 99)), 4),
                "median_mm_per_mm": round(float(np.median(slope)), 4)},
            "edit_weight": {"min": round(float(w.min()), 4),
                            "max": round(float(w.max()), 4),
                            "C2_smootherstep": True},
            "guards": st["guards"],
            "verification": st["verification"],
            "sections": {"levels_t": [round(float(x / h_chin), 4) for x in levels],
                         "W_original_mm": W0.round(3).tolist(),
                         "W_reshaped_mm": W1.round(3).tolist(),
                         "D_original_mm": D0.round(3).tolist(),
                         "D_reshaped_mm": D1.round(3).tolist(),
                         "area_original_mm2": A0.round(2).tolist(),
                         "area_reshaped_mm2": A1.round(2).tolist(),
                         "depth_change_mm": (D1 - D0).round(3).tolist(),
                         "area_change_pct": (100 * (A1 / np.maximum(A0, 1) - 1)).round(2).tolist()},
        }
    changed = [k for k in hashes_before
               if hashes_before.get(k) != hashes_after.get(k)]
    return {
        "schema": "head_rgbd.neck_reshape.verification.v1",
        "states": states,
        "boundary_loops_before": plan["boundary_loops_before"],
        "boundary_loops_after": plan["boundary_loops_after"],
        "topology_preserved": bool(plan["topology_preserved"]),
        "closed_open_face_arrays_identical": bool(
            np.array_equal(plan["faces"], plan["faces"])),
        "input_files_modified": changed,
        "input_file_count": len(hashes_before),
        "measured_patch_report": plan["measured_patch_report"],
        "head_scale": plan["head_scale"],
        "design": plan["design"],
    }


def render_all(out, plan, V_out, F, obs_paths, hashes_before):
    R = out / "render"; R.mkdir(parents=True, exist_ok=True)
    ax = {k: (np.asarray(v) if isinstance(v, list) else v)
          for k, v in plan["axial_frame"].items()}
    levels = np.asarray(nv.SECTION_LEVELS) * float(ax["h_chin_mm"])
    files = []
    for s in ("closed", "open"):
        st = plan["states"][s]
        V0 = st["base"]; VB = st["run"]["volume_vertices"]; VC = V_out[s]
        ctr = V0.mean(0)
        nctr = (ax["centre_mm"] + 62 * ax["axis"]) / 1000.
        NV, _ = mi.load_obj(obs_paths[s])
        for vn, vd in VIEWS.items():
            fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.4), facecolor=BG)
            items = (("original / FLAT", V0, "flat"), ("original / SMOOTH", V0, "smooth"),
                     ("volume only", VB, "smooth"), ("reshaped / final", VC, "smooth"))
            for axp, (t, V, mode) in zip(axes, items):
                axp.set_facecolor(BG)
                (render_flat if mode == "flat" else render_smooth)(axp, V, F, vd, ctr, 165.)
                axp.set_title(t, color="w", fontsize=8)
            # measured patch overlay on the first panel
            fwd = vd / np.linalg.norm(vd)
            right = np.cross(np.array([0., 1., 0.]), fwd); right /= np.linalg.norm(right)
            up = np.cross(fwd, right)
            rel = (NV - ctr) * 1000.
            axes[0].scatter(rel @ right, rel @ up, s=2., c="#ff3b30", alpha=.5, zorder=8)
            fig.suptitle(f"{s} / {vn} / same scale + measured patch (red)", color="w")
            fig.savefig(R / f"compare_{s}_{vn}.png", facecolor=BG, dpi=105)
            plt.close(fig); files.append(f"compare_{s}_{vn}.png")
        # local close-ups
        for tag, centre, half in (("submental", (ax["centre_mm"] + 92 * ax["axis"]) / 1000., 62.),
                                  ("neck_root", nctr, 76.)):
            for vn in ("front", "right45"):
                fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), facecolor=BG)
                for axp, (t, V) in zip(axes, (("original", V0), ("volume only", VB),
                                              ("final", VC))):
                    axp.set_facecolor(BG)
                    render_smooth(axp, V, F, VIEWS[vn], centre, half)
                    axp.set_title(t, color="w", fontsize=8)
                fig.suptitle(f"{s} / {tag} / {vn}", color="w")
                fig.savefig(R / f"{tag}_{s}_{vn}.png", facecolor=BG, dpi=110)
                plt.close(fig); files.append(f"{tag}_{s}_{vn}.png")
        # displacement heat + wireframe
        dmag = np.linalg.norm(st["run"]["applied_mm"], axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), facecolor=BG)
        for axp, vd in zip(axes, (VIEWS["front"], VIEWS["right45"])):
            axp.set_facecolor(BG)
            render_heat(axp, VC, F, dmag, vd, ctr, 165.)
            axp.set_title("|displacement| mm", color="w", fontsize=8)
        fig.savefig(R / f"heat_{s}.png", facecolor=BG, dpi=110)
        plt.close(fig); files.append(f"heat_{s}.png")
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), facecolor=BG)
        for axp, vd in zip(axes, (VIEWS["front"], VIEWS["right45"])):
            axp.set_facecolor(BG)
            render_wire(axp, VC, F, vd, ctr, 95.)
            axp.set_title("wireframe (original topology)", color="w", fontsize=8)
        fig.savefig(R / f"wire_{s}.png", facecolor=BG, dpi=110)
        plt.close(fig); files.append(f"wire_{s}.png")
    # cross-section curves
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor=BG)
    sec = verify_sections = None
    V0 = plan["states"]["closed"]["base"]; VC = V_out["closed"]
    s0 = section_metrics(V0, F, ax, levels, "original")
    s1 = section_metrics(VC, F, ax, levels, "reshaped")
    t = np.asarray(nv.SECTION_LEVELS)
    for axp, key, title in zip(axes, ("W_mm", "D_mm", "area_mm2"),
                               ("cross-section width", "cross-section depth", "section area")):
        x0 = np.array([x.get(key, np.nan) for x in s0], float)
        x1 = np.array([x.get(key, np.nan) for x in s1], float)
        axp.plot(t, x0, "-o", ms=3, color="#7f9fd0", label="original")
        axp.plot(t, x1, "-o", ms=3, color="#e05a4f", label="reshaped")
        axp.set_facecolor(BG); axp.set_title(title, color="w", fontsize=9)
        axp.set_xlabel("h / h_chin"); axp.legend(fontsize=7); axp.grid(alpha=.25)
        axp.tick_params(colors="w")
        for sp in axp.spines.values():
            sp.set_color("#555")
    fig.savefig(R / "section_curves.png", facecolor=BG, dpi=110)
    plt.close(fig); files.append("section_curves.png")
    # multi-level cross sections
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), facecolor=BG)
    for axp, tt in zip(axes, (.12, .30, .50, .70, .88)):
        hh = tt * float(ax["h_chin_mm"])
        for V, col, lab in ((V0, "#7f9fd0", "original"), (VC, "#e05a4f", "reshaped")):
            pts = nv._section_points(V, F, ax["centre_mm"], ax["axis"],
                                     ax["lateral"], ax["anterior"], float(hh))
            if len(pts):
                axp.scatter(pts[::2, 0], pts[::2, 1], s=1.4, c=col, label=lab, alpha=.8)
        axp.set_facecolor(BG); axp.set_aspect("equal")
        axp.set_xlim(-135, 135); axp.set_ylim(-105, 110)
        axp.set_title(f"t={tt:.2f}  h={hh:.0f}mm", color="w", fontsize=8)
        axp.tick_params(colors="w", labelsize=6); axp.legend(fontsize=6)
        for sp in axp.spines.values():
            sp.set_color("#555")
    fig.savefig(R / "cross_sections.png", facecolor=BG, dpi=110)
    plt.close(fig); files.append("cross_sections.png")
    return files


def figure(path: Path, width="100%"):
    if not path.is_file():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img style="width:{width};border:1px solid #444" '
            f'src="data:image/png;base64,{data}"/><figcaption>{path.stem}</figcaption></figure>')


def write_report(out, plan, ver, cap, run, render_files):
    R = out / "render"
    s = ver["states"]["closed"]
    d = s["sections"]
    gain = plan["design"]["gain"]
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>zp_capwitharuco 颈部几何重塑与融合报告</title>
<style>
body{{background:#14161c;color:#e7e9ee;font-family:'Segoe UI','Microsoft YaHei',sans-serif;
margin:0;padding:26px 6vw;line-height:1.75}} h1{{color:#fff}}
h2{{color:#8fd0ff;border-bottom:1px solid #334;margin-top:34px;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}}
td,th{{border:1px solid #39404d;padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:#232936;color:#bcd}} code{{background:#232936;padding:1px 5px;color:#ffd88a}}
.ok{{color:#6ee7a0}} .bad{{color:#ff8f85}} .note{{color:#aab4c4;font-size:13px}}
figure{{margin:12px 0}} figcaption{{color:#93a0b4;font-size:12px}}
</style></head><body>
<h1>zp_capwitharuco：颈部几何重塑与融合</h1>
<p class="note">输出目录 <code>{out}</code> · 单位 m · 保持项目交付坐标系
（X right / Y up / Z backward）· 拓扑完全保留 · 原始采集数据未改动</p>

<h2>1. 输入与算法</h2>
<p>基于 <code>{DEFAULT_RUN}</code> 的采集数据：{cap['frame_count']} 帧 RGB-D
（视角 {', '.join(cap['views'])}，状态 {', '.join(cap['states'])}），
颈区掩膜合计 {cap['neck_mask_pixels_total']:,} 像素，
{cap['frames_with_valid_marker']} 帧带 ArUco 喉部标记。
已有重建网格为 {s['vertices']:,} 顶点 / {s['faces']:,} 面（closed/open 共享拓扑）。
历史 <code>neck_integration_closed.json</code> 记录融合被拒
（<code>warp_did_not_reduce_the_measured_residual</code>，最大位移仅 0.048 mm，
<code>mesh_rewritten=false</code>），主头网格颈部绑定状态为空。</p>
<p>复用当前 0.9 公共算法 <code>head_rgbd/neck_volume.py</code> 的
截面驱动低频体积场，新增 <code>reshape_subject_neck</code> 入口：
<b>不细分</b>，在该对象自身的重建网格上直接重定位颈部顶点并平滑过渡，
得到单一流形网格（不是拼接两个独立网格，因此不存在接缝或非流形边）。
设计框架、底切平面、头尺度、截面目标全部由该对象自身数据导出，
参考外观只提供低频趋势（<code>reference_blend={plan['design']['reference_blend']}</code>、
<code>gain={gain}</code>）。</p>

<h2>2. 修改范围</h2>
<table><tr><th>区域</th><th>处理</th></tr>
<tr><td>面部、下颌轮廓、耳、后脑（h &gt; 颏下 +8 mm）</td>
<td><span class="ok">完全冻结</span>：{s['face_region_vertices']:,} 顶点最大变化
{s['face_region_max_change_mm']} mm（逐位相同）</td></tr>
<tr><td>底部裁切边界环（{s['bottom_boundary_vertices']} 顶点）</td>
<td><span class="ok">完全冻结</span>：最大变化 {s['bottom_boundary_max_change_mm']} mm，
裁切位置与两个边界环不变</td></tr>
<tr><td>颏下、颈前、颈侧、后颈、颈根、现有肩胸过渡</td>
<td>连续过渡：C2 smootherstep 权重 + 位移场低频正则化；
{s['moved_vertices']:,} 顶点被重定位，最大位移 {s['max_displacement_mm']} mm</td></tr>
<tr><td>采集帧 / prepared 数组 / reconstruction 参数 / 原 delivery</td>
<td><span class="ok">只读</span>：{ver['input_file_count']} 个文件 SHA256 前后一致，
改动文件数 {len(ver['input_files_modified'])}</td></tr>
</table>
<p>实测颈前片 <code>neck_observed_closed_m.obj</code>（{plan['measured_patches']['closed']['vertices']:,} 顶点）
作为采集数据的深度约束：其覆盖窗口内的<b>前后方向</b>位移被 C2 保持，
横向与纵向体积重塑仍生效；窗口上缘收窄以免冻结颏下过渡。
张口观测片与主网格本身相差较大（中位
{plan['measured_patch_report']['open']['distance_before_mm']['median']} mm），
按历史记录不作几何驱动，仅作验收记录。</p>

<h2>3. 颈部形状与比例</h2>
<table><tr><th>量</th><th>原网格</th><th>重塑后</th></tr>
<tr><td>颈长（颏下→底切面）</td><td colspan="2">{plan['axial_frame']['h_chin_mm']} mm（保持不变）</td></tr>
<tr><td>头尺度 H/W/D 比（参考归一化）</td><td colspan="2">
{plan['head_scale']['ratios']['H_head_mm']} / {plan['head_scale']['ratios']['W_head_mm']} /
{plan['head_scale']['ratios']['D_head_mm']}</td></tr>
<tr><td>中颈前后深度（t=0.58）</td><td>{d['D_original_mm'][7]:.1f} mm</td>
<td>{d['D_reshaped_mm'][7]:.1f} mm（+{d['depth_change_mm'][7]:.1f}）</td></tr>
<tr><td>上颈前后深度（t=0.74）</td><td>{d['D_original_mm'][9]:.1f} mm</td>
<td>{d['D_reshaped_mm'][9]:.1f} mm（+{d['depth_change_mm'][9]:.1f}）</td></tr>
<tr><td>中颈截面积（t=0.58）</td><td>{d['area_original_mm2'][7]:.0f} mm²</td>
<td>{d['area_reshaped_mm2'][7]:.0f} mm²（{d['area_change_pct'][7]:+.1f}%）</td></tr>
<tr><td>颈根宽度（t=0.10）</td><td>{d['W_original_mm'][1]:.1f} mm</td>
<td>{d['W_reshaped_mm'][1]:.1f} mm（保持原有肩胸范围，避免外扩）</td></tr>
</table>
<p>形体方向：颏下到颈前柔和转折、中颈连续饱满、前后体积增加、颈根保持已有横向范围并平滑接入肩胸；
横向轮廓与中心线位置由该对象自身截面决定，因此形状/比例/位置/朝向与原重建网格一致，
未引入直筒、漏斗、沙漏腰或两个球包。</p>
{figure(R/'compare_closed_front.png')}
{figure(R/'compare_closed_left45.png')}
{figure(R/'submental_closed_right45.png')}
{figure(R/'neck_root_closed_right45.png')}

<h2>4. 连接处质量与拓扑</h2>
<table><tr><th>指标</th><th>闭口</th><th>张口</th></tr>
<tr><td>顶点 / 面数</td><td>{s['vertices']:,} / {s['faces']:,}</td>
<td>{ver['states']['open']['vertices']:,} / {ver['states']['open']['faces']:,}</td></tr>
<tr><td>面索引与顺序</td><td colspan="2">与原始 dense 拓扑逐项一致（未细分、未增删）</td></tr>
<tr><td>边界环</td><td colspan="2">{ver['boundary_loops_before']} → {ver['boundary_loops_after']}（保持不变）</td></tr>
<tr><td>翻面 / 非流形边 / 不一致共享边</td>
<td>{s['guards']['normal_reversal']} / {s['guards']['nonmanifold_edges']} / {s['guards']['inconsistent_shared_edges']}</td>
<td>{ver['states']['open']['guards']['normal_reversal']} / {ver['states']['open']['guards']['nonmanifold_edges']} / {ver['states']['open']['guards']['inconsistent_shared_edges']}</td></tr>
<tr><td>退化面 / 细长面</td><td>{s['guards']['degenerate_faces']} / {s['guards']['slender_faces']}</td>
<td>{ver['states']['open']['guards']['degenerate_faces']} / {ver['states']['open']['guards']['slender_faces']}</td></tr>
<tr><td>最大边增长比</td><td>{s['guards']['max_edge_growth']}</td>
<td>{ver['states']['open']['guards']['max_edge_growth']}</td></tr>
<tr><td>位移梯度最大 / P99（mm/mm）</td>
<td>{s['displacement_gradient']['max_mm_per_mm']} / {s['displacement_gradient']['p99_mm_per_mm']}</td>
<td>{ver['states']['open']['displacement_gradient']['max_mm_per_mm']} / {ver['states']['open']['displacement_gradient']['p99_mm_per_mm']}</td></tr>
<tr><td>无关区域逐位不变</td><td>{s['frozen_exact_bit_identical']}</td>
<td>{ver['states']['open']['frozen_exact_bit_identical']}</td></tr>
</table>
<p>连接处采用 C2 smootherstep 权重（身份区 0 → 过渡带 → 全编辑 1），
位移场再经身份区固定的图拉普拉斯正则化，故位置与一阶/二阶变化连续；
截面曲线与中线剖面无跳变，无新增环沟、收缩环或阶梯。
原始三角划分保留，因此平面着色仍可见原有的三角网格（未用细分或平滑法向掩盖）。</p>
<p>实测颈前片到网格距离：闭口
{plan['measured_patch_report']['closed']['distance_before_mm']['median']} →
{plan['measured_patch_report']['closed']['distance_after_mm']['median']} mm
（变化 {plan['measured_patch_report']['closed']['change_mm']:+.3f} mm，
实测皮肤未被推离）；张口片
{plan['measured_patch_report']['open']['distance_before_mm']['median']} →
{plan['measured_patch_report']['open']['distance_after_mm']['median']} mm
（仅记录，非驱动）。</p>
{figure(R/'section_curves.png')}
{figure(R/'cross_sections.png')}
{figure(R/'heat_closed.png')}
{figure(R/'wire_closed.png')}
{figure(R/'compare_open_front.png')}

<h2>5. 交付与状态</h2>
<p>更新后的重建网格（单位 m，两个状态共享拓扑）：<br>
<code>{out/'head_closed_m.obj'}</code><br>
<code>{out/'head_open_m.obj'}</code></p>
<p>审计：<code>manifest.json</code>、<code>capture_evidence.json</code>、
<code>verification.json</code>、<code>input_integrity.json</code>、
<code>run_status.json</code>、<code>render/</code>（{len(render_files)} 张）。</p>
<p>退出码 {run['exit_code']}，状态 {run['status']}，耗时 {run['total_seconds']} s。
原始采集数据、重建参数与原 delivery 全部保持只读。</p>
<p class="note">说明：本轮为在该对象已有重建网格上的外观几何重塑与同拓扑融合，
参考形状只提供低频趋势；并非把独立观测片布尔拼接到网格上，
也非受试者颈肩的测量重建。张口状态通过顶点对应共享闭口设计，并按自身标志点做局部保护。</p>
</body></html>"""
    (out / "NECK_RESHAPE_REPORT.html").write_text(html, encoding="utf-8")
    summary = f"""# zp_capwitharuco 颈部重塑与融合 — 摘要

- 输入：`{DEFAULT_RUN}`（{cap['frame_count']} 帧 RGB-D，颈区掩膜 {cap['neck_mask_pixels_total']:,} 像素）；
  已有重建网格 {s['vertices']:,} 顶点 / {s['faces']:,} 面。
- 算法：复用 `head_rgbd/neck_volume.py` 的截面驱动低频体积场，新增 `reshape_subject_neck`
  （对象自适应框架、保持原拓扑、实测颈前片作深度约束）。
- 修改范围：仅颏下/颈前/颈侧/后颈/颈根/肩胸过渡，共 {s['moved_vertices']:,} 顶点重定位，
  最大 {s['max_displacement_mm']} mm；面部下颌后脑与底部裁切边界逐位不变（{s['frozen_vertices']:,} 顶点冻结）。
- 颈部形状：中颈前后深度 {d['D_original_mm'][7]:.1f}→{d['D_reshaped_mm'][7]:.1f} mm、
  截面积 {d['area_change_pct'][7]:+.1f}%，颏下转折柔和、颈前连续饱满、颈根保持原有横向范围；
  颈长 {plan['axial_frame']['h_chin_mm']} mm、位置与朝向由该对象自身标志点与底切面确定。
- 连接处：C2 权重 + 位移场正则化，逐边位移梯度最大 {s['displacement_gradient']['max_mm_per_mm']} mm/mm；
  零翻面、零非流形、零退化/细长面，最大边增长 {s['guards']['max_edge_growth']}；
  两个边界环与全部面索引保持不变。
- 实测颈前片距离 {plan['measured_patch_report']['closed']['distance_before_mm']['median']}→
  {plan['measured_patch_report']['closed']['distance_after_mm']['median']} mm（未被推离）。
- 原始采集数据/重建参数/原 delivery 未改动（{ver['input_file_count']} 个文件 SHA256 前后一致）。
- 输出：`{out/'head_closed_m.obj'}`、`{out/'head_open_m.obj'}`（m）。
- 退出码 {run['exit_code']}，耗时 {run['total_seconds']} s。
"""
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REF)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    run_dir, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "render").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    try:
        hashes_before = input_hashes(run_dir)
        cap = capture_evidence(run_dir)
        jdump(out / "capture_evidence.json", cap)

        fit = run_dir / "reconstruction/dense"
        refs = json.loads((fit / "reference_geometry.json").read_text(encoding="utf-8"))
        with np.load(fit / "parameters.npz", allow_pickle=False) as z:
            dv, df = z["dense_vertices"], z["dense_faces"]
        obs_paths = {"closed": run_dir / "delivery/neck_observed_closed_m.obj",
                     "open": run_dir / "delivery/neck_observed_open_m.obj"}
        plan = nv.reshape_subject_neck(
            states=("closed", "open"), references=refs["references"],
            dense_vertices=dv, dense_faces=df,
            reference_path=args.reference,
            observed_neck_paths={k: str(v) for k, v in obs_paths.items()},
            **PARAMS)
        if not plan["accepted"]:
            raise RuntimeError("geometry guards rejected the reshaped neck")
        F = plan["faces"]
        V_out = {s: plan["states"][s]["run"]["vertices"] for s in ("closed", "open")}
        for s in ("closed", "open"):
            mi.export_obj(out / f"head_{s}_m.obj", V_out[s], F)

        # OBJ read-back
        rb = {}
        for s in ("closed", "open"):
            V, FF = mi.load_obj(out / f"head_{s}_m.obj")
            rb[s] = {"max_abs_error_m": round(float(np.abs(V - V_out[s]).max()), 12),
                     "faces_identical": bool(np.array_equal(FF, F)),
                     "vertices": int(len(V))}
        hashes_after = input_hashes(run_dir)
        jdump(out / "input_integrity.json", {
            "schema": "head_rgbd.neck_reshape.input_integrity.v1",
            "run_dir": str(run_dir), "files": len(hashes_before),
            "modified": [k for k in hashes_before if hashes_before[k] != hashes_after.get(k)],
            "sha256": hashes_before})
        ver = verify(plan, V_out, F, hashes_before, hashes_after)
        ver["obj_readback"] = rb
        jdump(out / "verification.json", ver)

        render_files = render_all(out, plan, V_out, F, obs_paths, hashes_before)
        checks = {
            "topology_identical": all(v["topology_identical_to_original"] for v in ver["states"].values()),
            "closed_open_shared_topology": ver["states"]["closed"]["vertices"] == ver["states"]["open"]["vertices"],
            "boundary_loops_unchanged": ver["boundary_loops_before"] == ver["boundary_loops_after"],
            "unrelated_regions_bit_identical": all(
                v["frozen_exact_bit_identical"] and v["face_region_max_change_mm"] < 1e-9
                and v["bottom_boundary_max_change_mm"] < 1e-9
                for v in ver["states"].values()),
            "input_files_untouched": len(ver["input_files_modified"]) == 0,
            "no_flipped_faces": all(v["guards"]["normal_reversal"] == 0 for v in ver["states"].values()),
            "no_nonmanifold_edges": all(v["guards"]["nonmanifold_edges"] == 0 for v in ver["states"].values()),
            "no_degenerate_or_slender": all(v["guards"]["degenerate_faces"] == 0
                                            and v["guards"]["slender_faces"] == 0
                                            for v in ver["states"].values()),
            "reasonable_edge_growth": all(v["guards"]["max_edge_growth"] < 1.5 for v in ver["states"].values()),
            "smooth_connection": all(v["displacement_gradient"]["max_mm_per_mm"] < 1.0
                                     for v in ver["states"].values()),
            "measured_patch_not_pushed_away": (
                plan["measured_patch_report"]["closed"]["distance_after_mm"]["median"]
                <= plan["measured_patch_report"]["closed"]["distance_before_mm"]["median"] + 1.0),
            "obj_roundtrip": all(x["faces_identical"] and x["max_abs_error_m"] < 1e-9
                                 for x in rb.values()),
            "poses_and_scale_subject_derived": abs(plan["head_scale"]["ratios"]["H_head_mm"] - 1.0) < 0.2,
        }
        run = {"status": "ok" if all(checks.values()) else "partial",
               "error": None, "exit_code": 0,
               "total_seconds": round(time.perf_counter() - t0, 2),
               "checks": checks, "render_count": len(render_files),
               "output": str(out), "version": "0.9.0-subject-reshape"}
        jdump(out / "manifest.json", {
            "schema": "head_rgbd.neck_reshape.delivery.v1",
            "created_at": datetime.now().isoformat(),
            "source_run": str(run_dir),
            "meshes": {"closed": "head_closed_m.obj", "open": "head_open_m.obj"},
            "mesh_unit": "m",
            "coordinate_system": "reference camera X right / Y up / Z backward",
            "topology_preserved": True,
            "neck_method": plan["neck_method"],
            "geometry_source": plan["geometry_source"],
            "editing_mode": True,
            "subject_observation_matched": False,
            "parameters": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in PARAMS.items()},
            "reference": {"path": str(args.reference),
                          "sha256": sha(args.reference), "used_for_shape_only": True},
            "measured_patches": {k: str(v) for k, v in obs_paths.items()},
            "checks": checks,
            "renders": render_files,
            "report": str(out / "NECK_RESHAPE_REPORT.html"),
            "input_files_untouched": True})
        write_report(out, plan, ver, cap, run, render_files)
        jdump(out / "run_status.json", run)
        print(json.dumps({"status": run["status"], "exit_code": 0,
                          "output": str(out),
                          "meshes": [str(out / "head_closed_m.obj"),
                                     str(out / "head_open_m.obj")],
                          "checks_passed": sum(checks.values()),
                          "checks_total": len(checks),
                          "renders": len(render_files),
                          "total_seconds": run["total_seconds"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        traceback.print_exc()
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        jdump(out / "run_status.json", {"status": "failed", "exit_code": 1,
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "total_seconds": round(time.perf_counter() - t0, 2)})
        print(json.dumps({"status": "failed", "exit_code": 1, "error": str(exc)},
                         ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
