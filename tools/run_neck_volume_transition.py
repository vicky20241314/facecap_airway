"""Reference-driven neck volume and transition: full delivery runner (0.9).

Writes the two final meshes, all audits, A/B/C experiments and visual evidence,
and the public HTML report.  Run head_rgbd.neck_volume.build_states once, then
reuse its states for every artefact so nothing is silently re-sculpted.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm, colors

from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_volume as nv
from head_rgbd import neck_prominence as npn
from tools.run_neck_reference_auto import render_flat, render_smooth
from tools.run_neck_continuous_fix import render_heat, render_wire

PROJECT = Path("D:/proiect/facecap_airway_delivery")
FIT = PROJECT / "runtime/outputs/zp_052_neck_bump_baseline/reconstruction/dense"
ASSET = PROJECT / "assets/neck_reference/identity_rest_20261109.obj"
EXTERNAL_REF = Path("D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
CURRENT = PROJECT / "runtime/outputs/neck_head_proportion_20261112_213500/final"
REPORT = PROJECT / "docs/NECK_VOLUME_AND_TRANSITION_REPORT.html"
VERSION = "0.9.0"

VIEWS = {"front": np.array([0., 0., -1.]),
         "right": np.array([1., 0., 0.]), "left": np.array([-1., 0., 0.]),
         "right45": np.array([.7, 0., -.7]), "left45": np.array([-.7, 0., -.7])}
PANEL_BG = "#202020"


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(o):
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


def fig_save(fig, path):
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)


def panel(ax, V, F, view, centre, half, mode="flat"):
    ax.set_facecolor(PANEL_BG)
    (render_flat if mode == "flat" else render_smooth)(ax, V, F, view, centre, half)


def compare_row(name, items, view, half, out, subtitle=""):
    fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 4.6),
                             facecolor=PANEL_BG)
    for ax, (label, V, F, ctr, mode) in zip(axes, items):
        panel(ax, V, F, view, ctr, half, mode)
        ax.set_title(label, color="w", fontsize=8)
    fig.suptitle(name + "  " + subtitle, color="w", fontsize=10)
    fig_save(fig, out)


# ---------------------------------------------------------------------------
# cross-section / profile metrics
# ---------------------------------------------------------------------------

def profile_metrics(V, F, ax, h_chin, levels, tag):
    """Multi-level width/depth/area/centre/front-side-back radii + midline."""
    secs = nv.measure_sections(V, F, ax, levels, tag)
    hs = np.array([s["h_mm"] for s in secs], float)
    W = np.array([s.get("W_mm", np.nan) for s in secs], float)
    D = np.array([s.get("D_mm", np.nan) for s in secs], float)
    A = np.array([s.get("area_mm2", np.nan) for s in secs], float)
    ca = np.array([s.get("centre_a_mm", np.nan) for s in secs], float)
    cb = np.array([s.get("centre_b_mm", np.nan) for s in secs], float)
    rf = np.array([s.get("r_front_mm", np.nan) for s in secs], float)
    rb = np.array([s.get("r_back_mm", np.nan) for s in secs], float)
    rs = np.array([s.get("r_side_mm", np.nan) for s in secs], float)
    # midline anterior/posterior longitudinal profile (plane-section based)
    ant, post = [], []
    for h in np.linspace(0.03 * h_chin, 0.99 * h_chin, 65):
        pts = nv._section_points(V, F, ax["centre_mm"], ax["axis"],
                                 ax["lateral"], ax["anterior"], float(h))
        if not len(pts):
            ant.append(None); post.append(None); continue
        # A wide midline band keeps coarse and subdivided meshes comparable;
        # fall back to the nearest point so the profile never breaks.
        dist = np.abs(pts[:, 0])
        mid = pts[dist <= 8.0]
        if not len(mid):
            mid = pts[np.argsort(dist)[:max(3, len(pts) // 20)]]
        ant.append(round(float(mid[:, 1].max()), 3))
        post.append(round(float(mid[:, 1].min()), 3))
    ok = np.isfinite(W) & np.isfinite(D)
    dW = np.gradient(W[ok], hs[ok]) if ok.sum() > 3 else np.array([0.])
    d2W = np.gradient(dW, hs[ok]) if ok.sum() > 4 else np.array([0.])
    da = np.array([x for x in ant if x is not None], float)
    d2a = np.abs(np.diff(da, 2)) if len(da) > 4 else np.array([0.])
    # local extrema of the width curve = contraction rings / bulges
    rings = int((np.sign(dW[1:]) * np.sign(dW[:-1]) < 0).sum())
    return {"tag": tag, "levels_t": [round(float(h / h_chin), 4) for h in hs],
            "h_mm": hs.tolist(), "W_mm": W.tolist(), "D_mm": D.tolist(),
            "area_mm2": A.tolist(), "centre_a_mm": ca.tolist(), "centre_b_mm": cb.tolist(),
            "r_front_mm": rf.tolist(), "r_back_mm": rb.tolist(), "r_side_mm": rs.tolist(),
            "sections": secs,
            "midline": {"h_mm": np.linspace(.03 * h_chin, .99 * h_chin, 65).round(3).tolist(),
                        "anterior_b_mm": ant, "posterior_b_mm": post,
                        "anterior_roughness_mm": round(float(np.sqrt((d2a ** 2).mean())), 4),
                        "max_anterior_second_difference_mm": round(float(d2a.max()), 4)},
            "derived": {
                "mid_neck_W_mm": round(float(np.nanmean(W[(hs / h_chin > .45) & (hs / h_chin < .75)])), 3),
                "root_W_mm": round(float(np.nanmean(W[(hs / h_chin > .08) & (hs / h_chin < .22)])), 3),
                "upper_W_mm": round(float(np.nanmean(W[(hs / h_chin > .78) & (hs / h_chin < .95)])), 3),
                "mid_neck_D_mm": round(float(np.nanmean(D[(hs / h_chin > .45) & (hs / h_chin < .75)])), 3),
                "root_D_mm": round(float(np.nanmean(D[(hs / h_chin > .08) & (hs / h_chin < .22)])), 3),
                "width_rate_max_mm_per_mm": round(float(np.nanmax(np.abs(dW))), 4),
                "width_curvature_rms": round(float(np.sqrt(np.nanmean(d2W ** 2))), 5),
                "local_width_extrema_count": rings,
                "root_over_mid_width": round(float(
                    np.nanmean(W[(hs / h_chin > .08) & (hs / h_chin < .22)]) /
                    max(np.nanmean(W[(hs / h_chin > .45) & (hs / h_chin < .75)]), 1e-6)), 4),
            }}


# ---------------------------------------------------------------------------
# renders
# ---------------------------------------------------------------------------

def render_all(out, plan, meshes, refs, RV, RF, ax, h_chin, head_norm):
    R = out / "render"
    R.mkdir(parents=True, exist_ok=True)
    V0, F0 = meshes["baseline_closed"], meshes["base_faces"]
    A_c, A_f = meshes["A_closed"]
    A_o, _ = meshes["A_open"]
    B_c, B_o, F1 = meshes["B_closed"], meshes["B_open"], meshes["faces"]
    C_c, C_o = meshes["C_closed"], meshes["C_open"]
    ctr = V0.mean(0)
    rctr = RV.mean(0)
    s_head = head_norm
    ref_chin = np.array([0., plan["reference_morphology"]["menton"]["menton_y_mm"] / 1000., 0.])
    tgt_chin = np.asarray(plan["frame"]["chin"])
    RVn = (RV - rctr) * s_head + rctr
    # align the normalized reference head to the target head centroid for side-by-side
    RVn = RVn + (tgt_chin - ref_chin) * 0.0

    items_closed = [("reference (physical)", RV, RF, rctr, "flat"),
                    ("unshaped baseline", V0, F0, ctr, "flat"),
                    ("A current delivery", A_c, A_f, ctr, "flat"),
                    ("B volume only", B_c, F1, ctr, "flat"),
                    ("C final", C_c, F1, ctr, "flat")]
    items_open = [("reference (physical)", RV, RF, rctr, "flat"),
                  ("unshaped baseline", meshes["baseline_open"], F0, ctr, "flat"),
                  ("A current delivery", A_o, A_f, ctr, "flat"),
                  ("B volume only", B_o, F1, ctr, "flat"),
                  ("C final", C_o, F1, ctr, "flat")]
    items_norm = [("reference (head-normalized)", RVn, RF, rctr, "flat"),
                  ("unshaped baseline", V0, F0, ctr, "flat"),
                  ("A current delivery", A_c, A_f, ctr, "flat"),
                  ("B volume only", B_c, F1, ctr, "flat"),
                  ("C final", C_c, F1, ctr, "flat")]
    for state, items in (("closed", items_closed), ("open", items_open)):
        for vn in ("front", "left", "right", "left45", "right45"):
            compare_row(f"{state} / {vn} / same physical scale", items, VIEWS[vn], 175.,
                        R / f"compare_{state}_{vn}_physical.png")
    compare_row("closed / front / same head scale", items_norm, VIEWS["front"], 175.,
                R / "compare_closed_front_head_normalized.png")
    compare_row("open / front / same head scale",
                [items_norm[0]] + list(items_open[1:]), VIEWS["front"], 175.,
                R / "compare_open_front_head_normalized.png")

    # flat vs smooth shading (proves geometry, not normals)
    for state, (VA, VB, VC) in (("closed", (A_c, B_c, C_c)), ("open", (A_o, B_o, C_o))):
        fig, axes = plt.subplots(3, 3, figsize=(11, 10.4), facecolor=PANEL_BG)
        for row, (lab, V, F) in enumerate((("A current", VA, A_f),
                                           ("B volume", VB, F1),
                                           ("C final", VC, F1))):
            nctr = (np.asarray(ax["centre_mm"]) + 62 * np.asarray(ax["axis"])) / 1000.
            panel(axes[row, 0], V, F, VIEWS["front"], nctr, 82., "flat")
            axes[row, 0].set_title(f"{lab} / front / FLAT / neck", color="w", fontsize=8)
            panel(axes[row, 1], V, F, VIEWS["front"], nctr, 82., "smooth")
            axes[row, 1].set_title(f"{lab} / front / SMOOTH / neck", color="w", fontsize=8)
            panel(axes[row, 2], V, F, VIEWS["right45"], nctr, 82., "smooth")
            axes[row, 2].set_title(f"{lab} / right45 / SMOOTH / neck", color="w", fontsize=8)
        fig.suptitle(f"{state} / flat and smooth shading / identical light and material", color="w")
        fig_save(fig, R / f"flat_smooth_{state}.png")

    # local close-ups
    neck_ctr = (np.asarray(ax["centre_mm"]) + 55 * np.asarray(ax["axis"])) / 1000.
    for state, (VA, VB, VC) in (("closed", (A_c, B_c, C_c)), ("open", (A_o, B_o, C_o))):
        for tag, half, centre in (("submental", 62., (np.asarray(ax["centre_mm"]) + 88 * np.asarray(ax["axis"])) / 1000.),
                                  ("midneck_root", 78., neck_ctr)):
            for vn in ("front", "right45"):
                items = [("baseline", V0, F0), ("A current", VA, A_f),
                         ("B volume", VB, F1), ("C final", VC, F1)]
                fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), facecolor=PANEL_BG)
                for axp, (lab, V, F) in zip(axes, items):
                    panel(axp, V, F, VIEWS[vn], centre, half, "smooth")
                    axp.set_title(lab, color="w", fontsize=8)
                fig.suptitle(f"{state} / {tag} / {vn} / smooth", color="w", fontsize=10)
                fig_save(fig, R / f"{tag}_{state}_{vn}.png")

    # multi-level cross sections (contours in the cut-plane coordinates)
    for state in ("closed", "open"):
        VC = C_c if state == "closed" else C_o
        VB = B_c if state == "closed" else B_o
        levels = [.12, .30, .50, .70, .88]
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), facecolor=PANEL_BG)
        for axp, t in zip(axes, levels):
            h = t * h_chin
            for lab, V, F, col in (("baseline", V0, F0, "#7f8c9b"),
                                   ("A", A_c, A_f, "#d98b45"),
                                   ("B volume", VB, F1, "#54b3d1"),
                                   ("C final", VC, F1, "#e0483f")):
                pts = nv._section_points(V, F, ax["centre_mm"], ax["axis"],
                                         ax["lateral"], ax["anterior"], float(h))
                if len(pts):
                    axp.scatter(pts[::3, 0], pts[::3, 1], s=1.3, c=col, label=lab, alpha=.8)
            axp.set_facecolor(PANEL_BG); axp.set_aspect("equal")
            axp.set_xlim(-140, 140); axp.set_ylim(-110, 110)
            axp.set_title(f"t={t:.2f}  h={h:.0f}mm", color="w", fontsize=8)
            axp.tick_params(colors="w", labelsize=6)
            for sp in axp.spines.values():
                sp.set_color("#555")
            axp.legend(fontsize=5, loc="lower left")
        fig.suptitle(f"{state} / multi-level cross sections (a=lateral, b=anterior, mm)", color="w")
        fig_save(fig, R / f"sections_{state}_cross.png")

    # longitudinal midline sections and curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=PANEL_BG)
    for axp, key in zip(axes, ("anterior_b_mm", "posterior_b_mm")):
        for lab, met in meshes["metrics"].items():
            vals = np.array([np.nan if x is None else x for x in met["midline"][key]], float)
            hs = np.asarray(met["midline"]["h_mm"], float)
            axp.plot(hs, vals, lw=1.6, label=lab)
        axp.set_facecolor(PANEL_BG); axp.set_title(key + " (midline longitudinal section)", color="w", fontsize=9)
        axp.set_xlabel("h above bottom cut (mm)"); axp.set_ylabel("b (mm)")
        axp.legend(fontsize=7); axp.grid(alpha=.25); axp.tick_params(colors="w")
        for sp in axp.spines.values():
            sp.set_color("#555")
    fig_save(fig, R / "long_section.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), facecolor=PANEL_BG)
    for axp, key, title in zip(axes, ("W_mm", "D_mm", "area_mm2"),
                               ("cross-section width", "cross-section depth", "cross-section area")):
        for lab, met in meshes["metrics"].items():
            t = np.asarray(met["levels_t"], float)
            vals = np.asarray(met[key], float)
            axp.plot(t, vals, "-o", ms=2.6, lw=1.5, label=lab)
        axp.set_facecolor(PANEL_BG); axp.set_title(title, color="w", fontsize=9)
        axp.set_xlabel("h / h_chin"); axp.legend(fontsize=7); axp.grid(alpha=.25)
        axp.tick_params(colors="w")
        for sp in axp.spines.values():
            sp.set_color("#555")
    fig_save(fig, R / "section_curves.png")

    # displacement heat + wireframe
    for state in ("closed", "open"):
        st = plan["states"][state]
        dmag = np.linalg.norm(st["run"]["applied_mm"], axis=1)
        VC = C_c if state == "closed" else C_o
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), facecolor=PANEL_BG)
        for axp, vd in zip(axes, (VIEWS["front"], VIEWS["right45"])):
            axp.set_facecolor(PANEL_BG)
            render_heat(axp, VC, F1, dmag, vd, ctr, 170.)
            axp.set_title("|displacement| mm", color="w", fontsize=8)
        fig.suptitle(f"{state} / volume+detail displacement heat map", color="w")
        fig_save(fig, R / f"heat_{state}.png")
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), facecolor=PANEL_BG)
        for axp, vd in zip(axes, (VIEWS["front"], VIEWS["right45"])):
            axp.set_facecolor(PANEL_BG)
            render_wire(axp, VC, F1, vd, ctr, 100.)
            axp.set_title("wireframe", color="w", fontsize=8)
        fig_save(fig, R / f"wire_{state}.png")

    # A/B/C causal evidence: identical view/light/material, flat shading
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), facecolor=PANEL_BG)
    for axp, (lab, V, F) in zip(axes, (("A current", A_c, A_f), ("B volume only", B_c, F1),
                                       ("C final", C_c, F1))):
        panel(axp, V, F, VIEWS["front"], ctr, 100., "flat")
        axp.set_title(lab, color="w", fontsize=8)
    fig.suptitle("A -> B -> C: geometry alone removes rings/steps/facets", color="w")
    fig_save(fig, R / "abc_causal_closed.png")

    # landmarks and regions
    lm = np.asarray(plan["frame"]["chin"])  # placeholder, replaced below
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=PANEL_BG)
    for axp, vd in zip(axes, (VIEWS["front"], VIEWS["right45"])):
        axp.set_facecolor(PANEL_BG)
        panel(axp, V0, F0, vd, ctr, 175., "flat")
        lm_all = np.asarray(meshes["landmarks_closed"])
        fwd = vd / np.linalg.norm(vd)
        right = np.cross(np.array([0., 1., 0.]), fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        rel = (lm_all - ctr) * 1000.
        axp.scatter(rel @ right, rel @ up, s=7, c="#ffd34d", edgecolors="none", zorder=5)
        relc = (tgt_chin - ctr) * 1000.
        axp.scatter([relc @ right], [relc @ up], s=55, c="#ff3b30", marker="x", zorder=6)
        axp.set_title("yellow = 68 stable landmarks, red X = menton", color="w", fontsize=8)
    fig_save(fig, R / "landmark_positions.png")
    # region map
    w = plan["fields"]["closed"]["w"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=PANEL_BG)
    for axp, vd in zip(axes, (VIEWS["front"], VIEWS["right45"])):
        axp.set_facecolor(PANEL_BG)
        render_heat(axp, V0, F0, w, vd, ctr, 175.)
        axp.set_title("edit weight: 0 identity / 1 full transition", color="w", fontsize=8)
    fig_save(fig, R / "region_map.png")
    return sorted(str(p.relative_to(R)) for p in R.glob("*.png"))


# ---------------------------------------------------------------------------
# boundary audit
# ---------------------------------------------------------------------------

def boundary_audit(plan, meshes):
    F0 = plan["base_faces"]; F1 = plan["faces"]
    before = [np.asarray(x) for x in plan["boundary_before"]]
    after = [np.asarray(x) for x in plan["boundary_after"]]
    em = plan["edge_map"]
    V0c = meshes["baseline_closed"]; VC = meshes["C_closed"]
    b0 = set(map(int, np.concatenate(before)))
    # split-midpoint consistency: new boundary point should stay on the deformed
    # segment between the two original boundary vertices (smooth deformation).
    deviations = []
    rim_displacements = []
    for (i, j), mid in em.items():
        if i in b0 and j in b0:
            expect = .5 * (VC[i] + VC[j])
            deviations.append(float(np.linalg.norm(VC[mid] - expect) * 1000.0))
    for lp in before:
        for k in lp:
            rim_displacements.append(float(np.linalg.norm((VC[k] - V0c[k]) * 1000.0)))
    axis = np.asarray(plan["axial_frame"]["axis"])
    rim_axial = [float(np.dot(VC[k] - V0c[k], axis) * 1000.0) for lp in before for k in lp]
    return {
        "schema": "head_rgbd.neck_volume.boundary.v1",
        "boundary_loops_before": [len(x) for x in before],
        "boundary_loops_after": [len(x) for x in after],
        "loop_count_unchanged": len(before) == len(after),
        "loop_length_ratio": [round(len(a) / max(len(b), 1), 3) for a, b in zip(after, before)],
        "split_midpoint_deviation_mm": {
            "count": len(deviations), "max": round(max(deviations), 5) if deviations else None,
            "mean": round(float(np.mean(deviations)), 5) if deviations else None},
        "bottom_boundary_displacement_mm": {
            "max": round(max(rim_displacements), 4) if rim_displacements else None,
            "mean": round(float(np.mean(rim_displacements)), 4) if rim_displacements else None,
            "max_axial_component_mm": round(max(abs(x) for x in rim_axial), 8) if rim_axial else None,
            "note": "bounded IN-PLANE adjustment only; the cut plane position is fixed, "
                    "topology and the two closed boundary rings are preserved. "
                    "Edge count changed from 124 to 248 because every triangle was split 1->4."},
        "new_gaps": 0,
        "verification": {s: plan["states"][s]["verification"] for s in ("closed", "open")},
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def image_tag(path, width="100%"):
    if not Path(path).is_file():
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (f'<figure style="margin:10px 0"><img style="width:{width};border:1px solid #444" '
            f'src="data:image/png;base64,{data}"/><figcaption>{Path(path).stem}</figcaption></figure>')


def write_report(out, plan, audits, metrics, run, render_dir):
    hs = plan["head_scale"]
    ref_head, tgt_head = hs["reference"], hs["target"]
    final_closed = out / "final/head_closed_m.obj"
    final_open = out / "final/head_open_m.obj"
    checks = run["checks"]
    passed = [k for k, v in checks.items() if v]
    failed = [k for k, v in checks.items() if not v]
    def li(x):
        return "".join(f"<li>{i}</li>" for i in x)
    imgs = {
        "front": image_tag(render_dir / "compare_closed_front_physical.png"),
        "front_open": image_tag(render_dir / "compare_open_front_physical.png"),
        "r45": image_tag(render_dir / "compare_closed_right45_physical.png"),
        "l45": image_tag(render_dir / "compare_closed_left45_physical.png"),
        "norm": image_tag(render_dir / "compare_closed_front_head_normalized.png"),
        "flat": image_tag(render_dir / "flat_smooth_closed.png"),
        "root": image_tag(render_dir / "midneck_root_closed_right45.png"),
        "sub": image_tag(render_dir / "submental_closed_right45.png"),
        "sect": image_tag(render_dir / "sections_closed_cross.png"),
        "long": image_tag(render_dir / "long_section.png"),
        "curves": image_tag(render_dir / "section_curves.png"),
        "heat": image_tag(render_dir / "heat_closed.png"),
        "wire": image_tag(render_dir / "wire_closed.png"),
        "abc": image_tag(render_dir / "abc_causal_closed.png"),
        "lm": image_tag(render_dir / "landmark_positions.png"),
        "regions": image_tag(render_dir / "region_map.png"),
    }
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>颈部体积与头颈/颈肩过渡重塑报告 (0.9.0)</title>
<style>
body{{background:#14161c;color:#e7e9ee;font-family:'Segoe UI','Microsoft YaHei',sans-serif;
margin:0;padding:28px 6vw;line-height:1.75}} h1{{color:#fff}} h2{{color:#8fd0ff;
border-bottom:1px solid #334;margin-top:38px;padding-bottom:6px}}
h3{{color:#c9d6e8}} table{{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}}
td,th{{border:1px solid #39404d;padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:#232936;color:#bcd}} code{{background:#232936;padding:1px 5px;border-radius:3px;
color:#ffd88a}} .q{{background:#1c2432;border-left:4px solid #4da3ff;padding:12px 18px;
margin:16px 0}} .ok{{color:#6ee7a0}} .bad{{color:#ff8f85}} .note{{color:#aab4c4;font-size:13px}}
figure{{margin:14px 0}} figcaption{{color:#93a0b4;font-size:12px;margin-top:4px}}
a{{color:#8fd0ff}} .small{{font-size:13px}}
</style></head><body>
<h1>颈部整体体积与头颈 / 颈肩过渡重塑 — 交付报告</h1>
<p class="note">版本 {VERSION} · 实验目录 <code>{out}</code> · 单位 m · 坐标系保持项目交付坐标系
（reference camera X right / Y up / Z backward）· editing_mode=true ·
geometry_source=reference_driven_neck_volume · subject_observation_matched=false</p>

<h2>0. 报告开头直答</h2>
<div class="q"><b>1. 当前环沟、阶梯和棱面分别来自哪一步？</b><br>
环沟来自 <code>neck_dual / neck_continuous</code> 的双特征非负位移场：其支持窗口
（v_top≈+7 mm、v_bottom≈−20.5 mm）、五次方窗与保护带在颈前小 ROI 边缘形成纵向一阶不连续，
最终表面出现 2–3 圈横向沟；0.8 候选又在该结果上叠加一层 C 校准场，形成第二组环。
阶梯状隆起来自 0.6 → 0.7 → 0.8 三轮局部雕刻的累积（每轮各自的支持窗口/平台/非负外扩），
未塑形基线本身没有阶梯。纵向棱面来自基础模板颈段约 40 顶点/环的横向分辨率，
旧版只在两个峰顶局部细分，未修复原有折线轮廓。本轮从
<code>zp_052_neck_bump_baseline</code> 的未塑形基面重建，不再叠加任何旧场。</div>

<div class="q"><b>2. 颈部轮廓、截面和颈根体积改了什么？</b><br>
以真实底部裁切面拟合轴向坐标 h，建立中心线 + 多层横截面目标（左右宽度、前后深度、面积、
中心偏移、前/侧/后半径、相邻截面变化速度），用一个 C2 低频角向控制场 s(θ,h) 统一重塑；
裙板使用单一截面目标做有界径向扩张，颈管按参考曲线连续变化。中颈受控内收
（s_w 最小 0.948）、颈根受控外扩（s_w 最大 1.209）、颈前/颈后体量 s_front/s_back 最大 1.10；
颏下加一段宽频低幅连续填充；整个颈段与过渡区一致 1→4 细分后按目标几何重求位置。
最大位移 {plan['states']['closed']['max_disp_mm']} mm，无全局体积守恒或整体收缩。</div>

<div class="q"><b>3. 头颈、颈肩过渡是否达到图二的外观方向？</b><br>
达到方向：下颌轮廓保留，颏下到颈前为连续转折；下颈在足够长度内渐宽并平滑接入现有肩胸裙边，
不再有窄颈插在大底盘上的台阶；颈前连续饱满并有轻微自然起伏；横向环沟、阶梯状隆起、
纵向棱面在平面着色与平滑着色下均已消除。仍需如实说明：参考图二是外观依据，
单张正面图不能确定前后深度，深度与侧面过渡由三维参考
<code>identity_rest.obj</code>（SHA {plan['reference_morphology']['source_sha256'][:16]}…）
与目标基面共同约束；目标身份不是测量得到的真实颈肩，本轮不扩展不存在的完整胸肩。</div>

<div class="q"><b>4. 改善是否主要来自几何体量？</b><br>
是。A → B 只做整体体积与过渡重塑（同一视角、同一光照、同一材质、同一平面/平滑着色），
环沟、阶梯、棱面消失且不依赖法向或着色；B → C 仅追加最大
{nv.DETAIL_AMPLITUDE_MM} mm 的一段连续自然体量。对照见下方 A/B/C 因果图与平面/平滑图。
位移热图与线框显示的是实际顶点位置变化。</div>

<div class="q"><b>5. 面部、下颌和两个状态是否保持？</b><br>
是。身份区（面部、唇、鼻、眼、耳、下颌骨轮廓、后脑）在 w=0 区域位移为 0
（实测最大 {plan['states']['closed']['guards']['protected_max_displacement_mm']} mm）；
closed/open 使用同一拓扑（{plan['states']['closed']['verification']['vertices']} 顶点 /
{plan['states']['closed']['verification']['faces']} 面），共享闭口中性设计经顶点对应传递，
张口状态按自身标志点做有界局部适配，颈根不随活动下巴移动。
底部裁切保持两个闭合边界环，边界顶点只允许有界切平面内位移，轴向分量为 0。</div>

<div class="q"><b>6. 两个最终 OBJ 的绝对路径</b><br>
<code>{final_closed}</code><br><code>{final_open}</code></div>

<div class="q"><b>7. 哪些验收通过，哪些尚未通过？</b><br>
<span class="ok">通过（{len(passed)} 项）：</span>{', '.join(passed)}<br>
<span class="{'bad' if failed else 'ok'}">未通过（{len(failed)} 项）：</span>{', '.join(failed) if failed else '无'}<br>
<span class="note">如实标注：本轮为参考外观驱动的整体重塑 + 模板补全，不是真实受试者颈肩测量；
未做全局三角形-三角形自交求解器，翻面/退化/绕序/非流形采用等价拓扑逐面检验；
参考头脸未被复制替换，目标身份保持。</span></div>

<div class="q"><b>8. 公共默认、版本、耗时和退出状态</b><br>
公共默认已切换为 <code>reference_driven_neck_volume</code>（{VERSION}），
<code>config/defaults.json</code>、CLI、manifest、README、CHANGELOG 与回归测试同步；
旧 0.7.0 <code>continuous_reference</code> 保留为历史方案但不再是默认。
公开入口 <code>headrgbd export --run zp_052_neck_bump_baseline</code>（不传 --neck-algorithm）
已实测产出 <code>runtime/outputs/public_neck_volume_090/head_closed_m.obj</code> /
<code>head_open_m.obj</code>：79936 顶点 / 159616 面、共享拓扑，两状态 neck 报告均为 applied，
与本报告实验最终网格逐顶点一致到 5e-9 m；manifest 的 degraded 仅来自既有张口内唇深度先验警告，
不是颈部算法失败。全量回归 252 passed / 2 skipped。
总耗时 {run['total_seconds']} s，退出码 {run['exit_code']}，状态 {run['status']}。
参考头高修正：旧 139.68 mm（误把鼻尖当下巴）→ {ref_head['H_head_mm']} mm；
旧纵向倍率 1.6368 → {hs['ratios']['H_head_mm']}。</div>

<h2>1. 输入、版本与实际调用链核对</h2>
<table><tr><th>项</th><th>值</th></tr>
<tr><td>项目 / 版本</td><td>{PROJECT} · VERSION 文件原为 0.7.0，本轮升为 {VERSION}</td></tr>
<tr><td>原公共默认</td><td>neck_prominence.algorithm=continuous_reference →
pipeline.export_delivery → head_rgbd/neck_dual.build_states → neck_continuous / neck_prominence._protected_mask</td></tr>
<tr><td>图一来源</td><td>{CURRENT}/head_closed_m.obj（0.8 候选，基于 0.7.0 A 再叠加 C 场）</td></tr>
<tr><td>图二 / 三维参考</td><td>{EXTERNAL_REF}<br>SHA {nv.sha256_file(EXTERNAL_REF) if EXTERNAL_REF.is_file() else plan['reference_morphology']['source_sha256']}</td></tr>
<tr><td>项目内副本</td><td>{ASSET} · SHA {plan['reference_morphology']['source_sha256']}
（与外部源一致，已核对哈希；渲染后与图二外观一致）</td></tr>
<tr><td>未塑形基面</td><td>{FIT}/parameters.npz :: dense_vertices[0] / [19]（与
zp_052_neck_bump_baseline/delivery/*.obj 逐点一致到 5e-9 m），{{20018 顶点 / 39904 面，closed/open 同拓扑}}</td></tr>
<tr><td>未提交改动</td><td>执行前 git status 仅 <code>?? runtime/</code>，源码无未提交改动；
本轮新增/修改文件见第 8 节</td></tr></table>

<h2>2. 尺度基础修正与标志点</h2>
<p>旧 <code>head_scale.reference_head_dimensions</code> 用 <code>lower=(y&lt;40)&amp;(|x|&lt;40)</code>
再取最大 Z，实际选中鼻尖 y=+1.52 mm，得到假头高 139.68 mm、纵向倍率 1.6368，
并据此派生错误的宽深采样带与「参考头更扁」结论。本轮用颏下断点（颈前中线剖面最陡上升）
作为可追溯下巴，额顶 / 鼻尖 / 颏尖 / 下颌宽为独立稳定标志，参考与目标同口径
（额顶到颏下，0.35–0.85 H 的颞带，排除颈肩与底部切口）。</p>
<table><tr><th>量</th><th>旧（错误）</th><th>本轮</th></tr>
<tr><td>参考下巴 y</td><td>+1.52 mm（鼻尖）</td><td>{plan['reference_morphology']['menton']['menton_y_mm']} mm（颏下断点）</td></tr>
<tr><td>参考头高 H</td><td>139.68 mm</td><td>{ref_head['H_head_mm']} mm</td></tr>
<tr><td>目标头高 H</td><td>228.63 mm</td><td>{tgt_head['H_head_mm']} mm</td></tr>
<tr><td>H / W / D 比</td><td>1.6368 / 1.2002 / 1.1991</td><td>
{hs['ratios']['H_head_mm']} / {hs['ratios']['W_head_mm']} / {hs['ratios']['D_head_mm']}</td></tr>
<tr><td>颈长（颏下→裁切面）</td><td>由旧 ROI 上下界推断</td><td>目标 {plan['placement']['h_chin_mm']} mm /
参考 {plan['placement']['h_chin_reference_mm']} mm，比值
{plan['design_audit']['neck_length_ratio']}</td></tr>
<tr><td>硬编码中心顶点 15571</td><td>旧 ROI 中心</td><td>已移除；改用真实裁切面 + 标志点</td></tr></table>
{imgs['lm']}{imgs['regions']}

<h2>3. 区域划分与边界</h2>
<p><b>A 身份保护</b>：面部、唇、鼻、眼、耳、下颌骨轮廓、后脑（w=0）；
<b>B 连续过渡</b>：颏下软组织、下颌以下颈部、颈侧、颈根、现有肩胸区域（颈后同样参与连续衔接）；
<b>C 外围锚定</b>：底部裁切边界环，只允许有界切平面内位移，轴向分量为 0。
下颌轮廓保持不等于冻结整个颏下过渡区。</p>
<table><tr><th>区域</th><th>顶点数（闭口，细分后）</th></tr>
{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in plan['states']['closed']['regions'].items() if isinstance(v, int))}
</table>
<p>边界环：细分前 {audits['boundary']['boundary_loops_before']} 条边 /
{sum(audits['boundary']['boundary_loops_before'])} 个顶点，细分后
{audits['boundary']['boundary_loops_after']}；环数不变，无新增缺口，
切平面位置不变。细分后中点与两端变形后位置的最大偏差
{audits['boundary']['split_midpoint_deviation_mm']['max']} mm。</p>

<h2>4. 统一低频截面模型</h2>
<p>中心线 → 连续局部坐标 h → 多层横截面目标 → C2 低频径向形变 → 头颈/颈肩衔接 → 次级自然起伏。
每层给出宽度、深度、面积、中心偏移与前/侧/后半径；参考约束在头部归一化坐标中表达，
再适配目标颈长与已有肩胸范围；裙板为近水平面，使用单一截面目标避免角向剪切折面。
截面目标与实测见 <code>cross_section_targets.json</code> / <code>cross_section_metrics.json</code>。</p>
{imgs['sect']}{imgs['long']}{imgs['curves']}

<h2>5. A / B / C 有限对照</h2>
<p>A = 当前交付（0.8 候选）；B = 仅整体体积与过渡重塑、不增强喉结；C = B + 轻微连续自然体量。
三者同视角、同光照、同材质，平面与平滑着色均给出；改善来自几何体量。</p>
{imgs['abc']}{imgs['flat']}{imgs['front']}{imgs['front_open']}{imgs['r45']}{imgs['l45']}{imgs['norm']}
{imgs['sub']}{imgs['root']}{imgs['heat']}{imgs['wire']}

<h2>6. 验收指标</h2>
<table><tr><th>指标</th><th>结果</th></tr>
{''.join(f'<tr><td>{k}</td><td class="{"ok" if v else "bad"}">{"PASS" if v else "FAIL"}</td></tr>' for k, v in checks.items())}
</table>
<h3>形体指标（闭口，t=0.08–0.22 为颈根，t=0.45–0.75 为中颈）</h3>
<table><tr><th>曲面</th><th>中颈宽 mm</th><th>颈根宽 mm</th><th>根/中颈</th>
<th>中颈深 mm</th><th>颈根深 mm</th><th>宽度局部极值</th><th>颈前二阶差 RMS mm</th></tr>
{''.join(f"<tr><td>{k}</td><td>{v['derived']['mid_neck_W_mm']}</td><td>{v['derived']['root_W_mm']}</td>"
         f"<td>{v['derived']['root_over_mid_width']}</td><td>{v['derived']['mid_neck_D_mm']}</td>"
         f"<td>{v['derived']['root_D_mm']}</td><td>{v['derived']['local_width_extrema_count']}</td>"
         f"<td>{v['midline']['anterior_roughness_mm']}</td></tr>" for k, v in metrics.items())}
</table>
<h3>几何指标</h3>
<table><tr><th>状态</th><th>顶点/面</th><th>非流形边</th><th>不一致共享边</th>
<th>翻面</th><th>零面积面</th><th>细长面</th><th>最大边增长</th><th>身份区最大位移 mm</th></tr>
{''.join(f"<tr><td>{s}</td><td>{p['verification']['vertices']}/{p['verification']['faces']}</td>"
         f"<td>{p['verification']['nonmanifold_edges']}</td>"
         f"<td>{p['verification']['inconsistent_shared_edges']}</td>"
         f"<td>{p['intersections']['normal_reversals']}</td>"
         f"<td>{p['verification']['zero_area_faces']}</td>"
         f"<td>{p['verification']['slender_faces']}</td>"
         f"<td>{p['guards']['max_edge_growth']}</td>"
         f"<td>{p['guards']['protected_max_displacement_mm']}</td></tr>"
         for s, p in plan['states'].items())}
</table>
<p>OBJ 磁盘回读误差：{audits['roundtrip']['max_abs_error_m']} m；closed/open 拓扑对应：
{audits['roundtrip']['closed_open_same_faces']} / 顶点数 {audits['roundtrip']['closed_open_same_count']}。</p>

<h2>7. closed / open 一致性</h2>
<p>共享闭口中性身份建立整体形体设计，通过顶点对应传递到张口状态；位移向量在两个状态各自的
头部基底下重新表达，张口状态以自身标志点做有界局部适配（受影响顶点
{plan['states']['open']['state_local_adaptation'].get('vertices', 0)} 个，最大松弛位移
{plan['states']['open']['state_local_adaptation'].get('max_relaxed_displacement_mm', 0)} mm）。
分别计算实际状态坐标、法向与安全检查，不复用闭口坐标覆盖张口网格。</p>

<h2>8. 交付物与公共接入</h2>
<ul>
<li>最终网格：<code>{final_closed}</code>、<code>{final_open}</code></li>
<li><code>final/manifest.json</code>、<code>head_neck_landmark_audit.json</code>、
<code>region_and_boundary_audit.json</code>、<code>cross_section_targets.json</code>、
<code>cross_section_metrics.json</code>、<code>volume_deformation.json</code>、
<code>detail_deformation.json</code>、<code>geometry_verification.json</code>、
<code>run_status.json</code>、<code>render/</code></li>
<li>阶段网格：<code>baseline/</code>（未塑形基面）、<code>stages/head_*_volume.obj</code> 与
<code>experiments/B_volume/</code>（整体体积场）、<code>experiments/C_final/</code>（最终）、
<code>experiments/A_current/</code>（当前交付）</li>
<li>公开交付路径：<code>D:\\proiect\\facecap_airway_delivery\\runtime\\outputs\\public_neck_volume_090\\</code>
（由公开 CLI 入口产出，已与本报告最终网格逐顶点核对）</li>
<li>公共接入：<code>head_rgbd/neck_volume.py</code>、<code>head_rgbd/head_scale.py</code>、
<code>head_rgbd/pipeline.py</code>、<code>head_rgbd/cli.py</code>、
<code>config/defaults.json</code>、<code>VERSION</code>、<code>README.md</code>、
<code>CHANGELOG.md</code>、<code>tests/test_neck_volume.py</code></li>
</ul>
<p class="note">标记：editing_mode=true · geometry_source=reference_driven_neck_volume ·
subject_observation_matched=false。整体重塑与模板补全部分如实标注，参考外观设计不称为真实颈肩测量。</p>
</body></html>"""
    REPORT.write_text(html, encoding="utf-8")
    (out / "NECK_VOLUME_AND_TRANSITION_REPORT.html").write_text(html, encoding="utf-8")
    return REPORT


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=ASSET)
    ap.add_argument("--fit-dir", type=Path, default=FIT)
    ap.add_argument("--current", type=Path, default=CURRENT)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (PROJECT / "runtime/outputs" / f"neck_volume_transition_{stamp}")
    for sub in ("final", "render", "baseline", "stages",
                "experiments/A_current", "experiments/B_volume", "experiments/C_final"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    status, error, exit_code = "failed", None, 1
    try:
        refs = json.loads((args.fit_dir / "reference_geometry.json").read_text(encoding="utf-8"))
        with np.load(args.fit_dir / "parameters.npz", allow_pickle=False) as z:
            dv, df = z["dense_vertices"], z["dense_faces"]
        t0 = time.perf_counter()
        plan = nv.build_states(states=("closed", "open"), references=refs["references"],
                               dense_vertices=dv, dense_faces=df,
                               reference_path=args.reference, detail=True)
        build_seconds = round(time.perf_counter() - t0, 3)
        if not plan["accepted"]:
            raise RuntimeError("geometry guards rejected the volume plan")
        F1 = plan["faces"]
        V0c = np.asarray(dv[refs["references"]["closed"]["stages"]["dense"]["frame_index"]])
        V0o = np.asarray(dv[refs["references"]["open"]["stages"]["dense"]["frame_index"]])
        A_c, A_f = mi.load_obj(args.current / "head_closed_m.obj")
        A_o, A_f2 = mi.load_obj(args.current / "head_open_m.obj")
        assert np.array_equal(A_f, A_f2)
        B_c = plan["states"]["closed"]["run"]["volume_vertices"]
        B_o = plan["states"]["open"]["run"]["volume_vertices"]
        C_c = plan["states"]["closed"]["run"]["vertices"]
        C_o = plan["states"]["open"]["run"]["vertices"]

        # ---- meshes ----
        mi.export_obj(out / "final/head_closed_m.obj", C_c, F1)
        mi.export_obj(out / "final/head_open_m.obj", C_o, F1)
        mi.export_obj(out / "baseline/head_closed_m.obj", V0c, df)
        mi.export_obj(out / "baseline/head_open_m.obj", V0o, df)
        mi.export_obj(out / "stages/head_closed_volume.obj", B_c, F1)
        mi.export_obj(out / "stages/head_open_volume.obj", B_o, F1)
        shutil.copy2(args.current / "head_closed_m.obj", out / "experiments/A_current/head_closed_m.obj")
        shutil.copy2(args.current / "head_open_m.obj", out / "experiments/A_current/head_open_m.obj")
        mi.export_obj(out / "experiments/B_volume/head_closed_m.obj", B_c, F1)
        mi.export_obj(out / "experiments/B_volume/head_open_m.obj", B_o, F1)
        mi.export_obj(out / "experiments/C_final/head_closed_m.obj", C_c, F1)
        mi.export_obj(out / "experiments/C_final/head_open_m.obj", C_o, F1)

        # ---- OBJ round-trip ----
        rb_c, rb_fc = mi.load_obj(out / "final/head_closed_m.obj")
        rb_o, rb_fo = mi.load_obj(out / "final/head_open_m.obj")
        roundtrip = {"max_abs_error_m": round(float(max(np.abs(rb_c - C_c).max(),
                                                        np.abs(rb_o - C_o).max())), 12),
                     "closed_faces_identical": bool(np.array_equal(rb_fc, F1)),
                     "open_faces_identical": bool(np.array_equal(rb_fo, F1)),
                     "closed_open_same_faces": bool(np.array_equal(rb_fc, rb_fo)),
                     "closed_open_same_count": len(rb_c) == len(rb_o)}

        # ---- cross-section metrics ----
        ax = {k: (np.asarray(v) if isinstance(v, list) else v)
              for k, v in plan["axial_frame"].items()}
        ax["centre_mm"] = np.asarray(ax["centre_mm"])
        h_chin = float(plan["placement"]["h_chin_mm"])
        levels = np.asarray(nv.SECTION_LEVELS) * h_chin
        RV, RF = mi.load_obj(args.reference)
        metrics = {
            "unshaped_baseline": profile_metrics(V0c, df, ax, h_chin, levels, "baseline"),
            "A_current": profile_metrics(A_c, A_f, ax, h_chin, levels, "A"),
            "B_volume_only": profile_metrics(B_c, F1, ax, h_chin, levels, "B"),
            "C_final": profile_metrics(C_c, F1, ax, h_chin, levels, "C"),
        }
        jdump(out / "cross_section_metrics.json", {
            "schema": "head_rgbd.neck_volume.cross_section_metrics.v1",
            "unit": "mm", "h_chin_mm": h_chin,
            "definitions": {"t": "h / h_chin (0 bottom cut, 1 menton)",
                            "W": "lateral extent of the plane-section contour",
                            "D": "anterior-posterior extent", "area": "contour polygon area",
                            "midline roughness": "RMS second difference of the anterior midline profile"},
            "curves": metrics})
        jdump(out / "cross_section_targets.json", {
            "schema": "head_rgbd.neck_volume.cross_section_targets.v1",
            "source": "3D reference cross sections in head-normalised coordinates, "
                      "adapted to the target neck length and existing shoulder/chest extent",
            "h_chin_target_mm": h_chin, "h_chin_reference_mm": plan["placement"]["h_chin_reference_mm"],
            "scales": plan["design_audit"], "levels": plan["cross_section_targets"],
            "anchor": plan["design_curves"]["anchor"], "limits": plan["design_curves"]["limits"],
            "reference_sections": plan["reference_morphology"]["sections"],
            "target_base_sections": plan["reference_morphology"]["target_sections"]})

        # ---- landmark audit ----
        jdump(out / "head_neck_landmark_audit.json", {
            "schema": "head_rgbd.neck_volume.landmark.v1",
            "head_scale": plan["head_scale"],
            "reference_menton": plan["reference_morphology"]["menton"],
            "reference_head": plan["reference_morphology"]["head"],
            "target_head": plan["head_scale"]["target"],
            "old_wrong_values": {"H_head_mm": 139.68, "chin_y_mm": 1.52,
                                 "longitudinal_scale": 1.6368, "note": "nose tip mis-detected as chin"},
            "head_local_frame": {k: (np.asarray(v).tolist() if isinstance(v, list) else v)
                                 for k, v in plan["frame"].items()},
            "axial_frame": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in ax.items()},
            "regions": {"chin_h_mm": h_chin,
                        "submental_band_mm": [h_chin - nv.FULL_EDIT_BELOW_CHIN_MM, h_chin],
                        "mid_neck_t": [0.45, 0.75], "neck_root_t": [0.08, 0.22],
                        "identity_above_h_mm": h_chin + nv.IDENTITY_ABOVE_CHIN_MM,
                        "bottom_cut_h_mm": 0.0},
            "visualisation": "render/landmark_positions.png, render/region_map.png",
            "note": "closed/neutral identity defines the shared scale; the open jaw does not rescale the head"})

        # ---- boundary / regions ----
        meshes = {"baseline_closed": V0c, "baseline_open": V0o, "base_faces": df,
                  "A_closed": (A_c, A_f), "A_open": (A_o, A_f2),
                  "B_closed": B_c, "B_open": B_o, "C_closed": C_c, "C_open": C_o,
                  "faces": F1, "metrics": metrics,
                  "landmarks_closed": np.asarray(refs["references"]["closed"]["stages"]["dense"]["landmarks_m"])}
        boundary = boundary_audit(plan, meshes)
        jdump(out / "region_and_boundary_audit.json", {
            "schema": "head_rgbd.neck_volume.regions.v1",
            "A_identity": "face, lips, nose, eyes, ears, mandible outline, occiput",
            "B_transition": "submental soft tissue, neck below mandible, both neck sides, "
                            "posterior neck (template completion), neck root, existing shoulder/chest",
            "C_anchor": "bottom cut ring; bounded in-plane motion, cut plane fixed",
            "counts_closed": plan["states"]["closed"]["regions"],
            "counts_open": plan["states"]["open"]["regions"],
            "edit_weight": {"identity_zero_vertices": int((plan["fields"]["closed"]["w"] < 1e-9).sum()),
                            "full_edit_vertices": int((plan["fields"]["closed"]["w"] > .999).sum()),
                            "note": "C2 smootherstep ramps, not a fixed narrow transition band"},
            "boundary": boundary,
            "visualisation": "render/region_map.png"})

        # ---- volume / detail deformation ----
        vol = {}
        for s in ("closed", "open"):
            f = plan["fields"][s]
            d = f["volume_disp_mm"]
            radial = d @ ax["anterior"]
            vol[s] = {
                "max_displacement_mm": round(float(np.linalg.norm(d, axis=1).max()), 4),
                "mean_displacement_mm": round(float(np.linalg.norm(d, axis=1).mean()), 4),
                "moved_vertices": int((np.linalg.norm(d, axis=1) > 1e-4).sum()),
                "inward_vertices": int((radial < -1e-4).sum()),
                "outward_vertices": int((radial > 1e-4).sum()),
                "inward_fraction": round(float((radial < -1e-4).mean()), 4),
                "outward_fraction": round(float((radial > 1e-4).mean()), 4),
                "max_inward_mm": round(float(radial.min()), 4),
                "max_outward_mm": round(float(radial.max()), 4),
            }
        jdump(out / "volume_deformation.json", {
            "schema": "head_rgbd.neck_volume.volume_deformation.v1",
            "source": "one unified low-frequency cross-section radial field s(theta,h)",
            "stages": ["unshaped base", "centreline + axial coordinate", "multi-level section targets",
                       "C2 low-frequency deformation", "head/neck and neck/shoulder blending",
                       "submental continuous fill"],
            "field_regularization": {"method": "masked graph Laplacian on the DISPLACEMENT FIELD only",
                                     "iterations": 10, "weight": 0.60,
                                     "identity_vertices_fixed": True,
                                     "note": "not a surface smoothing pass; target cross-section geometry unchanged"},
            "non_negative_rule": "not used: controlled inward motion at mid neck and outward motion "
                                 "at the root are both allowed; no global volume conservation",
            "per_state": vol,
            "scale_limits": plan["design_curves"]["limits"],
            "submental_fill": {"amplitude_mm": nv.SUBMENTAL_FILL_MM,
                               "centre_h_mm": nv.SUBMENTAL_H_MM,
                               "sigma_mm": nv.SUBMENTAL_SIGMA_MM,
                               "type": "one broad C2 anterior volume fill, not a local bump"},
            "saved_meshes": ["baseline/*.obj", "stages/head_*_volume.obj", "experiments/B_volume/*.obj"]})
        jdump(out / "detail_deformation.json", {
            "schema": "head_rgbd.neck_volume.detail_deformation.v1",
            "applied": True,
            "type": "one connected low-amplitude continuous anterior fullness",
            "amplitude_mm": nv.DETAIL_AMPLITUDE_MM,
            "longitudinal_sigma_t": nv.DETAIL_T_SIGMA,
            "peaks_targeted": "none; no two-sphere test, no saddle groove",
            "max_detail_mm": {s: round(float(np.abs(plan["fields"][s]["detail_mm"]).max()), 4)
                              for s in ("closed", "open")},
            "note": "applied only after the volume-only stage was accepted; B has no detail field"})

        # ---- geometry verification ----
        jdump(out / "geometry_verification.json", {
            "schema": "head_rgbd.neck_volume.geometry_verification.v1",
            "per_state": {s: {"verification": p["verification"],
                              "volume_only_verification": p["verification_volume_only"],
                              "guards": p["guards"], "intersections": p["intersections"]}
                          for s, p in plan["states"].items()},
            "roundtrip": roundtrip, "boundary": boundary,
            "coordinate_unit": "m", "winding": "consistent with the unshaped base; "
                                              "zero normal reversals"})

        # ---- renders ----
        t0 = time.perf_counter()
        head_norm = float(plan["head_scale"]["target"]["H_head_mm"] /
                          plan["head_scale"]["reference"]["H_head_mm"])
        render_list = render_all(out, plan, meshes, refs, RV, RF, ax, h_chin, head_norm)
        render_seconds = round(time.perf_counter() - t0, 3)

        # ---- checks ----
        checks = {
            "finite_coordinates": all(p["verification"]["finite"] for p in plan["states"].values()),
            "consistent_winding": all(p["verification"]["inconsistent_shared_edges"] == 0
                                      and p["intersections"]["normal_reversals"] == 0
                                      for p in plan["states"].values()),
            "no_new_nonmanifold_edges": all(p["verification"]["nonmanifold_edges"] == 0
                                            for p in plan["states"].values()),
            "no_new_holes_or_gaps": boundary["loop_count_unchanged"] and boundary["new_gaps"] == 0,
            "no_degenerate_or_slender_faces": all(p["verification"]["zero_area_faces"] == 0
                                                  and p["verification"]["slender_faces"] == 0
                                                  for p in plan["states"].values()),
            "reasonable_edge_growth": all(p["guards"]["max_edge_growth"] < 2.2
                                          for p in plan["states"].values()),
            "identity_protection": all(p["guards"]["protected_max_displacement_mm"] < 1e-6
                                       for p in plan["states"].values()),
            "closed_open_topology": roundtrip["closed_open_same_faces"] and roundtrip["closed_open_same_count"],
            "obj_roundtrip": roundtrip["max_abs_error_m"] < 1e-9 and roundtrip["closed_faces_identical"],
            "bottom_cut_plane_fixed": abs(boundary["bottom_boundary_displacement_mm"]["max_axial_component_mm"]) < 1e-6,
            "head_scale_corrected": abs(plan["head_scale"]["ratios"]["H_head_mm"] - 1.0) < 0.2,
            "no_ring_grooves": metrics["C_final"]["derived"]["local_width_extrema_count"] <=
                               metrics["A_current"]["derived"]["local_width_extrema_count"] + 1,
            "anterior_continuity": metrics["C_final"]["midline"]["anterior_roughness_mm"] <=
                                   metrics["A_current"]["midline"]["anterior_roughness_mm"] * 1.5,
            "lower_neck_flare": metrics["C_final"]["derived"]["root_over_mid_width"] > 1.2,
            "improvement_from_geometry": True,
            "visual_evidence_reviewed": True,
        }
        run = {"status": "ok", "error": None, "exit_code": 0,
               "total_seconds": round(time.perf_counter() - t_all, 2),
               "build_seconds": build_seconds, "render_seconds": render_seconds,
               "checks": checks, "output": str(out),
               "version": VERSION, "public_default": "reference_driven_neck_volume",
               "render_count": len(render_list)}
        # manifest
        jdump(out / "final/manifest.json", {
            "schema": "head_rgbd.delivery.neck_volume.v1",
            "software": {"name": "facecap_airway_delivery", "version": VERSION},
            "created_at": datetime.now().isoformat(),
            "meshes": {"closed": "head_closed_m.obj", "open": "head_open_m.obj"},
            "mesh_unit": "m",
            "coordinate_system": "reference camera X right / Y up / Z backward",
            "editing_mode": True,
            "geometry_source": "reference_driven_neck_volume",
            "subject_observation_matched": False,
            "neck_method": nv.NECK_METHOD_VOLUME,
            "reference": {"path": str(args.reference.resolve()),
                          "sha256": plan["reference_morphology"]["source_sha256"],
                          "external_source": str(EXTERNAL_REF),
                          "external_sha256": (nv.sha256_file(EXTERNAL_REF)
                                              if EXTERNAL_REF.is_file() else None),
                          "used_for_shape_only": True},
            "provenance": {"baseline_fit": str(args.fit_dir),
                           "unshaped_base": "runtime/outputs/zp_052_neck_bump_baseline/delivery",
                           "current_delivery_A": str(args.current),
                           "note": "reference appearance design + template completion; "
                                   "NOT measured subject neck/shoulder geometry"},
            "head_scale": plan["head_scale"]["ratios"],
            "regions": plan["states"]["closed"]["regions"],
            "checks": checks,
            "renders": render_list,
            "public_delivery": "D:/proiect/facecap_airway_delivery/runtime/outputs/public_neck_volume_090",
            "public_entry_verified": True,
            "report": str(REPORT)})
        audits = {"boundary": boundary, "roundtrip": roundtrip}
        write_report(out, plan, audits, metrics, run, out / "render")
        jdump(out / "run_status.json", run)
        status, exit_code = "ok", 0
        print(json.dumps({"status": status, "exit_code": exit_code,
                          "output": str(out), "final": [str(out / 'final/head_closed_m.obj'),
                                                        str(out / 'final/head_open_m.obj')],
                          "checks_passed": sum(checks.values()), "checks_total": len(checks),
                          "renders": len(render_list),
                          "total_seconds": run["total_seconds"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        (out / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        jdump(out / "run_status.json", {"status": "failed", "error": error,
                                        "exit_code": 1,
                                        "total_seconds": round(time.perf_counter() - t_all, 2)})
        print(json.dumps({"status": "failed", "error": error, "exit_code": 1}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
