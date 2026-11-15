"""Larynx-focused visualization for the anatomy-prior reconstruction.

Produces four same-camera/same-light panels under ``<out>/render``:

``larynx_compare_smooth.png``  input / old 1.10 / gain 1.3 / anatomy+prior,
                                front and oblique, smooth shading;
``larynx_compare_wire.png``    the same four models as actual wireframes;
``larynx_sections.png``        horizontal sections at t=0.68/0.73/0.78,
                                showing the central ridge and lateral hollows;
``larynx_profile.png``         delivered relief and target applied relief along
                                theta ~ 0, with the laryngeal local maximum.

All models are framed from the **original capture** with one centre/half and
one light; no extra normal smoothing is used beyond the stated 2.5 mm vertex
normals for the smooth panels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection      # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi             # noqa: E402
from head_rgbd import neck_direct as nd                 # noqa: E402
from head_rgbd import neck_stage as ns                  # noqa: E402
from head_rgbd import raster as rs                      # noqa: E402
from tools import build_neck_stage as bns              # noqa: E402
from tools import neck_direct_render as r1             # noqa: E402
from tools import neck_stage_render as r0             # noqa: E402

BG = r0.BG
FG = r0.FG
COLOURS = ("#9a9a9a", "#4aa3ff", "#7ee0a0", "#ff9f40")


def _models(out, old_dir, gain_dir):
    """Input and the three delivered closed meshes, all in the same pose."""
    out = Path(out)
    meta = json.load(open(out / "build_direct.json", encoding="utf-8"))
    fit = Path(meta["provenance"]["fit_dir"])
    states, F0, _ = bns.load_subject(fit)
    V0 = np.asarray(states["closed"]["V"], float)
    models = [("original capture", V0, F0)]
    paths = [("old 1.10", old_dir), ("gain 1.3", gain_dir),
             ("anatomy + prior", out)]
    for label, d in paths:
        d = Path(d)
        if not d.is_absolute():
            d = PROJ / d
        obj = d / "head_closed_direct.obj"
        if not obj.exists():
            continue
        V, F = mi.load_obj(obj)
        models.append((label, V, F))
    return models, fit, meta


def _larynx_frame(V0, ax, hc, curve, view):
    return r1._roi_frame(V0, ax, hc, curve, view,
                         t_lo=0.62, t_hi=0.88, deg_max=55.0, margin=1.30)


def _smooth(V, F, view, centre, half):
    vn = r0._soft_normals(V, F, mm=2.5)
    return rs.render(V, F, view, centre, half, px=r0.PX, ss=2,
                     light=rs.light_vector(view, -20.0, 10.0), vn=vn)


def _panel_grid(models, V0, ax, hc, curve, out_path, wire=False):
    views = [("front", r0.VIEWS["front"]),
             ("oblique left", r0.VIEWS["oblique_left"]),
             ("side left", r0.VIEWS["side_left"])]
    fig, axs = plt.subplots(len(views), len(models),
                            figsize=(3.4 * len(models), 3.7 * len(views)),
                            facecolor=BG)
    axs = np.atleast_2d(axs)
    for r, (vname, view) in enumerate(views):
        centre, half = _larynx_frame(V0, ax, hc, curve, view)
        for c, (label, V, F) in enumerate(models):
            img = (r1.render_wire(V, F, view, centre, half) if wire
                   else _smooth(V, F, view, centre, half))
            axs[r, c].imshow(img, interpolation="bilinear")
            axs[r, c].axis("off")
            axs[r, c].set_title(f"{label} - {vname}", color=FG, fontsize=8.0,
                                pad=3)
    title = ("laryngeal region: actual wireframe" if wire
             else "laryngeal region: smooth shading (2.5 mm normals)")
    fig.suptitle(title, color=FG, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6)
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    return out_path


def _sections(models, ax, curve, hc, out_path):
    levels = [0.68, 0.73, 0.78]
    fig, axs = plt.subplots(1, len(levels), figsize=(4.3 * len(levels), 4.4),
                            facecolor=BG, sharex=True, sharey=True)
    if len(levels) == 1:
        axs = [axs]
    for a, t in enumerate(levels):
        segs = []
        for label, V, F in models:
            s = ns.section_segments(V, F, ax, float(t * hc))
            if len(s):
                segs.append(s)
        if not segs:
            continue
        allp = np.concatenate(segs, axis=0).reshape(-1, 2)
        # front half only: anterior b>0; a lateral
        front = allp[:, 1] > 0
        if front.any():
            lo_a, hi_a = np.percentile(allp[front, 0], [1, 99])
            lo_b, hi_b = np.percentile(allp[front, 1], [1, 99])
            pad = 0.08 * max(hi_a - lo_a, hi_b - lo_b)
            axs[a].set_xlim(lo_a - pad, hi_a + pad)
            axs[a].set_ylim(max(0.0, lo_b - pad), hi_b + pad)
        for (label, V, F), col in zip(models, COLOURS):
            s = ns.section_segments(V, F, ax, float(t * hc))
            if not len(s):
                continue
            lc = LineCollection(s, colors=col, linewidths=0.7, alpha=0.95,
                                label=label)
            axs[a].add_collection(lc)
        axs[a].set_title(f"t = {t:.2f} ({t * hc:.1f} mm)", color=FG, fontsize=9)
        axs[a].set_xlabel("a lateral (mm)", color=FG)
        axs[a].tick_params(colors=FG)
        axs[a].grid(alpha=0.20)
        axs[a].set_aspect("equal", adjustable="box")
    axs[0].set_ylabel("b anterior (mm)", color=FG)
    axs[-1].legend(fontsize=7, facecolor=BG, labelcolor=FG, loc="lower right")
    fig.suptitle("laryngeal horizontal sections: ridge and lateral hollows",
                 color=FG, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6)
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    return out_path


def _profile(models, meta, old_dir, gain_dir, prior_dir, out_png, ax, hc, curve):
    sig = float(meta["params"].get("structure_sigma_mm", 13.0))
    fine = float(meta["params"].get("fine_sigma_mm", 1.2))
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.8), facecolor=BG)
    # delivered relief on each actual surface
    for (label, V, F), col in zip(models, COLOURS):
        f = ns.analyse_surface(V, F, ax, hc, curve=curve,
                               structure_sigma=sig, fine_sigma=fine,
                               leak_mask=True)
        j = int(np.argmin(np.abs(f["theta"])))
        rel = np.asarray(f["relief"], float)[:, j]
        t = np.asarray(f["t"], float)
        axs[0].plot(t, rel, color=col, lw=1.1, label=label)
        m = (t > 0.65) & (t < 0.82)
        if m.any():
            k = np.argmax(rel[m])
            tt, rr = t[m][k], rel[m][k]
            axs[0].plot([tt], [rr], "o", color=col, ms=4)
            axs[0].annotate(f"{rr:.2f} @ {tt:.3f}", (tt, rr), color=col,
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
    axs[0].set_xlabel("material t", color=FG)
    axs[0].set_ylabel("delivered relief at theta ~0 (mm)", color=FG)
    axs[0].set_title("delivered surface: laryngeal local maximum marked", color=FG,
                     fontsize=9)
    axs[0].tick_params(colors=FG)
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=7, facecolor=BG, labelcolor=FG)
    # target applied profile from each build's own 2-D field
    for (label, d), col in ((("old 1.10", old_dir), COLOURS[1]),
                            (("gain 1.3", gain_dir), COLOURS[2]),
                            (("anatomy + prior", prior_dir), COLOURS[3])):
        d = Path(d)
        if not d.is_absolute():
            d = PROJ / d
        z = d / "stage_direct.npz"
        if not z.exists():
            continue
        npz = np.load(z)
        if "applied_closed" not in npz or "theta" not in npz:
            continue
        th = np.asarray(npz["theta"], float)
        j = int(np.argmin(np.abs(th)))
        applied = np.asarray(npz["applied_closed"], float)[:, j]
        t = np.asarray(npz["t"], float)
        axs[1].plot(t, applied, color=col, lw=1.0, ls="--", label=label + " target")
    axs[1].set_xlabel("material t", color=FG)
    axs[1].set_ylabel("target applied relief at theta ~0 (mm)", color=FG)
    axs[1].set_title("target applied (dashed) vs delivered (solid)", color=FG,
                     fontsize=9)
    axs[1].tick_params(colors=FG)
    axs[1].grid(alpha=0.25)
    axs[1].legend(fontsize=7, facecolor=BG, labelcolor=FG)
    fig.suptitle("laryngeal profile: delivered shape and target",
                 color=FG, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6)
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)
    return out_png


def render_larynx(out, old_dir, gain_dir):
    out = Path(out)
    if not out.is_absolute():
        out = PROJ / out
    rnd = out / "render"
    rnd.mkdir(parents=True, exist_ok=True)
    models, fit, meta = _models(out, old_dir, gain_dir)
    states, F0, _ = bns.load_subject(fit)
    V0 = np.asarray(states["closed"]["V"], float)
    _, ax, hc = ns.subject_frame(V0, F0, states["closed"]["landmarks"])
    curve, _ = ns.spine_curve(V0, F0, ax, hc)
    prod = {}
    prod["smooth"] = str(_panel_grid(models, V0, ax, hc, curve,
                                     rnd / "larynx_compare_smooth.png",
                                     wire=False))
    prod["wire"] = str(_panel_grid(models, V0, ax, hc, curve,
                                   rnd / "larynx_compare_wire.png",
                                   wire=True))
    prod["sections"] = str(_sections(models, ax, curve, hc,
                                     rnd / "larynx_sections.png"))
    prod["profile"] = str(_profile(models, meta, old_dir, gain_dir, out,
                                   rnd / "larynx_profile.png", ax, hc, curve))
    return prod


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/liguanghao_cap/neck_direct_v2_anatomy_prior")
    ap.add_argument("--old", default="runtime/outputs/liguanghao_cap/neck_direct_v2")
    ap.add_argument("--gain13", default="runtime/outputs/liguanghao_cap/neck_direct_v2_larynx13")
    a = ap.parse_args()
    print(json.dumps(render_larynx(a.out, a.old, a.gain13), indent=1))