"""Same-camera rendering for the direct-shaping acceptance.

Three models are always shown side by side under one orthographic camera, one
framing and one light: the **original** capture, the **direct** result and the
**accepted** appearance (regression only).  Structure is judged from real
shading and from the wireframe; no texture, bump map or normal smoothing is used
to hide geometry except in the one panel that states it does so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Patch                  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi             # noqa: E402
from head_rgbd import neck_stage as ns                 # noqa: E402
from head_rgbd import raster as rs                     # noqa: E402
from head_rgbd import neck_direct as nd                # noqa: E402
from tools import build_neck_stage as bns              # noqa: E402
from tools.neck_stage_render import (                  # noqa: E402
    VIEWS, HALF, PX, BG, FG, _centre, _figure, _show, _render_flat, _soft_normals)


def render_wire(V, F, view, centre, half, px=PX, ss=2,
                edge=(0.80, 0.83, 0.90), bg=(0.105, 0.105, 0.115)):
    """Hidden-line wireframe: a faint fill, then only edges that own the z-buffer.

    Each projected edge is sampled once per pixel of its own length, so a 7 mm
    edge drawn in a 3.3 px/mm window gets ~23 samples and reads as a line, not
    as a few dots (undersampling is what makes a correct mesh look like noise).
    """
    size = int(px * ss)
    Vmm = np.asarray(V, float) * rs.MM_PER_UNIT
    F = np.asarray(F, np.int64)
    cmm = np.asarray(centre, float) * rs.MM_PER_UNIT
    xy, Z, Fr, fwd = rs._project(Vmm, F, view, cmm, half, size, keep_backface=True)
    tri = Vmm[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    face_light = fn @ (-fwd)
    light = rs.light_vector(view, 0.0, 15.0)
    shade = (0.10 + 0.22 * np.clip(fn @ light, 0.0, 1.0))[fn @ (-fwd) > 0.0]
    zbuf = np.full((size, size), np.inf, np.float32)
    sbuf = np.zeros((size, size), np.float32)
    rs._rasterise(xy, Z, Fr, shade[:, None], size, zbuf, sbuf)
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    fill = np.isfinite(zbuf)
    img[fill] = (sbuf[fill][:, None] * np.array([0.55, 0.55, 0.62], np.float32))

    # only the clearly camera-facing part: grazing faces add edge clutter without
    # adding any information about the tessellation
    Fr = F[face_light > 0.30]
    e0 = np.concatenate([Fr[:, 0], Fr[:, 1], Fr[:, 2]])
    e1 = np.concatenate([Fr[:, 1], Fr[:, 2], Fr[:, 0]])
    L = np.hypot(xy[e1, 0] - xy[e0, 0], xy[e1, 1] - xy[e0, 1])
    kmax = 64
    k = np.clip(np.ceil(L * 1.5), 2, kmax).astype(np.int64)
    idx = np.repeat(np.arange(len(e0)), k)
    step = 1.0 / (k[idx] - 1)
    off = np.concatenate([np.arange(kk) * st for kk, st in zip(k, step)])
    t = off
    X = xy[e0[idx], 0] * (1 - t) + xy[e1[idx], 0] * t
    Y = xy[e0[idx], 1] * (1 - t) + xy[e1[idx], 1] * t
    Zs = Z[e0[idx]] * (1 - t) + Z[e1[idx]] * t
    xi = np.clip(X.astype(np.int32), 0, size - 1)
    yi = np.clip(Y.astype(np.int32), 0, size - 1)
    vis = Zs <= zbuf[yi, xi] + 0.05
    img[yi[vis], xi[vis]] = np.asarray(edge, np.float32)
    if ss > 1:
        img = img.reshape(px, ss, px, ss, 3).mean(axis=(1, 3))
    return np.clip(img, 0.0, 1.0)


def render_scalar(V, F, view, centre, half, values, px=PX, ss=2,
                  vmin=-3.0, vmax=5.0, keep=None, grey=0.26, bg=(0.105, 0.105, 0.115)):
    """Flat colour from a per-vertex scalar: a measured quantity, not shading.

    ``keep`` marks where the scalar is meaningful; elsewhere the same silhouette
    is drawn in flat grey, so the reader can tell "zero" from "not defined here".
    """
    from matplotlib import colormaps
    size = int(px * ss)
    Vmm = np.asarray(V, float) * rs.MM_PER_UNIT
    F = np.asarray(F, np.int64)
    cmm = np.asarray(centre, float) * rs.MM_PER_UNIT
    xy, Z, Fr, _ = rs._project(Vmm, F, view, cmm, half, size)
    val = np.asarray(values, float).reshape(-1)
    zbuf = np.full((size, size), np.inf, np.float32)
    sbuf = np.zeros((size, size), np.float32)
    rs._rasterise(xy, Z, Fr, val[:, None], size, zbuf, sbuf)
    filled = np.isfinite(zbuf)
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    n = np.clip((sbuf - vmin) / (vmax - vmin), 0.0, 1.0)
    img[filled] = np.asarray(colormaps["coolwarm"](n[filled]))[:, :3]
    if keep is not None:
        kb = np.full((size, size), np.inf, np.float32)
        ks = np.zeros((size, size), np.float32)
        rs._rasterise(xy, Z, Fr, np.asarray(keep, float).reshape(-1, 1), size, kb, ks)
        out = np.isfinite(kb) & (ks < 0.6) & filled
        img[out] = np.array([grey, grey, grey * 1.05], np.float32)
    if ss > 1:
        img = img.reshape(px, ss, px, ss, 3).mean(axis=(1, 3))
    return np.clip(img, 0.0, 1.0)


REGION_PALETTE = np.array([[0.60, 0.60, 0.64],     # A  fixed head
                           [0.98, 0.60, 0.16],     # B  submental bridge
                           [0.28, 0.60, 0.95],     # C  shaped neck
                           [0.36, 0.78, 0.46]],    # hand-over (no budget)
                          np.float32)
REGION_NAMES = ("A fixed head", "B submental bridge", "C shaped neck", "hand-over")


def _draw_edges(img, xy, Z, zbuf, size, e0, e1, colour, width=1):
    """Sample every selected edge at ~1.5 samples/pixel and keep the visible ones."""
    L = np.hypot(xy[e1, 0] - xy[e0, 0], xy[e1, 1] - xy[e0, 1])
    k = np.clip(np.ceil(L * 3.0), 3, 128).astype(np.int64)
    idx = np.repeat(np.arange(len(e0)), k)
    t = np.concatenate([np.arange(kk) / max(kk - 1, 1) for kk in k])
    X = xy[e0[idx], 0] * (1 - t) + xy[e1[idx], 0] * t
    Y = xy[e0[idx], 1] * (1 - t) + xy[e1[idx], 1] * t
    Zs = Z[e0[idx]] * (1 - t) + Z[e1[idx]] * t
    xi = np.clip(X.astype(np.int32), 0, size - 1)
    yi = np.clip(Y.astype(np.int32), 0, size - 1)
    vis = Zs <= zbuf[yi, xi] + 0.8
    col = np.asarray(colour, np.float32)
    for dy in range(-(width // 2), width // 2 + 1):
        for dx in range(-(width // 2), width // 2 + 1):
            img[np.clip(yi[vis] + dy, 0, size - 1),
                np.clip(xi[vis] + dx, 0, size - 1)] = col


def render_region(V, F, view, centre, half, region, palette=REGION_PALETTE, px=PX, ss=2,
                  overlay_edges=None, overlay_col=(1.0, 1.0, 1.0), width=1,
                  bg=(0.105, 0.105, 0.115)):
    """Discrete region colouring: one flat colour per label, no lighting, no ramp.

    A continuous colour map would imply the labels are ordered and would
    interpolate across a boundary that is a *decision*, so the label is rounded
    back to its own index after rasterising.
    """
    size = int(px * ss)
    Vmm = np.asarray(V, float) * rs.MM_PER_UNIT
    F = np.asarray(F, np.int64)
    cmm = np.asarray(centre, float) * rs.MM_PER_UNIT
    xy, Z, Fr, _ = rs._project(Vmm, F, view, cmm, half, size)
    zbuf = np.full((size, size), np.inf, np.float32)
    sbuf = np.zeros((size, size), np.float32)
    rs._rasterise(xy, Z, Fr, np.asarray(region, float).reshape(-1, 1), size, zbuf, sbuf)
    filled = np.isfinite(zbuf)
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    idx = np.clip(np.rint(sbuf), 0, len(palette) - 1).astype(np.int64)
    img[filled] = np.asarray(palette, np.float32)[idx[filled]]
    if overlay_edges is not None and len(overlay_edges):
        e = np.asarray(overlay_edges, np.int64)
        _draw_edges(img, xy, Z, zbuf, size, e[:, 0], e[:, 1], overlay_col, width)
    if ss > 1:
        img = img.reshape(px, ss, px, ss, 3).mean(axis=(1, 3))
    return np.clip(img, 0.0, 1.0)


def _panels(out, accepted):
    states, F, _ = bns.load_subject(PROJ / "runtime/outputs/zp_capwitharuco/reconstruction/dense")
    fr, ax, hc = ns.subject_frame(states["closed"]["V"], F, states["closed"]["landmarks"])
    pan = [("original capture", states["closed"]["V"], F)]
    Vn, Fn = mi.load_obj(Path(out) / "head_closed_direct.obj")
    pan.append(("direct shaping", Vn, Fn))
    acc = Path(accepted) / "head_closed_m.obj"
    pan.append(("accepted appearance (regression)", *mi.load_obj(acc)))
    # one stated normal treatment for all three models: vertices are averaged over
    # the SAME 2.5 mm of surface, so the finer capture mesh does not read as
    # mottling that the coarser one cannot show.  No normal map, no texture.
    vns = [_soft_normals(V, Ff, mm=2.5) for _, V, Ff in pan]
    return pan, ax, hc, vns


def _roi_frame(V, ax, hc, curve, view, t_lo=0.72, t_hi=1.12, deg_max=100.0, margin=1.10):
    """Frame a region of interest instead of guessing a centre.

    Blind centre/width guesses put the submental fold in a corner: the window is
    an orthographic box in the camera plane, so it has to be computed from the
    projected extent of the region that must be visible.  The frame is derived
    once, from the original capture, and reused for all three models.
    """
    P = np.asarray(V, float) * 1000.0
    h, th, r, t = nd.axial_mm(P, ax, curve, hc)
    sel = (h / hc > t_lo) & (h / hc < t_hi) & (np.abs(np.degrees(th)) < deg_max)
    Q = P[sel]
    fwd, right, up = rs.basis(view)
    X, Y = Q @ right, Q @ up
    cx, cy = 0.5 * (X.min() + X.max()), 0.5 * (Y.min() + Y.max())
    half = 0.5 * max(X.max() - X.min(), Y.max() - Y.min()) * margin
    return (right * cx + up * cy) / 1000.0, float(half)


def render_all(out: Path, accepted: Path, ref_field=None, tgt=None, subj_field=None,
               partition=None, h_material=None):
    out = Path(out)
    rnd = out / "render"
    rnd.mkdir(parents=True, exist_ok=True)
    pan, ax, hc, vns = _panels(out, accepted)
    c = _centre(ax, hc)
    curve, _ = ns.spine_curve(pan[0][1], pan[0][2], ax, hc)

    # 1. three models, three views, one camera ---------------------------
    views = ("front", "oblique_left", "side_left")
    fig, axs = plt.subplots(len(views), len(pan), figsize=(3.75 * len(pan), 3.95 * len(views)),
                            facecolor=BG)
    for r, vname in enumerate(views):
        for cix, (label, V, F) in enumerate(pan):
            img = rs.render(V, F, VIEWS[vname], c, HALF, px=PX, ss=2,
                            light=rs.light_vector(VIEWS[vname], 22.0, 16.0), vn=vns[cix])
            _show(axs[r][cix], img, label if r == 0 else "", vname)
    fig.suptitle("same camera / scale / light, Gouraud shading, vertex normals averaged over "
                 "2.5 mm identically for all three", color=FG, fontsize=9)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.93, bottom=0.01,
                        wspace=0.015, hspace=0.06)
    fig.savefig(rnd / "compare_views.png", facecolor=BG, dpi=125)
    plt.close(fig)

    # 2. connection close-ups -------------------------------------------
    # a front close-up of a 52 mm radius neck inside a 92 mm window is a wall of
    # surface with no silhouette, so the transition is shown the way it is
    # actually read: in profile.  Two different framings, both computed from the
    # projected extent of an explicit region of interest, because one window
    # cannot show the submental fold and the lateral blend at the same scale:
    #   top    profile  t 0.80-1.08  -- chin, submental fold, upper neck
    #   bottom three-quarter, raking light, t 0.74-0.98 -- anterior wall and the
    #          way it dies into the jaw line
    fig, axs = plt.subplots(2, len(pan), figsize=(3.7 * len(pan), 8.0), facecolor=BG)
    cc, hf = _roi_frame(pan[0][1], ax, hc, curve, VIEWS["side_left"],
                        t_lo=0.80, t_hi=1.08, margin=1.06)
    co, ho = _roi_frame(pan[0][1], ax, hc, curve, VIEWS["oblique_left"],
                        t_lo=0.74, t_hi=0.98, deg_max=80.0, margin=1.05)
    for cix, (label, V, F) in enumerate(pan):
        img = rs.render(V, F, VIEWS["side_left"], cc, hf, px=PX, ss=2,
                        light=rs.light_vector(VIEWS["side_left"], 18.0, 14.0), vn=vns[cix])
        _show(axs[0][cix], img, label,
              f"profile: chin, submental fold, upper neck, {2*hf:.0f} mm window")
        img = rs.render(V, F, VIEWS["oblique_left"], co, ho, px=PX, ss=2,
                        light=rs.light_vector(VIEWS["oblique_left"], -55.0, 6.0), vn=vns[cix])
        _show(axs[1][cix], img, label,
              f"three-quarter, grazing light, {2*ho:.0f} mm window")
    fig.suptitle("connection close-ups: the jaw-to-neck transition read off the silhouette",
                 color=FG, fontsize=9)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.925, bottom=0.045,
                        wspace=0.015, hspace=0.14)
    fig.savefig(rnd / "compare_closeups.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 3. smooth / flat / wireframe on the delivered mesh ------------------
    label, Vn, Fn = pan[1]
    vn1 = vns[1]
    fig, axs = plt.subplots(3, 2, figsize=(7.6, 11.0), facecolor=BG)
    for col, vname in enumerate(("front", "oblique_left")):
        img = rs.render(Vn, Fn, VIEWS[vname], c, HALF, px=PX, ss=2,
                        light=rs.light_vector(VIEWS[vname], 14.0, 16.0), vn=vn1)
        _show(axs[0][col], img, "smooth (Gouraud)", vname)
        img = _render_flat(Vn, Fn, VIEWS[vname], c, HALF, PX)
        _show(axs[1][col], img, "per-face flat", vname)
        img = render_wire(Vn, Fn, VIEWS[vname], c, HALF, px=620, ss=2)
        _show(axs[2][col], img, "wireframe (hidden lines removed)", vname)
    fig.suptitle("delivered mesh: shading and tessellation, no texture or normal map",
                 color=FG, fontsize=9)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.955, bottom=0.01,
                        wspace=0.015, hspace=0.06)
    fig.savefig(rnd / "shading_and_wireframe.png", facecolor=BG, dpi=120)
    plt.close(fig)

    # 4. wireframe of the original vs the delivered (density by eye) ------
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.1), facecolor=BG)
    for axp, (lbl, V, F) in zip(axs, (pan[0], pan[1])):
        img = render_wire(V, F, VIEWS["oblique_left"], c, 62.0, px=620, ss=2)
        _show(axp, img, lbl, "wireframe, 124 x 124 mm")
    fig.suptitle("tessellation: original capture vs direct result (same camera)",
                 color=FG, fontsize=9)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.01, wspace=0.015)
    fig.savefig(rnd / "wire_original_vs_direct.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 5. the measured prominence itself, coloured ------------------------
    # Each mesh is analysed with the same ray field and the same definition the
    # descriptors use (fine radius minus the smooth axial background), so the
    # three panels show the SAME quantity on the same colour scale.  No radius is
    # ever compared across two different meshes, and no lighting is involved.
    fig = plt.figure(figsize=(3.75 * len(pan), 4.55), facecolor=BG)
    peaks = []
    for cix, (label, V, F) in enumerate(pan):
        an = ns.analyse_surface(V, F, ax, hc, curve=curve, structure_sigma=13.0,
                                fine_sigma=1.2, leak_mask=True)
        hh, tth, rr, _ = nd.axial_mm(np.asarray(V, float) * 1000.0, ax, curve, hc)
        relf = ns.Field2D(np.asarray(an["hs_mm"], float), np.asarray(an["theta"], float),
                          np.where(np.isfinite(an["relief"]), an["relief"], 0.0))
        val = np.asarray(relf(hh, tth), float)
        tt = hh / hc
        keep = (tt > 0.42) & (tt < 0.90) & (np.abs(np.degrees(tth)) < 100)
        peak = float(np.percentile(val[keep & (np.abs(np.degrees(tth)) < 22) &
                                           (tt > 0.62) & (tt < 0.84)], 99))
        peaks.append(peak)
        axp = fig.add_subplot(1, len(pan), cix + 1)
        img = render_scalar(V, F, VIEWS["oblique_left"], c, 64.0, val, vmin=-2.0, vmax=5.0,
                            keep=keep)
        _show(axp, img, label, f"relief p99 on the prominence {peak:+.2f} mm")
        if cix == len(pan) - 1:
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-2.0, 5.0))
            cb = fig.colorbar(sm, ax=axp, fraction=0.045, pad=0.01)
            cb.set_label("mm", color=FG)
            cb.ax.tick_params(colors="#999")
    fig.suptitle("the prominence as a measured field: surface radius minus its smooth axial\n"
                 "background, computed identically on each mesh; grey = undefined radial coord",
                 color=FG, fontsize=8.5)
    fig.subplots_adjust(left=0.004, right=0.96, top=0.84, bottom=0.01, wspace=0.015)
    fig.savefig(rnd / "prominence_map.png", facecolor=BG, dpi=130)
    plt.close(fig)
    print("  prominence map p99: original %+.2f / direct %+.2f / accepted %+.2f mm"
          % (peaks[0], peaks[1], peaks[2]))

    # 6. sections about each neck's own axis -----------------------------
    RV, RF, _, rax, rhc, _ = ns.reference_frame(
        "D:/project_neck/runtime/v8_tail_smoke/experiment_202609-200612/final/identity_rest.obj"
        if False else
        "D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
    rcurve, _ = ns.spine_curve(RV, RF, rax, rhc)
    levels = (0.45, 0.60, 0.75, 0.85)
    th = np.linspace(-np.pi, np.pi, 721)
    fig, axs = plt.subplots(2, len(levels), figsize=(3.05 * len(levels), 6.5), facecolor=BG,
                            gridspec_kw={"height_ratios": [1.0, 0.62]})
    curves = [("original", pan[0][1], pan[0][2], ax, curve, hc, "#9a9a9a", 1.0),
              ("direct", pan[1][1], pan[1][2], ax, curve, hc, "#ffd166", 1.8),
              ("accepted", pan[2][1], pan[2][2], ax, curve, hc, "#7ee0a0", 1.1),
              ("reference", RV, RF, rax, rcurve, rhc, "#4aa3ff", 1.1)]
    for k, lv in enumerate(levels):
        for row in (0, 1):
            axs[row][k].set_facecolor(BG)
        for lbl, V, F, axm, crv, hcm, col, lw in curves:
            ca, cb = crv(lv * hcm)
            r = ns.ray_radius(ns.section_segments(V, F, axm, lv * hcm), th, (ca, cb))
            axs[0][k].plot(r * np.sin(th), r * np.cos(th), color=col, lw=lw, label=lbl)
            axs[1][k].plot(np.degrees(th), r - np.nanmedian(r), color=col, lw=lw)
        axs[0][k].set_xlim(-78, 78)
        axs[0][k].set_ylim(-78, 78)
        axs[0][k].set_aspect("equal")
        axs[0][k].axhline(0, color="#3a3a40", lw=0.5)
        axs[0][k].axvline(0, color="#3a3a40", lw=0.5)
        axs[0][k].set_title(f"section t = {lv:.2f}  ({lv * hc:.0f} mm)", color=FG, fontsize=8)
        axs[0][k].tick_params(colors="#888", labelsize=6)
        if k == 0:
            axs[0][k].legend(fontsize=6.0, facecolor=BG, edgecolor="#555", labelcolor=FG,
                             loc="lower left")
        axs[1][k].axhline(0, color="#3a3a40", lw=0.5)
        axs[1][k].set_xlim(-120, 120)
        axs[1][k].set_ylim(-6.5, 6.5)
        axs[1][k].grid(alpha=0.12)
        axs[1][k].tick_params(colors="#888", labelsize=6)
        axs[1][k].set_xlabel("theta (deg), 0 = anterior", color=FG, fontsize=7)
    fig.suptitle("horizontal sections about each neck's own axis (top) and relief about the "
                 "ring median (bottom)", color=FG, fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.075, wspace=0.15, hspace=0.28)
    fig.savefig(rnd / "sections.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 6. measured anterior profile + the continuous target field ---------
    if tgt is not None:
        lv = np.arange(0.34, 0.93, 0.01)
        th1 = np.linspace(-np.pi, np.pi, 721)
        i0 = len(th1) // 2                       # theta = 0, anterior
        prof = {}
        # ray-cast the surface at each height: sampling nearest *vertices* cannot
        # build this curve, because the delivered neck spacing is 6-9 mm and a
        # +-0.5 mm window around a given height simply contains no vertex
        for lbl, V, Fm, axm, crv, hcm in (("original", pan[0][1], pan[0][2], ax, curve, hc),
                                          ("direct", pan[1][1], pan[1][2], ax, curve, hc),
                                          ("accepted", pan[2][1], pan[2][2], ax, curve, hc),
                                          ("reference", RV, RF, rax, rcurve, rhc)):
            rr = []
            for x in lv:
                hm = float(x * hcm)
                ca, cb = crv(hm)
                segs = ns.section_segments(V, Fm, axm, hm)
                rad = ns.ray_radius(segs, th1, (ca, cb))
                rr.append(float(rad[i0]))
            prof[lbl] = np.asarray(rr)
        fig, axs = plt.subplots(1, 3, figsize=(16.4, 4.6), facecolor=BG)
        for axp in axs:
            axp.set_facecolor(BG)
        for lbl, col, lw in (("original", "#9a9a9a", 1.2), ("direct", "#ffd166", 2.0),
                             ("accepted", "#7ee0a0", 1.2), ("reference", "#4aa3ff", 1.3)):
            axs[0].plot(lv, prof[lbl], color=col, lw=lw, label=lbl)
            if lbl in ("direct", "accepted"):
                axs[1].plot(lv, prof[lbl] - prof["original"], color=col, lw=lw, label=lbl)
        axs[0].set_xlabel("t = h / h_chin", color=FG)
        axs[0].set_ylabel("surface radius at theta = 0 (mm)", color=FG)
        axs[0].set_title("anterior radius measured on each mesh (t <= 0.92, "
                         "so the chin is out of frame)", color=FG, fontsize=8.5)
        axs[0].set_ylim(55.0, 66.0)
        axs[1].axhline(0.0, color="#3a3a40", lw=0.8)
        axs[1].set_xlabel("t = h / h_chin", color=FG)
        axs[1].set_ylabel("radius change vs original (mm)", color=FG)
        axs[1].set_title("what the stage actually adds, against the capture", color=FG,
                         fontsize=8.5)
        axs[1].legend(fontsize=6.5, facecolor=BG, edgecolor="#555", labelcolor=FG)
        for axp in axs[:2]:
            axp.tick_params(colors="#999")
            axp.grid(alpha=0.15)
        axs[0].legend(fontsize=6.5, facecolor=BG, edgecolor="#555", labelcolor=FG,
                      loc="lower left")
        rel = np.asarray(tgt["applied"], float)
        im = axs[2].imshow(np.where(np.isfinite(rel), rel, 0.0), aspect="auto", origin="lower",
                           extent=[np.degrees(tgt["theta"])[0], np.degrees(tgt["theta"])[-1],
                                   tgt["t"][0], tgt["t"][-1]], cmap="coolwarm", vmin=-4.5, vmax=4.5)
        axs[2].set_xlim(-100, 100)
        axs[2].set_ylim(0.32, 0.96)
        axs[2].set_xlabel("theta (deg), 0 = anterior", color=FG)
        axs[2].set_ylabel("t", color=FG)
        axs[2].set_title("relief added to the subject background (the continuous target)",
                         color=FG, fontsize=8.5)
        axs[2].tick_params(colors="#999")
        cb = fig.colorbar(im, ax=axs[2])
        cb.set_label("mm", color=FG)
        cb.ax.tick_params(colors="#999")
        fig.subplots_adjust(left=0.045, right=0.985, top=0.89, bottom=0.11, wspace=0.22)
        fig.savefig(rnd / "profiles_and_field.png", facecolor=BG, dpi=130)
        plt.close(fig)

    # 8. material partition, the two interfaces, and the density zones -----
    if partition is not None:
        npz = np.load(out / "stage_direct.npz")
        reg = npz["region"]
        Fc = npz["F"]
        Vc = npz["V_closed"]
        E = mi.edge_records(Fc)[0]
        ab = np.flatnonzero(((reg[E[:, 0]] == 0) & (reg[E[:, 1]] == 1)) |
                            ((reg[E[:, 0]] == 1) & (reg[E[:, 1]] == 0)))
        bc = np.flatnonzero(((reg[E[:, 0]] == 1) & (reg[E[:, 1]] == 2)) |
                            ((reg[E[:, 0]] == 2) & (reg[E[:, 1]] == 1)))
        # the obj and the npz carry the same vertex order (both are the compacted set)
        dn = npz["density_zone"]
        hh, tth, rr, tt = nd.axial_mm(np.asarray(Vc, float) * 1000.0, ax, curve, hc)
        keep = (tt > 0.20) & (tt < 1.10) & (np.abs(np.degrees(tth)) < 130)
        fig, axs = plt.subplots(2, 3, figsize=(11.4, 8.0), facecolor=BG)
        for col, vname in enumerate(("front", "oblique_left", "side_left")):
            cc, hf = _roi_frame(Vc, ax, hc, curve, VIEWS[vname],
                                t_lo=0.30, t_hi=1.10, deg_max=75.0, margin=1.03)
            img = render_region(Vc, Fc, VIEWS[vname], cc, hf, reg, px=430, ss=2)
            _show(axs[0][col], img, "A / B / C material partition", vname)
            img = render_region(Vc, Fc, VIEWS[vname], cc, hf, reg, px=430, ss=2,
                                overlay_edges=E[np.concatenate([ab, bc])],
                                overlay_col=(1.0, 1.0, 1.0), width=2)
            _show(axs[1][col], img, f"A-B {len(ab)} edges / B-C {len(bc)} edges (white)",
                  vname)
        patch = [Patch(color=REGION_PALETTE[k], label=REGION_NAMES[k])
                 for k in range(3)]
        axs[0][0].legend(handles=patch, fontsize=6.0, facecolor=BG, edgecolor="#555",
                         labelcolor=FG, loc="lower left")
        fig.suptitle("material partition in the neutral frame: regional, not positional -- the "
                     "same indices carry the same regions in both states\n"
                     "B is a 16.7 mm band, 126 v over 19.1 cm2, solved in 3-D between the "
                     "fixed head and the shaped neck; A-B and B-C are the lines the bridge "
                     "has to connect", color=FG, fontsize=8.5)
        fig.subplots_adjust(left=0.004, right=0.996, top=0.875, bottom=0.01,
                            wspace=0.015, hspace=0.12)
        fig.savefig(rnd / "material_partition.png", facecolor=BG, dpi=125)
        plt.close(fig)

        # density zones on the same mesh, with the frozen per-zone budgets
        fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.9), facecolor=BG)
        for col, vname in enumerate(("front", "oblique_left", "side_left")):
            cc, hf = _roi_frame(Vc, ax, hc, curve, VIEWS[vname],
                                t_lo=0.30, t_hi=1.10, deg_max=75.0, margin=1.03)
            img = render_region(Vc, Fc, VIEWS[vname], cc, hf, dn, px=430, ss=2)
            _show(axs[col], img, "density zone", vname)
        patch = [Patch(color=REGION_PALETTE[k],
                                          label=f"z{k} {REGION_NAMES[k]}")
                 for k in range(4)]
        axs[0].legend(handles=patch, fontsize=6.0, facecolor=BG, edgecolor="#555",
                      labelcolor=FG, loc="lower left")
        fig.suptitle("density zone (0 anterior 7.0 mm / 1 structure 6.1 mm / "
                     "2 sides-back 7.9 mm / 3 hand-over = keep the head's own)",
                     color=FG, fontsize=8.5)
        fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.01, wspace=0.015)
        fig.savefig(rnd / "density_zones.png", facecolor=BG, dpi=125)
        plt.close(fig)

        # 9. what the stage moved, and how far the normals turned -----------
        from head_rgbd import neck_prominence as npn
        dcl, dop = npz["d_closed"], npz["d_open"]
        drc, dro = npz["d_rad_closed"], npz["d_rad_open"]
        n_in = npn.vertex_normals(npz["V_closed_input"], Fc)
        n_out = npn.vertex_normals(Vc, Fc)
        ang = np.degrees(np.arccos(np.clip(np.einsum("nk,nk->n", n_in, n_out), -1, 1)))
        n_in_o = npn.vertex_normals(npz["V_open_input"], Fc)
        n_out_o = npn.vertex_normals(npz["V_open"], Fc)
        ang_o = np.degrees(np.arccos(np.clip(np.einsum("nk,nk->n", n_in_o, n_out_o), -1, 1)))
        mag = np.linalg.norm(dcl, axis=1)
        mag_o = np.linalg.norm(dop, axis=1)
        B = reg == 1
        stat = (f"closed |d| p95 {np.percentile(mag[B], 95):.2f} mm in B / "
                f"{np.percentile(mag[reg == 2], 95):.2f} in C; normal turn p95 "
                f"{np.percentile(ang[B], 95):.1f} deg in B")
        fig, axs = plt.subplots(2, 3, figsize=(11.2, 7.0), facecolor=BG)
        cc, hf = _roi_frame(Vc, ax, hc, curve, VIEWS["oblique_left"],
                            t_lo=0.36, t_hi=1.02, deg_max=70.0, margin=1.05)
        for row, (V_, mg, rdl, an, tag) in enumerate((
                (Vc, mag, drc, ang, "closed"), (npz["V_open"], mag_o, dro, ang_o, "open"))):
            for col, (vec, lo, hi, ttl, unit) in enumerate((
                    (mg, 0.0, 5.0, "|displacement| vs the input surface", "mm"),
                    (rdl, -4.5, 4.5, "signed radial component", "mm"),
                    (an, 0.0, 25.0, "vertex-normal turn vs the input", "deg"))):
                img = render_scalar(V_, Fc, VIEWS["oblique_left"], cc, hf, vec, vmin=lo,
                                    vmax=hi, keep=keep, px=430, ss=2)
                _show(axs[row][col], img, f"{tag}: {ttl}", unit)
                sm = plt.cm.ScalarMappable(cmap="coolwarm",
                                           norm=plt.Normalize(lo, hi))
                cb = fig.colorbar(sm, ax=axs[row][col], fraction=0.04, pad=0.01)
                cb.set_label(unit, color=FG)
                cb.ax.tick_params(colors="#999", labelsize=6)
        fig.suptitle("displacement and normal rotation of the delivered surface "
                     f"(grey = outside the density band)\n{stat}", color=FG, fontsize=8.5)
        fig.subplots_adjust(left=0.004, right=0.98, top=0.87, bottom=0.01,
                            wspace=0.06, hspace=0.13)
        fig.savefig(rnd / "displacement_and_normals.png", facecolor=BG, dpi=125)
        plt.close(fig)

        # 10. long sections at several angles ------------------------------
        lv = np.arange(0.34, 0.96, 0.01)
        angs = (-180, -90, -45, 0, 45, 90)
        fig, axs = plt.subplots(2, 3, figsize=(11.4, 6.4), facecolor=BG)
        prof = {}
        for lbl, V_, Fm, axm, crv, hcm, col, lw in (
                ("original", pan[0][1], pan[0][2], ax, curve, hc, "#9a9a9a", 1.3),
                ("direct", Vc, Fc, ax, curve, hc, "#ffd166", 1.9),
                ("accepted", pan[2][1], pan[2][2], ax, curve, hc, "#7ee0a0", 1.1)):
            band = []
            for x in lv:
                hm = float(x * hcm)
                ca, cb = crv(hm)
                segs = ns.section_segments(V_, Fm, axm, hm)
                thq = np.radians(np.asarray(angs, float))
                band.append(ns.ray_radius(segs, thq, (ca, cb)))
            prof[lbl] = np.asarray(band, float)
        for k, a_deg in enumerate(angs):
            axp = axs[k // 3][k % 3]
            axp.set_facecolor(BG)
            for lbl, col, lw in (("original", "#9a9a9a", 1.3), ("direct", "#ffd166", 1.9),
                                 ("accepted", "#7ee0a0", 1.1)):
                axp.plot(lv, prof[lbl][:, k], color=col, lw=lw, label=lbl)
            axp.set_title(f"theta = {a_deg} deg", color=FG, fontsize=8)
            axp.tick_params(colors="#999", labelsize=6)
            axp.grid(alpha=0.15)
            axp.set_xlabel("t = h / h_chin", color=FG, fontsize=7)
            if k % 3 == 0:
                axp.set_ylabel("surface radius (mm)", color=FG, fontsize=7)
            if k == 0:
                axp.legend(fontsize=6.0, facecolor=BG, edgecolor="#555", labelcolor=FG)
        fig.suptitle("long sections at six angles about the subject's own axis: the "
                     "anterior change and the untouched sides in the same frame",
                     color=FG, fontsize=9)
        fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.075,
                            wspace=0.20, hspace=0.32)
        fig.savefig(rnd / "sections_multi_angle.png", facecolor=BG, dpi=125)
        plt.close(fig)

    # 7. both states ------------------------------------------------------
    Vo, Fo = mi.load_obj(out / "head_open_direct.obj")
    fig, axs = _figure(2, 3.8, 4.0)
    for axp, (lbl, V, F) in zip(axs, (("closed", Vn, Fn), ("open (shared topology)", Vo, Fo))):
        img = rs.render(V, F, VIEWS["oblique_left"], c, HALF, px=PX, ss=2,
                        light=rs.light_vector(VIEWS["oblique_left"], 0.0, 18.0))
        _show(axp, img, lbl)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.02)
    fig.savefig(rnd / "states_closed_open.png", facecolor=BG, dpi=125)
    plt.close(fig)
    return rnd


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/zp_capwitharuco_neck_direct")
    ap.add_argument("--accepted", default="runtime/outputs/zp_capwitharuco_neck_stage_v2")
    a = ap.parse_args()
    d = np.load(Path(a.out) / "stage_direct.npz")
    print(render_all(Path(a.out), Path(a.accepted),
                     tgt={"applied": d["applied_closed"], "t": d["t"], "theta": d["theta"]},
                     partition=d["region"], h_material=d["h_material"]))
