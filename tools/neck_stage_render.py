"""Same-camera rendering for the neck-stage acceptance.

Every panel uses the same orthographic camera, framing, light direction and
shading model, so the comparison shows geometry rather than presentation.  Each
mesh is framed on its own axis at ``t = 0.70`` with the same half extent in mm,
which is what makes a subject/reference comparison meaningful.

Shading is Gouraud (per-vertex Lambert interpolated inside each triangle, real
z-buffer) rather than one flat colour per triangle: a per-face constant turns
even a 1.7 mm mesh into a visible mosaic and would be read as faceting that the
surface does not have, while a coarse mesh reads as smooth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi            # noqa: E402
from head_rgbd import neck_volume as nv               # noqa: E402
from head_rgbd import neck_stage as ns                # noqa: E402
from head_rgbd import raster as rs                    # noqa: E402
from head_rgbd import neck_prominence as npn          # noqa: E402

BG = "#1b1b1d"
FG = "#e8e8e8"
HALF = 76.0            # mm, half width of the framed window
PX = 460
VIEWS = {
    "front": np.array([0.0, 0.0, -1.0]),
    "oblique_left": np.array([-0.57, 0.0, -0.82]),
    "oblique_right": np.array([0.57, 0.0, -0.82]),
    "side_left": np.array([-1.0, 0.0, 0.0]),
    "side_right": np.array([1.0, 0.0, 0.0]),
    "submental": np.array([0.0, -0.72, -0.69]),
}


def _centre(ax, h_chin, t=0.70):
    return (ax["centre_mm"] + t * h_chin * ax["axis"]) / 1000.0


def _label(ax, text, sub=""):
    ax.set_title(text, color=FG, fontsize=8.5, pad=3)
    if sub:
        ax.text(0.5, -0.035, sub, color="#9a9a9a", fontsize=7, ha="center",
                transform=ax.transAxes)


def _figure(ncols, width=4.0, height=4.2):
    fig, axs = plt.subplots(1, ncols, figsize=(width * ncols, height), facecolor=BG)
    return fig, (axs if ncols > 1 else [axs])


def _show(ax, img, label, sub=""):
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")
    ax.set_facecolor(BG)
    _label(ax, label, sub)


def _cylindrical(V, ax, curve):
    """Radius and theta of every vertex in the spine-centred section frame."""
    h, a, b = nv.axial_coords(V, ax)
    ca, cb = curve(h)
    da, db = a - ca, b - cb
    return h, np.hypot(da, db), np.degrees(np.arctan2(da, db))


def radial_profile(V, ax, curve, h_chin, levels, half_theta_deg=2.0, half_h_mm=1.2):
    """Measured surface radius near theta = 0 on the design frame.

    This is the *same* quantity as the designed field ``R(theta, h)``, read back
    from the exported mesh, so design and delivered surface can share one axis
    instead of being plotted in two different coordinate systems.
    """
    h, r, th = _cylindrical(V, ax, curve)
    out = []
    for lv in levels:
        m = (np.abs(th) <= half_theta_deg) & (np.abs(h - lv * h_chin) <= half_h_mm)
        out.append(float(np.max(r[m])) if m.any() else np.nan)
    return np.asarray(out)


def _band_profile(V, ax, curve, h_chin, levels, angle_deg):
    h, r, th = _cylindrical(V, ax, curve)
    out = []
    for lv in levels:
        m = (np.abs(th - angle_deg) <= 3.0) & (np.abs(h - lv * h_chin) <= 1.2)
        out.append(float(np.median(r[m])) if m.any() else np.nan)
    return np.asarray(out)


def render_all(out: Path, closed: Path, open: Path, reference: Path, compare,
               ax, h_chin, rax, rh_chin, curve_c, rcurve,
               tgt, subj_field, ref_field, base_V=None):
    rnd = out / "render"
    rnd.mkdir(parents=True, exist_ok=True)

    Vn, Fn = mi.load_obj(closed)
    Vo, Fo = mi.load_obj(open)
    RV, RF = mi.load_obj(reference)
    c_new = _centre(ax, h_chin)
    c_ref = _centre(rax, rh_chin)

    panels = [("new stage", Vn, Fn)]
    if compare is not None:
        prev = Path(compare) / "head_closed_m.obj"
        if prev.exists():
            PV, PF = mi.load_obj(prev)
            panels.append(("previous result", PV, PF))
    panels.append(("reference", RV - (c_ref - c_new), RF))

    # 1. same-camera comparison, one file per view -------------------------
    for name, view in VIEWS.items():
        fig, axs = _figure(len(panels), 3.9, 4.15)
        for axp, (label, V, F) in zip(axs, panels):
            img = rs.render(V, F, view, c_new, HALF, px=PX, ss=2,
                            light=rs.light_vector(view, 25.0, 18.0))
            _show(axp, img, label)
        fig.suptitle(f"{name}  (orthographic, same camera / scale / light, Gouraud)",
                     color=FG, fontsize=9)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.02)
        fig.savefig(rnd / f"compare_{name}.png", facecolor=BG, dpi=125)
        plt.close(fig)

    # 2. prominence close-up: the region the stage is asked to create ------
    fig, axs = _figure(len(panels), 3.7, 3.9)
    for axp, (label, V, F) in zip(axs, panels):
        cc = _centre(ax, h_chin, 0.72)
        img = rs.render(V, F, VIEWS["front"], cc, 36.0, px=PX, ss=2,
                        light=rs.light_vector(VIEWS["front"], -32.0, 10.0))
        _show(axp, img, label, "front, grazing light, 72 x 72 mm")
    fig.suptitle("anterior neck close-up (t = 0.72 +- 0.37)", color=FG, fontsize=9)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.02)
    fig.savefig(rnd / "compare_prominence_zoom.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 3. grazing light on two views ---------------------------------------
    fig, axs = plt.subplots(2, len(panels), figsize=(3.9 * len(panels), 8.2),
                            facecolor=BG)
    for col, (label, V, F) in enumerate(panels):
        for row, (view, az, el) in enumerate((("front", -38.0, 9.0),
                                              ("oblique_left", -30.0, 9.0))):
            img = rs.render(V, F, VIEWS[view], c_new, HALF, px=PX, ss=2,
                            light=rs.light_vector(VIEWS[view], az, el))
            _show(axs[row][col], img, label, f"grazing light, {view}, elev {el:.0f} deg")
    fig.suptitle("grazing light: surface relief only", color=FG, fontsize=9)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01,
                        wspace=0.02, hspace=0.08)
    fig.savefig(rnd / "compare_grazing.png", facecolor=BG, dpi=125)
    plt.close(fig)

    # 4. structure comparison under grazing light, identical for all models
    fig, axs = plt.subplots(2, len(panels), figsize=(3.9 * len(panels), 8.2),
                            facecolor=BG)
    for col, (label, V, F) in enumerate(panels):
        vn = _soft_normals(V, F, 2.5)
        for row, view in enumerate(("front", "oblique_left")):
            img = rs.render(V, F, VIEWS[view], c_new, HALF, px=PX, ss=2,
                            light=rs.light_vector(VIEWS[view], -24.0, 8.0), vn=vn,
                            ambient=0.10, diffuse=0.90)
            _show(axs[row][col], img, label, f"{view}, grazing light")
    fig.suptitle("same camera, scale and light; normals averaged over 2.5 mm in every panel "
                 "(removes tessellation mottling, keeps structure)", color=FG, fontsize=8.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01,
                        wspace=0.02, hspace=0.08)
    fig.savefig(rnd / "compare_structure.png", facecolor=BG, dpi=125)
    plt.close(fig)

    # 5. flat vs smooth shading on the new result -------------------------
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 8.6), facecolor=BG)
    for row, smooth in enumerate((True, False)):
        for col, view in enumerate((VIEWS["front"], VIEWS["oblique_left"])):
            axp = axs[row][col]
            axp.set_facecolor(BG)
            if smooth:
                img = rs.render(Vn, Fn, view, c_new, HALF, px=PX, ss=2,
                                light=rs.light_vector(view, 0.0, 18.0))
            else:
                img = _render_flat(Vn, Fn, view, c_new, HALF, PX)
            _show(axp, img, ("smooth (Gouraud)" if smooth else "per-face flat")
                  + " / " + list(VIEWS)[col])
    fig.suptitle("new result: flat vs smooth shading", color=FG, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.02, hspace=0.08)
    fig.savefig(rnd / "shading_flat_vs_smooth.png", facecolor=BG, dpi=125)
    plt.close(fig)

    # 6. submental junction close-up --------------------------------------
    fig, axs = _figure(len(panels), 3.6, 3.6)
    for axp, (label, V, F) in zip(axs, panels):
        cc = c_new - np.array([0.0, 0.030, 0.0])
        img = rs.render(V, F, VIEWS["submental"], cc, 44.0, px=PX, ss=2,
                        light=rs.light_vector(VIEWS["submental"], -25.0, 15.0))
        _show(axp, img, label)
    fig.suptitle("submental junction close-up", color=FG, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.02)
    fig.savefig(rnd / "compare_submental_closeup.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 7. cross-sections and the ring profile at the prominence height -------
    # Every section is taken about its own neck axis (the spine point at that
    # height), never about the bottom-cut centre: the two meshes have different
    # cut planes, and comparing about a shared origin would compare offsets
    # instead of shapes.
    levels = (0.45, 0.60, 0.75, 0.85)
    th = np.linspace(-np.pi, np.pi, 721)
    fig, axs = plt.subplots(2, len(levels), figsize=(3.1 * len(levels), 6.6),
                            facecolor=BG, gridspec_kw={"height_ratios": [1.0, 0.62]})
    curves = [("final", Vn, Fn, ax, curve_c, h_chin, "#e8b23a", 1.8)]
    if base_V is not None:
        curves.insert(0, ("base", base_V, Fn, ax, curve_c, h_chin, "#9a9a9a", 1.0))
    curves.append(("reference", RV, RF, rax, rcurve, rh_chin, "#4aa3ff", 1.1))
    for k, lv in enumerate(levels):
        axp = axs[0, k]
        axp.set_facecolor(BG)
        for label, V, F, axm, crv, hcm, col, lw in curves:
            ca, cb = crv(lv * hcm)
            r = ns.ray_radius(ns.section_segments(V, F, axm, lv * hcm), th, (ca, cb))
            axp.plot(r * np.sin(th), r * np.cos(th), color=col, lw=lw, alpha=0.95,
                     label=label)
        axp.set_xlim(-78, 78)
        axp.set_ylim(-78, 78)
        axp.set_aspect("equal")
        axp.axhline(0, color="#3a3a40", lw=0.5)
        axp.axvline(0, color="#3a3a40", lw=0.5)
        axp.set_title(f"section t = {lv:.2f}  ({lv * h_chin:.0f} mm)", color=FG, fontsize=8)
        axp.tick_params(colors="#888", labelsize=6)
        if k == 0:
            axp.set_ylabel("anterior / posterior (mm)", color=FG, fontsize=7)
            axp.legend(fontsize=6.2, facecolor=BG, edgecolor="#555", labelcolor=FG,
                       loc="lower left")
        # lower row: the same ring as a profile about its own mean radius, which
        # is what "structure" means here and is unaffected by neck size
        axq = axs[1, k]
        axq.set_facecolor(BG)
        for label, V, F, axm, crv, hcm, col, lw in curves:
            ca, cb = crv(lv * hcm)
            r = ns.ray_radius(ns.section_segments(V, F, axm, lv * hcm), th, (ca, cb))
            axq.plot(np.degrees(th), r - np.nanmedian(r), color=col, lw=lw)
        axq.axhline(0, color="#3a3a40", lw=0.5)
        axq.set_xlim(-120, 120)
        axq.set_ylim(-4.5, 4.5)
        axq.axvline(0, color="#3a3a40", lw=0.5, ls=":")
        axq.tick_params(colors="#888", labelsize=6)
        axq.grid(alpha=0.12)
        if k == 0:
            axq.set_ylabel("r - median(r)  (mm)", color=FG, fontsize=7)
        axq.set_xlabel("theta (deg), 0 = anterior", color=FG, fontsize=7)
    fig.suptitle("horizontal sections about each neck's own axis (top) and the same rings "
                 "as relief about their own median radius (bottom)", color=FG, fontsize=9)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.075, wspace=0.16,
                        hspace=0.28)
    fig.savefig(rnd / "sections.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 8. measured radial profiles + the applied relief ---------------------
    lv = np.arange(0.34, 1.00, 0.01)
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8), facecolor=BG)
    for axp in axs:
        axp.set_facecolor(BG)
    series = [("new stage", Vn, ax, curve_c, h_chin, "#ffd166", 2.0),
              ("reference", RV, rax, rcurve, rh_chin, "#4aa3ff", 1.5)]
    if base_V is not None:
        series.insert(0, ("base (before stage)", base_V, ax, curve_c, h_chin, "#9a9a9a", 1.3))
    for label, V, axm, crv, hcm, col, lw in series:
        axs[0].plot(lv, radial_profile(V, axm, crv, hcm, lv), "-", color=col, lw=lw,
                    label=f"{label}: theta=0")
        for ang, ls in ((30.0, "--"), (-30.0, ":")):
            axs[0].plot(lv, _band_profile(V, axm, crv, hcm, lv, ang), ls,
                        color=col, lw=0.9, alpha=0.55)
    axs[0].set_xlabel("t = h / h_chin", color=FG)
    axs[0].set_ylabel("surface radius r (mm), spine-centred", color=FG)
    axs[0].set_title("radius measured on the mesh (theta=0 solid, +-30 deg thin, dotted)",
                     color=FG, fontsize=9)
    axs[0].set_ylim(45.0, 80.0)
    axs[0].tick_params(colors="#999")
    axs[0].grid(alpha=0.15)
    for s in axs[0].spines.values():
        s.set_color("#555")
    axs[0].legend(fontsize=6.4, facecolor=BG, edgecolor="#555", labelcolor=FG)

    relief = np.asarray(tgt["applied"], float)
    relief = np.where(np.isfinite(relief), relief, 0.0)
    im = axs[1].imshow(relief, aspect="auto", origin="lower",
                       extent=[np.degrees(tgt["theta"])[0], np.degrees(tgt["theta"])[-1],
                               tgt["t"][0], tgt["t"][-1]],
                       cmap="coolwarm", vmin=-4, vmax=4)
    axs[1].set_xlim(-120, 120)
    axs[1].set_ylim(0.30, 1.02)
    axs[1].set_xlabel("theta (deg, 0 = anterior)", color=FG)
    axs[1].set_ylabel("t", color=FG)
    axs[1].set_title(f"relief added to the subject background (gain {tgt.get('gain', 0):.2f})",
                     color=FG, fontsize=9)
    axs[1].tick_params(colors="#999")
    cb = fig.colorbar(im, ax=axs[1])
    cb.set_label("mm", color=FG)
    cb.ax.tick_params(colors="#999")
    fig.subplots_adjust(left=0.055, right=0.98, top=0.90, bottom=0.10, wspace=0.20)
    fig.savefig(rnd / "profiles_and_field.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 9. the three relief fields side by side -----------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9), facecolor=BG)
    for axp, (rel, tt, title) in zip(axs, (
            (subj_field["relief"], subj_field["t"], "subject relief (discarded)"),
            (ref_field["relief"], ref_field["t"], "reference relief (transferred)"),
            (relief, tgt["t"], "relief applied to the subject"))):
        axp.set_facecolor(BG)
        m = axp.imshow(rel, aspect="auto", origin="lower", cmap="coolwarm",
                       vmin=-4, vmax=4,
                       extent=[np.degrees(tgt["theta"])[0], np.degrees(tgt["theta"])[-1],
                               tt[0], tt[-1]])
        axp.set_xlim(-120, 120)
        axp.set_ylim(0.20, 1.00)
        axp.set_title(title, color=FG, fontsize=8.5)
        axp.set_xlabel("theta (deg)", color=FG, fontsize=7)
        axp.tick_params(colors="#999", labelsize=6)
    fig.colorbar(m, ax=axs, label="relief (mm)")
    fig.subplots_adjust(left=0.03, right=0.93, top=0.88, bottom=0.14, wspace=0.12)
    fig.savefig(rnd / "relief_fields.png", facecolor=BG, dpi=130)
    plt.close(fig)

    # 9b. base -> final on the same camera -------------------------------
    if base_V is not None:
        fig, axs = _figure(2, 3.9, 4.15)
        for axp, (label, V) in zip(axs, (("base (faired, before the design)", base_V),
                                         ("final (design applied)", Vn))):
            for vi, (view, az, el) in enumerate((("oblique_left", 25.0, 18.0),)):
                img = rs.render(V, Fn, VIEWS[view], c_new, HALF, px=PX, ss=2,
                                light=rs.light_vector(VIEWS[view], az, el))
                _show(axp, img, label, f"{view}, same camera and light")
        fig.suptitle("what the stage changed (both panels: identical camera, scale and light)",
                     color=FG, fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.02)
        fig.savefig(rnd / "compare_base_vs_final.png", facecolor=BG, dpi=125)
        plt.close(fig)

    # 10. open state sanity -----------------------------------------------
    fig, axs = _figure(2, 3.8, 4.0)
    for axp, (label, V, F) in zip(axs, (("closed", Vn, Fn), ("open (shared topology)", Vo, Fo))):
        img = rs.render(V, F, VIEWS["oblique_left"], c_new, HALF, px=PX, ss=2,
                        light=rs.light_vector(VIEWS["oblique_left"], 0.0, 18.0))
        _show(axp, img, label)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.02)
    fig.savefig(rnd / "states_closed_open.png", facecolor=BG, dpi=125)
    plt.close(fig)
    return rnd


def _soft_normals(V, F, mm=2.5):
    """Vertex normals averaged over roughly ``mm`` of surface.

    The new mesh is four times denser than the previous result, so with raw
    interpolated normals it resolves every sub-0.1 mm ripple of the capture and
    reads as mottling that the coarser mesh cannot show.  That is a property of
    the tessellation, not of the anatomy.  Averaging the *normals* over a stated
    scale, applied identically to all three models, compares structure instead.
    """
    tri = np.asarray(V, float)[np.asarray(F, np.int64)]
    # the OBJ is in metres and ``mm`` is in millimetres: without the conversion
    # the ring count becomes mm / 0.0017 = ~1500, the 1-ring average converges to
    # one normal for the whole head and every render turns into flat ambient
    # shading (measured: vn.light 0.94 on the coarse meshes, 0.50 on the dense
    # one, i.e. two constants, and no relief visible in either)
    edge_mm = float(np.median(np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1))) * 1000.0
    iters = max(1, int(round(mm / edge_mm))) if edge_mm > 0 else 1
    vn = npn.vertex_normals(V, F)
    vn = rs.neighbour_average(np.asarray(V, float), F, vn, iterations=iters)
    ln = np.linalg.norm(vn, axis=1); ln[ln == 0] = 1.0
    return vn / ln[:, None]


def _render_flat(V, F, view, centre, half, px):
    """Per-face constant shading: the control panel for the Gouraud render."""
    fwd, right, up = rs.basis(view)
    size = int(px * 2)
    # _project works in mm, the OBJ is in metres
    Vmm = np.asarray(V, float) * rs.MM_PER_UNIT
    cmm = np.asarray(centre, float) * rs.MM_PER_UNIT
    xy, Z, _, _ = rs._project(Vmm, F, view, cmm, half, size, keep_backface=False)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    light = rs.light_vector(view, 0.0, 18.0)
    val = 0.18 + 0.82 * np.clip(fn @ light, 0.0, 1.0)
    keep = np.flatnonzero((fn @ (-fwd)) > 0.0)
    ax0, ay0 = xy[:, 0], xy[:, 1]
    zbuf = np.full((size, size), np.inf, np.float32)
    sbuf = np.zeros((size, size), np.float32)
    for fi in keep:
        a, b, c = F[fi]
        x0, x1, x2 = ax0[a], ax0[b], ax0[c]
        y0, y1, y2 = ay0[a], ay0[b], ay0[c]
        xlo = max(int(np.floor(min(x0, x1, x2))), 0)
        xhi = min(int(np.ceil(max(x0, x1, x2))), size - 1)
        ylo = max(int(np.floor(min(y0, y1, y2))), 0)
        yhi = min(int(np.ceil(max(y0, y1, y2))), size - 1)
        if xhi < xlo or yhi < ylo:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(xlo, xhi + 1) + 0.5,
                             np.arange(ylo, yhi + 1) + 0.5)
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-7) & (l1 >= -1e-7) & (l2 >= -1e-7)
        zz = l0 * Z[a] + l1 * Z[b] + l2 * Z[c]
        sub = zbuf[ylo:yhi + 1, xlo:xhi + 1]
        take = inside & (zz < sub)
        if not take.any():
            continue
        sub[take] = zz[take]
        ssub = sbuf[ylo:yhi + 1, xlo:xhi + 1]
        ssub[take] = val[fi]
    img = np.empty((size, size, 3), np.float32)
    img[:] = np.array([0.105, 0.105, 0.115], np.float32)
    filled = np.isfinite(zbuf)
    img[filled] = (sbuf[..., None] * np.array([0.97, 0.95, 0.92], np.float32))[filled]
    return np.clip(img.reshape(px, 2, px, 2, 3).mean(axis=(1, 3)), 0, 1)
