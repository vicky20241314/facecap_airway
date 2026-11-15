"""Same-camera renders for the v2 direct-neck delivery.

The 12 baseline panels are produced by the existing `neck_direct_render` tool
(same camera, scale, light, stated 2.5 mm normal treatment for every model).
This module adds what the v2 brief asks for explicitly:

* actual wireframe of the whole head and of the upper-neck / submental ring
  bands (original vs delivered, identical camera);
* flat vs smooth shading of the delivered surface (same light, no hidden
  normal smoothing);
* density / local compression plot from the delivered measurements;
* two-state (closed / open) submental-to-root anterior profiles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from head_rgbd import mesh_integrity as mi             # noqa: E402
from head_rgbd import neck_direct as nd                # noqa: E402
from head_rgbd import neck_fit as nfit                 # noqa: E402
from head_rgbd import neck_stage as ns                 # noqa: E402
from head_rgbd import raster as rs                     # noqa: E402
from tools import build_neck_stage as bns             # noqa: E402
from tools import neck_direct_render as r1            # noqa: E402
from tools import neck_stage_render as r0             # noqa: E402


BG = r0.BG
FG = r0.FG


def _fig(ncols, title=None, width=4.0, height=4.2):
    fig, axs = plt.subplots(1, ncols, figsize=(width * ncols, height),
                            facecolor=BG)
    if ncols == 1:
        axs = [axs]
    if title:
        fig.suptitle(title, color=FG, fontsize=9)
    return fig, axs


def _save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


def _pair_wire(V0, F0, Vf, Ff, ax, curve, hc, view, t_lo, t_hi, deg_max,
               label, out_path):
    c, half = r1._roi_frame(V0, ax, hc, curve, view, t_lo=t_lo, t_hi=t_hi,
                            deg_max=deg_max, margin=1.18)
    a = r1.render_wire(V0, F0, view, c, half)
    b = r1.render_wire(Vf, Ff, view, c, half)
    fig, axs = _fig(2, title=label)
    for axx, img, txt in ((axs[0], a, "original capture"), (axs[1], b, "direct v2")):
        axx.imshow(img, interpolation="bilinear")
        axx.axis("off")
        axx.set_title(txt, color=FG, fontsize=8.5, pad=3)
    return _save(fig, out_path)


def _flat_smooth(Vf, Ff, ax, hc, curve, out_path):
    view = r0.VIEWS["front"]
    c, half = r1._roi_frame(Vf, ax, hc, curve, view, t_lo=0.55, t_hi=1.02,
                            deg_max=100.0, margin=1.12)
    flat = r0._render_flat(Vf, Ff, view, c, half, r0.PX)
    vn = r0._soft_normals(Vf, Ff, mm=2.5)
    smooth = rs.render(Vf, Ff, view, c, half, px=r0.PX, ss=2,
                       light=rs.light_vector(view, -18.0, 10.0), vn=vn)
    fig, axs = _fig(3, title="delivered surface: actual facets / flat shading / smooth shading")
    imgs = (r1.render_wire(Vf, Ff, view, c, half), flat, smooth)
    names = ("actual wireframe", "flat (face normal)", "smooth (2.5 mm vertex normals)")
    for axx, img, txt in zip(axs, imgs, names):
        axx.imshow(img, interpolation="bilinear")
        axx.axis("off")
        axx.set_title(txt, color=FG, fontsize=8.5, pad=3)
    return _save(fig, out_path)


def _ring_metrics(out, path):
    vpath = Path(out) / "verification.json"
    if not vpath.exists():
        return None
    v = json.load(open(vpath, encoding="utf-8"))
    rb = v["S1_density"]["ring_bands"]
    inp, abl, full = rb["input"], rb["zero_topology_ablation"], rb["full"]
    tc = [0.5 * (r["t"][0] + r["t"][1]) for r in inp]
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.2), facecolor=BG)
    axs = axs.ravel()

    def line(a, key, label, colour):
        def vals(rows):
            return [np.nan if r.get(key) is None else r[key] for r in rows]
        axs[a].plot(tc, vals(inp), color="#9a9a9a", lw=1.0,
                    label="input")
        axs[a].plot(tc, vals(abl), color="#4aa3ff", lw=1.2,
                    label="zero topology")
        axs[a].plot(tc, vals(full), color="#ff9f40", lw=1.4,
                    label="full v2")
        axs[a].set_xlabel("material t", color=FG)
        axs[a].set_ylabel(label, color=FG)
        axs[a].tick_params(colors=FG)
        axs[a].grid(alpha=0.25)
        axs[a].legend(fontsize=7, facecolor=BG, labelcolor=FG)

    line(0, "edge_p50_mm", "edge p50 (mm)", "")
    # area ratio is a property of the bin, not of a mesh value; compute directly
    axs[1].plot(tc, [r.get("area_cm2", np.nan) for r in inp], color="#9a9a9a", lw=1.0, label="input")
    axs[1].plot(tc, [r.get("area_cm2", np.nan) for r in abl], color="#4aa3ff", lw=1.2, label="zero topology")
    axs[1].plot(tc, [r.get("area_cm2", np.nan) for r in full], color="#ff9f40", lw=1.4, label="full v2")
    axs[1].set_xlabel("material t", color=FG)
    axs[1].set_ylabel("area (cm2)", color=FG)
    axs[1].tick_params(colors=FG)
    axs[1].grid(alpha=0.25)
    axs[1].legend(fontsize=7, facecolor=BG, labelcolor=FG)
    for a, key, lab in ((2, "vertices_per_cm2", "vertices / cm2"),
                        (3, "strain_p95_mm_per_mm", "strain p95 (mm/mm)")):
        axs[a].plot(tc, [np.nan if r.get(key) is None else r[key] for r in inp],
                    color="#9a9a9a", lw=1.0, label="input")
        axs[a].plot(tc, [np.nan if r.get(key) is None else r[key] for r in abl],
                    color="#4aa3ff", lw=1.2, label="zero topology")
        axs[a].plot(tc, [np.nan if r.get(key) is None else r[key] for r in full],
                    color="#ff9f40", lw=1.4, label="full v2")
        axs[a].set_xlabel("material t", color=FG)
        axs[a].set_ylabel(lab, color=FG)
        axs[a].tick_params(colors=FG)
        axs[a].grid(alpha=0.25)
        axs[a].legend(fontsize=7, facecolor=BG, labelcolor=FG)
    axs[1].set_title("area per ring bin (compression, not added density)",
                     color=FG, fontsize=8)
    axs[3].set_title("edge-length strain vs capture", color=FG, fontsize=8)
    fig.suptitle("ring-band density and local compression (closed state)",
                 color=FG, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


def _profiles(out, V0c, F0, V0o, ax, curve, hc, ax_o, curve_o, hc_o, path):
    npz = np.load(Path(out) / "stage_direct.npz")
    Vc = np.asarray(npz["V_closed"], float)
    Vo = np.asarray(npz["V_open"], float)
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 5.0), facecolor=BG)
    for axx, V0, Vf, axx_, cur, hcc, title in (
            (axs[0], V0c, Vc, ax, curve, hc, "closed"),
            (axs[1], V0o, Vo, ax_o, curve_o, hc_o, "open")):
        h0, th0, r0, _ = nd.axial_mm(np.asarray(V0, float) * 1000.0, axx_, cur, hcc)
        hf, thf, rf, _ = nd.axial_mm(np.asarray(Vf, float) * 1000.0, axx_, cur, hcc)
        m0 = np.abs(np.degrees(th0)) < 5.0
        mf = np.abs(np.degrees(thf)) < 5.0
        o0 = np.argsort(h0[m0])
        of = np.argsort(hf[mf])
        axx.plot(h0[m0][o0], r0[m0][o0], color="#9a9a9a", lw=1.0,
                 label="input")
        axx.plot(hf[mf][of], rf[mf][of], color="#ff9f40", lw=1.5,
                 label="direct v2")
        axx.set_xlim(0, 1.06 * hcc)
        axx.set_xlabel("axial h (mm, this state's frame)", color=FG)
        axx.set_ylabel("radius at |theta| < 5 deg (mm)", color=FG)
        axx.set_title(f"{title}: submental-to-root anterior profile", color=FG,
                      fontsize=8.5)
        axx.tick_params(colors=FG)
        axx.grid(alpha=0.25)
        axx.legend(fontsize=7, facecolor=BG, labelcolor=FG)
    fig.suptitle("two-state profile (closed / open), same material band", color=FG,
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return path


def _full_wire(V0, F0, Vf, Ff, ax, hc, out_path):
    view = r0.VIEWS["front"]
    c = r0._centre(ax, hc, 0.72)
    half = 1.12 * hc
    a = r1.render_wire(V0, F0, view, c, half)
    b = r1.render_wire(Vf, Ff, view, c, half)
    fig, axs = _fig(2, title="whole head + neck: actual wireframe (front)")
    for axx, img, txt in ((axs[0], a, "original capture"), (axs[1], b, "direct v2")):
        axx.imshow(img, interpolation="bilinear")
        axx.axis("off")
        axx.set_title(txt, color=FG, fontsize=8.5, pad=3)
    return _save(fig, out_path)


def _displacement(Vc, Fc, Vin, d, ax, hc, out_path):
    from head_rgbd import neck_prominence as npn
    view = r0.VIEWS["front"]
    c = r0._centre(ax, hc, 0.72)
    half = 1.12 * hc
    mag = np.linalg.norm(np.asarray(d, float), axis=1)
    n_out = npn.vertex_normals(Vc, Fc)
    n_in = npn.vertex_normals(Vin, Fc)
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n_out, n_in), -1.0, 1.0)))
    imgs = (r1.render_scalar(Vc, Fc, view, c, half, mag, vmin=0.0, vmax=5.0),
            r1.render_scalar(Vc, Fc, view, c, half, ang, vmin=0.0, vmax=60.0))
    fig, axs = _fig(2, title="delivered: displacement magnitude (mm) / normal change (deg)")
    for axx, img, txt in zip(axs, imgs, ("|d| (mm)", "normal change (deg)")):
        axx.imshow(img, interpolation="bilinear")
        axx.axis("off")
        axx.set_title(txt, color=FG, fontsize=8.5, pad=3)
    return _save(fig, out_path)


def render_all(out, fit_dir=None, baseline=True):
    out = Path(out)
    rnd = out / "render"
    rnd.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(out / "build_direct.json", encoding="utf-8"))
    p = meta["params"]
    npz = np.load(out / "stage_direct.npz")
    if fit_dir is None:
        fit_dir = Path((meta.get("provenance") or {}).get(
            "fit_dir", PROJ / "runtime/outputs/zp_capwitharuco/reconstruction/dense"))
    fit_dir = Path(fit_dir)
    if not fit_dir.is_absolute():
        fit_dir = PROJ / fit_dir
    default_fit = (PROJ / "runtime/outputs/zp_capwitharuco/reconstruction/dense").resolve()
    # the 12-panel baseline compares with an accepted appearance of the ORIGINAL
    # subject; never attach it to a different fit directory
    baseline = bool(baseline and fit_dir.resolve() == default_fit)
    states, F0, _ = bns.load_subject(fit_dir)
    V0c, V0o = states["closed"]["V"], states["open"]["V"]
    _, ax, hc = ns.subject_frame(V0c, F0, states["closed"]["landmarks"])
    curve, _ = ns.spine_curve(V0c, F0, ax, hc)
    _, ax_o, hc_o = ns.subject_frame(V0o, F0, states["open"]["landmarks"])
    curve_o, _ = ns.spine_curve(V0o, F0, ax_o, hc_o)
    Vc, Fc = mi.load_obj(out / "head_closed_direct.obj")
    Vo, Fo = mi.load_obj(out / "head_open_direct.obj")

    produced = {}
    if baseline:
        # the 12-panel baseline is for the original subject; it must not be used
        # when the fit directory is a different subject
        RV, RF, _, rax, rhc, _ = ns.reference_frame(
            "D:/project_neck/runtime/v8_tail_smoke/experiment_20261109-200612/final/identity_rest.obj")
        rcurve, _ = ns.spine_curve(RV, RF, rax, rhc)
        ref_field = ns.analyse_surface(RV, RF, rax, rhc, curve=rcurve,
                                       structure_sigma=p["structure_sigma_mm"],
                                       fine_sigma=p["fine_sigma_mm"], leak_mask=True)
        subj_c = ns.analyse_surface(V0c, F0, ax, hc, curve=curve,
                                    structure_sigma=p["structure_sigma_mm"],
                                    fine_sigma=p["fine_sigma_mm"], leak_mask=True)
        tg = bns.target_field(ref_field, subj_c, p)
        reg = np.asarray(npz["region"], np.int64)
        partition = {"A": reg == 0, "B": reg == 1, "C": reg == 2}
        try:
            baseline_result = r1.render_all(
                out, PROJ / "runtime/outputs/zp_capwitharuco_neck_stage_v2",
                ref_field=ref_field, tgt=tg, subj_field=subj_c,
                partition=partition,
                h_material=np.asarray(npz["h_material"], float))
            produced["baseline"] = str(baseline_result)
        except Exception as exc:
            print(f"baseline renderer failed: {type(exc).__name__}: {exc}", flush=True)
    produced["wire_full.png"] = str(_full_wire(
        V0c, F0, Vc, Fc, ax, hc, rnd / "wire_full.png"))
    produced["wire_upper_neck.png"] = str(_pair_wire(
        V0c, F0, Vc, Fc, ax, curve, hc, r0.VIEWS["front"], 0.86, 1.01, 100.0,
        "upper neck ring band: actual wireframe (front)",
        rnd / "wire_upper_neck.png"))
    produced["wire_submental.png"] = str(_pair_wire(
        V0c, F0, Vc, Fc, ax, curve, hc, r0.VIEWS["submental"], 0.78, 0.96, 85.0,
        "submental connection: actual wireframe (submental view)",
        rnd / "wire_submental.png"))
    produced["flat_vs_smooth.png"] = str(_flat_smooth(
        Vc, Fc, ax, hc, curve, rnd / "flat_vs_smooth.png"))
    produced["displacement.png"] = str(_displacement(
        Vc, Fc, np.asarray(npz["V_closed_input"], float),
        np.asarray(npz["d_closed"], float), ax, hc, rnd / "displacement.png"))
    produced["two_state_sections.png"] = str(_profiles(
        out, V0c, F0, V0o, ax, curve, hc, ax_o, curve_o, hc_o,
        rnd / "two_state_sections.png"))
    met = _ring_metrics(out, rnd / "ring_band_metrics.png")
    if met is not None:
        produced["ring_band_metrics.png"] = str(met)
    return produced


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runtime/outputs/zp_capwitharuco_neck_direct_v2")
    ap.add_argument("--fit", default=None,
                    help="subject dense reconstruction; default from build_direct.json provenance")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the 12-panel original-subject baseline renderer")
    a = ap.parse_args()
    p = Path(a.out)
    if not p.is_absolute():
        p = PROJ / p
    print(json.dumps(render_all(p, fit_dir=a.fit, baseline=not a.no_baseline), indent=1))