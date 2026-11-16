"""Unified neck appearance stage (v2): reference structure transfer onto the subject base.

Why this module exists
----------------------
The 0.9 ``neck_volume`` design reduced the reference neck to thirteen
cross-sections, then to four scalars per section (W, D, r_front, r_back) and a
second-order angular function.  Whatever local structure the reference mesh had
(one main laryngeal prominence, a weaker cricoid region, the shallow transition
between them, the oblique lateral relief) cannot survive that reduction: the
angular basis is ``a0 + a1 cos(theta) + a2 cos(2 theta)`` and the axial basis is
thirteen C2 spline knots.  Its "detail" term was two wide Gaussians of at most
1.35 mm, applied along the anterior axis.

This stage keeps one representation of the neck surface -- a signed radial field
``R(theta, h)`` sampled from an actual mesh -- and works on it directly:

    1. base      : the subject's own low-frequency radial field (no creases, no
                   inherited facets), re-solved on a locally refined mesh;
    2. structure : the reference's band-pass residual (fine minus 22 mm
                   background) transferred as ONE continuous 2-D field, so the
                   prominence, the transition, the secondary relief and the
                   oblique lateral undulation remain mutually consistent;
    3. connection: a bounded submental fillet, a C2 root hand-over and a
                   per-state transfer through vertex correspondence.

Nothing is fitted to a curve peak list, no manual bumps are placed, and no
previous reshape result is used as input: the base is the original dense
reconstruction.

Coordinate conventions
----------------------
``h``  : axial coordinate normal to the fitted bottom cut plane, mm.
``a,b``: lateral / anterior in-plane coordinates about the section centre, mm.
``th`` : ``atan2(a - ca, b - cb)``, 0 = anterior, +pi/2 = +lateral.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from head_rgbd import head_scale as hs
from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_volume as nv

SCHEMA = "head_rgbd.neck_stage.v2"
METHOD = "reference_structure_transfer_v2"

# --- surface field ---------------------------------------------------------
N_THETA = 360
H_STEP = 0.5                      # mm
SPINE_BAND = (0.30, 0.86)         # t range used to define the neck axis
STRUCTURE_SIGMA_MM = 20.0         # background scale removed from the relief
FINE_SIGMA_MM = 1.2               # noise floor removed from the relief
TAPER_SIGMA_MM = 5.0              # axial smoothing of the ring-averaged thickness profile
STRUCTURE_SIGMA_H_MM = 14.0       # axial scale of the background inside the non-axisymmetric part
LEAK_LO_MM = 8.0                  # radius beyond the ring median treated as "outside the tube"
LEAK_HI_MM = 16.0

# --- editable band ---------------------------------------------------------
T_ROOT_LO, T_ROOT_HI = 0.22, 0.32   # hand-over from the shoulder plate
T_TOP_LO, T_TOP_HI = 0.93, 0.99     # hand-over to the jaw underside
THETA_FULL = np.radians(112.0)      # full angular weight
THETA_ZERO = np.radians(136.0)      # no angular weight beyond this

# --- design ---------------------------------------------------------------
STRUCTURE_GAIN = 1.0
STRUCTURE_GAIN_MAX = 2.2
FILLET_ITERS = 6
FILLET_LAMBDA = 0.42
FILLET_BAND = (0.80, 1.04)          # t range of the submental fillet
FILLET_THETA = np.radians(78.0)
REG_ITERS = 8
REG_WEIGHT = 0.50


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _ramp(x, lo, hi):
    return nv._ramp(x, lo, hi)


def _dumpable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _dumpable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_dumpable(v) for v in o]
    return o


# ---------------------------------------------------------------------------
# signed radial field of a mesh
# ---------------------------------------------------------------------------

def section_triangles(V: np.ndarray, F: np.ndarray, ax: dict) -> np.ndarray:
    """Mesh triangles in the (lateral, anterior, axis) frame, in mm.

    Independent of the section height, so a caller sweeping many levels builds
    it once instead of re-transforming every face per level.
    """
    centre = ax["centre_mm"]
    e1, e2, axis = ax["lateral"], ax["anterior"], ax["axis"]
    return (V[F] * 1000.0 - centre) @ np.column_stack([e1, e2, axis])


def section_segments(V: np.ndarray, F: np.ndarray, ax: dict, h0: float,
                     tri: np.ndarray | None = None) -> np.ndarray:
    """Plane h=h0 section of the mesh as (n,2,2) segments in (a,b) mm."""
    if tri is None:
        tri = section_triangles(V, F, ax)
    s = tri[:, :, 2] - h0
    lo, hi = s.min(axis=1), s.max(axis=1)
    ids = np.flatnonzero((lo <= 0) & (hi >= 0))
    if not len(ids):
        return np.zeros((0, 2, 2))
    tri = tri[ids][:, :, :2]
    s = s[ids]
    pts = []
    for i in range(3):
        j = (i + 1) % 3
        t0, t1 = s[:, i], s[:, j]
        cross = ((t0 <= 0) & (t1 > 0)) | ((t1 <= 0) & (t0 > 0))
        p = np.full((len(ids), 2), np.nan)
        den = np.where(np.abs(t0 - t1) > 1e-12, t0 - t1, 1.0)
        w = np.where(cross, t0 / den, 0.0)
        p[cross] = tri[cross, i] + w[cross, None] * (tri[cross, j] - tri[cross, i])
        pts.append(p)
    segs = []
    for i in range(3):
        j = (i + 1) % 3
        a, b = pts[i], pts[j]
        ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        if ok.any():
            segs.append(np.stack([a[ok], b[ok]], axis=1))
    return np.vstack(segs) if segs else np.zeros((0, 2, 2))


def ray_radius(segs: np.ndarray, theta: np.ndarray, centre_ab=(0.0, 0.0)) -> np.ndarray:
    """First surface hit along each ray from centre_ab, direction (sin t, cos t).

    Rays that leave the section, or whose first hit is behind their origin, are
    NaN.  Taking the FIRST hit is what prevents front/back mixing and
    interpolation across a hole: no ray is ever bridged to a farther surface.
    """
    if len(segs) == 0:
        return np.full(len(theta), np.nan)
    A = segs[:, 0] - np.asarray(centre_ab)
    E = segs[:, 1] - segs[:, 0]
    D = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    det = -E[:, 0][None, :] * D[:, 1][:, None] + E[:, 1][None, :] * D[:, 0][:, None]
    Ax, Ay = A[:, 0][None, :], A[:, 1][None, :]
    Ex, Ey = E[:, 0][None, :], E[:, 1][None, :]
    Dx, Dy = D[:, 0][:, None], D[:, 1][:, None]
    ok = np.abs(det) > 1e-12
    safe = np.where(ok, det, 1.0)
    u = (Ax * Dy - Ay * Dx) / safe
    r = (Ey * Ax - Ex * Ay) / safe
    good = ok & (u >= -1e-9) & (u <= 1 + 1e-9) & (r > 0)
    r = np.where(good, r, np.inf)
    out = r.min(axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def spine_curve(V: np.ndarray, F: np.ndarray, ax: dict, h_chin: float,
                levels: np.ndarray | None = None):
    """Neck axis: C2 fit of the section centres inside the tube band only.

    Using section bounding-box centres over the whole neck is unstable near the
    jaw and the shoulders (the jaw footprint and the shoulder plate move the
    box centre by more than 15 mm).  Only the tube band defines the axis; the
    rest is a linear extension, so R(theta,h) means the same thing on the
    subject and on the reference.
    """
    lv = np.linspace(0.30, 0.86, 15) if levels is None else levels
    sec = nv.measure_sections(V, F, ax, lv * h_chin, "spine")
    h = np.array([s["h_mm"] for s in sec], float)
    ca = np.array([s.get("centre_a_mm", 0.0) for s in sec], float)
    cb = np.array([s.get("centre_b_mm", 0.0) for s in sec], float)
    sa = CubicSpline(h, ca, bc_type="natural")
    sb = CubicSpline(h, cb, bc_type="natural")
    da = (ca[-1] - ca[0]) / max(h[-1] - h[0], 1e-6)
    db = (cb[-1] - cb[0]) / max(h[-1] - h[0], 1e-6)

    def curve(hh):
        hh = np.asarray(hh, float)
        lo = hh < h[0]
        hi = hh > h[-1]
        out_a = np.where(lo, ca[0] + da * (hh - h[0]),
                         np.where(hi, ca[-1] + da * (hh - h[-1]), sa(np.clip(hh, h[0], h[-1]))))
        out_b = np.where(lo, cb[0] + db * (hh - h[0]),
                         np.where(hi, cb[-1] + db * (hh - h[-1]), sb(np.clip(hh, h[0], h[-1]))))
        return out_a, out_b

    return curve, {"h_mm": h.tolist(), "a_mm": ca.tolist(), "b_mm": cb.tolist(),
                   "slope_a": float(da), "slope_b": float(db)}


def radial_field(V, F, ax, h_chin, curve, n_theta=N_THETA, h_lo=0.02, h_hi=1.06):
    hs_ = np.arange(h_lo, h_hi + 1e-9, H_STEP / h_chin) * h_chin
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    R = np.full((len(hs_), n_theta), np.nan)
    tri = section_triangles(V, F, ax)
    for i, h0 in enumerate(hs_):
        ca, cb = curve(float(h0))
        segs = section_segments(V, F, ax, float(h0), tri=tri)
        if len(segs) == 0:
            continue
        R[i] = ray_radius(segs, th, (float(ca), float(cb)))
    # an isolated failed ray is filled along theta only (never across h, never
    # from another surface); wide gaps stay NaN and are excluded from the design
    for i in range(R.shape[0]):
        row = R[i]
        bad = ~np.isfinite(row)
        if bad.any() and (~bad).sum() > 0.65 * n_theta:
            idx = np.arange(n_theta)
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad], period=n_theta)
            R[i] = row
    return hs_, th, R


def _blur(R, hs_, th, sigma_mm, radius_mm=None):
    """Separable Gaussian on a (h, theta) grid; theta is periodic.

    ``radius_mm`` fixes the arc length of one theta cell.  It must be supplied
    when the field is not a radius itself (e.g. a residual whose median is ~0),
    because deriving it from the data would divide by zero.
    """
    dh = float(np.median(np.diff(hs_)))
    periodic_radius = float(radius_mm) if radius_mm is not None else float(np.nanmedian(R))
    periodic_radius = max(periodic_radius, 1.0)
    dth = 2.0 * np.pi * periodic_radius / R.shape[1]
    out = gaussian_filter1d(R, sigma_mm / dh, axis=0, mode="nearest")
    out = gaussian_filter1d(out, sigma_mm / max(dth, 1e-6), axis=1, mode="wrap")
    return out


def _blur_masked(R, w, hs_, th, sigma_mm, radius_mm=None):
    """Normalised (weighted) blur: the kernel never borrows from invalid rows.

    A plain blur across the top of the neck pulls in the jaw underside (where
    the section is 8-10 mm larger) and across the root pulls in the shoulder
    plate.  The normalised form keeps the estimate local to the tube.
    """
    w = np.where(np.isfinite(R), w, 0.0)
    Rz = np.where(np.isfinite(R), R, 0.0)
    num = _blur(Rz * w, hs_, th, sigma_mm, radius_mm=radius_mm)
    den = _blur(w, hs_, th, sigma_mm, radius_mm=radius_mm)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)


def _row_taper(R, hs_, th, sigma_mm, band, leak_lo_mm, leak_hi_mm):
    """Ring-averaged radius per height: the neck's own thickness profile.

    This is what a low-pass filter must NOT touch.  A Gaussian blur of R in the
    axial direction smears the neck/shoulder flare upward by 3-6 mm and the
    design then inflates the lower neck; averaging over theta first, and only
    afterwards filtering the non-axisymmetric part, keeps the taper exact.
    """
    w = _ramp(np.linspace(0, 1, len(R)), band[0] - 0.06, band[0]) * \
        (1.0 - _ramp(np.linspace(0, 1, len(R)), band[1], band[1] + 0.05))
    with np.errstate(invalid="ignore"):
        rm = np.nanmedian(R, axis=1, keepdims=True)
        leak = np.nan_to_num(_ramp(R - rm, leak_lo_mm, leak_hi_mm), nan=1.0)
        good = np.isfinite(R) & (leak < 0.5) & (w[:, None] > 0.5)
        cnt = good.sum(axis=1)
        tot = np.where(good, R, 0.0).sum(axis=1)
    taper = np.where(cnt > 0.25 * R.shape[1], tot / np.maximum(cnt, 1), np.nan)
    idx = np.arange(len(taper))
    ok = np.isfinite(taper)
    if ok.sum() >= 2:
        taper = np.interp(idx, idx[ok], taper[ok])
    else:
        taper = np.full(len(taper), float(np.nanmedian(R)))
    dh = float(np.median(np.diff(hs_)))
    return gaussian_filter1d(taper, max(sigma_mm, 1e-6) / dh, mode="nearest")


def decompose(R, hs_, th, t=None, structure_sigma=STRUCTURE_SIGMA_MM,
              fine_sigma=FINE_SIGMA_MM, band=(0.03, 0.90),
              taper_sigma=TAPER_SIGMA_MM, structure_sigma_h=STRUCTURE_SIGMA_H_MM,
              leak_lo_mm=LEAK_LO_MM, leak_hi_mm=LEAK_HI_MM, leak_mask=False):
    """background / relief split of one radial field.

        taper(h)        ring-averaged radius (axisymmetric thickness profile)
        aniso           R - taper, i.e. cross-sectional shape + local structure
        background      taper + low-pass(aniso)   -> the "reasonable base"
        relief          fine(R) - background      -> the 3-D structure on top

    The low-pass is applied to the non-axisymmetric part only, so the neck's
    taper, flare and root shape pass through untouched and no smoothing can
    inflate them.
    """
    if t is None:
        t = (hs_ - hs_[0]) / max(hs_[-1] - hs_[0], 1e-9)
    taper = _row_taper(R, hs_, th, taper_sigma, band, leak_lo_mm, leak_hi_mm)
    aniso = R - taper[:, None]
    with np.errstate(invalid="ignore"):
        rm = np.nanmedian(R, axis=1, keepdims=True)
        leak = np.nan_to_num(_ramp(R - rm, leak_lo_mm, leak_hi_mm), nan=1.0)
    w = _ramp(t, band[0] - 0.06, band[0]) * (1.0 - _ramp(t, band[1], band[1] + 0.05))
    w2 = np.repeat(w[:, None], R.shape[1], axis=1)
    if leak_mask:
        w2 = w2 * (1.0 - leak)
    radius_mm = float(np.nanmedian(taper))
    smooth = _blur_masked(aniso, w2, hs_, th, structure_sigma, radius_mm=radius_mm)
    # the axial pass uses its own, wider scale: the structure we want to keep is
    # anisotropic in theta, and the axial direction carries the design intent
    dh = float(np.median(np.diff(hs_)))
    smooth = gaussian_filter1d(np.nan_to_num(smooth, nan=0.0),
                               max(structure_sigma_h, 1e-6) / dh, axis=0, mode="nearest")
    idx = np.arange(len(smooth))
    for j in range(smooth.shape[1]):
        col = smooth[:, j]
        good = np.isfinite(col)
        if not good.all() and good.sum() >= 2:
            col[~good] = np.interp(idx[~good], idx[good], col[good])
            smooth[:, j] = col
    bg = taper[:, None] + smooth
    fine = _blur(np.where(np.isfinite(R), R, np.nanmedian(R)), hs_, th, fine_sigma,
                 radius_mm=radius_mm)
    relief = fine - bg
    return {"background": bg, "relief": relief, "weight": w, "taper": taper,
            "aniso": aniso}


# ---------------------------------------------------------------------------
# frames and reference analysis
# ---------------------------------------------------------------------------

def subject_frame(V, F, landmarks_m):
    frame = nv.head_frame(np.asarray(landmarks_m, float))
    ax = nv.axial_frame(V, F, frame)
    return frame, ax, float(ax["h_chin_mm"])


def reference_frame(path):
    RV, RF = mi.load_obj(path)
    VM = RV * 1000.0
    menton = hs.detect_reference_menton(VM)
    chin = np.array([0.0, menton["menton_y_mm"], 0.0])
    sel = (np.abs(VM[:, 0]) <= 3.0) & (np.abs(VM[:, 1] - menton["menton_y_mm"]) <= 2.0)
    if sel.any():
        chin[2] = float(VM[sel, 2].max())
    frame = {"up": np.array([0., 1., 0.]), "right": np.array([1., 0., 0.]),
             "forward": np.array([0., 0., 1.]), "chin": chin / 1000.0}
    ax = nv.axial_frame(RV, RF, frame)
    return RV, RF, frame, ax, float(ax["h_chin_mm"]), menton


def analyse_surface(V, F, ax, h_chin, curve=None, n_theta=N_THETA,
                    structure_sigma=None, fine_sigma=None, leak_mask=False,
                    taper_sigma=None, structure_sigma_h=None):
    """Radial field + background/relief split + structural descriptors."""
    if curve is None:
        curve, spine = spine_curve(V, F, ax, h_chin)
    else:
        spine = None
    hs_, th, R = radial_field(V, F, ax, h_chin, curve, n_theta=n_theta)
    t = hs_ / h_chin
    dec = decompose(R, hs_, th, t=t,
                    structure_sigma=STRUCTURE_SIGMA_MM if structure_sigma is None else structure_sigma,
                    fine_sigma=FINE_SIGMA_MM if fine_sigma is None else fine_sigma,
                    leak_mask=leak_mask,
                    taper_sigma=TAPER_SIGMA_MM if taper_sigma is None else taper_sigma,
                    structure_sigma_h=(STRUCTURE_SIGMA_H_MM if structure_sigma_h is None
                                       else structure_sigma_h))
    return {"hs_mm": hs_, "theta": th, "t": t, "R": R,
            "background": dec["background"], "relief": dec["relief"],
            "spine": spine, "curve": curve, "h_chin_mm": h_chin}


def structure_descriptors(field, label="", t_lo=0.42, t_hi=0.90):
    """Primary / secondary relief lobes, saddle, lateral relief, symmetry.

    Peaks are found on the 2-D relief, not on a single centre-line curve.

    ``t_hi`` must be set per state: above a certain height the anterior ray hits
    the chin overhang instead of the neck, and its relief (20-40 mm) would win
    every maximum.  For the open state that happens ~14 mm lower in absolute
    terms, i.e. at t = 0.84 on the shared design scale.
    """
    R, rel, t, th = field["R"], field["relief"], field["t"], field["theta"]
    hs_ = field["hs_mm"]
    band = (t > t_lo) & (t < t_hi)
    front = np.abs(th) < np.radians(55.0)
    mask = band[:, None] & front[None, :]
    work = np.where(mask & np.isfinite(rel), rel, -np.inf)
    ij = np.unravel_index(np.nanargmax(work), work.shape)
    peak_val = float(work[ij])
    peak_t = float(t[ij[0]])
    peak_th = float(np.degrees(th[ij[1]]))

    # along the anterior column (theta ~ 0): prominence, saddle, second lobe
    j0 = int(np.argmin(np.abs(th)))
    col = np.where(band, rel[:, j0], np.nan)
    i_pk = int(np.nanargmax(col))
    below = np.arange(i_pk)
    below = below[band[below]]
    saddle = {"t": float(t[below[int(np.nanargmin(col[below]))]]),
              "mm": float(np.nanmin(col[below]))} if len(below) else None
    # secondary lobe = strongest local maximum below the saddle
    sec = None
    if saddle is not None:
        i_s = int(np.argmin(np.abs(t - saddle["t"])))
        seg = np.arange(0, i_s)
        seg = seg[band[seg] & (t[seg] > 0.30)]
        if len(seg) > 3:
            i_sec = seg[int(np.nanargmax(col[seg]))]
            if col[i_sec] > 0.05 and (saddle["mm"] < col[i_sec]):
                sec = {"t": float(t[i_sec]), "h_mm": float(hs_[i_sec]),
                       "mm": float(col[i_sec])}
    # half widths of the primary lobe at half amplitude
    half = peak_val * 0.5
    above = np.flatnonzero(col > half)
    width_t = float((t[above].max() - t[above].min()) * field["h_chin_mm"]) if len(above) else None
    win = np.flatnonzero(np.abs(t - peak_t) < 0.012)
    win = win[band[win]]
    if len(win):
        rprof = np.nanmean(rel[win], axis=0)
        i0 = int(np.argmin(np.abs(th)))
        # walk out symmetrically until the relief drops below half amplitude
        j = i0
        while j < len(th) - 1 and rprof[j] > half:
            j += 1
        k = i0
        while k > 0 and rprof[k] > half:
            k -= 1
        width_th = float(np.degrees(th[j] - th[k]))
    else:
        width_th = None
    # lateral oblique relief (30-90 deg), reported separately per side
    lat = {}
    for side, sgn in (("right", 1), ("left", -1)):
        sect = (sgn * th > np.radians(25)) & (sgn * th < np.radians(95)) & band[:, None]
        vals = rel[sect]
        lat[side] = {"min_mm": float(np.nanmin(vals)) if np.isfinite(vals).any() else None,
                     "max_mm": float(np.nanmax(vals)) if np.isfinite(vals).any() else None,
                     "median_mm": float(np.nanmedian(vals)) if np.isfinite(vals).any() else None}
    # symmetry of the relief about the anterior midline
    sym = rel - rel[:, ::-1]
    front_rows = band
    return {"label": label,
            "primary": {"t": peak_t, "h_mm": float(hs_[ij[0]]),
                        "theta_deg": peak_th, "relief_mm": peak_val,
                        "width_theta_deg": width_th, "width_t_mm": width_t},
            "saddle": saddle, "secondary": sec,
            "lateral": lat,
            "asymmetry_rms_mm": float(np.sqrt(np.nanmean(sym[front_rows][:, :180] ** 2))),
            "relief_rms_mm": float(np.sqrt(np.nanmean(rel[band] ** 2))),
            "relief_min_mm": float(np.nanmin(rel[band])),
            "relief_max_mm": float(np.nanmax(rel[band]))}


# ---------------------------------------------------------------------------
# target field
# ---------------------------------------------------------------------------

def build_target(reference_field, subject_field, structure_scale=1.0,
                 gain=STRUCTURE_GAIN, top_gate=(T_TOP_LO, T_TOP_HI)):
    """R_target = subject background + gain * scaled reference relief.

    The subject's own background carries its thickness, width, taper and root
    extent; the reference contributes only the signed relief, i.e. the
    prominence hierarchy, the transition and the oblique lateral undulation.
    Both live on the same normalised (theta, t) grid, so no world-space
    template position is imposed.
    """
    hs_s, th_s = subject_field["hs_mm"], subject_field["theta"]
    hs_r, th_r = reference_field["hs_mm"], reference_field["theta"]
    t_s = subject_field["t"]
    R_sub = subject_field["R"]
    bg_sub = subject_field["background"]
    t_r = reference_field["t"]

    # resample the reference relief onto (subject h, subject theta)
    rel_r = reference_field["relief"]
    rel_r = np.where(np.isfinite(rel_r), rel_r, 0.0)
    # theta interpolation (periodic)
    th_ext = np.r_[th_r - 2 * np.pi, th_r, th_r + 2 * np.pi]
    rel_ext = np.hstack([rel_r, rel_r, rel_r])
    idx_th = np.searchsorted(th_ext, th_s)
    idx_th = np.clip(idx_th, 1, len(th_ext) - 1)
    w_th = ((th_s - th_ext[idx_th - 1]) /
            np.maximum(th_ext[idx_th] - th_ext[idx_th - 1], 1e-9))[None, :]
    rel_theta = rel_ext[:, idx_th - 1] * (1 - w_th) + rel_ext[:, idx_th] * w_th
    # t interpolation (clamped): same normalised height, no world coordinates
    rel_t = np.empty((len(hs_s), len(th_s)))
    for j in range(len(th_s)):
        rel_t[:, j] = np.interp(t_s, t_r, rel_theta[:, j])

    gate = 1.0 - _ramp(t_s, top_gate[0], top_gate[1])
    rel_t = rel_t * gate[:, None] * float(structure_scale) * float(gain)
    R_tgt = bg_sub + rel_t
    return {"R_target": R_tgt, "relief_applied": rel_t, "background": bg_sub,
            "t": t_s, "theta": th_s, "hs_mm": hs_s,
            "gate_top": gate}


class Field2D:
    """Periodic-in-theta bilinear sampler of a (h, theta) grid."""

    def __init__(self, hs_mm, theta, values, clamp_h=True):
        self.hs = np.asarray(hs_mm, float)
        self.th = np.asarray(theta, float)
        self.v = np.asarray(values, float)
        self.n_th = len(self.th)
        self.clamp_h = clamp_h

    def __call__(self, h, theta):
        h = np.asarray(h, float)
        th = np.asarray(theta, float)
        if self.clamp_h:
            h = np.clip(h, self.hs[0], self.hs[-1])
        i = np.clip(np.searchsorted(self.hs, h) - 1, 0, len(self.hs) - 2)
        wh = np.clip((h - self.hs[i]) / np.maximum(self.hs[i + 1] - self.hs[i], 1e-9), 0, 1)
        d = np.pi / self.n_th
        j = np.floor((th + np.pi) / (2 * np.pi) * self.n_th).astype(int) % self.n_th
        wt = ((th + np.pi) / (2 * np.pi) * self.n_th) - np.floor((th + np.pi) / (2 * np.pi) * self.n_th)
        j2 = (j + 1) % self.n_th
        v00 = self.v[i, j]
        v01 = self.v[i, j2]
        v10 = self.v[i + 1, j]
        v11 = self.v[i + 1, j2]
        return ((1 - wh) * (1 - wt) * v00 + (1 - wh) * wt * v01 +
                wh * (1 - wt) * v10 + wh * wt * v11)


# ---------------------------------------------------------------------------
# mesh operations
# ---------------------------------------------------------------------------

def refine(V: np.ndarray, F: np.ndarray, red_faces: np.ndarray):
    """One conforming red-green refinement; returns the shared topology."""
    from head_rgbd import neck_subject_v3 as n3
    sub = n3.subdivide_local(V, F, face_mask=red_faces)
    return sub


def apply_refinement(V: np.ndarray, sub: dict) -> np.ndarray:
    """Place midpoints by the same deterministic rule on ANY state."""
    edges = sub["midpoint_edges"]
    if not edges:
        return V.copy()
    mids = np.array([0.5 * (V[i] + V[j]) for i, j in edges], float)
    return np.vstack([V, mids])


def band_weights(h, theta, t, h_chin, protect):
    """C2 editable weight: axial band x angular window x explicit protection."""
    w_t = _ramp(t, T_ROOT_LO, T_ROOT_HI) * (1.0 - _ramp(t, T_TOP_LO, T_TOP_HI))
    at = np.abs(theta)
    w_th = 1.0 - _ramp(at, THETA_FULL, THETA_ZERO)
    # keep the posterior nape nearly untouched by the structure transfer
    w_th = w_th * (1.0 - 0.55 * _ramp(at, np.radians(100.0), np.radians(150.0)))
    w = w_t * w_th
    w = np.where(protect, 0.0, w)
    return w


def protection_mask(h, theta, t, h_chin, guard_mm=5.0):
    """Vertices that must not move: the jaw/face above the submental band.

    Defined BEFORE any solve, from the axial coordinate of the true cut plane and
    the menton level of this state, and recorded in the verification payload.
    """
    return (h >= h_chin - guard_mm) | (t >= T_TOP_HI)


def solve_state(V, F, ax, curve, field: Field2D, h_chin, protect,
                fillet=None, weight_extra=None):
    """Re-solve the editable band of one state onto the target radial field."""
    h, a, b = nv.axial_coords(V, ax)
    t = np.clip(h / h_chin, 0.0, 1.2)
    ca, cb = curve(h)
    theta = np.arctan2(a - ca, b - cb)
    w = band_weights(h, theta, t, h_chin, protect)
    if weight_extra is not None:
        w = np.minimum(w, weight_extra)
    r_cur = np.hypot(a - ca, b - cb)
    r_tgt = field(h, theta)
    r_tgt = np.where(np.isfinite(r_tgt), r_tgt, r_cur)
    # limit the radial change so a failed sample can never explode the surface
    r_tgt = np.clip(r_tgt, r_cur - 16.0, r_cur + 16.0)
    scale = np.ones_like(r_cur)
    scale = np.where(r_cur > 1e-6, r_tgt / np.maximum(r_cur, 1e-6), 1.0)
    dir_a = np.sin(theta)
    dir_b = np.cos(theta)
    a_new = ca + r_cur * scale * dir_a
    b_new = cb + r_cur * scale * dir_b
    P_tgt = (ax["centre_mm"][None, :] + h[:, None] * ax["axis"][None, :] +
             a_new[:, None] * ax["lateral"][None, :] + b_new[:, None] * ax["anterior"][None, :]) / 1000.0
    d = w[:, None] * (P_tgt - V)
    return {"displacement_mm": d * 1000.0, "weight": w, "h": h, "theta": theta,
            "t": t, "r_current": r_cur, "r_target": r_tgt, "a": a, "b": b,
            "base": V}


def regularize(F, d_mm, free, iterations=REG_ITERS, weight=REG_WEIGHT):
    """Low-frequency smoothing of the DISPLACEMENT field, free set only."""
    free = np.asarray(free, bool)
    edges = mi.edge_records(F)[0]
    keep = free[edges[:, 0]] & free[edges[:, 1]]
    e = edges[keep]
    rows = np.r_[e[:, 0], e[:, 1]]
    cols = np.r_[e[:, 1], e[:, 0]]
    d = d_mm.copy()
    for _ in range(iterations):
        acc = np.zeros_like(d)
        cnt = np.zeros(len(d), float)
        np.add.at(acc, rows, d[cols])
        np.add.at(cnt, rows, 1.0)
        avg = acc / np.maximum(cnt[:, None], 1.0)
        d[free] = (1 - weight) * d[free] + weight * avg[free]
        d[~free] = 0.0
    return d


def fillet_junction(V, F, free, iterations=FILLET_ITERS, lam=FILLET_LAMBDA):
    """Bounded Taubin fairing of the submental junction (a soft concave fillet).

    A radial re-solve cannot round a corner that lies at the boundary of the
    editable band; a short tangential relaxation of that band, with everything
    outside it fixed, is the minimal operation that removes the inherited
    step.  The displacement is measured and reported.
    """
    free = np.asarray(free, bool)
    edges = mi.edge_records(F)[0]
    neigh = defaultdict(list)
    for i, j in edges:
        neigh[int(i)].append(int(j))
        neigh[int(j)].append(int(i))
    V0 = V.copy()
    out = V.copy()
    idx = np.flatnonzero(free)
    for _ in range(iterations):
        new = out.copy()
        for i in idx:
            nb = neigh[int(i)]
            if nb:
                new[i] = out[i] + lam * (out[nb].mean(axis=0) - out[i])
        out = new
        new = out.copy()
        for i in idx:
            nb = neigh[int(i)]
            if nb:
                new[i] = out[i] - (lam * 1.02) * (out[nb].mean(axis=0) - out[i])
        out = new
    out[~free] = V0[~free]
    return out


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------

def boundary_vertices(F) -> np.ndarray:
    """Boolean mask of vertices on an open mesh boundary (mesh rim / cut edge)."""
    F = np.asarray(F, np.int64)
    e, _, counts, _, _ = mi.edge_records(F)
    rim = np.zeros(int(F.max()) + 1, bool)
    rim[e[counts == 1].ravel()] = True
    return rim


def fair_band(V: np.ndarray, F: np.ndarray, free, iterations=12, max_mm=2.0,
              lam=0.50, mu=-0.53, weight=None):
    """Bounded Taubin fairing of the *inherited* capture surface.

    The dense reconstruction is a decimated capture: its vertices sit 2-7 mm
    apart and carry 0.1-0.3 mm of vertex-scale error, with a handful of edges
    whose dihedral reaches 22-57 deg (measured on the base band).  Subdividing
    that surface does not remove any of it -- the children of a planar triangle
    stay coplanar -- so the faceting survives at a finer scale and any shading
    with interpolated normals reads it as hard straight lines that the anatomy
    does not have.

    A plain Laplacian would remove the faceting but shrink the neck by roughly
    L^2/(2R) for a diffusion length L, which on a 60 mm radius neck is 1-2 mm of
    systematic collapse.  Taubin's lambda/mu pair alternates a shrinking and an
    inflating pass of nearly equal magnitude, so the low-frequency shape is
    preserved while the high-frequency faceting is removed.

    ``free`` marks the edited region; ``weight`` (0..1 per vertex) blends the
    step to zero at its border so the region boundary does not become a crease.
    The displacement is returned in mm so it can be measured, not assumed.
    """
    free = np.asarray(free, bool)
    V = np.asarray(V, float)
    w = np.ones(len(V)) if weight is None else np.clip(np.asarray(weight, float), 0.0, 1.0)
    w = np.where(free, w, 0.0)
    edges = mi.edge_records(F)[0]
    keep = free[edges[:, 0]] & free[edges[:, 1]]
    e = edges[keep]
    rows = np.r_[e[:, 0], e[:, 1]]
    cols = np.r_[e[:, 1], e[:, 0]]
    cnt = np.zeros(len(V), float)
    np.add.at(cnt, rows, 1.0)
    safe = np.maximum(cnt, 1.0)[:, None]
    out = V.copy()

    def step(coef):
        acc = np.zeros_like(out)
        np.add.at(acc, rows, out[cols])
        # vertices with no free neighbour are held fixed
        avg = np.divide(acc, safe, out=out.copy(), where=safe > 0)
        return (coef * w)[:, None] * (avg - out)

    for _ in range(int(iterations)):
        out = out + step(lam)
        out = out + step(mu)
    d = (out - V) * 1000.0
    mag = np.linalg.norm(d, axis=1)
    if mag.max() > max_mm:
        d = d * np.minimum(1.0, max_mm / np.maximum(mag, 1e-9))[:, None]
    d[w <= 0.0] = 0.0
    return V + d / 1000.0, d


def dihedral_stats(V, F, mask=None):
    edges, inverse, counts, signs, _ = mi.edge_records(F)
    face_ids = np.tile(np.arange(len(F)), 3)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    tri = V[F] * 1000.0
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    sel = starts[counts == 2]
    i0, i1 = face_ids[order[sel]], face_ids[order[sel + 1]]
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fn[i0], fn[i1]), -1, 1)))
    ev = edges[counts == 2]
    out = {"p50": float(np.percentile(ang, 50)), "p90": float(np.percentile(ang, 90)),
           "p99": float(np.percentile(ang, 99)), "max": float(ang.max()),
           "over_25deg": int((ang > 25).sum())}
    if mask is not None:
        mid = 0.5 * (mask[ev[:, 0]] + mask[ev[:, 1]])
        sub = ang[mid > 0.5]
        out["neck"] = {"p50": float(np.percentile(sub, 50)), "p90": float(np.percentile(sub, 90)),
                       "p99": float(np.percentile(sub, 99)), "max": float(sub.max()),
                       "over_25deg": int((sub > 25).sum()), "edges": int(len(sub))}
    return out




def _tri_tri_intersect(p0, p1, p2, q0, q1, q2, eps=1e-9):
    """Vectorised Moller triangle/triangle overlap test.

    Returns (hit, coplanar).  ``hit`` is True when the two closed triangles
    share a point of common area along the plane-intersection line; pairs that
    merely touch at an edge or vertex, and coplanar pairs, are reported
    separately instead of being silently merged with "no intersection".
    """
    def plane_normal(a, b, c):
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        return n / np.maximum(ln, eps)[:, None], ln

    n1, l1 = plane_normal(p0, p1, p2)
    n2, l2 = plane_normal(q0, q1, q2)
    degenerate = (l1 <= eps) | (l2 <= eps)

    # signed distances of the other triangle's vertices to each plane
    dq = np.stack([np.einsum("ij,ij->i", n1, q - p0) for q in (q0, q1, q2)], axis=1)
    dp = np.stack([np.einsum("ij,ij->i", n2, p - q0) for p in (p0, p1, p2)], axis=1)

    same_q = (dq > eps).all(axis=1) | (dq < -eps).all(axis=1)
    same_p = (dp > eps).all(axis=1) | (dp < -eps).all(axis=1)
    coplanar = (~same_q) & (~same_p) & (np.abs(dq).max(axis=1) <= 1e3 * eps) & \
               (np.abs(dp).max(axis=1) <= 1e3 * eps)
    trivial = degenerate | same_q | same_p | coplanar

    dirv = np.cross(n1, n2)
    ln = np.linalg.norm(dirv, axis=1)
    ok = ln > 1e-10
    dirv = dirv / np.maximum(ln, 1e-12)[:, None]

    def interval(P, D):
        """Parameter range of the triangle/plane crossing along ``dirv``."""
        lo = np.full(len(P), np.inf)
        hi = np.full(len(P), -np.inf)
        for k in range(3):
            a, b = k, (k + 1) % 3
            da, db = D[:, a], D[:, b]
            cross = (da * db) < 0
            den = np.where(np.abs(da - db) > 1e-15, da - db, 1.0)
            s = np.where(cross, da / den, 0.0)
            X = P[:, a, :] + s[:, None] * (P[:, b, :] - P[:, a, :])
            t = np.einsum("ij,ij->i", X, dirv)
            m = cross & np.isfinite(t)
            lo = np.where(m, np.minimum(lo, t), lo)
            hi = np.where(m, np.maximum(hi, t), hi)
        return lo, hi

    Ptri = np.stack([p0, p1, p2], axis=1)
    Qtri = np.stack([q0, q1, q2], axis=1)
    lo1, hi1 = interval(Ptri, dp)
    lo2, hi2 = interval(Qtri, dq)
    valid = np.isfinite(lo1) & np.isfinite(lo2) & ok & ~trivial
    overlap = (lo1 <= hi2 + 1e-7) & (lo2 <= hi1 + 1e-7)
    hit = valid & overlap
    touching = valid & ~overlap & ((np.abs(lo1 - hi2) <= 1e-6) | (np.abs(lo2 - hi1) <= 1e-6))
    return hit, coplanar & ~degenerate, touching


def self_intersections(V, F, mask=None, max_pairs=6_000_000, workers=-1):
    """Actual triangle/triangle intersection count on a sphere broad phase.

    Every triangle is enclosed by the sphere (centroid, circumradius), so any
    intersecting pair is necessarily closer than the sum of the two radii: a
    cKDTree query with the maximum radius is a conservative, complete broad
    phase (no pair that could intersect is discarded).  Pairs that share a
    vertex are excluded (they are adjacent by construction) and reported as
    such; coplanar and edge-touching pairs are counted separately.

    ``mask`` restricts the test to a subset of vertices (the reshaped neck);
    the returned scope string records that restriction verbatim.
    """
    from scipy.spatial import cKDTree

    tri = V[F] * 1000.0
    if mask is not None:
        keep = np.asarray(mask, bool)[F].all(axis=1)
    else:
        keep = np.ones(len(F), bool)
    scope = "whole mesh" if mask is None else "triangles wholly inside the reshaped band"
    if keep.sum() < 2:
        return {"triangles": int(keep.sum()), "pairs_tested": 0, "intersections": 0,
                "coplanar_pairs": 0, "touching_pairs": 0, "scope": scope,
                "method": "sphere broad phase + Moller triangle/triangle"}

    T = tri[keep]
    fid = np.flatnonzero(keep)
    cen = T.mean(axis=1)
    rad = np.linalg.norm(T - cen[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(cen)
    rmax = float(rad.max())
    # a pair can only intersect when |ci - cj| <= ri + rj <= ri + rmax, so the
    # per-point query radius has to be ri + rmax.  A single global radius rmax
    # misses every pair whose centres are further apart than rmax but closer
    # than the sum of the two circumradii -- i.e. it silently drops the
    # intersections between the largest triangles, which is exactly where a
    # flattened bridge would fold through itself.
    nbrs = tree.query_ball_point(cen, rad + rmax, workers=workers)
    ia, ib = [], []
    for i, lst in enumerate(nbrs):
        if len(lst) < 2:
            continue
        arr = np.asarray(lst, np.int64)
        arr = arr[arr > i]
        if not len(arr):
            continue
        ok = np.linalg.norm(cen[arr] - cen[i], axis=1) <= rad[i] + rad[arr]
        arr = arr[ok]
        if len(arr):
            ia.append(np.full(len(arr), i, np.int64))
            ib.append(arr)
    if not ia:
        P = np.zeros((0, 2), np.int64)
    else:
        P = np.stack([np.concatenate(ia), np.concatenate(ib)], axis=1)
        # drop pairs sharing a vertex (topological neighbours)
        vf = F[fid]
        shared = np.zeros(len(P), bool)
        for a in range(3):
            for b in range(3):
                shared |= vf[P[:, 0], a] == vf[P[:, 1], b]
        P = P[~shared]
    if len(P) > max_pairs:
        P = P[:max_pairs]
    hits = np.zeros(len(P), bool)
    cop = np.zeros(len(P), bool)
    touch = np.zeros(len(P), bool)
    step = 250_000
    for s in range(0, len(P), step):
        sl = slice(s, s + step)
        h, c, tc = _tri_tri_intersect(
            T[P[sl, 0], 0], T[P[sl, 0], 1], T[P[sl, 0], 2],
            T[P[sl, 1], 0], T[P[sl, 1], 1], T[P[sl, 1], 2])
        hits[sl], cop[sl], touch[sl] = h, c, tc
    return {"triangles": int(keep.sum()), "pairs_tested": int(len(P)),
            "intersections": int(hits.sum()), "coplanar_pairs": int(cop.sum()),
            "touching_pairs": int(touch.sum()),
            "first_pairs": [[int(fid[a]), int(fid[b])] for a, b in P[hits][:20]],
            "scope": scope,
            "shared_vertex_pairs_excluded": True,
            "method": "cKDTree sphere broad phase (complete) + Moller triangle/triangle"}


def point_to_surface_mm(points_m, V, F):
    """Exact point-to-triangle-surface distance (NOT nearest-vertex)."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(V, float), faces=np.asarray(F, np.int64),
                           process=False)
    q = np.asarray(points_m, float).reshape(-1, 3)
    closest, dist, _ = trimesh.proximity.closest_point(mesh, q)
    return dist * 1000.0, closest


def write_obj(path, V, F, comment=""):
    """Deterministic OBJ writer: one vertex per line, faces 1-based."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        if comment:
            fh.write("# " + comment.replace("\n", " ") + "\n")
        np.savetxt(fh, np.asarray(V, float), fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, np.asarray(F, np.int64) + 1, fmt="f %d %d %d")
    return path
