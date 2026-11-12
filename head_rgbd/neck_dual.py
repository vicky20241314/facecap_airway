"""Two-feature continuous reference appearance (0.7).

The reference (Y up, +Z anterior) has two vertically aligned anterior features
on the neck midline:
    feature_A: upper/main, x~0, y~-105.5 mm;
    feature_B: lower/secondary, x~0, y~-127.0 mm; B/A height ~0.37,
               saddle/A ~0.29, both measured on a sigma=8 mm Gaussian
               background of the raw front surface.
The previous single-feature route sampled y in [-125,-81], gated to zero at and
below -125, restricted peak selection to y in [-116,-94], and then fitted one
C2 bump; feature_B was cut before the fit.

This module bakes the measured two-feature midline profile as a smooth control
curve (not two independent spheres) and couples it to a lateral width that
blends between the measured A and B widths.  The target field is one connected
non-negative C2 surface; the saddle is positive and shallow.  All dimensions
are scaled together to the target neck; the lower support is constrained by the
true mesh bottom cut and the protection field, not by the old visual root
heuristic (the root still informs the neck-length mapping).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from head_rgbd import neck_continuous as nc
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_reference_auto as nra
from head_rgbd import neck_subject_v3 as v3
from head_rgbd.mesh_integrity import edge_records, load_obj

NECK_METHOD_DUAL = "two_feature_reference_appearance"
GEOMETRY_SOURCE_DUAL = "reference_driven_appearance"
# compatibility names used by the unified public delivery branch
NECK_METHOD_CONTINUOUS = NECK_METHOD_DUAL
GEOMETRY_SOURCE_CONTINUOUS = GEOMETRY_SOURCE_DUAL
na = nc.na
TAPER_MM = nc.TAPER_MM
MIN_CLEARANCE_MM = nc.MIN_TEN_CLEARANCE_MM
DEFAULT_ALPHA = nc.DEFAULT_ALPHA
SIGMA_MM = 8.0


def _smoothstep(x):
    t = np.clip(x, 0.0, 1.0)
    return t ** 3 * (10.0 + t * (-15.0 + 6.0 * t))


def _lateral(u, half, tip):
    half = np.maximum(half, 1e-3)
    tip = np.maximum(tip, 1e-3)
    e = tip / half
    s = u / half
    bracket = 1.0 - (np.sqrt(s * s + e * e) - e)
    support_end = 1.0 + e
    flare = 0.22 * support_end
    win = nc.na._quintic_window((np.abs(s) - (support_end - flare)) / np.maximum(flare, 1e-6))
    return np.clip(bracket, 0.0, None) * win


def _fit_lateral(row, x):
    edge = float(np.nanmedian(np.r_[row[np.abs(x) >= 22.0], row[np.abs(x) <= -22.0]]))
    h = np.maximum(row - edge, 0.0)
    peak = float(np.max(h)) if len(h) else 0.0
    if peak <= 1e-6:
        return 16.0, 4.0, 0.0, edge
    hn = h / peak
    take = (np.abs(x) <= 24.0) & np.isfinite(hn)
    try:
        fit = least_squares(lambda p: _lateral(x[take], p[0], p[1]) - hn[take],
                            [14.0, 4.0], bounds=([5.0, .5], [26.0, 12.0]))
        half, tip = float(fit.x[0]), float(fit.x[1])
        rms = float(np.sqrt(np.mean((_lateral(x[take], half, tip) - hn[take]) ** 2)))
    except Exception:
        half, tip, rms = 16.0, 4.0, None
    return half, tip, rms, edge


def extract_two_feature(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    V, F = load_obj(path)
    V = V * 1000.0
    N = npn.vertex_normals(V / 1000.0, F)
    x = np.arange(-30.0, 30.01, 0.5)
    y = np.arange(-150.0, -87.99, 0.5)
    X, Y = np.meshgrid(x, y)
    sel = (N[:, 2] > .05) & (np.abs(V[:, 0]) < 40.) & (V[:, 1] > -155.) & (V[:, 1] < -85.)
    Z = LinearNDInterpolator(V[sel, :2], V[sel, 2])(X, Y)
    valid = np.isfinite(Z)
    if valid.mean() < .98:
        raise ValueError("reference anterior height field has gaps")
    R = Z - gaussian_filter(Z, SIGMA_MM / .5)
    prof = np.nanmedian(R[:, np.abs(x) <= 2.0], axis=1)
    peaks, props = find_peaks(prof, prominence=.2, distance=int(18 / .5))
    if len(peaks) < 2:
        raise ValueError("reference does not expose two anterior features")
    order = peaks[np.argsort(-prof[peaks])]
    a = int(order[0])
    b_candidates = [p for p in order[1:] if 12.0 <= y[a] - y[p] <= 35.0]
    if not b_candidates:
        raise ValueError("reference second feature is not in the expected lower band")
    b = int(b_candidates[0])
    lo, hi = sorted((a, b))
    saddle_i = int(lo + np.argmin(prof[lo:hi + 1]))
    if prof[b] / prof[a] < .15 or prof[saddle_i] <= 0:
        raise ValueError("reference second feature is too weak or the saddle is not positive")
    # support: the positive profile extent on both sides, with a 2 mm margin
    y_top = None
    for yy in np.arange(y[a], y.max() + 1e-9, .5):
        if np.interp(yy, y, prof, left=0., right=0.) <= 0.0:
            y_top = float(yy)
            break
    if y_top is None:
        y_top = float(y[a] + 10.0)
    y_bottom = None
    for yy in np.arange(y[b], y.min() - 1e-9, -.5):
        if np.interp(yy, y, prof, left=0., right=0.) <= 0.0:
            y_bottom = float(yy)
            break
    if y_bottom is None:
        y_bottom = float(y[b] - 12.0)
    y_top = float(min(y_top + 2.0, y[a] + 16.0))
    y_bottom = float(max(y_bottom - 2.0, y[b] - 18.0))
    v_top = y_top - float(y[a])
    v_bottom = y_bottom - float(y[a])
    v_b = float(y[b] - y[a])
    v_s = float(y[saddle_i] - y[a])
    if not (v_top > 5.0 and v_bottom < -20.0 and v_bottom < v_b - 2.0):
        raise ValueError("degenerate two-feature support")
    # measured profile, normalized to A, with a C2 end window
    vv = np.linspace(v_bottom, v_top, 321)
    hh = np.maximum(np.interp(vv, y - y[a], prof, left=0., right=0.), 0.0)
    t = (v_top - vv) / (v_top - v_bottom)
    window = nc.nra.smooth_gate(np.clip(t / .10, 0, 1)) * \
             nc.nra.smooth_gate(np.clip((1 - t) / .10, 0, 1))
    hh = hh * window
    order = np.argsort(t)
    t, hh = t[order], hh[order]
    t_a = (v_top - 0.0) / (v_top - v_bottom)
    h_at_a = float(np.interp(t_a, t, hh))
    if h_at_a <= 1e-9:
        raise ValueError("degenerate reference profile at feature A")
    hh = hh / h_at_a
    # lateral fits at the two feature rows
    rows = {}
    for name, yy in (("A", float(y[a])), ("B", float(y[b]))):
        row = np.nanmedian(R[np.abs(y - yy) <= 1.0], axis=0)
        half, tip, rms, edge = _fit_lateral(row, x)
        rows[name] = {"half_mm": round(half, 3), "tip_mm": round(tip, 3),
                      "rms": None if rms is None else round(rms, 4),
                      "row_peak_mm": round(float(np.nanmax(row)), 3),
                      "row_edge_mm": round(edge, 3),
                      "w50_row_mm": round(float(np.ptp(x[np.maximum(row - edge, 0) >=
                                                          .5 * max(np.max(row - edge), 1e-9)])), 2)}
    ref_width = float(nra.extract_reference(path)[0]["reference_neck_width_mm"])
    return {
        "source_path": str(path.resolve()), "source_sha256": nc.sha256_file(path),
        "axes": "Y up, +Z anterior", "units": "m",
        "background": {"method": f"raw front z minus Gaussian sigma={SIGMA_MM:g} mm",
                       "sigma_mm": SIGMA_MM, "window_x_mm": [-30, 30],
                       "window_y_mm": [-150, -88]},
        "feature_A": {"name": "reference_feature_A", "x_mm": 0.0,
                      "y_mm": round(float(y[a]), 2), "height_mm": round(float(prof[a]), 3),
                      "v_mm": 0.0},
        "feature_B": {"name": "reference_feature_B", "x_mm": 0.0,
                      "y_mm": round(float(y[b]), 2), "height_mm": round(float(prof[b]), 3),
                      "v_mm": round(v_b, 2)},
        "saddle": {"y_mm": round(float(y[saddle_i]), 2), "v_mm": round(v_s, 2),
                   "height_mm": round(float(prof[saddle_i]), 3),
                   "relative_to_A": round(float(prof[saddle_i] / prof[a]), 4),
                   "B_above_saddle_mm": round(float(prof[b] - prof[saddle_i]), 3)},
        "relative": {"B_over_A": round(float(prof[b] / prof[a]), 4),
                     "centre_distance_mm": round(abs(v_b), 2),
                     "A_over_saddle": round(float(prof[a] - prof[saddle_i]), 3),
                     "B_over_saddle": round(float(prof[b] - prof[saddle_i]), 3)},
        "support": {"v_top_mm": round(v_top, 3), "v_bottom_mm": round(v_bottom, 3),
                    "y_top_mm": round(y_top, 2), "y_bottom_mm": round(y_bottom, 2)},
        "lateral": rows,
        "reference_neck_width_mm": ref_width,
        "profile": {"t": t.tolist(), "h": hh.tolist()},
        "s_ref": round(((-81.0) - float(y[a])) / ((-81.0) - (-125.0)), 3),
        "old_loss": {"old_roi_y": [-125, -81], "old_gate_zero_at_or_below_y": -125,
                     "old_core_y": [-116, -94],
                     "why": "feature_B centre y=-127 is outside the old ROI and gate; "
                            "its upper wing inside the ROI is faded to zero by the lower "
                            "9 mm gate, and the old core never selects it. The single C2 "
                            "profile then cannot represent two peaks."},
        "sensitivity": _feature_sensitivity(V, F, x, y),
        "editing_mode": True, "geometry_source": GEOMETRY_SOURCE_DUAL,
        "subject_observation_matched": False,
        "note": "positive central features only; the sigma=8 background is an appearance "
                "definition, not an anatomical measurement. The lateral x=+/-20 mm, y=-75 mm "
                "peaks are under-chin/jaw contours and are excluded; the old 1.55 mm islands "
                "at |x|~26-31 mm are lateral shoulder residuals, not feature_B.",
    }


def _feature_sensitivity(V, F, x, y):
    """Same raw z field at three Gaussian scales; records the B/A and saddle/A."""
    N = npn.vertex_normals(V / 1000.0, F)
    X, Y = np.meshgrid(x, y)
    sel = (N[:, 2] > .05) & (np.abs(V[:, 0]) < 40.) & (V[:, 1] > -155.) & (V[:, 1] < -85.)
    Z = LinearNDInterpolator(V[sel, :2], V[sel, 2])(X, Y)
    out = {}
    for sigma in (6., 8., 10.):
        R = Z - gaussian_filter(Z, sigma / .5)
        prof = np.nanmedian(R[:, np.abs(x) <= 2.0], axis=1)
        peaks, _ = find_peaks(prof, prominence=.2, distance=int(18 / .5))
        if len(peaks) < 2:
            out[str(sigma)] = {"features": 1}
            continue
        order = peaks[np.argsort(-prof[peaks])]
        a = int(order[0])
        bs = [p for p in order[1:] if 12 <= y[a] - y[p] <= 35]
        if not bs:
            out[str(sigma)] = {"features": 1}
            continue
        b = int(bs[0]); lo, hi = sorted((a, b)); s = int(lo + np.argmin(prof[lo:hi + 1]))
        out[str(sigma)] = {"features": 2, "A_y": round(float(y[a]), 1),
                           "B_y": round(float(y[b]), 1),
                           "B_over_A": round(float(prof[b] / prof[a]), 4),
                           "saddle_over_A": round(float(prof[s] / prof[a]), 4)}
    return out


@dataclass
class DualDesign:
    height_mm: float
    v_top_mm: float
    v_bottom_mm: float
    v_b_mm: float
    t: np.ndarray
    h: np.ndarray
    half_a_mm: float
    half_b_mm: float
    tip_a_mm: float
    tip_b_mm: float
    source_sigma_mm: float = SIGMA_MM

    def __post_init__(self):
        self._spline = CubicSpline(self.t, self.h, extrapolate=False)
        self.t_a = (self.v_top_mm - 0.0) / (self.v_top_mm - self.v_bottom_mm)
        self.t_b = (self.v_top_mm - self.v_b_mm) / (self.v_top_mm - self.v_bottom_mm)

    @property
    def support_lateral_mm(self) -> float:
        return max(self.half_a_mm + self.tip_a_mm, self.half_b_mm + self.tip_b_mm)

    @property
    def vertical_extent_mm(self) -> float:
        return max(abs(self.v_top_mm), abs(self.v_bottom_mm))

    def scaled(self, qf: float) -> "DualDesign":
        return self.scaled_axes(qf, qf, qf)

    def scaled_axes(self, vertical: float, lateral: float, height: float) -> "DualDesign":
        return DualDesign(self.height_mm * height,
                          self.v_top_mm * vertical, self.v_bottom_mm * vertical,
                          self.v_b_mm * vertical, self.t.copy(), self.h.copy(),
                          self.half_a_mm * lateral, self.half_b_mm * lateral,
                          self.tip_a_mm * lateral, self.tip_b_mm * lateral,
                          self.source_sigma_mm)

    def profile(self, v):
        t = (self.v_top_mm - np.asarray(v)) / (self.v_top_mm - self.v_bottom_mm)
        p = self._spline(np.clip(t, 0., 1.))
        return np.where((t >= 0.) & (t <= 1.), np.maximum(np.nan_to_num(p), 0.), 0.)

    def evaluate(self, u, v):
        t = (self.v_top_mm - np.asarray(v)) / (self.v_top_mm - self.v_bottom_mm)
        inside = (t >= 0.) & (t <= 1.)
        p = self.profile(v)
        frac = np.clip((t - self.t_a) / max(self.t_b - self.t_a, 1e-9), 0., 1.)
        blend = _smoothstep(frac)
        half = self.half_a_mm + (self.half_b_mm - self.half_a_mm) * blend
        tip = self.tip_a_mm + (self.tip_b_mm - self.tip_a_mm) * blend
        cross = _lateral(np.asarray(u), half, tip)
        return np.where(inside, self.height_mm * p * cross, 0.)

    def feature_positions(self):
        return {"A_v_mm": 0.0, "B_v_mm": float(self.v_b_mm),
                "saddle_v_mm": self._saddle_v(),
                "A_height_mm": float(self.height_mm),
                "B_height_mm": float(self.height_mm * np.interp(self.t_b, self.t, self.h)),
                "saddle_height_mm": float(self.height_mm * np.interp(self.t_b, self.t, self.h) * 0)}

    def _saddle_v(self):
        t = np.linspace(self.t_a, self.t_b, 101)
        h = self._spline(t)
        k = int(np.argmin(h))
        v = self.v_top_mm - t[k] * (self.v_top_mm - self.v_bottom_mm)
        return float(v)

    def saddle_height_mm(self):
        v = self._saddle_v()
        return float(self.profile(v))


def design_from_morph(morph: dict[str, Any], qf: float = 1.0) -> DualDesign:
    prof = morph["profile"]
    design = DualDesign(
        height_mm=morph["feature_A"]["height_mm"],
        v_top_mm=morph["support"]["v_top_mm"],
        v_bottom_mm=morph["support"]["v_bottom_mm"],
        v_b_mm=morph["feature_B"]["v_mm"],
        t=np.asarray(prof["t"], float), h=np.asarray(prof["h"], float),
        half_a_mm=morph["lateral"]["A"]["half_mm"],
        half_b_mm=morph["lateral"]["B"]["half_mm"],
        tip_a_mm=morph["lateral"]["A"]["tip_mm"],
        tip_b_mm=morph["lateral"]["B"]["tip_mm"])
    return design.scaled(qf)


# ---------------------------------------------------------------------------
# Target, placement, refinement, application and audits
# ---------------------------------------------------------------------------

def _bottom_cut_v(base_state, frame, faces):
    edges, _, counts, _, _ = edge_records(faces)
    bnd = np.unique(edges[counts == 1])
    uv = frame.uv(base_state["vertices"][bnd])
    front = (base_state["normals"][bnd] @ frame.n) > .2
    v = uv[:, 1]
    take = front & (v < -5.)
    return float(np.max(v[take])) if take.any() else float(np.min(v))


def _refined_mesh(base, faces, frames, design):
    pad = int(np.ceil((design.vertical_extent_mm + 12.0) / 1000.0 / .005)) * .005
    pad = max(pad, .030)
    centroid = {s: base[s]["vertices"][faces].mean(axis=1) for s in base}
    mask1 = np.zeros(len(faces), bool)
    for s in base:
        mask1 |= np.linalg.norm(centroid[s] - frames[s].origin, axis=1) < pad
    sub1 = v3.subdivide_local(base["closed"]["vertices"], faces, face_mask=mask1)
    l1 = sub1["midpoint_edges"]
    r1 = {s: nc.na._rebuild_refined(base[s]["vertices"], l1, []) for s in base}
    pad2 = max(.017, pad * .82)
    mask2 = np.zeros(len(sub1["faces"]), bool)
    for s in base:
        c = r1[s][sub1["faces"]].mean(axis=1)
        mask2 |= np.linalg.norm(c - frames[s].origin, axis=1) < pad2
    sub2 = v3.subdivide_local(r1["closed"], sub1["faces"], face_mask=mask2)
    l2 = sub2["midpoint_edges"]
    final = {s: nc.na._rebuild_refined(base[s]["vertices"], l1, l2) for s in base}
    return {"faces": sub2["faces"], "l1": l1, "l2": l2,
            "refined": final, "base_count": len(base["closed"]["vertices"])}


def _protected(base, refined, l1, l2, base_count, state):
    V = refined[state]
    prot = np.zeros(len(V), bool)
    prot[:base_count] = base[state]["protection"]
    off = base_count
    for k, (i, j) in enumerate(l1):
        prot[off + k] = base[state]["protection"][i] or base[state]["protection"][j]
    off2 = off + len(l1)
    for k, (i, j) in enumerate(l2):
        prot[off2 + k] = prot[i] or prot[j]
    return V, prot


def _field(vertices, faces, frame, design, protected, alpha, mode, local_normals):
    uv = frame.uv(vertices)
    p = design.evaluate(uv[:, 0], uv[:, 1])
    ids = np.flatnonzero(protected)
    dist = cKDTree(vertices[ids]).query(vertices)[0] * 1000. if len(ids) else \
        np.full(len(vertices), 1e9)
    gate = nra.smooth_gate(dist / TAPER_MM) * v3._region_window(vertices, frame, protected)
    D, direction = nc._direction_field(vertices, faces, frame, local_normals, alpha, mode)
    applied = p * gate
    disp = applied[:, None] * D * 1e-3
    shaped = vertices + disp
    normal = np.einsum("ij,ij->i", disp, local_normals) * 1000.
    tang = np.linalg.norm(disp - normal[:, None] * 1e-3 * local_normals, axis=1) * 1000.
    return {"vertices": shaped, "sculpt_mm": p, "gate": gate, "applied_mm": applied,
            "direction": D, "normal_mm": normal, "tangential_mm": tang,
            "distance_to_protected_mm": dist, "direction_audit": direction,
            "front_mask": (local_normals @ frame.n) > .2}


def _design_extents(design):
    lo = design.v_bottom_mm - 2.
    hi = design.v_top_mm + 2.
    uu = np.linspace(-design.support_lateral_mm - 2., design.support_lateral_mm + 2., 241)
    vv = np.linspace(lo, hi, 321)
    U, V = np.meshgrid(uu, vv)
    P = design.evaluate(U, V)
    out = {"max": float(P.max())}
    for frac in (.1, .5):
        m = P >= frac * design.height_mm
        out[f"u{int(frac*100)}"] = [float(U[m].min()), float(U[m].max())] if m.any() else None
        out[f"v{int(frac*100)}"] = [float(V[m].min()), float(V[m].max())] if m.any() else None
    return out


def _center(base, scale, s):
    return nc._find_center(base, scale, s)


def _search(base, scale, morph, faces):
    s_ref = morph["s_ref"]
    q = scale.W_neck_mm / morph["reference_neck_width_mm"]
    cands = []
    for f in (1.0, .96, .92, .88, .84, .80, .76, .72, .68, .64, .60):
        design = design_from_morph(morph, q * f)
        ext = _design_extents(design)
        v10 = ext.get("v10")
        if not v10:
            continue
        for s in np.arange(max(.38, s_ref - .10), min(.82, s_ref + .24) + 1e-9, .02):
            frames, audit = _center(base, scale, s)
            if frames is None:
                continue
            ok = True
            details = {}
            cut = {}
            for st in base:
                V = base[st]["vertices"]
                fr = frames[st]
                uv = fr.uv(V)
                front = (base[st]["normals"] @ fr.n) > .2
                p = design.evaluate(uv[:, 0], uv[:, 1])
                p10 = (p >= .1 * design.height_mm) & front
                if not p10.any():
                    ok = False
                    break
                ids = np.flatnonzero(base[st]["protection"])
                dist = cKDTree(V[ids]).query(V)[0] * 1000. if len(ids) else np.full(len(V), 1e9)
                clear = float(dist[p10].min())
                bottom = float(uv[p10, 1].min())
                cutv = _bottom_cut_v(base[st], fr, faces)
                details[st] = {"clearance_mm": round(clear, 2), "p10_bottom_mm": round(bottom, 2),
                               "bottom_cut_mm": round(cutv, 2), "front_dot": round(float(
                                   (base[st]["normals"][p10] @ fr.n).min()), 3)}
                cut[st] = cutv
                if clear < MIN_CLEARANCE_MM or bottom < cutv + 8.0:
                    ok = False
                    break
            if not ok:
                continue
            fr = frames["closed"]
            crease = float((scale.s0_point_camera - fr.origin) @ fr.v_axis * 1000)
            if ext["v10"][1] > crease - 1.0:
                continue
            cands.append({"f": f, "s": float(s), "design": design, "frames": frames,
                          "center": audit, "details": details, "extents": ext,
                          "crease_mm": crease, "bottom_cut_mm": cut})
    for c in cands:
        ratios = []
        for st, d in c["details"].items():
            ratios.append(d["clearance_mm"] / MIN_CLEARANCE_MM)
            ratios.append((d["p10_bottom_mm"] - d["bottom_cut_mm"]) / 8.0)
        c["coarse_score"] = min(ratios) if ratios else 0.0
    cands.sort(key=lambda c: (c["coarse_score"] >= 1.0, c["f"],
                              min(c["coarse_score"], 1.5),
                              -abs(c["s"] - s_ref)), reverse=True)
    return cands


def _features_from_field(uv, applied, front, design):
    """Detect the A/B longitudinal maxima and local saddle on the delivered field."""
    m = front & (np.abs(uv[:, 0]) <= 2.5)
    if m.sum() < 10:
        return {"detected": False, "reason": "no central front samples"}
    v = uv[m, 1]
    a = applied[m]
    bins = np.arange(-45., 20.1, .5)
    prof = np.full(len(bins), np.nan)
    for i, vc in enumerate(bins):
        take = np.abs(v - vc) <= .5
        if take.any():
            prof[i] = float(np.median(a[take]))
    peaks, props = find_peaks(prof, prominence=.03, distance=int(12. / .5))
    if len(peaks) < 2:
        return {"detected": False, "reason": "fewer than two longitudinal maxima"}
    order = peaks[np.argsort(-prof[peaks])]
    ia = int(order[0])
    ibs = [p for p in order[1:] if 8.0 <= bins[ia] - bins[p] <= 40.0]
    if not ibs:
        return {"detected": False, "reason": "no lower secondary maximum"}
    ib = int(ibs[0])
    lo, hi = sorted((ia, ib))
    is_ = int(lo + np.argmin(prof[lo:hi + 1]))
    if prof[is_] <= 0 or prof[ib] <= 0:
        return {"detected": False, "reason": "saddle or secondary not positive"}
    # lateral widths at the two peaks (front field, local edge baseline)
    widths = {}
    for name, idx in (("A", ia), ("B", ib)):
        vc = bins[idx]
        row = uv[(np.abs(uv[:, 1] - vc) <= 1.5) & front]
        val = applied[(np.abs(uv[:, 1] - vc) <= 1.5) & front]
        uu = np.arange(-35., 35.1, .5)
        rp = np.full(len(uu), np.nan)
        for i, uc in enumerate(uu):
            take = np.abs(row[:, 0] - uc) <= .5
            if take.any():
                rp[i] = float(np.median(val[take]))
        if np.isfinite(rp).sum() < 10:
            widths[name] = None
            continue
        edge = float(np.nanmedian(np.r_[rp[np.abs(uu) >= 22], rp[np.abs(uu) <= -22]]))
        half = edge + .5 * (np.nanmax(rp) - edge)
        take = np.flatnonzero(rp >= half)
        widths[name] = {"u_mm": [round(float(uu[take.min()]), 2), round(float(uu[take.max()]), 2)],
                        "w50_mm": round(float(uu[take.max()] - uu[take.min()]), 2)} \
            if len(take) else None
    # stability of the B shoulder at lateral offsets: a shoulder can be a
    # local maximum or a positive shelf above the inter-feature saddle.
    stable = 0
    for off in (0., 3., -3., 5., -5.):
        take = (np.abs(uv[:, 0] - off) <= 1.0) & front
        if take.sum() < 8:
            continue
        vv = uv[take, 1]; aa = applied[take]
        bp = np.arange(-45., 20.1, .5)
        pp = np.full(len(bp), np.nan)
        for i, vc in enumerate(bp):
            k = np.abs(vv - vc) <= .5
            if k.any():
                pp[i] = float(np.median(aa[k]))
        bv = float(bins[ib]); sv = float(bins[is_])
        bh = np.nanmedian(pp[np.abs(bp - bv) <= 1.5])
        sh = np.nanmedian(pp[np.abs(bp - sv) <= 1.5])
        if np.isfinite(bh) and np.isfinite(sh) and bh - sh > .02:
            stable += 1
    return {
        "detected": True,
        "A": {"v_mm": round(float(bins[ia]), 2), "height_mm": round(float(prof[ia]), 3),
              "prominence_mm": round(float(props["prominences"][list(peaks).index(ia)]), 3),
              "width": widths.get("A")},
        "B": {"v_mm": round(float(bins[ib]), 2), "height_mm": round(float(prof[ib]), 3),
              "prominence_mm": round(float(props["prominences"][list(peaks).index(ib)]), 3),
              "width": widths.get("B")},
        "saddle": {"v_mm": round(float(bins[is_]), 2), "height_mm": round(float(prof[is_]), 3)},
        "relative": {"B_over_A": round(float(prof[ib] / prof[ia]), 4),
                     "saddle_over_A": round(float(prof[is_] / prof[ia]), 4),
                     "centre_distance_mm": round(float(abs(bins[ib] - bins[ia])), 2)},
        "B_stable_offsets": int(stable),
        "design": {"A_v_mm": 0., "B_v_mm": float(design.v_b_mm),
                   "A_height_mm": float(design.height_mm),
                   "B_height_mm": float(design.height_mm * design.profile(design.v_b_mm)),
                   "saddle_height_mm": float(design.height_mm * design.saddle_height_mm())},
    }


def audit_state(plan, state):
    item = plan["states"][state]
    frame = item["frame"]
    Vr = np.asarray(item["vertices"], float)
    run = item["run"]
    shaped = np.asarray(run["vertices"], float)
    base = plan["base_states"][state]
    base_faces = plan["base_faces"]
    front = run["front_mask"]
    uv = frame.uv(Vr)
    analysis = nra.analyse_applied(
        {"uv_mm": uv, "applied_mm": run["applied_mm"], "refined_vertices": Vr,
         "frame": frame}, plan["faces"])
    features = _features_from_field(uv, run["applied_mm"], front, plan["design"])
    base_audit = nc._base_relative_audit(
        base["vertices"], base["normals"], shaped[:len(base["vertices"])], base_faces)
    refined_normals = npn.vertex_normals(Vr, plan["faces"])
    d_ref = (shaped - Vr) * 1000.
    comp = np.einsum("ij,ij->i", d_ref, refined_normals)
    tri = Vr[plan["faces"]]
    area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 1e6
    neg = comp[plan["faces"]].mean(axis=1) < -.05
    p10 = front & (run["applied_mm"] >= .1 * max(float(run["applied_mm"].max()), 1e-9))
    clear = float(run["distance_to_protected_mm"][p10].min()) if p10.any() else None
    gate = float(run["gate"][p10].min()) if p10.any() else None
    cutv = _bottom_cut_v(base, frame, base_faces)
    v = uv[p10, 1] if p10.any() else np.array([0.])
    return {
        "state": state,
        "applied_peak_mm": round(float(run["applied_mm"].max()), 3),
        "normal_peak_mm": round(float(run["normal_mm"].max()), 3),
        "tangential_max_mm": round(float(run["tangential_mm"].max()), 3),
        "base_relative": base_audit,
        "refined_relative": {"min_outward_mm": round(float(comp.min()), 3),
                             "negative_vertices_below_tolerance": int((comp < -.05).sum()),
                             "negative_face_area_mm2": round(float(area[neg].sum()), 2)},
        "features": features,
        "levels": {k: {"component_count": analysis["levels"][k]["component_count"],
                       "central": analysis["levels"][k]["central"],
                       "global_width_mm": analysis["levels"][k]["global_width_mm"],
                       "global_length_mm": analysis["levels"][k]["global_length_mm"]}
                   for k in ("0.1", "0.5")},
        "analysis": analysis,
        "support": {"p10_vertices": int(p10.sum()),
                    "clearance_min_mm": None if clear is None else round(clear, 2),
                    "gate_min": None if gate is None else round(gate, 4),
                    "inside_protected": int((p10 & item["protected"]).sum()),
                    "v_top_mm": round(float(v.max()), 2),
                    "v_bottom_mm": round(float(v.min()), 2),
                    "bottom_cut_mm": round(cutv, 2),
                    "bottom_margin_mm": round(float(v.min()) - cutv, 2)},
        "bottom_cut_mm": round(cutv, 2),
    }


def build_states(*, states=None, references, dense_vertices, dense_faces, reference_path,
                 alpha=nc.DEFAULT_ALPHA, direction_modes=("constant", "smooth"),
                 fixed_center_vertex=None, q_override=None, f_override=None,
                 require_support=True, morph_override=None):
    morph = morph_override if morph_override is not None else extract_two_feature(reference_path)
    base, scale = nc.measure_target(references, dense_vertices, dense_faces)
    q = float(q_override) if q_override is not None else \
        scale.W_neck_mm / morph["reference_neck_width_mm"]
    if fixed_center_vertex is not None:
        vertex = int(fixed_center_vertex)
        frames = {st: nc._state_frame(base[st], vertex, "historical_center_vertex")
                  for st in base}
        design = design_from_morph(morph, q * (f_override or 1.0))
        cands = [{"f": f_override or 1.0, "s": None, "design": design, "frames": frames,
                  "center": {"vertex": vertex, "source": "historical_center_vertex"}}]
    else:
        cands = _search(base, scale, morph, dense_faces)
        if not cands:
            raise RuntimeError("no two-feature placement fits the target support")
    tried = []
    for cand in cands[:16]:
        design = cand["design"]
        frames = cand["frames"]
        ref = _refined_mesh(base, dense_faces, frames, design)
        faces = ref["faces"]; l1, l2, base_count = ref["l1"], ref["l2"], ref["base_count"]
        states = {}
        for st in base:
            verts, prot = _protected(base, ref["refined"], l1, l2, base_count, st)
            n = npn.vertex_normals(verts, faces)
            runs = {}
            for dm in direction_modes:
                run = _field(verts, faces, frames[st], design, prot, alpha, dm, n)
                moved = np.abs(run["applied_mm"]) > 1e-9
                runs[dm] = {"guards": nc.na.geometry_guards(
                    rest=verts, shaped=run["vertices"], faces=faces, moved=moved),
                    "intersections": nra.local_intersections(run["vertices"], faces, moved),
                    "max_tangential_mm": round(float(run["tangential_mm"].max()), 3),
                    "mean_tangential_mm": round(float(run["tangential_mm"].mean()), 3),
                    "min_normal_mm": round(float(run["normal_mm"].min()), 3)}
            states[st] = {"vertices": verts, "protected": prot, "frame": frames[st],
                          "base_vertices": base[st]["vertices"],
                          "base_normals": base[st]["normals"],
                          "protection": base[st]["protection"],
                          "reference": base[st]["reference"],
                          "direction_runs": runs}
        safe = [dm for dm in direction_modes if all(
            states[s]["direction_runs"][dm]["intersections"]["count"] == 0
            and nc.na.accept_guards(states[s]["direction_runs"][dm]["guards"]) for s in base)]
        unsafe = False
        if not safe:
            if require_support:
                tried.append({"f": cand.get("f"), "s": cand.get("s"), "stage": "unsafe"})
                continue
            # causal diagnostic only: export a labelled unsafe candidate
            mode = direction_modes[0]
            unsafe = True
        else:
            mode = min(safe, key=lambda dm: max(
                states[s]["direction_runs"][dm]["mean_tangential_mm"] for s in base))
        for st in base:
            states[st]["run"] = _field(
                states[st]["vertices"], faces, frames[st], design, states[st]["protected"],
                alpha, mode, npn.vertex_normals(states[st]["vertices"], faces))
            moved = np.abs(states[st]["run"]["applied_mm"]) > 1e-9
            states[st]["moved"] = moved
            states[st]["guards"] = nc.na.geometry_guards(
                rest=states[st]["vertices"], shaped=states[st]["run"]["vertices"],
                faces=faces, moved=moved)
            states[st]["intersections"] = nra.local_intersections(
                states[st]["run"]["vertices"], faces, moved)
        placement = {
            **cand.get("center", {}), "source": "proportional_two_feature" if fixed_center_vertex is None
            else "historical_center_vertex",
            "s_requested": morph["s_ref"], "s_chosen": cand.get("s"),
            "fit_scale": cand.get("f"), "q": q, "details": cand.get("details", {}),
            "crease_mm": cand.get("crease_mm"), "bottom_cut_mm": cand.get("bottom_cut_mm"),
            "feature_A_v_mm": 0.0, "feature_B_v_mm": round(float(design.v_b_mm), 2),
            "support_v_mm": [round(float(design.v_bottom_mm), 2), round(float(design.v_top_mm), 2)],
            "design_extents": cand.get("extents"),
            "rule": "A and B share one base binding; the whole design (heights, widths, "
                    "lengths, tip radii, A-B distance) scales together. The 10% contour must "
                    "clear protection by >=8 mm, the A top stays below the submental crease, "
                    "and the B bottom stays 8 mm above the true mesh bottom cut. The visual "
                    "root informs proportion only.",
        }
        provisional = {
            "states": states, "faces": faces, "base_faces": np.asarray(dense_faces, int),
            "base_states": base, "design": design, "morphology": morph,
            "target_scale": scale, "q": q, "level1_edges": l1, "level2_edges": l2,
            "base_count": base_count, "base_vertex_count": base_count,
            "refined_vertices_count": len(states["closed"]["vertices"]),
            "placement": placement, "direction_mode": mode, "alpha": alpha,
            "reference_morphology": morph,
            "direction_comparison": {s: states[s]["direction_runs"] for s in base},
            "design_audit": {"height_mm": round(design.height_mm, 3),
                             "v_top_mm": round(design.v_top_mm, 3),
                             "v_bottom_mm": round(design.v_bottom_mm, 3),
                             "v_b_mm": round(design.v_b_mm, 3),
                             "half_a_mm": round(design.half_a_mm, 3),
                             "half_b_mm": round(design.half_b_mm, 3),
                             "tip_a_mm": round(design.tip_a_mm, 3),
                             "tip_b_mm": round(design.tip_b_mm, 3),
                             "support_lateral_mm": round(design.support_lateral_mm, 3),
                             "source_sigma_mm": design.source_sigma_mm,
                             "reference_feature_A_height_mm": morph["feature_A"]["height_mm"],
                             "reference_feature_B_height_mm": morph["feature_B"]["height_mm"],
                             "reference_B_over_A": morph["relative"]["B_over_A"],
                             "reference_saddle_over_A": morph["saddle"]["relative_to_A"],
                             "reference_centre_distance_mm": morph["relative"]["centre_distance_mm"]},
        }
        audits = {s: audit_state(provisional, s) for s in base}
        support_ok = True
        for s in base:
            a = audits[s]
            if a["support"]["inside_protected"] or a["support"]["clearance_min_mm"] is None \
                    or a["support"]["clearance_min_mm"] < MIN_CLEARANCE_MM \
                    or a["support"]["gate_min"] < .98 \
                    or a["support"]["bottom_margin_mm"] < 8.0:
                support_ok = False
            f = a["features"]
            if not f.get("detected") or f["B"]["height_mm"] <= .05 \
                    or f["relative"]["B_over_A"] < .15 \
                    or f["saddle"]["height_mm"] <= .02 \
                    or f["B_stable_offsets"] < 3:
                support_ok = False
        tried.append({"f": cand.get("f"), "s": cand.get("s"), "support_ok": support_ok,
                      "clearance": {s: audits[s]["support"]["clearance_min_mm"] for s in base},
                      "bottom_margin": {s: audits[s]["support"]["bottom_margin_mm"] for s in base},
                      "features": {s: audits[s]["features"].get("relative") for s in base},
                      "stable": {s: audits[s]["features"].get("B_stable_offsets") for s in base},
                      "gate": {s: audits[s]["support"]["gate_min"] for s in base},
                      "inside": {s: audits[s]["support"]["inside_protected"] for s in base}})
        if support_ok or not require_support:
            provisional["audits"] = audits
            provisional["accepted"] = bool(support_ok and not unsafe and all(
                nc.na.accept_guards(states[s]["guards"])
                and states[s]["intersections"]["count"] == 0 for s in base))
            provisional["unsafe_diagnostic"] = bool(unsafe)
            provisional["tried"] = tried
            return provisional
    raise RuntimeError("no two-feature candidate passed the refined A/B/support audit: "
                       + repr(tried[-3:]))
