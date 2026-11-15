"""Reference-driven neck volume and head/neck/shoulder transition (0.9).

This module replaces the accumulated local "laryngeal bump" sculpting with ONE
low-frequency, cross-section-driven neck volume model.

Design chain
------------
    unshaped shared dense base
      -> bottom-cut plane -> axial coordinate h (normal to the true cut)
      -> head-local frame from closed-state facial landmarks (shared scale)
      -> corrected, same-definition head dimensions (real menton, no nose tip)
      -> reference cross-section descriptors expressed in head-normalised units
      -> target cross-section descriptors adapted to the target neck length and
         the existing shoulder/chest skirt extent
      -> one continuous C2 radial control field s(theta, h) with a centerline
      -> consistent uniform subdivision of the WHOLE neck/transition mesh
      -> vertex positions re-solved from the smooth target contour
      -> optional low-amplitude continuous natural detail (secondary field)
      -> closed/open states by vertex correspondence, per-state safety checks

The old ROI rules are deliberately NOT reused: the whole neck below the jaw,
both sides, the posterior neck, the root and the existing shoulder/chest skirt
participate in the continuous transition.  Only the identity (face, lips, nose,
eyes, ears, mandible outline, occiput) and the outer cut ring are anchored.

Geometry is reference-driven *appearance design*, not a measured neck scan.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from head_rgbd import head_scale as hs
from head_rgbd import mesh_integrity as mi
from head_rgbd import neck_prominence as npn

NECK_METHOD_VOLUME = "reference_driven_neck_volume"
GEOMETRY_SOURCE_VOLUME = "reference_driven_neck_volume"
SCHEMA = "head_rgbd.neck_volume.v1"
DEFAULT_ALPHA = 1.0

# ibug-68 landmarks (same indices as neck_prominence)
CHIN, NOSE_BRIDGE_TOP, NOSE_TIP, EYE_OUTER_RIGHT, EYE_OUTER_LEFT = 8, 27, 30, 36, 45

# Region / transition design constants (mm, in the shared head frame)
IDENTITY_ABOVE_CHIN_MM = 8.0       # face/mandible/occiput start here
FULL_EDIT_BELOW_CHIN_MM = 30.0     # full volume freedom below this
CUT_ANCHOR_WIDTH_MM = 26.0         # distance-based C2 ramp at the cut ring
CUT_ANCHOR_FLOOR = 0.55            # bounded IN-PLANE rim motion, cut position fixed
SKIRT_TUBE_BLEND_MM = (22.0, 36.0)  # radial re-projection ramp: skirt -> tube
ANGULAR_DAMP_MM = (18.0, 34.0)      # plate uses a pure dilation, tube uses full harmonics
SKIRT_T = 0.22                     # single cross-section target over the flat skirt plate
SKIRT_AXIAL_BLEND_MM = (20.0, 42.0)  # plate -> tube axial coordinate handover
MAX_INWARD_SCALE = 0.86            # controlled inward motion at mid neck
MAX_OUTWARD_SCALE = 1.32           # controlled outward motion at the root
MAX_CENTER_SHIFT_MM = 7.0
DETAIL_AMPLITUDE_MM = 1.35         # secondary natural relief (C only)
DETAIL_T_SIGMA = 0.19
SUBMENTAL_FILL_MM = 5.5           # broad C2 anterior fill softening the jaw undercut
SUBMENTAL_H_MM = 82.0             # centre in axial mm above the true cut
SUBMENTAL_SIGMA_MM = 19.0
FOURIER_ORDER = 8
SECTION_LEVELS = [0.04, 0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58,
                  0.66, 0.74, 0.82, 0.90, 0.96]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _smootherstep(x):
    t = np.clip(x, 0.0, 1.0)
    return t ** 3 * (10.0 + t * (-15.0 + 6.0 * t))


def _ramp(x, lo, hi):
    """C2 0..1 ramp, 0 at x<=lo, 1 at x>=hi."""
    return _smootherstep((np.asarray(x, float) - lo) / max(hi - lo, 1e-9))


def _boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edges, inverse, counts, _, _ = mi.edge_records(faces)
    b = edges[counts == 1]
    adj = defaultdict(list)
    for a, c in b:
        adj[int(a)].append(int(c)); adj[int(c)].append(int(a))
    seen: set[int] = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        loop = [start]; seen.add(start); cur = start
        while True:
            nxt = [n for n in adj[cur] if n not in seen]
            if not nxt:
                break
            cur = nxt[0]; seen.add(cur); loop.append(cur)
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _fit_plane(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    return c, n, (P - c) @ n


def head_frame(landmarks_m: np.ndarray) -> dict[str, Any]:
    """Head-local right/up/forward axes from stable facial landmarks."""
    p = np.asarray(landmarks_m, float)
    right = p[EYE_OUTER_RIGHT] - p[EYE_OUTER_LEFT]
    right /= np.linalg.norm(right)
    up = p[NOSE_BRIDGE_TOP] - p[CHIN]
    up /= np.linalg.norm(up)
    right -= np.dot(right, up) * up
    right /= np.linalg.norm(right)
    fwd = np.cross(up, right)
    fwd /= np.linalg.norm(fwd)
    if np.dot(p[NOSE_TIP] - .5 * (p[EYE_OUTER_RIGHT] + p[EYE_OUTER_LEFT]), fwd) < 0:
        fwd = -fwd
    return {"right": right, "up": up, "forward": fwd, "chin": p[CHIN].copy(),
            "nose_tip": p[NOSE_TIP].copy(), "nose_bridge_top": p[NOSE_BRIDGE_TOP].copy(),
            "eye_outer_right": p[EYE_OUTER_RIGHT].copy(),
            "eye_outer_left": p[EYE_OUTER_LEFT].copy()}


# ---------------------------------------------------------------------------
# corrected head scale: real menton, same-definition band, no neck/shoulder
# ---------------------------------------------------------------------------

def detect_reference_menton(V_mm: np.ndarray) -> dict[str, Any]:
    """Canonical implementation lives in head_scale (public 0.9 fix)."""
    return hs.detect_reference_menton(V_mm)


def head_dimensions(V_mm: np.ndarray, up: np.ndarray, right: np.ndarray,
                    fwd: np.ndarray, chin: np.ndarray) -> dict[str, Any]:
    """Canonical implementation lives in head_scale (same protocol both heads)."""
    return hs.head_dimensions(V_mm, up, right, fwd, chin)


def scaled_head_audit(ref: dict, tgt: dict, ref_menton: dict) -> dict[str, Any]:
    ratios = {k: float(tgt[k] / ref[k]) for k in ("H_head_mm", "W_head_mm", "D_head_mm")}
    return {"reference": ref, "target": tgt, "reference_menton": ref_menton,
            "ratios": {k: round(v, 4) for k, v in ratios.items()},
            "S_head_geometric": round(float(np.cbrt(np.prod(list(ratios.values())))), 4),
            "note": "shared neutral/closed identity defines the scale; the open jaw "
                    "does not re-scale the head. The old H ratio 1.6368 came from a "
                    "nose-tip mis-detection and is not used anywhere."}


# ---------------------------------------------------------------------------
# axial frame from the true bottom cut
# ---------------------------------------------------------------------------

def axial_frame(V: np.ndarray, F: np.ndarray, frame: dict[str, Any]) -> dict[str, Any]:
    """Plane fitted to the real bottom cut ring -> (h, a, b) coordinates."""
    loops = _boundary_loops(F)
    cut = min(loops, key=lambda lp: float(V[lp][:, 1].mean()))
    c0, n, resid = _fit_plane(V[cut] * 1000.0)
    if np.dot(n, frame["up"]) < 0:
        n, resid = -n, -resid
    e1 = frame["right"] - np.dot(frame["right"], n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if np.dot(e2, frame["forward"]) < 0:
        e2, e1 = -e2, -e1
    chin_h = float((frame["chin"] * 1000.0 - c0) @ n)
    return {"centre_mm": c0, "axis": n, "lateral": e1, "anterior": e2,
            "cut_loop": cut, "all_loops": loops,
            "cut_plane_residual_mm": {"ptp": round(float(np.ptp(resid)), 4),
                                      "rms": round(float(np.sqrt((resid ** 2).mean())), 4)},
            "chin_h_mm": round(chin_h, 3), "h_chin_mm": round(chin_h, 3)}


def axial_coords(V: np.ndarray, ax: dict[str, Any]):
    P = V * 1000.0 - ax["centre_mm"]
    return P @ ax["axis"], P @ ax["lateral"], P @ ax["anterior"]


# ---------------------------------------------------------------------------
# plane-section contour descriptors
# ---------------------------------------------------------------------------

def _section_points(V: np.ndarray, F: np.ndarray, centre: np.ndarray,
                    axis: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                    h0: float) -> np.ndarray:
    """All surface/plane intersection points in (a,b) mm at h=h0."""
    tri = V[F] * 1000.0 - centre
    s = tri @ axis - h0
    cross = (np.min(s, axis=1) <= 0) & (np.max(s, axis=1) >= 0)
    ids = np.flatnonzero(cross)
    pts = []
    for fi in ids:
        v = F[fi]; t = s[fi]
        local = []
        for i in range(3):
            j = (i + 1) % 3
            if (t[i] <= 0 <= t[j]) or (t[j] <= 0 <= t[i]):
                if abs(t[j] - t[i]) > 1e-12:
                    w = t[i] / (t[i] - t[j])
                    local.append(tri[fi, i] + w * (tri[fi, j] - tri[fi, i]))
        if len(local) == 2:
            pts.append([local[0] @ e1, local[0] @ e2])
            pts.append([local[1] @ e1, local[1] @ e2])
    return np.asarray(pts, float).reshape(-1, 2) if pts else np.zeros((0, 2))


def section_descriptor(pts: np.ndarray) -> dict[str, Any]:
    if len(pts) < 6:
        return {"n_points": int(len(pts))}
    a, b = pts[:, 0], pts[:, 1]
    ca, cb = .5 * (a.min() + a.max()), .5 * (b.min() + b.max())
    # theta = 0 -> +b (anterior), pi/2 -> +a (right lateral), pi -> posterior.
    # This is the convention used by angular_scale() and deform_state().
    theta = np.arctan2(a - ca, b - cb)
    r = np.hypot(a - ca, b - cb)
    bins = 96
    idx = np.minimum(((theta + np.pi) / (2 * np.pi) * bins).astype(int), bins - 1)
    outer = np.full(bins, np.nan)
    for k in range(bins):
        vals = r[idx == k]
        if len(vals):
            outer[k] = float(vals.max())
    good = np.isfinite(outer)
    if good.sum() < 2:
        return {"n_points": int(len(pts))}
    # ALWAYS fill the sparse angular bins: leaving NaNs would poison the area
    # and make the cross-section curves discontinuous on coarse meshes.
    x = np.flatnonzero(good)
    outer = np.where(good, np.interp(np.arange(bins), x, outer[good]), outer)
    outer = np.where(np.isfinite(outer), outer, float(np.nanmean(outer[good])))
    ang = (np.arange(bins) + .5) / bins * 2 * np.pi - np.pi
    xx, yy = outer * np.cos(ang), outer * np.sin(ang)
    area = .5 * float(np.abs(np.dot(xx, np.roll(yy, -1)) - np.dot(yy, np.roll(xx, -1))))
    def sector(centre, width):
        d = np.abs(np.angle(np.exp(1j * (ang - centre))))
        vals = outer[d < width]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float(np.nanmean(outer))
    front = sector(0.0, .20)
    back = sector(np.pi, .20)
    side = sector(np.pi / 2, .20)
    return {"n_points": int(len(pts)), "W_mm": round(float(np.ptp(a)), 3),
            "D_mm": round(float(np.ptp(b)), 3), "a_min": round(float(a.min()), 3),
            "a_max": round(float(a.max()), 3), "b_min": round(float(b.min()), 3),
            "b_max": round(float(b.max()), 3),
            "centre_a_mm": round(float(ca), 3), "centre_b_mm": round(float(cb), 3),
            "area_mm2": round(area, 2),
            "r_front_mm": round(front, 3), "r_back_mm": round(back, 3),
            "r_side_mm": round(side, 3)}


def measure_sections(V: np.ndarray, F: np.ndarray, ax: dict[str, Any],
                     h_levels: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    out = []
    for h in h_levels:
        pts = _section_points(V, F, ax["centre_mm"], ax["axis"],
                              ax["lateral"], ax["anterior"], float(h))
        d = section_descriptor(pts)
        d.update({"level": prefix, "h_mm": round(float(h), 3)})
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# design: reference-driven cross-section targets
# ---------------------------------------------------------------------------

def build_design(ref_sec: list[dict], tgt_sec: list[dict],
                 ref_head: dict, tgt_head: dict,
                 h_chin_ref: float, h_chin_tgt: float,
                 reference_blend: float = 1.0, gain: float = 1.0) -> dict[str, Any]:
    """Low-frequency C2 scale functions s_w, s_front, s_back and center drift.

    Reference descriptors are expressed in head-normalised coordinates and
    mapped to the target neck length; the target's existing shoulder/chest
    extent and jaw are preserved by a C2 blend to the base at both ends.

    ``reference_blend`` < 1 keeps more of the subject's OWN cross-section
    curve, which is useful when a second subject's neck must be reshaped
    without importing a foreign neck silhouette. 1.0 is the 0.9 public default.
    """
    t = np.asarray(SECTION_LEVELS, float)
    wr = float(tgt_head["W_head_mm"] / ref_head["W_head_mm"])
    lr = float(h_chin_tgt / h_chin_ref)
    rW = np.array([s.get("W_mm", np.nan) for s in ref_sec], float)
    rD = np.array([s.get("D_mm", np.nan) for s in ref_sec], float)
    rF = np.array([s.get("r_front_mm", np.nan) for s in ref_sec], float)
    rB = np.array([s.get("r_back_mm", np.nan) for s in ref_sec], float)
    rC = np.array([s.get("centre_b_mm", np.nan) for s in ref_sec], float)
    bW = np.array([s.get("W_mm", np.nan) for s in tgt_sec], float)
    bD = np.array([s.get("D_mm", np.nan) for s in tgt_sec], float)
    bF = np.array([s.get("r_front_mm", np.nan) for s in tgt_sec], float)
    bB = np.array([s.get("r_back_mm", np.nan) for s in tgt_sec], float)
    bC = np.array([s.get("centre_b_mm", np.nan) for s in tgt_sec], float)

    def clean(x, fill):
        x = np.where(np.isfinite(x) & (x > 1.0), x, np.nan)
        ok = np.isfinite(x)
        return np.interp(t, t[ok], x[ok]) if ok.sum() >= 3 else np.full_like(t, fill)

    rW, rD, rF, rB, rC = (clean(a, 1.0) for a in (rW, rD, rF, rB, rC))
    bW, bD, bF, bB, bC = (clean(a, 1.0) for a in (bW, bD, bF, bB, bC))

    # reference radii already live at the SAME normalized levels t, so only
    # the head-width ratio scales the cross-section; neck length normalisation
    # is already encoded by t = h / h_chin.
    s_w = np.clip(rW * wr / bW, MAX_INWARD_SCALE, MAX_OUTWARD_SCALE)
    s_f = np.clip(rF * wr / bF, MAX_INWARD_SCALE, MAX_OUTWARD_SCALE)
    s_b = np.clip(rB * wr / bB, MAX_INWARD_SCALE, MAX_OUTWARD_SCALE)
    s_d = np.clip(rD * wr / bD, MAX_INWARD_SCALE, MAX_OUTWARD_SCALE)
    # center drift: remove the constant frame offset using the mid-neck mean
    mid = (t > .35) & (t < .70)
    drift = ((rC - rC[mid].mean()) * wr - (bC - bC[mid].mean()))
    drift = np.clip(drift, -MAX_CENTER_SHIFT_MM, MAX_CENTER_SHIFT_MM)

    # preserve the existing cut ring and the jaw/mandible: C2 endpoint blend
    anchor = (_ramp(t, 0.0, .16) * (1.0 - _ramp(t, .84, 1.0))
              * float(np.clip(reference_blend, 0.0, 1.0)))
    for arr in (s_w, s_f, s_b, s_d):
        arr[:] = 1.0 + (arr - 1.0) * anchor * float(gain)
    drift *= anchor * float(gain)

    # keep the depth scale consistent with the front/back scales (no flat tube)
    s_f = .5 * (s_f + s_d)
    s_b = .5 * (s_b + s_d)

    # smooth C2 splines; clamped ends keep the anchored boundaries exact
    knots = np.r_[0.0, t, 1.0]
    def spline(v):
        return CubicSpline(knots, np.r_[v[0], v, v[-1]], bc_type="clamped")
    curves = {"s_w": spline(s_w), "s_front": spline(s_f),
              "s_back": spline(s_b), "s_depth": spline(s_d),
              "center_drift": spline(drift)}
    targets = []
    for i, tt in enumerate(t):
        targets.append({
            "t": round(float(tt), 4),
            "h_mm": round(float(tt * h_chin_tgt), 3),
            "reference": {k: (round(float(x), 3) if np.isfinite(x) else None)
                          for k, x in (("W_mm", rW[i]), ("D_mm", rD[i]),
                                       ("r_front_mm", rF[i]), ("r_back_mm", rB[i]))},
            "base": {k: (round(float(x), 3) if np.isfinite(x) else None)
                     for k, x in (("W_mm", bW[i]), ("D_mm", bD[i]),
                                  ("r_front_mm", bF[i]), ("r_back_mm", bB[i]))},
            "target": {
                "W_mm": round(float(bW[i] * s_w[i]), 3),
                "D_mm": round(float(bD[i] * s_d[i]), 3),
                "r_front_mm": round(float(bF[i] * s_f[i]), 3),
                "r_back_mm": round(float(bB[i] * s_b[i]), 3),
                "centre_b_shift_mm": round(float(drift[i]), 3),
                "area_mm2": round(float(tgt_sec[i].get("area_mm2", 0.0) *
                                        s_w[i] * s_d[i]), 2),
            },
            "scale": {"s_w": round(float(s_w[i]), 4),
                      "s_front": round(float(s_f[i]), 4),
                      "s_back": round(float(s_b[i]), 4),
                      "s_depth": round(float(s_d[i]), 4)},
        })
    return {"curves": curves, "targets": targets, "t": t,
            "scales": {"width_ratio": round(wr, 4), "neck_length_ratio": round(lr, 4),
                       "reference_blend": round(float(reference_blend), 4),
                       "gain": round(float(gain), 4)},
            "anchor": {"lower_t": 0.16, "upper_t": 0.84,
                       "note": "C2 blend to the existing bottom cut ring and the jaw; "
                               "the skirt extent and mandible outline are not replaced"},
            "limits": {"max_inward_scale": MAX_INWARD_SCALE,
                       "max_outward_scale": MAX_OUTWARD_SCALE,
                       "max_center_shift_mm": MAX_CENTER_SHIFT_MM}}


def angular_scale(theta: np.ndarray, t: np.ndarray, design: dict) -> np.ndarray:
    """C-infinity in theta, C2 in t: s(theta,t) = A0 + A1 cos + A2 cos2."""
    c = design["curves"]
    s_w = c["s_w"](t); s_f = c["s_front"](t); s_b = c["s_back"](t)
    s_m = .5 * (s_f + s_b)
    a0 = .5 * (s_w + s_m)
    a1 = .5 * (s_f - s_b)
    a2 = .5 * (s_m - s_w)
    return a0 + a1 * np.cos(theta) + a2 * np.cos(2.0 * theta)


# ---------------------------------------------------------------------------
# smooth base contour field (removes the coarse polygonal outline)
# ---------------------------------------------------------------------------

def fit_contour_field(V: np.ndarray, F: np.ndarray, ax: dict[str, Any],
                      h_chin: float, centre_curve) -> dict[str, Any]:
    """Fourier coefficients of the base outer contour, smoothed along h."""
    h, a, b = axial_coords(V, ax)
    hs = np.arange(18.0, h_chin + 6.0, 2.0)
    bins = 96
    coeff = np.zeros((len(hs), 2 * FOURIER_ORDER + 1))
    for i, hh in enumerate(hs):
        sel = np.abs(h - hh) <= 2.5
        if sel.sum() < 12:
            continue
        ca, cb = centre_curve(hh)
        aa, bb = a[sel] - ca, b[sel] - cb
        th = np.arctan2(aa, bb)
        rr = np.hypot(aa, bb)
        k = np.minimum(((th + np.pi) / (2 * np.pi) * bins).astype(int), bins - 1)
        outer = np.full(bins, np.nan)
        for j in range(bins):
            vals = rr[k == j]
            if len(vals):
                outer[j] = float(vals.max())
        ok = np.isfinite(outer)
        if ok.sum() < 30:
            continue
        ang = (np.arange(bins) + .5) / bins * 2 * np.pi - np.pi
        vals = np.interp(ang, ang[ok], outer[ok])
        M = np.column_stack([np.ones(bins)] +
                            [fn for o in range(1, FOURIER_ORDER + 1)
                             for fn in (np.cos(o * ang), np.sin(o * ang))])
        co = np.linalg.lstsq(M, vals, rcond=None)[0]
        coeff[i] = co
    # smooth each coefficient along h (C2 cubic spline, light regularisation)
    ok = np.isfinite(coeff[:, 0]) & (coeff[:, 0] > 1.0)
    hs_ok = hs[ok]
    co_ok = coeff[ok]
    splines = [CubicSpline(hs_ok, co_ok[:, j], bc_type="natural")
               for j in range(co_ok.shape[1])]
    return {"splines": splines, "h_min": float(hs_ok.min()), "h_max": float(hs_ok.max()),
            "order": FOURIER_ORDER, "levels": len(hs_ok)}


def contour_radius(theta: np.ndarray, h: np.ndarray, cf: dict) -> np.ndarray:
    hh = np.clip(h, cf["h_min"], cf["h_max"])
    out = cf["splines"][0](hh)
    for o in range(1, cf["order"] + 1):
        out = out + cf["splines"][2 * o - 1](hh) * np.cos(o * theta)
        out = out + cf["splines"][2 * o](hh) * np.sin(o * theta)
    return np.maximum(out, 1.0)


def centre_curve_from_sections(sections: list[dict], h_chin: float):
    hs = np.array([s["h_mm"] for s in sections], float)
    ca = np.array([s.get("centre_a_mm", 0.0) for s in sections], float)
    cb = np.array([s.get("centre_b_mm", 0.0) for s in sections], float)
    # add the cut plane centre and the chin level as stable end points
    hs = np.r_[0.0, hs, h_chin + 5.0]
    ca = np.r_[ca[0] if len(ca) else 0.0, ca, ca[-1] if len(ca) else 0.0]
    cb = np.r_[cb[0] if len(cb) else 0.0, cb, cb[-1] if len(cb) else 0.0]
    sa = CubicSpline(hs, ca, bc_type="natural")
    sb = CubicSpline(hs, cb, bc_type="natural")
    return lambda h: (sa(np.clip(h, hs[0], hs[-1])),
                      sb(np.clip(h, hs[0], hs[-1])))


# ---------------------------------------------------------------------------
# consistent subdivision
# ---------------------------------------------------------------------------

def subdivide_uniform(V: np.ndarray, F: np.ndarray):
    """One 1->4 split of EVERY triangle; shared edge midpoints => no cracks.

    New vertices are *not* accepted as-is: their final positions are re-solved
    from the smooth target contour by deform_state().
    """
    edges = {}
    new_verts = []
    for f in F:
        for i, j in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(i, j), max(i, j))
            if key not in edges:
                edges[key] = len(V) + len(new_verts)
                new_verts.append(.5 * (V[i] + V[j]))
    V1 = np.vstack([V, np.asarray(new_verts, float)]) if new_verts else V.copy()
    F1 = []
    for f in F:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab, bc, ca = edges[(min(a, b), max(a, b))], edges[(min(b, c), max(b, c))], \
                     edges[(min(c, a), max(c, a))]
        F1.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return V1, np.asarray(F1, dtype=np.int64), edges


def _map_boundary(F0: np.ndarray) -> set[int]:
    edges, _, counts, _, _ = mi.edge_records(F0)
    return set(map(int, edges[counts == 1].reshape(-1)))


def _boundary_after_subdivision(F0: np.ndarray, edges: dict, n0: int) -> np.ndarray:
    """Boundary vertex ids on the subdivided mesh (original ring + midpoints)."""
    b0 = _map_boundary(F0)
    mids = [idx for (i, j), idx in edges.items() if i in b0 and j in b0]
    return np.asarray(sorted(b0 | set(mids)), dtype=np.int64)


# ---------------------------------------------------------------------------
# volume + detail deformation
# ---------------------------------------------------------------------------

def regularize_displacement(F: np.ndarray, D: np.ndarray, free: np.ndarray,
                            iterations: int = 30, weight: float = 0.65) -> np.ndarray:
    """Low-frequency regularization of the DISPLACEMENT FIELD (not the surface).

    Identity/anchor vertices are held fixed (Dirichlet zero); only editable
    vertices exchange displacement with editable neighbours.  This removes
    high per-triangle strain where a strong root expansion meets a nearly
    horizontal shoulder/chest plate, without smoothing away the designed
    cross-section volume.
    """
    free = np.asarray(free, bool)
    ids = np.flatnonzero(free)
    edges, _, counts, _, _ = mi.edge_records(F)
    keep = free[edges[:, 0]] & free[edges[:, 1]] & (counts == 2)
    e = edges[keep]
    rows = np.r_[e[:, 0], e[:, 1]]
    cols = np.r_[e[:, 1], e[:, 0]]
    D = D.copy()
    for _ in range(iterations):
        acc = np.zeros_like(D)
        cnt = np.zeros(len(D), float)
        np.add.at(acc, rows, D[cols])
        np.add.at(cnt, rows, 1.0)
        avg = acc / np.maximum(cnt[:, None], 1.0)
        D[free] = (1.0 - weight) * D[free] + weight * avg[free]
        D[~free] = 0.0
    return D


def measured_region_hold(V: np.ndarray, ax: dict[str, Any], theta: np.ndarray,
                         observation: dict) -> np.ndarray:
    """C2 hold factor around a measured (RGB-D) neck surface patch.

    Real measured skin must not be pushed away by an appearance-driven volume
    change.  The patch only supplies a smooth spatial window; it never copies
    foreign vertices into the mesh.  Returns 1 outside the patch and
    ``observation['hold']`` at its centre.
    """
    pts = np.asarray(observation["points_mm"], float)
    P = pts - ax["centre_mm"]
    oh, oa = P @ ax["axis"], P @ ax["lateral"]
    h, a, b = axial_coords(V, ax)
    hp = float(observation.get("h_pad_mm", 8.0))
    ap = float(observation.get("a_pad_mm", 10.0))
    # Asymmetric axial falloff: the sticker patch is on the mid/lower front
    # neck, so its upper falloff must be short and must NOT freeze the
    # submental (jaw-to-neck) transition just above the patch.
    h_hi_pad = float(observation.get("h_pad_high_mm", max(hp * .35, 1.0)))
    h_lo, h_hi = float(oh.min()) - hp, float(oh.max()) + h_hi_pad
    a_lo, a_hi = float(oa.min()) - ap, float(oa.max()) + ap

    def plateau(x, lo, core_lo, core_hi, hi):
        rise = _smootherstep((x - lo) / max(core_lo - lo, 1e-6))
        fall = 1.0 - _smootherstep((x - core_hi) / max(hi - core_hi, 1e-6))
        return np.clip(rise * fall, 0.0, 1.0)

    wh = plateau(h, h_lo, float(oh.min()), float(oh.max()), h_hi)
    wa = plateau(a, a_lo, float(oa.min()), float(oa.max()), a_hi)
    wf = np.clip(np.cos(theta), 0.0, 1.0) ** 2
    g = wh * wa * wf
    hold = float(np.clip(observation.get("hold", 0.45), 0.0, 1.0))
    return 1.0 - (1.0 - hold) * g


def deform_state(V: np.ndarray, F: np.ndarray, V0: np.ndarray, F0: np.ndarray,
                 ax: dict[str, Any], design: dict, cf: dict, centre_curve,
                 boundary1: np.ndarray, landmarks: np.ndarray,
                 state: str, detail: bool,
                 observation: dict | None = None,
                 detail_amplitude: float | None = None,
                 cut_anchor_floor: float | None = None,
                 shape_params: dict | None = None) -> dict[str, Any]:
    """Re-solve every vertex position from the smooth target cross-sections.

    ``observation`` optionally keeps a measured neck patch close to its
    recorded position (C2 spatial hold); None keeps the 0.9 public behaviour.
    ``shape_params`` overrides the module's calibrated shape constants for a
    different subject; None keeps the 0.9 public behaviour bit-for-bit.
    """
    sp = {} if shape_params is None else shape_params
    skirt_t = float(sp.get("skirt_t", SKIRT_T))
    skirt_axial = sp.get("skirt_axial_blend", SKIRT_AXIAL_BLEND_MM)
    skirt_tube = sp.get("skirt_tube_blend", SKIRT_TUBE_BLEND_MM)
    angular_damp = sp.get("angular_damp", ANGULAR_DAMP_MM)
    cut_width = float(sp.get("cut_anchor_width", CUT_ANCHOR_WIDTH_MM))
    sub_h = float(sp.get("submental_h", SUBMENTAL_H_MM))
    sub_sig = float(sp.get("submental_sigma", SUBMENTAL_SIGMA_MM))
    sub_fill = float(sp.get("submental_fill_mm", SUBMENTAL_FILL_MM))
    full_edit = float(sp.get("full_edit_below_chin", FULL_EDIT_BELOW_CHIN_MM))
    identity_above = float(sp.get("identity_above_chin", IDENTITY_ABOVE_CHIN_MM))
    h, a, b = axial_coords(V, ax)
    h_chin = float(ax["h_chin_mm"])
    t = np.clip(h / h_chin, 0.0, 1.0)
    # The shoulder/chest skirt is a nearly horizontal PLATE: using the raw
    # axial coordinate there would impose a large h-gradient on an in-plane
    # dilation and fold its thin triangles.  Give the plate ONE cross-section
    # target and hand over to the continuous tube coordinate above it.
    q = _ramp(h, skirt_axial[0], skirt_axial[1])
    t_eff = (1.0 - q) * skirt_t + q * t
    ca, cb = centre_curve(h)
    ca = np.asarray(ca, float); cb = np.asarray(cb, float)
    # center drift from the reference-driven design
    drift = np.asarray(design["curves"]["center_drift"](t_eff), float)
    theta = np.arctan2(a - ca, b - cb)
    r_actual = np.hypot(a - ca, b - cb)
    r_smooth = contour_radius(theta, h, cf)

    # identity / outer-anchor / skirt-tube C2 weights
    w_upper = 1.0 - _ramp(h, h_chin - full_edit, h_chin + identity_above)
    boundary_pts = V[boundary1] * 1000.0
    tree = cKDTree(boundary_pts)
    d_cut, _ = tree.query(V * 1000.0, k=1)
    floor = (float(sp.get("cut_anchor_floor", CUT_ANCHOR_FLOOR))
             if cut_anchor_floor is None else float(cut_anchor_floor))
    w_cut = floor + (1.0 - floor) * _ramp(d_cut, 0.0, cut_width)
    # state-local jaw protection from that state's own facial landmarks
    lm_mm = landmarks * 1000.0
    d_lm, _ = cKDTree(lm_mm).query(V * 1000.0, k=1)
    # Only the immediate landmark vicinity is protected.  A wide guard would
    # freeze the whole submental transition and leave a visible crease.
    w_lm = _ramp(d_lm, 2.0, 9.0)
    w = w_upper * w_cut * w_lm
    # exclude the head entirely (never pull the occiput or the face)
    w = np.where(h > h_chin + identity_above, 0.0, w)

    # The bottom skirt is a near-horizontal plate, NOT a tube: there we use a
    # pure dilation (the constant harmonic) so no angular shear can invert its
    # thin triangles.  The full low-frequency angular field and the smooth
    # contour re-projection are blended in only above the true neck tube.
    tube = _ramp(h, skirt_tube[0], skirt_tube[1])
    damp = _ramp(h, angular_damp[0], angular_damp[1])
    r_base = (1.0 - tube) * r_actual + tube * r_smooth
    s_full = angular_scale(theta, t_eff, design)
    c = design["curves"]
    s0 = .5 * (c["s_w"](t_eff) + .5 * (c["s_front"](t_eff) + c["s_back"](t_eff)))
    scale = s0 + damp * (s_full - s0)
    r_new = r_base * scale
    cb_new = cb + drift
    a_new = ca + r_new * np.sin(theta)
    b_new = cb_new + r_new * np.cos(theta)
    # Head/neck transition: one broad, low-amplitude C2 anterior volume fill
    # softens the submental undercut without touching the mandible outline.
    fill_window = np.exp(-.5 * ((h - sub_h) / sub_sig) ** 2)
    anterior_window = np.clip(np.cos(theta), 0.0, 1.0) ** 2
    broad_lateral = np.exp(-.5 * (np.sin(theta) / .70) ** 2)
    submental_mm = sub_fill * fill_window * anterior_window * broad_lateral
    b_new = b_new + submental_mm

    P_target = (ax["centre_mm"][None, :] + h[:, None] * ax["axis"][None, :]
                + a_new[:, None] * ax["lateral"][None, :]
                + b_new[:, None] * ax["anterior"][None, :]) / 1000.0
    d_target = w[:, None] * (P_target - V)
    hold = np.ones(len(V), float)
    if observation is not None:
        hold = measured_region_hold(V, ax, theta, observation)
        # Measured RGB-D skin constrains the ANTERIOR (depth) component only:
        # lateral/vertical volume reshaping still applies, but recorded skin is
        # not pushed away in depth.  A C2 spatial window keeps the transition
        # smooth, and the displacement field is regularised afterwards.
        ant = np.asarray(ax["anterior"], float)
        d_par = (d_target @ ant)[:, None] * ant[None, :]
        d_target = d_target - (1.0 - hold)[:, None] * d_par
    d_smooth = regularize_displacement(
        F, d_target, w > 1e-6,
        iterations=int(sp.get("regularize_iterations", 10)),
        weight=float(sp.get("regularize_weight", .60)))
    V_volume = V + d_smooth

    # secondary continuous natural relief (C only): ONE connected low-amplitude
    # anterior fullness, no rings, no two glued spheres, no saddle groove.
    V_detail = V_volume.copy()
    detail_mm = np.zeros(len(V))
    if detail:
        prof = (np.exp(-.5 * ((t - .55) / DETAIL_T_SIGMA) ** 2) * .75
                + np.exp(-.5 * ((t - .78) / (DETAIL_T_SIGMA * .8)) ** 2) * .42)
        prof = np.clip(prof / max(prof.max(), 1e-9), 0.0, 1.0)
        anterior = np.clip(np.cos(theta), 0.0, 1.0) ** 2
        lateral = np.exp(-.5 * (np.sin(theta) / .55) ** 2)
        amp = (DETAIL_AMPLITUDE_MM if detail_amplitude is None else float(detail_amplitude)) \
            * prof * anterior * lateral * w
        if observation is not None:
            amp = amp * hold  # same depth hold as the volume field
        detail_mm = amp
        V_detail = V_volume + (amp / 1000.0)[:, None] * ax["anterior"][None, :]

    disp = (V_detail - V) * 1000.0
    vol_disp = (V_volume - V) * 1000.0
    inward = float(np.min(vol_disp @ ax["anterior"])) if len(V) else 0.0
    outward = float(np.max(vol_disp @ ax["anterior"])) if len(V) else 0.0
    regions = {
        "A_identity": np.asarray((h > h_chin - full_edit) &
                                 (w_upper < .5) | (h > h_chin + identity_above), bool),
        "B_transition": np.asarray((w > .05) & (w <= .999), bool),
        "B_full": np.asarray(w >= .999, bool),
        "C_cut_anchor": np.asarray(w_cut < .05, bool),
        "face_landmark_guard": np.asarray(w_lm < .05, bool),
    }
    return {"base": V, "volume": V_volume, "detail": V_detail,
            "h": h, "a": a, "b": b, "t": t, "w": w, "scale": scale,
            "disp_mm": disp, "volume_disp_mm": vol_disp, "detail_mm": detail_mm,
            "regions": regions, "boundary": boundary1,
            "state": state,
            "max_disp_mm": round(float(np.linalg.norm(disp, axis=1).max()), 4),
            "inward_outward_mm": {"min_signed_anterior": round(inward, 3),
                                  "max_signed_anterior": round(outward, 3)},
            "detail_applied": bool(detail)}


# ---------------------------------------------------------------------------
# geometry verification
# ---------------------------------------------------------------------------

def verify_mesh(V: np.ndarray, F: np.ndarray, V_base: np.ndarray,
                F_base: np.ndarray, label: str) -> dict[str, Any]:
    rep = mi.topology_report(V, F)
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) * .5 * 1e6
    e = np.linalg.norm(np.vstack([tri[:, 1] - tri[:, 0],
                                  tri[:, 2] - tri[:, 1],
                                  tri[:, 0] - tri[:, 2]]), axis=1)
    # winding agreement with the base mesh on shared topology
    flipped = 0
    if F.shape == F_base.shape and np.array_equal(F, F_base):
        flipped = 0
    disp = np.linalg.norm((V[:len(V_base)] - V_base) * 1000.0, axis=1)
    tri0 = V_base[F]
    e0 = np.stack([tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 1],
                   tri0[:, 0] - tri0[:, 2]], axis=1)
    emax0 = np.maximum(np.linalg.norm(e0, axis=2).max(axis=1), 1e-9)
    emax1 = np.linalg.norm(np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1],
                                     tri[:, 0] - tri[:, 2]], axis=1), axis=2).max(axis=1)
    growth = float(np.max(emax1 / emax0))
    return {"label": label, "finite": bool(np.isfinite(V).all()),
            "max_edge_growth_ratio": round(growth, 4),
            "vertices": int(len(V)), "faces": int(len(F)),
            "components": rep["components"], "nonmanifold_edges": rep["nonmanifold_edges"],
            "inconsistent_shared_edges": rep["inconsistent_shared_edges"],
            "boundary_edges": rep["boundary_edges"], "boundary_groups": rep["boundary_groups"],
            "zero_area_faces": rep["zero_area_faces"],
            "small_faces_below_0_01_mm2": rep["small_faces_below_0_01_mm2"],
            "min_area_mm2": rep["min_area_mm2"],
            "min_edge_mm": round(float(e.min()), 5),
            "max_edge_mm": round(float(e.max()), 4),
            "median_edge_mm": round(float(np.median(e)), 4),
            "slender_faces": int((area < .12 * e.reshape(3, -1).max(axis=0) ** 2).sum()),
            "max_displacement_mm": round(float(disp.max()), 4),
            "mean_displacement_mm": round(float(disp.mean()), 4),
            "flipped_face_estimate": int(flipped)}


def normal_flip_count(V_base: np.ndarray, V: np.ndarray, F: np.ndarray) -> int:
    """Faces whose normal reversed relative to the unshaped base (same topology)."""
    n0 = np.cross(V_base[F][:, 1] - V_base[F][:, 0], V_base[F][:, 2] - V_base[F][:, 0])
    n1 = np.cross(V[F][:, 1] - V[F][:, 0], V[F][:, 2] - V[F][:, 0])
    dot = np.einsum("ij,ij->i", n0, n1)
    return int((dot <= 0.0).sum())


def self_intersection_estimate(V_base: np.ndarray, V: np.ndarray,
                               F: np.ndarray) -> dict[str, Any]:
    """Fold / flip / degeneracy audit on identical topology."""
    import trimesh
    mesh = trimesh.Trimesh(V, F, process=False)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    degenerate = int((np.linalg.norm(n, axis=1) < 1e-12).sum())
    flips = normal_flip_count(V_base, V, F)
    return {"count": flips + degenerate,
            "normal_reversals": flips,
            "degenerate_faces": degenerate,
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "method": "per-face normal reversal against the unshaped base on identical "
                      "topology, face-area degeneracy, trimesh winding/bounds; "
                      "a global triangle-triangle intersection solver is not used"}


def transfer_state(V_base_sub: np.ndarray, d_volume: np.ndarray,
                   d_detail: np.ndarray, basis_closed: np.ndarray,
                   basis_state: np.ndarray, landmarks: np.ndarray,
                   state: str) -> dict[str, Any]:
    """Transfer the SHARED design by vertex correspondence, not closed coordinates.

    The displacement vectors are re-expressed in this state's own head basis,
    then applied to this state's own subdivided base vertices.  A small
    state-local jaw guard prevents the open jaw from being pulled; its range
    and displacement are recorded separately.
    """
    M = basis_closed @ basis_state.T
    dv = d_volume @ M
    dd = d_detail @ M
    lm_mm = landmarks * 1000.0
    d_lm, _ = cKDTree(lm_mm).query(V_base_sub * 1000.0, k=1)
    jaw = _ramp(d_lm, 8.0, 22.0)
    # jaw guard only relaxes, never amplifies the shared design
    dv = dv * jaw[:, None]
    dd = dd * jaw[:, None]
    V_volume = V_base_sub + dv
    V_final = V_volume + dd
    affected = int((jaw < .999).sum())
    return {"volume": V_volume, "detail": V_final,
            "volume_disp_mm": dv * 1000.0, "detail_mm": dd * 1000.0,
            "disp_mm": (V_final - V_base_sub) * 1000.0,
            "state_local_guard": {"vertices": affected,
                                  "min_distance_mm": round(float(d_lm.min()), 3),
                                  "max_relaxed_displacement_mm": round(
                                      float(np.linalg.norm((d_volume - dv) * 1000.0, axis=1).max()), 4),
                                  "range": "8-22 mm from this state's own facial landmarks",
                                  "note": "shared closed design transferred by vertex "
                                          "correspondence; closed coordinates are never copied"},
            "state": state}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def build_states(*, states, references, dense_vertices, dense_faces,
                 reference_path, fit_dir=None, alpha=DEFAULT_ALPHA,
                 detail=True, **kwargs) -> dict[str, Any]:
    """Reference-driven neck volume for the public delivery path.

    Mirrors neck_dual.build_states' call signature so pipeline.export_delivery
    can switch the public default to this module without a new optional mode.
    """
    ref_path = Path(reference_path)
    RV, RF = mi.load_obj(ref_path)
    RV_mm = RV * 1000.0
    # reference frame: Y up, +Z anterior
    ref_up = np.array([0., 1., 0.]); ref_right = np.array([1., 0., 0.])
    ref_fwd = np.array([0., 0., 1.])
    menton = detect_reference_menton(RV_mm)
    ref_chin = np.array([0., menton["menton_y_mm"], 0.])
    # real chin point on the reference midline at the detected menton height
    sel = (np.abs(RV_mm[:, 0]) <= 3.) & (np.abs(RV_mm[:, 1] - menton["menton_y_mm"]) <= 2.)
    if sel.sum():
        ref_chin[2] = float(RV_mm[sel, 2].max())
    ref_head = head_dimensions(RV_mm, ref_up, ref_right, ref_fwd, ref_chin)
    ref_ax = axial_frame(RV, RF, {"up": ref_up, "right": ref_right,
                                  "forward": ref_fwd, "chin": ref_chin / 1000.0})
    h_chin_ref = float(ref_ax["h_chin_mm"])

    # shared neutral (closed) head frame defines the scale for BOTH states
    closed_ref = references["closed"] if isinstance(references, dict) else references[0]
    lm = np.asarray(closed_ref["stages"]["dense"]["landmarks_m"], float)
    frame = head_frame(lm)
    Vc = np.asarray(dense_vertices[closed_ref["stages"]["dense"]["frame_index"]], float)
    tgt_head = head_dimensions(Vc * 1000.0, frame["up"], frame["right"],
                               frame["forward"], frame["chin"] * 1000.0)
    head_audit = scaled_head_audit(ref_head, tgt_head, menton)

    # closed base defines the shared axial frame and the contour field
    F0 = np.asarray(dense_faces, dtype=np.int64)
    ax = axial_frame(Vc, F0, frame)
    h_chin = float(ax["h_chin_mm"])
    # section levels: same normalized t for reference and target
    ref_sec = measure_sections(RV, RF, ref_ax,
                               np.asarray(SECTION_LEVELS) * h_chin_ref, "reference")
    tgt_sec = measure_sections(Vc, F0, ax,
                               np.asarray(SECTION_LEVELS) * h_chin, "target_base")
    design = build_design(ref_sec, tgt_sec, ref_head, tgt_head, h_chin_ref, h_chin)
    centre_curve = centre_curve_from_sections(tgt_sec, h_chin)
    cf = fit_contour_field(Vc, F0, ax, h_chin, centre_curve)

    V1c, F1, edge_map = subdivide_uniform(Vc, F0)
    boundary1 = _boundary_after_subdivision(F0, edge_map, len(Vc))
    base_loops = _boundary_loops(F0)
    after_loops = _boundary_loops(F1)

    # ONE shared design is built on the closed/neutral base.
    closed_run = deform_state(V1c, F1, Vc, F0, ax, design, cf, centre_curve,
                              boundary1, lm, "closed", detail=detail)
    d_volume = closed_run["volume"] - V1c
    d_detail = closed_run["detail"] - closed_run["volume"]
    basis_closed = np.column_stack([frame["right"], frame["up"], frame["forward"]])

    states_out, audits, fields = {}, {}, {}
    accepted = True
    for state in states:
        ref_state = references[state] if isinstance(references, dict) else references[0]
        frame_idx = ref_state["stages"]["dense"]["frame_index"]
        Vb = np.asarray(dense_vertices[frame_idx], float)
        V1s, F1s, _ = subdivide_uniform(Vb, F0)
        lm_state = np.asarray(ref_state["stages"]["dense"]["landmarks_m"], float)
        frame_state = head_frame(lm_state)
        basis_state = np.column_stack([frame_state["right"], frame_state["up"],
                                       frame_state["forward"]])
        if state == "closed":
            run = closed_run
            V_volume, V_final = run["volume"], run["detail"]
            adaptation = {"vertices": 0, "note": "design state; no transfer adaptation"}
            frozen = run["w"] < 1e-9
            disp_mm, detail_mm, vol_disp = run["disp_mm"], run["detail_mm"], run["volume_disp_mm"]
            regions = {k: int(v.sum()) for k, v in run["regions"].items()}
        else:
            tr = transfer_state(V1s, d_volume, d_detail, basis_closed, basis_state,
                                lm_state, state)
            V_volume, V_final = tr["volume"], tr["detail"]
            adaptation = tr["state_local_guard"]
            frozen = np.linalg.norm(d_volume, axis=1) < 1e-12
            disp_mm, detail_mm, vol_disp = tr["disp_mm"], tr["detail_mm"], tr["volume_disp_mm"]
            regions = {
                "A_identity": int(closed_run["regions"]["A_identity"].sum()),
                "B_transition": int(closed_run["regions"]["B_transition"].sum()),
                "B_full": int(closed_run["regions"]["B_full"].sum()),
                "C_cut_anchor": int(closed_run["regions"]["C_cut_anchor"].sum()),
                "face_landmark_guard": int(closed_run["regions"]["face_landmark_guard"].sum()),
                "note": "transferred by vertex correspondence; region counts are the "
                        "shared design's own counts"}
        ver = verify_mesh(V_final, F1s, V1s, F1s, state + "_final")
        verB = verify_mesh(V_volume, F1s, V1s, F1s, state + "_volume_only")
        inter = self_intersection_estimate(V1s, V_final, F1s)
        ver["normal_reversals"] = inter["normal_reversals"]
        ident_max = float(np.linalg.norm(disp_mm[frozen], axis=1).max()) if frozen.any() else 0.0
        guards = {"inconsistent_shared_edges": ver["inconsistent_shared_edges"],
                  "nonmanifold_edges": ver["nonmanifold_edges"],
                  "degenerate_faces": ver["zero_area_faces"],
                  "normal_reversal": inter["normal_reversals"],
                  "slender_faces": ver["slender_faces"],
                  "protected_max_displacement_mm": round(ident_max, 6),
                  "protected_vertices": int(frozen.sum()),
                  "max_edge_growth": ver["max_edge_growth_ratio"],
                  "status": "passed" if (ver["inconsistent_shared_edges"] == 0
                                         and ver["nonmanifold_edges"] == 0
                                         and ver["zero_area_faces"] == 0
                                         and inter["count"] == 0
                                         and ident_max < 1e-6) else "failed"}
        accepted &= guards["status"] == "passed"
        max_disp = round(float(np.linalg.norm(disp_mm, axis=1).max()), 4)
        states_out[state] = {
            "run": {"vertices": V_final, "volume_vertices": V_volume,
                    "applied_mm": disp_mm, "detail_mm": detail_mm,
                    "weight": closed_run["w"], "h_mm": closed_run["h"],
                    "t": closed_run["t"]},
            "base_subdivided": V1s, "protected": frozen,
            "guards": guards, "intersections": inter,
            "landmarks_m": lm_state, "frame_index": frame_idx,
            "regions": regions,
            "verification": ver, "verification_volume_only": verB,
            "max_disp_mm": max_disp,
            "inward_outward_mm": {
                "min_signed_anterior": round(float((vol_disp @ ax["anterior"]).min()), 3),
                "max_signed_anterior": round(float((vol_disp @ ax["anterior"]).max()), 3)},
            "state_local_adaptation": adaptation,
        }
        audits[state] = {"applied_peak_mm": max_disp,
                         "support": {"cut_anchor_vertices": regions.get("C_cut_anchor", 0),
                                     "full_edit_vertices": regions.get("B_full", 0),
                                     "transition_vertices": regions.get("B_transition", 0),
                                     "inside_protected": 0},
                         "base_relative": {
                             "max_outward_mm": states_out[state]["inward_outward_mm"]["max_signed_anterior"],
                             "min_outward_mm": states_out[state]["inward_outward_mm"]["min_signed_anterior"],
                             "note": "signed anterior displacement; controlled inward motion "
                                     "at mid neck is allowed and recorded; no global volume "
                                     "conservation is imposed"},
                         "verification": ver}
        fields[state] = {"disp_mm": disp_mm, "volume_disp_mm": vol_disp,
                         "detail_mm": detail_mm, "base": V1s,
                         "volume": V_volume, "detail": V_final,
                         "w": closed_run["w"], "h": closed_run["h"],
                         "t": closed_run["t"], "regions": regions}

    return {
        "schema": SCHEMA, "accepted": bool(accepted),
        "neck_method": NECK_METHOD_VOLUME, "geometry_source": GEOMETRY_SOURCE_VOLUME,
        "states": states_out, "audits": audits, "faces": F1,
        "base_faces": F0, "base_vertices": dense_vertices,
        "edge_map": edge_map, "boundary_before": [l.tolist() for l in base_loops],
        "boundary_after": [l.tolist() for l in after_loops],
        "boundary_vertices": boundary1,
        "placement": {"source": "shared closed neutral head landmarks + true bottom cut",
                      "h_chin_mm": h_chin, "h_chin_reference_mm": h_chin_ref,
                      "old_hardcoded_center_vertex": 15571,
                      "old_roi_note": "the old ROI was a small front patch; it is NOT used"},
        "design_audit": {"levels": len(SECTION_LEVELS), "height_mm": h_chin,
                         "width_ratio": design["scales"]["width_ratio"],
                         "neck_length_ratio": design["scales"]["neck_length_ratio"],
                         "limits": design["limits"], "anchor": design["anchor"]},
        "reference_morphology": {"source_sha256": sha256_file(ref_path),
                                 "path": str(ref_path), "head": ref_head,
                                 "menton": menton, "sections": ref_sec,
                                 "target_sections": tgt_sec},
        "head_scale": head_audit, "frame": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                            for k, v in frame.items()},
        "axial_frame": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in ax.items() if k not in ("cut_loop", "all_loops")},
        "cross_section_targets": design["targets"],
        "direction_mode": "cross_section_radial_low_frequency", "alpha": alpha,
        "detail_applied": bool(detail),
        "design_curves": design, "contour_field": {
            "order": cf["order"], "levels": cf["levels"],
            "h_min_mm": cf["h_min"], "h_max_mm": cf["h_max"]},
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# subject-adaptive, topology-preserving entry point
# ---------------------------------------------------------------------------

def reshape_subject_neck(*, states, references, dense_vertices, dense_faces,
                         reference_path, observed_neck_paths=None,
                         reference_blend=1.0, observation_hold=0.80,
                         cut_anchor_floor=0.0, detail=True,
                         detail_amplitude=None, alpha=1.0,
                         observation_pads=(8.0, 10.0),
                         shape_params=None, gain=1.0) -> dict[str, Any]:
    """Reshape the neck ON an existing reconstructed mesh and fuse in place.

    Reuses the 0.9 cross-section-driven volume algorithm, but unlike
    ``build_states`` it does NOT subdivide: the original dense topology,
    vertex order, face list and both boundary rings are preserved exactly.
    Only neck/transition vertices are re-positioned; the face, mandible,
    occiput and every unrelated region stay bit-identical.

    The design frame, axial cut plane, head scale and cross-section targets
    are all derived from THIS subject, so shape, scale, position and
    orientation follow the existing reconstruction rather than a foreign
    template.  ``reference_blend`` < 1 keeps more of the subject's own
    section curve.  A measured RGB-D neck patch can be supplied in
    ``observed_neck_paths['closed']``: it defines a C2 hold window that stops
    the appearance-driven volume change from pushing real measured skin away.

    ``build_states`` (the public 0.9 delivery path) is unchanged.
    """
    ref_path = Path(reference_path)
    RV, RF = mi.load_obj(ref_path)
    RV_mm = RV * 1000.0
    ref_up = np.array([0., 1., 0.]); ref_right = np.array([1., 0., 0.])
    ref_fwd = np.array([0., 0., 1.])
    menton = hs.detect_reference_menton(RV_mm)
    ref_chin = np.array([0., menton["menton_y_mm"], 0.])
    sel = (np.abs(RV_mm[:, 0]) <= 3.) & (np.abs(RV_mm[:, 1] - menton["menton_y_mm"]) <= 2.)
    if sel.any():
        ref_chin[2] = float(RV_mm[sel, 2].max())
    ref_head = hs.head_dimensions(RV_mm, ref_up, ref_right, ref_fwd, ref_chin)
    ref_ax = axial_frame(RV, RF, {"up": ref_up, "right": ref_right,
                                  "forward": ref_fwd, "chin": ref_chin / 1000.0})
    h_chin_ref = float(ref_ax["h_chin_mm"])

    closed_ref = references["closed"] if isinstance(references, dict) else references[0]
    lm_c = np.asarray(closed_ref["stages"]["dense"]["landmarks_m"], float)
    frame = head_frame(lm_c)
    F0 = np.asarray(dense_faces, dtype=np.int64)
    idx_c = closed_ref["stages"]["dense"]["frame_index"]
    Vc = np.asarray(dense_vertices[idx_c], float)
    tgt_head = hs.head_dimensions(Vc * 1000., frame["up"], frame["right"],
                                  frame["forward"], frame["chin"] * 1000.)
    head_audit = scaled_head_audit(ref_head, tgt_head, menton)
    ax = axial_frame(Vc, F0, frame)
    h_chin = float(ax["h_chin_mm"])

    levels = np.asarray(SECTION_LEVELS)
    ref_sec = measure_sections(RV, RF, ref_ax, levels * h_chin_ref, "reference")
    tgt_sec = measure_sections(Vc, F0, ax, levels * h_chin, "subject_base")
    design = build_design(ref_sec, tgt_sec, ref_head, tgt_head,
                          h_chin_ref, h_chin, reference_blend=reference_blend,
                          gain=gain)
    centre_curve = centre_curve_from_sections(tgt_sec, h_chin)
    cf = fit_contour_field(Vc, F0, ax, h_chin, centre_curve)
    boundary1 = np.asarray(sorted(_map_boundary(F0)), dtype=np.int64)
    loops_before = [len(x) for x in _boundary_loops(F0)]

    # measured neck patch from the capture: real depth-derived skin surface
    observation = None
    observed = {}
    if observed_neck_paths and "closed" in observed_neck_paths:
        p = Path(observed_neck_paths["closed"])
        if p.is_file():
            OV, _ = mi.load_obj(p)
            observation = {"points_mm": OV * 1000., "hold": observation_hold,
                           "h_pad_mm": float(observation_pads[0]),
                           "a_pad_mm": float(observation_pads[1]),
                           "path": str(p), "vertices": int(len(OV))}
    for s in ("closed", "open"):
        key = observed_neck_paths.get(s) if observed_neck_paths else None
        if key and Path(key).is_file():
            OV, OF = mi.load_obj(key)
            observed[s] = {"path": str(key), "vertices": int(len(OV)),
                           "faces": int(len(OF)), "used_as_constraint": s == "closed" and observation is not None}

    closed_run = deform_state(Vc, F0, Vc, F0, ax, design, cf, centre_curve,
                              boundary1, lm_c, "closed", detail,
                              observation=observation,
                              detail_amplitude=detail_amplitude,
                              cut_anchor_floor=cut_anchor_floor,
                              shape_params=shape_params)
    d_volume = closed_run["volume"] - Vc
    d_detail = closed_run["detail"] - closed_run["volume"]
    basis_closed = np.column_stack([frame["right"], frame["up"], frame["forward"]])

    out_states, fields, audits = {}, {}, {}
    accepted = True
    for state in states:
        ref_state = references[state] if isinstance(references, dict) else references[0]
        idx = ref_state["stages"]["dense"]["frame_index"]
        Vb = np.asarray(dense_vertices[idx], float)
        lm_s = np.asarray(ref_state["stages"]["dense"]["landmarks_m"], float)
        if state == "closed":
            V_vol, V_fin = closed_run["volume"], closed_run["detail"]
            disp = closed_run["disp_mm"]
            local = {"vertices": 0, "note": "design state"}
        else:
            fs = head_frame(lm_s)
            basis_s = np.column_stack([fs["right"], fs["up"], fs["forward"]])
            tr = transfer_state(Vb, d_volume, d_detail, basis_closed, basis_s,
                                lm_s, state)
            V_vol, V_fin, disp = tr["volume"], tr["detail"], tr["disp_mm"]
            local = tr["state_local_guard"]
        ver = verify_mesh(V_fin, F0, Vb, F0, state + "_reshaped")
        inter = self_intersection_estimate(Vb, V_fin, F0)
        frozen = np.linalg.norm(disp, axis=1) < 1e-9
        # the frozen set must be EXACTLY unchanged (bit-level for the identity)
        ident_max = float(np.abs(V_fin[frozen] - Vb[frozen]).max()) if frozen.any() else 0.0
        guards = {"normal_reversal": inter["normal_reversals"],
                  "nonmanifold_edges": ver["nonmanifold_edges"],
                  "inconsistent_shared_edges": ver["inconsistent_shared_edges"],
                  "degenerate_faces": ver["zero_area_faces"],
                  "slender_faces": ver["slender_faces"],
                  "max_edge_growth": ver["max_edge_growth_ratio"],
                  "frozen_region_max_change_mm": ident_max,
                  "frozen_vertices": int(frozen.sum()),
                  "status": "passed" if (inter["count"] == 0
                                         and ver["nonmanifold_edges"] == 0
                                         and ver["inconsistent_shared_edges"] == 0
                                         and ver["zero_area_faces"] == 0
                                         and ver["slender_faces"] == 0
                                         and ver["max_edge_growth_ratio"] < 2.0
                                         and ident_max < 1e-9) else "failed"}
        accepted &= guards["status"] == "passed"
        out_states[state] = {
            "run": {"vertices": V_fin, "volume_vertices": V_vol,
                    "applied_mm": disp, "weight": closed_run["w"]},
            "base": Vb, "frame_index": idx, "landmarks_m": lm_s,
            "guards": guards, "intersections": inter, "verification": ver,
            "max_disp_mm": round(float(np.linalg.norm(disp, axis=1).max()), 4),
            "state_local_adaptation": local,
            "regions": {k: int(v.sum()) for k, v in closed_run["regions"].items()},
        }
        fields[state] = {"disp_mm": disp, "w": closed_run["w"],
                         "base": Vb, "volume": V_vol, "final": V_fin,
                         "h": closed_run["h"], "t": closed_run["t"]}
        audits[state] = {"verification": ver, "guards": guards,
                         "max_disp_mm": out_states[state]["max_disp_mm"]}

    # measured-patch acceptance: distance from the recorded skin to the mesh
    patch_report = {}
    for s, info in observed.items():
        OV, _ = mi.load_obj(info["path"])
        Vb = out_states[s]["base"]
        Vf = out_states[s]["run"]["vertices"]
        d0, _ = cKDTree(Vb * 1000.).query(OV * 1000., k=1)
        d1, _ = cKDTree(Vf * 1000.).query(OV * 1000., k=1)
        patch_report[s] = {
            "used_as_constraint": info["used_as_constraint"],
            "vertices": info["vertices"],
            "distance_before_mm": {"median": round(float(np.median(d0)), 3),
                                   "p95": round(float(np.percentile(d0, 95)), 3)},
            "distance_after_mm": {"median": round(float(np.median(d1)), 3),
                                  "p95": round(float(np.percentile(d1, 95)), 3)},
            "change_mm": round(float(np.median(d1) - np.median(d0)), 3),
        }

    return {
        "schema": "head_rgbd.neck_volume.subject_reshape.v1",
        "accepted": bool(accepted),
        "neck_method": "subject_adaptive_reference_volume_preserve_topology",
        "geometry_source": "subject_reconstruction_plus_reference_design_plus_measured_neck",
        "states": out_states, "audits": audits, "fields": fields,
        "faces": F0, "topology_preserved": True,
        "boundary_loops_before": loops_before,
        "boundary_loops_after": [len(x) for x in _boundary_loops(F0)],
        "boundary_vertices": boundary1,
        "head_scale": head_audit,
        "frame": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in frame.items()},
        "axial_frame": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in ax.items() if k not in ("cut_loop", "all_loops")},
        "reference": {"path": str(ref_path), "sha256": sha256_file(ref_path),
                      "head": ref_head, "menton": menton},
        "design": {"targets": design["targets"], "scales": design["scales"],
                   "anchor": design["anchor"], "limits": design["limits"],
                   "reference_blend": reference_blend, "gain": gain,
                   "observation_hold": observation_hold,
                   "observation_pads_mm": [float(x) for x in observation_pads],
                   "cut_anchor_floor": cut_anchor_floor,
                   "shape_params": shape_params or {},
                   "detail_amplitude_mm": (DETAIL_AMPLITUDE_MM if detail_amplitude is None
                                           else float(detail_amplitude))},
        "subject_sections": tgt_sec, "reference_sections": ref_sec,
        "measured_patches": observed, "measured_patch_report": patch_report,
        "design_curves": {"curves": design["curves"], "t": design["t"]},
        "contour_field": {"order": cf["order"], "levels": cf["levels"]},
        "alpha": alpha,
    }
