"""Continuous non-negative laryngeal appearance from a reference (0.6).

The reference supplies shape *proportions* only: peak height, 50%/10% width and
length, upper/lower transitions and apex bluntness, all measured on the positive
central lobe of a bounded anterior field.  Its negative background residual is
never transported.  The target receives one analytic, non-negative C2 bump whose
support is fitted to the measured reference profiles and whose position is
mapped from the reference neck window into the target's measured reliable neck
segment (submental boundary to neck root).  If the full support does not fit in
both states, the whole design (height, width, length, tip) is scaled together;
the width alone is never squeezed.

Displacement is along a smooth outward direction field, not one constant vector.
Everything relative to the unshaped baseline is audited in millimetres; the
normal component is required to stay at or above zero (0.05 mm numerical
tolerance).  This remains an appearance edit (editing_mode=true,
geometry_source=reference_driven_appearance, subject_observation_matched=false).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from head_rgbd import neck_appearance as na
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_proportional as prop
from head_rgbd import neck_reference_auto as nra
from head_rgbd import neck_subject_v3 as v3

NECK_METHOD_CONTINUOUS = "continuous_reference_appearance"
GEOMETRY_SOURCE_CONTINUOUS = "reference_driven_appearance"
TAPER_MM = 8.0                     # protection ramp (mm); 10% contour must clear it
MIN_TEN_CLEARANCE_MM = 8.0         # 10% contour must be this clear of protection
DESIGN_MARGIN_MM = 1.0             # 10% contour must sit inside the neck window
FIT_STEPS = (1.0, .96, .92, .88, .84, .80, .76, .72)
DEFAULT_ALPHA = .55                # frame normal / smoothed surface normal blend


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Reference morphology
# ---------------------------------------------------------------------------

def _plateau(dv_mm, up_mm, dn_mm):
    return na._vertical_envelope(dv_mm, up_mm, dn_mm, "plateau")


def _lateral_shape(u_rel, half, tip, offset=0.0):
    half = max(float(half), 1e-3)
    tip = max(float(tip), 1e-3)
    e = tip / half
    s = (np.asarray(u_rel) - float(offset)) / half
    bracket = 1.0 - (np.sqrt(s * s + e * e) - e)
    support_end = 1.0 + e
    flare = 0.22 * support_end
    window = na._quintic_window((np.abs(s) - (support_end - flare)) / max(flare, 1e-6))
    return np.clip(bracket, 0.0, None) * window


def _central_component(field, peak, fraction):
    m = field >= peak * fraction
    lab, n = label(m)
    iy, ix = np.unravel_index(np.argmax(field), field.shape)
    k = lab[iy, ix]
    if k == 0:
        return np.zeros_like(m), []
    comp = lab == k
    islands = []
    for other in range(1, n + 1):
        if other == k:
            continue
        c = lab == other
        if c.sum() < 4:
            continue
        yy, xx = np.where(c)
        islands.append({"vertices": int(c.sum()),
                        "peak_mm": round(float(field[c].max()), 3),
                        "u_mm": [round(float(xx.min()), 1), round(float(xx.max()), 1)],
                        "v_mm": [round(float(yy.min()), 1), round(float(yy.max()), 1)]})
    return comp, islands


def reference_window(Vr, face_peak_y):
    """Diagnostic reference neck window from the front outline and widths.

    Returns the ROI window mapping used for placement plus the geometric
    under-chin / shoulder-widening window for the record.  The geometric window
    is not used when the shoulder widening overlaps the larynx (short reference
    neck), which is reported rather than silently ignored.
    """
    x, y, z = Vr[:, 0], Vr[:, 1], Vr[:, 2]
    rows = []
    for yc in np.arange(40.0, -220.0, -2.0):
        band = (y <= yc) & (y > yc - 2.0)
        if band.sum() < 4:
            continue
        mid = band & (np.abs(x) < 20.0)
        if mid.sum() < 4:
            continue
        zmax = float(np.max(z[mid]))
        width = float(np.ptp(np.percentile(x[band], [2, 98])))
        rows.append((yc, zmax, width))
    if not rows:
        return {"roi_window_mm": [-125.0, -81.0], "s_roi": None,
                "geometric": {"status": "unavailable"}}
    rows = np.asarray(rows)
    # under-chin turn: strongest negative outline gradient below the chin peak
    top = rows[rows[:, 0] > -60.0]
    chin_y = float(top[np.argmax(top[:, 1]), 0]) if len(top) else 0.0
    below = rows[(rows[:, 0] < chin_y - 5.0) & (rows[:, 0] > chin_y - 100.0)]
    geom = {"chin_y_mm": round(chin_y, 2), "status": "partial"}
    if len(below) >= 4:
        grad = np.diff(below[:, 1]) / np.diff(below[:, 0])
        # gradient is per mm downward (dy negative); the submental turn is the
        # strongest negative drop in front of the throat
        i = int(np.argmin(grad))
        turn_y = float(below[i, 0])
        neck_width = float(np.median(rows[(rows[:, 0] < turn_y - 2.0) &
                                          (rows[:, 0] > turn_y - 25.0), 2]))
        lower = rows[rows[:, 0] < turn_y - 10.0]
        root_y = None
        if len(lower) >= 3:
            for row in lower:
                if row[2] > 1.18 * neck_width:
                    root_y = float(row[0])
                    break
        geom.update({
            "under_chin_turn_y_mm": round(turn_y, 2),
            "neck_width_mm": round(neck_width, 2),
            "shoulder_widening_y_mm": None if root_y is None else round(root_y, 2),
            "s_larynx": (None if root_y is None or root_y >= turn_y
                         else round((turn_y - face_peak_y) / (turn_y - root_y), 3)),
            "status": "ok" if root_y is not None else "no_widening",
        })
        if root_y is not None and root_y >= turn_y:
            geom["status"] = "invalid_widening"
    roi_top, roi_bottom = -81.0, -125.0
    s_roi = (roi_top - face_peak_y) / (roi_top - roi_bottom)
    return {"roi_window_mm": [roi_bottom, roi_top],
            "s_roi": round(float(s_roi), 3),
            "geometric": geom,
            "used": "roi_window",
            "why": "the geometric shoulder widening overlaps the reference larynx; "
                   "the bounded anterior ROI (under-chin turn excluded above -81 mm, "
                   "shoulders excluded below -125 mm) is the reliable window"}


def extract_reference_morphology(path: str | Path) -> dict[str, Any]:
    """Measure positive-lobe proportions; negative residual is never used."""
    path = Path(path)
    asset, Vr, Fr = nra.extract_reference(path)
    x = np.asarray(asset["x_mm"], float)
    y = np.asarray(asset["y_mm"], float)
    field = np.asarray(asset["field_mm"], float)
    peak = float(field.max())
    iy, ix = np.unravel_index(np.argmax(field), field.shape)
    comp10, islands = _central_component(field, peak, .1)
    comp50, _ = _central_component(field, peak, .5)
    for isl in islands:
        isl["u_mm"] = [round(float(x[int(isl["u_mm"][0])]), 1),
                       round(float(x[int(isl["u_mm"][1])]), 1)]
        isl["v_mm"] = [round(float(y[int(isl["v_mm"][0])]), 1),
                       round(float(y[int(isl["v_mm"][1])]), 1)]
    X, Y = np.meshgrid(x, y)
    ys, xs = np.where(comp10)
    ys50, xs50 = np.where(comp50)
    W10, L10 = float(x[xs].max() - x[xs].min()), float(y[ys].max() - y[ys].min())
    W50, L50 = float(x[xs50].max() - x[xs50].min()), float(y[ys50].max() - y[ys50].min())
    up10 = float(y[iy] - y[ys].min()); dn10 = float(y[ys].max() - y[iy])
    up50 = float(y[iy] - y[ys50].min()); dn50 = float(y[ys50].max() - y[iy])
    # midline vertical profile through the apex column
    v_rel = y - y[iy]
    col = field[:, ix]
    take = comp10[:, ix] & (col > 0.02 * peak)
    w = np.ones(int(take.sum()))
    def vres(p):
        model = _plateau(v_rel[take], p[0], p[1])
        return (model - col[take] / peak) * w
    vfit = least_squares(vres, [max(up10 * 1.4, 4.0), max(dn10 * 1.4, 4.0)],
                         bounds=([3.0, 3.0], [40.0, 40.0]))
    L_up, L_dn = (float(v) for v in vfit.x)
    v_rms = float(np.sqrt(np.mean(vres(vfit.x) ** 2)))
    # transverse profile through the apex row, limited to the central lobe
    u_rel = x - x[ix]
    row = field[iy, :]
    take_u = comp10[iy, :] & (row > 0.02 * peak)
    take_u &= np.abs(u_rel) <= (W10 / 2.0 + 4.0)
    def lres(p):
        model = _lateral_shape(u_rel[take_u], p[0], p[1], p[2])
        return model - row[take_u] / peak
    lfit = least_squares(lres, [max(W50 / 2.0, 4.0), 3.0, 0.0],
                         bounds=([4.0, .5, -5.0], [25.0, 14.0, 5.0]))
    half, tip, offset = (float(v) for v in lfit.x)
    l_rms = float(np.sqrt(np.mean(lres(lfit.x) ** 2)))
    # realized extents of the fitted analytic design at the apex
    uu = np.arange(-30., 30.001, .1)
    prof = _lateral_shape(uu, half, tip, offset)
    over = prof >= .5
    W50_fit = float(np.ptp(uu[over])) if over.any() else None
    vv = np.arange(-40., 40.001, .1)
    env = _plateau(vv, L_up, L_dn)
    L50_fit = float(np.ptp(vv[env >= .5])) if (env >= .5).any() else None
    up50_fit = float(np.max(vv[env >= .5])) if (env >= .5).any() else None
    dn50_fit = float(-np.min(vv[env >= .5])) if (env >= .5).any() else None
    # apex bluntness on the positive field
    curve = []
    for win in (2., 3., 4.):
        m = np.hypot(X - x[ix], Y - y[iy]) <= win
        du, dv = X[m] - x[ix], Y[m] - y[iy]
        A = np.stack([du * du, du * dv, dv * dv, du, dv, np.ones_like(du)], 1)
        coef = np.linalg.lstsq(A, field[m], rcond=None)[0]
        H = np.array([[2 * coef[0], coef[1]], [coef[1], 2 * coef[2]]])
        ev = np.linalg.eigvalsh(H)
        curve.append({"window_mm": win, "radius_mm": round(float(1 / max(abs(ev))), 2),
                      "anisotropy": round(float(np.abs(ev).max() / max(np.abs(ev).min(), 1e-12)), 2),
                      "fit_rms_mm": round(float(np.sqrt(np.mean((field[m] - A @ coef) ** 2))), 3)})
    ids = np.linspace(0, len(vv) - 1, 121).astype(int)
    window = reference_window(Vr * 1000.0, y[iy])
    return {
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "source_axes": asset["source_axes"], "source_units": asset["source_units"],
        "reference_neck_width_mm": float(asset["reference_neck_width_mm"]),
        "peak_mm": round(peak, 3), "peak_xy_mm": [round(float(x[ix]), 2), round(float(y[iy]), 2)],
        "negative_extent_mm": round(float(field.min()), 3),
        "width50_mm": round(W50, 3), "length50_mm": round(L50, 3),
        "width10_mm": round(W10, 3), "length10_mm": round(L10, 3),
        "up50_mm": round(up50, 3), "down50_mm": round(dn50, 3),
        "up10_mm": round(up10, 3), "down10_mm": round(dn10, 3),
        "islands_10pct": islands,
        "shoulders_used": False,
        "fit": {
            "vertical_support_up_mm": round(L_up, 3), "vertical_support_down_mm": round(L_dn, 3),
            "vertical_rms": round(v_rms, 4),
            "lateral_half_width_mm": round(half, 3), "tip_radius_mm": round(tip, 3),
            "centre_offset_mm": round(offset, 3), "lateral_rms": round(l_rms, 4),
            "width50_fit_mm": None if W50_fit is None else round(W50_fit, 2),
            "length50_fit_mm": None if L50_fit is None else round(L50_fit, 2),
            "up50_fit_mm": None if up50_fit is None else round(up50_fit, 2),
            "down50_fit_mm": None if dn50_fit is None else round(dn50_fit, 2),
        },
        "apex_curvature": curve,
        "window": window,
        "background_sensitivity": nra.background_sensitivity(path),
        "midline_profile": {"v_rel_mm": [round(float(v), 2) for v in vv[ids]],
                            "height_mm": [round(float(v), 3) for v in (peak * env)[ids]]},
        "transverse_profile": {"u_rel_mm": [round(float(v), 2) for v in uu[::50]],
                               "height_mm": [round(float(peak * v), 3) for v in prof[::50]]},
        "note": "positive central lobe only; the separate 1.55 mm shoulder islands at "
                "|u|~26-32 mm and the negative background residual are recorded but not "
                "transported. Peak is an appearance residual against the reference background, "
                "not an anatomical larynx height.",
    }


# ---------------------------------------------------------------------------
# Target measurement and placement
# ---------------------------------------------------------------------------

def _base_states(references, dense_vertices, dense_faces):
    base = {}
    for state, ref in references.items():
        V = np.asarray(dense_vertices[ref["stages"]["dense"]["frame_index"]], float)
        M = np.asarray(ref["animated_model_to_reference_camera_4x4"], float)
        R, t = M[:3, :3], M[:3, 3]
        lm = np.asarray(ref["stages"]["dense"]["landmarks_m"], float)
        lm_model = (lm - t) @ R
        axes = npn._direction_frame(lm_model)
        normals_model = npn.vertex_normals((V - t) @ R, dense_faces)
        protection = npn._protected_mask((V - t) @ R, lm_model, normals_model, axes)
        base[state] = {
            "vertices": V, "protection": protection["mask"],
            "up": R @ axes["up"], "forward": R @ axes["forward"], "right": R @ axes["right"],
            "chin": lm_model[npn.CHIN] @ R.T + t, "normals": normals_model @ R.T,
            "reference": ref,
        }
    return base


def measure_target(references, dense_vertices, dense_faces):
    base = _base_states(references, dense_vertices, dense_faces)
    closed = base["closed"]
    scale = prop.measure_neck_scale(
        {s: {"vertices": base[s]["vertices"], "faces": dense_faces,
             "protection": base[s]["protection"]} for s in base},
        chin_camera=closed["chin"], head_up=closed["up"],
        head_forward=closed["forward"], head_right=closed["right"], label_point=None)
    return base, scale


def _state_frame(state_item, vertex, source):
    V = state_item["vertices"]
    normal = state_item["normals"][vertex].copy()
    normal /= max(np.linalg.norm(normal), 1e-12)
    up = state_item["up"] - np.dot(state_item["up"], normal) * normal
    up /= max(np.linalg.norm(up), 1e-12)
    u_axis = np.cross(up, normal); u_axis /= max(np.linalg.norm(u_axis), 1e-12)
    v_axis = np.cross(normal, u_axis); v_axis /= max(np.linalg.norm(v_axis), 1e-12)
    return v3.LocalNeckFrame(origin=V[vertex].copy(), n=normal, up=up,
                             u_axis=u_axis, v_axis=v_axis,
                             audit={"frame_source": source, "vertex": int(vertex)})


def _center_vertex(base, scale, s):
    closed, other = base["closed"], base["open"]
    chin, up, right, forward = closed["chin"], closed["up"], closed["right"], closed["forward"]
    v_s0 = float((scale.s0_point_camera - chin) @ up)
    v_root = float((scale.root_point_camera - chin) @ up)
    v_c = v_s0 + s * (v_root - v_s0)
    rel = closed["vertices"] - chin
    u, v, f = rel @ right, rel @ up, rel @ forward
    band_mid = np.abs(v - v_c) < 0.004
    u_mid = float(np.median(u[band_mid])) if band_mid.sum() >= 5 else 0.0
    free = ~closed["protection"] & ~other["protection"]
    front = ((closed["normals"] @ closed["forward"]) > 0.35) & \
            ((other["normals"] @ other["forward"]) > 0.35)
    for du, dv in ((.012, .004), (.018, .008), (.025, .012)):
        band = (np.abs(v - v_c) < dv) & (np.abs(u - u_mid) < du) & free & front
        if band.any():
            ids = np.flatnonzero(band)
            score = f[ids] - 0.002 * ((u[ids] - u_mid) ** 2 + (v[ids] - v_c) ** 2)
            j = int(ids[int(np.argmax(score))])
            return j, {"s": round(float(s), 3), "v_c_mm": round(float(v_c) * 1000, 2),
                       "u_mid_mm": round(u_mid * 1000, 2),
                       "vertex": j, "source": "anterior_midline_proportional_binding"}
    return None, {"s": round(float(s), 3), "reason": "no_shared_unprotected_front_vertex"}


def measure_reference_segment(morph):
    return float(morph["window"]["s_roi"])


def design_from_morphology(morph, q, f=1.0, width_override=None):
    fit = morph["fit"]
    H = morph["peak_mm"] * q * f
    half = fit["lateral_half_width_mm"] * q * f
    tip = fit["tip_radius_mm"] * q * f
    offset = fit["centre_offset_mm"] * q * f
    L_up = fit["vertical_support_up_mm"] * q * f
    L_dn = fit["vertical_support_down_mm"] * q * f
    offset = float(np.clip(offset, -0.15 * half, 0.15 * half))
    if width_override is not None:
        half = 0.5 * float(width_override) * q * f
    design = na.SculptDesign(
        height_mm=float(H), width_total_mm=float(2.0 * half),
        length_up_mm=float(L_up), length_down_mm=float(L_dn),
        tip_radius_mm=float(tip), centre_offset_mm=float(offset),
        envelope="plateau")
    realized_w50 = _analytic_width50(design)
    return design, {
        "q_scale": round(float(q), 5), "fit_scale": round(float(f), 4),
        "height_mm": round(float(H), 3), "width_total_mm": round(float(2 * half), 3),
        "support_up_mm": round(float(L_up), 3), "support_down_mm": round(float(L_dn), 3),
        "tip_radius_mm": round(float(tip), 3), "centre_offset_mm": round(float(offset), 3),
        "analytic_w50_mm": None if realized_w50 is None else round(realized_w50, 2),
        "envelope": "plateau (1-t^2)^3; C2 zero value and slope at the support boundary",
    }


def _analytic_width50(design):
    u = np.arange(-60., 60.001, .1)
    p = na.sculpt_profile(u, np.full_like(u, design.v_shift_mm), design)
    over = p >= .5 * design.height_mm
    return float(np.ptp(u[over])) if over.any() else None


def _design_extents(design):
    u = np.arange(-60., 60.001, .1)
    lat = na.sculpt_profile(u, np.zeros_like(u) + 0.0, design)
    v = np.arange(-60., 60.001, .1)
    env = _plateau(v, design.length_up_mm, design.length_down_mm) * design.height_mm
    out = {}
    for name, arr, axis, peak in (("u", lat, u, design.height_mm),
                                  ("v", env, v, design.height_mm)):
        for frac in (.1, .5):
            m = arr >= peak * frac
            out[f"{name}{int(frac*100)}"] = [float(axis[m].min()), float(axis[m].max())] if m.any() else None
    return out


# ---------------------------------------------------------------------------
# Direction field and application
# ---------------------------------------------------------------------------

def _smooth_normals(vertices, faces, iterations=8):
    normals = npn.vertex_normals(vertices, faces)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    import scipy.sparse as sparse
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vertices),) * 2).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    smooth = normals.copy()
    for _ in range(iterations):
        smooth = 0.5 * smooth + 0.5 * (A @ smooth) / deg[:, None]
        smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-12)
    return normals, smooth


def _direction_field(vertices, faces, frame, local_normals, alpha, mode):
    if mode == "constant":
        D = np.repeat(frame.n[None, :], len(vertices), axis=0)
    else:
        raw, smooth = _smooth_normals(vertices, faces)
        D = alpha * frame.n[None, :] + (1.0 - alpha) * smooth
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    dots = np.einsum("ij,ij->i", D, local_normals)
    bad = dots < 0.2
    if bad.any():
        D[bad] = 0.5 * D[bad] + 0.5 * local_normals[bad]
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    dots = np.einsum("ij,ij->i", D, local_normals)
    return D, {"min_dot_local_normal": round(float(dots.min()), 4),
               "mean_dot_local_normal": round(float(dots.mean()), 4),
               "mode": mode, "alpha": float(alpha)}


def _apply(vertices, faces, frame, design, protected, alpha, mode, local_normals):
    uv = frame.uv(vertices)
    p = np.clip(na.sculpt_profile(uv[:, 0], uv[:, 1], design), 0.0, None)
    ids = np.flatnonzero(protected)
    dist = cKDTree(vertices[ids]).query(vertices)[0] * 1000.0 if len(ids) else \
        np.full(len(vertices), 1e9)
    gate = nra.smooth_gate(dist / TAPER_MM) * v3._region_window(vertices, frame, protected)
    D, direction_audit = _direction_field(vertices, faces, frame, local_normals, alpha, mode)
    applied = p * gate  # mm along D
    disp = applied[:, None] * D * 1e-3
    shaped = vertices + disp
    normal = np.einsum("ij,ij->i", disp, local_normals) * 1000.0
    tang = np.linalg.norm(disp - normal[:, None] * 1e-3 * local_normals, axis=1) * 1000.0
    return {
        "vertices": shaped, "sculpt_mm": p, "gate": gate, "applied_mm": applied,
        "direction": D, "normal_mm": normal, "tangential_mm": tang,
        "distance_to_protected_mm": dist, "direction_audit": direction_audit,
        "front_mask": (local_normals @ frame.n) > .2,
    }


def _build_refined(base, base_faces, frames):
    centroid = {s: base[s]["vertices"][base_faces].mean(axis=1) for s in base}
    support_pad = 0.024
    mask1 = np.zeros(len(base_faces), bool)
    for s in base:
        mask1 |= np.linalg.norm(centroid[s] - frames[s].origin, axis=1) < support_pad
    sub1 = v3.subdivide_local(base["closed"]["vertices"], base_faces, face_mask=mask1)
    l1 = sub1["midpoint_edges"]
    refined1 = {s: na._rebuild_refined(base[s]["vertices"], l1, []) for s in base}
    mask2 = np.zeros(len(sub1["faces"]), bool)
    for s in base:
        c = refined1[s][sub1["faces"]].mean(axis=1)
        mask2 |= np.linalg.norm(c - frames[s].origin, axis=1) < 0.017
    sub2 = v3.subdivide_local(refined1["closed"], sub1["faces"], face_mask=mask2)
    l2 = sub2["midpoint_edges"]
    faces = sub2["faces"]
    base_count = len(base["closed"]["vertices"])
    final = {s: na._rebuild_refined(base[s]["vertices"], l1, l2) for s in base}
    return {"faces": faces, "level1_edges": l1, "level2_edges": l2,
            "base_count": base_count, "refined1": refined1, "final": final}


def _refined_vertices_and_protection(base, state, refined, l1, l2, base_count):
    verts = refined[state]
    protected = np.zeros(len(verts), bool)
    protected[:base_count] = base[state]["protection"]
    off1 = base_count
    for k, (i, j) in enumerate(l1):
        protected[off1 + k] = base[state]["protection"][i] or base[state]["protection"][j]
    off2 = off1 + len(l1)
    for k, (i, j) in enumerate(l2):
        protected[off2 + k] = protected[i] or protected[j]
    return verts, protected


def _find_center(base, scale, s, frames_from=None):
    vertex, audit = _center_vertex(base, scale, s)
    if vertex is None:
        return None, audit
    frames = {st: _state_frame(base[st], vertex, "proportional_midline_binding")
              for st in base}
    return frames, {**audit, "frames": {st: frames[st].as_dict() for st in frames}}


def _ten_contour_uv(design):
    half = design.half_width_mm + design.tip_radius_mm + 3.0
    vpad = max(design.length_up_mm, design.length_down_mm) + 3.0
    u = np.arange(-half, half + 1e-9, .5)
    v = np.arange(-vpad, vpad + 1e-9, .5)
    U, V = np.meshgrid(u, v)
    p = na.sculpt_profile(U, V, design)
    m = np.abs(p - .1 * design.height_mm) <= .015 * design.height_mm
    return np.stack([U[m], V[m]], axis=1)


def _search_placement(base, scale, morph, q):
    """Collect feasible (fit scale, centre) candidates for the full support.

    The cheap screen uses the design's analytic 10% extents for the neck-window
    check and the front base vertices for a clearance proxy.  The final choice
    is made on the refined mesh by build_states; the coarse mesh alone is not
    trusted to represent the 10% contour.
    """
    s_ref = measure_reference_segment(morph)
    candidates = []
    for f in FIT_STEPS:
        design, design_audit = design_from_morphology(morph, q, f)
        ext = _design_extents(design)
        v10 = ext.get("v10")
        if not v10:
            continue
        contour_uv = _ten_contour_uv(design)
        up10, dn10 = float(v10[1]), float(-v10[0])
        for s in np.arange(max(.36, s_ref - .12), min(.84, s_ref + .28) + 1e-9, .02):
            frames, audit = _find_center(base, scale, s)
            if frames is None:
                continue
            details = {}
            ok = True
            for st in base:
                V = base[st]["vertices"]
                uv = frames[st].uv(V)
                p = np.clip(na.sculpt_profile(uv[:, 0], uv[:, 1], design), 0., None)
                front = (base[st]["normals"] @ frames[st].n) > .2
                p10 = (p >= .1 * design.height_mm) & front
                if not p10.any():
                    ok = False
                    break
                ids = np.flatnonzero(base[st]["protection"])
                dist = cKDTree(V[ids]).query(V)[0] * 1000 if len(ids) else np.full(len(V), 1e9)
                clearance = float(dist[p10].min())
                front_dot = float((base[st]["normals"][p10] @ frames[st].n).min())
                front_ids = np.flatnonzero(front)
                contour_clearance = clearance
                if len(front_ids) and len(contour_uv):
                    tree = cKDTree(uv[front_ids])
                    _, nearest = tree.query(contour_uv)
                    contour_clearance = float(dist[front_ids][nearest].min())
                details[st] = {"clearance_proxy_mm": round(clearance, 2),
                               "contour_clearance_proxy_mm": round(contour_clearance, 2),
                               "front_dot": round(front_dot, 3),
                               "p10_base_vertices": int(p10.sum())}
                if clearance < MIN_TEN_CLEARANCE_MM or contour_clearance < MIN_TEN_CLEARANCE_MM:
                    ok = False
                    break
            if not ok:
                continue
            fr = frames["closed"]
            crease = float((scale.s0_point_camera - fr.origin) @ fr.v_axis * 1000)
            root = float((scale.root_point_camera - fr.origin) @ fr.v_axis * 1000)
            if up10 > crease - 0.5 or -dn10 < root + 0.5:
                continue
            candidates.append({
                "f": f, "s": float(s), "design": design, "design_audit": design_audit,
                "frames": frames, "center": audit, "details": details,
                "top10_mm": up10, "bottom10_mm": -dn10,
                "crease_mm": crease, "root_mm": root,
                "extents": ext,
            })
    candidates.sort(key=lambda c: (c["f"], -abs(c["s"] - s_ref)), reverse=True)
    return candidates


def build_states(*, states, references, dense_vertices, dense_faces, reference_path,
                 alpha=DEFAULT_ALPHA, mode="smooth", fixed_center_vertex=None,
                 q_override=None, f_override=None, s_override=None,
                 direction_modes=None, require_support=True):
    """Build both states; the same base-topology vertex is the shared binding."""
    morph = extract_reference_morphology(reference_path)
    base, scale = measure_target(references, dense_vertices, dense_faces)
    q = float(q_override) if q_override is not None else \
        scale.W_neck_mm / morph["reference_neck_width_mm"]
    if fixed_center_vertex is not None:
        vertex = int(fixed_center_vertex)
        frames = {st: _state_frame(base[st], vertex, "historical_center_vertex")
                  for st in base}
        design, design_audit = design_from_morphology(
            morph, q, 1.0 if f_override is None else f_override)
        candidates = [{"source": "fixed_center_vertex", "vertex": vertex,
                       "s": s_override, "design": design, "design_audit": design_audit,
                       "frames": frames, "details": {}, "f": 1.0}]
    else:
        candidates = _search_placement(base, scale, morph, q)
        if not candidates:
            raise RuntimeError("no feasible placement: the continuous support does not fit "
                               "the measured neck window in both states")
    s_ref = measure_reference_segment(morph)
    tried = []
    chosen = None
    for cand in candidates[:10]:
        frames = cand["frames"]
        design = cand["design"]
        design_audit = cand["design_audit"]
        refined = _build_refined(base, dense_faces, frames)
        faces = refined["faces"]
        l1, l2 = refined["level1_edges"], refined["level2_edges"]
        base_count = refined["base_count"]
        results = {}
        for state in states:
            verts, protected = _refined_vertices_and_protection(
                base, state, refined["final"], l1, l2, base_count)
            local_normals = npn.vertex_normals(verts, faces)
            modes = direction_modes or [mode]
            direction_runs = {}
            for dm in modes:
                run = _apply(verts, faces, frames[state], design, protected,
                             alpha, dm, local_normals)
                moved = np.abs(run["applied_mm"]) > 1e-9
                guards = na.geometry_guards(rest=verts, shaped=run["vertices"],
                                            faces=faces, moved=moved)
                inter = nra.local_intersections(run["vertices"], faces, moved)
                direction_runs[dm] = {
                    "guards": guards, "intersections": inter,
                    "max_tangential_mm": round(float(run["tangential_mm"].max()), 3),
                    "mean_tangential_mm": round(float(run["tangential_mm"].mean()), 3),
                    "min_normal_mm": round(float(run["normal_mm"].min()), 3),
                    "max_normal_mm": round(float(run["normal_mm"].max()), 3),
                    "direction": run["direction_audit"],
                    "moved_vertices": int(moved.sum()),
                }
            results[state] = {
                "vertices": verts, "protected": protected, "frame": frames[state],
                "base_vertices": base[state]["vertices"],
                "base_normals": base[state]["normals"],
                "protection": base[state]["protection"],
                "reference": base[state]["reference"],
                "direction_runs": direction_runs,
            }
        safe = [dm for dm in (direction_modes or [mode]) if all(
            results[s]["direction_runs"][dm]["intersections"]["count"] == 0
            and na.accept_guards(results[s]["direction_runs"][dm]["guards"]) for s in states)]
        if not safe:
            tried.append({"f": cand.get("f"), "s": cand.get("s"),
                          "stage": "direction_modes_unsafe",
                          "details": {s: results[s]["direction_runs"] for s in states}})
            continue
        mode_choice = min(safe, key=lambda dm: max(
            results[s]["direction_runs"][dm]["mean_tangential_mm"] for s in states))
        for state in states:
            results[state]["run"] = _apply(
                results[state]["vertices"], faces, results[state]["frame"], design,
                results[state]["protected"], alpha, mode_choice,
                npn.vertex_normals(results[state]["vertices"], faces))
            moved = np.abs(results[state]["run"]["applied_mm"]) > 1e-9
            results[state]["moved"] = moved
            results[state]["guards"] = na.geometry_guards(
                rest=results[state]["vertices"], shaped=results[state]["run"]["vertices"],
                faces=faces, moved=moved)
            results[state]["intersections"] = nra.local_intersections(
                results[state]["run"]["vertices"], faces, moved)
        placement = {
            **cand.get("center", {}), "source": "proportional_search",
            "s_requested": s_ref, "s_chosen": cand.get("s"), "fit_scale": cand.get("f"),
            "q": q, "details": cand.get("details", {}),
            "top10_mm": cand.get("top10_mm"), "bottom10_mm": cand.get("bottom10_mm"),
            "crease_mm": cand.get("crease_mm"), "root_mm": cand.get("root_mm"),
            "extents": cand.get("extents"),
            "support_rule": f"10% contour >= {MIN_TEN_CLEARANCE_MM} mm from protection; "
                            f"within the measured neck window with {DESIGN_MARGIN_MM} mm margin",
        } if fixed_center_vertex is None else {
            "source": "fixed_center_vertex", "vertex": int(fixed_center_vertex),
            "s_requested": s_ref, "s_chosen": cand.get("s"), "fit_scale": cand.get("f"),
            "q": q, "details": cand.get("details", {}),
        }
        provisional = {
            "states": results, "faces": faces, "base_faces": np.asarray(dense_faces, int),
            "level1_edges": l1, "level2_edges": l2, "base_count": base_count,
            "base_vertex_count": base_count,
            "refined_vertices_count": len(results[states[0]]["vertices"]),
            "design": design, "design_audit": design_audit, "placement": placement,
            "reference_morphology": morph, "target_scale": scale, "q": q,
            "direction_mode": mode_choice, "alpha": alpha,
            "direction_comparison": {s: results[s]["direction_runs"] for s in states},
        }
        audits = {s: audit_state(provisional, s) for s in states}
        support_ok = True
        for s in states:
            sup = audits[s]["support"]
            if sup["inside_protected"] != 0:
                support_ok = False
            if sup["clearance_min_mm"] is None or sup["clearance_min_mm"] < MIN_TEN_CLEARANCE_MM:
                support_ok = False
            if sup["gate_min"] is None or sup["gate_min"] < .98:
                support_ok = False
        tried.append({"f": cand.get("f"), "s": cand.get("s"),
                      "support_ok": bool(support_ok),
                      "clearance": {s: audits[s]["support"]["clearance_min_mm"] for s in states},
                      "gate": {s: audits[s]["support"]["gate_min"] for s in states}})
        if support_ok or not require_support:
            provisional["audits"] = audits
            provisional["accepted"] = bool(
                support_ok and all(na.accept_guards(results[s]["guards"])
                                   and results[s]["intersections"]["count"] == 0
                                   for s in states))
            provisional["tried_placements"] = tried
            return provisional
        # if the refined support is too close, try the next candidate
        if support_ok is False and cand.get("f") == FIT_STEPS[0]:
            continue
    raise RuntimeError(
        "no continuous placement passed the refined 10% support audit: "
        + repr(tried[-3:] if tried else []))


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def _base_relative_audit(base_vertices, base_normals, shaped_vertices, base_faces):
    """Displacement relative to the unshaped baseline, normal/tangential split."""
    d = (np.asarray(shaped_vertices) - np.asarray(base_vertices)) * 1000.0
    comp = np.einsum("ij,ij->i", d, base_normals)
    tang = np.linalg.norm(d - comp[:, None] * base_normals, axis=1)
    tri = np.asarray(base_vertices)[base_faces]
    area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 1e6
    face_comp = comp[base_faces].mean(axis=1)
    negative = face_comp < -0.05
    return {
        "max_outward_mm": round(float(comp.max()), 3),
        "min_outward_mm": round(float(comp.min()), 3),
        "min_tolerance_mm": -0.05,
        "negative_vertices_below_tolerance": int((comp < -0.05).sum()),
        "negative_face_area_mm2": round(float(area[negative].sum()), 2),
        "negative_face_fraction": round(float(area[negative].sum() / max(area.sum(), 1e-9)), 5),
        "max_tangential_mm": round(float(tang.max()), 3),
        "mean_tangential_mm": round(float(tang.mean()), 3),
        "moved_vertices": int((np.abs(comp) > 1e-6).sum()),
    }


def _field_components(analysis, level):
    lvl = analysis["levels"][level]
    return {"component_count": lvl["component_count"],
            "central": lvl["central"],
            "global_width_mm": lvl["global_width_mm"],
            "global_length_mm": lvl["global_length_mm"]}


def _radial_monotonicity(uv, applied, front):
    from scipy.interpolate import LinearNDInterpolator
    front_idx = np.flatnonzero(front)
    if len(front_idx) < 20:
        return {"violations": None, "max_ring_drop_mm": None}
    interp = LinearNDInterpolator(uv[front_idx], applied[front_idx])
    apex = uv[int(np.argmax(np.where(front, applied, -np.inf)))]
    support = 30.0
    radii = np.linspace(.5, support, 60)
    angles = np.linspace(0., 2 * np.pi, 48, endpoint=False)
    vals = []
    for r in radii:
        pts = np.stack([apex[0] + r * np.cos(angles), apex[1] + r * np.sin(angles)], 1)
        vals.append(interp(pts))
    vals = np.asarray(vals)
    drops = 0
    max_drop = 0.0
    for k in range(1, vals.shape[0]):
        for a in range(vals.shape[1]):
            previous = np.nanmax(vals[:k, a])
            if np.isfinite(previous) and np.isfinite(vals[k, a]):
                rise = float(vals[k, a] - previous)
                if rise > .01:
                    drops += 1
                    max_drop = max(max_drop, rise)
    return {"violations": int(drops), "max_ring_drop_mm": round(max_drop, 3),
            "angles": len(angles), "radii": len(radii),
            "method": "front-sheet applied field interpolated on polar rings; an increase "
                      "with radius is a groove/valley indicator"}


def _boundary_metrics(design, applied, uv, peak):
    """Analytic support boundary and profile slope metrics (mm)."""
    u = np.arange(-40., 40.001, .1)
    lat = na.sculpt_profile(u, np.zeros_like(u), design)
    v = np.arange(-40., 40.001, .1)
    env = _plateau(v, design.length_up_mm, design.length_down_mm) * design.height_mm
    take_u = np.abs(u) <= design.half_width_mm + design.tip_radius_mm + 2.
    take_v = (v <= design.length_up_mm + 2.) & (v >= -design.length_down_mm - 2.)
    du = np.gradient(lat[take_u], u[take_u])
    dv = np.gradient(env[take_v], v[take_v])
    edge_value = 0.0
    return {
        "max_lateral_slope_mm_per_mm": round(float(np.max(np.abs(du))), 4),
        "max_vertical_slope_mm_per_mm": round(float(np.max(np.abs(dv))), 4),
        "lateral_boundary_value_mm": round(edge_value, 4),
        "vertical_boundary_value_mm": round(edge_value, 4),
        "envelope": "plateau C2; analytic value and slope vanish at the support boundary",
    }


def audit_state(plan, state):
    item = plan["states"][state]
    frame = item["frame"]
    Vr = np.asarray(item["vertices"], float)
    run = item["run"]
    shaped = np.asarray(run["vertices"], float)
    faces = plan["faces"]
    base_faces = plan["base_faces"]
    V0 = np.asarray(item["base_vertices"], float)
    N0 = np.asarray(item["base_normals"], float)
    front = run["front_mask"]
    applied = run["applied_mm"]
    protected = item["protected"]
    dist = run["distance_to_protected_mm"]
    base_audit = _base_relative_audit(V0, N0, shaped[:len(V0)], base_faces)
    refined_normals = npn.vertex_normals(Vr, faces)
    d_ref = (shaped - Vr) * 1000.0
    comp_ref = np.einsum("ij,ij->i", d_ref, refined_normals)
    tri_ref = Vr[faces]
    area_ref = .5 * np.linalg.norm(np.cross(tri_ref[:, 1] - tri_ref[:, 0],
                                            tri_ref[:, 2] - tri_ref[:, 0]), axis=1) * 1e6
    neg_ref = comp_ref[faces].mean(axis=1) < -.05
    uv = frame.uv(Vr)
    analysis = nra.analyse_applied(
        {"uv_mm": uv, "applied_mm": applied, "refined_vertices": Vr, "frame": frame}, faces)
    peak_applied = float(np.max(np.where(front, applied, -np.inf)))
    p10 = front & (applied >= .1 * peak_applied)
    clearance = float(dist[p10].min()) if p10.any() else None
    gate_min = float(run["gate"][p10].min()) if p10.any() else None
    front_dot = float((refined_normals[p10] @ frame.n).min()) if p10.any() else None
    level10 = _field_components(analysis, "0.1")
    level50 = _field_components(analysis, "0.5")
    base_uv = frame.uv(V0)
    base_front = (N0 @ frame.n) > .2
    bv = base_uv[:, 1]
    bn = frame.n_coord(V0) if hasattr(frame, "n_coord") else np.zeros(len(V0))
    v_lo = -(plan["design"].length_down_mm + 6.0)
    v_hi = plan["design"].length_up_mm + 6.0
    band = (base_front & (np.abs(base_uv[:, 0]) < 8.)
            & (base_uv[:, 1] > v_lo) & (base_uv[:, 1] < v_hi))
    if band.sum() >= 10:
        vv = bv[band]; nn = bn[band]
        order = np.argsort(vv)
        vv, nn = vv[order], nn[order]
        chord = nn[0] + (nn[-1] - nn[0]) * (vv - vv[0]) / max(vv[-1] - vv[0], 1e-9)
        base_concavity = {"deviation_min_mm": round(float((nn - chord).min()), 3),
                          "deviation_max_mm": round(float((nn - chord).max()), 3),
                          "range_mm": [round(float(vv.min()), 2), round(float(vv.max()), 2)],
                          "method": "front-sheet base normal coordinate along the midline "
                                    "minus its chord over the neck support range"}
    else:
        base_concavity = {"status": "unavailable"}
    return {
        "state": state,
        "base_relative": base_audit,
        "refined_relative": {
            "min_outward_mm": round(float(comp_ref.min()), 3),
            "negative_vertices_below_tolerance": int((comp_ref < -.05).sum()),
            "negative_face_area_mm2": round(float(area_ref[neg_ref].sum()), 2),
        },
        "applied_peak_mm": round(float(peak_applied), 3),
        "normal_peak_mm": round(float(run["normal_mm"].max()), 3),
        "tangential_max_mm": round(float(run["tangential_mm"].max()), 3),
        "support": {
            "p10_vertices": int(p10.sum()),
            "clearance_min_mm": None if clearance is None else round(clearance, 2),
            "gate_min": None if gate_min is None else round(gate_min, 4),
            "front_dot_min": None if front_dot is None else round(front_dot, 3),
            "inside_protected": int((p10 & protected).sum()),
        },
        "levels": {"0.5": level50, "0.1": level10},
        "analysis": analysis,
        "radial_monotonicity": _radial_monotonicity(uv, applied, front),
        "boundary": _boundary_metrics(plan["design"], applied, uv, peak_applied),
        "base_concavity": base_concavity,
    }
