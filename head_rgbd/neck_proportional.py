"""Proportion-driven laryngeal appearance sculpt (0.5.7).

The 0.5.6 sculpt used fixed millimetre sizes and a Wendland vertical envelope
that decayed too fast, so the delivered ridge was a short, wide mound.  This
module replaces the sizing with measurements taken from the unshaped mesh
itself:

* ``W_head``: robust lower-face width (front-facing band, ears excluded by
  orientation);
* ``W_neck``: median full neck cross-section width over the middle of the neck;
* ``L_neck``: submental crease to neck root, where the root is found from a
  cross-section widening when available and otherwise from a documented
  margin above the mesh cut (lower confidence, recorded);
* a robust neck axis from slice centroids and a normalised coordinate
  ``s in [0, 1]`` along the front midline.

The appearance presets become relative strength factors; the design height,
width and length are derived from the measured scale with the project's
appearance ranges (0.045-0.065 W_neck, 0.26-0.34 W_neck, 0.35-0.50 L_neck for
natural), then clamped by slope, curvature and space limits.  The visible 10 %
and 50 % extents are measured on the delivered mesh, never only on the input
parameters.  The vertical envelope can be switched to a wider C2 plateau
``(1 - t^2)^3`` so the effective ridge stays long without inflating the support.

The centre lives on the shared base topology as a vertex binding; closed and
open map the same binding through their own poses.  A recorded label is only a
suggestion: if its normalised position is implausible the design centre is
chosen from the normalised rule and the original label plus the offset and the
reason are recorded.

This remains a manual appearance design (editing_mode=true,
geometry_source=procedural_appearance, subject_observation_matched=false).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from head_rgbd import neck_appearance as na
from head_rgbd import neck_prominence as npn
from head_rgbd import neck_subject_v3 as v3

PRESET_STRENGTH = {"light": 0.6, "natural": 1.0, "pronounced": 1.4}
HEIGHT_RATIO = 0.055          # natural H / W_neck (design range 0.045-0.065)
WIDTH10_RATIO = 0.30          # W10 / W_neck (design range 0.26-0.34)
LENGTH10_RATIO = 0.46         # L10 / L_neck (design range 0.35-0.50)
UP_FRACTION = 0.38            # of L10 above the centre (longer lower transition)
PLATEAU_L10_RATIO = 0.886     # L10 / support length for the (1-t^2)^3 envelope
WENDLAND_L10_RATIO = 0.56     # L10 / support length for the Wendland envelope
TIP_OF_HEIGHT = 0.42          # tip radius / height
MAX_SLOPE = 1.1               # mm/mm on either profile
CENTRE_S = 0.36               # natural initialisation (design range 0.30-0.42)
S_LABEL_MIN, S_LABEL_MAX = 0.25, 0.60
S_TOP_MARGIN, S_BOTTOM_MARGIN = 0.06, 0.94


@dataclass
class NeckScale:
    W_head_mm: float
    W_neck_mm: float
    L_neck_mm: float
    W_neck_per_state: dict[str, float]
    L_neck_per_state: dict[str, float]
    axis_up: np.ndarray            # camera-frame along-neck direction (downward)
    midline_u_camera: float        # lateral coordinate of the neck midline
    s0_point_camera: np.ndarray    # submental crease on the front midline
    root_point_camera: np.ndarray
    centre_point_camera: np.ndarray
    centre_base_vertex: int
    centre_s: float
    centre_s_requested: float
    label_s: float | None
    label_offset_mm: float | None
    label_reason: str | None
    confidence: dict[str, Any] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    clearance_by_s: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "W_head_mm": round(self.W_head_mm, 2),
            "W_neck_mm": round(self.W_neck_mm, 2),
            "L_neck_mm": round(self.L_neck_mm, 2),
            "W_neck_per_state": {k: round(v, 2) for k, v in self.W_neck_per_state.items()},
            "L_neck_per_state": {k: round(v, 2) for k, v in self.L_neck_per_state.items()},
            "axis_up": [round(float(v), 4) for v in self.axis_up],
            "centre_s": round(self.centre_s, 3),
            "centre_s_requested": round(self.centre_s_requested, 3),
            "centre_base_vertex": int(self.centre_base_vertex),
            "label_s": None if self.label_s is None else round(self.label_s, 3),
            "label_offset_mm": None if self.label_offset_mm is None else round(self.label_offset_mm, 2),
            "label_reason": self.label_reason,
            "confidence": self.confidence,
            "fallbacks": self.fallbacks,
            "clearance_by_s_mm": self.clearance_by_s,
        }


def _slice_profile(vertices, faces, chin_cam, up, right, forward, normals):
    rel = vertices - chin_cam
    v = rel @ up                    # downward is negative
    u = rel @ right
    f = rel @ forward
    front = (normals @ forward > 0.25) & (v < -0.008)
    vmin = float(v[front].min())
    bins = np.arange(-0.008, vmin - 1e-9, -0.002)   # every 2 mm downward
    rows = []
    for vc in bins:
        in_bin = (v <= vc) & (v > vc - 0.002) & (v < -0.008)
        if in_bin.sum() < 6:
            continue
        width = float(u[in_bin].max() - u[in_bin].min())
        front_bin = in_bin & front
        outline = float(f[front_bin].max()) if front_bin.any() else np.nan
        rows.append({"v": float(vc), "width": width, "outline": outline,
                     "u_mid": float((u[in_bin].max() + u[in_bin].min()) * 0.5)})
    return rows, front, v, u, f, vmin


def _submental_crease(rows: list[dict]) -> tuple[float | None, str]:
    usable = [r for r in rows if np.isfinite(r["outline"])]
    if len(usable) < 6:
        return None, "insufficient_outline"
    # the chin itself protrudes; the crease is the local minimum of the front
    # outline in the upper part of the profile
    best = None
    for i in range(2, len(usable) - 2):
        if usable[i]["outline"] <= usable[i - 1]["outline"] and usable[i]["outline"] <= usable[i + 1]["outline"]:
            if best is None or usable[i]["outline"] < usable[best]["outline"]:
                best = i
    if best is None:
        return None, "no_outline_minimum"
    # a crease must sit at least 6 mm below the chin and above the mid-profile
    if usable[best]["v"] > -0.006:
        return None, "crease_too_close_to_chin"
    return usable[best]["v"], "outline_minimum"


def _neck_root(rows: list[dict], s0_v: float) -> tuple[float, str, bool]:
    below = [r for r in rows if r["v"] < s0_v]
    if len(below) < 6:
        return below[-1]["v"] if below else s0_v, "insufficient_slices", False
    widths = np.asarray([r["width"] for r in below])
    vv = np.asarray([r["v"] for r in below])
    upper = widths[len(widths) // 5: max(len(widths) // 2, len(widths) // 5 + 2)]
    reference = float(np.median(upper)) if len(upper) else float(np.median(widths))
    for i in range(len(below)):
        if widths[i] > 1.18 * reference and i > len(below) // 3:
            return float(vv[i]), "cross_section_widening", True
    span = abs(float(vv[-1] - s0_v))
    margin = 0.10 * span
    return float(vv[-1] + margin), "mesh_cut_minus_10pct_margin", False


def measure_neck_scale(
    state_data: dict[str, dict[str, Any]],
    *,
    chin_camera: np.ndarray,
    head_up: np.ndarray,
    head_forward: np.ndarray,
    head_right: np.ndarray,
    label_point: np.ndarray | None,
    label_state: str = "closed",
) -> NeckScale:
    """Measure W_head / W_neck / L_neck and pick the normalised design centre."""
    up = np.asarray(head_up, float) / np.linalg.norm(head_up)
    forward = np.asarray(head_forward, float) / np.linalg.norm(head_forward)
    right = np.asarray(head_right, float) / np.linalg.norm(head_right)

    per_state_w, per_state_l = {}, {}
    rows_ref = None
    normals_ref = None
    for state, item in state_data.items():
        vertices = item["vertices"]
        faces = item["faces"]
        normals = npn.vertex_normals(vertices, faces)
        rows, front, v, u, f, vmin = _slice_profile(vertices, faces, chin_camera, up, right, forward, normals)
        if rows_ref is None and state == label_state:
            rows_ref, normals_ref = rows, normals
        usable = [r for r in rows if r["v"] < -0.010]
        widths_upper = [r["width"] for r in usable if r["v"] > -0.040]
        W_ref = float(np.median(widths_upper)) if widths_upper else float(np.median([r["width"] for r in usable]))
        s0_v = None
        for r in usable:                       # scan down: first full-width neck slice
            if r["width"] >= 0.90 * W_ref:
                s0_v = r["v"]
                break
        crease_method = "width_reaches_90pct_of_neck_reference"
        if s0_v is None:
            s0_v = -0.014
            crease_method = "chin_offset_fallback"
        root_v, root_method, root_grown = _neck_root(rows, s0_v)
        length = abs(root_v - s0_v) * 1000.0
        lower = [r for r in rows if root_v < r["v"] < s0_v]
        mid = [r for r in lower if 0.20 <= (s0_v - r["v"]) / max(s0_v - root_v, 1e-9) <= 0.80]
        width = float(np.median([r["width"] for r in mid])) * 1000.0 if mid else 0.0
        per_state_w[state] = width
        per_state_l[state] = length
    if rows_ref is None:
        rows_ref = rows
        normals_ref = normals
    conf = {
        "W_neck_method": "median full cross-section width over s in [0.25, 0.75]",
        "L_neck_method": f"submental({crease_method}) to neck root({root_method})",
        "neck_root_from_widening": bool(root_grown),
        "states": list(state_data.keys()),
    }
    fallbacks = []
    if not root_grown:
        fallbacks.append("neck_root_from_mesh_cut_margin")
    W_neck = float(np.mean(list(per_state_w.values())))
    L_neck = float(np.mean(list(per_state_l.values())))
    if L_neck <= 0 or W_neck <= 0:
        raise RuntimeError("degenerate neck scale")
    spread = max(per_state_w.values()) - min(per_state_w.values())
    conf["W_neck_state_spread_mm"] = round(float(spread), 2)
    conf["confidence"] = "high" if spread <= 0.15 * W_neck and root_grown else "medium"
    if crease_method != "outline_minimum":
        fallbacks.append("submental_crease_fallback")

    # robust neck axis from the slice midpoints of the reference state
    mids = np.asarray([[r["u_mid"], r["v"]] for r in rows_ref if np.isfinite(r["width"])])
    if len(mids) >= 6:
        # remove the widening skirt before fitting
        keep = mids[:, 1] > (mids[:, 1].min() + 0.08 * (mids[:, 1].max() - mids[:, 1].min()))
        fit = np.polyfit(mids[keep, 1], mids[keep, 0], 1) if keep.sum() >= 4 else np.polyfit(mids[:, 1], mids[:, 0], 1)
        u_at = lambda vv: float(np.polyval(fit, vv))
    else:
        u_at = lambda vv: float(np.median(mids[:, 0])) if len(mids) else 0.0
        fallbacks.append("neck_axis_low_sample")

    # head width: front-facing lower-face band, ears excluded by orientation
    vertices_ref = state_data[label_state]["vertices"]
    rel = vertices_ref - chin_camera
    vv = rel @ up
    uu = rel @ right
    ff = rel @ forward
    face_band = (vv > 0.0) & (vv < 0.060) & (normals_ref @ forward > 0.35)
    if face_band.sum() >= 40:
        u_mid_face = float(np.median(uu[face_band]))
        W_head = 2000.0 * float(np.percentile(np.abs(uu[face_band] - u_mid_face), 90))
    else:
        W_head = float("nan")
        fallbacks.append("head_width_unavailable")

    # normalised centre
    label_s = None
    label_offset_mm = None
    label_reason = None
    s_requested = CENTRE_S
    if label_point is not None:
        rel_l = np.asarray(label_point, float) - chin_camera
        v_label = float(rel_l @ up)
        label_s = float((s0_v - v_label) / max(s0_v - root_v, 1e-9))
        if S_LABEL_MIN <= label_s <= S_LABEL_MAX:
            s_requested = float(label_s)
            label_reason = "label_used_within_allowed_band"
        else:
            label_reason = "label_outside_allowed_band_kept_as_provenance"
    # scan the allowed band for a centre with clearance in BOTH states: the
    # same base vertex must stay clear of the chin protection in closed and open
    base_closed = state_data[label_state]["vertices"]
    other = [s for s in state_data if s != label_state]
    clearance = {}
    best_s, best_score = None, -1e9
    for s_try in np.arange(0.28, 0.561, 0.02):
        v_try = s0_v + s_try * (root_v - s0_v)
        rows_try = [r for r in rows_ref if abs(r["v"] - v_try) < 0.005 and np.isfinite(r["outline"])]
        f_try = float(np.median([r["outline"] for r in rows_try])) if rows_try else 0.0
        point = chin_camera + u_at(v_try) * right + v_try * up + f_try * forward
        vertex = int(np.argmin(np.linalg.norm(base_closed - point, axis=1)))
        d_min = None
        for st, item in state_data.items():
            prot = item.get("protection")
            if prot is None:
                continue
            d = float(np.min(np.linalg.norm(item["vertices"][prot] - item["vertices"][vertex], axis=1)))
            d_min = d if d_min is None else min(d_min, d)
        if d_min is None:
            d_min = 0.0
        clearance[round(float(s_try), 3)] = round(d_min * 1000.0, 2)
        # prefer s closest to the 0.36 initialisation among candidates whose
        # weakest-state clearance covers the protection taper ramp (12 mm)
        enough = d_min >= 0.012
        score = (1000.0 if enough else 0.0) + min(d_min, 0.030) * 1000.0 - 40.0 * abs(s_try - 0.36)
        if score > best_score:
            best_score, best_s = score, float(s_try)
    if best_s is not None and abs(best_s - s_requested) > 0.005:
        s_c = best_s
        label_reason = (label_reason or "") + ";centred_for_two_state_protection_clearance"
    else:
        s_c = float(np.clip(s_requested, 0.28, 0.55))
    v_c = s0_v + s_c * (root_v - s0_v)
    # front midline point at the centre height
    front_rows = [r for r in rows_ref if abs(r["v"] - v_c) < 0.004 and np.isfinite(r["outline"])]
    if front_rows:
        f_c = float(np.median([r["outline"] for r in front_rows]))
    else:
        vp = v_c - 0.004
        front_rows = [r for r in rows_ref if abs(r["v"] - vp) < 0.004 and np.isfinite(r["outline"])]
        f_c = float(np.median([r["outline"] for r in front_rows])) if front_rows else 0.0
        fallbacks.append("centre_front_depth_interpolated")
    centre = chin_camera + (u_at(v_c)) * right + v_c * up + f_c * forward
    # bind to the shared base topology
    base_vertices = state_data[label_state]["vertices"]
    centre_vertex = int(np.argmin(np.linalg.norm(base_vertices - centre, axis=1)))
    centre = base_vertices[centre_vertex].copy()
    if label_s is not None:
        label_offset_mm = float(abs(label_s - s_c) * abs(s0_v - root_v) * 1000.0)
    return NeckScale(
        W_head_mm=W_head,
        clearance_by_s=clearance,
        W_neck_mm=W_neck,
        L_neck_mm=L_neck,
        W_neck_per_state=per_state_w,
        L_neck_per_state=per_state_l,
        axis_up=up,
        midline_u_camera=float(u_at(v_c)),
        s0_point_camera=(chin_camera + u_at(s0_v) * right + s0_v * up
                         + (np.nan_to_num(np.nan) if False else 0.0) * forward
                         + (np.median([r["outline"] for r in rows_ref if abs(r["v"] - s0_v) < 0.004 and np.isfinite(r["outline"])])
                            if any(r["v"] - s0_v < 0.004 and np.isfinite(r["outline"]) for r in rows_ref) else 0.0) * forward),
        root_point_camera=(chin_camera + u_at(root_v) * right + root_v * up + 0.0 * forward),
        centre_point_camera=centre,
        centre_base_vertex=centre_vertex,
        centre_s=s_c,
        centre_s_requested=s_requested,
        label_s=label_s,
        label_offset_mm=label_offset_mm,
        label_reason=label_reason,
        confidence=conf,
        fallbacks=fallbacks,
    )


@dataclass
class ProportionalDesign:
    design: na.SculptDesign
    audit: dict[str, Any]
    envelope: str


def proportional_design(
    scale: NeckScale,
    preset: str,
    *,
    envelope: str = "plateau",
    min_support_gap_mm: float = 4.0,
) -> ProportionalDesign:
    """Derive the millimetre design from the measured head/neck scale."""
    strength = PRESET_STRENGTH[preset]
    W_neck = scale.W_neck_mm
    L_neck = scale.L_neck_mm
    head_cap = 0.07 * scale.W_head_mm if np.isfinite(scale.W_head_mm) else 10.0
    H = float(np.clip(strength * HEIGHT_RATIO * W_neck, 1.5, head_cap))
    L10 = LENGTH10_RATIO * L_neck
    # lateral size follows the coordinated ridge ratio rather than a fixed
    # fraction of a wide neck: for a short neck a wide design would become a
    # squashed mound.  L50/W50 targets 1.1-1.5; W50 ~= 0.65 W10, L50 ~= 0.686 L10.
    target_l50_w50 = 1.25
    W10_coordinated = L10 * (0.686 / 0.65) / target_l50_w50
    W10 = float(np.clip(W10_coordinated, 0.10 * W_neck, WIDTH10_RATIO * W_neck))
    ratio_deviation = None
    nominal_lo, nominal_hi = 0.26 * W_neck, 0.34 * W_neck
    if not (nominal_lo <= W10 <= nominal_hi):
        ratio_deviation = {
            "reason": "short_neck_coordination" if L10 < 0.42 * L_neck * 0.999 else "neck_width_mismatch",
            "nominal_band_mm": [round(nominal_lo, 2), round(nominal_hi, 2)],
            "used_mm": round(W10, 2),
            "l50_w50_target": target_l50_w50,
        }
    # space available above/below the centre along the neck (mm)
    space_up = scale.centre_s * L_neck - min_support_gap_mm
    space_dn = (1.0 - scale.centre_s) * L_neck - min_support_gap_mm
    up_frac = UP_FRACTION
    while up_frac > 0.2:
        l10_up = L10 * up_frac
        l10_dn = L10 * (1.0 - up_frac)
        ratio = PLATEAU_L10_RATIO if envelope == "plateau" else WENDLAND_L10_RATIO
        sup_up = l10_up / ratio
        sup_dn = l10_dn / ratio
        if sup_up <= space_up and sup_dn <= space_dn:
            break
        up_frac -= 0.05
    # shrink the whole design if the space still does not fit (never squash it)
    fit = min(1.0, space_up / sup_up, space_dn / sup_dn)
    if fit < 1.0:
        H *= fit
        W10 *= fit
        L10 *= fit
        l10_up *= fit
        l10_dn *= fit
        sup_up *= fit
        sup_dn *= fit
    width_total = W10 / 0.97
    half_width = width_total * 0.5
    # slope limits and the head-scale cap on the height
    H = float(min(H, head_cap))
    if half_width < H / MAX_SLOPE:
        width_total = 2.0 * H / MAX_SLOPE
        half_width = width_total * 0.5
    H = float(min(H, 1.0 * width_total))
    min_l = 0.75 * H / MAX_SLOPE * (l10_up + l10_dn) / max(sup_up + sup_dn, 1e-9)
    if sup_up + sup_dn < min_l:
        grow = min_l / max(sup_up + sup_dn, 1e-9)
        sup_up *= grow
        sup_dn *= grow
    tip = float(np.clip(TIP_OF_HEIGHT * H, 0.9, 0.22 * half_width))
    design = na.SculptDesign(
        height_mm=H,
        width_total_mm=width_total,
        length_up_mm=sup_up,
        length_down_mm=sup_dn,
        tip_radius_mm=tip,
        envelope=envelope,
    )
    audit = {
        "preset": preset,
        "strength": strength,
        "envelope": envelope,
        "W_head_mm": round(scale.W_head_mm, 2) if np.isfinite(scale.W_head_mm) else None,
        "W_neck_mm": round(W_neck, 2),
        "L_neck_mm": round(L_neck, 2),
        "H_mm": round(H, 3),
        "H_over_Wneck": round(H / W_neck, 4),
        "W10_target_mm": round(W10, 2),
        "W10_over_Wneck": round(W10 / W_neck, 4),
        "L10_target_mm": round(L10, 2),
        "L10_over_Lneck": round(L10 / L_neck, 4),
        "support_up_mm": round(sup_up, 2),
        "support_down_mm": round(sup_dn, 2),
        "tip_radius_mm": round(tip, 2),
        "space_up_mm": round(space_up, 2),
        "space_down_mm": round(space_dn, 2),
        "overall_scale_fit": round(fit, 3),
        "up_fraction": round(up_frac, 3),
        "slope_limit_mm_per_mm": MAX_SLOPE,
        "width_ratio_deviation": ratio_deviation,
    }
    return ProportionalDesign(design=design, audit=audit, envelope=envelope)


def _profile_curvature_radius(section_u: np.ndarray, section_n: np.ndarray) -> float | None:
    """Parabolic fit near the apex: R = 1/|2a| in mm."""
    if len(section_u) < 5:
        return None
    order = np.argsort(section_u)
    u = np.asarray(section_u)[order]
    n = np.asarray(section_n)[order]
    centre_u = u[np.argmax(n)]
    near = np.abs(u - centre_u) < max(4.0, 0.5 * (u.max() - u.min()))
    if near.sum() < 5:
        return None
    window = max(1.5, 1.3 * float(np.max(np.abs(u[near] - centre_u))))
    near = np.abs(u - centre_u) <= window
    if near.sum() < 5:
        return None
    coeffs = np.polyfit(u[near] - centre_u, n[near], 2)
    a = coeffs[0]
    if a >= -1e-6:
        return None
    return float(1.0 / (2.0 * abs(a)))


def build_proportional_states(
    *,
    states,
    references: dict[str, Any],
    dense_vertices: np.ndarray,
    dense_faces: np.ndarray,
    prepared_dir,
    landmarks_payload: dict[str, Any] | None,
    preset: str,
    sequence_rows: list[dict[str, Any]] | None = None,
    envelope: str = "plateau",
    keep_background: bool = False,
    label_state: str = "closed",
) -> dict[str, Any]:
    """Measure the scale, derive the proportional design, build both states."""
    # per-state geometry, frame (mesh/midline based), protection
    base = {}
    chin_camera = None
    head_up = head_forward = head_right = None
    for state in states:
        ref = references[state]
        vertices = np.asarray(dense_vertices[ref["stages"]["dense"]["frame_index"]], dtype=float)
        M = np.asarray(ref["animated_model_to_reference_camera_4x4"], dtype=float)
        rot, trans = M[:3, :3], M[:3, 3]
        lm_cam = np.asarray(ref["stages"]["dense"]["landmarks_m"], dtype=float)
        lm_model = (lm_cam - trans) @ rot
        axes = npn._direction_frame(lm_model)
        normals_model = npn.vertex_normals((vertices - trans) @ rot, dense_faces)
        protection = npn._protected_mask((vertices - trans) @ rot, lm_model, normals_model, axes)
        up = rot @ axes["up"]
        forward = rot @ axes["forward"]
        right = rot @ axes["right"]
        chin = (lm_model[npn.CHIN] @ rot.T + trans)
        base[state] = {
            "vertices": vertices, "protection": protection["mask"],
            "up": up, "forward": forward, "right": right, "chin": chin,
            "reference": ref, "normals": normals_model @ rot.T,
        }
        if state == label_state:
            chin_camera = chin
            head_up, head_forward, head_right = up, forward, right

    record = ((landmarks_payload or {}).get("states", {}) or {}).get(label_state) or {}
    label_point = None
    if record.get("status") == v3.nl.STATUS_OBSERVED and record.get("point_observed_m"):
        if record.get("frame_id") == base[label_state]["reference"]["frame_id"]:
            label_point = np.asarray(record["point_observed_m"], dtype=float)
        elif sequence_rows:
            tform = v3.nl.resolve_reference_transform(record, sequence_rows,
                                                      base[label_state]["reference"]["frame_id"])
            if tform.get("status") == v3.nl.STATUS_OBSERVED:
                label_point = np.asarray(v3.nl.apply_transform(record["point_observed_m"], tform), dtype=float)

    scale = measure_neck_scale(
        {s: {"vertices": base[s]["vertices"], "faces": dense_faces,
             "protection": base[s]["protection"]} for s in states},
        chin_camera=chin_camera, head_up=head_up, head_forward=head_forward,
        head_right=head_right, label_point=label_point, label_state=label_state,
    )
    prop = proportional_design(scale, preset, envelope=envelope)
    design = prop.design

    # per-state frames centred on the shared base vertex (same binding both states)
    centre_vertex = scale.centre_base_vertex
    for state in states:
        item = base[state]
        centre_cam = item["vertices"][centre_vertex]
        normals = item["normals"]
        normal = normals[centre_vertex]
        up = item["up"] - np.dot(item["up"], normal) * normal
        up /= np.linalg.norm(up)
        u_axis = np.cross(up, normal); u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis); v_axis /= np.linalg.norm(v_axis)
        item["frame"] = v3.LocalNeckFrame(
            origin=centre_cam.copy(), n=np.cross(u_axis, v_axis) / np.linalg.norm(np.cross(u_axis, v_axis)),
            up=up, u_axis=u_axis, v_axis=v_axis,
            audit={"frame_source": "shared_centre_vertex", "vertex": int(centre_vertex),
                   "position_source": "normalised_midline_binding",
                   "label_suggestion_s": scale.label_s, "label_reason": scale.label_reason},
        )
        # background correction (optional, off by default)
        item["background"] = None
        if keep_background:
            record_s = ((landmarks_payload or {}).get("states", {}) or {}).get(state) or {}
            label_pt = None
            if record_s.get("status") == v3.nl.STATUS_OBSERVED and record_s.get("point_observed_m"):
                label_pt = np.asarray(record_s["point_observed_m"], dtype=float)
            if label_pt is not None:
                from pathlib import Path as _P
                path = _P(prepared_dir) / f"{item['reference']['frame_id']}.npz"
                if path.is_file():
                    data = np.load(path, allow_pickle=False)
                    try:
                        obs = v3.collect_observations(
                            state=state, frame_id=item["reference"]["frame_id"], depth=data["depth"],
                            neck_mask=data["neck_mask"], intrinsics=data["intrinsics"],
                            label_pixel=tuple(record_s.get("pixel_uv") or [0.0, 0.0]),
                            label_point=label_pt, head_up_camera=item["up"],
                            baseline_vertices=item["vertices"], faces=dense_faces)
                        item["background"] = v3.robust_background(obs)
                    except Exception:
                        item["background"] = None

    # shared two-level refinement (union masks, per-state midpoints)
    centroid = {s: base[s]["vertices"][dense_faces].mean(axis=1) for s in states}
    mask1 = np.zeros(len(dense_faces), dtype=bool)
    for s in states:
        mask1 |= np.linalg.norm(centroid[s] - base[s]["frame"].origin, axis=1) < 0.042
    sub1 = v3.subdivide_local(base[states[0]]["vertices"], dense_faces, face_mask=mask1)
    l1 = sub1["midpoint_edges"]
    refined1 = {s: na._rebuild_refined(base[s]["vertices"], l1, []) for s in states}
    mask2 = np.zeros(len(sub1["faces"]), dtype=bool)
    for s in states:
        c2 = refined1[s][sub1["faces"]].mean(axis=1)
        mask2 |= np.linalg.norm(c2 - base[s]["frame"].origin, axis=1) < 0.022
    sub2 = v3.subdivide_local(refined1[states[0]], sub1["faces"], face_mask=mask2)
    l2 = sub2["midpoint_edges"]
    refined_faces = sub2["faces"]
    base_count = len(base[states[0]]["vertices"])

    results = {}
    for state in states:
        item = base[state]
        verts = na._rebuild_refined(item["vertices"], l1, l2)
        protected = np.zeros(len(verts), dtype=bool)
        protected[:base_count] = item["protection"]
        off1 = base_count
        for k, (i, j) in enumerate(l1):
            protected[off1 + k] = item["protection"][i] or item["protection"][j]
        off2 = off1 + len(l1)
        for k, (i, j) in enumerate(l2):
            protected[off2 + k] = protected[i] or protected[j]
        bg_values = None
        bg_reference = None
        if item["background"] is not None:
            uv_all = item["frame"].uv(verts)
            bg_values = item["background"].evaluate(uv_all[:, 0], uv_all[:, 1])
            taper_all = na._taper(verts, protected)
            window_all = v3._region_window(verts, item["frame"], protected)
            bg_reference = taper_all * window_all * bg_values
        applied = na.apply_sculpt(vertices=verts, faces=refined_faces, frame=item["frame"],
                                  design=design, protected=protected, background_mm=bg_values)
        shaped = applied["shaped_vertices"]
        moved = np.linalg.norm(shaped - verts, axis=1) > 1e-6
        guards = na.geometry_guards(rest=verts, shaped=shaped, faces=refined_faces, moved=moved)
        adjacency = npn.vertex_adjacency(refined_faces, len(shaped))
        self_int = npn._count_self_intersections(
            shaped, refined_faces, moved, adjacency, item["frame"].origin,
            (design.support_radius_mm + 20.0) * 1e-3)
        metrics = na.realized_metrics(
            shaped=shaped, base=verts, frame=item["frame"], design=design,
            background_mm=bg_values, background_reference=bg_reference)
        # section through the apex for the actual curvature radius
        frame = item["frame"]
        up_samples, n_samples = [], []
        tri = shaped[refined_faces]
        xs = (tri - frame.origin) @ frame.v_axis
        cross = (xs[:, 0] * xs[:, 1] <= 0) | (xs[:, 1] * xs[:, 2] <= 0) | (xs[:, 0] * xs[:, 2] <= 0)
        near = (np.abs(xs).max(axis=1) < 0.006) | (np.abs(xs).min(axis=1) < 1e-9)
        for fi in np.flatnonzero(near & cross):
            vf = refined_faces[fi]
            s = xs[fi]
            pts = []
            for a in range(3):
                b = (a + 1) % 3
                if s[a] * s[b] <= 0 and abs(s[a] - s[b]) > 1e-12:
                    t = s[a] / (s[a] - s[b])
                    pts.append(shaped[vf[a]] + t * (shaped[vf[b]] - shaped[vf[a]]))
            if len(pts) == 2:
                r0 = pts[0] - frame.origin
                r1 = pts[1] - frame.origin
                u0, u1 = float(r0 @ frame.u_axis) * 1000.0, float(r1 @ frame.u_axis) * 1000.0
                n0, n1 = float(r0 @ frame.n) * 1000.0, float(r1 @ frame.n) * 1000.0
                steps = max(1, int(np.ceil(abs(u1 - u0) / 0.5)))
                for step in range(steps + 1):
                    t = step / steps
                    up_samples.append(u0 + t * (u1 - u0))
                    n_samples.append(n0 + t * (n1 - n0))
        tip_radius_measured = _profile_curvature_radius(np.asarray(up_samples), np.asarray(n_samples))
        # distances and centre audit
        rel_c = frame.origin - chin_camera
        centre_s = None
        if np.isfinite(scale.L_neck_mm) and scale.L_neck_mm > 0:
            # recompute s using the shared axis (same for both states)
            v0 = None
            rel_s0 = scale.s0_point_camera - chin_camera
            rel_r = scale.root_point_camera - chin_camera
            v_c = float(rel_c @ scale.axis_up)
            v_s0 = float(rel_s0 @ scale.axis_up)
            v_root = float(rel_r @ scale.axis_up)
            centre_s = float((v_s0 - v_c) / max(v_s0 - v_root, 1e-9))
        chin_dist = float(np.min(np.linalg.norm(verts[protected] - frame.origin, axis=1)) * 1000.0) if protected.any() else None
        results[state] = {
            **item,
            "refined_vertices": verts,
            "protected_refined": protected,
            "shaped_vertices": shaped,
            "applied_mm": applied["applied_mm"],
            "sculpt_mm": applied["sculpt_mm"],
            "support": applied["support"],
            "guards": guards,
            "self_intersections": self_int,
            "metrics": metrics,
            "tip_radius_measured_mm": None if tip_radius_measured is None else round(tip_radius_measured, 2),
            "centre_s": None if centre_s is None else round(centre_s, 3),
            "midline_offset_mm": round(float(abs(frame.uv(frame.origin[None, :])[0, 0])), 3),
            "protection_distance_mm": None if chin_dist is None else round(chin_dist, 2),
            "protection_check": {"protected_vertices_moved": int((np.abs(applied["applied_mm"][protected]) > 1e-9).sum())},
            "background_available": item["background"] is not None,
        }
    accepted = all(
        na.accept_guards(results[s]["guards"]) and results[s]["self_intersections"]["count"] == 0
        and results[s]["protection_check"]["protected_vertices_moved"] == 0 for s in states)
    return {
        "states": results,
        "refined_faces": refined_faces,
        "level1_edges": l1,
        "level2_edges": l2,
        "base_vertex_count": base_count,
        "refined_vertex_count": len(results[states[0]]["refined_vertices"]),
        "design": design.as_dict(),
        "design_audit": prop.audit,
        "envelope": envelope,
        "scale": scale.as_dict(),
        "centre_binding": {"base_vertex": int(centre_vertex), "states": list(states)},
        "accepted": accepted,
    }


def _rebuild_refined(base_vertices, level1_edges, level2_edges):
    return na._rebuild_refined(base_vertices, level1_edges, level2_edges)
