"""Subject-observed neck reconstruction V3 (0.5.5 on the 0.5.2 baseline).

This stage replaces the fixed-height procedural bump with an observation-driven
local fit of the neck front:

1. local neck frame per state from the depth observations (label point, plane
   normal, observed midline direction; the head pose only initializes);
2. a low-frequency background field b(u, v) fitted on the *surrounding* neck
   skin and frozen before the prominence is measured, correcting the template
   neck offset instead of stacking a bump on a misplaced surface;
3. an individualized rounded-ridge model p(u, v): peak height, lateral width,
   tip rounding, centre offset/tilt, independent upper/lower support lengths,
   height envelope as a low-order cubic B-spline, blended C2 into the
   background;
4. a bounded residual field r(u, v) for reliable asymmetric detail with a
   strong prior, never allowed to become an independent bump;
5. observation residuals E_depth (b, p, r against the measured skin depth),
   E_contour (observed neck silhouette width), E_landmark (recorded label
   position and its stated uncertainty), E_shape (weak prior bounds),
   E_deformation (edge stretch) and E_bending (displacement Laplacian);
6. a deterministic red-green local subdivision of the front-neck patch so the
   narrowest target feature spans about six sampling intervals (the same
   refined topology for closed and open, parent-face bindings recorded);
7. independent acceptance: exact first-hit depth/coverage metrics on a fixed
   observation set, geometric safety (protected vertices, first-hit sources,
   self-intersection, flips, spikes, slender faces) and fixed-protocol
   measurements (prominence height above the frozen background, width, length,
   area, slopes, tip bluntness, peak position and uncertainty).

The stage never treats the binding depth gap as prominence height, never snaps
the centre to the template ridge, and never demands a preset final height.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import binary_erosion, median_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from head_rgbd import neck_landmark as nl
from head_rgbd import neck_prominence as npn

NECK_METHOD_V3 = "subject_observed_neck_background_ridge_v3"
GEOMETRY_SOURCE_V3 = "template_plus_observed_local_deformation"

STATUS_APPLIED = npn.STATUS_APPLIED
STATUS_SKIPPED = npn.STATUS_SKIPPED

# Observation region and fitting scales.
OBS_RADIUS_PX = 82           # observed pixel radius around the label
FIT_RADIUS_MM = 52.0         # local (u,v) radius used for fitting and windowing
OUTER_RING_IN_MM = 24.0      # background fit annulus inner radius (u,v)
OUTER_RING_OUT_MM = 40.0     # background fit annulus outer radius
WINDOW_FADE_MM = 26.0        # C2 fade band of the applied field before the region edge
TAPER_MM = 12.0              # continuous distance ramp from the nearest protected vertex
DEPTH_SIGMA_MM = 2.0         # depth residual scale for the observation term
CONTOUR_SIGMA_MM = 10.0      # observed silhouette width scale
LABEL_SIGMA_MM = 4.0         # recorded label position uncertainty (engineering)
STRETCH_SIGMA = 0.25         # edge stretch excess scale
BENDING_SIGMA_MM = 1.0       # displacement Laplacian scale
RESIDUAL_SIGMA_MM = 0.8      # residual field prior scale
RESIDUAL_BOUND_MM = 2.5      # hard bound on the residual field (signed correction)
RESIDUAL_CTRL = 4            # residual B-spline control grid (n x n)

# Subdivision requirement: the narrowest reliable target feature should span
# at least this many sampling intervals (project requirement, not anatomy).
MIN_SAMPLES_ACROSS_FEATURE = 6
SUBMIT_RADIUS_MM = 40.0      # patch radius for the local red-green refinement

# Ridge parameterization: support lengths and bounds are broad appearance
# priors, explicitly not anatomical standards.
RIDGE_CTRL = 5               # cubic B-spline coefficients for h(v)
H_BOUNDS_MM = (-8.0, 12.0)
W_BOUNDS_MM = (8.0, 26.0)
RTIP_BOUNDS_MM = (0.5, 8.0)
LUP_BOUNDS_MM = (5.0, 30.0)
LDN_BOUNDS_MM = (5.0, 30.0)
C0_BOUNDS_MM = (-10.0, 10.0)
C1_BOUNDS = (-0.6, 0.6)

GROW_LIMIT_FACTOR = npn.EDGE_GROW_FACTOR   # reused V2 limits as loose guards


# ---------------------------------------------------------------------------
# Local neck frame
# ---------------------------------------------------------------------------

@dataclass
class LocalNeckFrame:
    origin: np.ndarray            # camera-frame label point
    n: np.ndarray                 # outward normal
    up: np.ndarray                # along-neck (v) direction
    u_axis: np.ndarray            # lateral (u) direction
    v_axis: np.ndarray            # orthogonalized v
    audit: dict[str, Any] = field(default_factory=dict)

    def uv(self, points: np.ndarray) -> np.ndarray:
        rel = np.asarray(points, dtype=np.float64) - self.origin
        return np.stack([rel @ self.u_axis, rel @ self.v_axis], axis=1) * 1000.0

    def n_coord(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.origin) @ self.n * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin_m": [float(v) for v in self.origin],
            "normal": [float(v) for v in self.n],
            "up": [float(v) for v in self.v_axis],
            "u_axis": [float(v) for v in self.u_axis],
            **self.audit,
        }


def build_local_frame(
    depth: np.ndarray,
    neck_mask: np.ndarray,
    intrinsics: np.ndarray,
    label_pixel: tuple[float, float],
    label_point: np.ndarray,
    head_up_camera: np.ndarray,
) -> LocalNeckFrame:
    """Observation-driven local frame; the head pose only initializes."""
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    inner = binary_erosion(np.asarray(neck_mask, dtype=bool), iterations=3)
    u0, v0 = float(label_pixel[0]), float(label_pixel[1])
    yy, xx = np.mgrid[0:height, 0:width]

    def backproject(px, py):
        z = depth[py, px].astype(np.float64) * 1e-3
        return np.stack([(px - cx) * z / fx, -(py - cy) * z / fy, -z], axis=1)

    # outward normal from a local depth plane
    window = ((xx - u0) ** 2 + (yy - v0) ** 2) < 14 ** 2
    sel = window & inner & (depth > 0)
    pts = backproject(xx[sel], yy[sel])
    a = np.stack([pts[:, 0], pts[:, 1], np.ones(len(pts))], axis=1)
    coef, *_ = np.linalg.lstsq(a, pts[:, 2], rcond=None)
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)
    rms = float(np.sqrt((a @ coef - pts[:, 2]).prod() if False else ((a @ coef - pts[:, 2]) ** 2).mean()) * 1000.0)

    # axis initialization from the head pose, projected into the tangent plane
    up0 = np.asarray(head_up_camera, dtype=np.float64)
    up = up0 - np.dot(up0, normal) * normal
    up /= np.linalg.norm(up)

    # refinement from full contiguous mask rows around the label column
    rows = []
    widths = []
    for v in range(max(0, int(v0) - 110), min(height, int(v0) + 110)):
        us = np.where(inner[v])[0]
        if len(us) < 20:
            continue
        runs = np.split(us, np.where(np.diff(us) > 1)[0] + 1)
        run = next((r for r in runs if r.min() <= u0 <= r.max()), None)
        if run is None or len(run) < 20:
            continue
        rows.append((v, run))
        widths.append(run.max() - run.min())
    audit: dict[str, Any] = {
        "plane_rms_mm": round(rms, 3),
        "full_rows": len(rows),
        "axis_refined": False,
    }
    if len(rows) >= 8:
        med = float(np.median(widths))
        cents = []
        for (v, run), w in zip(rows, widths):
            if not (0.8 * med <= w <= 1.2 * med):
                continue
            z = np.median(depth[v, run][depth[v, run] > 0]) * 1e-3
            cents.append([(np.median(run) - cx) * z / fx, -(v - cy) * z / fy, -z])
        cents = np.asarray(cents)
        if len(cents) >= 6:
            centered = cents - cents.mean(axis=0)
            eigval, eigvec = np.linalg.eigh(centered.T @ centered)
            axis = eigvec[:, int(np.argmax(eigval))]
            if axis @ up < 0:
                axis = -axis
            angle = float(np.degrees(np.arccos(np.clip(axis @ up, -1, 1))))
            audit["centroid_axis_angle_deg"] = round(angle, 2)
            audit["centroid_rows"] = int(len(cents))
            if angle < 35.0:
                up = axis
                audit["axis_refined"] = True
    u_axis = np.cross(up, normal)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    n_axis = np.cross(u_axis, v_axis)
    n_axis /= np.linalg.norm(n_axis)
    return LocalNeckFrame(
        origin=np.asarray(label_point, dtype=np.float64).copy(),
        n=n_axis,
        up=up,
        u_axis=u_axis,
        v_axis=v_axis,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Observations and baseline depth
# ---------------------------------------------------------------------------

@dataclass
class NeckObservations:
    state: str
    frame_id: str
    intrinsics: np.ndarray
    image_shape: tuple[int, int]
    pixels: np.ndarray            # (N, 2) float
    points: np.ndarray            # (N, 3) camera
    uv_mm: np.ndarray             # (N, 2) local coordinates
    observed_depth_m: np.ndarray  # (N,)
    template_depth_m: np.ndarray  # (N,)
    ray_nz: np.ndarray            # (N,) depth response per unit normal displacement
    weights: np.ndarray           # (N,)
    frame: LocalNeckFrame
    mask_inner: np.ndarray
    depth: np.ndarray
    neck_mask: np.ndarray
    region_rows: np.ndarray       # observed full-row widths for the contour term
    subset_faces: np.ndarray = None   # faces used by the z-buffer (fit + eval)


def backproject_pixels(pixels: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    px = np.rint(pixels[:, 0]).astype(np.int64)
    py = np.rint(pixels[:, 1]).astype(np.int64)
    z = depth[py, px].astype(np.float64) * 1e-3
    return np.stack([(px - cx) * z / fx, -(py - cy) * z / fy, -z], axis=1)


def rasterize_depth(
    vertices: np.ndarray, faces: np.ndarray, intrinsics: np.ndarray,
    shape: tuple[int, int], subset: np.ndarray | None = None,
) -> np.ndarray:
    """Z-buffer depth of the template surface at every pixel (first surface)."""
    fx, fy, cx, cy = intrinsics
    height, width = shape
    zbuf = np.full((height, width), np.inf)
    face_ids = faces if subset is None else faces[subset]
    tri = vertices[face_ids]
    z = -tri[:, :, 2]
    ok = (z > 0.05).all(axis=1)
    u = cx + fx * tri[:, :, 0] / np.maximum(z, 1e-6)
    v = cy - fy * tri[:, :, 1] / np.maximum(z, 1e-6)
    for i in np.flatnonzero(ok):
        uu, vv, zz = u[i], v[i], z[i]
        umin, umax = int(np.floor(uu.min())), int(np.ceil(uu.max()))
        vmin, vmax = int(np.floor(vv.min())), int(np.ceil(vv.max()))
        if umax < 0 or umin >= width or vmax < 0 or vmin >= height:
            continue
        umin, umax = max(0, umin), min(width - 1, umax)
        vmin, vmax = max(0, vmin), min(height - 1, vmax)
        if umax - umin > 240 or vmax - vmin > 240:
            continue
        px, py = np.meshgrid(np.arange(umin, umax + 1), np.arange(vmin, vmax + 1))
        d = (uu[1] - uu[0]) * (vv[2] - vv[0]) - (uu[2] - uu[0]) * (vv[1] - vv[0])
        if abs(d) < 1e-14:
            continue
        w1 = ((px - uu[0]) * (vv[2] - vv[0]) - (py - vv[0]) * (uu[2] - uu[0])) / d
        w2 = ((uu[1] - uu[0]) * (py - vv[0]) - (vv[1] - vv[0]) * (px - uu[0])) / d
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
        if not inside.any():
            continue
        zi = w0 * zz[0] + w1 * zz[1] + w2 * zz[2]
        sub = zbuf[vmin:vmax + 1, umin:umax + 1]
        update = inside & (zi < sub)
        sub[update] = zi[update]
    return zbuf


def collect_observations(
    *,
    state: str,
    frame_id: str,
    depth: np.ndarray,
    neck_mask: np.ndarray,
    intrinsics: np.ndarray,
    label_pixel: tuple[float, float],
    label_point: np.ndarray,
    head_up_camera: np.ndarray,
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
) -> NeckObservations:
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    u0, v0 = float(label_pixel[0]), float(label_pixel[1])
    yy, xx = np.mgrid[0:height, 0:width]
    inner = binary_erosion(np.asarray(neck_mask, dtype=bool), iterations=5)
    # local depth consistency: drop pixels that differ from a 5x5 median
    # (jewellery, mask edge bleed, sensor holes)
    depth_f = depth.astype(np.float64)
    valid_depth = depth_f > 0
    median = median_filter(np.where(valid_depth, depth_f, 0.0), size=5)
    consistent = valid_depth & (np.abs(depth_f - median) < 2.5)

    frame = build_local_frame(depth, neck_mask, intrinsics, label_pixel, label_point, head_up_camera)

    # exact first-hit depth of the baseline template at every observed pixel
    # (the z-buffer aliases on 5-8 mm triangles; the fit and the evaluation
    # must share one true-first-hit definition)
    tri_c = baseline_vertices[faces].mean(axis=1)
    zc = -tri_c[:, 2]
    uc = cx + fx * tri_c[:, 0] / np.maximum(zc, 1e-6)
    vc = cy - fy * tri_c[:, 1] / np.maximum(zc, 1e-6)
    subset = np.flatnonzero(((uc - u0) ** 2 + (vc - v0) ** 2) < (OBS_RADIUS_PX + 40) ** 2)
    near = ((xx - u0) ** 2 + (yy - v0) ** 2) < OBS_RADIUS_PX ** 2
    sel = near & inner & consistent
    pixels = np.stack([xx[sel], yy[sel]], axis=1).astype(np.float64)
    points = backproject_pixels(pixels, depth, intrinsics)
    baseline_hits = first_hit_batch(baseline_vertices, faces[subset], points)
    hit = baseline_hits["hit"]
    pixels = pixels[hit]
    points = points[hit]
    uv_mm = frame.uv(points)
    observed_depth = depth_at_pixels(pixels, depth, intrinsics)
    template_depth = -baseline_hits["point_m"][hit][:, 2]
    # per-pixel depth response to a 1 mm displacement along the local surface
    # normal: moving the surface along its normal by d changes the camera-axis
    # depth of the first hit by d * |u_z| / |n . u| (u = ray direction)
    face_ids = subset[baseline_hits["triangle"][hit]]
    tri_n = np.cross(
        baseline_vertices[faces[face_ids, 1]] - baseline_vertices[faces[face_ids, 0]],
        baseline_vertices[faces[face_ids, 2]] - baseline_vertices[faces[face_ids, 0]],
    )
    tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-12)
    ray_dir = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    cos_nu = np.abs((tri_n * ray_dir).sum(axis=1))
    depth_factor = np.abs(ray_dir[:, 2]) / np.maximum(cos_nu, 0.2)
    # depth change along the camera axis for a displacement along n:
    # depth = -z so a displacement d*n changes depth by -d*n_z
    ray_nz = np.full(len(pixels), float(frame.n[2]))
    # exclude rays where the observation is implausibly far from the template
    gap_mm = (template_depth - observed_depth) * 1000.0
    plaus = np.abs(gap_mm) < 45.0
    weights = plaus.astype(np.float64)
    # observed full-row widths for the contour term (against background only)
    rows = []
    for v in range(max(0, int(v0) - 70), min(height, int(v0) + 70), 10):
        us = np.where(inner[v])[0]
        if len(us) < 20:
            continue
        runs = np.split(us, np.where(np.diff(us) > 1)[0] + 1)
        run = next((r for r in runs if r.min() <= u0 <= r.max()), None)
        if run is None or len(run) < 20:
            continue
        z = np.median(depth[v, run][depth[v, run] > 0]) * 1e-3
        widths_mm = (run.max() - run.min()) * z / fx * 1000.0
        rows.append({"v_px": int(v), "v_mm": float((frame.uv(
            np.array([[(np.median(run) - cx) * z / fx, -(v - cy) * z / fy, -z]]))[0, 1])),
            "width_mm": float(widths_mm)})
    subset_eval = np.flatnonzero(((uc - u0) ** 2 + (vc - v0) ** 2) < 150 ** 2)
    return NeckObservations(
        state=state,
        frame_id=frame_id,
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        image_shape=(height, width),
        pixels=pixels[plaus],
        points=points[plaus],
        uv_mm=uv_mm[plaus],
        observed_depth_m=observed_depth[plaus],
        template_depth_m=template_depth[plaus],
        ray_nz=depth_factor[plaus],
        weights=weights[plaus],
        frame=frame,
        mask_inner=inner,
        depth=depth,
        neck_mask=neck_mask,
        region_rows=np.asarray(rows),
        subset_faces=subset_eval,
    )


# ---------------------------------------------------------------------------
# Background field (fitted on surrounding skin, frozen)
# ---------------------------------------------------------------------------

BG_TERMS = ("1", "u", "v", "u2", "uv", "v2")
BG_LAMBDA = 1e-3             # ridge regularisation of the quadratic background


def background_design(u_mm: np.ndarray, v_mm: np.ndarray) -> np.ndarray:
    """Low-order quadratic background: it cannot form a localised bump."""
    un = np.asarray(u_mm) / 100.0
    vn = np.asarray(v_mm) / 100.0
    return np.stack([np.ones_like(un), un, vn, un * un, un * vn, vn * vn], axis=1)


@dataclass
class BackgroundFit:
    coefficients_mm: np.ndarray
    residual_stats: dict[str, Any]

    def evaluate(self, u_mm: np.ndarray, v_mm: np.ndarray) -> np.ndarray:
        return background_design(u_mm, v_mm) @ self.coefficients_mm


def fit_background(obs: NeckObservations, iterations: int = 4) -> BackgroundFit:
    """Robust quadratic background fitted on the skin *around* the larynx.

    A global quadratic cannot represent a localised prominence, so fitting it
    on the ring outside the core (|uv| >= 24 mm) and freezing it gives a
    background reference that does not absorb the larynx: the natural neck
    front convexity is carried by the quadratic, and any residual localised
    excess is what the ridge model must explain.
    """
    radius = np.linalg.norm(obs.uv_mm, axis=1)
    ring = (radius >= OUTER_RING_IN_MM) & (radius <= OUTER_RING_OUT_MM) & (obs.weights > 0)
    if ring.sum() < 300:
        ring = (radius <= OUTER_RING_OUT_MM) & (obs.weights > 0)
    design = background_design(obs.uv_mm[ring, 0], obs.uv_mm[ring, 1])
    target = (obs.template_depth_m[ring] - obs.observed_depth_m[ring]) * 1000.0 / obs.ray_nz[ring]
    lam = BG_LAMBDA * np.eye(design.shape[1])
    lam[0, 0] = 0.0
    weights = np.ones(int(ring.sum()))
    coefficients = np.zeros(design.shape[1])
    for _ in range(iterations):
        wd = design * weights[:, None]
        coefficients = np.linalg.solve(design.T @ wd + lam, wd.T @ target)
        residual = design @ coefficients - target
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weights = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
    residual = design @ coefficients - target
    all_design = background_design(obs.uv_mm[:, 0], obs.uv_mm[:, 1])
    stats = {
        "model": "quadratic_ring_background",
        "terms": list(BG_TERMS),
        "core_excluded_mm": OUTER_RING_IN_MM,
        "ring_samples": int(ring.sum()),
        "residual_median_mm": round(float(np.median(residual)), 3),
        "residual_mad_mm": round(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 3),
        "coefficients_mm": [round(float(c), 4) for c in coefficients],
        "all_sample_median_mm": round(float(np.median(all_design @ coefficients)), 3),
        "centre_mm": round(float(background_design(np.array([0.0]), np.array([0.0]))[0] @ coefficients), 3),
    }
    return BackgroundFit(coefficients_mm=coefficients, residual_stats=stats)


# ---------------------------------------------------------------------------
# Ridge model
# ---------------------------------------------------------------------------

def _smoothstep_c2(x: np.ndarray) -> np.ndarray:
    """C2 ramp: 1 -> 0 over x in [0,1] (zero slope at both ends)."""
    t = np.clip(x, 0.0, 1.0)
    return 1.0 - (6.0 * t ** 5 - 15.0 * t ** 4 + 10.0 * t ** 3)


def _soft_positive(x: np.ndarray, eps: float = 0.01) -> np.ndarray:
    return 0.5 * (x + np.sqrt(x * x + eps * eps))


def _bspline_basis(v_mm: np.ndarray, vmax: float, n_ctrl: int = RIDGE_CTRL) -> np.ndarray:
    """Clamped B-spline basis over [0, vmax] with n_ctrl coefficients."""
    vmax = max(float(vmax), 1.0)
    degree = max(1, min(3, n_ctrl - 2))
    n_interior = max(n_ctrl - degree - 1, 0)
    interior = np.linspace(0.0, vmax, n_interior + 2)[1:-1] if n_interior else np.zeros(0)
    knots = np.concatenate([
        np.repeat(0.0, degree + 1), interior, np.repeat(vmax, degree + 1),
    ])
    t = np.clip(v_mm, 0.0, vmax)
    basis = BSpline.design_matrix(t, knots, degree, extrapolate=True).toarray()
    if basis.shape[1] != n_ctrl:
        raise RuntimeError(f"bspline basis has {basis.shape[1]} coefficients, expected {n_ctrl}")
    return basis


def _bspline_basis_unit(x_mm: np.ndarray, span: float, n_ctrl: int) -> np.ndarray:
    """Symmetric basis over [-span, span] mapped onto the unit domain."""
    t = np.clip((np.asarray(x_mm) / max(span, 1e-6) + 1.0) * 0.5, 0.0, 1.0)
    return _bspline_basis(t, 1.0, n_ctrl)


@dataclass
class RidgeParameters:
    h_ctrl_mm: np.ndarray        # (5,) non-negative
    width_mm: float
    rtip_mm: float
    c0_mm: float
    c1: float
    l_up_mm: float
    l_dn_mm: float
    v_shift_mm: float
    residual_ctrl_mm: np.ndarray  # (RESIDUAL_CTRL**2,)


N_RIDGE = RIDGE_CTRL + 7 + RESIDUAL_CTRL * RESIDUAL_CTRL


def unpack_ridge(theta: np.ndarray) -> RidgeParameters:
    i = 0
    h = np.asarray(theta[i:i + RIDGE_CTRL], dtype=np.float64)
    i += RIDGE_CTRL
    width = float(theta[i]); i += 1
    rtip = float(theta[i]); i += 1
    c0 = float(theta[i]); i += 1
    c1 = float(theta[i]); i += 1
    l_up = float(theta[i]); i += 1
    l_dn = float(theta[i]); i += 1
    v_shift = float(theta[i]); i += 1
    residual = np.asarray(theta[i:i + RESIDUAL_CTRL * RESIDUAL_CTRL], dtype=np.float64)
    return RidgeParameters(h, width, rtip, c0, c1, l_up, l_dn, v_shift, residual)


def pack_ridge_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = [H_BOUNDS_MM[0]] * RIDGE_CTRL + [
        W_BOUNDS_MM[0], RTIP_BOUNDS_MM[0], C0_BOUNDS_MM[0], C1_BOUNDS[0],
        LUP_BOUNDS_MM[0], LDN_BOUNDS_MM[0], -25.0,
    ] + [-RESIDUAL_BOUND_MM] * (RESIDUAL_CTRL * RESIDUAL_CTRL)
    upper = [H_BOUNDS_MM[1]] * RIDGE_CTRL + [
        W_BOUNDS_MM[1], RTIP_BOUNDS_MM[1], C0_BOUNDS_MM[1], C1_BOUNDS[1],
        LUP_BOUNDS_MM[1], LDN_BOUNDS_MM[1], 25.0,
    ] + [RESIDUAL_BOUND_MM] * (RESIDUAL_CTRL * RESIDUAL_CTRL)
    return np.asarray(lower), np.asarray(upper)


def initial_ridge_theta(h0_mm: float = 3.0) -> np.ndarray:
    theta = np.zeros(N_RIDGE)
    theta[:RIDGE_CTRL] = h0_mm
    theta[RIDGE_CTRL] = 14.0        # width
    theta[RIDGE_CTRL + 1] = 2.0     # rtip
    theta[RIDGE_CTRL + 2] = 0.0     # c0
    theta[RIDGE_CTRL + 3] = 0.0     # c1
    theta[RIDGE_CTRL + 4] = 14.0    # l_up
    theta[RIDGE_CTRL + 5] = 12.0    # l_dn
    theta[RIDGE_CTRL + 6] = 0.0     # v_shift
    return theta


def _residual_basis(u_mm: np.ndarray, v_mm: np.ndarray) -> np.ndarray:
    span = FIT_RADIUS_MM * 0.9
    bu = _bspline_basis_unit(u_mm, span, RESIDUAL_CTRL)
    bv = _bspline_basis_unit(v_mm, span, RESIDUAL_CTRL)
    return (bu[:, :, None] * bv[:, None, :]).reshape(len(u_mm), -1)


def ridge_eval(theta: np.ndarray, u_mm: np.ndarray, v_mm: np.ndarray) -> dict[str, np.ndarray]:
    p = unpack_ridge(theta)
    h = p.h_ctrl_mm
    span_up = p.l_up_mm + p.v_shift_mm
    span_dn = p.l_dn_mm - p.v_shift_mm
    vmax_up = max(p.l_up_mm, 1.0)
    vmax_dn = max(p.l_dn_mm, 1.0)
    # height envelope: one clamped cubic B-spline per side, joined at v = v_shift
    vel = v_mm - p.v_shift_mm
    basis_up = _bspline_basis(np.clip(vel, 0.0, vmax_up), vmax_up)
    basis_dn = _bspline_basis(np.clip(-vel, 0.0, vmax_dn), vmax_dn)
    envelope = np.where(vel >= 0, basis_up @ h, basis_dn @ h)
    # independent upper/lower C2 windows
    win_up = _smoothstep_c2((vel - 0.55 * p.l_up_mm) / max(0.45 * p.l_up_mm, 1e-6))
    win_dn = _smoothstep_c2((-vel - 0.55 * p.l_dn_mm) / max(0.45 * p.l_dn_mm, 1e-6))
    win_v = np.where(vel >= 0, np.maximum(win_up, 0.0), np.maximum(win_dn, 0.0))
    win_v = np.where(np.abs(vel) < 5.0, 1.0, win_v)
    c = p.c0_mm + p.c1 * v_mm
    width = np.maximum(p.width_mm, 1.0)
    rtip = np.maximum(p.rtip_mm, 0.05)
    k = envelope / np.maximum(np.sqrt(width ** 2 + rtip ** 2) - rtip, 1e-3)
    du = u_mm - c
    bracket = envelope - k * (np.sqrt(du * du + rtip * rtip) - rtip)
    lateral = _smoothstep_c2((np.abs(du) - 0.60 * width) / max(0.40 * width, 1e-6))
    # signed ridge: positive adds a rounded laryngeal ridge, negative removes a
    # wrong template bulge; the bracket reaches zero exactly at |u-c| = W
    p_field = win_v * lateral * bracket
    r_field = _residual_basis(u_mm, v_mm) @ p.residual_ctrl_mm
    r_field = r_field * lateral * win_v
    return {"background_like": np.zeros_like(p_field), "prominence_mm": p_field,
            "residual_mm": r_field, "envelope_mm": envelope, "centre_mm": c,
            "slope_mm_per_mm": k, "window_v": win_v, "lateral_window": lateral}


# ---------------------------------------------------------------------------
# Mesh-side evaluation terms
# ---------------------------------------------------------------------------

def _field_at_vertices(
    theta: np.ndarray, vertices: np.ndarray, frame: LocalNeckFrame, background: BackgroundFit,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    uv = frame.uv(vertices)
    u, v = uv[:, 0], uv[:, 1]
    parts = ridge_eval(theta, u, v)
    total = background.evaluate(u, v) + parts["prominence_mm"] + parts["residual_mm"]
    return total, parts


def _vertex_adjacency(faces: np.ndarray, n: int) -> list[np.ndarray]:
    return npn.vertex_adjacency(faces, n)


def _distance_taper(vertices: np.ndarray, protected: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Continuous 0->1 ramp over TAPER_MM distance from the nearest protected vertex."""
    ids = np.flatnonzero(protected)
    if not len(ids):
        return np.ones(len(vertices)), {"taper_mm": TAPER_MM, "nearest_protected_mm": None}
    tree = cKDTree(vertices[ids])
    distance, _ = tree.query(vertices, k=1)
    taper = np.clip(distance / (TAPER_MM * 1e-3), 0.0, 1.0)
    return taper, {"taper_mm": TAPER_MM, "nearest_protected_mm": round(float(distance.min() * 1000.0), 3)}


def _region_window(vertices: np.ndarray, frame: LocalNeckFrame, protected: np.ndarray) -> np.ndarray:
    uv = frame.uv(vertices)
    radius = np.linalg.norm(uv, axis=1)
    window = _smoothstep_c2((radius - (FIT_RADIUS_MM - WINDOW_FADE_MM)) / WINDOW_FADE_MM)
    window = np.where(radius < FIT_RADIUS_MM - WINDOW_FADE_MM, 1.0, window)
    window = np.where(radius >= FIT_RADIUS_MM, 0.0, window)
    window = np.where(protected, 0.0, window)
    return window


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

@dataclass
class StateFitResult:
    theta: np.ndarray
    background: BackgroundFit
    diagnostics: dict[str, Any]
    residuals_history: list[dict[str, Any]]
    covariance: np.ndarray | None
    joint_theta: np.ndarray | None = None
    residual_coefficients: np.ndarray | None = None


BG_DIM = 6
N_TOTAL = BG_DIM + N_RIDGE


def _split_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(theta[:BG_DIM]), np.asarray(theta[BG_DIM:])


def ridge_profile(
    u_mm: np.ndarray, v_mm: np.ndarray,
    height_mm: float, c0_mm: float, c1: float, width_mm: float, rtip_mm: float,
    v0_mm: float, l_up_mm: float, l_dn_mm: float,
) -> np.ndarray:
    """Signed rounded-ridge profile in the local neck frame.

    z(u,v) = h(v) - k(v)[sqrt((u-c(v))^2 + r_tip^2) - r_tip], with h = height,
    k = height / (sqrt(W^2 + r_tip^2) - r_tip) so the bracket reaches zero
    exactly at |u - c| = W, c(v) = c0 + c1*(v - v0), and independent C2
    smoothstep vertical transitions of length l_up / l_dn.  A positive height
    adds a laryngeal ridge, a negative height removes a wrong template bulge.
    """
    u_mm = np.asarray(u_mm, dtype=np.float64)
    v_mm = np.asarray(v_mm, dtype=np.float64)
    width = max(float(width_mm), 1.0)
    rtip = max(float(rtip_mm), 0.05)
    centre = c0_mm + c1 * (v_mm - v0_mm)
    k = height_mm / max(np.sqrt(width ** 2 + rtip ** 2) - rtip, 1e-3)
    du = u_mm - centre
    bracket = height_mm - k * (np.sqrt(du * du + rtip * rtip) - rtip)
    lateral = _smoothstep_c2((np.abs(du) - 0.60 * width) / max(0.40 * width, 1e-6))
    dv = v_mm - v0_mm
    l_up = max(float(l_up_mm), 1.0)
    l_dn = max(float(l_dn_mm), 1.0)
    win_up = _smoothstep_c2((dv - 0.55 * l_up) / max(0.45 * l_up, 1e-6))
    win_dn = _smoothstep_c2((-dv - 0.55 * l_dn) / max(0.45 * l_dn, 1e-6))
    win_v = np.where(dv >= 0, win_up, win_dn)
    win_v = np.where(np.abs(dv) <= max(0.55 * l_up, 0.55 * l_dn), 1.0, win_v)
    return win_v * lateral * bracket


RIDGE8_NAMES = ("height_mm", "c0_mm", "c1", "width_mm", "rtip_mm", "v0_mm", "l_up_mm", "l_dn_mm")
RIDGE8_BOUNDS = np.array([
    [-3.0, 3.0],         # height (conservative: the depth evidence is weak)
    [-15.0, 15.0],       # c0
    [-0.5, 0.5],         # c1
    [8.0, 30.0],         # width
    [0.5, 8.0],          # rtip
    [-12.0, 12.0],       # v0
    [6.0, 35.0],         # l_up
    [6.0, 35.0],         # l_dn
])
RESIDUAL_GRID = 3            # residual control grid (n x n) over the core


def robust_background(obs: NeckObservations, radius_max_mm: float = 40.0) -> BackgroundFit:
    """Robust quadratic background fitted on all reliable observations.

    Only six coefficients, so the background cannot represent a localised
    prominence; it carries the template-vs-subject offset and the neck-front
    curvature.  The fitted background is frozen and reused by the measurement.
    """
    sel = (np.linalg.norm(obs.uv_mm, axis=1) <= radius_max_mm) & (obs.weights > 0)
    design = background_design(obs.uv_mm[sel, 0], obs.uv_mm[sel, 1])
    target = (obs.template_depth_m[sel] - obs.observed_depth_m[sel]) * 1000.0 / obs.ray_nz[sel]
    lam = 1e-3 * np.eye(design.shape[1])
    lam[0, 0] = 0.0
    weights = np.ones(int(sel.sum()))
    coefficients = np.zeros(design.shape[1])
    for _ in range(5):
        wd = design * weights[:, None]
        coefficients = np.linalg.solve(design.T @ wd + lam, wd.T @ target)
        residual = design @ coefficients - target
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weights = 1.0 / (1.0 + (residual / (2.5 * scale)) ** 2)
    residual = design @ coefficients - target
    stats = {
        "model": "quadratic_background_all_observations",
        "terms": list(BG_TERMS),
        "radius_max_mm": radius_max_mm,
        "samples": int(sel.sum()),
        "residual_median_mm": round(float(np.median(residual)), 3),
        "residual_mad_mm": round(float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 3),
        "coefficients_mm": [round(float(c), 4) for c in coefficients],
        "centre_mm": round(float(background_design(np.array([0.0]), np.array([0.0]))[0] @ coefficients), 3),
    }
    return BackgroundFit(coefficients_mm=coefficients, residual_stats=stats)


def bin_residuals(obs: NeckObservations, background: BackgroundFit, step_mm: float = 2.5,
                  radius_mm: float = 32.0, min_count: int = 8):
    """Median displacement excess above the background on a fixed (u,v) grid."""
    excess = (obs.template_depth_m - obs.observed_depth_m) * 1000.0 / obs.ray_nz \
        - background.evaluate(obs.uv_mm[:, 0], obs.uv_mm[:, 1])
    u = obs.uv_mm[:, 0]
    v = obs.uv_mm[:, 1]
    keep = np.linalg.norm(obs.uv_mm, axis=1) <= radius_mm
    u = u[keep]
    v = v[keep]
    excess = excess[keep]
    ug = np.arange(-radius_mm, radius_mm + 1e-6, step_mm)
    vg = np.arange(-radius_mm, radius_mm + 1e-6, step_mm)
    cells = []
    for uc in ug:
        for vc in vg:
            sel = (np.abs(u - uc) < step_mm * 0.5) & (np.abs(v - vc) < step_mm * 0.5)
            if sel.sum() >= min_count:
                cells.append((uc, vc, float(np.median(excess[sel])), int(sel.sum())))
    cells = np.asarray(cells)
    return cells


def fit_state(
    obs: NeckObservations,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    frame: LocalNeckFrame,
    protected: np.ndarray,
    vertex_normals: np.ndarray,
    region_radius_mm: float = FIT_RADIUS_MM,
    max_nfev: int = 80,
) -> StateFitResult:
    """Robust observation fit: frozen quadratic background + 8-parameter ridge.

    The binned median excess above the background is the observation vector;
    the ridge (signed height, lateral width, tip rounding, centre offset/tilt,
    peak height position, independent up/down transition lengths) is fitted to
    it with bounded least squares and soft appearance priors.  A small smooth
    residual field absorbs reliable asymmetric detail (bounded, priored).
    One rendered first-surface pass then corrects the fitted height for the
    linearisation gap between the depth-domain model and the actual mesh.
    """
    background = robust_background(obs)
    cells = bin_residuals(obs, background)
    if len(cells) < 30:
        raise RuntimeError("insufficient binned observations for the ridge fit")

    # robust weights on the binned medians (reject isolated bins)
    cell_u = cells[:, 0]
    cell_v = cells[:, 1]
    cell_r = cells[:, 2]
    med = np.median(cell_r)
    mad = 1.4826 * np.median(np.abs(cell_r - med)) + 1e-6
    cell_w = 1.0 / (1.0 + (np.abs(cell_r - med) / (3.0 * mad)) ** 2)

    history: list[dict[str, Any]] = []
    prior_scale = np.array([2.0, 4.0, 0.15, 6.0, 2.5, 4.0, 8.0, 8.0])
    prior_centre = np.array([0.0, 0.0, 0.0, 15.0, 2.5, 0.0, 15.0, 15.0])
    prior_weight = np.array([1.0, 0.8, 0.8, 0.7, 0.6, 0.8, 0.6, 0.6])

    def ridge_fun(theta: np.ndarray) -> np.ndarray:
        model = ridge_profile(cell_u, cell_v, *theta)
        data = (model - cell_r) / DEPTH_SIGMA_MM * cell_w * 2.0
        priors = (theta - prior_centre) / prior_scale * prior_weight
        return np.concatenate([data, priors])

    lo = RIDGE8_BOUNDS[:, 0]
    hi = RIDGE8_BOUNDS[:, 1]
    theta0 = np.array([1.0, 0.0, 0.0, 15.0, 2.5, 0.0, 15.0, 15.0])
    theta0 = np.clip(theta0, lo + 1e-6, hi - 1e-6)
    counter = {"n": 0}

    def tracked(theta: np.ndarray) -> np.ndarray:
        out = ridge_fun(theta)
        counter["n"] += 1
        if counter["n"] <= 160:
            history.append({
                "iteration": counter["n"],
                "ridge_height_mm": round(float(theta[0]), 3),
                "data_rms": round(float(np.sqrt(np.mean((ridge_profile(cell_u, cell_v, *theta) - cell_r) ** 2))), 3),
            })
        return out

    result = least_squares(tracked, theta0, bounds=(lo, hi), x_scale="jac",
                           loss="soft_l1", f_scale=0.75, max_nfev=max_nfev, verbose=0)
    ridge_theta = result.x

    # residual field on the grid, linear with a strong prior
    grid_span = 34.0
    def grid_basis(values: np.ndarray) -> np.ndarray:
        return _bspline_basis_unit(values, grid_span, RESIDUAL_GRID)

    bu = grid_basis(cell_u)
    bv = grid_basis(cell_v)
    design = (bu[:, :, None] * bv[:, None, :]).reshape(len(cell_u), -1)
    leftover = cell_r - ridge_profile(cell_u, cell_v, *ridge_theta)
    lam_r = (DEPTH_SIGMA_MM / RESIDUAL_SIGMA_MM) ** 2 * np.eye(design.shape[1]) / max(len(cells), 1) * 20.0
    wd = design * cell_w[:, None]
    resid_coeffs = np.linalg.solve(design.T @ wd + lam_r, wd.T @ (leftover * cell_w))
    resid_coeffs = np.clip(resid_coeffs, -RESIDUAL_BOUND_MM, RESIDUAL_BOUND_MM)

    def full_field(u_mm: np.ndarray, v_mm: np.ndarray) -> np.ndarray:
        r = (_bspline_basis_unit(np.asarray(u_mm), grid_span, RESIDUAL_GRID)[:, :, None]
             * _bspline_basis_unit(np.asarray(v_mm), grid_span, RESIDUAL_GRID)[:, None, :]
             ).reshape(len(np.asarray(u_mm)), -1) @ resid_coeffs
        return ridge_profile(u_mm, v_mm, *ridge_theta) + r

    # one rendered first-surface correction of the fitted height
    render_history: list[dict[str, Any]] = []
    core = np.linalg.norm(obs.uv_mm, axis=1) <= 22.0
    # conservative mode: the ridge height is bounded to +-3 mm and is not
    # re-scaled after fitting (the render diagnostics are still reported).
    for outer in range(0):
        shaped = apply_field_arrays(vertices, frame, background, ridge_theta, resid_coeffs,
                                    protected, vertex_normals)
        hits = first_hit_batch(shaped, faces[obs.subset_faces] if obs.subset_faces is not None else faces,
                               obs.points)
        rendered = np.where(hits["hit"], -hits["point_m"][:, 2], np.nan)
        ok = np.isfinite(rendered) & core
        if ok.sum() < 100:
            break
        err_mm = (obs.observed_depth_m[ok] - rendered[ok]) * 1000.0
        # positive error = rendered surface in front of the skin -> reduce the ridge
        correction = -float(np.median(err_mm)) / float(np.median(obs.ray_nz[ok]))
        ridge_theta = ridge_theta.copy()
        ridge_theta[0] = float(np.clip(ridge_theta[0] + correction, lo[0], hi[0]))
        shaped_after = apply_field_arrays(vertices, frame, background, ridge_theta, resid_coeffs,
                                          protected, vertex_normals)
        hits2 = first_hit_batch(shaped_after, faces[obs.subset_faces] if obs.subset_faces is not None else faces,
                                obs.points)
        rendered2 = np.where(hits2["hit"], -hits2["point_m"][:, 2], np.nan)
        ok2 = np.isfinite(rendered2) & core
        err2 = (obs.observed_depth_m[ok2] - rendered2[ok2]) * 1000.0
        render_history.append({
            "outer": outer,
            "height_correction_mm": round(correction, 3),
            "depth_median_before_mm": round(float(np.median(err_mm)), 3),
            "depth_median_after_mm": round(float(np.median(err2)), 3),
            "depth_mae_after_mm": round(float(np.mean(np.abs(err2))), 3),
            "covered": int(ok2.sum()),
        })

    # covariance of the ridge parameters from the binned fit
    sigma2 = float(np.mean(np.asarray(ridge_fun(result.x)) ** 2))
    try:
        jac = result.jac
        covariance = np.linalg.pinv(jac.T @ jac) * sigma2
    except Exception:
        covariance = None
    binned_rms = float(np.sqrt(np.mean((full_field(cell_u, cell_v) - cell_r) ** 2)))
    diagnostics = {
        "method": "robust_binned_ridge",
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "depth_rms_mm": round(binned_rms, 3),
        "binned_depth_median_mm": round(float(np.median(cell_r - full_field(cell_u, cell_v))), 3),
        "bins": int(len(cells)),
        "central_observations": int(core.sum()),
        "background": background.residual_stats,
        "render_refit": render_history,
        "sigma2": round(sigma2, 5),
        "parameters": {name: round(float(value), 3) for name, value in zip(RIDGE8_NAMES, ridge_theta)},
        "residual_coefficients_mm": [round(float(c), 3) for c in resid_coeffs],
        "residual_max_mm": round(float(np.max(np.abs(resid_coeffs))), 3),
    }
    return StateFitResult(theta=ridge_theta, background=background, diagnostics=diagnostics,
                          residuals_history=history, covariance=covariance, joint_theta=None,
                          residual_coefficients=resid_coeffs)


def apply_field_arrays(
    vertices: np.ndarray, frame: LocalNeckFrame, background: BackgroundFit,
    ridge_theta: np.ndarray, residual_coefficients: np.ndarray,
    protected: np.ndarray, vertex_normals: np.ndarray,
) -> np.ndarray:
    """Displace vertices by background + ridge + residual along their normals."""
    uv = frame.uv(vertices)
    bg = background.evaluate(uv[:, 0], uv[:, 1])
    prof = ridge_profile(uv[:, 0], uv[:, 1], *ridge_theta)
    resid = (_bspline_basis_unit(uv[:, 0], 34.0, RESIDUAL_GRID)[:, :, None]
             * _bspline_basis_unit(uv[:, 1], 34.0, RESIDUAL_GRID)[:, None, :]
             ).reshape(len(vertices), -1) @ residual_coefficients
    window = _region_window(vertices, frame, protected)
    taper, _ = _distance_taper(vertices, protected)
    total = window * taper * (bg + prof + resid)
    return vertices + vertex_normals * (total * 1e-3)[:, None]



def render_depth_at_observations_joint(
    theta_joint: np.ndarray, obs: NeckObservations,
    vertices: np.ndarray, faces: np.ndarray, frame: LocalNeckFrame,
    protected: np.ndarray, vertex_normals: np.ndarray,
) -> np.ndarray:
    btheta, rtheta = _split_theta(theta_joint)
    background = BackgroundFit(coefficients_mm=btheta, residual_stats={})
    applied = apply_field(vertices=vertices, faces=faces, frame=frame,
                          background=background, theta=rtheta, protected=protected,
                          vertex_normals=vertex_normals)
    subset = obs.subset_faces if obs.subset_faces is not None else None
    faces_eval = faces[subset] if subset is not None else faces
    hits = first_hit_batch(applied["shaped_vertices"], faces_eval, obs.points)
    return np.where(hits["hit"], -hits["point_m"][:, 2], np.nan)



def render_depth_at_observations(
    theta: np.ndarray, background: BackgroundFit, obs: NeckObservations,
    vertices: np.ndarray, faces: np.ndarray, frame: LocalNeckFrame,
    protected: np.ndarray, vertex_normals: np.ndarray,
) -> np.ndarray:
    """Real first-surface depth of the shaped mesh at the observed pixels."""
    applied = apply_field(vertices=vertices, faces=faces, frame=frame,
                          background=background, theta=theta, protected=protected,
                          vertex_normals=vertex_normals)
    subset = obs.subset_faces if obs.subset_faces is not None else None
    faces_eval = faces[subset] if subset is not None else faces
    hits = first_hit_batch(applied["shaped_vertices"], faces_eval, obs.points)
    depth = np.where(hits["hit"], -hits["point_m"][:, 2], np.nan)
    return depth


# ---------------------------------------------------------------------------
# Applying the fitted field to the mesh
# ---------------------------------------------------------------------------

def smooth_displacement(disp: np.ndarray, faces: np.ndarray, protected: np.ndarray,
                        iterations: int = 2) -> np.ndarray:
    """Laplacian smoothing of the displacement field (keeps protected at zero).

    The template neck has a few sharp creases (bottom cut, submental fold).  A
    displacement along per-vertex normals folds tiny triangles at those
    creases; smoothing the *field* (not clipping per vertex) removes the fold
    without changing the target height noticeably.
    """
    n = len(disp)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    out = disp.copy()
    for _ in range(iterations):
        acc = np.zeros_like(out)
        count = np.zeros(n)
        np.add.at(acc, edges[:, 0], out[edges[:, 1]])
        np.add.at(acc, edges[:, 1], out[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1.0)
        np.add.at(count, edges[:, 1], 1.0)
        avg = acc / np.maximum(count, 1.0)[:, None]
        out = 0.5 * out + 0.5 * avg
        out[protected] = 0.0
    return out


def apply_field(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    frame: LocalNeckFrame,
    background: BackgroundFit,
    theta: np.ndarray,
    protected: np.ndarray,
    vertex_normals: np.ndarray,
    residual_coefficients: np.ndarray | None = None,
) -> dict[str, Any]:
    uv = frame.uv(vertices)
    bg = background.evaluate(uv[:, 0], uv[:, 1])
    prof = ridge_profile(uv[:, 0], uv[:, 1], *theta)
    if residual_coefficients is None:
        resid = np.zeros(len(vertices))
    else:
        resid = (_bspline_basis_unit(uv[:, 0], 34.0, RESIDUAL_GRID)[:, :, None]
                 * _bspline_basis_unit(uv[:, 1], 34.0, RESIDUAL_GRID)[:, None, :]
                 ).reshape(len(vertices), -1) @ residual_coefficients
    window = _region_window(vertices, frame, protected)
    taper, _ = _distance_taper(vertices, protected)
    total = window * taper * (bg + prof + resid)
    disp = vertex_normals * (total * 1e-3)[:, None]
    disp = smooth_displacement(disp, faces, protected, iterations=2)
    total = np.sign(total) * np.linalg.norm(disp, axis=1) * 1e3
    shaped = vertices + disp
    return {
        "shaped_vertices": shaped,
        "applied_mm": total,
        "window": window,
        "prominence_mm": prof,
        "residual_mm": resid,
        "background_mm": bg,
    }


# ---------------------------------------------------------------------------
# Deterministic local red-green subdivision
# ---------------------------------------------------------------------------

def subdivide_local(
    vertices: np.ndarray, faces: np.ndarray, centre: np.ndarray = None,
    radius_mm: float = SUBMIT_RADIUS_MM, face_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Red-green refinement of the front-neck patch, deterministic and shared.

    Red faces (centroid within radius) split 1->4; green faces with 1 or 2
    split edges split 1->2 / 1->3.  Edge midpoints are created deterministically
    from the sorted (i, j) endpoints, so the same construction can be replayed
    for any pose of the shared topology.
    """
    if face_mask is not None:
        red = np.asarray(face_mask, dtype=bool)
    else:
        centroid = vertices[faces].mean(axis=1)
        dist = np.linalg.norm(centroid - centre, axis=1)
        red = dist < radius_mm * 1e-3
    edge_key = {}
    for f in faces[red]:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edge_key[key] = True
    split_edges = sorted(edge_key.keys())
    new_vertices = list(np.asarray(vertices, dtype=np.float64))
    midpoint_index: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in midpoint_index:
            return midpoint_index[key]
        index = len(new_vertices)
        new_vertices.append(0.5 * (vertices[i] + vertices[j]))
        midpoint_index[key] = index
        return index

    new_faces: list[list[int]] = []
    parent_faces: list[int] = []
    split_set = set(split_edges)
    for fid, f in enumerate(faces):
        keys = [(min(int(f[a]), int(f[b])), max(int(f[a]), int(f[b])))
                for a, b in ((0, 1), (1, 2), (2, 0))]
        n_split = sum(1 for k in keys if k in split_set)
        if n_split == 3:
            a, b, c = (int(v) for v in f)
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            for tri in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)):
                new_faces.append(list(tri))
                parent_faces.append(fid)
        elif n_split == 1:
            axis = next(i for i, k in enumerate(keys) if k in split_set)
            a, b, c = (int(v) for v in f)
            if axis == 0:
                m = midpoint(a, b)
                new_faces.extend([[a, m, c], [m, b, c]])
            elif axis == 1:
                m = midpoint(b, c)
                new_faces.extend([[a, b, m], [a, m, c]])
            else:
                m = midpoint(c, a)
                new_faces.extend([[a, b, m], [m, b, c]])
            parent_faces.extend([fid, fid])
        elif n_split == 2:
            a, b, c = (int(v) for v in f)
            if keys[0] in split_set and keys[1] in split_set:
                m1, m2 = midpoint(a, b), midpoint(b, c)
                new_faces.extend([[a, m1, m2], [a, m2, c], [m1, b, m2]])
            elif keys[1] in split_set and keys[2] in split_set:
                m1, m2 = midpoint(b, c), midpoint(c, a)
                new_faces.extend([[a, b, m1], [a, m1, m2], [m2, m1, c]])
            else:
                m1, m2 = midpoint(c, a), midpoint(a, b)
                new_faces.extend([[a, m2, m1], [m2, b, c], [m1, m2, c]])
            parent_faces.extend([fid, fid, fid])
        else:
            new_faces.append([int(v) for v in f])
            parent_faces.append(fid)
    new_vertices = np.asarray(new_vertices, dtype=np.float64)
    new_faces = np.asarray(new_faces, dtype=np.int64)
    parent_faces = np.asarray(parent_faces, dtype=np.int64)
    ordered_edges = [None] * len(midpoint_index)
    for key, index in midpoint_index.items():
        ordered_edges[index - len(vertices)] = key
    return {
        "vertices": new_vertices,
        "faces": new_faces,
        "parent_faces": parent_faces,
        "midpoint_map": midpoint_index,
        "midpoint_edges": ordered_edges,
        "red_faces": int(red.sum()),
        "split_edges": int(len(split_edges)),
        "base_vertices": int(len(vertices)),
        "base_faces": int(len(faces)),
        "refined_vertices": int(len(new_vertices)),
        "refined_faces": int(len(new_faces)),
    }


# ---------------------------------------------------------------------------
# Independent acceptance: exact first-hit metrics on a fixed observation set
# ---------------------------------------------------------------------------

def first_hit_batch(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray,
                    chunk: int = 256) -> dict[str, np.ndarray]:
    """Batched camera-origin first hits for many rays (vectorised over faces)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    tri0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - tri0
    e2 = vertices[faces[:, 2]] - tri0
    normal = np.cross(e1, e2)
    n_len = np.linalg.norm(normal, axis=1)
    n_dir = normal / np.maximum(n_len, 1e-12)[:, None]
    hits = np.zeros(len(points), dtype=bool)
    depth = np.full(len(points), np.nan)
    tri = np.full(len(points), -1, dtype=np.int64)
    point = np.full((len(points), 3), np.nan)
    for start in range(0, len(points), chunk):
        block = np.asarray(points[start:start + chunk], dtype=np.float64)
        d = block[:, None, :]                       # (R,1,3)
        pvec = np.cross(d, e2[None, :, :])          # (R,F,3)
        det = (e1[None, :, :] * pvec).sum(axis=2)   # (R,F)
        usable = np.abs(det) > 1e-14
        inv = np.zeros_like(det)
        inv[usable] = 1.0 / det[usable]
        tvec = -tri0[None, :, :]
        u = (tvec * pvec).sum(axis=2) * inv
        qvec = np.cross(tvec, e1[None, :, :])
        v = (d * qvec).sum(axis=2) * inv
        t = (e2[None, :, :] * qvec).sum(axis=2) * inv
        ray_len = np.linalg.norm(block, axis=1)
        incidence = (n_dir[None, :, :] * (-d / ray_len[:, None, None])).sum(axis=2)
        keep = usable & (t > 1e-9) & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (incidence > 0.15)
        t_masked = np.where(keep, t, np.inf)
        best = np.argmin(t_masked, axis=1)
        best_t = t_masked[np.arange(len(block)), best]
        found = np.isfinite(best_t)
        ids = np.arange(len(block))[found]
        hits[start:start + len(block)][ids] = True
        depth[start:start + len(block)][ids] = best_t[ids] * ray_len[found]
        tri[start:start + len(block)][ids] = best[ids]
        point[start:start + len(block)][ids] = block[ids] * best_t[ids, None]
    return {"hit": hits, "depth_m": depth, "triangle": tri, "point_m": point}


def first_hit_metrics(
    vertices: np.ndarray, faces: np.ndarray, obs: NeckObservations,
    eval_pixels: np.ndarray | None = None, subset_radius_mm: float = 120.0,
) -> dict[str, Any]:
    """Fixed-set first-hit depth error: no-hit penalty 30 mm, no truncation.

    The depth compared is the camera-axis depth of the first hit (not the
    along-ray distance).  Only faces with a vertex within subset_radius_mm of
    the label are considered: rays inside the observed neck region cannot pass
    through geometry farther away without hitting these faces first (the
    region is a frontal patch), and this keeps the batched ray casting fast.
    """
    if eval_pixels is None:
        eval_pixels = obs.pixels
    points = backproject_pixels(eval_pixels, obs.depth, obs.intrinsics)
    observed = depth_at_pixels(eval_pixels, obs.depth, obs.intrinsics)
    if subset_radius_mm:
        near_vertex = np.linalg.norm(vertices - obs.frame.origin, axis=1) < subset_radius_mm * 1e-3
        subset = np.flatnonzero(near_vertex[faces].any(axis=1))
        faces_eval = faces[subset]
    else:
        faces_eval = faces
    result = first_hit_batch(vertices, faces_eval, points)
    hit = result["hit"]
    rendered = -result["point_m"][:, 2]
    error_mm = np.where(hit, (observed - rendered) * 1000.0, 30.0)
    return {
        "samples": int(len(points)),
        "coverage": round(float(hit.mean()), 4),
        "mae_mm": round(float(np.mean(np.abs(error_mm))), 3),
        "rmse_mm": round(float(np.sqrt(np.mean(error_mm ** 2))), 3),
        "p95_abs_mm": round(float(np.percentile(np.abs(error_mm), 95)), 3),
        "median_signed_mm": round(float(np.median(error_mm)), 3),
        "max_abs_mm": round(float(np.max(np.abs(error_mm))), 3),
        "hit_mae_mm": round(float(np.mean(np.abs(error_mm[hit]))) if hit.any() else 30.0, 3),
        "no_hit_penalty_mm": 30.0,
        "error_mm": error_mm,
    }


def depth_at_pixels(pixels: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    px = np.rint(np.asarray(pixels)[:, 0]).astype(np.int64)
    py = np.rint(np.asarray(pixels)[:, 1]).astype(np.int64)
    return depth[py, px].astype(np.float64) * 1e-3
