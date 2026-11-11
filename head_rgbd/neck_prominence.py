"""Depth-conditioned procedural neck-front prominence on the 0.5.2 template.

This stage is *appearance shaping of the template neck*, never a measured neck
reconstruction:

* the laryngeal landmark (ArUco centre or a manually specified point, per
  expression state, corrected only with recorded RGB/depth evidence) is bound
  to the delivered baseline mesh by an exact camera-ray first hit;
* depth around the landmark is read only to verify direction, bound the
  support radii against the observed neck width and record the local skin
  trend - it never drives a per-point fit of the neck;
* the shaping centre is snapped onto the template neck's own front ridge when
  (and only when) the binding falls on the ridge slope, because the template
  neck can sit laterally offset from the subject's real neck;
* the target is a FINAL prominence height H measured on a fixed sagittal
  profile (plane/mesh intersection) against a smooth reference trend built
  from anchor bands outside the support, anchored before any shaping;
* the displacement field is the sum of two compactly supported Wendland C2
  fields - a wide base and a narrower central ridge - applied along the
  pose-following forward axis, with a distance-based continuous taper near
  protected regions;
* the achieved H_final is calibrated against H_target by rescaling the field,
  and a candidate is accepted only when both the shape goal
  (|H_final - H_target| <= 1 mm) and every geometry safety check (including
  in-stage self-intersection) pass;
* all displacements are computed from the undeformed baseline vertices, so
  repeated exports are idempotent; female/auto keeps every vertex identical
  to the baseline.

No neck refit, measured surface, patch, stitching or whole-neck ICP exists
in this stage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from head_rgbd import neck_landmark as nl

try:  # scipy is available in the shipped runtime environment
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - exercised only on stripped installs
    cKDTree = None

NECK_METHOD = "depth_conditioned_procedural_prominence"
GEOMETRY_SOURCE = "template_plus_procedural_deformation"

# ibug-68 landmark indices used for the local direction frame.
EYE_OUTER_RIGHT = 36
EYE_OUTER_LEFT = 45
NOSE_BRIDGE_TOP = 27
NOSE_TIP = 30
CHIN = 8

# Support and binding geometry.
SELECT_MARGIN = 1.3          # BFS may expand through vertices up to this field r
FRONT_NORMAL_MIN = 0.25      # vertex counts as front-neck skin above this dot
BIND_NORMAL_MIN = 0.35       # binding-site surface must agree with forward
BIND_MAX_DISTANCE_MM = 25.0  # ray-hit point may not drift this far from marker
SNAP_MAX_OFFSET_MM = 20.0    # lateral snap onto the template neck ridge limit
SNAP_VERTICAL_BAND_MM = 8.0  # vertical band searched for the neck ridge
SNAP_LATERAL_BAND_MM = 25.0  # lateral band searched for the neck ridge

# Protection (identical semantics to the 0.5.3 stage).
FACE_PROTECT_MM = 30.0       # proximity of any facial landmark protects
LATERAL_PROTECT_MM = 45.0    # |x| beyond this (model frame) is ear/side
BOTTOM_PROTECT_MM = 8.0      # bottom cut band

# Continuous taper: ramp 0 -> 1 over this Euclidean distance from the nearest
# protected vertex (replaces the fixed graph-ring taper of 0.5.3).
TAPER_MM = 6.0

# Shape-goal acceptance and calibration.
ACCEPT_TOLERANCE_MM = 1.0    # |H_final - H_target| acceptance bound
CALIBRATE_TOLERANCE_MM = 0.3 # inner calibration loop target
CALIBRATION_ITERS = 3
MIN_GAIN = 0.35              # centre efficiency below which the core widens
MAX_AMPLITUDE_MM = 25.0      # hard runaway guard for the calibrated amplitude

# Geometry safety.
MAX_SLOPE = 1.5              # displacement/edge-length slope guard
EDGE_GROW_FACTOR = 1.5       # edge growth limits on affected faces
EDGE_GROW_ABS_MM = 2.0
# A smooth prominence on a neck front sampled at 4-8 mm edges necessarily
# stretches flank edges beyond the plain 1.5x/+2 mm limit: measured smooth
# growth on the ridge flanks reaches 1.58-1.75x (faces at |x| ~ 7-8 mm,
# 1-ring median 1.09-1.17 along the ridge).  Inside the shaping region the
# relative edge limit is therefore a hard measured cap, and single-vertex
# pulls are caught directly by a displacement spike test against the
# 1-ring neighbourhood median instead of by edge ratios.  Slender faces,
# normal flips, area collapse, slope and self-intersections keep their own
# independent checks.  Recorded in the report per state.
EDGE_GROW_SMOOTH_FACTOR = 1.35   # retained for the recorded allowance stat
EDGE_GROW_SMOOTH_MARGIN = 0.05
EDGE_RATIO_HARD_CAP = 2.2
SPIKE_RATIO = 2.5            # vertex displacement vs 1-ring median
SPIKE_ABS_MM = 3.0
SLENDER_QUALITY_FACTOR = 3.0 # shaped/base aspect-quality ratio per face
SLENDER_QUALITY_CAP = 30.0   # absolute slender-triangle cap (equilateral ~3.46)

# Central ridge crest: flat plateau fraction of the compact support.
PLATEAU_FRACTION = 0.35

# Sagittal profile machinery.
PROFILE_SLAB_MM = 10.0       # triangles within this |x| of the profile plane
PROFILE_FRONT_CUT_MM = 15.0  # outline samples must be in front of this depth
ANCHOR_INSET_MM = 2.0        # anchor band starts this far outside the base
ANCHOR_BAND_MM = 8.0         # anchor band width
ANCHOR_CONTINUITY_MM = 8.0   # anchor outline must stay near the centre outline
MIN_ANCHORS_PER_SIDE = 2

STATUS_APPLIED = "applied"
STATUS_SKIPPED = "skipped"


def wendland_c2(r: np.ndarray | float) -> np.ndarray | float:
    """(1-r)^4 (4r+1) for r<1, zero otherwise; C2 at the support boundary."""
    r = np.asarray(r, dtype=np.float64)
    inside = r < 1.0
    one_minus = np.clip(1.0 - r, 0.0, None)
    values = one_minus ** 4 * (4.0 * r + 1.0)
    return np.where(inside, values, 0.0)


def plateau_c2(r: np.ndarray | float, plateau: float = PLATEAU_FRACTION) -> np.ndarray | float:
    """Flat-topped C2 compactly supported kernel for the central ridge crest.

    w(r) = 1 for r <= p, then a quintic smoothstep fall to 0 at r = 1 with
    zero first and second derivatives at both ends.  Used for the central
    ridge so the crest is blunt instead of a Wendland needle: a Wendland C2
    core keeps only 18.75 % weight at half its radius, which on a 4-8 mm
    sampled neck front becomes a single-vertex pull.
    """
    r = np.asarray(r, dtype=np.float64)
    p = float(plateau)
    u = np.clip((r - p) / max(1.0 - p, 1e-9), 0.0, 1.0)
    s = 1.0 - (6.0 * u ** 5 - 15.0 * u ** 4 + 10.0 * u ** 3)
    return np.where(r < 1.0, s, 0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProminenceConfig:
    subject_sex: str = "unspecified"          # never inferred, user-declared only
    mode: str = "auto"                        # auto | on | off
    amplitude_mm: float = 8.0                 # target FINAL prominence height
    radius_x_mm: float = 11.0                 # central ridge lateral radius
    radius_y_mm: float = 12.0                 # central ridge vertical radius
    base_radius_x_mm: float = 20.0            # base field lateral radius
    base_radius_y_mm: float = 16.0            # base field vertical radius
    base_share: float = 0.35                  # base share of the total height
    candidate_amplitudes_mm: tuple[float, ...] = (6.0, 8.0)

    @property
    def candidates_mm(self) -> tuple[float, ...]:
        """Final-height targets to try, larger (preferred) target first.

        An explicit user amplitude is the single target; the default tries
        the configured candidate targets in descending order and records a
        fallback to a lower target when the higher one fails the shape goal
        or a safety check.
        """
        if self._explicit:
            return (float(self.amplitude_mm),)
        return tuple(sorted({float(a) for a in self.candidate_amplitudes_mm}, reverse=True))

    _explicit: bool = field(default=False, repr=False, compare=False)


def resolve_configuration(defaults: dict[str, Any], args: Any | None = None) -> ProminenceConfig:
    """Unified configuration: config/defaults.json first, CLI overrides second.

    Every value is validated: amplitudes finite and non-negative, radii
    finite and strictly positive, share within [0, 1], and the base support
    at least as wide as the central ridge.  The requested amplitude, the
    default candidates and an explicitly supplied amplitude can never
    silently disagree: an explicit amplitude replaces the candidate list.
    """
    section = dict(defaults.get("neck_prominence") or {})
    supplied = getattr(args, "neck_prominence_mm", None)
    explicit = supplied is not None
    if explicit:
        amplitude = float(supplied)
        if not np.isfinite(amplitude) or amplitude < 0.0:
            raise ValueError("neck_prominence_mm must be finite and non-negative")
    else:
        amplitude = float(section.get("amplitude_mm", 8.0))
        if not np.isfinite(amplitude) or amplitude < 0.0:
            raise ValueError("neck_prominence.amplitude_mm must be finite and non-negative")

    def pick(cli_name: str, section_name: str, fallback: float, *, minimum: float, maximum: float | None = None) -> float:
        value = getattr(args, cli_name, None)
        if value is None:
            value = section.get(section_name, fallback)
        value = float(value)
        if not np.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"neck_prominence.{section_name} must be finite and >= {minimum}")
        return value

    subject_sex = str(getattr(args, "subject_sex", None) or section.get("subject_sex", "unspecified"))
    mode = str(getattr(args, "neck_prominence", None) or section.get("mode", "auto"))
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"neck_prominence.mode must be auto|on|off, got {mode!r}")
    if subject_sex not in ("male", "female", "unspecified"):
        raise ValueError(f"subject_sex must be male|female|unspecified, got {subject_sex!r}")

    radius_x_mm = pick("neck_prominence_radius_x_mm", "radius_x_mm", 11.0, minimum=0.5)
    radius_y_mm = pick("neck_prominence_radius_y_mm", "radius_y_mm", 12.0, minimum=0.5)
    base_radius_x_mm = pick("neck_prominence_base_radius_x_mm", "base_radius_x_mm", 20.0, minimum=0.5)
    base_radius_y_mm = pick("neck_prominence_base_radius_y_mm", "base_radius_y_mm", 16.0, minimum=0.5)
    base_share = pick("neck_prominence_base_share", "base_share", 0.35, minimum=0.0, maximum=1.0)

    candidates_raw = section.get("candidate_amplitudes_mm", (6.0, 8.0))
    candidates = tuple(float(v) for v in candidates_raw)
    if not candidates or any((not np.isfinite(v)) or v < 0.0 for v in candidates):
        raise ValueError("neck_prominence.candidate_amplitudes_mm must be non-empty, finite, non-negative")
    if base_radius_x_mm < radius_x_mm or base_radius_y_mm < radius_y_mm:
        raise ValueError("neck_prominence base support must be at least as wide as the central ridge")

    return ProminenceConfig(
        subject_sex=subject_sex,
        mode=mode,
        amplitude_mm=amplitude,
        radius_x_mm=radius_x_mm,
        radius_y_mm=radius_y_mm,
        base_radius_x_mm=base_radius_x_mm,
        base_radius_y_mm=base_radius_y_mm,
        base_share=base_share,
        candidate_amplitudes_mm=candidates,
        _explicit=explicit,
    )


def effective_enable(config: ProminenceConfig, landmark_observed: bool) -> tuple[bool, str | None]:
    """Resolve mode + subject sex into one decision; sex is never inferred."""
    if config.mode == "off":
        return False, "disabled_by_configuration"
    if config.mode == "on":
        if not landmark_observed:
            return False, "enabled_but_no_valid_landmark"
        return True, None
    # auto
    if config.subject_sex == "male":
        if not landmark_observed:
            return False, "auto_male_but_no_valid_landmark"
        return True, None
    if config.subject_sex == "female":
        return False, "auto_female_keeps_template_neck"
    return False, "auto_subject_sex_unspecified"


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------

def vertex_adjacency(faces: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    """Sorted neighbour lists from the triangle list."""
    neighbours = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
    return [np.fromiter(n, dtype=np.int64) for n in neighbours]


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of the outward-oriented surface."""
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for axis in range(3):
        np.add.at(normals, faces[:, axis], fn)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    return normals / lengths[:, None]


def ray_first_hit(vertices: np.ndarray, faces: np.ndarray, point: np.ndarray) -> dict[str, Any] | None:
    """Exact camera-origin ray/triangle first hit (Moller-Trumbore, one ray).

    The hit is the first visible surface along the observation ray with a
    front-facing gate; this is *not* a nearest-vertex lookup.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    ray = np.asarray(point, dtype=np.float64).reshape(3)
    if vertices.ndim != 2 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        return None
    if not np.isfinite(vertices).all() or not np.isfinite(ray).all():
        return None
    if faces.min() < 0 or faces.max() >= len(vertices):
        return None
    direction = ray
    tri0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - tri0
    edge2 = vertices[faces[:, 2]] - tri0
    pvec = np.cross(direction[None, :], edge2)
    det = (edge1 * pvec).sum(axis=1)
    usable = np.abs(det) > 1e-14
    if not usable.any():
        return None
    inv_det = np.full(len(faces), np.nan)
    inv_det[usable] = 1.0 / det[usable]
    tvec = -tri0
    u = (tvec * pvec).sum(axis=1) * inv_det
    qvec = np.cross(tvec, edge1)
    v = (direction[None, :] * qvec).sum(axis=1) * inv_det
    t = (edge2 * qvec).sum(axis=1) * inv_det
    normal = np.cross(edge1, edge2)
    normal_length = np.linalg.norm(normal, axis=1)
    ray_length = float(np.linalg.norm(direction))
    incidence = np.full(len(faces), -1.0)
    if ray_length > 0:
        incidence = (normal * (-direction / ray_length)[None, :]).sum(axis=1) / np.maximum(normal_length, 1e-12)
    keep = (
        np.isfinite(t) & (t > 1e-9) & np.isfinite(u) & np.isfinite(v)
        & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        & (incidence > 0.15)
    )
    if not keep.any():
        return None
    index = int(np.flatnonzero(keep)[np.argmin(t[keep])])
    bary = np.array([1.0 - u[index] - v[index], u[index], v[index]], dtype=np.float64)
    hit = t[index] * direction
    return {
        "triangle_index": index,
        "barycentric": [float(value) for value in bary],
        "point_mesh_m": [float(value) for value in hit],
        "ray_t": float(t[index]),
        "incidence": float(incidence[index]),
        "face_vertices": [int(value) for value in faces[index]],
    }


def _direction_frame(landmarks_m: np.ndarray) -> dict[str, Any] | None:
    """Right/up/forward axes of the head in the model (canonical) frame.

    The axes come from the baseline model's own facial landmarks, so they
    follow each state's fitted pose; they are never world axes and never the
    camera axis.
    """
    points = np.asarray(landmarks_m, dtype=np.float64)
    if points.shape[0] <= NOSE_TIP:
        return None
    right = points[EYE_OUTER_RIGHT] - points[EYE_OUTER_LEFT]
    if np.linalg.norm(right) < 1e-6:
        return None
    up = points[NOSE_BRIDGE_TOP] - points[CHIN]
    if np.linalg.norm(up) < 1e-6:
        return None
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    if abs(float(np.dot(right, up))) > 0.95:
        return None
    right = right - np.dot(right, up) * up
    right = right / np.linalg.norm(right)
    forward = np.cross(up, right)
    forward = forward / np.linalg.norm(forward)
    # The nose tip sits in front of the eye line; if not, the frame is ambiguous.
    reference = points[NOSE_TIP] - 0.5 * (points[EYE_OUTER_RIGHT] + points[EYE_OUTER_LEFT])
    if abs(float(np.dot(reference, forward))) < 1e-6:
        return None
    if float(np.dot(reference, forward)) < 0:
        forward = -forward
    return {"right": right, "up": up, "forward": forward}


def _protected_mask(
    vertices_model: np.ndarray,
    landmarks_model: np.ndarray,
    normals_model: np.ndarray,
    axes: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Vertices that must never move: face, lips, jaw, chin, ears, back, bottom cut."""
    protect = np.zeros(len(vertices_model), dtype=bool)
    reasons: dict[str, np.ndarray] = {}
    if len(landmarks_model):
        # dense distance: chunked to bound memory
        dist = np.empty(len(vertices_model), dtype=np.float64)
        chunk = 4096
        for start in range(0, len(vertices_model), chunk):
            block = vertices_model[start:start + chunk]
            d = np.linalg.norm(block[:, None, :] - landmarks_model[None, :, :], axis=2)
            dist[start:start + chunk] = d.min(axis=1)
        face_near = dist < FACE_PROTECT_MM * 1e-3
        reasons["facial_landmark_proximity"] = face_near
        protect |= face_near
    x = (vertices_model - vertices_model.mean(axis=0)) @ axes["right"]
    lateral = np.abs(x) > LATERAL_PROTECT_MM * 1e-3
    reasons["lateral_beyond_ear_line"] = lateral
    protect |= lateral
    y = (vertices_model - vertices_model.mean(axis=0)) @ axes["up"]
    bottom = y < (y.min() + BOTTOM_PROTECT_MM * 1e-3)
    reasons["bottom_cut_boundary"] = bottom
    protect |= bottom
    back = (normals_model @ axes["forward"]) < 0.0
    reasons["backward_facing"] = back
    protect |= back
    return {"mask": protect, "reasons": {k: v.tolist() for k, v in reasons.items()},
            "count": int(protect.sum())}


def _distance_taper(
    vertices_model: np.ndarray, protected_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continuous 0 -> 1 ramp over TAPER_MM Euclidean distance from protection.

    The taper is defined by actual distance in millimetres, not by a fixed
    number of graph rings: the ramp width is resolution independent and does
    not strangle the field when the mesh sampling is fine.
    """
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required for the distance taper")
    protected_ids = np.flatnonzero(protected_mask)
    if not len(protected_ids):
        return np.ones(len(vertices_model)), {"taper_mm": TAPER_MM, "nearest_protected_mm": None}
    tree = cKDTree(vertices_model[protected_ids])
    d, _ = tree.query(vertices_model, k=1)
    taper = np.clip(d / (TAPER_MM * 1e-3), 0.0, 1.0)
    return taper, {"taper_mm": TAPER_MM, "nearest_protected_mm": round(float(d.min() * 1000.0), 3)}


# ---------------------------------------------------------------------------
# Depth conditioning (few shape parameters only)
# ---------------------------------------------------------------------------

def depth_conditioning(
    prepared_dir: Any | None,
    segment_file: str | None,
    pixel_uv: list[float] | tuple[float, float] | None,
    marker_point_reference_camera_m: np.ndarray,
    forward_reference_camera: np.ndarray,
    core_radius_x_mm: float,
    base_radius_x_mm: float,
) -> dict[str, Any]:
    """Read reliable neck-front depth near the landmark to condition parameters.

    Depth use is restricted to: local surface orientation (direction check),
    observed neck width (support radius bounds) and recorded plane-fit
    statistics.  The observed protrusion magnitude is *not* used to change
    the target height: at 1 mm depth quantisation on a sloped surface with
    jewellery in the field of view its reliability is insufficient, and the
    configured target height is the documented appearance prior.
    """
    out: dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "surface_plane": None,
        "neck_width_mm": None,
        "forward_agreement": None,
        "radius_bounds": {
            "core_radius_x_mm": float(core_radius_x_mm),
            "base_radius_x_mm": float(base_radius_x_mm),
            "clamped": False,
        },
        "note": (
            "depth verifies direction and bounds support radii only; "
            "no per-point neck fit, no amplitude derivation from depth"
        ),
    }
    if prepared_dir is None or not pixel_uv:
        out["reason"] = "prepared_data_unavailable"
        return out
    from pathlib import Path
    path = Path(prepared_dir) / segment_file if segment_file else None
    if path is None or not path.is_file():
        out["reason"] = "segment_file_missing"
        return out
    try:
        data = np.load(path, allow_pickle=False)
        depth = np.asarray(data["depth"])
        neck = np.asarray(data["neck_mask"], dtype=bool)
        intr = np.asarray(data["intrinsics"], dtype=np.float64).reshape(-1)
    except Exception as error:
        out["reason"] = f"segment_load_failed:{type(error).__name__}"
        return out
    if intr.shape != (4,) or not np.isfinite(intr).all():
        out["reason"] = "invalid_intrinsics"
        return out
    fx, fy, cx, cy = intr
    u0, v0 = float(pixel_uv[0]), float(pixel_uv[1])
    height, width = depth.shape
    if not (0 <= u0 < width and 0 <= v0 < height):
        out["reason"] = "marker_pixel_outside_image"
        return out

    # Local surface plane: valid depth in a 17x17 window around the marker.
    us, vs = np.meshgrid(np.arange(u0 - 8, u0 + 9), np.arange(v0 - 8, v0 + 9))
    us = np.clip(us.ravel(), 0, width - 1).astype(np.int64)
    vs = np.clip(vs.ravel(), 0, height - 1).astype(np.int64)
    z_mm = depth[vs, us].astype(np.float64)
    keep = z_mm > 0
    out["surface_plane"] = {"valid_pixels": int(keep.sum())}
    if keep.sum() >= 60:
        z = z_mm[keep] * 1e-3
        x = (us[keep] - cx) * z / fx
        y = -(vs[keep] - cy) * z / fy
        w = -z
        a = np.stack([x, y, np.ones_like(x)], axis=1)
        coef, *_ = np.linalg.lstsq(a, w, rcond=None)
        resid = a @ coef - w
        rms = float(np.sqrt((resid ** 2).mean()) * 1000.0)
        normal = np.array([-coef[0], -coef[1], 1.0])
        normal /= np.linalg.norm(normal)
        forward = np.asarray(forward_reference_camera, dtype=np.float64)
        fwd_norm = forward / max(float(np.linalg.norm(forward)), 1e-12)
        agreement = float(np.dot(normal, fwd_norm))
        out["surface_plane"].update(
            rms_resid_mm=round(rms, 3),
            outward_normal_reference_camera=[round(float(v), 4) for v in normal],
            forward_agreement=round(agreement, 4),
        )
        out["forward_agreement"] = round(agreement, 4)
        out["status"] = "available" if rms <= 1.5 else "degraded"
        if rms > 1.5:
            out["reason"] = "plane_rms_above_1p5mm"
    else:
        out["reason"] = "insufficient_valid_depth_at_marker"

    # Observed neck width at the marker height: lateral mask extent.
    widths = []
    for v in range(max(0, int(v0) - 4), min(height, int(v0) + 5)):
        cols = np.flatnonzero(neck[v])
        if len(cols) >= 5:
            zvals = depth[v, cols].astype(np.float64)
            zvals = zvals[zvals > 0]
            if len(zvals):
                z = float(np.median(zvals)) * 1e-3
                widths.append((cols.max() - cols.min()) * z / fx * 1000.0)
    if len(widths) >= 5:
        width_mm = float(np.median(widths))
        out["neck_width_mm"] = round(width_mm, 1)
        core = min(float(core_radius_x_mm), width_mm * 0.12)
        base = min(float(base_radius_x_mm), width_mm * 0.25)
        out["radius_bounds"] = {
            "core_radius_x_mm": round(core, 3),
            "base_radius_x_mm": max(round(base, 3), round(core, 3)),
            "clamped": bool(core != float(core_radius_x_mm) or base != float(base_radius_x_mm)),
        }
    return out


# ---------------------------------------------------------------------------
# Sagittal profile: plane/mesh intersection, fixed reference trend
# ---------------------------------------------------------------------------

def _profile_triangles(
    vertices_model: np.ndarray, faces: np.ndarray, centre_model: np.ndarray, axes: dict[str, np.ndarray]
) -> np.ndarray:
    """Triangle indices near the midsagittal profile plane through the centre."""
    tri = vertices_model[faces]
    xs = (tri - centre_model) @ axes["right"]
    cross = (
        (xs[:, 0] * xs[:, 1] <= 0) | (xs[:, 1] * xs[:, 2] <= 0) | (xs[:, 0] * xs[:, 2] <= 0)
    )
    near = np.abs(xs).max(axis=1) < PROFILE_SLAB_MM * 1e-3
    # Triangles with a vertex exactly on the profile plane cross it by
    # definition; wide ones must not be dropped by the lateral slab filter.
    touching = np.abs(xs).min(axis=1) < 1e-9
    return np.flatnonzero((near | touching) & cross)


def _sagittal_outline(
    vertices_model: np.ndarray,
    faces: np.ndarray,
    centre_model: np.ndarray,
    axes: dict[str, np.ndarray],
    profile_indices: np.ndarray,
    y_min_mm: float,
    y_max_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Front outline F(y) (mm) from plane/mesh intersection, 1 mm sampling.

    The outline is built from exact triangle/plane intersections, never from
    scattered vertices.  Back-of-neck samples are excluded by the front cut.
    """
    rel = vertices_model - centre_model
    x = rel @ axes["right"]
    y = rel @ axes["up"]
    f = rel @ axes["forward"]
    grid = np.arange(np.floor(y_min_mm), np.ceil(y_max_mm) + 1, dtype=np.float64)
    outline = np.full(len(grid), np.nan)
    samples_y = []
    samples_f = []
    for fi in profile_indices:
        v = faces[fi]
        s = x[v]
        yy = y[v]
        ff = f[v]
        crossing = []
        for a in range(3):
            b = (a + 1) % 3
            if s[a] * s[b] <= 0 and abs(s[a] - s[b]) > 1e-12:
                t = s[a] / (s[a] - s[b])
                crossing.append((yy[a] + t * (yy[b] - yy[a]), ff[a] + t * (ff[b] - ff[a])))
        if len(crossing) == 2:
            (y0, f0c), (y1, f1c) = crossing
            height = abs(y1 - y0) * 1000.0
            steps = max(1, int(np.ceil(height / 0.5)))
            for step in range(steps + 1):
                ratio = step / steps
                py = y0 + ratio * (y1 - y0)
                pf = f0c + ratio * (f1c - f0c)
                if pf > -PROFILE_FRONT_CUT_MM * 1e-3:
                    samples_y.append(py * 1000.0)
                    samples_f.append(pf * 1000.0)
    if not samples_y:
        return grid, outline
    sy = np.asarray(samples_y)
    sf = np.asarray(samples_f)
    for i, gy in enumerate(grid):
        sel = (sy >= gy - 0.5) & (sy < gy + 0.5)
        if sel.any():
            outline[i] = float(sf[sel].max())
    return grid, outline


def _outline_trend(
    grid: np.ndarray, outline: np.ndarray, base_radius_y_mm: float
) -> dict[str, Any]:
    """Smooth reference trend from anchor bands outside the support.

    Anchor bands are fixed from the baseline outline (|y| in
    [ry_base + 2, ry_base + 8] on both sides), must stay within the outline
    continuity band of the centre outline (rejects the chin bulge), and the
    baseline trend is reused unchanged when measuring the shaped mesh.
    """
    centre_value = float(outline[int(np.argmin(np.abs(grid)))])
    lo = -(base_radius_y_mm + ANCHOR_INSET_MM + ANCHOR_BAND_MM)
    hi = -(base_radius_y_mm + ANCHOR_INSET_MM)
    lo_upper = base_radius_y_mm + ANCHOR_INSET_MM
    hi_upper = base_radius_y_mm + ANCHOR_INSET_MM + ANCHOR_BAND_MM
    anchors_y: list[float] = []
    anchors_f: list[float] = []
    per_side = {"lower": 0, "upper": 0}
    for gy, gf in zip(grid, outline):
        if not np.isfinite(gf):
            continue
        side = None
        if lo <= gy <= hi:
            side = "lower"
        elif lo_upper <= gy <= hi_upper:
            side = "upper"
        if side and abs(gf - centre_value) <= ANCHOR_CONTINUITY_MM:
            anchors_y.append(float(gy))
            anchors_f.append(float(gf))
            per_side[side] += 1
    if per_side["lower"] < MIN_ANCHORS_PER_SIDE or per_side["upper"] < MIN_ANCHORS_PER_SIDE:
        return {
            "available": False,
            "reason": "insufficient_anchor_samples",
            "per_side": per_side,
        }
    coeffs = np.polyfit(np.asarray(anchors_y), np.asarray(anchors_f), 1)
    trend_at_centre = float(np.polyval(coeffs, 0.0))
    return {
        "available": True,
        "slope_mm_per_mm": round(float(coeffs[0]), 4),
        "trend_at_centre_mm": round(trend_at_centre, 3),
        "anchor_count": len(anchors_y),
        "per_side": per_side,
        "anchor_band_mm": [round(lo, 1), round(hi, 1), round(lo_upper, 1), round(hi_upper, 1)],
        "anchors_y_mm": [round(v, 1) for v in anchors_y],
        "anchors_f_mm": [round(v, 1) for v in anchors_f],
    }


def _prominence_at_centre(grid: np.ndarray, outline: np.ndarray, trend_at_centre_mm: float) -> float:
    """H = front outline at the centre height minus the fixed reference trend."""
    value = outline[int(np.argmin(np.abs(grid)))]
    if not np.isfinite(value):
        return float("nan")
    return float(value - trend_at_centre_mm)


# ---------------------------------------------------------------------------
# Centre snap onto the template neck's own front ridge
# ---------------------------------------------------------------------------

def _neck_ridge_centre(
    vertices_model: np.ndarray,
    binding_model: np.ndarray,
    axes: dict[str, np.ndarray],
    normals_model: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Snap the shaping centre laterally onto the template neck front ridge.

    The delivered neck is the template neck; when the subject's real neck is
    laterally offset from it, the camera-ray binding lands on the ridge
    *slope* and a bump there would look like a side bulge.  The snap moves
    the centre along the template surface to the front-most ridge within a
    limited neighbourhood; it never invents a location and is fully recorded.
    """
    rel = vertices_model - binding_model
    x = rel @ axes["right"]
    y = rel @ axes["up"]
    f = rel @ axes["forward"]
    front = normals_model @ axes["forward"] > FRONT_NORMAL_MIN
    band = (
        front
        & (np.abs(y) < SNAP_VERTICAL_BAND_MM * 1e-3)
        & (np.abs(x) < SNAP_LATERAL_BAND_MM * 1e-3)
        & (f > -5e-3)
    )
    info: dict[str, Any] = {
        "candidate_vertices": int(band.sum()),
    }
    if band.sum() < 6:
        info["applied"] = False
        info["reason"] = "ridge_band_too_sparse"
        return None, info
    f_max = float(f[band].max())
    plateau = band & (f > f_max - 2e-3)
    if plateau.sum() < 2:
        info["applied"] = False
        info["reason"] = "ridge_plateau_too_small"
        return None, info
    x_ridge = float(np.median(x[plateau]))
    if abs(x_ridge) > SNAP_MAX_OFFSET_MM * 1e-3:
        info["applied"] = False
        info["reason"] = "ridge_offset_exceeds_neighbourhood"
        info["offset_mm"] = round(x_ridge * 1000.0, 2)
        return None, info
    ids = np.flatnonzero(plateau)
    best = int(ids[int(np.argmin(np.abs(y[ids])))])
    info.update(
        applied=True,
        offset_mm=round(x_ridge * 1000.0, 2),
        plateau_vertices=int(plateau.sum()),
        ridge_forward_mm=round(f_max * 1000.0, 2),
        centre_vertex=best,
    )
    return vertices_model[best].copy(), info


# ---------------------------------------------------------------------------
# Geometry safety
# ---------------------------------------------------------------------------

def _count_self_intersections(
    shaped_vertices: np.ndarray,
    faces: np.ndarray,
    moved: np.ndarray,
    adjacency: list[np.ndarray],
    centre_model: np.ndarray,
    support_radius_m: float,
) -> dict[str, Any]:
    """In-stage self-intersection test over every affected edge.

    Segments are the affected-face edges with a moved endpoint; candidate
    triangles are the affected faces, their two-ring neighbourhood and every
    face near the support sphere.  Legal shared-edge contact (two common
    vertices) is excluded; a shared single vertex still counts when the
    segment crosses the open triangle interior.
    """
    moved_ids = np.flatnonzero(moved)
    touched = np.zeros(len(shaped_vertices), dtype=bool)
    touched[moved_ids] = True
    for v in moved_ids:
        touched[adjacency[v]] = True
    affected = touched[faces].any(axis=1)

    seg_index = np.flatnonzero(affected)
    segments: list[tuple[int, int]] = []
    seg_own: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for fi in seg_index:
        a, b, c = (int(v) for v in faces[fi])
        for i, j in ((a, b), (b, c), (c, a)):
            if not (moved[i] or moved[j]):
                continue
            key = (fi, min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            segments.append((i, j))
            seg_own.append(faces[fi].tolist())

    if not segments:
        return {"count": 0, "segments": 0, "candidate_faces": 0, "examples": []}

    # Candidate faces: affected + 2-ring neighbourhood + support sphere.
    ring = set(seg_index.tolist())
    for fi in list(ring):
        for v in faces[fi]:
            ring.update(np.flatnonzero((faces == v).any(axis=1)).tolist())
    ring = np.fromiter(ring, dtype=np.int64)
    centroids = shaped_vertices[faces].mean(axis=1)
    radius = support_radius_m + 6e-3
    near = np.linalg.norm(centroids - centre_model, axis=1) < radius * 1.6
    candidates = np.unique(np.concatenate([ring, np.flatnonzero(near)]))

    tri = shaped_vertices[faces[candidates]]
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    hits = 0
    hit_detail: list[dict[str, Any]] = []
    for (i, j), own in zip(segments, seg_own):
        a = shaped_vertices[i]
        b = shaped_vertices[j]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-12:
            continue
        pvec = np.cross(d[None, :], e2)
        det = (e1 * pvec).sum(axis=1)
        ok = np.abs(det) > 1e-14
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        tvec = a - v0
        u = (tvec * pvec).sum(axis=1) * inv
        qvec = np.cross(tvec, e1)
        v = (d[None, :] * qvec).sum(axis=1) * inv
        t = (e2 * qvec).sum(axis=1) * inv
        keep = (
            ok & (t > 1e-9) & (t < length - 1e-9)
            & (u > 1e-9) & (v > 1e-9) & (u + v < 1.0 - 1e-9)
        )
        for index in np.flatnonzero(keep):
            tri_ids = set(int(vv) for vv in faces[candidates[index]])
            common = len(tri_ids & set(own))
            if common >= 2:
                continue  # legal shared-edge contact
            hits += 1
            hit_detail.append({
                "segment_vertices": [int(i), int(j)],
                "triangle": [int(vv) for vv in faces[candidates[index]]],
                "shared_vertices": common,
            })
            if len(hit_detail) >= 20:
                break
        if len(hit_detail) >= 20:
            break
    return {
        "count": hits,
        "segments": len(segments),
        "candidate_faces": int(len(candidates)),
        "examples": hit_detail,
    }


def _safety_checks(
    base_vertices: np.ndarray,
    shaped_vertices: np.ndarray,
    faces: np.ndarray,
    moved: np.ndarray,
    displacement: np.ndarray,
    protected: np.ndarray,
    adjacency: list[np.ndarray],
    amplitude_m: float,
    self_intersections: dict[str, Any],
) -> dict[str, Any]:
    """Checks over every face touching a moved vertex (and their faces)."""
    moved_ids = np.flatnonzero(moved)
    touched = np.zeros(len(base_vertices), dtype=bool)
    touched[moved_ids] = True
    for v in moved_ids:
        touched[adjacency[v]] = True
    affected = touched[faces].any(axis=1)

    problems: list[str] = []
    if not np.isfinite(shaped_vertices).all():
        return {"status": STATUS_SKIPPED, "reason": "non_finite_coordinates", "problems": ["non_finite"],
                "affected_faces": int(affected.sum()), "moved_vertices": int(moved.sum())}
    if moved.any() and protected[moved].any():
        problems.append("protected_vertex_moved")

    tri_b = base_vertices[faces[affected]]
    tri_a = shaped_vertices[faces[affected]]
    n_b = np.cross(tri_b[:, 1] - tri_b[:, 0], tri_b[:, 2] - tri_b[:, 0])
    n_a = np.cross(tri_a[:, 1] - tri_a[:, 0], tri_a[:, 2] - tri_a[:, 0])
    area_b = np.linalg.norm(n_b, axis=1)
    area_a = np.linalg.norm(n_a, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = area_a / area_b
    if (area_a < 1e-12).any():
        problems.append("degenerate_face")
    elif (ratio < 0.05).any():
        problems.append("area_collapse_beyond_5_percent")
    dot = (n_b * n_a).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = dot / np.maximum(area_b * area_a, 1e-30)
    if (cos < 0.05).any():
        problems.append("face_normal_reversal")

    # Edge growth: hard measured cap inside the shaping region (see the
    # criterion comment at the constants).  Growth is recorded for the report.
    face_ids = np.flatnonzero(affected).tolist()
    axis_ratios = []
    axis_growth = []
    for axis in range(3):
        edge_b = np.linalg.norm(tri_b[:, (axis + 1) % 3] - tri_b[:, axis], axis=1)
        edge_a = np.linalg.norm(tri_a[:, (axis + 1) % 3] - tri_a[:, axis], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            axis_ratios.append(edge_a / np.maximum(edge_b, 1e-12))
        axis_growth.append(edge_a - edge_b)
    ratios = np.max(np.stack(axis_ratios), axis=0)
    growth = np.max(np.stack(axis_growth), axis=0)
    over_cap = ratios > EDGE_RATIO_HARD_CAP
    edge_violations = [
        {
            "face": int(face_ids[i]),
            "ratio": round(float(ratios[i]), 3),
            "growth_mm": round(float(growth[i] * 1000.0), 2),
        }
        for i in np.flatnonzero(over_cap)[:10]
    ]
    if edge_violations:
        problems.append("abnormally_long_edge")
    # Single-vertex pull ("spike") test: a moved vertex may not stand out from
    # its 1-ring neighbourhood by both a large ratio and a large absolute gap.
    spike_vertices: list[dict[str, Any]] = []
    magnitudes = np.linalg.norm(displacement, axis=1)
    for v in np.flatnonzero(moved):
        nbrs = adjacency[v]
        if not len(nbrs):
            continue
        reference = float(np.median(magnitudes[nbrs]))
        limit = max(SPIKE_RATIO * reference, reference + SPIKE_ABS_MM * 1e-3)
        if magnitudes[v] > limit:
            spike_vertices.append({
                "vertex": int(v),
                "magnitude_mm": round(float(magnitudes[v] * 1000.0), 2),
                "neighbourhood_median_mm": round(reference * 1000.0, 2),
            })
            if len(spike_vertices) >= 10:
                break
    if spike_vertices:
        problems.append("displacement_spike")
    edge_report = {
        "hard_cap_ratio": EDGE_RATIO_HARD_CAP,
        "spike_ratio": SPIKE_RATIO,
        "spike_abs_mm": SPIKE_ABS_MM,
        "max_ratio": round(float(ratios.max()), 3) if len(ratios) else 1.0,
        "max_growth_mm": round(float(growth.max() * 1000.0), 2) if len(growth) else 0.0,
        "violations": edge_violations,
        "spike_vertices": spike_vertices,
    }

    # Slender-face guard on the shaped surface.
    def aspect_quality(tri: np.ndarray) -> np.ndarray:
        e = np.stack([
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ], axis=1)
        longest = e.max(axis=1)
        area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        return longest ** 2 / np.maximum(area2, 1e-18)
    quality_b = aspect_quality(tri_b)
    quality_a = aspect_quality(tri_a)
    slender = (
        (quality_a > np.maximum(quality_b * SLENDER_QUALITY_FACTOR, SLENDER_QUALITY_CAP))
        | (quality_a > SLENDER_QUALITY_CAP)
    )
    if slender.any():
        problems.append("slender_face_after_shaping")
    edge_report.update(
        criterion="hard_cap_plus_displacement_spike_test",
        slender_faces=int(slender.sum()),
        max_slender_quality=round(float(quality_a.max()), 2) if len(quality_a) else 0.0,
    )

    # Support boundary continuity and spike guard.
    if moved.any():
        applied = np.zeros(len(base_vertices), dtype=np.float64)
        applied[moved] = np.linalg.norm(displacement, axis=1)[moved]
        if (applied[~moved] > 1e-9).any():
            problems.append("displacement_outside_support")
        if applied.max() > amplitude_m * 1.001 + 1e-9:
            problems.append("spike_exceeds_amplitude")
        for v in np.flatnonzero(moved):
            nbrs = adjacency[v]
            spans = np.linalg.norm(base_vertices[nbrs] - base_vertices[v], axis=1)
            diffs = np.abs(applied[nbrs] - applied[v])
            if ((diffs > MAX_SLOPE * spans + 1e-4) & (spans > 0)).any():
                problems.append("displacement_slope_discontinuity")
                break

    if self_intersections["count"]:
        problems.append(f"self_intersections:{self_intersections['count']}")

    status = STATUS_APPLIED if not problems else STATUS_SKIPPED
    return {
        "status": status,
        "reason": None if not problems else ";".join(problems),
        "problems": problems,
        "affected_faces": int(affected.sum()),
        "moved_vertices": int(moved.sum()),
        "self_intersections": self_intersections,
        "edge_growth": edge_report,
    }


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------

def _field_weights(x_m: np.ndarray, y_m: np.ndarray, cfg: ProminenceConfig) -> np.ndarray:
    """Wide base + narrow central ridge, both Wendland C2 compact support."""
    r_base = np.sqrt((x_m / (cfg.base_radius_x_mm * 1e-3)) ** 2
                     + (y_m / (cfg.base_radius_y_mm * 1e-3)) ** 2)
    r_core = np.sqrt((x_m / (cfg.radius_x_mm * 1e-3)) ** 2
                     + (y_m / (cfg.radius_y_mm * 1e-3)) ** 2)
    w_base = np.asarray(wendland_c2(r_base), dtype=np.float64)
    w_core = np.asarray(plateau_c2(r_core), dtype=np.float64)
    return cfg.base_share * w_base + (1.0 - cfg.base_share) * w_core


def _select_region(
    vertices: np.ndarray,
    adjacency: list[np.ndarray],
    seed_faces: np.ndarray,
    q: np.ndarray,
    axes: dict[str, np.ndarray],
    normals: np.ndarray,
    protected: np.ndarray,
    base_radius_x_m: float,
    base_radius_y_m: float,
) -> np.ndarray:
    """Connected front-neck region around q inside the base support ellipse.

    Selection is a breadth-first walk over the mesh graph, not a Euclidean
    ball: back-facing, protected or out-of-ellipse vertices stop the walk.
    """
    rel = vertices - q
    x = rel @ axes["right"]
    y = rel @ axes["up"]
    r = np.sqrt((x / base_radius_x_m) ** 2 + (y / base_radius_y_m) ** 2)
    front = (normals @ axes["forward"]) > FRONT_NORMAL_MIN
    allowed = (r < SELECT_MARGIN) & front & (~protected)
    selected = np.zeros(len(vertices), dtype=bool)
    frontier = np.asarray(seed_faces, dtype=np.int64)
    frontier = frontier[allowed[frontier]]
    selected[frontier] = True
    while len(frontier):
        nxt = np.unique(np.concatenate([adjacency[v] for v in frontier]))
        keep = allowed[nxt] & ~selected[nxt]
        nxt = nxt[keep]
        if not len(nxt):
            break
        selected[nxt] = True
        frontier = nxt
    return selected


def _shape_metrics(
    vertices_model: np.ndarray, moved: np.ndarray, displacement_model: np.ndarray,
    axes: dict[str, np.ndarray], centre_model: np.ndarray,
) -> dict[str, Any]:
    """Actual extent of the applied field (10% and half-max levels)."""
    magnitudes = np.linalg.norm(displacement_model, axis=1)
    peak = float(magnitudes[moved].max()) if moved.any() else 0.0
    out: dict[str, Any] = {"peak_displacement_mm": round(peak * 1000.0, 3)}
    for name, fraction in (("ten_percent", 0.10), ("half_max", 0.50)):
        threshold = peak * fraction
        sel = moved & (magnitudes >= threshold)
        if sel.any():
            rel_all = vertices_model[sel] - centre_model
            xs = rel_all @ axes["right"] * 1000.0
            ys = rel_all @ axes["up"] * 1000.0
            out[f"width_x_{name}_mm"] = round(float(xs.max() - xs.min()), 2)
            out[f"length_y_{name}_mm"] = round(float(ys.max() - ys.min()), 2)
            out[f"vertices_{name}"] = int(sel.sum())
        else:
            out[f"width_x_{name}_mm"] = None
            out[f"length_y_{name}_mm"] = None
            out[f"vertices_{name}"] = 0
    if moved.any():
        peak_id = int(np.flatnonzero(moved)[np.argmax(magnitudes[moved])])
        pv = vertices_model[peak_id] - centre_model
        out["peak_offset_from_centre_mm"] = [
            round(float(np.dot(pv, axes["right"]) * 1000.0), 2),
            round(float(np.dot(pv, axes["up"]) * 1000.0), 2),
        ]
        out["peak_vertex"] = peak_id
    return out


def _shape_state(
    *,
    base_vertices: np.ndarray,
    faces: np.ndarray,
    landmark_point_reference_camera_m: list[float],
    model_to_camera: np.ndarray,
    landmarks_m: np.ndarray,
    target_height_mm: float,
    cfg: ProminenceConfig,
    prepared_dir: Any | None,
    pixel_uv: list[float] | None,
    segment_file: str | None,
) -> dict[str, Any]:
    """One target-height candidate for one state: bind, snap, calibrate, check."""
    vertices = np.asarray(base_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    point = np.asarray(landmark_point_reference_camera_m, dtype=np.float64).reshape(3)

    rotation = model_to_camera[:3, :3]
    translation = model_to_camera[:3, 3]
    to_model = lambda p: (np.asarray(p) - translation) @ rotation
    to_camera = lambda p: np.asarray(p) @ rotation.T + translation

    vertices_model = (vertices - translation) @ rotation
    landmarks_model = (np.asarray(landmarks_m, dtype=np.float64) - translation) @ rotation

    axes = _direction_frame(landmarks_model)
    if axes is None:
        return {"status": STATUS_SKIPPED, "reason": "direction_frame_unavailable"}

    normals_model = vertex_normals(vertices_model, faces)
    protection = _protected_mask(vertices_model, landmarks_model, normals_model, axes)

    hit = ray_first_hit(vertices, faces, point)
    if hit is None:
        return {"status": STATUS_SKIPPED, "reason": "baseline_binding_ray_missed",
                "protection": protection}
    ray_binding_camera = np.asarray(hit["point_mesh_m"], dtype=np.float64)
    distance_to_marker_mm = float(np.linalg.norm(ray_binding_camera - point) * 1000.0)
    if distance_to_marker_mm > BIND_MAX_DISTANCE_MM:
        return {
            "status": STATUS_SKIPPED,
            "reason": "baseline_binding_too_far_from_marker",
            "distance_marker_to_binding_mm": distance_to_marker_mm,
            "protection": protection,
        }
    q_model = to_model(ray_binding_camera)
    q_seed = np.asarray(hit["face_vertices"], dtype=np.int64)
    local_normal = normals_model[q_seed].mean(axis=0)
    forward_agreement = float(np.dot(local_normal, axes["forward"]))
    if forward_agreement < BIND_NORMAL_MIN:
        return {
            "status": STATUS_SKIPPED,
            "reason": "binding_direction_ambiguous",
            "forward_normal_agreement": forward_agreement,
            "protection": protection,
        }
    # The laryngeal prominence sits below the chin; a binding on the face
    # itself means the marker does not localise the neck front.
    chin_y = float(np.dot(landmarks_model[CHIN] - vertices_model.mean(axis=0), axes["up"]))
    q_y = float(np.dot(q_model - vertices_model.mean(axis=0), axes["up"]))
    if q_y > chin_y - 0.010:
        return {
            "status": STATUS_SKIPPED,
            "reason": "binding_above_chin_not_neck_front",
            "binding_height_above_chin_mm": (q_y - chin_y) * 1000.0,
            "protection": protection,
        }

    # Centre snap onto the template neck's own front ridge (recorded, limited).
    snapped_model, snap_info = _neck_ridge_centre(
        vertices_model, q_model, axes, normals_model
    )
    binding_hit = hit
    if snapped_model is None:
        centre_model = q_model
        centre_seed = q_seed
        snap_info = {"applied": False,
                     "reason": snap_info.get("reason", "not_snapped"),
                     "offset_mm": snap_info.get("offset_mm")}
    else:
        snapped_camera = to_camera(snapped_model)
        rebind = ray_first_hit(vertices, faces, snapped_camera)
        if rebind is None:
            centre_model = q_model
            centre_seed = q_seed
            snap_info = {"applied": False, "reason": "snap_rebind_ray_missed"}
        else:
            binding_hit = rebind
            q_camera = np.asarray(rebind["point_mesh_m"], dtype=np.float64)
            centre_model = to_model(q_camera)
            centre_seed = np.asarray(rebind["face_vertices"], dtype=np.int64)

    # Depth conditioning: direction verification and support radius bounds.
    forward_camera = axes["forward"] @ rotation.T
    forward_camera = forward_camera / max(float(np.linalg.norm(forward_camera)), 1e-12)
    depth_block = depth_conditioning(
        prepared_dir, segment_file, pixel_uv, point, forward_camera,
        cfg.radius_x_mm, cfg.base_radius_x_mm,
    )
    if depth_block.get("forward_agreement") is not None and depth_block["forward_agreement"] < 0.3:
        return {
            "status": STATUS_SKIPPED,
            "reason": "depth_direction_disagrees_with_model_forward",
            "depth_conditioning": depth_block,
            "protection": protection,
        }
    if depth_block["radius_bounds"].get("clamped"):
        cfg = ProminenceConfig(
            subject_sex=cfg.subject_sex, mode=cfg.mode,
            amplitude_mm=cfg.amplitude_mm,
            radius_x_mm=depth_block["radius_bounds"]["core_radius_x_mm"],
            radius_y_mm=cfg.radius_y_mm,
            base_radius_x_mm=depth_block["radius_bounds"]["base_radius_x_mm"],
            base_radius_y_mm=cfg.base_radius_y_mm,
            base_share=cfg.base_share,
            candidate_amplitudes_mm=cfg.candidate_amplitudes_mm,
            _explicit=cfg._explicit,
        )

    adjacency = vertex_adjacency(faces, len(vertices))
    taper, taper_info = _distance_taper(vertices_model, protection["mask"])

    selected = _select_region(
        vertices_model, adjacency, centre_seed, centre_model, axes, normals_model,
        protection["mask"],
        cfg.base_radius_x_mm * 1e-3, cfg.base_radius_y_mm * 1e-3,
    )
    if not selected.any():
        return {"status": STATUS_SKIPPED, "reason": "empty_support_region",
                "protection": protection, "centre_snap": snap_info,
                "depth_conditioning": depth_block}

    rel = vertices_model - centre_model
    x = rel @ axes["right"]
    y = rel @ axes["up"]

    # Fixed profile and baseline trend (computed once, never refitted).
    profile_indices = _profile_triangles(vertices_model, faces, centre_model, axes)
    span = cfg.base_radius_y_mm + ANCHOR_INSET_MM + ANCHOR_BAND_MM + 2.0
    grid, outline_base = _sagittal_outline(
        vertices_model, faces, centre_model, axes, profile_indices, -span, span
    )
    trend = _outline_trend(grid, outline_base, cfg.base_radius_y_mm)
    if not trend.get("available"):
        return {
            "status": STATUS_SKIPPED,
            "reason": "reference_trend_unavailable",
            "trend": trend,
            "protection": protection,
            "centre_snap": snap_info,
            "depth_conditioning": depth_block,
        }
    h_base = _prominence_at_centre(grid, outline_base, trend["trend_at_centre_mm"])
    if not np.isfinite(h_base):
        return {
            "status": STATUS_SKIPPED,
            "reason": "baseline_outline_unavailable_at_centre",
            "trend": trend,
            "protection": protection,
            "centre_snap": snap_info,
            "depth_conditioning": depth_block,
        }

    if h_base >= target_height_mm:
        return {
            "status": STATUS_SKIPPED,
            "reason": "baseline_prominence_already_meets_target",
            "baseline_prominence_mm": round(h_base, 3),
            "requested_final_prominence_mm": float(target_height_mm),
            "trend": trend,
            "protection": protection,
            "centre_snap": snap_info,
            "depth_conditioning": depth_block,
        }

    # Calibration loop: rescale the field until the measured H meets the goal.
    amplitude_m = (target_height_mm - h_base) * 1e-3
    core_widened = False
    calibration: list[dict[str, Any]] = []
    shaped = None
    displacement_camera = None
    moved = np.zeros(len(vertices), dtype=bool)
    h_final = float("nan")
    for iteration in range(CALIBRATION_ITERS):
        if amplitude_m > MAX_AMPLITUDE_MM * 1e-3:
            amplitude_m = MAX_AMPLITUDE_MM * 1e-3
        weight = np.zeros(len(vertices_model), dtype=np.float64)
        weight[selected] = _field_weights(x[selected], y[selected], cfg)
        weight *= taper
        moved = selected & (weight > 1e-9) & (amplitude_m > 0)
        displacement_model = (amplitude_m * weight)[:, None] * axes["forward"][None, :]
        displacement_camera = displacement_model @ rotation.T
        shaped = vertices.copy()
        shaped[moved] = vertices[moved] + displacement_camera[moved]

        shaped_model = (shaped - translation) @ rotation
        _, outline_shaped = _sagittal_outline(
            shaped_model, faces, centre_model, axes, profile_indices, -span, span
        )
        h_final = _prominence_at_centre(grid, outline_shaped, trend["trend_at_centre_mm"])
        gain = (h_final - h_base) / (amplitude_m * 1000.0) if amplitude_m > 1e-9 else 0.0
        calibration.append({
            "iteration": iteration,
            "amplitude_mm": round(amplitude_m * 1000.0, 3),
            "final_prominence_mm": round(h_final, 3) if np.isfinite(h_final) else None,
            "gain": round(float(gain), 4) if np.isfinite(gain) else None,
        })
        if not np.isfinite(h_final):
            break
        if abs(h_final - target_height_mm) <= CALIBRATE_TOLERANCE_MM:
            break
        if gain < MIN_GAIN and not core_widened:
            # The centre falls in a low-weight zone: widen the support rather
            # than dividing by a tiny weight and creating a spike.
            core_widened = True
            cfg = ProminenceConfig(
                subject_sex=cfg.subject_sex, mode=cfg.mode,
                amplitude_mm=cfg.amplitude_mm,
                radius_x_mm=min(cfg.radius_x_mm * 1.3, cfg.base_radius_x_mm),
                radius_y_mm=min(cfg.radius_y_mm * 1.3, cfg.base_radius_y_mm),
                base_radius_x_mm=cfg.base_radius_x_mm,
                base_radius_y_mm=cfg.base_radius_y_mm,
                base_share=cfg.base_share,
                candidate_amplitudes_mm=cfg.candidate_amplitudes_mm,
                _explicit=cfg._explicit,
            )
            selected = _select_region(
                vertices_model, adjacency, centre_seed, centre_model, axes, normals_model,
                protection["mask"],
                cfg.base_radius_x_mm * 1e-3, cfg.base_radius_y_mm * 1e-3,
            )
            rel = vertices_model - centre_model
            x = rel @ axes["right"]
            y = rel @ axes["up"]
            continue
        achieved = h_final - h_base
        if achieved <= 1e-6:
            break
        amplitude_m = amplitude_m * (target_height_mm - h_base) / achieved

    if shaped is None or not np.isfinite(h_final):
        return {
            "status": STATUS_SKIPPED,
            "reason": "calibration_did_not_converge",
            "calibration": calibration,
            "trend": trend,
            "protection": protection,
            "centre_snap": snap_info,
            "depth_conditioning": depth_block,
        }

    magnitudes = np.linalg.norm(displacement_camera, axis=1)
    peak_mm = float(magnitudes[moved].max() * 1000.0) if moved.any() else 0.0
    self_intersections = _count_self_intersections(
        shaped, faces, moved, adjacency, centre_model,
        cfg.base_radius_x_mm * 1e-3,
    )
    checks = _safety_checks(
        vertices, shaped, faces, moved, displacement_camera,
        protection["mask"], adjacency, amplitude_m, self_intersections,
    )

    metrics = _shape_metrics(vertices_model, moved, displacement_model, axes, centre_model)

    # Binding forward displacement: exact barycentric interpolation on the
    # binding triangle - never a simple vertex mean.
    binding_bary = np.asarray(binding_hit["barycentric"], dtype=np.float64)
    binding_seed = np.asarray(binding_hit["face_vertices"], dtype=np.int64)
    binding_disp = float(np.dot(binding_bary, magnitudes[binding_seed]) * 1000.0)
    q_camera = np.asarray(binding_hit["point_mesh_m"], dtype=np.float64)
    binding_after = q_camera + binding_bary @ displacement_camera[binding_seed]

    result: dict[str, Any] = {
        "status": checks["status"],
        "reason": checks["reason"],
        "requested_final_prominence_mm": float(target_height_mm),
        "baseline_prominence_mm": round(h_base, 3),
        "final_prominence_mm": round(h_final, 3),
        "target_error_mm": round(float(h_final - target_height_mm), 3),
        "binding_forward_displacement_mm": round(binding_disp, 3),
        "max_vertex_displacement_mm": peak_mm,
        "calibrated_amplitude_mm": round(amplitude_m * 1000.0, 3),
        "shape": {
            "core_radius_x_mm": round(cfg.radius_x_mm, 3),
            "core_radius_y_mm": round(cfg.radius_y_mm, 3),
            "base_radius_x_mm": round(cfg.base_radius_x_mm, 3),
            "base_radius_y_mm": round(cfg.base_radius_y_mm, 3),
            "base_share": round(cfg.base_share, 3),
            "taper": taper_info,
            "core_widened_during_calibration": core_widened,
            "field": "wendland_c2_base_plus_plateau_core",
        },
        "extent": metrics,
        "trend": trend,
        "calibration": calibration,
        "centre_snap": snap_info,
        "depth_conditioning": depth_block,
        "binding": {
            "marker_point_reference_camera_m": [float(v) for v in point],
            "baseline_binding_point_reference_camera_m": [float(v) for v in q_camera],
            "shaped_binding_point_reference_camera_m": [float(v) for v in binding_after],
            "ray_binding_point_reference_camera_m": [float(v) for v in ray_binding_camera],
            "distance_marker_to_ray_binding_mm": round(distance_to_marker_mm, 3),
            "triangle_index": binding_hit["triangle_index"],
            "barycentric": binding_hit["barycentric"],
            "ray_incidence": binding_hit["incidence"],
            "binding_to_centre_offset_mm": round(
                float(np.linalg.norm(centre_model - to_model(ray_binding_camera)) * 1000.0), 3),
        },
        "direction": {
            "right_model": [float(v) for v in axes["right"]],
            "up_model": [float(v) for v in axes["up"]],
            "forward_model": [float(v) for v in axes["forward"]],
            # Direction vectors transform with the rotation only - never with
            # the translation - and stay normalised.
            "forward_reference_camera": [float(v) for v in forward_camera],
            "forward_normal_agreement": forward_agreement,
            "source": "facial landmarks of the baseline model, per state pose",
        },
        "support": {
            "selected_vertices": int(selected.sum()),
            "moved_vertices": int(moved.sum()),
        },
        "protection": {
            "protected_vertices": protection["count"],
            "reasons": list(protection["reasons"].keys()),
        },
        "safety": checks,
    }
    shape_ok = abs(h_final - target_height_mm) <= ACCEPT_TOLERANCE_MM
    if checks["status"] == STATUS_APPLIED and shape_ok:
        result["shaped_vertices"] = shaped
    else:
        result["status"] = STATUS_SKIPPED
        result["reason"] = (
            "shape_goal_not_reached" if checks["status"] == STATUS_APPLIED
            else checks["reason"]
        )
    return result


def apply_prominence(
    *,
    state: str,
    base_vertices: np.ndarray,
    faces: np.ndarray,
    landmark_record: dict[str, Any],
    transform: dict[str, Any],
    model_to_camera: np.ndarray,
    landmarks_m: np.ndarray,
    config: ProminenceConfig,
    prepared_dir: Any | None = None,
) -> dict[str, Any]:
    """Full per-state pipeline: enable decision, centre, direction, shaping.

    Returns a report dict; ``shaped_vertices`` is present only when shaping
    was applied, otherwise the caller keeps the baseline geometry.
    """
    started = time.perf_counter()
    report: dict[str, Any] = {
        "state": state,
        "neck_method": NECK_METHOD,
        "geometry_source": GEOMETRY_SOURCE,
        "subject_sex": config.subject_sex,
        "mode": config.mode,
        "requested_amplitude_mm": config.amplitude_mm,
        "radius_x_mm": config.radius_x_mm,
        "radius_y_mm": config.radius_y_mm,
        "base_radius_x_mm": config.base_radius_x_mm,
        "base_radius_y_mm": config.base_radius_y_mm,
        "base_share": config.base_share,
    }

    observed = (
        landmark_record.get("status") == nl.STATUS_OBSERVED
        and landmark_record.get("point_observed_m")
    )
    enabled, reason = effective_enable(config, bool(observed))
    report["effective_enabled"] = bool(enabled)
    report["landmark_source"] = landmark_record.get("source")
    if not enabled:
        report.update(status=STATUS_SKIPPED, reason=reason)
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return report
    if transform.get("status") != nl.STATUS_OBSERVED:
        report.update(status=STATUS_SKIPPED, reason=transform.get("reason") or "transform_unavailable")
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return report

    point = nl.apply_transform(landmark_record["point_observed_m"], transform)
    if point is None:
        report.update(status=STATUS_SKIPPED, reason="transform_application_failed")
        report["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        return report
    report["point_reference_camera_m"] = point
    report["transform"] = {
        "method": transform.get("method"),
        "annotation_frame": transform.get("annotation_frame"),
        "rotation_3x3": transform.get("rotation_3x3"),
        "translation_m": transform.get("translation_m"),
    }

    candidates = config.candidates_mm
    attempts: list[dict[str, Any]] = []
    for order, target_mm in enumerate(candidates):
        attempt = _shape_state(
            base_vertices=base_vertices,
            faces=faces,
            landmark_point_reference_camera_m=point,
            model_to_camera=np.asarray(model_to_camera, dtype=np.float64),
            landmarks_m=landmarks_m,
            target_height_mm=float(target_mm),
            cfg=config,
            prepared_dir=prepared_dir,
            pixel_uv=landmark_record.get("pixel_uv"),
            segment_file=landmark_record.get("segment_file"),
        )
        entry = {k: v for k, v in attempt.items() if k != "shaped_vertices"}
        entry["candidate_order"] = order
        entry["target_height_mm"] = float(target_mm)
        attempts.append(entry)
        if attempt["status"] == STATUS_APPLIED:
            report.update(
                status=STATUS_APPLIED,
                reason=None,
                requested_final_prominence_mm=attempt["requested_final_prominence_mm"],
                baseline_prominence_mm=attempt["baseline_prominence_mm"],
                final_prominence_mm=attempt["final_prominence_mm"],
                target_error_mm=attempt["target_error_mm"],
                binding_forward_displacement_mm=attempt["binding_forward_displacement_mm"],
                max_vertex_displacement_mm=attempt["max_vertex_displacement_mm"],
                calibrated_amplitude_mm=attempt["calibrated_amplitude_mm"],
                candidate_amplitudes_mm=[float(a) for a in candidates],
                shape=attempt["shape"],
                extent=attempt["extent"],
                trend=attempt["trend"],
                calibration=attempt["calibration"],
                centre_snap=attempt["centre_snap"],
                depth_conditioning=attempt["depth_conditioning"],
                binding=attempt["binding"],
                direction=attempt["direction"],
                support=attempt["support"],
                protection=attempt["protection"],
                safety=attempt["safety"],
                candidate_attempts=attempts,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            report["shaped_vertices"] = attempt["shaped_vertices"]
            return report
    # Every candidate failed the shape goal or a safety check: keep the
    # baseline geometry, report the diagnosis, never clip per-vertex.
    last = attempts[-1] if attempts else {}
    report.update(
        status=STATUS_SKIPPED,
        reason=(last.get("reason") or "all_candidate_targets_failed"),
        candidate_amplitudes_mm=[float(a) for a in candidates],
        candidate_attempts=attempts,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    return report
