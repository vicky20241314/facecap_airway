"""Marker-guided neck-front prominence on the 0.5.2 template surface.

This is *label-localised template shaping*, not a measured neck reconstruction:

* the laryngeal landmark (ArUco sticker centre or a manually clicked point)
  supplies one surface location per expression state;
* that location is bound to the delivered baseline mesh by an exact camera-ray
  first hit, giving the shaping centre ``q``;
* a compactly supported Wendland C2 displacement field moves nearby front-neck
  vertices a few millimetres along the pose-following model-forward direction;
* face, lips, jaw, chin, ears, the back of the neck and the bottom cut boundary
  are protected and strictly unchanged.

No neck refit, measured surface, patch, stitching or whole-neck ICP exists in
this stage.  All displacements are computed from the undeformed baseline
vertices, so repeated exports are idempotent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from head_rgbd import neck_landmark as nl

NECK_METHOD = "marker_guided_template_prominence"
GEOMETRY_SOURCE = "template_plus_procedural_deformation"

# ibug-68 landmark indices used for the local direction frame.
EYE_OUTER_RIGHT = 36
EYE_OUTER_LEFT = 45
NOSE_BRIDGE_TOP = 27
NOSE_TIP = 30
CHIN = 8

# Wendland C2 on the unit interval: w(r) = (1-r)^4 (4r+1), support r < 1.
SELECT_MARGIN = 1.3          # BFS may expand through vertices up to this r
FRONT_NORMAL_MIN = 0.25      # vertex counts as front-neck skin above this dot
BIND_NORMAL_MIN = 0.35       # binding-site surface must agree with forward
BIND_MAX_DISTANCE_MM = 25.0  # ray-hit point may not drift this far from marker
FACE_PROTECT_MM = 30.0       # proximity of any facial landmark protects a vertex
LATERAL_PROTECT_MM = 45.0    # |x| beyond this (model frame) is ear/side, protected
BOTTOM_PROTECT_MM = 8.0      # bottom cut band, protected
TAPER_RINGS = 3              # graph rings over which the taper ramps 0 -> 1
MAX_SLOPE = 1.5              # displacement/edge-length slope guard

STATUS_APPLIED = "applied"
STATUS_SKIPPED = "skipped"


def wendland_c2(r: np.ndarray | float) -> np.ndarray | float:
    """(1-r)^4 (4r+1) for r<1, zero otherwise; C2 at the support boundary."""
    r = np.asarray(r, dtype=np.float64)
    inside = r < 1.0
    one_minus = np.clip(1.0 - r, 0.0, None)
    values = one_minus ** 4 * (4.0 * r + 1.0)
    return np.where(inside, values, 0.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProminenceConfig:
    subject_sex: str = "unspecified"          # never inferred, user-declared only
    mode: str = "auto"                        # auto | on | off
    amplitude_mm: float = 3.0
    radius_x_mm: float = 12.0
    radius_y_mm: float = 18.0
    candidate_amplitudes_mm: tuple[float, ...] = (2.0, 3.0, 4.0)

    @property
    def candidates_mm(self) -> tuple[float, ...]:
        """Amplitudes to try, smallest natural bump first.

        An explicit user amplitude is the single candidate (with automatic
        safety reductions inside the shaping loop); the default compares only
        the few trial amplitudes, no parameter search.
        """
        if self._explicit:
            return (float(self.amplitude_mm),)
        return tuple(sorted({float(a) for a in self.candidate_amplitudes_mm}))

    _explicit: bool = field(default=False, repr=False, compare=False)


def resolve_configuration(defaults: dict[str, Any], args: Any | None = None) -> ProminenceConfig:
    """Unified configuration: config/defaults.json first, CLI overrides second."""
    section = dict(defaults.get("neck_prominence") or {})
    supplied = getattr(args, "neck_prominence_mm", None)
    explicit = supplied is not None
    amplitude = float(supplied) if explicit else float(section.get("amplitude_mm", 3.0))
    return ProminenceConfig(
        subject_sex=str(
            getattr(args, "subject_sex", None)
            or section.get("subject_sex", "unspecified")
        ),
        mode=str(
            getattr(args, "neck_prominence", None)
            or section.get("mode", "auto")
        ),
        amplitude_mm=amplitude,
        radius_x_mm=float(
            getattr(args, "neck_prominence_radius_x_mm", None)
            or section.get("radius_x_mm", 12.0)
        ),
        radius_y_mm=float(
            getattr(args, "neck_prominence_radius_y_mm", None)
            or section.get("radius_y_mm", 18.0)
        ),
        candidate_amplitudes_mm=tuple(
            float(v) for v in section.get("candidate_amplitudes_mm", (2.0, 3.0, 4.0))
        ),
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
        tree_points = landmarks_model
        # dense distance: chunked to bound memory
        dist = np.empty(len(vertices_model), dtype=np.float64)
        chunk = 4096
        for start in range(0, len(vertices_model), chunk):
            block = vertices_model[start:start + chunk]
            d = np.linalg.norm(block[:, None, :] - tree_points[None, :, :], axis=2)
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


def _graph_distance_rings(adjacency: list[np.ndarray], sources: np.ndarray) -> np.ndarray:
    """BFS ring distance from the protected set over the full adjacency."""
    distance = np.full(len(adjacency), np.iinfo(np.int32).max, dtype=np.int32)
    frontier = np.asarray(sources, dtype=np.int64)
    distance[frontier] = 0
    ring = 0
    while len(frontier):
        ring += 1
        nxt = np.unique(np.concatenate([adjacency[v] for v in frontier]))
        nxt = nxt[distance[nxt] == np.iinfo(np.int32).max]
        if not len(nxt):
            break
        distance[nxt] = ring
        frontier = nxt
    return distance


def _select_region(
    vertices: np.ndarray,
    adjacency: list[np.ndarray],
    seed_faces: np.ndarray,
    q: np.ndarray,
    axes: dict[str, np.ndarray],
    normals: np.ndarray,
    protected: np.ndarray,
    radius_x_m: float,
    radius_y_m: float,
) -> np.ndarray:
    """Connected front-neck region around q inside the elliptical support.

    Selection is a breadth-first walk over the mesh graph, not a Euclidean
    ball: back-facing, protected or out-of-ellipse vertices stop the walk, so
    the far side of a thin wall or the chin can never be selected.
    """
    rel = vertices - q
    x = rel @ axes["right"]
    y = rel @ axes["up"]
    r = np.sqrt((x / radius_x_m) ** 2 + (y / radius_y_m) ** 2)
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


def _safety_checks(
    base_vertices: np.ndarray,
    shaped_vertices: np.ndarray,
    faces: np.ndarray,
    moved: np.ndarray,
    displacement: np.ndarray,
    protected: np.ndarray,
    adjacency: list[np.ndarray],
    amplitude_m: float,
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
        return {"status": STATUS_SKIPPED, "reason": "non_finite_coordinates", "problems": problems,
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

    # Spike / abnormal-long-edge guard on affected faces.
    for axis in range(3):
        edge_b = np.linalg.norm(tri_b[:, (axis + 1) % 3] - tri_b[:, axis], axis=1)
        edge_a = np.linalg.norm(tri_a[:, (axis + 1) % 3] - tri_a[:, axis], axis=1)
        grown = edge_a > np.maximum(edge_b * 1.5, edge_b + 2e-3)
        if grown.any():
            problems.append("abnormally_long_edge")
            break

    # Support-boundary continuity: the applied field must be zero outside the
    # support, adjacent magnitudes must not jump beyond a bounded slope, and no
    # spike may exceed the requested amplitude.
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
    status = STATUS_APPLIED if not problems else STATUS_SKIPPED
    return {
        "status": status,
        "reason": None if not problems else ";".join(problems),
        "problems": problems,
        "affected_faces": int(affected.sum()),
        "moved_vertices": int(moved.sum()),
    }


def _shape_state(
    *,
    base_vertices: np.ndarray,
    faces: np.ndarray,
    landmark_point_reference_camera_m: list[float],
    model_to_camera: np.ndarray,
    landmarks_m: np.ndarray,
    amplitude_m: float,
    radius_x_m: float,
    radius_y_m: float,
) -> dict[str, Any]:
    """One candidate shaping attempt for one state. Raises nothing."""
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
    q_camera = np.asarray(hit["point_mesh_m"], dtype=np.float64)
    distance_to_marker_mm = float(np.linalg.norm(q_camera - point) * 1000.0)
    if distance_to_marker_mm > BIND_MAX_DISTANCE_MM:
        return {
            "status": STATUS_SKIPPED,
            "reason": "baseline_binding_too_far_from_marker",
            "distance_marker_to_binding_mm": distance_to_marker_mm,
            "protection": protection,
        }
    q_model = to_model(q_camera)
    # Direction sanity: the local surface at q must face the model-forward axis.
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

    adjacency = vertex_adjacency(faces, len(vertices))
    taper_distance = _graph_distance_rings(adjacency, np.flatnonzero(protection["mask"]))
    taper = np.clip(taper_distance / float(TAPER_RINGS), 0.0, 1.0)

    selected = _select_region(
        vertices_model, adjacency, q_seed, q_model, axes, normals_model,
        protection["mask"], radius_x_m, radius_y_m,
    )
    if not selected.any():
        return {"status": STATUS_SKIPPED, "reason": "empty_support_region",
                "protection": protection}

    rel = vertices_model - q_model
    x = rel @ axes["right"]
    y = rel @ axes["up"]
    r = np.sqrt((x / radius_x_m) ** 2 + (y / radius_y_m) ** 2)
    weight = np.asarray(wendland_c2(r), dtype=np.float64)
    moved = selected & (weight > 0)
    displacement_model = (
        amplitude_m * weight * taper
    )[:, None] * axes["forward"][None, :]
    displacement_camera = displacement_model @ rotation.T
    shaped = vertices.copy()
    shaped[moved] = vertices[moved] + displacement_camera[moved]

    checks = _safety_checks(
        vertices, shaped, faces, moved, displacement_camera,
        protection["mask"], adjacency, amplitude_m,
    )

    peak_mm = 0.0
    if moved.any():
        magnitudes = np.linalg.norm(displacement_camera, axis=1)
        peak_mm = float(magnitudes[moved].max() * 1000.0)
    result = {
        "status": checks["status"],
        "reason": checks["reason"],
        "binding": {
            "marker_point_reference_camera_m": [float(v) for v in point],
            "baseline_binding_point_reference_camera_m": [float(v) for v in q_camera],
            "baseline_binding_point_model_m": [float(v) for v in q_model],
            "triangle_index": hit["triangle_index"],
            "barycentric": hit["barycentric"],
            "distance_marker_to_binding_mm": distance_to_marker_mm,
            "ray_incidence": hit["incidence"],
        },
        "direction": {
            "right_model": [float(v) for v in axes["right"]],
            "up_model": [float(v) for v in axes["up"]],
            "forward_model": [float(v) for v in axes["forward"]],
            "forward_reference_camera": [float(v) for v in to_camera(axes["forward"])],
            "forward_normal_agreement": forward_agreement,
            "source": "facial landmarks of the baseline model, per state pose",
        },
        "support": {
            "selected_vertices": int(selected.sum()),
            "moved_vertices": int(moved.sum()),
            "radius_x_mm": round(radius_x_m * 1000.0, 6),
            "radius_y_mm": round(radius_y_m * 1000.0, 6),
        },
        "protection": {
            "protected_vertices": protection["count"],
            "reasons": list(protection["reasons"].keys()),
        },
        "safety": checks,
        "requested_amplitude_mm": amplitude_m * 1000.0,
        "actual_amplitude_mm": peak_mm,
    }
    if checks["status"] == STATUS_APPLIED:
        result["shaped_vertices"] = shaped
        binding_after = q_camera + displacement_camera[
            np.asarray(hit["face_vertices"], dtype=np.int64)
        ].mean(axis=0)
        result["binding"]["shaped_binding_point_reference_camera_m"] = [
            float(v) for v in binding_after
        ]
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
    for order, amplitude_mm in enumerate(candidates):
        attempt = _shape_state(
            base_vertices=base_vertices,
            faces=faces,
            landmark_point_reference_camera_m=point,
            model_to_camera=np.asarray(model_to_camera, dtype=np.float64),
            landmarks_m=landmarks_m,
            amplitude_m=amplitude_mm * 1e-3,
            radius_x_m=config.radius_x_mm * 1e-3,
            radius_y_m=config.radius_y_mm * 1e-3,
        )
        entry = {k: v for k, v in attempt.items() if k != "shaped_vertices"}
        entry["candidate_order"] = order
        attempts.append(entry)
        if attempt["status"] == STATUS_APPLIED:
            report.update(
                status=STATUS_APPLIED,
                reason=None,
                actual_amplitude_mm=attempt["actual_amplitude_mm"],
                candidate_amplitudes_mm=[float(a) for a in candidates],
                selected_amplitude_mm=float(amplitude_mm),
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
    # Every candidate unsafe: keep the smallest amplitude's diagnosis, and
    # never hand-pick per-vertex clipping to force a pass.
    last = attempts[-1] if attempts else {}
    report.update(
        status=STATUS_SKIPPED,
        reason=(last.get("reason") or "all_candidate_amplitudes_failed_safety"),
        candidate_amplitudes_mm=[float(a) for a in candidates],
        candidate_attempts=attempts,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )
    return report
