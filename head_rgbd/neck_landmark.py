"""Minimal laryngeal-landmark localisation ported from the marker experiments.

Only the localisation capability is kept:

* manual click -> depth-checked backprojected surface point (``query_observed_point``);
* frame binding verification (``verify_frame_binding``);
* the annotation-frame -> reference-camera transform (``resolve_reference_transform``),
  same-frame identity, an explicit transform supplied with the record, or the
  neck-region optical-flow chain when the prepared sequence carries one.

No neck refit, measured surface, patch, stitching or whole-neck ICP is ported
here; a laryngeal landmark record is a single surface point and nothing else.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from head_rgbd import neck_anatomy as na
from head_rgbd import neck_direct as nd
from head_rgbd import neck_prominence as npn

NAME = "laryngeal_prominence"
DEFINITION = "人工指定的喉结体表位置"
DEFINITION_SCOPE = (
    "体表人工指定点；不是内部甲状软骨的直接测量，也不是独立几何真值"
)
OBSERVATION_FRAME_COORDINATES = (
    "annotation frame camera X_right_Y_up_Z_backward; metres"
)
COORDINATE_CONVENTION = "X_right_Y_up_Z_backward"
STATES = ("closed", "open")
STATUS_OBSERVED = "observed"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"
SOURCE_MANUAL = "manual_click"
SOURCE_ARUCO = "aruco_marker"
SOURCES = (SOURCE_ARUCO, SOURCE_MANUAL)

# Landmark sidecar handed to reconstruct/export (one record per state).
LANDMARKS_SCHEMA = "head_rgbd.neck_landmarks.v1"
LANDMARKS_FILE = "neck_landmarks.json"

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 1500
LOCAL_RADIUS_PX = 4
LOCAL_MIN_VALID_FRACTION = 0.60
LOCAL_MAX_CENTER_JUMP_MM = 12.0
LOCAL_MAX_SPREAD_MM = 25.0
EDGE_MAX_SPREAD_MM = 60.0
MAX_TEMPORAL_GAP_S = 0.2


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def array_sha256(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array)
    return hashlib.sha256(data.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Manual click -> observed surface point
# ---------------------------------------------------------------------------

def neck_roi(image_width: int, image_height: int, face_bbox: list | tuple | None) -> dict[str, Any]:
    """Rectangular neck search region below the detected face box."""
    if face_bbox is None:
        return {
            "status": "unavailable",
            "reason": "no_face_bbox",
            "x0": 0, "y0": 0, "x1": image_width, "y1": image_height,
            "full_image": True,
        }
    x, y, w, h = [float(v) for v in face_bbox]
    x0 = max(0, int(x - 0.25 * w))
    x1 = min(image_width, int(x + 1.25 * w))
    # The laryngeal prominence sits directly below the chin; a 10%-of-box
    # margin pushed this ROI onto the collar and forced mis-localised clicks.
    y0 = min(image_height, int(y + h + 0.02 * h))
    y1 = min(image_height, int(y + h + 2.4 * h))
    if x1 <= x0 or y1 <= y0:
        return {
            "status": "unavailable",
            "reason": "degenerate_roi",
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "full_image": False,
        }
    return {"status": "available", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "full_image": False}


def pixel_in_roi(uv: tuple[float, float], roi: dict[str, Any] | None) -> bool:
    if not roi or roi.get("full_image"):
        return True
    return (
        roi["x0"] <= uv[0] <= roi["x1"] and roi["y0"] <= uv[1] <= roi["y1"]
    )


def _smooth_window(depth: np.ndarray, y: int, x: int, radius: int) -> np.ndarray:
    height, width = depth.shape[:2]
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    return depth[y0:y1, x0:x1]


def query_observed_point(
    uv: tuple[float, float] | list[float],
    depth: np.ndarray,
    intrinsics: np.ndarray | list[float],
    *,
    roi: dict[str, Any] | None = None,
    radius_px: int = LOCAL_RADIUS_PX,
) -> dict[str, Any]:
    """Backproject one manually clicked pixel using the project's ``backproject``.

    The centre pixel must carry valid, finite depth.  No distant pixel, fixed
    depth or template point is ever substituted.  A tight local support window
    is used only to *reject* holes, depth jumps and occlusion edges.
    """
    from head_rgbd.prepare import backproject

    result: dict[str, Any] = {
        "status": STATUS_FAILED,
        "reason": None,
        "pixel_uv": [float(uv[0]), float(uv[1])],
        "point_observed_m": None,
        "coordinate_system": OBSERVATION_FRAME_COORDINATES,
        "unit": "m",
        "depth": {
            "method": "center_pixel_with_local_consistency_check",
            "query_reuses": "head_rgbd.prepare.backproject",
            "support_region_limited": True,
            "support_radius_px": int(radius_px),
            "local_plane_fit_used": False,
            "substitute_used": False,
            "valid": False,
        },
    }
    u, v = float(uv[0]), float(uv[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        result["reason"] = "nonfinite_pixel"
        return result
    if depth.ndim != 2:
        result["reason"] = "depth_is_not_an_image"
        return result
    height, width = depth.shape
    if not (0 <= u < width and 0 <= v < height):
        result["reason"] = "pixel_outside_image"
        return result
    if not pixel_in_roi((u, v), roi):
        result["reason"] = "pixel_outside_neck_region"
        return result

    intr = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if intr.shape != (4,) or not np.isfinite(intr).all() or min(intr[:2]) <= 0:
        result["reason"] = "invalid_intrinsics"
        return result

    x, y = int(round(u)), int(round(v))
    x, y = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
    center = float(depth[y, x])
    if not np.isfinite(center) or not (DEPTH_MIN_MM < center < DEPTH_MAX_MM):
        result["reason"] = "center_depth_invalid"
        result["depth"].update(center_depth_mm=center if np.isfinite(center) else None)
        return result

    window = _smooth_window(depth, y, x, radius_px)
    valid = np.isfinite(window) & (window > DEPTH_MIN_MM) & (window < DEPTH_MAX_MM)
    fraction = float(valid.mean())
    if fraction < LOCAL_MIN_VALID_FRACTION:
        result["reason"] = "depth_local_holes"
        result["depth"].update(valid_fraction=fraction)
        return result
    values = window[valid]
    median = float(np.median(values))
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    jump = float(abs(center - median))
    edge_window = _smooth_window(depth, y, x, radius_px * 2)
    edge_valid = np.isfinite(edge_window) & (edge_window > DEPTH_MIN_MM) & (edge_window < DEPTH_MAX_MM)
    edge_spread = (
        float(np.percentile(edge_window[edge_valid], 95) - np.percentile(edge_window[edge_valid], 5))
        if edge_valid.any() else None
    )
    result["depth"].update(
        valid=True,
        center_depth_mm=center,
        local_median_depth_mm=median,
        local_valid_fraction=fraction,
        local_spread_p90_p10_mm=spread,
        center_vs_median_jump_mm=jump,
        edge_spread_p95_p5_mm=edge_spread,
        depth_min_mm=DEPTH_MIN_MM,
        depth_max_mm=DEPTH_MAX_MM,
    )
    if jump > LOCAL_MAX_CENTER_JUMP_MM:
        result["reason"] = "depth_local_jump"
        return result
    if spread > LOCAL_MAX_SPREAD_MM:
        result["reason"] = "depth_local_discontinuity"
        return result
    if edge_spread is not None and edge_spread > EDGE_MAX_SPREAD_MM:
        result["reason"] = "depth_edge_or_occlusion"
        return result

    xyz, ok = backproject(np.asarray([[u, v]], dtype=np.float64), depth, intr)
    if not bool(ok[0]) or not np.isfinite(xyz[0]).all():
        result["reason"] = "backprojection_invalid"
        return result
    result["status"] = STATUS_OBSERVED
    result["point_observed_m"] = [float(value) for value in xyz[0]]
    result["depth"]["intrinsics_fx_fy_cx_cy"] = [float(value) for value in intr]
    return result


# ---------------------------------------------------------------------------
# Frame binding verification
# ---------------------------------------------------------------------------

def verify_frame_binding(
    record: dict[str, Any],
    *,
    frame_id: str,
    segment_file: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[bool, str | None]:
    """Reject an annotation that is not bound to this exact saved observation."""
    binding = record.get("frame_binding") or {}
    if record.get("frame_id") != frame_id:
        return False, "annotation_frame_id_mismatch"
    if record.get("segment_file") != segment_file:
        return False, "annotation_segment_file_mismatch"
    if int(record.get("source_frame", -1)) != int(binding.get("source_frame", -2)):
        return False, "annotation_source_frame_mismatch"
    if list(record.get("image_size") or []) != [int(depth.shape[1]), int(depth.shape[0])]:
        return False, "annotation_image_size_mismatch"
    if binding.get("rgb_sha256") and binding["rgb_sha256"] != array_sha256(rgb):
        return False, "annotation_rgb_frame_mismatch"
    if binding.get("depth_sha256") and binding["depth_sha256"] != array_sha256(depth):
        return False, "annotation_depth_frame_mismatch"
    if binding.get("intrinsics_sha256") and binding["intrinsics_sha256"] != array_sha256(intrinsics):
        return False, "annotation_intrinsics_mismatch"
    if not np.allclose(
        np.asarray(binding.get("intrinsics_fx_fy_cx_cy", []), dtype=float),
        np.asarray(intrinsics, dtype=float).reshape(-1),
        rtol=0, atol=1e-9,
    ):
        return False, "annotation_intrinsics_value_mismatch"
    return True, None


# ---------------------------------------------------------------------------
# Annotation frame -> delivery reference frame
# ---------------------------------------------------------------------------

def _rotation_translation(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    motion = row["neck_overlap"]
    return (
        np.asarray(motion["previous_to_current_R"], dtype=np.float64),
        np.asarray(motion["previous_to_current_t"], dtype=np.float64),
    )


def _chain(rows: list[dict[str, Any]], start: int, end: int) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    """Compose ``p_next = R @ p_previous + t`` from start to end (end > start)."""
    rotation = np.eye(3)
    translation = np.zeros(3)
    links = []
    for index in range(start + 1, end + 1):
        current = rows[index]
        previous = rows[index - 1]
        motion = current.get("neck_overlap") or {}
        if (
            current.get("status") != "candidate"
            or previous.get("status") != "candidate"
            or current.get("segment") != previous.get("segment")
            or current.get("tracking_barrier")
            or motion.get("status") != "tracked"
            or "previous_to_current_R" not in motion
            or "previous_to_current_t" not in motion
        ):
            return None
        try:
            gap = float(current["timestamp_s"]) - float(previous["timestamp_s"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 < gap < MAX_TEMPORAL_GAP_S:
            return None
        r, t = _rotation_translation(current)
        if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
            return None
        median_m = float(motion.get("median_m", 0.0))
        if not np.isfinite(median_m) or median_m > 0.008:
            return None
        rotation = r @ rotation
        translation = r @ translation + t
        links.append({
            "from": previous["id"],
            "to": current["id"],
            "segment": current.get("segment"),
            "median_residual_m": median_m,
            "timestamp_gap_s": round(gap, 6),
        })
    return rotation, translation, links


def resolve_reference_transform(
    record: dict[str, Any],
    rows: list[dict[str, Any]],
    reference_frame_id: str,
) -> dict[str, Any]:
    """Validate and record a neck-specific transform between two frame cameras.

    Supported, in order: an explicit transform stored with the record (source
    ``explicit_transform``, e.g. produced and recorded by an ArUco capture
    helper), the exact same frame (identity), and the neck-region optical-flow
    chain from the prepared sequence.  A head/FLAME rigid registration is never
    substituted here.
    """
    base = {
        "status": STATUS_FAILED,
        "reason": None,
        "method": None,
        "direction": None,
        "unit": "m",
        "applies_to": "neck surface observation only",
        "head_rigid_registration_used": False,
        "independent_extrinsics": False,
        "annotation_frame": record.get("frame_id"),
        "reference_frame": reference_frame_id,
        "rotation_3x3": None,
        "translation_m": None,
        "links": [],
    }
    frame_id = record.get("frame_id")
    if record.get("status") != STATUS_OBSERVED or not record.get("point_observed_m"):
        base["reason"] = "annotation_not_observed"
        return base

    explicit = record.get("explicit_transform")
    if explicit:
        rotation = np.asarray(explicit.get("rotation_3x3"), dtype=np.float64)
        translation = np.asarray(explicit.get("translation_m"), dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not (
            np.isfinite(rotation).all() and np.isfinite(translation).all()
        ):
            base["reason"] = "explicit_transform_invalid"
            return base
        base.update(
            status=STATUS_OBSERVED,
            method="explicit_transform_from_record",
            direction="p_reference_camera = R @ p_annotation_camera + t",
            rotation_3x3=rotation.tolist(),
            translation_m=translation.tolist(),
            scope="transform supplied with the landmark record; provenance recorded there",
        )
        return base

    ids = [row.get("id") for row in rows]
    if frame_id not in ids or reference_frame_id not in ids:
        base["reason"] = "annotation_or_reference_frame_absent_from_sequence"
        return base
    start, end = ids.index(frame_id), ids.index(reference_frame_id)
    if rows[start].get("status") != "candidate":
        base["reason"] = "annotated_frame_excluded_from_prepared_candidates"
        return base
    if rows[end].get("status") != "candidate":
        base["reason"] = "reference_frame_excluded_from_prepared_candidates"
        return base
    if start == end:
        base.update(
            status=STATUS_OBSERVED,
            method="same_frame_observation",
            direction="p_reference_camera = p_observation_camera",
            rotation_3x3=np.eye(3).tolist(),
            translation_m=[0.0, 0.0, 0.0],
            scope="annotation and delivery share the exact same RGB-D frame",
            note="no head or neck registration is needed or applied",
        )
        return base
    if rows[start].get("segment") != rows[end].get("segment"):
        base["reason"] = "annotation_and_reference_frames_are_in_different_segments"
        return base
    if start < end:
        composed = _chain(rows, start, end)
        if composed is None:
            base["reason"] = "neck_frame_transform_unavailable"
            return base
        rotation, translation, links = composed
    else:
        composed = _chain(rows, end, start)
        if composed is None:
            base["reason"] = "neck_frame_transform_unavailable"
            return base
        forward_r, forward_t, links = composed
        inverse = np.linalg.inv(forward_r)
        rotation = inverse
        translation = -inverse @ forward_t
        links = list(reversed(links))
    base.update(
        status=STATUS_OBSERVED,
        method="neck_region_optical_flow_depth_rigid_chain",
        direction="p_reference_camera = R @ p_annotation_camera + t",
        rotation_3x3=rotation.tolist(),
        translation_m=translation.tolist(),
        links=links,
        scope=(
            "contiguous frames of one capture segment; neck-mask optical flow and "
            "backprojected neck depth only; no cross-gap neck identity registration"
        ),
        note="R/t are in metres; each link was accepted with median residual < 8 mm",
    )
    return base


def apply_transform(point_m: list[float], transform: dict[str, Any]) -> list[float] | None:
    rotation = np.asarray(transform.get("rotation_3x3"), dtype=np.float64)
    translation = np.asarray(transform.get("translation_m"), dtype=np.float64)
    point = np.asarray(point_m, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,) or point.shape != (3,):
        return None
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all() and np.isfinite(point).all()):
        return None
    return [float(value) for value in (rotation @ point + translation)]


# ---------------------------------------------------------------------------
# Landmark sidecar file (neck_landmarks.json)
# ---------------------------------------------------------------------------

def validate_marker_position(point_reference_m: list[float], *, state: str,
                             V: np.ndarray, F: np.ndarray, ax: dict,
                             curve, h_chin: float,
                             p: dict[str, Any] | None = None) -> dict[str, Any]:
    """Subject-space front-neck position gate for the marker centre.

    The marker is only allowed to drive C when it lies on the subject's anterior
    neck: ``0.35 <= t <= t_gate + 0.025``, ``|theta| <= 20 deg``, the nearest
    surface vertex still faces up/outward (the same normal threshold used by the
    larynx prior), and the point is not far from the subject surface.  A point on
    the jaw underside, clavicle or shoulder fails with a named reason; it is
    never clamped into range.
    """
    p = dict(p or {})
    result: dict[str, Any] = {
        "status": STATUS_FAILED, "reason": None, "state": state,
        "point_reference_m": [float(v) for v in point_reference_m],
        "t_marker": None, "theta_marker_deg": None, "t_gate": None,
        "t_center_used": None, "theta_used_deg": 0.0,
        "source": SOURCE_ARUCO,
        "t_range": [0.35, None], "theta_limit_deg": 20.0,
        "checks": {},
    }
    note = point_reference_m
    if not isinstance(note, (list, tuple)) or len(note) != 3 or not np.isfinite(note).all():
        result["reason"] = "marker_point_not_finite"
        return result
    gate, profile = na.anterior_gate_top(V, F, ax, curve, h_chin, p)
    t_hi = float(gate) + 0.025
    point_mm = np.asarray(note, dtype=np.float64) * 1000.0
    h, th, r, t = nd.axial_mm(point_mm[None, :], ax, curve, h_chin)
    t = float(t[0])
    theta_deg = float(np.degrees(th[0]))
    t = float(t)
    # nearest surface vertex and its outward normal
    Vmm = np.asarray(V, dtype=np.float64) * 1000.0
    tree = cKDTree(Vmm)
    distance_mm, index = tree.query(point_mm, k=1)
    normals = npn.vertex_normals(V, F)
    dn = float(normals[int(index)] @ np.asarray(ax["axis"], float))
    max_distance = float(p.get("marker_max_surface_distance_mm", 30.0))
    gate_dn = float(p.get("larynx_prior_gate_dn", -0.30))
    checks = {
        "t_in_range": bool(0.35 <= t <= t_hi),
        "theta_in_range": bool(abs(theta_deg) <= 20.0),
        "normal_gate": bool(dn > gate_dn),
        "near_surface": bool(float(distance_mm) <= max_distance),
    }
    result.update({
        "t_marker": t, "theta_marker_deg": theta_deg, "t_gate": float(gate),
        "t_center_used": t, "theta_used_deg": 0.0,
        "t_range": [0.35, t_hi],
        "nearest_surface_distance_mm": float(distance_mm),
        "nearest_vertex_index": int(index),
        "nearest_normal_dn": dn,
        "gate_dn_threshold": gate_dn,
        "checks": checks,
        "gate_profile": profile,
    })
    if not checks["t_in_range"]:
        result["reason"] = ("marker_t_below_0_35" if t < 0.35
                            else "marker_t_above_gate")
        return result
    if not checks["theta_in_range"]:
        result["reason"] = "marker_theta_out_of_range"
        return result
    if not checks["normal_gate"]:
        result["reason"] = "marker_normal_gate_failed"
        return result
    if not checks["near_surface"]:
        result["reason"] = "marker_far_from_subject_surface"
        return result
    result["status"] = STATUS_OBSERVED
    return result


def _validate_record(record: Any, state: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(record, dict):
        return None, f"{state}_record_not_an_object"
    status = record.get("status")
    if status not in (STATUS_OBSERVED, STATUS_FAILED, STATUS_UNAVAILABLE):
        return None, f"{state}_invalid_status"
    if status != STATUS_OBSERVED:
        return dict(record), None
    point = record.get("point_observed_m")
    if not isinstance(point, (list, tuple)) or len(point) != 3:
        return None, f"{state}_point_observed_m_must_be_3_vector"
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in point):
        return None, f"{state}_point_observed_m_not_finite"
    if not record.get("frame_id"):
        return None, f"{state}_missing_frame_id"
    return dict(record), None


def load_neck_landmarks(path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """Load and validate the one-record-per-state landmark sidecar."""
    if path is None:
        return None, "no_neck_landmarks_file"
    path = Path(path)
    if not path.is_file():
        return None, "neck_landmarks_file_not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"neck_landmarks_unreadable: {type(exc).__name__}: {exc}"
    if payload.get("schema") != LANDMARKS_SCHEMA:
        return None, f"unsupported_neck_landmarks_schema: {payload.get('schema')!r}"
    states = payload.get("states")
    if not isinstance(states, dict):
        return None, "neck_landmarks_missing_states"
    validated: dict[str, Any] = {}
    for state in STATES:
        record, reason = _validate_record(states.get(state), state)
        if reason:
            return None, reason
        validated[state] = record
    return {
        "schema": LANDMARKS_SCHEMA,
        "path": str(path),
        "sha256": file_sha256(path),
        "states": validated,
        "sources": {state: (record.get("source") or SOURCE_MANUAL) for state, record in validated.items()},
        "landmark_name": payload.get("landmark_name", NAME),
        "created_at": payload.get("created_at"),
        "note": payload.get("note"),
    }, None


def unavailable_record(state: str, reason: str, *, source: str | None = None) -> dict[str, Any]:
    return {
        "name": NAME,
        "definition": DEFINITION,
        "definition_scope": DEFINITION_SCOPE,
        "source": source,
        "state": state,
        "status": STATUS_UNAVAILABLE,
        "reason": reason,
        "frame_id": None,
        "point_observed_m": None,
        "coordinate_system": None,
        "unit": "m",
    }


def empty_landmarks_payload(*, note: str | None = None) -> dict[str, Any]:
    return {
        "schema": LANDMARKS_SCHEMA,
        "created_at": now_iso(),
        "landmark_name": NAME,
        "definition": DEFINITION,
        "definition_scope": DEFINITION_SCOPE,
        "unit": "m",
        "coordinate_convention": COORDINATE_CONVENTION,
        "states": {
            state: unavailable_record(state, "landmark_not_yet_specified")
            for state in STATES
        },
        "note": note,
    }


def write_landmarks_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
    return path
