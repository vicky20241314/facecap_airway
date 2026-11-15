"""ArUco laryngeal marker: detection, centre, depth binding and print assets.

Replaces the QR payload marker with a 4x4 ArUco marker from ``DICT_4X4_50``.
What is reused from the reference implementation (``subdivide_scan/core/aruco.py``)
is the dictionary and the detector call; its centre convention is **not** reused:

* the reference takes ``corners.mean(axis=0)``, which is not the projection of the
  square centre under perspective;
* the reference accepts whichever marker is returned first;
* the reference back-projects with a bilinear helper that skips invalid
  neighbours and renormalises, which can interpolate across a depth hole.

All three are replaced here. What is reused from this project is the image
convention: ``prepare.backproject`` already returns metres with
``X_right_Y_up_Z_backward`` and negative Z, so nothing is flipped again.

The marker is defined as a *surface* point: the operator places the printed
centre on the laryngeal prominence they can feel. It is not a measurement of the
cartilage underneath.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from head_rgbd import neck_landmark as al

MARKER_TYPE = "aruco"
DICTIONARY_NAME = "DICT_4X4_50"
DEFAULT_TARGET_ID = 0
DATA_CELLS = 4
BORDER_CELLS = 1
# The printed black square is the 4 data cells plus a 1-cell black border.
QUAD_SPAN_CELLS = DATA_CELLS + 2 * BORDER_CELLS
DEFAULT_BLACK_SQUARE_MM = 16.0
DEFAULT_WHITE_MARGIN_MM = 3.0
DEFAULT_STICKER_THICKNESS_MM = 0.10
CANDIDATE_BLACK_SQUARE_MM = (12.0, 16.0, 20.0)

STATUS_OBSERVED = al.STATUS_OBSERVED
STATUS_FAILED = al.STATUS_FAILED
STATUS_UNAVAILABLE = al.STATUS_UNAVAILABLE

_DETECTOR_CACHE: dict[tuple[str, bool], Any] = {}


# --------------------------------------------------------------------------- spec
@dataclass(frozen=True)
class ArucoSpec:
    """Physical layout of one printed sticker.

    ``black_square_mm`` is the outer edge of the black square, i.e. the 6x6 cell
    block the detector reports corners on. It excludes the white margin;
    ``sticker_mm`` is the full adhesive outline.
    """

    dictionary: str = DICTIONARY_NAME
    target_id: int = DEFAULT_TARGET_ID
    black_square_mm: float = DEFAULT_BLACK_SQUARE_MM
    white_margin_mm: float = DEFAULT_WHITE_MARGIN_MM
    # None means "black square plus twice the white margin", recomputed from the
    # actual sizes so a caller changing one size cannot inherit a stale total.
    sticker_mm: float | None = None
    thickness_mm: float = DEFAULT_STICKER_THICKNESS_MM
    thickness_source: str = "nominal, not measured"
    # Detection records carry the measured black-square size and its source;
    # print assets keep the nominal values unless the caller sets these.
    size_source: str = "nominal"
    size_uncertainty_mm: float | None = None

    def __post_init__(self):
        if self.sticker_mm is None:
            object.__setattr__(self, "sticker_mm",
                               float(self.black_square_mm) + 2 * float(self.white_margin_mm))

    @property
    def cell_mm(self) -> float:
        return self.black_square_mm / QUAD_SPAN_CELLS

    @property
    def data_span_mm(self) -> float:
        return self.cell_mm * DATA_CELLS

    def validate(self) -> None:
        if self.dictionary != DICTIONARY_NAME:
            raise ValueError(f"unsupported dictionary {self.dictionary!r}")
        if not 0 <= int(self.target_id) < 50:
            raise ValueError("DICT_4X4_50 ids are 0..49")
        if self.black_square_mm <= 0 or self.white_margin_mm < 0 or self.thickness_mm < 0:
            raise ValueError("print dimensions must be positive")
        expected = self.black_square_mm + 2 * self.white_margin_mm
        if abs(expected - self.sticker_mm) > 1e-6:
            raise ValueError(f"sticker_mm {self.sticker_mm} != black + 2*margin ({expected})")

    def as_dict(self) -> dict[str, Any]:
        return {
            "marker_type": MARKER_TYPE,
            "dictionary": self.dictionary,
            "target_id": int(self.target_id),
            "data_cells": DATA_CELLS,
            "border_cells": BORDER_CELLS,
            "quad_span_cells": QUAD_SPAN_CELLS,
            "black_square_mm": round(self.black_square_mm, 4),
            "cell_mm": round(self.cell_mm, 4),
            "data_span_mm": round(self.data_span_mm, 4),
            "white_margin_mm": round(self.white_margin_mm, 4),
            "sticker_mm": round(self.sticker_mm, 4),
            "thickness_mm": round(self.thickness_mm, 4),
            "thickness_source": self.thickness_source,
            "size_source": self.size_source,
            "size_uncertainty_mm": (None if self.size_uncertainty_mm is None
                                    else round(float(self.size_uncertainty_mm), 4)),
            "size_definition": "black_square_mm is the outer edge of the black 6x6 cell block",
        }


def spec_from_defaults(defaults: dict[str, Any] | None = None) -> ArucoSpec:
    block = (defaults or {}).get("laryngeal_marker", {}).get("aruco", {})
    spec = ArucoSpec(
        dictionary=block.get("dictionary", DICTIONARY_NAME),
        target_id=int(block.get("target_id", DEFAULT_TARGET_ID)),
        black_square_mm=float(block.get("black_square_mm", DEFAULT_BLACK_SQUARE_MM)),
        white_margin_mm=float(block.get("white_margin_mm", DEFAULT_WHITE_MARGIN_MM)),
        thickness_mm=float(block.get("thickness_mm", DEFAULT_STICKER_THICKNESS_MM)),
        thickness_source=block.get("thickness_source", "nominal, not measured"),
    )
    spec = replace(spec, sticker_mm=spec.black_square_mm + 2 * spec.white_margin_mm)
    spec.validate()
    return spec


# ----------------------------------------------------------------------- detector
def _dictionary(name: str = DICTIONARY_NAME):
    if name != DICTIONARY_NAME:
        raise ValueError(f"unsupported dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detector(subpix: bool = True) -> Any:
    """Cached detector: creation is cheap but per-frame creation is not."""
    key = (DICTIONARY_NAME, bool(subpix))
    if key not in _DETECTOR_CACHE:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX if subpix else cv2.aruco.CORNER_REFINE_NONE)
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01
        _DETECTOR_CACHE[key] = cv2.aruco.ArucoDetector(_dictionary(), params)
    return _DETECTOR_CACHE[key]


# ------------------------------------------------------------------- quad checks
def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def homography_center(corners: np.ndarray) -> list[float]:
    """Centre of the black square as the projection of the canonical centre.

    ``corners.mean`` is the centroid of four projected points, which is not the
    projection of the square centre unless the quad is affine. The homography
    below maps the unit square to the detected quad, so (0.5, 0.5) maps to the
    true centre under perspective.
    """
    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    source = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float64)
    matrix = cv2.getPerspectiveTransform(source.astype(np.float32), quad.astype(np.float32))
    centre = cv2.perspectiveTransform(np.array([[[0.5, 0.5]]], np.float32), matrix)
    return [float(centre[0, 0, 0]), float(centre[0, 0, 1])]


def quad_geometry(corners: np.ndarray, spec: ArucoSpec) -> dict[str, Any]:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    area = _polygon_area(points)
    cross = []
    for index in range(4):
        a = points[(index + 1) % 4] - points[index]
        b = points[(index + 2) % 4] - points[(index + 1) % 4]
        cross.append(float(a[0] * b[1] - a[1] * b[0]))
    convex = all(value > 0 for value in cross) or all(value < 0 for value in cross)
    angles = []
    for index in range(4):
        prev = points[(index - 1) % 4] - points[index]
        nxt = points[(index + 1) % 4] - points[index]
        denominator = np.linalg.norm(prev) * np.linalg.norm(nxt)
        angles.append(float("nan") if denominator <= 1e-9
                      else float(math.degrees(math.acos(float(np.clip(
                          np.dot(prev, nxt) / denominator, -1.0, 1.0))))))
    centre = homography_center(points)
    return {
        "corners_px": points.tolist(),
        "corner_order": "detector_order_tl_tr_br_bl",
        "centre_px": [round(centre[0], 4), round(centre[1], 4)],
        "centre_method": "homography_unit_square_centre",
        "side_lengths_px": [round(float(v), 3) for v in sides],
        "min_side_px": round(float(sides.min()), 3),
        "max_side_px": round(float(sides.max()), 3),
        "side_ratio": round(float(sides.max() / max(sides.min(), 1e-9)), 4),
        "quad_area_px": round(area, 3),
        "corner_angles_deg": [round(v, 3) if np.isfinite(v) else None for v in angles],
        "convex": bool(convex),
        "px_per_cell_min": round(float(sides.min()) / QUAD_SPAN_CELLS, 4),
        "px_per_cell_mean": round(float(math.sqrt(max(area, 1e-9))) / QUAD_SPAN_CELLS, 4),
        "black_square_span_px": round(float(math.sqrt(max(area, 1e-9))), 3),
        "signed_area_sign": 1 if cross[0] > 0 else -1,
    }


def quality_gate(geometry: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str | None]:
    """Explicit, order-stable rejection reasons. No invented confidence."""
    if not geometry["convex"]:
        return False, "quad_not_convex"
    if geometry["quad_area_px"] < float(cfg.get("min_area_px2", 400.0)):
        return False, "quad_area_too_small"
    if geometry["min_side_px"] < float(cfg.get("min_side_px", 18.0)):
        return False, "quad_side_too_short"
    if geometry["side_ratio"] > float(cfg.get("max_side_ratio", 2.2)):
        return False, "quad_side_ratio_excessive"
    if geometry["px_per_cell_mean"] < float(cfg.get("min_px_per_cell", 3.0)):
        return False, "cell_pitch_below_limit"
    angles = [a for a in geometry["corner_angles_deg"] if a is not None]
    if angles and (min(angles) < float(cfg.get("min_corner_angle_deg", 45.0))
                   or max(angles) > float(cfg.get("max_corner_angle_deg", 135.0))):
        return False, "quad_corner_angle_out_of_range"
    return True, None


def blur_score(gray: np.ndarray, corners: np.ndarray, pad: int = 4) -> float:
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    x0, y0 = np.floor(points.min(0)).astype(int)
    x1, y1 = np.ceil(points.max(0)).astype(int)
    patch = gray[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    if patch.size < 9:
        return 0.0
    return float(cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F).var())


# -------------------------------------------------------------------- detection
def detect_candidates(rgb: np.ndarray, spec: ArucoSpec, cfg: dict[str, Any] | None = None,
                      *, subpix: bool = True, depth: np.ndarray | None = None,
                      intrinsics: np.ndarray | None = None) -> dict[str, Any]:
    """Detect every candidate, auto-identify it and record why it was selected.

    The detector returns markers in image order, which is not a selection rule.
    The target ID is **not** taken from ``spec``: any ``DICT_4X4_50`` ID is
    accepted, and the physical black-square size is estimated from depth and the
    four 3D corners when ``depth``/``intrinsics`` are supplied.  The old
    ``ArucoSpec.target_id`` remains the *print* asset ID; detection is automatic.

    Selection is deliberately conservative:

    * two different IDs in the frame -> ``multiple_marker_ids`` (no picking by
      order or quality);
    * two copies of the same ID -> ``duplicate_target_id``;
    * one ID with a quality or size failure -> that explicit reason, no target.
    """
    cfg = dict(cfg or {})
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb
    corners_list, ids, _ = detector(subpix).detectMarkers(gray)
    candidates: list[dict[str, Any]] = []
    if ids is None:
        return {"schema": "head_rgbd.aruco_detection.v1", "markers": [], "candidates": [],
                "detected_ids": [], "selected": None, "rejection": "no_marker_detected",
                "detector": detector_characteristics(subpix)}
    for index, (corners, marker_id) in enumerate(zip(corners_list, ids.ravel())):
        quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        geometry = quad_geometry(quad, spec)
        ok, reason = quality_gate(geometry, cfg)
        size = None
        if depth is not None and intrinsics is not None:
            size = estimate_marker_size(quad, depth, intrinsics, cfg)
            if ok and cfg.get("enforce_size", True) and size["status"] != "ok":
                ok, reason = False, size["reason"]
        entry = {
            "index": index,
            "detected_id": int(marker_id),
            "expected_id": None,
            "auto_id": True,
            "corners_px": quad.tolist(),
            "centre_px": geometry["centre_px"],
            "geometry": geometry,
            "blur_score": round(blur_score(gray, quad), 3),
            "size": size or {"status": "not_evaluated",
                             "reason": "depth_not_supplied",
                             "size_source": None},
            "accepted": bool(ok),
            "rejection": reason,
        }
        candidates.append(entry)
    accepted = [c for c in candidates if c["accepted"]]
    detected_ids = [c["detected_id"] for c in candidates]
    payload = {
        "schema": "head_rgbd.aruco_detection.v1",
        "markers": [{"id": c["detected_id"], "corners_px": c["corners_px"]}
                    for c in candidates],
        "candidates": candidates,
        "detected_ids": detected_ids,
        "selected": None,
        "rejection": None,
        "detector": detector_characteristics(subpix),
    }
    if not candidates:
        payload["rejection"] = "no_marker_detected"
        return payload
    unique_ids = sorted(set(detected_ids))
    if len(unique_ids) > 1:
        # Multiple physical IDs are not disambiguated by detector order or
        # sharpness.  The operator must keep one sticker in view.
        payload["rejection"] = "multiple_marker_ids"
        return payload
    if len(candidates) > 1:
        # Two copies of the same ID are still ambiguous.
        payload["rejection"] = "duplicate_target_id"
        return payload
    if not accepted:
        payload["rejection"] = candidates[0]["rejection"] or "marker_rejected"
        return payload
    payload["selected"] = accepted[0]
    return payload


def detector_characteristics(subpix: bool = True) -> dict[str, Any]:
    return {
        "dictionary": DICTIONARY_NAME,
        "opencv_version": cv2.__version__,
        "corner_refinement": ("CORNER_REFINE_SUBPIX" if subpix else "CORNER_REFINE_NONE"),
        "auto_id": True,
        "target_id": None,
        "id_range": [0, 49],
    }


# ------------------------------------------------------------------ depth binding
def centre_depth(centre_px: list[float], depth: np.ndarray, intr: np.ndarray,
                 cfg: dict[str, Any] | None = None, spec: ArucoSpec | None = None) -> dict[str, Any]:
    """Depth at the marker centre with explicit hole and jump rejection.

    The window is kept inside the black square so it cannot straddle the white
    margin. Invalid neighbours are never skipped or renormalised: a hole at the
    centre, a spread that indicates two surfaces, or a footprint that leaves the
    same local surface all fail with a reason.
    """
    cfg = dict(cfg or {})
    half = int(cfg.get("centre_window_half", 2))
    max_jump = float(cfg.get("max_local_jump_mm", 4.0))
    min_valid = int(cfg.get("min_valid_centre_pixels", 6))
    height, width = depth.shape[:2]
    u, v = int(round(centre_px[0])), int(round(centre_px[1]))
    if not (0 <= u < width and 0 <= v < height):
        return {"status": STATUS_FAILED, "reason": "centre_outside_image", "depth_mm": None}
    window = depth[max(v - half, 0):v + half + 1, max(u - half, 0):u + half + 1].astype(np.float64)
    valid = window > 0
    if not valid.any():
        return {"status": STATUS_FAILED, "reason": "depth_hole_at_centre", "depth_mm": None,
                "window_px": int(window.size), "valid_px": 0}
    values = window[valid]
    if len(values) < min_valid:
        return {"status": STATUS_FAILED, "reason": "depth_centre_too_few_valid",
                "depth_mm": None, "window_px": int(window.size), "valid_px": int(len(values))}
    median = float(np.median(values))
    spread = float(values.max() - values.min())
    if spread > max_jump:
        return {"status": STATUS_FAILED, "reason": "depth_jump_at_centre", "depth_mm": None,
                "window_px": int(window.size), "valid_px": int(len(values)),
                "spread_mm": round(spread, 4)}
    if float(np.abs(values - median).max()) > max_jump:
        return {"status": STATUS_FAILED, "reason": "depth_not_single_surface", "depth_mm": None,
                "window_px": int(window.size), "valid_px": int(len(values)),
                "spread_mm": round(spread, 4)}
    centre_value = depth[v, u]
    if centre_value <= 0:
        # The centre pixel itself is void: the marker would sit on a hole, which a
        # median across the neighbourhood must not paper over.
        return {"status": STATUS_FAILED, "reason": "depth_hole_at_centre_pixel", "depth_mm": None,
                "window_px": int(window.size), "valid_px": int(len(values))}
    return {
        "status": "ok",
        "reason": None,
        "depth_mm": round(float(centre_value), 4),
        "window_median_mm": round(median, 4),
        "window_spread_mm": round(spread, 4),
        "window_px": int(window.size),
        "valid_px": int(len(values)),
        "max_local_jump_mm": max_jump,
    }


# --------------------------------------------------------------- automatic size
def _corner_depth_mm(corner_px, depth: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """Median depth at one marker corner, with hole/jump rejection.

    The corner sits on the outer edge of the black square.  A small window is
    used instead of one pixel because the printed corner can fall between
    pixels; invalid neighbours are not skipped/re-normalised, and a window that
    spans two surfaces is rejected rather than averaged.
    """
    half = int(cfg.get("size_corner_window_half", 1))
    max_jump = float(cfg.get("size_corner_max_jump_mm", 4.0))
    height, width = depth.shape[:2]
    u, v = int(round(float(corner_px[0]))), int(round(float(corner_px[1])))
    if not (0 <= u < width and 0 <= v < height):
        return {"depth_mm": None, "reason": "corner_outside_image"}
    window = depth[max(v - half, 0):v + half + 1,
                   max(u - half, 0):u + half + 1].astype(np.float64)
    valid = window > 0
    if not valid.any():
        return {"depth_mm": None, "reason": "corner_depth_hole", "valid_px": 0}
    values = window[valid]
    spread = float(values.max() - values.min())
    if spread > max_jump:
        return {"depth_mm": None, "reason": "corner_depth_jump",
                "spread_mm": round(spread, 4), "valid_px": int(len(values))}
    return {"depth_mm": float(np.median(values)), "reason": None,
            "spread_mm": round(spread, 4), "valid_px": int(len(values))}


def estimate_marker_size(corners: np.ndarray, depth: np.ndarray,
                         intrinsics: np.ndarray,
                         cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Black-square edge length (mm) from depth + the four 3D corners.

    The detector's quad spans the outer edge of the black 6x6 cell block, so the
    median 3D side length is the physical black-square size.  This is a
    *measurement*, not the nominal print size: the field sticker may be larger
    or smaller than the PDF (the historical ``zp_capwitharuco`` sticker measured
    ~33.6 mm while its sidecar said 16 mm).

    Validity follows the spec: at least 3 valid corners, size 8-40 mm, and a
    stable estimate (side ratio / half-range below the configured limits).
    ``size_source=measured_from_depth`` even when the estimate is rejected, so
    the failure is auditable.
    """
    cfg = dict(cfg or {})
    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    corner_info = [_corner_depth_mm(point, depth, cfg) for point in quad]
    valid = [i for i, info in enumerate(corner_info) if info["depth_mm"] is not None]
    base = {"size_source": "measured_from_depth",
            "valid_corners": int(len(valid)),
            "corner_depths": corner_info}
    if len(valid) < 3:
        return {**base, "status": "invalid", "reason": "size_insufficient_corners",
                "black_square_mm": None, "size_uncertainty_mm": None,
                "side_lengths_mm": []}
    points: list[list[float] | None] = []
    for index in range(4):
        info = corner_info[index]
        points.append(None if info["depth_mm"] is None else
                      backproject_centre(quad[index], info["depth_mm"], intrinsics))
    sides: list[float] = []
    side_info: list[dict[str, Any]] = []
    for index in range(4):
        nxt = (index + 1) % 4
        if points[index] is None or points[nxt] is None:
            continue
        length = float(np.linalg.norm(np.asarray(points[index]) -
                                      np.asarray(points[nxt])) * 1000.0)
        sides.append(length)
        side_info.append({"edge": [index, nxt], "mm": round(length, 4)})
    if len(sides) < 2:
        return {**base, "status": "invalid", "reason": "size_insufficient_sides",
                "black_square_mm": None, "size_uncertainty_mm": None,
                "side_lengths_mm": [round(v, 4) for v in sides]}
    sides_arr = np.asarray(sides, dtype=np.float64)
    size = float(np.median(sides_arr))
    spread = float(sides_arr.max() - sides_arr.min())
    uncertainty = 0.5 * spread
    ratio = float(sides_arr.max() / max(sides_arr.min(), 1e-9))
    min_mm = float(cfg.get("min_black_square_mm", 8.0))
    max_mm = float(cfg.get("max_black_square_mm", 40.0))
    max_ratio = float(cfg.get("size_max_side_ratio", 1.30))
    max_unc = float(cfg.get("size_max_uncertainty_mm", 2.0))
    status, reason = "ok", None
    if size < min_mm or size > max_mm:
        status, reason = "invalid", "size_out_of_range"
    elif ratio > max_ratio or uncertainty > max_unc:
        status, reason = "invalid", "size_unstable"
    return {**base, "status": status, "reason": reason,
            "black_square_mm": round(size, 4),
            "size_uncertainty_mm": round(uncertainty, 4),
            "side_lengths_mm": [round(float(v), 4) for v in sides],
            "side_ratio": round(ratio, 4),
            "side_info": side_info,
            "min_black_square_mm": min_mm, "max_black_square_mm": max_mm}


def _fit_plane(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return {"normal": None, "rms_mm": None, "centroid_m": None}
    centroid = points.mean(axis=0)
    centred = points - centroid
    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    if float(np.dot(normal, centroid)) < 0:
        normal = -normal
    residual = centred @ normal
    return {"normal": [float(v) for v in normal],
            "rms_mm": round(float(np.sqrt(np.mean(residual ** 2)) * 1000.0), 4),
            "centroid_m": [float(v) for v in centroid], "samples": int(len(points))}


def surface_point(corners: np.ndarray, depth: np.ndarray, intr: np.ndarray,
                  cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Marker surface points from this frame's aligned depth, and their plane."""
    from head_rgbd.prepare import backproject

    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    grid = int(dict(cfg or {}).get("plane_grid", 5))
    inside = []
    for i in range(grid):
        for j in range(grid):
            inside.append([0.1 + 0.8 * i / max(grid - 1, 1), 0.1 + 0.8 * j / max(grid - 1, 1)])
    source = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float64)
    matrix = cv2.getPerspectiveTransform(source.astype(np.float32), quad.astype(np.float32))
    pts = cv2.perspectiveTransform(np.array([inside], np.float32), matrix)[0]
    xyz, ok = backproject(pts, depth, intr)
    used = xyz[ok]
    return {"samples": int(ok.sum()), "requested": len(pts),
            "points_m": used.tolist(), "plane": _fit_plane(used) if len(used) >= 3 else _fit_plane(np.zeros((0, 3)))}


def backproject_centre(centre_px: list[float], depth_mm: float, intr: np.ndarray) -> list[float]:
    """Camera-frame metres for one pixel, using the project's exact convention.

    Mirrors ``prepare.backproject`` (x = (u-cx)z/fx, y = -(v-cy)z/fy, z = -depth)
    so the marker point lands in the same frame as every other landmark. No axis
    is flipped afterwards.
    """
    fx, fy, cx, cy = [float(v) for v in intr]
    if not depth_mm or depth_mm <= 0:
        raise ValueError("depth must be positive millimetres")
    z = float(depth_mm) * 0.001
    return [float((centre_px[0] - cx) * z / fx),
            float(-(centre_px[1] - cy) * z / fy),
            float(-z)]


# --------------------------------------------------------------------- printing
def marker_cells(spec: ArucoSpec) -> np.ndarray:
    """Binary cell matrix of the black square (QUAD_SPAN_CELLS x QUAD_SPAN_CELLS)."""
    side = QUAD_SPAN_CELLS * 10
    image = cv2.aruco.generateImageMarker(_dictionary(spec.dictionary), int(spec.target_id), side)
    cell = side // QUAD_SPAN_CELLS
    return (image[cell // 2::cell, cell // 2::cell][:QUAD_SPAN_CELLS, :QUAD_SPAN_CELLS] < 128)


def sticker_image(spec: ArucoSpec, px_per_mm: float = 24.0) -> np.ndarray:
    """White margin + black square + marker, in millimetres times ``px_per_mm``."""
    margin_px = int(round(spec.white_margin_mm * px_per_mm))
    square_px = int(round(spec.black_square_mm * px_per_mm))
    cells = marker_cells(spec)
    cell_px = square_px / QUAD_SPAN_CELLS
    square = np.full((square_px, square_px), 255, np.uint8)
    for row in range(QUAD_SPAN_CELLS):
        for column in range(QUAD_SPAN_CELLS):
            if cells[row, column]:
                y0 = int(round(row * cell_px)); y1 = int(round((row + 1) * cell_px))
                x0 = int(round(column * cell_px)); x1 = int(round((column + 1) * cell_px))
                square[y0:y1, x0:x1] = 0
    total = square_px + 2 * margin_px
    canvas = np.full((total, total), 255, np.uint8)
    canvas[margin_px:margin_px + square_px, margin_px:margin_px + square_px] = square
    return canvas


def _file_stem(spec: ArucoSpec) -> str:
    return (f"aruco_{spec.dictionary}_id{spec.target_id}"
            f"_black{spec.black_square_mm:g}mm_margin{spec.white_margin_mm:g}mm"
            f"_sticker{spec.sticker_mm:g}mm")


def marker_svg(spec: ArucoSpec, scale: float = 3.7795) -> str:
    """SVG in millimetres (1 mm = 3.7795 px at 96 dpi)."""
    cells = marker_cells(spec)
    cell = spec.cell_mm * scale
    margin = spec.white_margin_mm * scale
    total = spec.sticker_mm * scale
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total:.3f}" height="{total:.3f}" '
        f'viewBox="0 0 {total:.3f} {total:.3f}">',
        f'<rect width="{total:.3f}" height="{total:.3f}" fill="#ffffff"/>',
    ]
    for row in range(QUAD_SPAN_CELLS):
        for column in range(QUAD_SPAN_CELLS):
            if cells[row, column]:
                parts.append(
                    f'<rect x="{margin + column * cell:.3f}" y="{margin + row * cell:.3f}" '
                    f'width="{cell:.3f}" height="{cell:.3f}" fill="#000000"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_assets(directory: str | Path, spec: ArucoSpec, *, px_per_mm: float = 24.0) -> dict[str, Path]:
    """PNG, SVG, spec JSON and an A4 print sheet, all named after the layout."""
    spec.validate()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    stem = _file_stem(spec)
    png = out / f"{stem}_150dpi.png"
    svg = out / f"{stem}.svg"
    sheet = out / f"{stem}_printsheet.png"
    spec_json = out / f"{stem}.json"

    image = sticker_image(spec, px_per_mm)
    cv2.imwrite(str(png), image)
    svg.write_text(marker_svg(spec), encoding="utf-8")
    cv2.imwrite(str(sheet), print_sheet(spec, image, px_per_mm))

    payload = spec.as_dict()
    payload.update({
        "schema": "head_rgbd.aruco_print_spec.v1",
        "files": {"png": png.name, "svg": svg.name, "print_sheet": sheet.name},
        "png_px_per_mm": px_per_mm,
        "marker_sha256_black_square": hashlib.sha256(
            (cv2.aruco.generateImageMarker(_dictionary(spec.dictionary), int(spec.target_id),
                                           600) > 0).astype(np.uint8).tobytes()).hexdigest(),
        "print_note": ("打印时禁用“适应页面”，按 100% 比例；用页面左上角的 50 mm 校验尺核对。"
                       "标签文字与裁切线都放在白色留边之外。"),
        "thickness_note": "厚度为标称值，未实测；贴纸下表面不可观测。",
    })
    spec_json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"png": png, "svg": svg, "print_sheet": sheet, "spec": spec_json}


def print_sheet(spec: ArucoSpec, sticker: np.ndarray, px_per_mm: float, page_mm=(210.0, 297.0)) -> np.ndarray:
    """A4 page: mm ruler, crop marks and a label kept outside the quiet zone."""
    width = int(page_mm[0] * px_per_mm)
    height = int(page_mm[1] * px_per_mm)
    page = np.full((height, width), 255, np.uint8)
    side = sticker.shape[0]
    y0 = int(60 * px_per_mm); x0 = int(60 * px_per_mm)
    page[y0:y0 + side, x0:x0 + side] = sticker
    # 50 mm ruler with 10 mm ticks
    ruler_y = int(20 * px_per_mm)
    cv2.line(page, (x0, ruler_y), (x0 + int(50 * px_per_mm), ruler_y), 0, 1)
    for step in range(6):
        x = x0 + int(step * 10 * px_per_mm)
        cv2.line(page, (x, ruler_y), (x, ruler_y - int((4 if step % 5 == 0 else 2) * px_per_mm)), 0, 1)
    label_y = y0 + side + int(8 * px_per_mm)
    cv2.putText(page, f"{spec.dictionary} id={spec.target_id} black={spec.black_square_mm:g}mm "
                      f"sticker={spec.sticker_mm:g}mm", (x0, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    cv2.putText(page, "print at 100%; verify the 50 mm ruler before cutting", (x0, label_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv2.LINE_AA)
    # crop marks outside the white margin
    for corner in ((x0, y0), (x0 + side, y0), (x0, y0 + side), (x0 + side, y0 + side)):
        cv2.drawMarker(page, corner, 0, cv2.MARKER_CROSS, int(6 * px_per_mm), 1)
    return page


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="ArUco laryngeal marker print assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--black-square-mm", type=float, action="append", default=None,
                        help=f"repeatable; default {list(CANDIDATE_BLACK_SQUARE_MM)}")
    parser.add_argument("--target-id", type=int, default=DEFAULT_TARGET_ID)
    parser.add_argument("--white-margin-mm", type=float, default=DEFAULT_WHITE_MARGIN_MM)
    args = parser.parse_args(argv)
    sizes = args.black_square_mm or list(CANDIDATE_BLACK_SQUARE_MM)
    written = []
    for size in sizes:
        spec = ArucoSpec(target_id=args.target_id, black_square_mm=float(size),
                         white_margin_mm=args.white_margin_mm,
                         sticker_mm=float(size) + 2 * args.white_margin_mm)
        written.append(write_assets(args.output, spec))
    for entry in written:
        print(json.dumps({k: str(v) for k, v in entry.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ----------------------------------------------------------------- frame record
# --------------------------------------------------------------- live gate
def _frame_attr(frame: Any, name: str, default: Any = None) -> Any:
    if isinstance(frame, dict):
        return frame.get(name, default)
    return getattr(frame, name, default)


def _compact_marker_record(record: dict[str, Any]) -> dict[str, Any]:
    marker = record.get("marker") or {}
    size = marker.get("size") or {}
    return {
        "frame_id": record.get("frame_id"),
        "status": record.get("status"),
        "reason": record.get("reason"),
        "blur_score": record.get("blur_score"),
        "detected_id": record.get("detected_id"),
        "detected_ids": marker.get("detected_ids"),
        "black_square_mm": size.get("black_square_mm"),
        "size_source": size.get("size_source"),
        "size_uncertainty_mm": size.get("size_uncertainty_mm"),
        "centre_px": record.get("centre_px"),
    }


def evaluate_frames(frames, state: str, spec: ArucoSpec | None = None,
                    cfg: dict[str, Any] | None = None,
                    frame_id_prefix: str | None = None) -> dict[str, Any]:
    """Live front-frame gate: best sharpness among frames with a valid marker.

    This is the same ``evaluate_frame`` used by the prepared recheck; the live
    gate is advisory only.  A frame with no marker contributes its explicit
    failure reason to ``attempts`` and never becomes a target.
    """
    spec = spec or ArucoSpec()
    attempts: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None
    for index, frame in enumerate(frames):
        frame_id = f"{frame_id_prefix or state}_f{index}"
        record = evaluate_frame(
            np.asarray(_frame_attr(frame, "rgb")),
            np.asarray(_frame_attr(frame, "depth")),
            np.asarray(_frame_attr(frame, "intrinsics"), dtype=np.float64).reshape(-1),
            spec, state=state, frame_id=frame_id, cfg=cfg,
            timestamp={"host_monotonic_s": float(_frame_attr(frame, "host_monotonic_s", 0.0) or 0.0)},
        )
        compact = _compact_marker_record(record)
        compact["index"] = index
        attempts.append(compact)
        if record.get("status") == STATUS_OBSERVED:
            score = float(record.get("blur_score") or 0.0)
            if best is None or score > best[0]:
                best = (score, compact)
    if best is None:
        reason = next((a["reason"] for a in attempts if a["reason"]),
                      "no_frames" if not attempts else "no_front_frame_with_valid_marker")
        return {"state": state, "status": STATUS_FAILED, "reason": reason,
                "frame_id": None, "detected_id": None, "black_square_mm": None,
                "size": None, "attempts": attempts, "best": None}
    compact = best[1]
    return {"state": state, "status": STATUS_OBSERVED, "reason": None,
            "frame_id": compact["frame_id"], "detected_id": compact["detected_id"],
            "black_square_mm": compact["black_square_mm"],
            "size": {k: compact.get(k) for k in
                     ("black_square_mm", "size_source", "size_uncertainty_mm")},
            "attempts": attempts, "best": compact}


def marker_gate_decision(gate: dict[str, Any], auto: bool,
                         allow_fallback: bool) -> str:
    """One place for the live failure policy: continue / retake / abort / fallback."""
    if gate.get("status") == STATUS_OBSERVED:
        return "continue"
    if allow_fallback:
        return "fallback"
    if auto:
        return "abort"
    return "retake"


def preview_marker_line(detection: dict[str, Any]) -> str:
    """Short, non-fabricated preview text: ID/size/quality or the reason."""
    selected = detection.get("selected")
    if selected is None:
        return f"ARUCO: {detection.get('rejection') or 'no_marker_detected'}"
    size = selected.get("size") or {}
    if size.get("status") == "ok":
        return (f"ARUCO id={selected['detected_id']} "
                f"{size['black_square_mm']:.1f}mm ok")
    return f"ARUCO id={selected['detected_id']} size {size.get('reason', 'unknown')}"


def _sha(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def page_area_mm2(spec: ArucoSpec) -> dict[str, float]:
    """Occluded areas, so the sticker's footprint is reported, not inferred."""
    black = spec.black_square_mm ** 2
    sticker = spec.sticker_mm ** 2
    return {"black_square_mm2": round(black, 3),
            "white_margin_mm2": round(sticker - black, 3),
            "sticker_mm2": round(sticker, 3),
            "white_margin_fraction": round((sticker - black) / sticker, 4)}


def evaluate_frame(rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray,
                   spec: ArucoSpec, *, state: str, frame_id: str,
                   cfg: dict[str, Any] | None = None,
                   timestamp: dict[str, Any] | None = None) -> dict[str, Any]:
    """One frame -> one record. Closed and open frames are evaluated separately.

    Nothing is invented: a frame either has a selected marker with a measured
    surface point, or it carries the reason it does not.
    """
    spec.validate()
    cfg = dict(cfg or {})
    detection = detect_candidates(rgb, spec, cfg, depth=depth, intrinsics=intrinsics)
    selected = detection["selected"]
    detected_id = None if selected is None else int(selected["detected_id"])
    size = None if selected is None else selected.get("size")
    if (selected is not None and isinstance(size, dict)
            and size.get("status") == "ok"):
        # The field sticker may not match the printed PDF size (the historical
        # zp sticker measured ~33.6 mm against a 16 mm sidecar).  The record uses
        # the measured size; the nominal spec is only a print-asset value.
        print_spec = replace(
            spec, target_id=detected_id,
            black_square_mm=float(size["black_square_mm"]), sticker_mm=None,
            size_source="measured_from_depth",
            size_uncertainty_mm=float(size["size_uncertainty_mm"]))
    else:
        print_spec = spec
    marker = {
        "marker_type": MARKER_TYPE,
        "dictionary": spec.dictionary,
        "target_id": detected_id,
        "print": print_spec.as_dict(),
        "size": size,
        "occlusion": page_area_mm2(print_spec),
        "detected_ids": detection["detected_ids"],
        "candidates": detection["candidates"],
        "detector": detection["detector"],
    }
    record: dict[str, Any] = {
        "name": al.NAME,
        "definition": al.DEFINITION,
        "definition_scope": al.DEFINITION_SCOPE,
        "source": MARKER_TYPE,
        "state": state,
        "frame_id": frame_id,
        "timestamp": timestamp or {},
        "image_size": [int(depth.shape[1]), int(depth.shape[0])],
        "unit": "m",
        "coordinate_system": al.OBSERVATION_FRAME_COORDINATES,
        "observation_is_direct_measurement": False,
        "note": ("ArUco 标记中心所指示的喉结体表位置；不是内部甲状软骨的直接测量"),
        "marker": marker,
        "rgb_sha256": _sha(rgb),
        "depth_sha256": _sha(depth),
        "intrinsics_sha256": _sha(np.asarray(intrinsics, np.float64)),
        "intrinsics": [float(v) for v in intrinsics],
    }
    if selected is None:
        record.update({"status": STATUS_FAILED, "reason": detection["rejection"],
                       "centre_px": None, "point_observed_m": None,
                       "skin_estimate_m": None, "transform_source": None})
        return record
    centre = selected["centre_px"]
    depth_check = centre_depth(centre, depth, intrinsics, cfg, spec)
    surface = surface_point(np.asarray(selected["corners_px"]), depth, intrinsics, cfg)
    record.update({
        "centre_px": centre,
        "centre_method": selected["geometry"]["centre_method"],
        "corners_px": selected["corners_px"],
        "detected_id": selected["detected_id"],
        "geometry": selected["geometry"],
        "blur_score": selected["blur_score"],
        "depth_check": depth_check,
        "marker_plane": surface["plane"],
        "transform_source": "rgbd_backprojection_same_frame",
    })
    if depth_check["status"] != "ok":
        record.update({"status": STATUS_FAILED, "reason": depth_check["reason"],
                       "point_observed_m": None, "skin_estimate_m": None})
        return record
    point = backproject_centre(centre, depth_check["depth_mm"], intrinsics)
    normal = surface["plane"]["normal"]
    thickness_m = spec.thickness_mm * 0.001
    skin = None if normal is None else [float(point[i] + normal[i] * 0) for i in range(3)]
    if normal is not None:
        # The sticker surface lies in front of the skin along the outward normal;
        # the estimate moves back along it by the nominal thickness.
        skin = [float(point[i] - normal[i] * thickness_m) for i in range(3)]
    record.update({
        "status": STATUS_OBSERVED,
        "reason": None,
        "point_observed_m": [float(v) for v in point],
        "marker_surface_point_m": [float(v) for v in point],
        "skin_estimate_m": skin,
        "skin_estimate_is_direct_measurement": False,
        "uncertainty_not_included": [
            "贴纸下表面不可观测，厚度为标称值",
            "贴纸贴合气泡、翘边与局部弯曲",
            "打印/裁切导致贴纸中心与符号中心的装配误差",
        ],
        "sticker_under_surface_observable": False,
    })
    return record
