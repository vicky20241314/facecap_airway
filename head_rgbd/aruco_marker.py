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
                      *, subpix: bool = True) -> dict[str, Any]:
    """Detect every candidate and record why each was kept or rejected.

    The detector returns markers in image order, which is not a selection rule.
    Every candidate is logged with its id, corners and the reason it did or did
    not become the target.
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
        if int(marker_id) != int(spec.target_id):
            ok, reason = False, "wrong_id"
        entry = {
            "index": index,
            "detected_id": int(marker_id),
            "expected_id": int(spec.target_id),
            "corners_px": quad.tolist(),
            "centre_px": geometry["centre_px"],
            "geometry": geometry,
            "blur_score": round(blur_score(gray, quad), 3),
            "accepted": bool(ok),
            "rejection": reason,
        }
        candidates.append(entry)
    accepted = [c for c in candidates if c["accepted"]]
    payload = {
        "schema": "head_rgbd.aruco_detection.v1",
        "markers": [{"id": c["detected_id"], "corners_px": c["corners_px"]} for c in candidates],
        "candidates": candidates,
        "detected_ids": [c["detected_id"] for c in candidates],
        "selected": None,
        "rejection": None,
        "detector": detector_characteristics(subpix),
    }
    if not candidates:
        payload["rejection"] = "no_marker_detected"
        return payload
    if len(accepted) > 1:
        # Two accepted copies of the target: the correspondence is not unique, so
        # the frame is rejected rather than resolved by detector order.
        payload["rejection"] = "duplicate_target_id"
        return payload
    if not accepted:
        target_seen = [c for c in candidates if c["detected_id"] == spec.target_id]
        payload["rejection"] = ("target_marker_failed_quality" if target_seen
                                else "target_id_not_found")
        return payload
    payload["selected"] = accepted[0]
    return payload


def detector_characteristics(subpix: bool = True) -> dict[str, Any]:
    return {
        "dictionary": DICTIONARY_NAME,
        "opencv_version": cv2.__version__,
        "corner_refinement": ("CORNER_REFINE_SUBPIX" if subpix else "CORNER_REFINE_NONE"),
        "target_id": DEFAULT_TARGET_ID,
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
    detection = detect_candidates(rgb, spec, cfg)
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
        "marker": {
            "marker_type": MARKER_TYPE,
            "dictionary": spec.dictionary,
            "target_id": int(spec.target_id),
            "print": spec.as_dict(),
            "occlusion": page_area_mm2(spec),
            "detected_ids": detection["detected_ids"],
            "candidates": detection["candidates"],
            "detector": detection["detector"],
        },
        "rgb_sha256": _sha(rgb),
        "depth_sha256": _sha(depth),
        "intrinsics_sha256": _sha(np.asarray(intrinsics, np.float64)),
        "intrinsics": [float(v) for v in intrinsics],
    }
    selected = detection["selected"]
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
