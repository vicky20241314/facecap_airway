"""Lightweight continuous head-turn tracking on the RGB-D sequence.

This stage does not fit a model. It answers three questions per frame, cheaply:

1. where the head is (rigid pose from the 68 landmarks against the neutral FLAME
   template, hence yaw/pitch/roll), and how reliable that answer is;
2. whether the mouth stayed closed (an inner-lip gap proxy, no expression fitting);
3. when the sequence must be cut, because continuity is no longer supported.

Detector inference (YuNet + TUFA) runs only every ``detect_stride`` frames or after
a tracking loss. In between, the previous landmark set is propagated with
forward-backward KLT on the head region and the rigid pose is solved directly from
those correspondences. No RGB-D rigid ICP is run here; the per-frame pose comes
from the landmark correspondences and the depth map at those pixels.

Output continuity is expressed with ``tracking_group``. It is deliberately
different from the angle grouping used later: one tracking group may cover many
yaw angles, and one yaw angle may appear in several tracking groups.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time

import cv2
import numpy as np

from head_rgbd.prepare import backproject, rigid
from head_rgbd.reference_contract import inner_lip_gap_mm

UPPER_FACE = np.arange(27, 48)
INTEROCULAR_PAIR = (36, 45)


@dataclass
class _Track:
    """Internal per-frame accumulator; serialised later."""

    source_index: int
    group: str
    timestamp_s: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    quality: float
    residual_mm: float
    sharpness: float
    face_depth_pixels: int
    valid_landmarks: int
    detected: bool
    usable: bool
    reason: str | None
    inner_lip_gap_mm: float | None


def analyze(sequence, config, role: str) -> dict:
    """Track one input role and return the serialisable result."""
    started = time.perf_counter()
    os.environ.setdefault("HEADRGBD_TUFA_DEVICE", "cpu")
    template = template_landmarks(config.flame)
    tracking = config.tracking
    quality = tracking.quality

    count = len(sequence)
    uv_all = np.full((count, 68, 2), np.nan, dtype=np.float64)
    uv_valid_all = np.zeros((count, 68), dtype=bool)
    xyz_all = np.full((count, 68, 3), np.nan, dtype=np.float64)
    depth_valid_all = np.zeros((count, 68), dtype=bool)
    rotation_all = np.tile(np.eye(3), (count, 1, 1))
    translation_all = np.zeros((count, 3))
    tracks: list[_Track] = []

    previous_gray = None
    previous_uv = None
    previous_timestamp = None
    previous_yaw = None
    previous_group_owner = None
    gap_reference = None
    ratio_reference = None
    group_index = -1
    lost = True

    for index, entry in enumerate(sequence.entries):
        frame = sequence.read(index)
        rgb, depth = frame["rgb"], frame["depth"]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detected = False
        uv = None
        bbox = None
        must_detect = (index % tracking.detect_stride == 0) or lost or previous_uv is None
        if not must_detect:
            uv = _propagate(previous_gray, gray, previous_uv, tracking)
            if uv is None:
                must_detect = True
        if must_detect:
            uv, bbox = detect_landmarks(rgb)
            detected = True
        if uv is None:
            tracks.append(_failed_track(index, entry, group_index, "landmark_detection_failed"))
            previous_uv = None
            lost = True
            previous_timestamp = entry.timestamp_s
            continue

        intrinsics = frame["intrinsics"]
        xyz, depth_valid = backproject(uv, depth, intrinsics)
        uv_valid = np.isfinite(uv).all(axis=1)
        uv_all[index] = np.where(uv_valid[:, None], uv, np.nan)
        uv_valid_all[index] = uv_valid
        xyz_all[index] = xyz
        depth_valid_all[index] = depth_valid

        mask = uv_valid & depth_valid
        mask[: UPPER_FACE[0]] = False
        mask[UPPER_FACE[-1] + 1:] = False
        pose, residual_mm = None, float("inf")
        if int(mask.sum()) >= tracking.min_valid_upper_landmarks:
            try:
                rotation, translation = rigid(template[mask], xyz[mask])
                pose = (rotation, translation)
                residual_mm = float(np.median(
                    np.linalg.norm(template[mask] @ rotation.T + translation - xyz[mask], axis=1)) * 1000.)
            except (ValueError, np.linalg.LinAlgError):
                pose = None

        sharpness = _sharpness(rgb, bbox, uv)
        face_pixels = _face_depth_pixels(depth, bbox, uv)
        valid_landmarks = int((uv_valid & depth_valid).sum())
        yaw = pitch = roll = float("nan")
        if pose is not None:
            rotation, translation = pose
            rotation_all[index] = rotation
            translation_all[index] = translation
            pitch, yaw, roll = _euler(rotation)
        gap = inner_lip_gap_mm(xyz, depth_valid)
        ratio = _mouth_ratio(xyz, depth_valid)

        reasons = []
        usable = True
        if pose is None:
            usable, reason = False, "rigid_pose_unavailable"
            reasons.append(reason)
        elif residual_mm > tracking.max_fit_residual_mm:
            usable, reason = False, "landmark_fit_residual_too_large"
            reasons.append(reason)
        elif sharpness < quality.min_sharpness:
            usable, reason = False, "frame_too_blurred"
            reasons.append(reason)
        elif face_pixels < quality.min_face_depth_pixels:
            usable, reason = False, "insufficient_face_depth_pixels"
            reasons.append(reason)
        elif valid_landmarks < quality.min_valid_landmarks:
            usable, reason = False, "insufficient_valid_landmarks"
            reasons.append(reason)
        elif abs(pitch) > tracking.max_pitch_deg or abs(roll) > tracking.max_roll_deg:
            usable, reason = False, "out_of_range_pitch_or_roll"
            reasons.append(reason)
        else:
            reason = None

        if gap_reference is None and usable:
            gap_reference = gap
            ratio_reference = ratio
        expression_changed = bool(
            usable and gap_reference is not None and gap is not None
            and (abs(gap - gap_reference) > tracking.expression_gap_mm
                 or (ratio is not None and ratio_reference is not None
                     and abs(ratio - ratio_reference) > tracking.expression_ratio))
        )

        boundary = []
        if previous_timestamp is None:
            boundary.append("sequence_start")
        else:
            delta = entry.timestamp_s - previous_timestamp
            if delta <= 0:
                boundary.append("nonmonotonic_timestamp")
            elif delta > config.input.max_adjacent_gap_s:
                boundary.append("temporal_gap")
            if lost:
                boundary.append("tracking_relocalized_after_loss")
            if previous_yaw is not None and usable and np.isfinite(previous_yaw) \
                    and abs(yaw - previous_yaw) > tracking.max_yaw_jump_deg:
                boundary.append("yaw_jump")
            if previous_group_owner is not None and previous_group_owner != entry.group:
                boundary.append("source_group_change")
        if expression_changed:
            boundary.append("expression_change")
        if boundary or group_index < 0:
            group_index += 1
        group = f"track{group_index:02d}"

        frame_quality = _quality(sharpness, face_pixels, valid_landmarks, residual_mm, quality) if usable else 0.0
        record = _Track(source_index=index, group=group, timestamp_s=entry.timestamp_s,
                        yaw_deg=float(yaw), pitch_deg=float(pitch), roll_deg=float(roll),
                        quality=float(frame_quality), residual_mm=float(residual_mm),
                        sharpness=float(sharpness), face_depth_pixels=int(face_pixels),
                        valid_landmarks=int(valid_landmarks), detected=detected,
                        usable=usable, reason=reason, inner_lip_gap_mm=None if gap is None else float(gap))
        record.__dict__["boundary_reasons"] = boundary
        tracks.append(record)

        previous_gray = gray
        previous_uv = np.where(uv_valid[:, None], uv, np.nan)
        lost = bool(not usable or "yaw_jump" in boundary)
        previous_timestamp = entry.timestamp_s
        previous_yaw = float(yaw) if usable else previous_yaw
        previous_group_owner = entry.group
        if index % 10 == 0 or index == count - 1:
            print(json.dumps(dict(role=role, frame=index, of=count, group=group, yaw_deg=round(float(yaw), 2),
                                  quality=round(float(frame_quality), 3), detected=detected)), flush=True)

    records = [_serialise(track, sequence) for track in tracks]
    summary = _summary(records, sequence, role)
    summary["elapsed_s"] = time.perf_counter() - started
    summary["detector_calls"] = int(sum(1 for track in tracks if track.detected))
    summary["tracked_by_klt"] = int(sum(1 for track in tracks if not track.detected))
    return {
        "role": role,
        "records": records,
        "summary": summary,
        "arrays": {
            "uv": uv_all, "uv_valid": uv_valid_all, "xyz": xyz_all, "depth_valid": depth_valid_all,
            "R": rotation_all, "t": translation_all,
            "yaw_deg": np.asarray([track.yaw_deg for track in tracks], dtype=np.float64),
            "pitch_deg": np.asarray([track.pitch_deg for track in tracks], dtype=np.float64),
            "roll_deg": np.asarray([track.roll_deg for track in tracks], dtype=np.float64),
            "quality": np.asarray([track.quality for track in tracks], dtype=np.float64),
            "residual_mm": np.asarray([track.residual_mm for track in tracks], dtype=np.float64),
            "usable": np.asarray([track.usable for track in tracks], dtype=bool),
            "detected": np.asarray([track.detected for track in tracks], dtype=bool),
            "visible_landmarks": depth_valid_all & uv_valid_all,
        },
    }


def template_landmarks(flame_config) -> np.ndarray:
    """Neutral FLAME template landmarks in metres on the CPU, no gradient."""
    import torch
    from core.fit_options import FitOptions
    from core.integrated_fitting import make_flame

    flame = make_flame(FitOptions(shape_components=flame_config.shape_components,
                                  expression_components=flame_config.expression_components), "cpu", batch=1)
    with torch.no_grad():
        _, landmarks = flame(
            shape_params=torch.zeros(1, flame_config.shape_components),
            expression_params=torch.zeros(1, flame_config.expression_components),
            pose_params=torch.zeros(1, 6),
            transl=torch.zeros(1, 3),
        )
    return landmarks[0].detach().cpu().numpy().astype(np.float64)


def detect_landmarks(rgb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """YuNet box plus 68 TUFA points for one RGB frame.

    Returns ``(uv, bbox)``; ``(None, None)`` when the detector finds no face. The
    caller decides whether that is fatal, so a single lost frame cannot abort a
    long turn.
    """
    from core.face_test import detect_face_bbox

    bbox = detect_face_bbox(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if bbox is None:
        return None, None
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= 1 or bbox[3] <= 1:
        return None, None
    from third_party.tufa.Test import get_lmk_single

    landmarks = np.asarray(get_lmk_single(rgb, bbox), dtype=np.float64)
    if landmarks.size != 136:
        raise RuntimeError(f"TUFA returned {landmarks.size} values; expected 136")
    return landmarks.reshape(68, 2), bbox


# ------------------------------------------------------------------------- detail
def _propagate(previous_gray, gray, previous_uv, tracking):
    if previous_gray is None or previous_uv is None:
        return None
    points = np.where(np.isfinite(previous_uv).all(axis=1)[:, None], previous_uv, -1.0)
    good = np.isfinite(points).all(axis=1)
    if int(good.sum()) < tracking.klt_min_landmarks:
        return None
    source = points[good].astype(np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, source, None,
        winSize=(tracking.klt_window, tracking.klt_window), maxLevel=tracking.klt_levels, criteria=criteria)
    if forward is None or status is None:
        return None
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, forward, None,
        winSize=(tracking.klt_window, tracking.klt_window), maxLevel=tracking.klt_levels, criteria=criteria)
    if backward is None or status_back is None:
        return None
    keep = (status[:, 0].astype(bool) & status_back[:, 0].astype(bool)
            & (np.linalg.norm(backward[:, 0] - source[:, 0], axis=1) < tracking.klt_forward_backward_px))
    if int(keep.sum()) < tracking.klt_min_landmarks:
        return None
    uv = np.full(previous_uv.shape, np.nan)
    uv[np.flatnonzero(good)[keep]] = forward[keep, 0]
    return uv


def _sharpness(rgb, bbox, uv):
    height, width = rgb.shape[:2]
    if bbox is not None:
        x, y, w, h = [int(round(value)) for value in bbox]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + max(w, 1)), min(height, y + max(h, 1))
    else:
        finite = np.isfinite(uv).all(axis=1)
        if not finite.any():
            return 0.0
        x0, y0 = np.floor(np.nanmin(uv[finite], axis=0)).astype(int)
        x1, y1 = np.ceil(np.nanmax(uv[finite], axis=0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(cv2.resize(crop, (128, 128)), cv2.CV_64F).var())


def _face_depth_pixels(depth, bbox, uv):
    height, width = depth.shape
    if bbox is not None:
        x, y, w, h = [int(round(value)) for value in bbox]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + max(w, 1)), min(height, y + max(h, 1))
    else:
        finite = np.isfinite(uv).all(axis=1)
        if not finite.any():
            return 0
        x0, y0 = np.floor(np.nanmin(uv[finite], axis=0)).astype(int)
        x1, y1 = np.ceil(np.nanmax(uv[finite], axis=0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    patch = depth[y0:y1, x0:x1]
    return int(np.count_nonzero((patch > 100) & (patch < 1500)))


def _quality(sharpness, face_pixels, valid_landmarks, residual_mm, quality) -> float:
    sharp = np.clip((sharpness - quality.min_sharpness)
                    / max(quality.reference_sharpness - quality.min_sharpness, 1e-6), 0.0, 1.0)
    depth = np.clip(face_pixels / max(float(quality.reference_face_depth_pixels), 1.0), 0.0, 1.0)
    landmarks = np.clip((valid_landmarks - quality.min_valid_landmarks)
                        / max(68.0 - quality.min_valid_landmarks, 1.0), 0.0, 1.0)
    fit = np.clip((quality.max_residual_mm - residual_mm)
                  / max(quality.max_residual_mm - quality.reference_residual_mm, 1e-6), 0.0, 1.0)
    factors = np.asarray([max(float(sharp), 1e-6), max(float(depth), 1e-6),
                          max(float(landmarks), 1e-6), max(float(fit), 1e-6)])
    return float(np.exp(np.log(factors).mean()))


def _euler(rotation):
    from scipy.spatial.transform import Rotation

    pitch, yaw, roll = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    return float(pitch), float(yaw), float(roll)


def _mouth_ratio(xyz, valid):
    left, right = INTEROCULAR_PAIR
    if not (valid[left] and valid[right]):
        return None
    span = float(np.linalg.norm(xyz[left] - xyz[right]))
    if span <= 1e-6:
        return None
    gap = inner_lip_gap_mm(xyz, valid)
    return None if gap is None else float(gap / 1000.0 / span)


def _failed_track(index, entry, group_index, reason) -> _Track:
    record = _Track(source_index=index, group=f"track{max(group_index, 0):02d}", timestamp_s=entry.timestamp_s,
                    yaw_deg=float("nan"), pitch_deg=float("nan"), roll_deg=float("nan"),
                    quality=0.0, residual_mm=float("inf"), sharpness=0.0, face_depth_pixels=0,
                    valid_landmarks=0, detected=False, usable=False, reason=reason, inner_lip_gap_mm=None)
    record.__dict__["boundary_reasons"] = ["landmark_detection_failed"]
    return record


def _serialise(track: _Track, sequence) -> dict:
    entry = sequence.entries[track.source_index]
    return {
        "source_index": track.source_index,
        "timestamp_s": track.timestamp_s,
        "yaw_deg": None if not np.isfinite(track.yaw_deg) else round(track.yaw_deg, 4),
        "pitch_deg": None if not np.isfinite(track.pitch_deg) else round(track.pitch_deg, 4),
        "roll_deg": None if not np.isfinite(track.roll_deg) else round(track.roll_deg, 4),
        "quality": round(track.quality, 4),
        "fit_residual_mm": None if not np.isfinite(track.residual_mm) else round(track.residual_mm, 4),
        "sharpness": round(track.sharpness, 3),
        "face_depth_pixels": track.face_depth_pixels,
        "valid_landmarks": track.valid_landmarks,
        "inner_lip_gap_mm": track.inner_lip_gap_mm,
        "detected": track.detected,
        "usable": track.usable,
        "unusable_reason": track.reason,
        "tracking_group": track.group,
        "boundary_reasons": track.__dict__.get("boundary_reasons", []),
        "source": entry.source,
        "source_frame": entry.source_frame,
        "source_group": entry.group,
        "state_label": entry.state,
        "view_label": entry.view,
    }


def _summary(records: list[dict], sequence, role: str) -> dict:
    usable = [row for row in records if row["usable"]]
    reasons: dict[str, int] = {}
    for row in records:
        if row["unusable_reason"]:
            reasons[row["unusable_reason"]] = reasons.get(row["unusable_reason"], 0) + 1
    yaw = np.asarray([row["yaw_deg"] for row in usable if row["yaw_deg"] is not None], dtype=np.float64)
    groups = list(dict.fromkeys(row["tracking_group"] for row in records))
    return {
        "role": role,
        "frames": len(records),
        "usable_frames": len(usable),
        "tracking_groups": groups,
        "tracking_group_count": len(groups),
        "unusable_reasons": reasons,
        "yaw_span_deg": None if not len(yaw) else [float(yaw.min()), float(yaw.max())],
        "yaw_left_frames": int((yaw < 0).sum()),
        "yaw_right_frames": int((yaw > 0).sum()),
        "expression_boundary_count": int(sum(1 for row in records if "expression_change" in row["boundary_reasons"])),
        "input": sequence.metadata.as_dict(),
        "note": "yaw is a landmark-rigid angle against the neutral template, not an independent goniometer",
    }
