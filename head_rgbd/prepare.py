"""Prepare real RGB-D captures as per-frame metric fitting observations.

The raw capture is never modified. Invalid landmark depth is represented as
NaN plus an explicit validity mask; it is never converted into a zero-depth
constraint.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_rgbd.audit import dump, sha
from head_rgbd.contracts import (
    CAPTURE_SCHEMA,
    COORDINATE_CONVENTION,
    DEPTH_UNIT,
    FRAME_MANIFEST_SCHEMA,
    MAX_TEMPORAL_GAP_S,
    MESH_UNIT,
    SEQUENCE_SCHEMA,
)
from head_rgbd.paths import asset_paths, configure_environment, missing_assets


VALID_STATES = {"closed", "open"}
VALID_VIEWS = {"front", "left", "right"}


def backproject(uv, depth, intr):
    """Backproject image points; invalid points are NaN and explicitly masked."""
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth)
    intr = np.asarray(intr, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if depth.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    if intr.shape != (4,):
        raise ValueError("intrinsics must have shape (4,)")

    pix = np.rint(np.nan_to_num(uv, nan=-1.0)).astype(np.int64)
    h, w = depth.shape
    valid = (
        np.isfinite(uv).all(axis=1)
        & (pix[:, 0] >= 0)
        & (pix[:, 0] < w)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] < h)
    )
    z = np.full(len(uv), np.nan, dtype=np.float64)
    ids = np.flatnonzero(valid)
    z[ids] = depth[pix[ids, 1], pix[ids, 0]].astype(np.float64) * 0.001
    valid &= np.isfinite(z) & (z > 0.1) & (z < 1.5)

    fx, fy, cx, cy = intr
    xyz = np.full((len(uv), 3), np.nan, dtype=np.float32)
    xyz[valid, 0] = (uv[valid, 0] - cx) * z[valid] / fx
    xyz[valid, 1] = -(uv[valid, 1] - cy) * z[valid] / fy
    xyz[valid, 2] = -z[valid]
    return xyz, valid


def rigid(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("rigid alignment requires equal (N, 3) arrays")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("rigid alignment requires at least three finite pairs")
    x = source - source.mean(axis=0)
    y = target - target.mean(axis=0)
    u, singular, vh = np.linalg.svd(x.T @ y)
    if singular[1] < 1e-10:
        raise ValueError("degenerate rigid alignment")
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(vh.T @ u.T)
    rotation = vh.T @ fix @ u.T
    translation = target.mean(axis=0) - source.mean(axis=0) @ rotation.T
    return rotation, translation


def track_overlap(
    previous_rgb,
    rgb,
    previous_depth,
    depth,
    intr,
    face_mask,
    upper_fraction=0.6,
    previous_intr=None,
):
    """Track upper-face or neck overlap and report a robust metric transform."""
    gray0 = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = face_mask.astype(np.uint8) * 255
    ys = np.flatnonzero(mask.any(axis=1))
    if not len(ys):
        return {"status": "lost", "reason": "no_face_mask"}
    if upper_fraction < 1:
        cutoff = int(np.percentile(ys, upper_fraction * 100))
        mask[cutoff:] = 0
    points = cv2.goodFeaturesToTrack(gray0, 160, 0.01, 6, mask=mask)
    if points is None:
        return {"status": "lost", "reason": "no_tracks"}
    tracked, ok_forward, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None)
    if tracked is None or ok_forward is None:
        return {"status": "lost", "reason": "forward_flow_failed"}
    tracked_back, ok_backward, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, tracked, None)
    if tracked_back is None or ok_backward is None:
        return {"status": "lost", "reason": "backward_flow_failed"}
    keep = (
        ok_forward[:, 0].astype(bool)
        & ok_backward[:, 0].astype(bool)
        & (np.linalg.norm(tracked_back[:, 0] - points[:, 0], axis=1) < 1.0)
    )
    previous_points, valid_previous = backproject(
        points[:, 0],
        previous_depth,
        intr if previous_intr is None else previous_intr,
    )
    current_points, valid_current = backproject(tracked[:, 0], depth, intr)
    keep &= valid_previous & valid_current
    if keep.sum() < 8:
        return {
            "status": "lost",
            "reason": "insufficient_overlap",
            "tracks": int(keep.sum()),
        }

    previous_points = previous_points[keep]
    current_points = current_points[keep]
    for _ in range(3):
        rotation, translation = rigid(previous_points, current_points)
        residual = np.linalg.norm(
            previous_points @ rotation.T + translation - current_points,
            axis=1,
        )
        inlier = residual < max(0.003, float(np.median(residual) * 2.5))
        if inlier.sum() < 8:
            break
        previous_points = previous_points[inlier]
        current_points = current_points[inlier]
    rotation, translation = rigid(previous_points, current_points)
    error = float(
        np.median(
            np.linalg.norm(
                previous_points @ rotation.T + translation - current_points,
                axis=1,
            )
        )
    )
    return {
        "status": "tracked" if error < 0.008 else "lost",
        "tracks": len(previous_points),
        "median_m": error,
        "previous_to_current_R": rotation,
        "previous_to_current_t": translation,
        "direction": "p_current = R @ p_previous + t",
        "independent_extrinsics": False,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_state_view(entry: dict) -> None:
    if entry.get("state") not in VALID_STATES or entry.get("view") not in VALID_VIEWS:
        raise ValueError(
            f"unsupported state/view {entry.get('state')!r}/{entry.get('view')!r}"
        )


def _validate_capture_manifest(manifest: dict, path: Path) -> None:
    required = {
        "schema": CAPTURE_SCHEMA,
        "status": "complete",
        "depth_unit": "mm",
        "mesh_unit": "m",
        "coordinate_convention": "X_right_Y_up_Z_backward",
        "rgb_depth_aligned": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"capture {key} must be {expected!r}, got {manifest.get(key)!r}"
            )
    segment_ids = set()
    for segment in manifest.get("segments", []):
        _validate_state_view(segment)
        segment_id = segment.get("id")
        if not segment_id or segment_id in segment_ids:
            raise ValueError(f"invalid or duplicate capture segment id: {segment_id!r}")
        segment_ids.add(segment_id)
        npz_path = (path.parent / segment["file"]).resolve()
        if npz_path.parent != path.parent or not npz_path.is_file():
            raise ValueError(f"capture segment is outside session or missing: {npz_path}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _capture_sources(capture_path: Path) -> Iterator[dict]:
    manifest_path = capture_path / "session.json" if capture_path.is_dir() else capture_path
    manifest_path = manifest_path.resolve()
    if manifest_path.name != "session.json" or not manifest_path.is_file():
        raise ValueError(f"capture manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    _validate_capture_manifest(manifest, manifest_path)
    manifest_hash = sha(manifest_path)
    for segment in manifest["segments"]:
        npz_path = (manifest_path.parent / segment["file"]).resolve()
        yield {
            "entry": {
                "view_id": segment["id"],
                "segment": segment["id"],
                "state": segment["state"],
                "view": segment["view"],
            },
            "arrays": _load_npz(npz_path),
            "frame_records": segment.get("frame_records", []),
            "provenance": {
                "kind": "capture_segment",
                "path": str(npz_path),
                "sha256": sha(npz_path),
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_hash,
            },
            "capture_manifest": manifest,
        }


def _frame_manifest_sources(frame_manifest: Path) -> Iterator[dict]:
    frame_manifest = frame_manifest.resolve()
    manifest = _load_json(frame_manifest)
    if (
        manifest.get("schema") != FRAME_MANIFEST_SCHEMA
        or manifest.get("depth_unit") != "mm"
        or manifest.get("rgb_depth_aligned") is not True
        or manifest.get("coordinate_convention", "X_right_Y_up_Z_backward")
        != "X_right_Y_up_Z_backward"
    ):
        raise ValueError(
            "frame manifest requires head_rgbd.raw.v1, mm depth, aligned RGB-D "
            "and X_right_Y_up_Z_backward coordinates"
        )
    for index, entry in enumerate(manifest.get("chunks", [])):
        _validate_state_view(entry)
        for key in ("segment", "view_id"):
            value = entry.get(key)
            if not value or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in value
            ):
                raise ValueError(f"unsafe or missing chunk {key}: {value!r}")
        npz_path = (frame_manifest.parent / entry["file"]).resolve()
        if npz_path.parent != frame_manifest.parent or not npz_path.is_file():
            raise ValueError(f"frame chunk is outside manifest or missing: {npz_path}")
        yield {
            "entry": entry,
            "arrays": _load_npz(npz_path),
            "frame_records": entry.get("frame_records", []),
            "provenance": {
                "kind": "frame_chunk",
                "index": index,
                "path": str(npz_path),
                "sha256": sha(npz_path),
                "manifest": str(frame_manifest),
                "manifest_sha256": sha(frame_manifest),
            },
            "capture_manifest": None,
        }


def _normalise_array_contract(arrays: dict, count: int) -> None:
    required = ("rgb", "depth", "intrinsics", "timestamps_s")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"segment is missing arrays: {missing}")
    for key in required:
        if len(arrays[key]) != count:
            raise ValueError(f"{key} has {len(arrays[key])} frames; expected {count}")
    if arrays["intrinsics"].shape == (4,):
        arrays["intrinsics"] = np.tile(arrays["intrinsics"], (count, 1))
    if arrays["intrinsics"].shape != (count, 4):
        raise ValueError("intrinsics must have shape (N, 4)")


def _landmark_detection(rgb: np.ndarray, detect_face_bbox, get_lmk_single):
    bbox = detect_face_bbox(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if bbox is None:
        raise ValueError("face_bbox_not_detected")
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= 1 or bbox[3] <= 1:
        raise ValueError("invalid_face_bbox")
    landmarks = np.asarray(get_lmk_single(rgb, bbox), dtype=np.float64)
    if landmarks.size != 136:
        raise ValueError(f"TUFA returned {landmarks.size} values; expected 136")
    landmarks = landmarks.reshape(68, 2)
    return bbox, landmarks


def _serializable_error(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--capture", type=Path, help="head_rgbd.capture.v1 session or directory")
    inputs.add_argument("--frame-manifest", type=Path, help="external head_rgbd.raw.v1 chunks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, help="runtime root containing assets")
    parser.add_argument(
        "--landmark-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="TUFA inference device; UniFace parsing always uses CPU",
    )
    args = parser.parse_args()

    paths = configure_environment(args.runtime)
    absent = missing_assets(paths)
    if absent:
        raise FileNotFoundError(
            "runtime assets are missing; run the assets command first: "
            + ", ".join(absent)
        )
    if args.landmark_device == "cuda":
        os.environ["HEADRGBD_TUFA_DEVICE"] = "cuda:0"
    elif args.landmark_device == "cpu":
        os.environ["HEADRGBD_TUFA_DEVICE"] = "cpu"
    else:
        os.environ.pop("HEADRGBD_TUFA_DEVICE", None)

    from core import mask_detection
    from core.camera import Camera
    from core.face_test import detect_face_bbox
    from third_party.tufa.Test import get_lmk_single

    mask_detection.require_local_parser()
    mask_detection._ONNX_PROVIDERS = ["CPUExecutionProvider"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sources = []
    rows = []
    counters: dict[str, int] = {}
    last_segment = None
    last_timestamp = None
    last_frame = None
    next_tracking_barrier = None
    capture_manifest_seen = None
    all_assets = asset_paths(paths.runtime)

    source_iter = (
        _capture_sources(args.capture)
        if args.capture
        else _frame_manifest_sources(args.frame_manifest)
    )
    for source in source_iter:
        entry = source["entry"]
        arrays = source["arrays"]
        records = source["frame_records"]
        segment = entry["segment"]
        provenance = source["provenance"]
        sources.append(provenance)
        if last_segment is not None and segment != last_segment:
            last_frame = None
            last_timestamp = None
            next_tracking_barrier = (
                f"capture_segment_boundary:{last_segment}->{segment}"
            )
        if source["capture_manifest"] is not None:
            capture_manifest_seen = source["capture_manifest"]
        try:
            count = len(arrays["rgb"])
            _normalise_array_contract(arrays, count)
        except (KeyError, ValueError) as exc:
            rows.append(
                {
                    "id": entry.get("view_id", f"segment_{len(sources) - 1}"),
                    "segment": segment,
                    "state": entry["state"],
                    "view": entry["view"],
                    "status": "excluded",
                    "reason": f"segment_contract_failed: {_serializable_error(exc)}",
                    "source": provenance["path"],
                    "source_sha256": provenance["sha256"],
                }
            )
            last_frame = None
            last_timestamp = None
            next_tracking_barrier = (
                f"chunk_contract_failed:{_serializable_error(exc)}"
            )
            continue

        supplied_uv = arrays.get("uv")
        supplied_valid = arrays.get("landmark_valid")
        if supplied_uv is not None:
            supplied_uv = np.asarray(supplied_uv)
            if supplied_uv.shape != (count, 68, 2):
                rows.append(
                    {
                        "id": entry["view_id"],
                        "segment": segment,
                        "state": entry["state"],
                        "view": entry["view"],
                        "status": "excluded",
                        "reason": "landmark_contract_failed: uv must have shape (N, 68, 2)",
                        "source": provenance["path"],
                        "source_sha256": provenance["sha256"],
                    }
                )
                last_frame = None
                last_timestamp = None
                next_tracking_barrier = "landmark_contract_failed"
                continue

        for frame_index in range(count):
            view_id = entry["view_id"]
            index = counters.get(view_id, 0)
            counters[view_id] = index + 1
            frame_id = f"{view_id}_{index}"
            rgb = arrays["rgb"][frame_index]
            depth = arrays["depth"][frame_index]
            intr = np.asarray(arrays["intrinsics"][frame_index], dtype=np.float64)
            timestamp = float(arrays["timestamps_s"][frame_index])
            record = records[frame_index] if frame_index < len(records) else {}
            tracking_barrier = next_tracking_barrier
            next_tracking_barrier = None
            row: dict[str, Any] = {
                "id": frame_id,
                "segment": segment,
                "state": entry["state"],
                "view": entry["view"],
                "index": index,
                "timestamp_s": timestamp,
                "source": provenance["path"],
                "source_sha256": provenance["sha256"],
                "source_frame": frame_index,
                "status": "excluded",
                "reason": None,
                "independent_extrinsics": False,
                "acquisition": record,
                "tracking_barrier": tracking_barrier,
            }
            rows.append(row)
            timestamp_for_state_update = None
            try:
                if not np.isfinite(timestamp):
                    raise ValueError("invalid_timestamp")
                if segment == last_segment and last_timestamp is not None:
                    if timestamp <= last_timestamp:
                        raise ValueError("nonmonotonic_timestamp")
                    if timestamp - last_timestamp >= MAX_TEMPORAL_GAP_S:
                        tracking_barrier = (
                            f"temporal_gap_{timestamp - last_timestamp:.3f}s"
                        )
                        row["tracking_barrier"] = tracking_barrier
                        last_frame = None
                        last_timestamp = None
                timestamp_for_state_update = timestamp
                if tracking_barrier is not None:
                    last_frame = None
                if rgb.shape != (*depth.shape, 3):
                    raise ValueError("unaligned_rgb_depth_sizes")
                if rgb.dtype != np.uint8 or depth.dtype != np.uint16:
                    raise ValueError("rgb must be uint8 and depth must be uint16 millimetres")
                if not np.isfinite(depth).all() or np.any(depth < 0):
                    raise ValueError("invalid_depth_values")
                if (
                    not np.isfinite(intr).all()
                    or intr.shape != (4,)
                    or min(intr[:2]) <= 0
                ):
                    raise ValueError("invalid_intrinsics")

                if supplied_uv is None:
                    bbox, uv = _landmark_detection(rgb, detect_face_bbox, get_lmk_single)
                    uv_valid = np.isfinite(uv).all(axis=1)
                    landmark_source = "YUNET_TUFA_SINGLE_FRAME"
                else:
                    uv = np.asarray(supplied_uv[frame_index], dtype=np.float64)
                    uv_valid = np.isfinite(uv).all(axis=1)
                    if supplied_valid is not None:
                        uv_valid &= np.asarray(supplied_valid[frame_index], dtype=bool)
                    bbox = None
                    landmark_source = "PROVIDED_PER_FRAME_DETECTIONS"

                xyz, depth_valid = backproject(uv, depth, intr)
                valid = uv_valid & depth_valid
                xyz[~valid] = np.nan
                if valid[27:48].sum() < 6:
                    raise ValueError("insufficient_upper_face_depth")

                segmentation_data = mask_detection.get_face_data_single(
                    Camera(*intr.tolist()),
                    rgb,
                    depth,
                    bbox_xywh=bbox,
                )
                segmentation = segmentation_data["parsing"]
                face = np.isin(segmentation, mask_detection.FACE_LABELS)
                face = cv2.erode(
                    face.astype(np.uint8), np.ones((3, 3), np.uint8)
                ).astype(bool)
                neck = segmentation == 14
                ys, xs = np.where(face & (depth > 100) & (depth < 1500))
                if len(xs) < 200:
                    raise ValueError("insufficient_face_depth")
                y0, y1 = max(0, int(ys.min())), min(depth.shape[0], int(ys.max()) + 1)
                x0, x1 = max(0, int(xs.min())), min(depth.shape[1], int(xs.max()) + 1)
                crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                sharpness = float(
                    cv2.Laplacian(cv2.resize(crop, (128, 128)), cv2.CV_64F).var()
                )
                overlap = (
                    track_overlap(
                        last_frame["rgb"],
                        rgb,
                        last_frame["depth"],
                        depth,
                        intr,
                        last_frame["face"],
                        previous_intr=last_frame["intrinsics"],
                    )
                    if last_frame is not None
                    else {"status": "segment_start"}
                )
                neck_overlap = (
                    track_overlap(
                        last_frame["rgb"],
                        rgb,
                        last_frame["depth"],
                        depth,
                        intr,
                        last_frame["neck"],
                        upper_fraction=1.0,
                        previous_intr=last_frame["intrinsics"],
                    )
                    if last_frame is not None
                    else {"status": "segment_start"}
                )
                neck_valid_pixels = int(
                    np.count_nonzero(neck & (depth > 100) & (depth < 1500))
                )
                degradation = []
                if neck_valid_pixels < 30:
                    degradation.append("neck_depth_unavailable; head reconstruction continues")
                if overlap.get("status") == "lost":
                    degradation.append("adjacent_head_tracking_lost; landmark relocalization allowed")
                if neck_overlap.get("status") == "lost":
                    degradation.append("adjacent_neck_tracking_lost; separate component allowed")

                frame_file = f"{frame_id}.npz"
                np.savez_compressed(
                    output / frame_file,
                    rgb=rgb,
                    depth=depth,
                    intrinsics=intr,
                    uv=uv,
                    xyz=xyz,
                    valid=valid,
                    landmark2d_valid=uv_valid,
                    face_mask=face,
                    neck_mask=neck,
                    timestamp_s=np.asarray(timestamp, dtype=np.float64),
                    source_frame=np.asarray(frame_index, dtype=np.int64),
                )
                row.update(
                    status="candidate",
                    reason=None,
                    file=frame_file,
                    sharpness=sharpness,
                    face_depth_pixels=len(xs),
                    neck_depth_pixels=neck_valid_pixels,
                    landmark_source=landmark_source,
                    face_bbox_xywh=None if bbox is None else bbox.tolist(),
                    overlap=overlap,
                    neck_overlap=neck_overlap,
                    reinitialized=overlap.get("status") == "lost",
                    degraded=degradation,
                    landmark_depth_invalid=np.flatnonzero(~valid).tolist(),
                    alignment_evidence=(
                        "capture declares HW D2C with equal RGB/depth dimensions"
                        if source["capture_manifest"] is not None
                        else "external manifest declares aligned RGB-D"
                    ),
                )
                last_frame = {
                    "rgb": rgb,
                    "depth": depth,
                    "face": face,
                    "neck": neck,
                    "intrinsics": intr,
                }
                last_segment = segment
                last_timestamp = timestamp
                print(f"{frame_id}: candidate {overlap.get('status')}", flush=True)
            except (KeyError, ValueError, RuntimeError, cv2.error) as exc:
                row["reason"] = _serializable_error(exc)
                last_frame = None
                if timestamp_for_state_update is not None:
                    last_segment = segment
                    last_timestamp = timestamp_for_state_update
                next_tracking_barrier = (
                    f"previous_frame_excluded:{row['id']}:{row['reason']}"
                )
                print(f"{frame_id}: excluded {row['reason']}", flush=True)

    if not rows:
        raise ValueError("no raw capture frames were found")
    candidate_count = sum(row["status"] == "candidate" for row in rows)
    if candidate_count == 0:
        dump(
            output / "sequence.json",
            {
                "schema": SEQUENCE_SCHEMA,
                "depth_unit": "mm",
                "mesh_unit": "m",
                "coordinate_convention": "X_right_Y_up_Z_backward",
                "sources": sources,
                "frames": rows,
                "status": "no_candidate_frames",
            },
        )
        raise ValueError("all prepared frames were excluded; see sequence.json")

    capture_device = (
        capture_manifest_seen.get("device") if capture_manifest_seen is not None else None
    )
    manifest = {
        "schema": SEQUENCE_SCHEMA,
        "status": "prepared",
        "created_at": datetime.now().astimezone().isoformat(),
        "depth_unit": DEPTH_UNIT,
        "mesh_unit": MESH_UNIT,
        "coordinate_convention": COORDINATE_CONVENTION,
        "sources": sources,
        "frames": rows,
        "candidate_frames": candidate_count,
        "excluded_frames": len(rows) - candidate_count,
        "continuous_full_turn": False,
        "independent_extrinsics": False,
        "human_silhouettes": False,
        "elapsed_s": time.perf_counter() - start,
        "camera": capture_device,
        "camera_model": (
            "Orbbec Gemini 215"
            if capture_device is not None
            else "declared by external acquisition metadata; hardware not probed"
        ),
        "runtime_dir": str(paths.runtime),
        "landmark_runtime": {
            "tufa_device_requested": args.landmark_device,
            "uniface_providers": ["CPUExecutionProvider"],
            "tufa_weight": str(all_assets.tufa_weight),
            "tufa_weight_sha256": sha(all_assets.tufa_weight),
            "tufa_prompt": str(all_assets.tufa_prompt),
            "tufa_prompt_sha256": sha(all_assets.tufa_prompt),
            "yunet_model": str(all_assets.yunet_model),
            "yunet_model_sha256": sha(all_assets.yunet_model),
            "uniface_model": str(all_assets.uniface_dir / "parsing_resnet34.onnx"),
            "uniface_model_sha256": sha(all_assets.uniface_dir / "parsing_resnet34.onnx"),
        },
        "timing_scope": (
            "prepare wall time includes per-frame TUFA/YuNet/UniFace inference, "
            "tracking, masks and NPZ export; model load is included on first use"
        ),
        "limitations": [
            "guided view labels are not independent pose measurements",
            "HW D2C is recorded from SDK configuration and equal dimensions",
            "no independent extrinsics, manual contour or reference scan",
            "invalid landmark depth remains absent and is never a zero constraint",
        ],
    }
    dump(output / "sequence.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output / "sequence.json"),
                "candidate_frames": candidate_count,
                "excluded_frames": len(rows) - candidate_count,
                "elapsed_s": manifest["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
