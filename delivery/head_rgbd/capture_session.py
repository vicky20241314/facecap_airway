"""Guided, segment-aware RGB-D capture for Orbbec Gemini cameras."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from typing import Iterable

import cv2
import numpy as np
import pyorbbecsdk

from core.capture import (
    CapturedFrame,
    OrbbecCapture,
    _camera_param_dict,
    _device_info_dict,
    _profile_dict,
)
from head_rgbd.paths import configure_environment, ensure_runtime_layout, write_json_exclusive


CAPTURE_SCHEMA = "head_rgbd.capture.v1"
DEFAULT_VIEWS = ("closed:front", "closed:left", "closed:right", "open:front")
STATES = {"closed", "open"}
VIEWS = {"front", "left", "right"}
VIEW_INSTRUCTIONS = {
    ("closed", "front"): "Look straight at the camera; close your mouth naturally.",
    ("closed", "left"): "Keep the mouth closed and turn the head about 25-35 degrees to image-left.",
    ("closed", "right"): "Keep the mouth closed and turn the head about 25-35 degrees to image-right.",
    ("open", "front"): "Face the camera and open the mouth clearly without moving the body.",
}


def _safe_identity(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    if not cleaned:
        raise ValueError("session name has no usable characters")
    return cleaned[:80]


def parse_view_specs(values: Iterable[str] | None = None) -> list[tuple[str, str]]:
    specs = list(values or DEFAULT_VIEWS)
    result: list[tuple[str, str]] = []
    for raw in specs:
        parts = raw.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"view must be state:view, got {raw!r}")
        state, view = parts
        if state not in STATES or view not in VIEWS:
            raise ValueError(f"unsupported state/view {state!r}/{view!r}")
        if (state, view) in result:
            raise ValueError(f"duplicate state/view {state}/{view}")
        result.append((state, view))
    if ("closed", "front") not in result:
        raise ValueError("capture must include closed:front")
    if ("open", "front") not in result:
        raise ValueError("capture must include open:front for the calibrated mouth reference")
    if not any(state == "closed" and view in {"left", "right"} for state, view in result):
        raise ValueError("capture must include at least one real closed oblique view")
    return result


def _noneable_int(value) -> int:
    return -1 if value is None else int(value)


def _save_segment(path: Path, frames: list[CapturedFrame]) -> dict:
    if not frames:
        raise ValueError("cannot save an empty segment")
    arrays = {
        "rgb": np.stack([frame.rgb for frame in frames]).astype(np.uint8, copy=False),
        "depth": np.stack([frame.depth for frame in frames]).astype(np.uint16, copy=False),
        "intrinsics": np.stack([frame.intrinsics for frame in frames]).astype(np.float64, copy=False),
        "timestamps_s": np.asarray([frame.host_monotonic_s for frame in frames], dtype=np.float64),
        "hardware_timestamp_us": np.asarray(
            [_noneable_int(frame.hardware_timestamp_us) for frame in frames], dtype=np.int64
        ),
        "global_timestamp_us": np.asarray(
            [_noneable_int(frame.global_timestamp_us) for frame in frames], dtype=np.int64
        ),
        "system_timestamp_us": np.asarray(
            [_noneable_int(frame.system_timestamp_us) for frame in frames], dtype=np.int64
        ),
        "color_index": np.asarray(
            [_noneable_int(frame.color_index) for frame in frames], dtype=np.int64
        ),
        "depth_index": np.asarray(
            [_noneable_int(frame.depth_index) for frame in frames], dtype=np.int64
        ),
        "color_depth_sync_delta_ms": np.asarray(
            [
                np.nan if frame.color_depth_sync_delta_ms is None
                else frame.color_depth_sync_delta_ms
                for frame in frames
            ],
            dtype=np.float64,
        ),
        "depth_scale_mm": np.asarray(
            [np.nan if frame.depth_scale_mm is None else frame.depth_scale_mm for frame in frames],
            dtype=np.float64,
        ),
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return arrays


def _segment_quality(arrays: dict, metadata: list[dict]) -> dict:
    times = arrays["timestamps_s"]
    gaps = np.diff(times)
    sync = arrays["color_depth_sync_delta_ms"]
    finite_sync = sync[np.isfinite(sync)]
    fractions = np.asarray([row["valid_depth_fraction"] for row in metadata])
    return {
        "frames": len(times),
        "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "timestamp_monotonic": bool(np.all(gaps > 0)),
        "timestamp_median_gap_s": float(np.median(gaps)) if len(gaps) else None,
        "timestamp_max_gap_s": float(np.max(gaps)) if len(gaps) else None,
        "color_depth_sync_median_ms": float(np.median(finite_sync)) if len(finite_sync) else None,
        "color_depth_sync_p95_ms": float(np.percentile(finite_sync, 95)) if len(finite_sync) else None,
        "valid_depth_fraction_median": float(np.median(fractions)),
        "valid_depth_fraction_min": float(np.min(fractions)),
        "face_detection_frames": int(sum(bool(row.get("face_bbox_xywh")) for row in metadata)),
        "tracking_status": "pending_offline_prepare",
        "angle_status": "pending_offline_fit",
    }


def _detect_face_metadata(rgb: np.ndarray) -> dict:
    try:
        from core.face_test import detect_face_bbox

        box = detect_face_bbox(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if box is None:
            return {"face_bbox_xywh": None, "face_detection_status": "not_detected"}
        return {
            "face_bbox_xywh": [int(v) for v in box],
            "face_detection_status": "detected",
        }
    except Exception as exc:
        return {
            "face_bbox_xywh": None,
            "face_detection_status": "unavailable",
            "face_detection_error": f"{type(exc).__name__}: {exc}",
        }


def _capture_frames(
    capture: OrbbecCapture,
    count: int,
    interval_s: float,
    preview: bool,
    wait_for_input: bool,
    instruction: str,
) -> list[CapturedFrame]:
    result: list[CapturedFrame] = []
    window = "RGB-D head capture | SPACE=start ESC=cancel"
    if preview and wait_for_input:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            frame = capture.read_with_metadata()
            if frame is not None:
                shown = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(shown, instruction, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 230, 255), 2)
                cv2.putText(shown, "SPACE: start | ESC: cancel", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(window)
                raise KeyboardInterrupt("capture cancelled")
            if key == 32:
                break

    next_capture = time.monotonic()
    while len(result) < count:
        frame = capture.read_with_metadata()
        if frame is None:
            continue
        now = time.monotonic()
        if preview:
            shown = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                shown,
                f"Recording {len(result) + 1}/{count}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 230, 80),
                2,
            )
            cv2.imshow(window, shown)
            cv2.waitKey(1)
        if now < next_capture:
            continue
        result.append(frame)
        next_capture = now + interval_s
    if preview:
        cv2.destroyWindow(window)
    return result


def _wait_for_user(instruction: str, auto: bool, countdown_s: int) -> None:
    print(f"\n{instruction}", flush=True)
    if not auto:
        input("Press Enter when ready, then hold the pose...")
    for remaining in range(countdown_s, 0, -1):
        print(f"Recording starts in {remaining}...", flush=True)
        time.sleep(1)


def capture_session(
    *,
    runtime: str | Path | None,
    session_name: str,
    views: Iterable[str] | None = None,
    frames_per_view: int = 5,
    interval_s: float = 0.12,
    warmup_s: float = 0.8,
    countdown_s: int = 2,
    preview: bool = True,
    auto: bool = False,
) -> Path:
    """Capture a guided, no-overwrite multi-segment session."""
    if frames_per_view < 2:
        raise ValueError("frames_per_view must be at least 2")
    if not (0.02 <= interval_s <= 2.0):
        raise ValueError("interval_s must be within 0.02-2.0 seconds")
    specs = parse_view_specs(views)
    paths = configure_environment(runtime)
    layout = ensure_runtime_layout(paths.runtime)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = layout["captures"] / f"{stamp}_{_safe_identity(session_name)}"
    folder.mkdir(parents=False, exist_ok=False)

    manifest = {
        "schema": CAPTURE_SCHEMA,
        "status": "started",
        "created_at": datetime.now().astimezone().isoformat(),
        "session_name": session_name,
        "depth_unit": "mm",
        "mesh_unit": "m",
        "coordinate_convention": "X_right_Y_up_Z_backward",
        "rgb_depth_aligned": False,
        "alignment": {
            "mode": "HW_D2C",
            "evidence": "pending runtime frame validation",
            "independent_physical_verification": False,
        },
        "sdk_version": str(pyorbbecsdk.get_version()),
        "segments": [],
        "limitations": [
            "HW D2C alignment is recorded, not independently calibrated in this run",
            "view labels describe guided acquisition; measured yaw is fitted offline",
            "there are no copied front frames labeled as oblique observations",
        ],
    }
    capture = OrbbecCapture(timeout_ms=1000)
    try:
        capture.start()
        manifest["device"] = _device_info_dict(capture.device)
        manifest["depth_work_mode"] = str(capture.device.get_depth_work_mode().name)
        manifest["color_profile"] = _profile_dict(capture.color_profile)
        if capture.depth_profile is not None:
            manifest["depth_profile"] = _profile_dict(capture.depth_profile)
        manifest["camera_param"] = _camera_param_dict(capture.pipeline)
        if warmup_s > 0:
            deadline = time.monotonic() + warmup_s
            while time.monotonic() < deadline:
                capture.read_with_metadata()

        for segment_index, (state, view) in enumerate(specs):
            instruction = VIEW_INSTRUCTIONS[(state, view)]
            _wait_for_user(instruction, auto, countdown_s)
            frames = _capture_frames(
                capture,
                frames_per_view,
                interval_s,
                preview,
                not auto,
                instruction,
            )
            filename = f"segment_{segment_index:02d}_{state}_{view}.npz"
            arrays = _save_segment(folder / filename, frames)
            metadata = []
            for frame in frames:
                row = frame.metadata()
                row.update(_detect_face_metadata(frame.rgb))
                metadata.append(row)
            segment = {
                "id": f"{state}_{view}",
                "state": state,
                "view": view,
                "file": filename,
                "frame_records": metadata,
                "quality": _segment_quality(arrays, metadata),
            }
            manifest["segments"].append(segment)
            with (folder / f"segment_{segment_index:02d}.json").open(
                "x", encoding="utf-8"
            ) as stream:
                json.dump(segment, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"saved {segment['id']} ({len(frames)} frames)", flush=True)

        dimensions = {
            (row["width"], row["height"])
            for segment in manifest["segments"]
            for row in segment["frame_records"]
        }
        sync_values = [
            row["color_depth_sync_delta_ms"]
            for segment in manifest["segments"]
            for row in segment["frame_records"]
            if row["color_depth_sync_delta_ms"] is not None
        ]
        manifest["rgb_depth_aligned"] = dimensions == {(1280, 720)}
        manifest["alignment"].update(
            rgb_depth_aligned=manifest["rgb_depth_aligned"],
            evidence="HW D2C equal-size RGB/depth frames and color intrinsics on every segment",
            observed_color_depth_sync_median_ms=(
                float(np.median(sync_values)) if sync_values else None
            ),
        )
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        write_json_exclusive(folder / "session.json", manifest)
        print(str(folder / "session.json"), flush=True)
        return folder
    except BaseException as exc:
        manifest["status"] = "aborted"
        manifest["aborted_at"] = datetime.now().astimezone().isoformat()
        manifest["abort_reason"] = f"{type(exc).__name__}: {exc}"
        partial = folder / "session.partial.json"
        if not partial.exists():
            write_json_exclusive(partial, manifest)
        raise
    finally:
        capture.stop()


def load_capture_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"unsupported capture schema: {manifest.get('schema')!r}")
    if manifest.get("depth_unit") != "mm" or manifest.get("mesh_unit") != "m":
        raise ValueError("capture units must be mm depth and metre mesh")
    if manifest.get("coordinate_convention") != "X_right_Y_up_Z_backward":
        raise ValueError("unsupported capture coordinate convention")
    if manifest.get("rgb_depth_aligned") is not True:
        raise ValueError("capture did not declare verified RGB/depth alignment")
    segment_ids = set()
    for segment in manifest.get("segments", []):
        if segment.get("state") not in STATES or segment.get("view") not in VIEWS:
            raise ValueError("capture contains unsupported state/view labels")
        if segment.get("id") in segment_ids:
            raise ValueError("capture contains duplicate segment IDs")
        segment_ids.add(segment.get("id"))
        file = (path.parent / segment["file"]).resolve()
        if file.parent != path.parent or not file.is_file():
            raise ValueError(f"invalid segment path: {file}")
    return manifest
