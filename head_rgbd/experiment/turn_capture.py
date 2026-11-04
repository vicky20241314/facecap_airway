"""Capture a frontal anchor and a continuous closed-mouth RGB-D turn.

This experimental entry point leaves the production four-pose capture unchanged.
Run with ``python -m head_rgbd.experiment.turn_capture``. Press SPACE once for
each clip in the preview window; ESC cancels and preserves a partial manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pyorbbecsdk

from core.capture import OrbbecCapture, _camera_param_dict, _device_info_dict, _profile_dict
from head_rgbd.capture_session import _CapturePreview, _save_segment, _segment_quality
from head_rgbd.paths import configure_environment, create_exclusive_dir
from head_rgbd.reporting import write_json


def _validate_frame(frame):
    if frame.rgb.dtype != np.uint8 or frame.depth.dtype != np.uint16:
        raise ValueError("Capture requires uint8 RGB and uint16 millimetre depth")
    if frame.rgb.shape != (*frame.depth.shape, 3) or frame.depth.shape != (720, 1280):
        raise ValueError("Hardware-aligned 1280x720 RGB-D is required")
    intrinsics = np.asarray(frame.intrinsics)
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0):
        raise ValueError("Capture has invalid color camera intrinsics")


def _write_role_manifests(folder, manifest):
    """Keep the unlabeled turn clip out of the frontal anchor input."""
    payloads = {}
    for role in ("anchor", "turn"):
        segments = [segment for segment in manifest["segments"]
                    if segment.get("acquisition_role") == role]
        if not segments:
            raise ValueError(f"Capture has no {role} segment")
        payloads[role] = dict(manifest, segments=segments, input_role=role)
    for role, payload in payloads.items():
        write_json(Path(folder) / f"{role}_session.json", payload, exclusive=True)


def _record(cap, window, count, interval_s, instruction, *, turn=False):
    last_frame = time.monotonic()

    def read():
        nonlocal last_frame
        frame = cap.read_with_metadata()
        if frame is not None:
            _validate_frame(frame)
            last_frame = time.monotonic()
        elif time.monotonic() - last_frame > 10:
            raise TimeoutError("No aligned RGB-D frame received for 10 seconds")
        return frame

    print(instruction + " | Press SPACE in the preview to start", flush=True)
    while True:
        if window.poll() == 32:
            break
        frame = read()
        if frame is not None:
            window.show(frame, instruction, "SPACE: start | ESC: cancel")
        if window.poll() == 32:
            break

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        window.poll()
        frame = read()
        if frame is not None:
            window.show(frame, instruction, f"Starting in {max(1, int(deadline - time.monotonic()) + 1)}...")
        window.poll()

    frames = []
    next_sample = time.monotonic()
    started = next_sample
    while len(frames) < count:
        window.poll()
        frame = read()
        now = time.monotonic()
        if frame is None:
            continue
        fraction = len(frames) / count
        cue = "HOLD STILL, FACE FRONT"
        if turn:
            cue = ("SLOWLY TURN TO IMAGE-LEFT" if fraction < .3 else
                   "SLOWLY TURN TO IMAGE-RIGHT" if fraction < .8 else "RETURN TO FRONT")
        window.show(frame, instruction, f"{cue} | {now - started:.1f}s | {len(frames)}/{count}")
        window.poll()
        if now >= next_sample:
            frames.append(frame)
            next_sample = now + interval_s
    return frames


def capture(*, output=None, runtime=None, turn_frames=100, interval_s=.12, preview_window=None):
    if turn_frames < 3 or not .02 <= interval_s <= .18:
        raise ValueError("turn_frames must be >=3 and interval_s within .02-.18 seconds")
    paths = configure_environment(runtime)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    folder = create_exclusive_dir(output or paths.runtime / "captures" / f"{stamp}_continuous_turn")
    manifest = {
        "schema": "head_rgbd.capture.v1", "status": "started",
        "created_at": datetime.now().astimezone().isoformat(),
        "session_name": "continuous_closed_turn", "experiment_capture": True,
        "depth_unit": "mm", "mesh_unit": "m",
        "coordinate_convention": "X_right_Y_up_Z_backward",
        "rgb_depth_aligned": False,
        "alignment": {"mode": "HW_D2C", "independent_physical_verification": False},
        "sdk_version": str(pyorbbecsdk.get_version()), "segments": [],
        "limitations": ["Turn view is intentionally unlabeled; offline tracking measures yaw.",
                        "Hardware RGB-D alignment has not been independently calibrated."],
    }
    cap = None
    window = preview_window if preview_window is not None else _CapturePreview()
    try:
        cap = OrbbecCapture(timeout_ms=1000)
        cap.start()
        manifest.update(device=_device_info_dict(cap.device),
                        color_profile=_profile_dict(cap.color_profile),
                        depth_profile=_profile_dict(cap.depth_profile),
                        camera_param=_camera_param_dict(cap.pipeline),
                        depth_work_mode=str(cap.device.get_depth_work_mode().name))
        specs = [("closed_front", "front", 5, "ANCHOR: face front and keep mouth naturally closed"),
                 ("closed_turn", None, turn_frames, "TURN: mouth closed; center -> left -> right -> center")]
        for index, (identity, view, count, instruction) in enumerate(specs):
            frames = _record(cap, window, count, interval_s, instruction, turn=index == 1)
            filename = f"segment_{index:02d}_{identity}.npz"
            arrays = _save_segment(folder / filename, frames)
            records = [frame.metadata() for frame in frames]
            segment = {"id": identity, "state": "closed", "view": view, "file": filename,
                       "acquisition_role": "anchor" if index == 0 else "turn",
                       "frame_records": records, "quality": _segment_quality(arrays, records)}
            manifest["segments"].append(segment)
            write_json(folder / f"segment_{index:02d}.json", segment, exclusive=True)
            del arrays, records, frames
            print(f"saved {identity} ({count} frames)", flush=True)
        manifest["rgb_depth_aligned"] = True
        manifest["alignment"]["evidence"] = "HW D2C; every recorded RGB-D pair validated at 1280x720 with color intrinsics"
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        _write_role_manifests(folder, manifest)
        write_json(folder / "session.json", manifest, exclusive=True)
        print(f"CAPTURE_COMPLETE={folder}", flush=True)
        return folder
    except BaseException as exc:
        manifest.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        write_json(folder / "session.partial.json", manifest, exclusive=True)
        raise
    finally:
        try:
            window.close()
        finally:
            if cap is not None:
                cap.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--turn-frames", type=int, default=100)
    parser.add_argument("--interval-s", type=float, default=.12)
    args = parser.parse_args(argv)
    try:
        capture(output=args.output, runtime=args.runtime,
                turn_frames=args.turn_frames, interval_s=args.interval_s)
    except KeyboardInterrupt:
        print("Capture cancelled; partial data preserved.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
