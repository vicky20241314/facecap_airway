"""T2 live capture gate: saved-frame simulation, no camera required."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from head_rgbd import aruco_marker as am  # noqa: E402

try:  # pyorbbecsdk is in the project runtime; skip cleanly elsewhere
    from head_rgbd import capture_session as cs  # noqa: E402
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    cs = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

INTR = np.array([849.22216796875, 849.1190795898438, 640.1256103515625, 361.0])
PX_PER_MM = float(INTR[0] / 470.0)
CANVAS = 900
DEPTH_MM = 470


def _marker_rgb() -> np.ndarray:
    sticker = am.sticker_image(am.ArucoSpec(), PX_PER_MM)
    page = np.full((CANVAS, CANVAS), 255, np.uint8)
    y0 = (CANVAS - sticker.shape[0]) // 2
    x0 = (CANVAS - sticker.shape[1]) // 2
    page[y0:y0 + sticker.shape[0], x0:x0 + sticker.shape[1]] = sticker
    return np.dstack([page] * 3)


class Frame:
    def __init__(self, rgb, ts):
        self.rgb = rgb
        self.depth = np.full((CANVAS, CANVAS), DEPTH_MM, np.uint16)
        self.intrinsics = INTR
        self.host_monotonic_s = float(ts)
        self.hardware_timestamp_us = None
        self.global_timestamp_us = None
        self.system_timestamp_us = None
        self.color_index = None
        self.depth_index = None
        self.color_depth_sync_delta_ms = 0.0
        self.depth_scale_mm = 1.0

    def metadata(self):
        return {"host_monotonic_s": self.host_monotonic_s,
                "width": CANVAS, "height": CANVAS,
                "valid_depth_fraction": 1.0,
                "color_depth_sync_delta_ms": 0.0,
                "face_bbox_xywh": None}


VALID = _marker_rgb()
BLANK = np.full((CANVAS, CANVAS, 3), 255, np.uint8)


def _frames(valid: bool, count: int = 2):
    return [Frame(VALID if valid else BLANK, i * 0.1) for i in range(count)]


class FakeCapture:
    def __init__(self, *args, **kwargs):
        self.device = SimpleNamespace(get_depth_work_mode=lambda: SimpleNamespace(name="DEFAULT"))
        self.color_profile = object()
        self.depth_profile = None
        self.pipeline = object()

    def start(self):
        pass

    def stop(self):
        pass

    def read_with_metadata(self):
        return None


def _fake_save(path: Path, frames):
    path.write_bytes(b"")
    return {"timestamps_s": np.asarray([f.host_monotonic_s for f in frames]),
            "color_depth_sync_delta_ms": np.asarray([0.0 for _ in frames])}


class FakePreview:
    instances: list["FakePreview"] = []

    def __init__(self, *args, **kwargs):
        self.shown: list[str] = []
        self.last_marker = None
        self.closed = False
        FakePreview.instances.append(self)

    def show(self, frame, instruction, status):
        self.shown.append(status)

    def poll(self):
        return -1

    def close(self):
        self.closed = True


class FakeWindow:
    def __init__(self, polls):
        self._polls = iter(polls)
        self.shown: list[str] = []

    def poll(self):
        return next(self._polls, -1)

    def show(self, frame, instruction, status):
        self.shown.append(status)


@unittest.skipIf(cs is None, f"capture_session unavailable: {_IMPORT_ERROR}")
class LiveCaptureGate(unittest.TestCase):
    def _run(self, batches, *, auto: bool, fallback: bool, runtime: Path,
             preview: bool = False):
        queue = list(batches)

        def fake_capture_frames(capture, count, interval_s, preview, wait_for_input,
                                instruction, *, countdown_s=0, preview_window=None,
                                retry_reason=None):
            if not queue:
                raise AssertionError("capture_segment called more times than scripted")
            return queue.pop(0)

        patches = [
            mock.patch.multiple(
                cs,
                OrbbecCapture=FakeCapture,
                _capture_frames=fake_capture_frames,
                _save_segment=_fake_save,
                _detect_face_metadata=lambda rgb: {
                    "face_bbox_xywh": None, "face_detection_status": "not_detected"},
                _device_info_dict=lambda device: {},
                _profile_dict=lambda profile: {},
                _camera_param_dict=lambda pipeline: {},
                _CapturePreview=FakePreview,
            ),
            mock.patch("builtins.input",
                       side_effect=AssertionError("preview retake must not wait on console input")
                       if preview else None,
                       return_value=None if preview else ""),
        ]
        for patch in patches:
            patch.start()
        try:
            return cs.capture_session(
                runtime=runtime, session_name="marker_gate", views=None,
                frames_per_view=2, interval_s=0.02, warmup_s=0.0, countdown_s=0,
                preview=preview, auto=auto, allow_marker_fallback=fallback,
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_preview_retake_uses_window_not_console_input(self):
        FakePreview.instances.clear()
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._run([_frames(False), _frames(True), _frames(True),
                                _frames(True), _frames(True)],
                               auto=False, fallback=False, runtime=Path(tmp),
                               preview=True)
            manifest = json.loads((folder / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["marker_status"], "usable")
            self.assertEqual(len(FakePreview.instances), 1)
            self.assertTrue(FakePreview.instances[0].closed)

    def test_capture_frames_preview_names_the_retry_reason(self):
        window = FakeWindow([0, 32, 0, 0, 0])

        class FakeCamera:
            def read_with_metadata(self):
                return object()

        frames = cs._capture_frames(
            FakeCamera(), 1, 0.0, True, True, "instruction",
            preview_window=window, retry_reason="open front: size_insufficient_corners")
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(window.shown), 2)
        self.assertTrue(window.shown[0].startswith("RETRY:"))
        self.assertIn("size_insufficient_corners", window.shown[0])
        self.assertTrue(window.shown[1].startswith("Recording"))

    def test_manual_retake_then_valid_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._run([_frames(False), _frames(True), _frames(True),
                                _frames(True), _frames(True)],
                               auto=False, fallback=False, runtime=Path(tmp))
            manifest = json.loads((folder / "session.json").read_text(encoding="utf-8"))
            validation = json.loads((folder / "marker_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["marker_status"], "usable")
            self.assertEqual(validation["status"], "usable")
            self.assertEqual(validation["states"]["closed"]["status"], "observed")
            self.assertEqual(validation["states"]["closed"]["attempt"], 1)
            self.assertGreaterEqual(len(validation["states"]["closed"]["attempts"]), 3)
            front = next(s for s in manifest["segments"] if s["id"] == "closed_front")
            self.assertTrue(front["marker_validation"]["status"] == "observed")

    def test_auto_invalid_aborts_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cs.MarkerValidationError):
                self._run([_frames(False)], auto=True, fallback=False,
                          runtime=Path(tmp))
            folders = list((Path(tmp) / "captures").glob("*marker_gate*"))
            self.assertEqual(len(folders), 1)
            folder = folders[0]
            self.assertTrue((folder / "session.partial.json").is_file())
            self.assertTrue((folder / "marker_validation.json").is_file())
            self.assertTrue((folder / "segment_00_closed_front.npz").is_file())
            validation = json.loads((folder / "marker_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["status"], "failed")
            self.assertEqual(validation["states"]["closed"]["status"], "failed")
            self.assertEqual(validation["states"]["closed"]["reason"], "no_marker_detected")

    def test_explicit_fallback_completes_not_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._run([_frames(False), _frames(True),
                                _frames(True), _frames(True)],
                               auto=False, fallback=True, runtime=Path(tmp))
            manifest = json.loads((folder / "session.json").read_text(encoding="utf-8"))
            validation = json.loads((folder / "marker_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["marker_status"], "not_usable")
            self.assertEqual(validation["status"], "not_usable")
            self.assertEqual(validation["states"]["closed"]["status"], "failed")
            self.assertTrue(validation["states"]["closed"]["fallback_used"])
            front = next(s for s in manifest["segments"] if s["id"] == "closed_front")
            self.assertTrue(front["marker_validation"]["fallback_used"])


if __name__ == "__main__":
    unittest.main()