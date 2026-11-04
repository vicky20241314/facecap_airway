"""Capture controls: one window, one start confirmation, reliable cancellation."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.capture import CapturedFrame
from head_rgbd import capture_session as session


class FakeCapture:
    def __init__(self, **_):
        self.reads = 0
        self.started = False
        self.stopped = False
        self.device = SimpleNamespace(
            get_depth_work_mode=lambda: SimpleNamespace(name="test")
        )
        self.color_profile = object()
        self.depth_profile = None
        self.pipeline = object()

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read_with_metadata(self):
        self.reads += 1
        return CapturedFrame(
            rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            depth=np.full((4, 6), 500, dtype=np.uint16),
            intrinsics=np.array([100., 100., 3., 2.]),
            host_monotonic_s=self.reads * .1,
            hardware_timestamp_us=None, global_timestamp_us=None,
            system_timestamp_us=None, color_index=self.reads,
            depth_index=self.reads, color_depth_sync_delta_ms=0., depth_scale_mm=1.,
        )


class FakeClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        self.now += .01
        return self.now


class FakeGUI:
    def __init__(self):
        self.visible = True
        self.names = []
        self.images = []
        self.destroyed = []
        self.statuses = []
        self.instruction = ""
        self.status = ""
        self.started = set()
        self.key_override = None
        self.mouse_callback = None

    def named_window(self, name, _flags):
        self.names.append(name)

    def render(self, rgb, title, guidance, status, *, original_instruction, pressed=False):
        self.instruction = original_instruction
        self.status = status
        self.statuses.append(status)
        return np.zeros((780, 960, 3), dtype=np.uint8)

    def set_mouse_callback(self, _name, callback):
        self.mouse_callback = callback

    def click(self, x=776, y=712):
        self.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        self.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)

    def wait_key(self, _delay):
        if self.key_override is not None:
            return self.key_override()
        if self.status.startswith("SPACE:") and self.instruction not in self.started:
            self.started.add(self.instruction)
            return 32
        return -1

    def install(self, stack):
        for name, replacement in {
            "namedWindow": self.named_window,
            "resizeWindow": lambda *_: None,
            "setMouseCallback": self.set_mouse_callback,
            "imshow": lambda name, _: self.images.append(name),
            "getWindowProperty": lambda *_: float(self.visible),
            "waitKey": self.wait_key,
            "destroyWindow": self.destroyed.append,
        }.items():
            stack.enter_context(patch.object(session.cv2, name, replacement))
        stack.enter_context(patch.object(session.CapturePanel, "render",
                                        lambda _panel, *a, **kw: self.render(*a, **kw)))


class CapturePreviewTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.gui = FakeGUI()
        self.gui.install(self.stack)
        self.clock = FakeClock()
        self.stack.enter_context(patch.object(session.time, "monotonic", self.clock.monotonic))
        self.capture = FakeCapture()

    def test_four_poses_keep_one_window_until_session_end(self):
        window = session._CapturePreview()
        for index in range(4):
            frames = session._capture_frames(
                self.capture, 2, .02, True, True, f"pose {index}",
                countdown_s=1, preview_window=window,
            )
            self.assertEqual(len(frames), 2)
            self.assertEqual(self.gui.destroyed, [])
        window.close()
        self.assertEqual(self.gui.names, [session.CAPTURE_WINDOW])
        self.assertEqual(set(self.gui.images), {session.CAPTURE_WINDOW})
        self.assertEqual(self.gui.destroyed, [session.CAPTURE_WINDOW])
        self.assertEqual(len(self.gui.started), 4)
        statuses = self.gui.statuses
        self.assertLess(next(i for i, x in enumerate(statuses) if x.startswith("SPACE:")),
                        next(i for i, x in enumerate(statuses) if x.startswith("Recording ")))
        self.assertFalse(any(x.startswith("Starting in") for x in statuses))

    def test_space_at_first_poll_of_existing_window_is_not_lost(self):
        window = session._CapturePreview()
        window.show(self.capture.read_with_metadata(), "previous", "Recording 2/2")
        pending = iter([32])
        self.gui.key_override = lambda: next(pending, -1)
        frames = session._capture_frames(
            self.capture, 2, .02, True, True, "next", preview_window=window,
        )
        self.assertEqual(len(frames), 2)
        window.close()

    def test_window_close_cancels_without_recreating(self):
        window = session._CapturePreview()
        frame = self.capture.read_with_metadata()
        window.show(frame, "pose", "SPACE: start")
        self.gui.visible = False
        with self.assertRaisesRegex(KeyboardInterrupt, "window closed"):
            window.show(frame, "pose", "SPACE: start")
        self.assertEqual(len(self.gui.names), 1)
        self.assertEqual(len(self.gui.images), 1)
        window.close()

    def test_closed_window_opencv_error_becomes_cancellation(self):
        window = session._CapturePreview()
        window.show(self.capture.read_with_metadata(), "pose", "SPACE: start")
        with patch.object(session.cv2, "getWindowProperty", side_effect=cv2.error("gone")):
            with self.assertRaisesRegex(KeyboardInterrupt, "window closed"):
                window.poll()
        window.close()

    def test_escape_while_recording_releases_locally_owned_window(self):
        self.gui.key_override = lambda: 27 if self.gui.status.startswith("Recording ") else -1
        with self.assertRaisesRegex(KeyboardInterrupt, "cancelled"):
            session._capture_frames(self.capture, 2, .02, True, False, "pose")
        self.assertEqual(self.gui.destroyed, [session.CAPTURE_WINDOW])

    def test_missing_frames_still_pump_cancel_events(self):
        window = session._CapturePreview()
        window.show(self.capture.read_with_metadata(), "pose", "Recording 1/2")
        missing = []

        def missing_frame():
            missing.append(True)
            return None

        self.gui.key_override = lambda: 27 if missing else -1
        with patch.object(self.capture, "read_with_metadata", side_effect=missing_frame) as read:
            with self.assertRaises(KeyboardInterrupt):
                session._capture_frames(
                    self.capture, 2, .02, True, False, "pose", preview_window=window,
                )
        read.assert_called_once()
        window.close()

    def test_auto_records_without_start_key(self):
        self.gui.key_override = lambda: -1
        frames = session._capture_frames(
            self.capture, 2, .02, True, False, "pose", countdown_s=1,
        )
        self.assertEqual(len(frames), 2)
        self.assertTrue(any(x.startswith("Starting in") for x in self.gui.statuses))
        self.assertFalse(any(x.startswith("SPACE:") for x in self.gui.statuses))

    def test_no_preview_does_not_create_window(self):
        frames = session._capture_frames(self.capture, 2, .02, False, True, "pose")
        self.assertEqual(len(frames), 2)
        self.assertEqual(self.gui.names, [])

    def test_mouse_button_starts_each_pose_in_one_window(self):
        def click_when_ready():
            if self.gui.status.startswith("SPACE:") and self.gui.instruction not in self.gui.started:
                self.gui.started.add(self.gui.instruction)
                self.gui.click()
            return -1

        self.gui.key_override = click_when_ready
        window = session._CapturePreview()
        for index in range(4):
            frames = session._capture_frames(
                self.capture, 2, .02, True, True, f"pose {index}",
                countdown_s=1, preview_window=window,
            )
            self.assertEqual(len(frames), 2)
        self.assertEqual(len(self.gui.started), 4)
        self.assertEqual(self.gui.names, [session.CAPTURE_WINDOW])
        self.assertFalse(any(x.startswith("Starting in") for x in self.gui.statuses))
        window.close()

    def test_click_and_space_capture_next_frame_without_countdown(self):
        for action in ("mouse", "space"):
            with self.subTest(action=action):
                started_after_read = []

                def start():
                    if self.gui.status.startswith("SPACE:") and not started_after_read:
                        started_after_read.append(self.capture.reads)
                        if action == "mouse":
                            self.gui.click()
                        else:
                            return 32
                    return -1

                self.gui.status = ""
                self.gui.key_override = start
                frames = session._capture_frames(
                    self.capture, 2, .02, True, True, action, countdown_s=3,
                )
                self.assertEqual(frames[0].color_index, started_after_read[0] + 1)
                self.assertFalse(any(x.startswith("Starting in") for x in self.gui.statuses))

    def test_terminal_confirmation_also_starts_without_countdown(self):
        with patch("builtins.input", return_value="") as confirm, \
                patch.object(session.time, "sleep") as sleep:
            session._wait_for_user("pose", False, 3)
        confirm.assert_called_once()
        sleep.assert_not_called()

    def test_automatic_terminal_capture_keeps_configured_countdown(self):
        with patch("builtins.input", side_effect=AssertionError("unexpected prompt")), \
                patch.object(session.time, "sleep") as sleep:
            session._wait_for_user("pose", True, 2)
        self.assertEqual(sleep.call_count, 2)

    def test_mouse_requires_left_click_inside_button(self):
        self.gui.key_override = lambda: -1
        window = session._CapturePreview()
        window.show(self.capture.read_with_metadata(), "pose", "SPACE: start")
        self.gui.click(20, 20)
        self.gui.mouse_callback(cv2.EVENT_RBUTTONDOWN, 776, 712, 0, None)
        self.gui.mouse_callback(cv2.EVENT_RBUTTONUP, 776, 712, 0, None)
        self.gui.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 20, 0, None)
        self.gui.mouse_callback(cv2.EVENT_LBUTTONUP, 776, 712, 0, None)
        self.assertNotEqual(window.poll(), 32)
        self.gui.click()
        self.assertEqual(window.poll(), 32)
        self.assertNotEqual(window.poll(), 32)
        window.close()

    def test_mouse_during_countdown_or_recording_does_not_start_next_pose(self):
        self.gui.key_override = lambda: -1
        window = session._CapturePreview()
        frame = self.capture.read_with_metadata()
        window.show(frame, "pose", "SPACE: start")
        self.gui.click()
        for status in ("Starting in 2...", "Recording 1/2"):
            window.show(frame, "pose", status)
            self.gui.click()
            self.assertNotEqual(window.poll(), 32)
        window.show(frame, "next", "SPACE: start")
        self.assertNotEqual(window.poll(), 32)
        window.close()

    def test_escape_takes_priority_over_mouse_start(self):
        window = session._CapturePreview()
        window.show(self.capture.read_with_metadata(), "pose", "SPACE: start")
        self.gui.click()
        self.gui.key_override = lambda: 27
        with self.assertRaisesRegex(KeyboardInterrupt, "cancelled"):
            window.poll()
        window.close()

    def setup_session(self, root):
        captures = root / "captures"
        captures.mkdir()
        for name, value in {
            "configure_environment": SimpleNamespace(runtime=root),
            "ensure_runtime_layout": {"captures": captures},
            "OrbbecCapture": self.capture,
            "_device_info_dict": {},
            "_profile_dict": {},
            "_camera_param_dict": {},
            "_detect_face_metadata": {"face_bbox_xywh": None},
        }.items():
            self.stack.enter_context(patch.object(session, name, return_value=value))

    def test_preview_session_needs_no_terminal_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.setup_session(root)
            with patch("builtins.input", side_effect=AssertionError("terminal prompt")):
                folder = session.capture_session(
                    runtime=root, session_name="test", frames_per_view=2,
                    warmup_s=0, countdown_s=0,
                )
            manifest = json.loads((folder / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["segments"]), 4)
            self.assertEqual(len(list(folder.glob("*.npz"))), 4)
            self.assertTrue(self.capture.stopped)
            self.assertEqual(self.gui.names, [session.CAPTURE_WINDOW])
            self.assertEqual(self.gui.destroyed, [session.CAPTURE_WINDOW])

    def test_session_cancel_closes_window_stops_camera_and_saves_partial(self):
        self.gui.key_override = lambda: 27
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.setup_session(root)
            with self.assertRaises(KeyboardInterrupt):
                session.capture_session(
                    runtime=root, session_name="cancel", frames_per_view=2,
                    warmup_s=0, countdown_s=0,
                )
            partial = next((root / "captures").glob("*/session.partial.json"))
            manifest = json.loads(partial.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "aborted")
            self.assertTrue(self.capture.stopped)
            self.assertEqual(self.gui.destroyed, [session.CAPTURE_WINDOW])

    def test_headless_session_keeps_terminal_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.setup_session(root)
            with patch("builtins.input", return_value="") as ready:
                session.capture_session(
                    runtime=root, session_name="headless", frames_per_view=2,
                    warmup_s=0, countdown_s=0, preview=False,
                )
            self.assertEqual(ready.call_count, 4)
            self.assertEqual(self.gui.names, [])


class CaptureLauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt" and shutil.which("pwsh.exe"), "Windows PowerShell launcher")
    def test_powershell_subcommand_help_is_forwarded(self):
        root = Path(__file__).resolve().parents[1]
        # Invalid device is a guard: even a regression must fail before capture.
        result = subprocess.run(
            ["pwsh.exe", "-NoProfile", "-File", str(root / "headrgbd.ps1"),
             "run", "--help", "--device", "__help_guard__"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: headrgbd run", result.stdout)
        self.assertIn("--frames-per-view", result.stdout)


class CapturePanelTests(unittest.TestCase):
    def test_controls_are_readable_outside_camera_without_changing_raw_rgb(self):
        panel = session.CapturePanel()
        instruction = session.VIEW_INSTRUCTIONS[("closed", "front")]
        title, guidance = session.PREVIEW_INSTRUCTIONS[instruction]
        dark = np.zeros((180, 320, 3), dtype=np.uint8)
        bright = np.full_like(dark, 255)
        ready = panel.render(dark, title, guidance, "SPACE: start", original_instruction=instruction)
        bright_ready = panel.render(bright, title, guidance, "SPACE: start", original_instruction=instruction)
        self.assertEqual(ready.shape, (780, 960, 3))
        np.testing.assert_array_equal(ready[:104], bright_ready[:104])
        np.testing.assert_array_equal(ready[644:], bright_ready[644:])
        self.assertTrue(np.all(dark == 0))
        self.assertTrue(np.all(bright == 255))
        self.assertGreater(int(ready[710, 632, 1]), 150)
        recording = panel.render(dark, title, guidance, "Recording 1/5", original_instruction=instruction)
        self.assertLess(int(recording[710, 632, 1]), 100)

    def test_english_fallback_and_unusual_aspect_ratio_render(self):
        panel = session.CapturePanel()
        panel.chinese = False
        shown = panel.render(np.zeros((100, 10, 3), dtype=np.uint8), "title", "guidance",
                             "Starting in 2...", original_instruction="Look straight at the camera.")
        self.assertEqual(shown.shape, (780, 960, 3))


if __name__ == "__main__":
    unittest.main()
