"""Exercise Windows HighGUI with synthetic frames or live Orbbec RGB frames.

Only sends window messages to windows owned by this test process. Camera access
requires --camera; live frames stay in memory and input is process-local.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_rgbd import __version__
from head_rgbd.capture_session import CAPTURE_WINDOW, VIEW_INSTRUCTIONS, _CapturePreview, _capture_frames
from core.capture import OrbbecCapture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="store_true", help="use live Orbbec RGB frames in memory")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this check requires Windows HighGUI")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.FindWindowExW.argtypes = [wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR]
    user32.FindWindowExW.restype = wintypes.HWND
    user32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ScreenToClient.restype = wintypes.BOOL

    def windows():
        result = []

        @callback_type
        def visit(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == os.getpid() and user32.IsWindowVisible(hwnd):
                title, kind = ctypes.create_unicode_buffer(512), ctypes.create_unicode_buffer(128)
                user32.GetWindowTextW(hwnd, title, len(title))
                user32.GetClassNameW(hwnd, kind, len(kind))
                if title.value == CAPTURE_WINDOW or "HighGUI" in kind.value:
                    result.append({"hwnd": int(hwnd), "title": title.value, "class": kind.value})
            return True

        user32.EnumWindows(visit, 0)
        return result

    report = {
        "version": __version__, "opencv": cv2.__version__,
        "camera_used": args.camera,
        "frame_source": "live Orbbec RGB frames (memory only)" if args.camera else "synthetic RGB arrays",
        "event_delivery": "PostMessageW to this process's preview only",
        "poses": [], "events": [], "checks": {}, "ui_snapshots": {},
    }
    deadline = time.monotonic() + 30

    class Preview(_CapturePreview):
        def __init__(self):
            super().__init__()
            self.instruction = ""
            self.status = ""
            self.statuses = []
            self.action = "space"
            self.sent = set()
            self.handles = set()
            self.max_windows = 0

        def show(self, frame, instruction, status):
            shown = super().show(frame, instruction, status)
            self.instruction, self.status = instruction, status
            self.statuses.append(status)
            stage = "ready" if status.startswith("SPACE:") else (
                "countdown" if status.startswith("Starting in") else "recording"
            )
            if not args.camera and stage not in report["ui_snapshots"]:
                path = args.output.parent / f"preview_{stage}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(path), shown)
                report["ui_snapshots"][stage] = str(path)
            current = windows()
            self.max_windows = max(self.max_windows, len(current))
            assert len(current) == 1, current
            self.handles.add(current[0]["hwnd"])

        def poll(self):
            if time.monotonic() > deadline:
                raise RuntimeError("native preview event check timed out")
            token = (self.instruction, self.action)
            ready = (
                self.action == "close"
                or (self.action in {"space", "mouse"} and self.status.startswith("SPACE:"))
                or (self.action == "escape" and self.status.startswith("Recording "))
            )
            if self.opened and ready and token not in self.sent:
                current = windows()
                assert len(current) == 1, current
                hwnd = current[0]["hwnd"]
                event = {"action": self.action, "pose": self.instruction, "hwnd": hwnd}
                if self.action == "mouse":
                    # HighGUI maps native image-child coordinates back into the
                    # displayed image. Exercise that mapping after resizing too.
                    child = user32.FindWindowExW(hwnd, None, "HighGUI class", None)
                    assert child, "HighGUI image child not found"
                    left, top, width, height = cv2.getWindowImageRect(CAPTURE_WINDOW)
                    x1, y1, x2, y2 = self._panel.BUTTON
                    point = wintypes.POINT(round(left + (x1 + x2) / 2 / self._panel.SIZE[0] * width),
                                           round(top + (y1 + y2) / 2 / self._panel.SIZE[1] * height))
                    assert user32.ScreenToClient(child, ctypes.byref(point))
                    coordinates = (point.y << 16) | (point.x & 0xFFFF)
                    for message, flags in ((0x0201, 1), (0x0202, 0)):
                        if not user32.PostMessageW(child, message, flags, coordinates):
                            raise ctypes.WinError(ctypes.get_last_error())
                    event.update(image_rect=[left, top, width, height], mouse_child=int(child))
                else:
                    message, key = (0x0010, 0) if self.action == "close" else (
                        0x0102, 32 if self.action == "space" else 27
                    )
                    if not user32.PostMessageW(hwnd, message, key, 0):
                        raise ctypes.WinError(ctypes.get_last_error())
                report["events"].append(event)
                self.sent.add(token)
            return super().poll()

    class Camera:
        def __init__(self):
            self.frame = SimpleNamespace(rgb=np.full((360, 640, 3), 96, dtype=np.uint8))

        def read_with_metadata(self):
            time.sleep(.01)
            return self.frame

    camera = OrbbecCapture(timeout_ms=1000) if args.camera else Camera()
    preview = Preview()
    escaped = Preview()
    try:
        if args.camera:
            camera.start()
        for index, (pose, instruction) in enumerate(VIEW_INSTRUCTIONS.items()):
            preview.action = "mouse" if index % 2 == 0 else "space"
            if index == 2:
                cv2.resizeWindow(CAPTURE_WINDOW, 768, 624)
                cv2.waitKey(1)
            frames = _capture_frames(
                camera, 2, .02, True, True, instruction,
                countdown_s=1, preview_window=preview,
            )
            report["poses"].append({"pose": pose, "frames": len(frames), "windows": windows()})
        report["checks"]["one_native_window_across_four_poses"] = (
            len(preview.handles) == 1 and preview.max_windows == 1
        )
        report["checks"]["keyboard_and_mouse_started_poses"] = (
            len([e for e in report["events"] if e["action"] == "space"]) == 2
            and len([e for e in report["events"] if e["action"] == "mouse"]) == 2
            and all(p["frames"] == 2 for p in report["poses"])
        )
        report["checks"]["manual_start_has_no_countdown"] = not any(
            status.startswith("Starting in") for status in preview.statuses
        )
        mouse_events = [e for e in report["events"] if e["action"] == "mouse"]
        report["checks"]["mouse_works_after_window_resize"] = (
            len(mouse_events) == 2 and mouse_events[0]["image_rect"][2:] != mouse_events[1]["image_rect"][2:]
        )
        preview.action = "close"
        try:
            _capture_frames(camera, 2, .02, True, True, "cancel", preview_window=preview)
        except KeyboardInterrupt as exc:
            report["checks"]["native_close_cancels"] = "window closed" in str(exc)
        else:
            report["checks"]["native_close_cancels"] = False
        preview.close()
        report["checks"]["close_does_not_reopen_window"] = not windows()

        escaped.action = "escape"
        try:
            _capture_frames(camera, 2, .02, True, False, "escape", preview_window=escaped)
        except KeyboardInterrupt as exc:
            report["checks"]["native_escape_cancels_recording"] = "cancelled" in str(exc)
        else:
            report["checks"]["native_escape_cancels_recording"] = False
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        preview.close()
        escaped.close()
        if args.camera:
            camera.stop()
        report["checks"]["no_test_windows_left"] = not windows()
        report["passed"] = not report.get("error") and all(report["checks"].values())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
