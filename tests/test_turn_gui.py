"""Exercise GUI controls, camera messages and real process-tree cancellation."""
import os
import re
import sys
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QProcess
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from head_rgbd.experiment import turn_gui


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app, tmp_path):
    result = turn_gui.TurnWindow(runtime=tmp_path)
    yield result
    if result.process.state() != QProcess.ProcessState.NotRunning:
        result.stop_task()
        result.process.waitForFinished(5000)
    result.worker = None
    result.task = ""
    result.close()
    qt_app.processEvents()


def test_invalid_recording_keeps_controls_ready(window):
    window.mode_input.setCurrentIndex(1)
    window.data_input.setText("missing recording")
    window.start_button.click()
    assert window.task == ""
    assert window.start_button.isEnabled()
    assert "缺少" in window.status_label.text()
    assert window.process.state() == QProcess.ProcessState.NotRunning


def test_queued_preview_cannot_start_a_clip_twice(window):
    window._set_busy("capture")
    window.worker = SimpleNamespace(start_event=threading.Event(), cancel_event=threading.Event())
    image = turn_gui._image(np.zeros((4, 4, 3), np.uint8))
    window._on_preview(image, image, "ANCHOR: test", "SPACE: start", 50.)
    assert window.capture_button.isEnabled()
    window.capture_button.click()
    assert window.worker.start_event.is_set()
    window._on_preview(image, image, "ANCHOR: test", "SPACE: start", 50.)
    assert not window.capture_button.isEnabled()
    window.worker.start_event.clear()
    window._on_preview(image, image, "TURN: test", "SPACE: start", 50.)
    assert window.capture_button.isEnabled()
    window.capture_button.click()
    assert window.worker.start_event.is_set()


def test_capture_bridge_owns_image_memory_and_delivers_start_cancel(qt_app):
    worker = turn_gui.CaptureWorker(None, None, 100)
    spy = QSignalSpy(worker.preview)
    bridge = turn_gui.CaptureBridge(worker)
    frame = SimpleNamespace(rgb=np.full((12, 16, 3), 100, np.uint8),
                            depth=np.full((12, 16), 500, np.uint16))
    bridge.show(frame, "ANCHOR", "SPACE",)
    assert len(spy) == 1
    frame.rgb[:] = 0
    assert spy[0][0].pixelColor(0, 0).red() == 100
    assert spy[0][-1] == 50.
    worker.start_event.set()
    assert bridge.poll() == 32
    assert bridge.poll() == -1
    worker.cancel_event.set()
    with pytest.raises(KeyboardInterrupt):
        bridge.poll()
    worker.deleteLater()


def test_failed_process_start_restores_gui(window, monkeypatch, tmp_path):
    monkeypatch.setattr(turn_gui, "runtime_python", lambda: tmp_path / "missing_python.exe")
    spy = QSignalSpy(window.workflow_finished)
    window.start_check()
    assert len(spy) or spy.wait(5000)
    assert spy[0][0] is False
    assert window.task == ""
    assert window.start_button.isEnabled()


def test_results_remain_readable_in_a_small_window(window, qt_app, tmp_path):
    window.run_dir = tmp_path / "result"
    window.resize(980, 700)
    window._show_results({"meshes": {"final_mesh.obj": {"vertices": 20018, "faces": 39904}},
                          "keyframes": 7, "coverage": {"yaw_span_deg": 57.2}, "elapsed_s": 132.76})
    window.show()
    qt_app.processEvents()
    assert "132.76" in window.result_detail.text()
    assert window.result_detail.height() >= window.result_detail.minimumSizeHint().height()
    assert window.height() <= 720


@pytest.mark.skipif(sys.platform != "win32", reason="Windows task tree cancellation")
def test_stop_ends_both_the_pipeline_and_its_child(window):
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    window._start_log()
    window._set_busy("reconstruct")
    script = ("import subprocess, sys, time; "
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
              "creationflags=subprocess.CREATE_NO_WINDOW); "
              "print('CHILD=' + str(child.pid), flush=True); time.sleep(60)")
    window.process.setProgram(sys.executable)
    window.process.setArguments(["-c", script])
    window.process.start()
    assert window.process.waitForStarted(3000)
    for _ in range(100):
        QTest.qWait(30)
        match = re.search(r"CHILD=(\d+)", window.log_view.toPlainText())
        if match:
            break
    assert match is not None
    handle = kernel.OpenProcess(0x100000, False, int(match.group(1)))
    assert handle
    spy = QSignalSpy(window.workflow_finished)
    try:
        window.stop_button.click()
        assert spy.wait(5000)
        assert spy[0][0] is False
        assert kernel.WaitForSingleObject(handle, 5000) == 0
        assert window.task == ""
        assert "已停止" in window.status_label.text()
    finally:
        kernel.CloseHandle(handle)
