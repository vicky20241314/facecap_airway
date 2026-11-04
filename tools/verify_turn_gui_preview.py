"""Check a connected camera in the actual GUI, then stop before recording a clip."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def main():
    from head_rgbd.experiment.turn_run import _configure_stdio
    _configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "preview_received": False, "errors": []}
    import pyorbbecsdk
    devices = pyorbbecsdk.Context().query_devices()
    if devices.get_count() == 0:
        result.update(status="skipped", reason="No RGB-D camera connected")
    else:
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import QApplication
        from head_rgbd.experiment.turn_gui import TurnWindow, STYLE

        app = QApplication([])
        app.setStyle("Fusion")
        app.setStyleSheet(STYLE)
        app.setFont(QFont("Microsoft YaHei UI", 10))
        window = TurnWindow()
        window.show()

        def preview(rgb, depth, instruction, status, distance):
            if result["preview_received"]:
                return
            result["preview_received"] = not rgb.isNull() and not depth.isNull()
            result["rgb_size"] = [rgb.width(), rgb.height()]
            result["clip_button_enabled"] = window.capture_button.isEnabled()
            window.depth_toggle.setChecked(True)
            app.processEvents()
            window.grab().save(str(args.output / "gui_depth_preview.png"))
            QTimer.singleShot(0, window.stop_button.click)

        def finished(success, message):
            try:
                if not result["preview_received"] or not result.get("clip_button_enabled"):
                    raise AssertionError("Camera did not reach its ready state: " + message)
                if window.worker is not None or window.task or not window.start_button.isEnabled():
                    raise AssertionError("Capture resources or controls were not released")
                partials = list((window.runtime / "captures").glob("gui_turn_*/session.partial.json"))
                result.update(status="passed", message=message, camera_released=True,
                              no_clip_recorded=True, partial_capture_count=len(partials))
            except Exception as exc:
                result["status"] = "failed"
                result["errors"].append(f"{type(exc).__name__}: {exc}")
            window.close()
            app.quit()

        def start():
            window.start_button.click()
            if window.worker is not None:
                window.worker.preview.connect(preview)

        window.workflow_finished.connect(finished)
        QTimer.singleShot(0, start)
        QTimer.singleShot(20000, window.stop_task)
        app.exec()
    (args.output / "gui_preview_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] in ("passed", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
