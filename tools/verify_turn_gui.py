"""Run an actual recorded RGB-D reconstruction through the desktop controls.

This opens the GUI, clicks its existing-data action, validates the successful
result state and saves window screenshots plus a machine-readable report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from head_rgbd.experiment.turn_gui import TurnWindow, STYLE


def main():
    from head_rgbd.experiment.turn_run import _configure_stdio
    _configure_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = TurnWindow(capture_folder=args.capture, device=args.device)
    window.output_input.setText(str(args.output / "runs"))
    window.show()
    app.processEvents()
    window.grab().save(str(args.output / "gui_existing.png"))
    result = {"status": "running", "capture": str(args.capture.resolve()), "errors": []}
    last_progress = -1

    def progress():
        nonlocal last_progress
        value = window.progress_bar.value()
        if value != last_progress:
            print(f"GUI_STAGE_PROGRESS={value}/8", flush=True)
            last_progress = value

    def finished(success, message):
        try:
            result.update(status="passed" if success else "failed", message=message,
                          run=str(window.run_dir), log=str(window.log_path))
            if success:
                if window.progress_bar.value() != 8:
                    raise AssertionError("GUI did not show all eight completed stages")
                if not all(button.isEnabled() for button in window.result_buttons):
                    raise AssertionError("Result buttons were not enabled")
                if not window.start_button.isEnabled() or window.stop_button.isEnabled():
                    raise AssertionError("GUI did not return to its ready state")
                verification = json.loads((window.run_dir / "verification.json").read_text(encoding="utf-8"))
                if verification["status"] != "passed" or verification["errors"]:
                    raise AssertionError("Artifact verification failed")
                result.update(elapsed_s=verification["elapsed_s"], keyframes=verification["keyframes"],
                              meshes=verification["meshes"])
                app.processEvents()
                if not window.grab().save(str(args.output / "gui_result.png")):
                    raise AssertionError("Could not save the result window screenshot")
            else:
                result["errors"].append(message)
                result["log_tail"] = window.log_view.toPlainText()[-5000:]
        except Exception as exc:
            result["status"] = "failed"
            result["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            (args.output / "gui_verification.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            window.close()
            app.quit()

    window.workflow_finished.connect(finished)
    timer = QTimer()
    timer.setInterval(500)
    timer.timeout.connect(progress)
    timer.start()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(window.stop_task)
    timeout.start(600000)
    QTimer.singleShot(0, window.start_button.click)
    app.exec()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
