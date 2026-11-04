"""Desktop interface for the continuous-turn experiment; model work stays off the GUI thread."""
from __future__ import annotations

import codecs
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from PyQt6.QtCore import QProcess, QProcessEnvironment, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont, QImage, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
    QFileDialog, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from head_rgbd.experiment.turn_app import capture_inputs, new_run_path, runtime_python
from head_rgbd.experiment.turn_config import TurnConfig
from head_rgbd.paths import PROJECT_ROOT, resolve_runtime_dir


STAGES = (("tracking", "轻量跟踪"), ("keyframe_selection", "角度选帧"),
          ("prepared_keyframes", "观测准备"), ("front_anchor", "正面初始化"),
          ("joint_shared_identity", "共享形状"), ("pose_correction", "姿态修正"),
          ("control_field", "位移场与细分"), ("delivery", "网格交付"))

STYLE = """
QWidget { font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 13px; color: #24344b; }
QMainWindow, QWidget#page { background: #f2f5f9; }
QWidget#options { background: white; }
QFrame#card { background: white; border: 1px solid #dfe6ef; border-radius: 14px; }
QLabel#title { font-size: 25px; font-weight: 700; color: #14243a; }
QLabel#subtitle, QLabel#muted { color: #6a7a90; }
QLabel#section { font-size: 16px; font-weight: 600; }
QLabel#tag { background: #def3ee; color: #087c6d; border-radius: 10px; padding: 7px 12px; }
QLineEdit, QComboBox, QSpinBox { border: 1px solid #d6dfe9; border-radius: 7px; padding: 7px; background: white; min-height: 19px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #159b89; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled { background: #f1f4f7; color: #8391a2; }
QPushButton { border: 1px solid #d3dce7; border-radius: 8px; background: white; padding: 8px 12px; }
QPushButton:hover { background: #eff6f6; border-color: #56b2a7; }
QPushButton:disabled { color: #9aa7b7; background: #f1f4f8; border-color: #e2e8ef; }
QPushButton#primary { background: #118b7b; color: white; border: 0; font-size: 15px; font-weight: 600; padding: 12px; }
QPushButton#primary:hover { background: #087566; }
QPushButton#primary:disabled { background: #a9cbc5; color: white; }
QPushButton#stop { color: #ba4b4b; }
QTabWidget::pane { border: 0; background: white; }
QTabBar::tab { padding: 10px 18px; color: #6a7a90; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #087c6d; border-bottom: 2px solid #118b7b; }
QProgressBar { background: #e7edf4; border: 0; border-radius: 4px; min-height: 8px; max-height: 8px; }
QProgressBar::chunk { background: #118b7b; border-radius: 4px; }
QPlainTextEdit { background: #152238; color: #dce9f4; border: 0; border-radius: 8px; font-family: Consolas; font-size: 12px; padding: 12px; }
QScrollArea { background: transparent; border: 0; }
"""


def _image(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


def _translated_status(status):
    if status.startswith("SPACE"):
        return "准备好后，点击下方采集按钮"
    if status.startswith("Starting in"):
        return status.replace("Starting in", "倒计时").replace("...", " 秒")
    replacements = {"HOLD STILL, FACE FRONT": "正对镜头，保持不动",
                    "SLOWLY TURN TO IMAGE-LEFT": "缓慢转向画面左侧",
                    "SLOWLY TURN TO IMAGE-RIGHT": "缓慢转向画面右侧",
                    "RETURN TO FRONT": "缓慢回到正面"}
    for source, target in replacements.items():
        status = status.replace(source, target)
    return status


class CaptureBridge:
    """Adapter for the existing capture protocol; all widgets stay on the main thread."""
    def __init__(self, worker):
        self.worker = worker
        self.last_preview = 0.

    def poll(self):
        if self.worker.cancel_event.is_set():
            raise KeyboardInterrupt("GUI capture cancelled")
        if self.worker.start_event.is_set():
            self.worker.start_event.clear()
            return 32
        return -1

    def show(self, frame, instruction, status):
        now = time.monotonic()
        if now - self.last_preview < .1:
            return
        self.last_preview = now
        import cv2
        depth = frame.depth.astype(np.float32)
        scaled = np.clip((1000. - depth) / 750. * 255., 0, 255).astype(np.uint8)
        colors = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
        colors[frame.depth == 0] = 0
        height, width = depth.shape
        center = depth[int(height * .3):int(height * .7), int(width * .35):int(width * .65)]
        valid = center[(center > 100) & (center < 1500)]
        distance_cm = float(np.median(valid) / 10.) if len(valid) else float("nan")
        self.worker.preview.emit(_image(frame.rgb), _image(colors), instruction, status, distance_cm)

    def close(self):
        pass


class CaptureWorker(QThread):
    preview = pyqtSignal(QImage, QImage, str, str, float)
    captured = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, output, runtime, frames, parent=None):
        super().__init__(parent)
        self.output, self.runtime, self.frames = output, runtime, frames
        self.start_event = threading.Event()
        self.cancel_event = threading.Event()

    def run(self):
        try:
            from head_rgbd.experiment.turn_capture import capture
            folder = capture(output=self.output, runtime=self.runtime, turn_frames=self.frames,
                             preview_window=CaptureBridge(self))
            self.captured.emit(str(folder))
        except KeyboardInterrupt:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class TurnWindow(QMainWindow):
    workflow_finished = pyqtSignal(bool, str)

    def __init__(self, *, runtime=None, config=None, device=None, capture_folder=None):
        super().__init__()
        self.runtime = resolve_runtime_dir(runtime)
        self.default_config = Path(config or PROJECT_ROOT / "config" / "turn_experiment.json").resolve()
        initial_config = TurnConfig.load(self.default_config)
        self.worker = None
        self.task = ""
        self.run_dir = None
        self.log_stream = None
        self.log_path = None
        self.cancel_requested = False
        self.closing = False
        self.pending_capture = None
        self.capture_error = ""
        self.rgb_image = self.depth_image = None
        self.last_capture_cue = ""
        self.started_clip = ""
        self.active_clip = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.setWindowTitle("HeadRGBD · 连续转头重建")
        self.setMinimumSize(980, 680)
        self._build_ui()
        self.config_input.setText(str(self.default_config))
        self.device_input.setCurrentIndex(0 if (device or initial_config.experiment.device) == "cuda" else 1)
        if capture_folder:
            self.data_input.setText(str(Path(capture_folder).resolve()))
            self.mode_input.setCurrentIndex(1)
        else:
            captures = self.runtime / "captures"
            candidates = [path for path in captures.glob("*")
                          if path.is_dir() and (path / "anchor_session.json").is_file()
                          and (path / "turn_session.json").is_file()]
            if candidates:
                self.data_input.setText(str(max(candidates, key=lambda path: path.stat().st_mtime)))
        self._mode_changed()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self._poll_progress)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1200, screen.width() - 40), min(830, screen.height() - 45))
        self._reset_progress()
        self._set_busy("")

    def _build_ui(self):
        page = QWidget()
        page.setObjectName("page")
        self.setCentralWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(16)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("连续转头 RGB-D 重建")
        title.setObjectName("title")
        titles.addWidget(title)
        subtitle = QLabel("正面锚点 + 连续转头，一次采集完成多视角重建")
        subtitle.setObjectName("subtitle")
        titles.addWidget(subtitle)
        heading.addLayout(titles, 1)
        tag = QLabel("HEADRGBD  /  TURN")
        tag.setObjectName("tag")
        heading.addWidget(tag, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(heading)
        body = QHBoxLayout()
        body.setSpacing(18)
        left = self._card()
        left.setFixedWidth(324)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 18, 18, 18)
        label = QLabel("运行设置")
        label.setObjectName("section")
        left_layout.addWidget(label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        options = QWidget()
        options.setObjectName("options")
        form = QVBoxLayout(options)
        form.setContentsMargins(0, 4, 0, 6)
        form.setSpacing(9)
        self.mode_input = QComboBox()
        self.mode_input.addItems(["现场采集并重建", "使用已有 RGB-D 数据"])
        self.mode_input.currentIndexChanged.connect(self._mode_changed)
        form.addWidget(QLabel("任务模式"))
        form.addWidget(self.mode_input)
        self.data_input = QLineEdit()
        self.data_input.setPlaceholderText("选择连续转头采集目录")
        self.data_row, self.data_browse = self._path_row(self.data_input, self._browse_data)
        form.addWidget(QLabel("已有采集目录"))
        form.addWidget(self.data_row)
        form.addWidget(QLabel("计算设备"))
        self.device_input = QComboBox()
        self.device_input.addItem("GPU · CUDA", "cuda")
        self.device_input.addItem("CPU", "cpu")
        form.addWidget(self.device_input)
        form.addWidget(QLabel("连续转头帧数"))
        self.frames_input = QSpinBox()
        self.frames_input.setRange(30, 200)
        self.frames_input.setValue(100)
        self.frames_input.setSuffix(" 帧")
        form.addWidget(self.frames_input)
        form.addWidget(QLabel("实验配置"))
        self.config_input = QLineEdit()
        config_row, self.config_browse = self._path_row(self.config_input, self._browse_config)
        form.addWidget(config_row)
        form.addWidget(QLabel("重建结果根目录"))
        self.output_input = QLineEdit(str(self.runtime / "turn_experiment"))
        output_row, self.output_browse = self._path_row(self.output_input, self._browse_output)
        form.addWidget(output_row)
        hint = QLabel("采集提示\n相机固定，距离约 0.3–0.7 米。\n保持闭口和自然表情，按提示缓慢\n完成正面 → 左侧 → 右侧 → 正面。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        hint.setStyleSheet("padding-top: 9px; line-height: 1.6;")
        form.addWidget(hint)
        form.addStretch(1)
        scroll.setWidget(options)
        left_layout.addWidget(scroll, 1)
        self.start_button = QPushButton("开始采集并重建")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_workflow)
        left_layout.addWidget(self.start_button)
        actions = QHBoxLayout()
        self.check_button = QPushButton("环境检查")
        self.check_button.clicked.connect(self.start_check)
        self.stop_button = QPushButton("停止任务")
        self.stop_button.setObjectName("stop")
        self.stop_button.clicked.connect(self.stop_task)
        actions.addWidget(self.check_button)
        actions.addWidget(self.stop_button)
        left_layout.addLayout(actions)
        body.addWidget(left)
        right = QVBoxLayout()
        right.setSpacing(14)
        progress_card = self._card()
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 14, 18, 14)
        progress_header = QHBoxLayout()
        label = QLabel("重建进度")
        label.setObjectName("section")
        progress_header.addWidget(label)
        progress_header.addStretch(1)
        self.progress_text = QLabel("0 / 8 个阶段完成")
        self.progress_text.setObjectName("muted")
        progress_header.addWidget(self.progress_text)
        progress_layout.addLayout(progress_header)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 8)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        progress_layout.addWidget(self.progress_bar)
        grid = QGridLayout()
        grid.setSpacing(6)
        self.stage_labels = {}
        for i, (key, text) in enumerate(STAGES):
            chip = QLabel(f"{i + 1:02d}  {text}")
            chip.setMinimumHeight(27)
            grid.addWidget(chip, i // 4, i % 4)
            self.stage_labels[key] = chip
        progress_layout.addLayout(grid)
        right.addWidget(progress_card)
        content = self._card()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 12)
        self.tabs = QTabWidget()
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(8, 12, 8, 4)
        row = QHBoxLayout()
        self.camera_status = QLabel("相机尚未启动")
        self.camera_status.setObjectName("muted")
        row.addWidget(self.camera_status, 1)
        self.depth_toggle = QCheckBox("显示深度图")
        self.depth_toggle.toggled.connect(self._refresh_preview)
        row.addWidget(self.depth_toggle)
        preview_layout.addLayout(row)
        self.preview_label = QLabel("准备开始采集\n\n启动后可在这里查看 RGB-D 实时画面")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(360, 180)
        self.preview_label.setStyleSheet("background: #eaf0f5; color: #72839a; border-radius: 10px; font-size: 15px;")
        preview_layout.addWidget(self.preview_label, 1)
        self.guide_label = QLabel("先采集正面锚点，再缓慢左右转头。采集结束后自动开始重建。")
        self.guide_label.setWordWrap(True)
        self.guide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.guide_label.setMinimumHeight(40)
        preview_layout.addWidget(self.guide_label)
        self.capture_button = QPushButton("等待相机就绪")
        self.capture_button.setObjectName("primary")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self._start_clip)
        preview_layout.addWidget(self.capture_button)
        self.tabs.addTab(preview, "相机预览")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.tabs.addTab(self.log_view, "运行日志")
        self.results_page = self._build_results()
        self.tabs.addTab(self.results_page, "重建结果")
        content_layout.addWidget(self.tabs)
        right.addWidget(content, 1)
        body.addLayout(right, 1)
        layout.addLayout(body, 1)
        self.status_label = QLabel("就绪 · 选择任务模式后开始")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(24)
        layout.addWidget(self.status_label)

    @staticmethod
    def _card():
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    @staticmethod
    def _path_row(edit, callback):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        button = QPushButton("选择")
        button.setFixedWidth(52)
        button.clicked.connect(callback)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return widget, button

    def _build_results(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        page.setObjectName("options")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 26, 20, 18)
        self.result_title = QLabel("等待重建完成")
        self.result_title.setObjectName("title")
        layout.addWidget(self.result_title)
        self.result_detail = QLabel("完成后会显示关键帧、网格信息和结果文件。")
        self.result_detail.setWordWrap(True)
        self.result_detail.setStyleSheet("font-size: 15px; color: #63768d; padding: 12px 0;")
        layout.addWidget(self.result_detail)
        layout.addWidget(QLabel("本次结果目录"))
        self.result_path = QLineEdit()
        self.result_path.setReadOnly(True)
        layout.addWidget(self.result_path)
        layout.addSpacing(16)
        self.result_buttons = []
        for text, suffix in (("打开结果目录", "delivery"), ("打开最终网格", "delivery/final_mesh.obj"),
                             ("打开验收报告", "verification.json")):
            button = QPushButton(text)
            button.setEnabled(False)
            button.clicked.connect(lambda checked=False, relative=suffix: self._open_result(relative))
            layout.addWidget(button)
            self.result_buttons.append(button)
        note = QLabel("验收检查流程执行、文件完整性和几何安全约束。独立几何精度仍需参考数据验证。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addStretch(1)
        layout.addWidget(note)
        scroll.setWidget(page)
        return scroll

    def _mode_changed(self):
        if not hasattr(self, "frames_input"):
            return
        live = self.mode_input.currentIndex() == 0
        self.data_row.setEnabled(not live and not self.task)
        self.frames_input.setEnabled(live and not self.task)
        self.start_button.setText("开始采集并重建" if live else "开始重建")

    def _browse_data(self):
        path = QFileDialog.getExistingDirectory(self, "选择连续转头采集目录", str(self.runtime / "captures"))
        if path:
            self.data_input.setText(path)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择实验配置", str(PROJECT_ROOT / "config"), "JSON (*.json)")
        if path:
            self.config_input.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择结果根目录", self.output_input.text())
        if path:
            self.output_input.setText(path)

    def _set_busy(self, task):
        self.task = task
        busy = bool(task)
        for widget in (self.mode_input, self.device_input, self.config_input, self.config_browse,
                       self.output_input, self.output_browse, self.start_button, self.check_button):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.capture_button.setVisible(task == "capture" or not busy)
        self.capture_button.setEnabled(False)
        self._mode_changed()

    def _start_log(self):
        if self.log_stream is not None:
            self.log_stream.close()
        logs = self.runtime / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log_path = new_run_path(logs, "turn_gui").with_suffix(".log")
        self.log_stream = self.log_path.open("x", encoding="utf-8")
        self.log_view.clear()
        self._append_log("日志：" + str(self.log_path) + "\n")

    def _append_log(self, text):
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.ensureCursorVisible()
        if self.log_stream is not None:
            self.log_stream.write(text)
            self.log_stream.flush()

    def _reset_progress(self):
        self.progress_bar.setRange(0, 8)
        self.progress_bar.setValue(0)
        self.progress_text.setText("0 / 8 个阶段完成")
        for label in self.stage_labels.values():
            label.setStyleSheet("color: #7a889c; background: #f1f4f8; border-radius: 5px; padding: 3px 6px;")

    def start_workflow(self):
        if self.task:
            return
        try:
            TurnConfig.load(Path(self.config_input.text()))
            if not self.output_input.text().strip():
                raise ValueError("请选择重建结果根目录。")
            live = self.mode_input.currentIndex() == 0
            if not live:
                capture_inputs(self.data_input.text())
            self._start_log()
            self.cancel_requested = False
            self.run_dir = new_run_path(Path(self.output_input.text()))
            self.result_path.clear()
            self.result_title.setText("等待重建完成")
            self.result_detail.setText("本次任务完成并通过验收后显示结果。")
            for button in self.result_buttons:
                button.setEnabled(False)
            self._reset_progress()
            if live:
                self._begin_capture()
            else:
                self._begin_reconstruction(Path(self.data_input.text()))
        except Exception as exc:
            self._show_error(str(exc))

    def _begin_capture(self):
        output = new_run_path(self.runtime / "captures", "gui_turn")
        self.pending_capture = None
        self.capture_error = ""
        self.last_capture_cue = ""
        self.started_clip = ""
        self.active_clip = ""
        self._set_busy("capture")
        self.tabs.setCurrentIndex(0)
        self.camera_status.setText("正在连接 RGB-D 相机…")
        self.status_label.setText("正在打开相机，请保持正对镜头、自然闭口。")
        self._append_log("打开相机；采集目录：" + str(output) + "\n")
        worker = CaptureWorker(output, self.runtime, self.frames_input.value(), self)
        self.worker = worker
        worker.preview.connect(self._on_preview)
        worker.captured.connect(self._capture_saved)
        worker.failed.connect(self._capture_failed)
        worker.finished.connect(self._capture_finished)
        worker.start()

    def _on_preview(self, rgb, depth, instruction, status, distance):
        if self.task != "capture" or self.cancel_requested:
            return
        self.rgb_image, self.depth_image = rgb, depth
        anchor = instruction.startswith("ANCHOR")
        phase = "正面锚点" if anchor else "连续转头"
        self.active_clip = phase
        distance_text = f"中心深度 {distance:.1f} cm" if np.isfinite(distance) else "中心深度暂不可用"
        self.camera_status.setText(f"{phase} · RGB-D 已对齐 · {distance_text}")
        waiting = status.startswith("SPACE") and self.started_clip != phase
        self.capture_button.setText("采集正面锚点" if anchor else "开始转头采集")
        self.capture_button.setEnabled(waiting)
        guide = _translated_status(status)
        self.guide_label.setText(guide)
        self.status_label.setText(f"{phase} · {guide}")
        cue = phase + " " + guide.split("|")[0].strip()
        if cue != self.last_capture_cue:
            self._append_log(cue + "\n")
            self.last_capture_cue = cue
        self._refresh_preview()

    def _refresh_preview(self):
        image = self.depth_image if self.depth_toggle.isChecked() else self.rgb_image
        if image is not None and not image.isNull():
            self.preview_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview_label"):
            self._refresh_preview()

    def _start_clip(self):
        if self.worker is not None and not self.cancel_requested:
            self.capture_button.setEnabled(False)
            self.started_clip = self.active_clip
            self.worker.start_event.set()

    def _capture_saved(self, path):
        self.pending_capture = Path(path)
        self.data_input.setText(path)
        self._append_log("采集保存完成：" + path + "\n")

    def _capture_failed(self, error):
        self.capture_error = error
        self._append_log(error + "\n")

    def _capture_finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        if self.pending_capture is not None and not self.cancel_requested:
            self._begin_reconstruction(self.pending_capture)
        else:
            self._set_busy("")
            if self.capture_error:
                self._show_error(self.capture_error.strip().splitlines()[-1])
            else:
                self.status_label.setText("采集已停止，已保存的数据会保留。")
            self.workflow_finished.emit(False, self.capture_error or "cancelled")
            if self.closing:
                self.close()

    def _begin_reconstruction(self, folder):
        arguments = ["reconstruct", "--capture", str(folder), "--runtime", str(self.runtime),
                     "--config", self.config_input.text(), "--device", self.device_input.currentData(),
                     "--output", str(self.run_dir)]
        self.status_label.setText("采集数据已就绪，正在启动重建…")
        self.tabs.setCurrentIndex(1)
        self._launch_process(arguments, "reconstruct")

    def start_check(self):
        if self.task:
            return
        try:
            self._start_log()
            self.cancel_requested = False
            self.tabs.setCurrentIndex(1)
            self.status_label.setText("正在检查运行环境、模型资产和计算设备…")
            self._launch_process(["check", "--runtime", str(self.runtime),
                                  "--device", self.device_input.currentData()], "check")
        except Exception as exc:
            self._show_error(str(exc))

    def _launch_process(self, arguments, task):
        self._set_busy(task)
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        env = QProcessEnvironment.systemEnvironment()
        for key, value in {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                           "PYTHONPATH": str(PROJECT_ROOT)}.items():
            env.insert(key, value)
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProgram(str(runtime_python()))
        self.process.setArguments(["-m", "head_rgbd.experiment.turn_app", *arguments])
        self._append_log("启动：" + subprocess.list2cmdline([str(runtime_python()), "-m", "head_rgbd.experiment.turn_app", *arguments]) + "\n")
        self.timer.start()
        self.process.start()

    def _read_output(self):
        data = bytes(self.process.readAllStandardOutput())
        self._append_log(self.decoder.decode(data))

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.timer.stop()
            message = self.process.errorString()
            self._set_busy("")
            self._show_error(message)
            self.workflow_finished.emit(False, message)
            if self.closing:
                self.close()

    def _poll_progress(self):
        if self.task != "reconstruct" or self.run_dir is None:
            return
        try:
            report = json.loads((self.run_dir / "audit" / "run_report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        statuses = {row["name"]: row["status"] for row in report["stages"]}
        colors = {"passed": ("#087c6d", "#e0f2ec"), "running": ("#1a63af", "#e2edfc"),
                  "failed": ("#b44343", "#fbe8e8"), "pending": ("#7a889c", "#f1f4f8")}
        for key, label in self.stage_labels.items():
            color, background = colors.get(statuses.get(key, "pending"), colors["pending"])
            label.setStyleSheet(f"color: {color}; background: {background}; border-radius: 5px; padding: 3px 6px;")
        completed = sum(value == "passed" for value in statuses.values())
        self.progress_bar.setValue(completed)
        self.progress_text.setText(f"{completed} / 8 个阶段完成")
        active = [text for key, text in STAGES if statuses.get(key) == "running"]
        if active and not self.cancel_requested:
            self.status_label.setText(f"正在执行：{active[0]} · 完整日志已保存")

    def _process_finished(self, code, exit_status):
        self._read_output()
        tail = self.decoder.decode(b"", final=True)
        if tail:
            self._append_log(tail)
        self._poll_progress()
        self.timer.stop()
        task = self.task
        self._set_busy("")
        success = code == 0 and exit_status == QProcess.ExitStatus.NormalExit and not self.cancel_requested
        message = ""
        if self.cancel_requested:
            message = "任务已停止，已保存的数据和阶段日志会保留。"
            self.status_label.setText(message)
        elif success and task == "check":
            message = "环境检查通过，可以开始采集或重建。"
            self.status_label.setText(message)
        elif success:
            try:
                verification = json.loads((self.run_dir / "verification.json").read_text(encoding="utf-8"))
                if verification["status"] != "passed":
                    raise ValueError("产物验收未通过")
                self._show_results(verification)
                message = "重建及验收完成，8 个阶段全部通过。"
                self.status_label.setText(message)
            except Exception as exc:
                success = False
                message = str(exc)
                self._show_error(message)
        else:
            message = f"任务失败，退出码 {code}。请查看运行日志。"
            self._show_error(message)
        self._append_log(f"任务结束：exit_code={code}，{message}\n")
        self.workflow_finished.emit(success, message)
        if self.closing:
            self.close()

    def _show_results(self, verification):
        mesh = verification["meshes"]["final_mesh.obj"]
        self.result_title.setText("重建与验收通过")
        self.result_detail.setText(
            f"{verification['keyframes']} 个关键帧  ·  左右跨度 {verification['coverage']['yaw_span_deg']:.1f}°\n"
            f"{mesh['vertices']:,} 个顶点  /  {mesh['faces']:,} 个面\n"
            f"完整重建用时 {verification['elapsed_s']:.2f} 秒  ·  8 个阶段全部通过")
        self.result_detail.setMinimumHeight(self.result_detail.sizeHint().height())
        self.result_path.setText(str(self.run_dir / "delivery"))
        for button in self.result_buttons:
            button.setEnabled(True)
        self.tabs.setCurrentIndex(2)

    def _show_error(self, message):
        self.status_label.setText("错误 · " + message)
        self._append_log("ERROR: " + message + "\n")
        self.tabs.setCurrentIndex(1)

    def _open_result(self, relative):
        if self.run_dir is not None:
            path = self.run_dir / relative
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
                self._show_error("无法打开文件，请从结果目录手动选择：" + str(path))

    def stop_task(self):
        if not self.task or self.cancel_requested:
            return
        self.cancel_requested = True
        self.stop_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.status_label.setText("正在停止任务并释放资源…")
        if self.worker is not None:
            self.worker.cancel_event.set()
        elif self.process.state() != QProcess.ProcessState.NotRunning:
            pid = int(self.process.processId())
            if sys.platform == "win32" and pid:
                subprocess.Popen(["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.process.terminate()

    def closeEvent(self, event):
        if self.task:
            self.closing = True
            self.stop_task()
            event.ignore()
            return
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None
        event.accept()


def launch(*, runtime=None, config=None, device=None, capture_folder=None):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("HeadRGBD Turn")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = TurnWindow(runtime=runtime, config=config, device=device, capture_folder=capture_folder)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch())
