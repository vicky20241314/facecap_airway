"""Readable capture controls outside the live camera image."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def _fonts():
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        (windows_fonts / "msyh.ttc", windows_fonts / "msyhbd.ttc"),
        (windows_fonts / "simhei.ttf", windows_fonts / "simhei.ttf"),
        (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
         Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")),
    ]
    for regular, bold in candidates:
        try:
            fonts = {size: ImageFont.truetype(str(regular), size) for size in (17, 19, 24)}
            fonts[30] = ImageFont.truetype(str(bold), 30)
            return fonts, True
        except OSError:
            continue
    # An English fallback is more useful than missing Chinese glyphs.
    return {size: ImageFont.load_default(size=size) for size in (17, 19, 24, 30)}, False


class CapturePanel:
    SIZE = (960, 780)
    BUTTON = (620, 668, 932, 756)
    CAMERA = (0, 104, 960, 644)
    TEXT_WIDTH = 578
    HEADLINE_Y = 655
    DETAIL_Y = 691
    CANCEL_Y = 750
    MARKER_XY = (400, 24)
    MARKER_WIDTH = 380

    def __init__(self):
        self.fonts, self.chinese = _fonts()
        self._key = None
        self._template = None

    def _wrap(self, draw, text, max_lines=2):
        font = self.fonts[19]
        tokens = re.findall(r"[A-Za-z0-9_.:;,%+\-/|]+|\s+|.", text)
        lines: list[str] = []
        current = ""
        for token in tokens:
            candidate = current + token
            if current and draw.textlength(candidate, font=font) > self.TEXT_WIDTH:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        if current.strip():
            lines.append(current.rstrip())
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and draw.textlength(last + "…", font=font) > self.TEXT_WIDTH:
                last = last[:-1]
            lines[-1] = last + "…"
        return lines

    def _fit_marker(self, draw, text):
        font = self.fonts[17]
        if draw.textlength(text, font=font) <= self.MARKER_WIDTH:
            return text
        while text and draw.textlength(text + "…", font=font) > self.MARKER_WIDTH:
            text = text[:-1]
        return text + "…"

    def _controls(self, title, guidance, status, original_instruction, pressed,
                  marker_line=""):
        retry = "RETRY:" in status
        ready = status.startswith("SPACE:") or retry
        countdown = re.search(r"Starting in (\d+)", status)
        recording = re.search(r"Recording (\d+)/(\d+)", status)
        if retry:
            reason = status.split("RETRY:", 1)[1].strip()[:120]
            # The pose is already named in the title, so the retry text keeps
            # only the machine reason and its hint.
            reason = re.sub(r"^(?:closed|open) front:\s*", "", reason)
            reason_en = reason.split(" — ", 1)[0]
            badge = "需要重拍" if self.chinese else "RETRY"
            headline = "标记无效：请调整后重拍" if self.chinese else "Marker invalid: adjust and retry"
            detail = (f"原因：{reason}。按空格或点击“重新采集”重拍。"
                      if self.chinese else
                      f"Reason: {reason_en}. Press SPACE or click Retry.")
            label = "重新采集" if self.chinese else "Retry"
            accent = (255, 178, 92)
        elif ready:
            badge = "准备采集" if self.chinese else "READY"
            headline = "准备就绪：点击按钮或按空格" if self.chinese else "Click Start capture or press SPACE"
            detail = "调整好姿态后点击，立即开始采集，无需倒计时。" if self.chinese else "Get into position; click to capture immediately."
            label = "开始采集" if self.chinese else "Start capture"
            accent = (255, 215, 92)
        elif countdown:
            seconds = countdown.group(1)
            badge = "倒计时" if self.chinese else "GET READY"
            headline = "即将开始：请保持当前姿态" if self.chinese else "Starting soon: hold your pose"
            detail = "无需再次点击或按键。" if self.chinese else "No further click or key is needed."
            label = f"{seconds} 秒后采集" if self.chinese else f"Starting in {seconds}"
            accent = (255, 215, 92)
        else:
            progress = f"{recording.group(1)}/{recording.group(2)}" if recording else ""
            badge = "采集中" if self.chinese else "RECORDING"
            headline = "正在采集：请保持不动" if self.chinese else "Recording: please keep still"
            detail = "本组结束后，按顶部提示调整下一姿态。" if self.chinese else "Follow the next pose prompt when this set is done."
            label = f"采集中 {progress}" if self.chinese else f"Recording {progress}"
            accent = (255, 132, 125)
        if not self.chinese:
            title, guidance = "RGB-D head capture", original_instruction

        panel = Image.new("RGB", self.SIZE, (12, 18, 28))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 960, 103), fill=(22, 32, 48))
        draw.rectangle((0, 644, 960, 780), fill=(22, 32, 48))
        draw.text((28, 16), title, font=self.fonts[30], fill=(245, 248, 255))
        draw.text((790, 24), badge, font=self.fonts[19], fill=accent)
        draw.text((28, 66), guidance, font=self.fonts[19], fill=(211, 224, 242))
        if marker_line:
            marker_ok = ("ARUCO:" not in marker_line and "ok" in marker_line)
            draw.text(self.MARKER_XY, self._fit_marker(draw, marker_line),
                      font=self.fonts[17],
                      fill=(130, 244, 191) if marker_ok else (255, 132, 125))
        draw.text((28, self.HEADLINE_Y), headline, font=self.fonts[24], fill=accent)
        y = self.DETAIL_Y
        for line in self._wrap(draw, detail):
            draw.text((28, y), line, font=self.fonts[19], fill=(220, 230, 244))
            y += 28
        cancel = "ESC 或关闭窗口可取消采集" if self.chinese else "ESC or close window to cancel"
        draw.text((28, self.CANCEL_Y), cancel, font=self.fonts[17], fill=(174, 194, 220))
        button_fill = ((24, 165, 98) if pressed else (48, 213, 136)) if ready else (48, 63, 83)
        draw.rounded_rectangle(self.BUTTON, radius=16, fill=button_fill,
                               outline=(130, 244, 191) if ready else (80, 99, 124), width=2)
        draw.text(((self.BUTTON[0] + self.BUTTON[2]) / 2,
                   (self.BUTTON[1] + self.BUTTON[3]) / 2), label,
                  font=self.fonts[30], fill=(6, 31, 20) if ready else (205, 219, 236),
                  anchor="mm")
        return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)

    def render(self, rgb, title, guidance, status, *, original_instruction,
               pressed=False, marker_line=""):
        key = (title, guidance, status, original_instruction, pressed, marker_line)
        if key != self._key:
            self._template = self._controls(*key)
            self._key = key
        shown = self._template.copy()
        height, width = rgb.shape[:2]
        scale = min(960 / width, 540 / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        camera = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size,
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        x = (960 - size[0]) // 2
        y = 104 + (540 - size[1]) // 2
        shown[y:y + size[1], x:x + size[0]] = camera
        return shown
