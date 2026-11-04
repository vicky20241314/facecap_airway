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

    def __init__(self):
        self.fonts, self.chinese = _fonts()
        self._key = None
        self._template = None

    def _controls(self, title, guidance, status, original_instruction, pressed):
        ready = status.startswith("SPACE:")
        countdown = re.match(r"Starting in (\d+)", status)
        recording = re.match(r"Recording (\d+)/(\d+)", status)
        if ready:
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
        draw.text((28, 661), headline, font=self.fonts[24], fill=accent)
        draw.text((28, 701), detail, font=self.fonts[19], fill=(220, 230, 244))
        cancel = "ESC 或关闭窗口可取消采集" if self.chinese else "ESC or close window to cancel"
        draw.text((28, 742), cancel, font=self.fonts[17], fill=(174, 194, 220))
        button_fill = ((24, 165, 98) if pressed else (48, 213, 136)) if ready else (48, 63, 83)
        draw.rounded_rectangle(self.BUTTON, radius=16, fill=button_fill,
                               outline=(130, 244, 191) if ready else (80, 99, 124), width=2)
        draw.text(((self.BUTTON[0] + self.BUTTON[2]) / 2,
                   (self.BUTTON[1] + self.BUTTON[3]) / 2), label,
                  font=self.fonts[30], fill=(6, 31, 20) if ready else (205, 219, 236),
                  anchor="mm")
        return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)

    def render(self, rgb, title, guidance, status, *, original_instruction, pressed=False):
        key = (title, guidance, status, original_instruction, pressed)
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
