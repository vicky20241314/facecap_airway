"""Project constants; all mutable assets resolve outside the Git worktree."""
import os
import sys
from pathlib import Path


"""
目录部分
"""

# 运行时_MEIPASS目录
if getattr(sys, 'frozen', False):
    RUN_DIR = Path(sys._MEIPASS)
else:
    RUN_DIR = Path(__file__).parent.parent

# 项目目录
if getattr(sys, 'frozen', False):
    PROJECT_DIR = Path(sys.executable).parent
else:
    PROJECT_DIR = Path(__file__).parent.parent

_DEFAULT_RUNTIME = PROJECT_DIR / "runtime"
RUNTIME_DIR = Path(os.environ.get("HEADRGBD_RUNTIME_DIR", _DEFAULT_RUNTIME)).resolve()
ASSETS_DIR = Path(os.environ.get("HEADRGBD_ASSET_ROOT", RUNTIME_DIR / "assets")).resolve()
OUTPUT_DIR = RUNTIME_DIR / "outputs"

"""
相机部分
"""

DEPTH_WORK_MODE = "Extended Distance Mode"

DEPTH_VMIN_MM = 200
DEPTH_VMAX_MM = 700

"""
bbox识别部分
"""

bbox_model_path = Path(os.environ.get(
    "HEADRGBD_YUNET_MODEL",
    ASSETS_DIR / "models" / "yunet" / "face_detection_yunet_2023mar.onnx",
)).resolve()
score_threshold = 0.6  # 置信度阈值
nms_threshold = 0.3  # NMS 阈值
top_k = 5000  # NMS 前保留候选框数量
