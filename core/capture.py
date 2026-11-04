from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import time
from typing import Any

import numpy as np
import pyorbbecsdk
from pyorbbecsdk import (
    Pipeline, Config,
    OBSensorType, OBFormat, OBAlignMode,
    OBFrameAggregateOutputMode, OBDepthWorkMode,
)
from core.camera import Camera
from core.config import DEPTH_WORK_MODE


@dataclass(frozen=True)
class CapturedFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    host_monotonic_s: float
    hardware_timestamp_us: int | None
    global_timestamp_us: int | None
    system_timestamp_us: int | None
    color_index: int | None
    depth_index: int | None
    color_depth_sync_delta_ms: float | None
    depth_scale_mm: float | None

    @property
    def camera(self) -> Camera:
        return Camera(*self.intrinsics.tolist())

    def metadata(self) -> dict[str, Any]:
        depth = self.depth
        return {
            "host_monotonic_s": self.host_monotonic_s,
            "hardware_timestamp_us": self.hardware_timestamp_us,
            "global_timestamp_us": self.global_timestamp_us,
            "system_timestamp_us": self.system_timestamp_us,
            "color_index": self.color_index,
            "depth_index": self.depth_index,
            "color_depth_sync_delta_ms": self.color_depth_sync_delta_ms,
            "depth_scale_mm": self.depth_scale_mm,
            "width": int(depth.shape[1]),
            "height": int(depth.shape[0]),
            "valid_depth_fraction": float(np.count_nonzero(depth) / depth.size),
            "depth_min_mm": int(depth[depth > 0].min()) if np.any(depth > 0) else None,
            "depth_max_mm": int(depth.max()) if np.any(depth > 0) else None,
        }


def _safe_call(obj: Any, name: str, default=None):
    try:
        value = getattr(obj, name)
        return value() if callable(value) else value
    except Exception:
        return default


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Convert SDK values into JSON-native scalars without NaN/Infinity."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item(), depth + 1)
        if value.size <= 256:
            return [_json_safe(item, depth + 1) for item in value.tolist()]
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if depth >= 4:
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    return str(value)


def _device_info_dict(device) -> dict[str, Any]:
    info = device.get_device_info()
    return _json_safe({
        "name": _safe_call(info, "get_name"),
        "serial_number": _safe_call(info, "get_serial_number"),
        "uid": _safe_call(info, "get_uid"),
        "vid": _safe_call(info, "get_vid"),
        "pid": _safe_call(info, "get_pid"),
        "connection_type": str(_safe_call(info, "get_connection_type")),
        "device_type": str(_safe_call(info, "get_device_type")),
        "firmware_version": _safe_call(info, "get_firmware_version"),
        "hardware_version": _safe_call(info, "get_hardware_version"),
        "supported_min_sdk_version": _safe_call(info, "get_supported_min_sdk_version"),
    })


def _profile_dict(profile) -> dict[str, Any]:
    result = {
        "width": int(profile.get_width()),
        "height": int(profile.get_height()),
        "fps": int(profile.get_fps()),
        "format": str(profile.get_format()),
        "type": str(profile.get_type()),
    }
    intrinsic = _safe_call(profile, "get_intrinsic")
    if intrinsic is not None:
        result["intrinsics"] = {
            "fx": float(intrinsic.fx),
            "fy": float(intrinsic.fy),
            "cx": float(intrinsic.cx),
            "cy": float(intrinsic.cy),
            "width": int(getattr(intrinsic, "width", result["width"])),
            "height": int(getattr(intrinsic, "height", result["height"])),
        }
    distortion = _safe_call(profile, "get_distortion")
    if distortion is not None:
        result["distortion"] = {
            name: _json_safe(getattr(distortion, name))
            for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
            if hasattr(distortion, name)
        }
    return _json_safe(result)


def _timestamp_us(frame, method: str) -> int | None:
    value = _safe_call(frame, method)
    if value is None:
        return None
    return int(value)


# -----------------------------------------------------------------------
# 私有辅助函数
# -----------------------------------------------------------------------

def _select_rgb_1280x720(pipeline: Pipeline):
    """从设备支持的彩色 profile 列表中筛选 RGB 格式, 分辨率 1280x720."""
    plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    if plist is None or len(plist) == 0:
        raise RuntimeError("No COLOR stream profiles found.")

    for i in range(len(plist)):
        p = plist[i]
        if p.get_format() == OBFormat.RGB and p.get_width() == 1280 and p.get_height() == 720:
            return p

    raise RuntimeError("No RGB 1280x720 color profile found.")


def _get_hw_d2c_config(pipeline: Pipeline, color_profile):
    """
    构建硬件 D2C (Depth-to-Color) 对齐配置.
    - 深度流在设备端直接对齐到彩色流, 无需软件对齐, 延迟更低
    - 按目标分辨率匹配深度 profile, 避免硬编码索引在不同固件版本下失效
    """
    hw_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
    if hw_list is None or len(hw_list) == 0:
        raise RuntimeError("HW D2C not supported for this color profile/device.")

    # 从 hw_list 中选择最优深度 profile:
    # 优先条件: Y16 格式 (SDK 标准输出格式) + 分辨率尽量大 + 帧率尽量高
    # 实测 hw_list 只有 1280x800 和 640x400 两种分辨率, 没有与彩色流完全对齐的尺寸,
    # D2C 后深度图会被裁剪/缩放到与彩色流相同的尺寸
    best = None
    best_score = (-1, -1, -1)   # (w*h, fps, is_y16)
    for i in range(len(hw_list)):
        p = hw_list[i]
        is_y16 = 1 if p.get_format() == OBFormat.Y16 else 0
        score = (p.get_width() * p.get_height(), p.get_fps(), is_y16)
        if score > best_score:
            best_score = score
            best = p
    depth_profile = best if best is not None else hw_list[0]

    config = Config()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_align_mode(OBAlignMode.HW_MODE)

    try:
        # 要求彩色帧和深度帧必须同时到达才输出, 避免时间戳不同步
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception:
        pass

    return config, depth_profile


# -----------------------------------------------------------------------
# 深度工作模式管理
# 设备支持两种模式:
#   - "Close_Up Precision Mode"  : 近距离精度模式
#   - "Extended Distance Mode"   : 远距离模式 (本项目使用)
# -----------------------------------------------------------------------

def list_depth_work_modes() -> list[OBDepthWorkMode]:
    """查询设备支持的所有深度工作模式, 返回模式对象列表."""
    pipeline = Pipeline()
    device = pipeline.get_device()
    mode_list = device.get_depth_work_mode_list()
    return [mode_list.get_depth_work_mode_by_index(i) for i in range(mode_list.get_count())]


def get_depth_work_mode() -> OBDepthWorkMode:
    """返回设备当前的深度工作模式."""
    pipeline = Pipeline()
    device = pipeline.get_device()
    return device.get_depth_work_mode()


def set_depth_work_mode(mode: OBDepthWorkMode | str) -> OBDepthWorkMode:
    """
    切换设备深度工作模式.

    参数:
        mode: OBDepthWorkMode 对象, 或模式名称字符串
              (例如 "Extended Distance Mode")
    返回:
        切换后实际生效的 OBDepthWorkMode 对象
    异常:
        ValueError: 名称字符串未匹配到任何可用模式时抛出
    """
    pipeline = Pipeline()
    device = pipeline.get_device()

    # 如果传入的是字符串, 先在模式列表中按名称查找对应对象
    if isinstance(mode, str):
        mode_list = device.get_depth_work_mode_list()
        target = None
        for i in range(mode_list.get_count()):
            m = mode_list.get_depth_work_mode_by_index(i)
            if m.name == mode:
                target = m
                break
        if target is None:
            available = [mode_list.get_depth_work_mode_by_index(i).name
                         for i in range(mode_list.get_count())]
            raise ValueError(f"Depth work mode {mode!r} not found. Available: {available}")
        mode = target

    device.set_depth_work_mode(mode.name)
    return device.get_depth_work_mode()


# -----------------------------------------------------------------------
# 流式采集器
# -----------------------------------------------------------------------

class OrbbecCapture:
    """
    Orbbec 相机流式采集器.

    典型用法:
        cap = OrbbecCapture(timeout_ms=100)
        cap.start()
        while True:
            out = cap.read()
            if out is None:
                continue
            camera, rgb, depth = out
        cap.stop()

    read() 返回值:
        (Camera, rgb, depth) 或 None (帧未就绪时)
        - rgb   : (720, 1280, 3) uint8, RGB 格式
        - depth : (720, 1280)    uint16, 单位 mm
    """

    def __init__(self, timeout_ms: int = 100, depth_work_mode: str = DEPTH_WORK_MODE):
        self.timeout_ms = timeout_ms
        self.depth_work_mode = depth_work_mode
        self.pipeline = Pipeline()
        self.camera = None       # 首帧到达后由内参初始化
        self._started = False
        self.device = None
        self.color_profile = None
        self.depth_profile = None

    def start(self):
        # 步骤 1: 在启动流之前切换到远距离深度工作模式
        # 官方文档要求: Depth mode 必须在 start stream 之前设置
        # 官方示例传入模式名称字符串 (device.set_depth_work_mode(mode.name))
        device = self.pipeline.get_device()
        if device is None:
            raise RuntimeError("No Orbbec device is connected.")
        self.device = device
        mode_list = device.get_depth_work_mode_list()
        available_modes = {
            str(mode_list.get_depth_work_mode_by_index(i).name)
            for i in range(mode_list.get_count())
        }
        if self.depth_work_mode not in available_modes:
            raise RuntimeError(
                f"Depth work mode {self.depth_work_mode!r} is unavailable; "
                f"available modes: {sorted(available_modes)}"
            )
        device.set_depth_work_mode(self.depth_work_mode)

        # 步骤 2: 选择彩色流 profile (RGB 1280x720)
        color_profile = _select_rgb_1280x720(self.pipeline)
        self.color_profile = color_profile

        # 步骤 3: 构建 HW D2C 配置并启动 pipeline
        config, self.depth_profile = _get_hw_d2c_config(self.pipeline, color_profile)
        self.pipeline.start(config)
        self._started = True

    def stop(self):
        # 只停止 pipeline, 窗口生命周期由调用方管理
        if self._started:
            self.pipeline.stop()
            self._started = False

    def read(self):
        """
        读取一帧, 返回 (camera, rgb, depth) 或 None.
        - 帧超时/未就绪/格式不符时均返回 None, 调用方应循环重试
        """
        result = self.read_with_metadata()
        if result is None:
            return None
        return result.camera, result.rgb, result.depth

    def read_with_metadata(self) -> CapturedFrame | None:
        """Read one aligned RGB-D frame and retain acquisition evidence."""
        if not self._started:
            raise RuntimeError("OrbbecCapture.start() must be called first.")

        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        # HW D2C 对齐后, 彩色帧和深度帧应都是 1280x720
        if color_frame.get_width() != 1280 or color_frame.get_height() != 720:
            return None
        if depth_frame.get_width() != 1280 or depth_frame.get_height() != 720:
            return None

        # 格式校验
        if color_frame.get_format() != OBFormat.RGB:
            return None
        if depth_frame.get_format() != OBFormat.Y16:
            return None

        # 将原始字节数据解析为独立 numpy 数组，避免帧对象释放后悬挂。
        rgb   = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape((720, 1280, 3)).copy()
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((720, 1280)).copy()

        # 首帧成功时从彩色流内参初始化相机模型
        intr = color_frame.get_stream_profile().get_intrinsic()
        intrinsics = np.array([intr.fx, intr.fy, intr.cx, intr.cy], dtype=np.float64)
        if self.camera is None:
            self.camera = Camera(*intrinsics.tolist())
        elif not np.allclose(
            [self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy],
            intrinsics,
            rtol=0,
            atol=1e-5,
        ):
            raise RuntimeError("Camera intrinsics changed during an active stream.")

        color_global = _timestamp_us(color_frame, "get_global_timestamp_us")
        depth_global = _timestamp_us(depth_frame, "get_global_timestamp_us")
        delta_ms = None
        if color_global is not None and depth_global is not None:
            delta_ms = abs(color_global - depth_global) / 1000.0
        depth_scale = _safe_call(depth_frame, "get_depth_scale")
        return CapturedFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            host_monotonic_s=time.monotonic(),
            hardware_timestamp_us=_timestamp_us(color_frame, "get_timestamp_us"),
            global_timestamp_us=color_global,
            system_timestamp_us=_timestamp_us(color_frame, "get_system_timestamp_us"),
            color_index=_safe_call(color_frame, "get_index"),
            depth_index=_safe_call(depth_frame, "get_index"),
            color_depth_sync_delta_ms=delta_ms,
            depth_scale_mm=float(depth_scale) if depth_scale is not None else None,
        )


def _camera_param_dict(pipeline: Pipeline) -> dict[str, Any] | None:
    params = _safe_call(pipeline, "get_camera_param")
    if params is None:
        return None
    result = {}
    for name in dir(params):
        if name.startswith("_"):
            continue
        try:
            value = getattr(params, name)
        except Exception:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, (str, int, float, bool, np.generic)) or value is None:
            result[name] = _json_safe(value)
    return result or {"repr": _json_safe(repr(params))}


def probe_camera(timeout_ms: int = 1000, sample_frames: int = 3) -> dict[str, Any]:
    """Open the first Orbbec device, acquire real frames and report sync evidence."""
    context = pyorbbecsdk.Context()
    devices = context.query_devices()
    result: dict[str, Any] = {
        "schema": "head_rgbd.camera_probe.v1",
        "sdk_version": str(pyorbbecsdk.get_version()),
        "device_count": int(devices.get_count()),
        "opened": False,
        "alignment": {
            "mode": "HW_D2C",
            "rgb_depth_aligned": False,
            "independent_physical_verification": False,
        },
        "frames": [],
        "errors": [],
    }
    if result["device_count"] < 1:
        result["errors"].append("No Orbbec device reported by Context.query_devices().")
        return result

    capture = OrbbecCapture(timeout_ms=timeout_ms)
    try:
        capture.start()
        result["opened"] = True
        result["device"] = _device_info_dict(capture.device)
        result["depth_work_mode"] = str(capture.device.get_depth_work_mode().name)
        result["color_profile"] = _profile_dict(capture.color_profile)
        if capture.depth_profile is not None:
            result["depth_profile"] = _profile_dict(capture.depth_profile)
        result["camera_param"] = _camera_param_dict(capture.pipeline)
        for _ in range(sample_frames):
            frame = capture.read_with_metadata()
            if frame is None:
                result["errors"].append(f"Timed out waiting for aligned frame {len(result['frames']) + 1}.")
                continue
            result["frames"].append(frame.metadata())
        if result["frames"]:
            result["alignment"].update(
                rgb_depth_aligned=all(
                    item["width"] == 1280 and item["height"] == 720
                    for item in result["frames"]
                ),
                evidence="HW D2C configuration plus equal aligned frame dimensions and valid-color intrinsics",
            )
        return result
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        return result
    finally:
        try:
            capture.stop()
        except Exception as exc:
            result["errors"].append(f"stop: {type(exc).__name__}: {exc}")
