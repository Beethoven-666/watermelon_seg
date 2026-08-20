#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib
import json
import logging
import math
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np


logger = logging.getLogger("tray_capture")


REPO_ROOT = Path(__file__).resolve().parents[1]
EXIT_SUCCESS = 0
EXIT_ARGUMENTS = 2
EXIT_NO_DEVICE = 3
EXIT_DEVICE_AMBIGUOUS = 4
EXIT_UNSUPPORTED_PROFILE = 5
EXIT_PIPELINE_FAIL = 6
EXIT_ENCODER_FAIL = 7
EXIT_CAPTURE_STREAM_FAIL = 8
EXIT_VERIFY_FAIL = 9
EXIT_UNKNOWN = 10

MANIFEST_FIELDS = [
    "capture_id",
    "batch_id",
    "video_path",
    "metadata_path",
    "frames_csv_path",
    "capture_date",
    "start_time",
    "end_time",
    "duration_seconds",
    "width",
    "height",
    "fps",
    "requested_fps",
    "measured_fps",
    "codec",
    "frame_count",
    "device_model",
    "device_serial",
    "lighting",
    "scene",
    "tray_count",
    "camera_moved",
    "conveyor_running",
    "planned_split",
    "fixed_test",
    "status",
    "video_sha256",
    "notes",
]


class CaptureError(RuntimeError):
    exit_code: int = EXIT_UNKNOWN

    def __init__(self, message: str, exit_code: int = EXIT_UNKNOWN):
        super().__init__(message)
        self.exit_code = exit_code


class FrameSourceError(CaptureError):
    pass


class VideoWriterError(CaptureError):
    pass


@dataclass(frozen=True)
class CaptureConfig:
    output_root: Path
    batch_id: Optional[str]
    serial_number: Optional[str]
    list_devices: bool
    list_profiles: bool
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    color_format: Optional[str]
    duration_seconds: float
    segment_seconds: float
    warmup_seconds: float
    no_preview: bool
    fixed_test: bool
    planned_split: str
    lighting: str
    scene_tags: str
    tray_count: str
    camera_moved: str
    conveyor_running: str
    notes: str
    codec: str
    container: str
    frame_timeout_ms: int
    log_level: str


@dataclass(frozen=True)
class DeviceInfo:
    model: str
    serial_number: str
    firmware_version: Optional[str] = None


@dataclass(frozen=True)
class ResolvedStreamProfile:
    requested_width: Optional[int]
    requested_height: Optional[int]
    requested_fps: Optional[float]
    requested_format: Optional[str]
    actual_width: int
    actual_height: int
    actual_fps: float
    actual_format: str


@dataclass(frozen=True)
class CapturePaths:
    capture_id: str
    segment_index: int
    video_partial_path: Path
    video_path: Path
    metadata_path: Path
    frames_csv_path: Path
    log_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SourceFrame:
    data: Any
    width: int
    height: int
    format: str
    device_timestamp_ms: Optional[int] = None
    frame_index: Optional[int] = None


@dataclass
class CaptureStatistics:
    frames_received: int = 0
    frames_written: int = 0
    conversion_failures: int = 0
    write_failures: int = 0
    timeouts: int = 0
    estimated_dropped_frames: int = 0

    def measured_fps(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0:
            return 0.0
        return self.frames_written / elapsed_seconds


@dataclass
class CaptureResult:
    capture_id: str
    batch_id: str
    status: str
    video_path: Optional[Path]
    metadata_path: Path
    frames_csv_path: Path
    log_path: Path
    capture_start_time: datetime
    capture_end_time: datetime
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    requested_fps: Optional[float]
    statistics: CaptureStatistics
    scene_tags: str
    fixed_test: bool
    planned_split: str
    error: Optional[str] = None


class FrameSourceBase:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def read_frame(self, timeout_ms: int) -> Optional[SourceFrame]:
        raise NotImplementedError

    def get_active_profile(self) -> ResolvedStreamProfile:
        raise NotImplementedError

    def get_device_info(self) -> DeviceInfo:
        raise NotImplementedError


class OrbbecFrameSource(FrameSourceBase):
    def __init__(
        self,
        serial_number: Optional[str],
        requested_profile: Optional[ResolvedStreamProfile],
        logger_: logging.Logger,
    ) -> None:
        self.serial_number = serial_number
        self.requested_profile = requested_profile
        self.logger = logger_
        self.pipeline = None
        self.context = None
        self.device = None
        self._device_info: Optional[DeviceInfo] = None
        self._active_profile: Optional[ResolvedStreamProfile] = None

    @staticmethod
    def _load_sdk():
        try:
            return importlib.import_module("pyorbbecsdk")
        except ModuleNotFoundError as exc:
            raise FrameSourceError(
                "未检测到 pyorbbecsdk，请先在当前环境安装 SDK 或确认当前环境有 Gemini 435Le 驱动。"
            ) from exc

    def start(self) -> None:
        sdk = self._load_sdk()
        Pipeline = getattr(sdk, "Pipeline", None)
        Config = getattr(sdk, "Config", None)
        SensorType = getattr(sdk, "OBSensorType", None)
        if Pipeline is None or Config is None or SensorType is None:
            raise FrameSourceError(
                "当前 pyorbbecsdk 版本未暴露 Pipeline/Config/OBSensorType，无法安全启动采集。"
            )

        try:
            context = sdk.Context()
            selected, device = _select_sdk_device(context.query_devices(), self.serial_number)
            pipeline = Pipeline(device)
            config = Config()
        except Exception as exc:
            raise FrameSourceError(f"实例化 Pipeline/Config 失败: {exc}") from exc

        try:
            color_profiles = pipeline.get_stream_profile_list(SensorType.COLOR_SENSOR)
            if self.requested_profile and self.requested_profile.requested_width is not None:
                if self.requested_profile.requested_format:
                    fmt = self._resolve_format_constant(sdk, self.requested_profile.requested_format)
                    color_profile = color_profiles.get_video_stream_profile(
                        self.requested_profile.requested_width,
                        self.requested_profile.requested_height,
                        fmt,
                        int(self.requested_profile.requested_fps),
                    )
                else:
                    matches = [
                        item for item in _iter_iterable_like(color_profiles)
                        if _to_int_or_none(_safe_call(item, "get_width")) == self.requested_profile.requested_width
                        and _to_int_or_none(_safe_call(item, "get_height")) == self.requested_profile.requested_height
                        and _safe_call(item, "get_fps") == int(self.requested_profile.requested_fps)
                    ]
                    if len(matches) != 1:
                        raise FrameSourceError(
                            "指定宽度/高度/FPS 时匹配到多个或零个颜色格式；请增加 --format 精确选择。",
                            exit_code=EXIT_UNSUPPORTED_PROFILE,
                        )
                    color_profile = matches[0]
            else:
                color_profile = color_profiles.get_default_video_stream_profile()
            if color_profile is None:
                raise FrameSourceError("设备没有可用的默认彩色流配置")
            config.enable_stream(color_profile)
        except FrameSourceError:
            raise
        except Exception as exc:
            available = _profiles_from_sdk_list(color_profiles) if "color_profiles" in locals() else []
            requested = self.requested_profile
            if requested is None:
                raise FrameSourceError("读取设备默认彩色流配置失败") from exc
            raise FrameSourceError(
                "当前设备不支持指定彩色配置: "
                f"{requested.requested_width}x{requested.requested_height} "
                f"fps={requested.requested_fps} format={requested.requested_format}; "
                f"可用配置: {available}",
                exit_code=EXIT_UNSUPPORTED_PROFILE,
            ) from exc

        try:
            pipeline.start(config)
        except Exception as exc:
            raise FrameSourceError(
                f"Pipeline 启动失败: {exc}", exit_code=EXIT_PIPELINE_FAIL
            ) from exc
        self.pipeline = pipeline
        self.context = context
        self.config = config
        self.device = device
        self._device_info = selected
        self._active_profile = _resolved_profile_from_sdk_profile(
            color_profile,
            requested=self.requested_profile,
        )
        self.logger.info("pipeline started serial=%s", selected.serial_number)

    def stop(self) -> None:
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        except Exception:
            self.logger.exception("pipeline stop failed")
        self.pipeline = None

    def get_active_profile(self) -> ResolvedStreamProfile:
        if self._active_profile is None:
            raise FrameSourceError("未获得活动颜色流参数，请先调用 start()")
        return self._active_profile

    def get_device_info(self) -> DeviceInfo:
        if self._device_info is None:
            raise FrameSourceError("未获得设备信息，请先调用 start()")
        return self._device_info

    def read_frame(self, timeout_ms: int) -> Optional[SourceFrame]:
        if self.pipeline is None:
            raise FrameSourceError("pipeline 未启动")
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                return None
        except Exception as exc:
            raise FrameSourceError(f"wait_for_frames 失败: {exc}", exit_code=EXIT_CAPTURE_STREAM_FAIL) from exc

        color_method = getattr(frames, "get_color_frame", None)
        if not callable(color_method):
            return None
        frame = color_method()
        if frame is None:
            return None
        width = _safe_call(frame, "get_width")
        height = _safe_call(frame, "get_height")
        if width is None or height is None:
            raise FrameSourceError("帧尺寸不可用，无法写盘")
        return SourceFrame(
            data=_safe_call(frame, "get_data"),
            width=int(width),
            height=int(height),
            format=_resolve_format_name(_safe_call(frame, "get_format")) or "BGR",
            device_timestamp_ms=_to_int_or_none(_safe_call(frame, "get_timestamp")),
            frame_index=_to_int_or_none(_safe_call(frame, "get_index")),
        )

    @staticmethod
    def _resolve_format_constant(sdk_module: Any, fmt: Optional[str]) -> Any:
        if not fmt:
            raise FrameSourceError("未指定 SDK color format")
        format_enum = getattr(sdk_module, "OBFormat", None)
        if format_enum is None or not hasattr(format_enum, fmt.upper()):
            raise FrameSourceError(f"未知的 color format 常量: {fmt}")
        return getattr(format_enum, fmt.upper())

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        sdk = OrbbecFrameSource._load_sdk()
        try:
            return [info for info, _device in _sdk_devices(sdk.Context().query_devices())]
        except CaptureError:
            raise
        except Exception as exc:
            raise FrameSourceError("查询 pyorbbec 设备列表失败") from exc

    @staticmethod
    def list_color_profiles(serial_number: Optional[str]) -> list[ResolvedStreamProfile]:
        sdk = OrbbecFrameSource._load_sdk()
        try:
            _selected, device = _select_sdk_device(sdk.Context().query_devices(), serial_number)
            pipeline = sdk.Pipeline(device)
            profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
            return _profiles_from_sdk_list(profiles)
        except CaptureError:
            raise
        except Exception as exc:
            raise FrameSourceError("读取颜色流配置列表失败，请检查 pyorbbecsdk 版本") from exc

def _safe_call(obj: Any, attr: str, *call_args: Any, default: Any = None) -> Any:
    if obj is None:
        return default
    value = getattr(obj, attr, None)
    if callable(value):
        try:
            return value(*call_args)
        except TypeError:
            return default
    return value if value is not None else default


def _iter_iterable_like(value: Any) -> Iterable[Any]:
    """Iterate the collection types exposed by pyorbbecsdk2 without guessing one accessor."""
    if value is None:
        return []
    count = _safe_call(value, "get_count", default=None)
    if isinstance(count, int):
        for method_name in ("get_device_by_index", "get_stream_profile_by_index", "get_profile_by_index"):
            method = getattr(value, method_name, None)
            if callable(method):
                return [method(index) for index in range(count)]
        return []
    try:
        return list(value)
    except Exception:
        return []


def _extract_device_info(device_obj: Any) -> DeviceInfo:
    info_obj = _safe_call(device_obj, "get_device_info", default=device_obj) or device_obj
    serial = _resolve_device_prop(info_obj, ["get_serial_number", "get_serial", "serial"])
    model = _resolve_device_prop(info_obj, ["get_name", "get_name_str", "name", "model"])
    firmware = _resolve_device_prop(
        info_obj, ["get_firmware_version", "get_firmware", "firmware_version"], optional=True
    )
    if not serial:
        raise FrameSourceError("发现设备但序列号为空，无法唯一标识设备")
    if not model:
        model = "Unknown"
    return DeviceInfo(model=model, serial_number=str(serial), firmware_version=firmware)


def _sdk_devices(device_list: Any) -> list[tuple[DeviceInfo, Any]]:
    return [(_extract_device_info(device), device) for device in _iter_iterable_like(device_list)]


def _select_sdk_device(device_list: Any, serial_number: Optional[str]) -> tuple[DeviceInfo, Any]:
    available = _sdk_devices(device_list)
    selected = _select_device_by_serial([info for info, _device in available], serial_number)
    for info, device in available:
        if info.serial_number == selected.serial_number:
            return info, device
    raise CaptureError("未能绑定所选相机对象", exit_code=EXIT_NO_DEVICE)


def _resolved_profile_from_sdk_profile(
    profile: Any,
    requested: Optional[ResolvedStreamProfile] = None,
) -> ResolvedStreamProfile:
    width = _to_int_or_none(_safe_call(profile, "get_width"))
    height = _to_int_or_none(_safe_call(profile, "get_height"))
    fps = _safe_call(profile, "get_fps")
    fmt = _resolve_format_name(_safe_call(profile, "get_format"))
    if width is None or height is None or fps is None or not fmt:
        raise FrameSourceError("SDK 返回的彩色流参数不完整")
    return ResolvedStreamProfile(
        requested_width=requested.requested_width if requested else None,
        requested_height=requested.requested_height if requested else None,
        requested_fps=requested.requested_fps if requested else None,
        requested_format=requested.requested_format if requested else None,
        actual_width=width,
        actual_height=height,
        actual_fps=float(fps),
        actual_format=fmt,
    )


def _profiles_from_sdk_list(profile_list: Any) -> list[ResolvedStreamProfile]:
    profiles: list[ResolvedStreamProfile] = []
    for item in _iter_iterable_like(profile_list):
        try:
            profile = _resolved_profile_from_sdk_profile(item)
        except FrameSourceError:
            continue
        if profile not in profiles:
            profiles.append(profile)
    return profiles


def _resolve_device_prop(device_obj: Any, names: Sequence[str], optional: bool = False) -> Optional[str]:
    for name in names:
        raw = _safe_call(device_obj, name, default=None)
        if raw is not None:
            return str(raw)
    if optional:
        return None
    return None


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_format_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.upper()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").upper()
    if hasattr(value, "name"):
        return str(getattr(value, "name")).upper()
    return str(value).upper()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="果托彩色视频采集脚本")
    p.add_argument("--output-root", type=Path, default=Path("raw/tray_detect"))
    p.add_argument("--batch-id", dest="batch_id")
    p.add_argument("--serial-number", dest="serial_number")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--list-profiles", action="store_true")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=float)
    p.add_argument("--format", dest="color_format")
    p.add_argument("--duration-seconds", type=float, default=0.0)
    p.add_argument("--segment-seconds", type=float, default=0.0)
    p.add_argument("--warmup-seconds", type=float, default=2.0)
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--fixed-test", action="store_true")
    p.add_argument(
        "--planned-split",
        choices=["unassigned", "train", "val", "test", "fixed_test"],
        default="unassigned",
    )
    p.add_argument("--lighting", default="normal")
    p.add_argument("--scene-tags", default="")
    p.add_argument("--tray-count", default="unknown")
    p.add_argument(
        "--camera-moved",
        choices=["yes", "no", "unknown"],
        default="unknown",
    )
    p.add_argument(
        "--conveyor-running",
        choices=["yes", "no", "mixed", "unknown"],
        default="unknown",
    )
    p.add_argument("--notes", default="")
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--container", choices=["mp4", "avi"], default="mp4")
    p.add_argument("--frame-timeout-ms", type=int, default=1000)
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return p.parse_args(argv)


def normalize_args_to_config(args: argparse.Namespace) -> CaptureConfig:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    if args.list_devices or args.list_profiles:
        planned_split = args.planned_split
    else:
        planned_split = args.planned_split
        if args.fixed_test:
            if planned_split != "fixed_test":
                planned_split = "fixed_test"
        elif planned_split == "fixed_test":
            raise CaptureError(
                "未设置 --fixed-test 时不能将 planned-split 设为 fixed_test。",
                exit_code=EXIT_ARGUMENTS,
            )

    batch_id = args.batch_id
    if not args.fixed_test and not batch_id and not (args.list_devices or args.list_profiles):
        raise CaptureError("普通采集必须提供 --batch-id", exit_code=EXIT_ARGUMENTS)

    has_profile_fields = any(v is not None for v in [args.width, args.height, args.fps, args.color_format])
    if has_profile_fields and not all([args.width, args.height, args.fps]):
        raise CaptureError(
            "当指定颜色配置时，width/height/fps 必须全部提供；format 为可选。",
            exit_code=EXIT_ARGUMENTS,
        )
    if args.width is not None and args.width <= 0:
        raise CaptureError("--width 必须大于 0", exit_code=EXIT_ARGUMENTS)
    if args.height is not None and args.height <= 0:
        raise CaptureError("--height 必须大于 0", exit_code=EXIT_ARGUMENTS)
    if args.fps is not None and args.fps <= 0:
        raise CaptureError("--fps 必须大于 0", exit_code=EXIT_ARGUMENTS)
    if args.duration_seconds < 0:
        raise CaptureError("--duration-seconds 不能小于 0", exit_code=EXIT_ARGUMENTS)
    if args.segment_seconds < 0:
        raise CaptureError("--segment-seconds 不能小于 0", exit_code=EXIT_ARGUMENTS)
    if args.warmup_seconds < 0:
        raise CaptureError("--warmup-seconds 不能小于 0", exit_code=EXIT_ARGUMENTS)
    if args.segment_seconds > 0 and args.duration_seconds > 0:
        if args.segment_seconds > args.duration_seconds:
            logger.warning("segment-seconds 大于 duration-seconds 时，最后一段会提前结束。")
    if args.no_preview and args.duration_seconds == 0:
        logger.warning("无窗口模式下 duration-seconds=0 时仅可通过 Ctrl+C 结束。")

    if args.frame_timeout_ms < 0:
        raise CaptureError("--frame-timeout-ms 不能小于 0", exit_code=EXIT_ARGUMENTS)

    return CaptureConfig(
        output_root=output_root,
        batch_id=batch_id,
        serial_number=args.serial_number,
        list_devices=args.list_devices,
        list_profiles=args.list_profiles,
        width=args.width,
        height=args.height,
        fps=args.fps,
        color_format=(args.color_format.upper() if args.color_format else None),
        duration_seconds=float(args.duration_seconds),
        segment_seconds=float(args.segment_seconds),
        warmup_seconds=float(args.warmup_seconds),
        no_preview=args.no_preview,
        fixed_test=args.fixed_test,
        planned_split=planned_split,
        lighting=args.lighting,
        scene_tags=args.scene_tags,
        tray_count=args.tray_count,
        camera_moved=args.camera_moved,
        conveyor_running=args.conveyor_running,
        notes=args.notes,
        codec=args.codec,
        container=args.container,
        frame_timeout_ms=args.frame_timeout_ms,
        log_level=args.log_level,
    )


def sanitize_token(value: str) -> str:
    import re

    token = re.sub(r"[\\/\s:*?\"<>|]+", "_", value.strip())
    token = re.sub(r"_+", "_", token)
    token = token.strip("._")
    return token or "unknown"


def _scene_token_for_name(config: CaptureConfig) -> str:
    tags = [t.strip() for t in (config.scene_tags or "").split(",") if t.strip()]
    if tags:
        return sanitize_token(tags[0])
    return sanitize_token(config.lighting)


def _capture_timestamp() -> datetime:
    return datetime.now().astimezone()


def _format_scene_date(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%S")


def _build_capture_stem(ts: datetime, model: str, serial: str, scene: str, segment_index: int) -> str:
    return f"{_format_scene_date(ts)}_{sanitize_token(model)}_{sanitize_token(serial)}_{sanitize_token(scene)}_{segment_index:03d}"


def _resolve_capture_dir(output_root: Path, config: CaptureConfig) -> Path:
    if config.fixed_test:
        return output_root / "fixed_test_videos"
    if not config.batch_id:
        raise CaptureError("fixed_test 模式不允许 batch-id 为空")
    return output_root / sanitize_token(config.batch_id)


def _final_and_partial_paths(
    base_dir: Path, stem: str, container: str
) -> tuple[Path, Path]:
    final_path = base_dir / f"{stem}.{container}"
    partial_path = final_path.with_name(f"{final_path.stem}.partial{final_path.suffix}")
    return final_path, partial_path


def create_capture_paths(
    output_root: Path,
    config: CaptureConfig,
    model: str,
    serial: str,
    scene: str,
    timestamp: datetime,
    capture_index: int,
) -> CapturePaths:
    base_dir = _resolve_capture_dir(output_root, config)
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = _build_capture_stem(timestamp, model, serial, scene, capture_index)
    final_path, partial_path = _final_and_partial_paths(base_dir, stem, config.container)
    metadata_path = final_path.with_name(f"{final_path.stem}.json")
    frames_csv_path = final_path.with_name(f"{final_path.stem}.frames.csv")
    log_path = final_path.with_name(f"{final_path.stem}.log")
    manifest_path = output_root / "capture_manifest.csv"
    return CapturePaths(
        capture_id=stem,
        segment_index=capture_index,
        video_partial_path=partial_path,
        video_path=final_path,
        metadata_path=metadata_path,
        frames_csv_path=frames_csv_path,
        log_path=log_path,
        manifest_path=manifest_path,
    )


def resolve_next_capture_paths(
    output_root: Path,
    config: CaptureConfig,
    model: str,
    serial: str,
    scene: str,
    timestamp: datetime,
    start_index: int = 1,
) -> tuple[CapturePaths, int]:
    index = start_index
    while True:
        paths = create_capture_paths(output_root, config, model, serial, scene, timestamp, index)
        if any(
            p.exists()
            for p in (
                paths.video_path,
                paths.video_partial_path,
                paths.video_partial_path.with_name(paths.video_partial_path.name + ".incomplete"),
                paths.metadata_path,
                paths.frames_csv_path,
                paths.log_path,
            )
        ):
            index += 1
            continue
        return paths, index


def convert_color_frame_to_bgr(frame: SourceFrame) -> np.ndarray:
    fmt = sanitize_token(frame.format).upper()
    if fmt == "":
        fmt = "BGR"
    data = frame.data
    if isinstance(data, np.ndarray):
        array = data
    else:
        try:
            array = np.frombuffer(data, dtype=np.uint8)
        except (TypeError, ValueError) as exc:
            raise CaptureError(f"无法从 SDK 彩色帧读取字节数据: {exc}") from exc
    if fmt in {"BGR", "BGR8"}:
        if array.ndim == 3 and array.shape[2] == 3:
            result = array
        else:
            result = array.reshape((frame.height, frame.width, 3))
    elif fmt in {"RGB", "RGB8"}:
        if array.ndim == 1:
            array = array.reshape((frame.height, frame.width, 3))
        result = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    elif fmt in {"MJPG", "MJPEG"}:
        result = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if result is None:
            raise CaptureError("MJPG 解码失败")
    elif fmt in {"YUYV", "YUY2", "YUV422", "YUV"}:
        if array.ndim == 1:
            # YUYV is usually interleaved byte stream, reshape by width*2 bytes per pixel pair.
            result = array.reshape((-1, 2))
            result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR_YUYV)
        else:
            result = cv2.cvtColor(array, cv2.COLOR_YUV2BGR_YUYV)
    else:
        raise CaptureError(f"不支持的彩色格式: {fmt}", exit_code=EXIT_UNSUPPORTED_PROFILE)
    if result is None:
        raise CaptureError(f"格式转换后未得到帧: {fmt}", exit_code=EXIT_UNSUPPORTED_PROFILE)
    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)
    if result.dtype != np.uint8:
        result = result.astype(np.uint8)
    if result.ndim != 3 or result.shape[2] != 3:
        raise CaptureError(f"转换后的帧形状不符合 BGR 预期: {result.shape}", exit_code=EXIT_UNSUPPORTED_PROFILE)
    return result


def create_video_writer(
    path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str,
    container: str,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        float(max(1.0, fps)),
        (int(width), int(height)),
        isColor=True,
    )
    if not writer.isOpened():
        raise VideoWriterError(
            f"VideoWriter 打开失败，container={container} codec={codec} size={width}x{height} fps={fps}",
            exit_code=EXIT_ENCODER_FAIL,
        )
    return writer


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_is_finite(value: Optional[float]) -> bool:
    return value is not None and isinstance(value, float) and math.isfinite(value)


def _serialize_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_json_safe(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_serialize_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_serialize_json_safe(v) for v in sorted(value)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def build_capture_metadata(
    config: CaptureConfig,
    result: CaptureResult,
    device: DeviceInfo,
    profile: ResolvedStreamProfile,
    software: dict[str, Optional[str]],
    frame_csv_path: Path,
    status: str,
    video_readable: bool,
    sha256_value: Optional[str],
) -> dict[str, Any]:
    elapsed = result.duration_seconds
    stats = result.statistics
    estimated_dropped = stats.estimated_dropped_frames
    if profile.actual_fps and elapsed > 0:
        estimated = round((elapsed * profile.actual_fps) - stats.frames_written)
        if estimated > 0:
            estimated_dropped = int(estimated)
    metadata = {
        "schema_version": "1.0",
        "capture_id": result.capture_id,
        "status": status,
        "video_path": _to_repo_relative(result.video_path) if result.video_path else None,
        "frames_csv_path": _to_repo_relative(frame_csv_path),
        "batch_id": config.batch_id,
        "fixed_test": config.fixed_test,
        "planned_split": config.planned_split,
        "capture_start_time": result.capture_start_time.isoformat(),
        "capture_end_time": result.capture_end_time.isoformat(),
        "duration_seconds": float(elapsed),
        "device": {
            "model": device.model,
            "serial_number": device.serial_number,
            "firmware_version": device.firmware_version,
        },
        "stream": {
            "sensor": "color",
            "requested_width": profile.requested_width,
            "requested_height": profile.requested_height,
            "requested_fps": profile.requested_fps,
            "requested_format": profile.requested_format,
            "actual_width": profile.actual_width,
            "actual_height": profile.actual_height,
            "actual_fps": profile.actual_fps,
            "actual_format": profile.actual_format,
        },
        "encoding": {"container": config.container, "codec": config.codec},
        "statistics": {
            "frames_received": stats.frames_received,
            "frames_written": stats.frames_written,
            "conversion_failures": stats.conversion_failures,
            "write_failures": stats.write_failures,
            "timeouts": stats.timeouts,
            "estimated_dropped_frames": estimated_dropped,
            "measured_fps": stats.measured_fps(elapsed),
        },
        "scene": {
            "lighting": config.lighting,
            "scene_tags": [t for t in config.scene_tags.split(",") if t],
            "tray_count": config.tray_count,
            "camera_moved": config.camera_moved,
            "conveyor_running": config.conveyor_running,
            "notes": config.notes,
        },
        "integrity": {
            "video_sha256": sha256_value,
            "video_readable_by_opencv": video_readable,
        },
        "software": software,
        "resolved_arguments": {
            "width": profile.requested_width or profile.actual_width,
            "height": profile.requested_height or profile.actual_height,
            "fps": profile.requested_fps or profile.actual_fps,
            "format": profile.requested_format or profile.actual_format,
            "codec": config.codec,
            "container": config.container,
            "frame_timeout_ms": config.frame_timeout_ms,
        },
    }
    return _serialize_json_safe(metadata)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)


@contextmanager
def _manifest_lock(manifest_path: Path) -> Iterable[None]:
    """Use a short Windows advisory lock while appending the shared manifest."""
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        locked = False
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            locked = True
        try:
            yield
        finally:
            if locked:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def append_capture_manifest(manifest_path: Path, row: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock(manifest_path):
        needs_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
        with manifest_path.open("a", encoding="utf-8-sig", newline="") as writer_file:
            writer = csv.DictWriter(writer_file, fieldnames=MANIFEST_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
            writer_file.flush()
            os.fsync(writer_file.fileno())


def validate_video_file(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    try:
        ok, frame = cap.read()
        if not ok:
            return False
        if frame is None:
            return False
        if frame.size == 0:
            return False
        return True
    finally:
        cap.release()


def _software_versions() -> dict[str, Optional[str]]:
    pyver = ".".join(map(str, sys.version_info[:3]))
    sdk_version = None
    try:
        sdk = importlib.import_module("pyorbbecsdk")
        sdk_version = getattr(sdk, "__version__", None)
    except ModuleNotFoundError:
        sdk_version = None
    return {
        "python": pyver,
        "opencv": cv2.__version__,
        "pyorbbecsdk": str(sdk_version) if sdk_version else None,
        "platform": platform.system(),
    }


def _select_device_by_serial(
    available: list[DeviceInfo],
    serial_number: Optional[str],
) -> DeviceInfo:
    if serial_number:
        for item in available:
            if item.serial_number == serial_number:
                return item
        raise CaptureError(f"未找到序列号 {serial_number} 的设备", exit_code=EXIT_NO_DEVICE)
    if len(available) == 0:
        raise CaptureError("未发现可用设备", exit_code=EXIT_NO_DEVICE)
    if len(available) > 1:
        raise CaptureError(
            "检测到多台设备，必须使用 --serial-number 明确指定采集设备。", exit_code=EXIT_DEVICE_AMBIGUOUS
        )
    return available[0]


def run_list_devices() -> int:
    devices = OrbbecFrameSource.list_devices()
    if not devices:
        raise CaptureError("未发现可用设备", exit_code=EXIT_NO_DEVICE)
    for idx, device in enumerate(devices, 1):
        logger.info(
            "DEVICE %d model=%s serial=%s firmware=%s",
            idx,
            device.model,
            device.serial_number,
            device.firmware_version or "unknown",
        )
    return EXIT_SUCCESS


def run_list_profiles(config: CaptureConfig) -> int:
    devices = OrbbecFrameSource.list_devices()
    if not devices:
        raise CaptureError("未发现可用设备", exit_code=EXIT_NO_DEVICE)
    selected = _select_device_by_serial(devices, config.serial_number)
    logger.info(
        "selected device serial=%s model=%s",
        selected.serial_number,
        selected.model,
    )
    profiles = OrbbecFrameSource.list_color_profiles(selected.serial_number)
    if not profiles:
        raise CaptureError("未读取到该设备颜色流配置", exit_code=EXIT_UNSUPPORTED_PROFILE)
    for profile in profiles:
        logger.info(
            "PROFILE %dx%d fps=%s format=%s",
            profile.actual_width,
            profile.actual_height,
            profile.actual_fps,
            profile.actual_format,
        )
    return EXIT_SUCCESS


def _log_status(
    frame: np.ndarray,
    segment_index: int,
    captured_seconds: float,
    stats: CaptureStatistics,
    status: str,
) -> Optional[str]:
    overlay = frame.copy()
    lines = [
        "model: pending",
        f"segment: {segment_index}",
        f"captured_s: {captured_seconds:.1f}",
        f"frames_written: {stats.frames_written}",
        f"timeouts: {stats.timeouts}",
        f"status: {status}",
        "q / Esc 结束 | n 新建片段",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (10, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow("tray_capture", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        return "stop"
    if key == ord("n"):
        return "next"
    return None


def _safe_close_preview_window(window_name: str) -> None:
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except Exception:
        return


def _preview_status_line(
    frame: np.ndarray,
    model: str,
    serial: str,
    profile: ResolvedStreamProfile,
    segment: int,
    duration: float,
    stats: CaptureStatistics,
) -> str:
    overlay = frame.copy()
    rows = [
        f"Device: {model} SN={serial}",
        f"Color: {profile.actual_width}x{profile.actual_height} @{profile.actual_fps:.2f}",
        f"Segment: #{segment}",
        f"Duration: {duration:.1f}s",
        f"Written: {stats.frames_written} Timeout: {stats.timeouts}",
    ]
    for i, row in enumerate(rows):
        cv2.putText(
            overlay,
            row,
            (10, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow("tray_capture", overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        return "stop"
    if key == ord("n"):
        return "next"
    return None


def run_capture(
    config: CaptureConfig,
    source: FrameSourceBase,
    video_writer_factory: Callable[
        [Path, int, int, float, str, str], cv2.VideoWriter
    ] = create_video_writer,
) -> list[CaptureResult]:
    results: list[CaptureResult] = []
    segment_index = 1
    scene = _scene_token_for_name(config)
    output_root = config.output_root
    manifest_path = config.output_root / "capture_manifest.csv"
    started = False

    def _finalize_segment(
        segment_start_time: datetime,
        segment_end_time: datetime,
        paths: CapturePaths,
        stats: CaptureStatistics,
        profile: ResolvedStreamProfile,
        device: DeviceInfo,
        status: str,
        error: Optional[str] = None,
    ) -> CaptureResult:
        segment_elapsed = (segment_end_time - segment_start_time).total_seconds()
        elapsed_for_fps = max(segment_elapsed, 1e-6)
        readable = False
        if stats.frames_written > 0:
            readable = validate_video_file(paths.video_partial_path)
            if readable:
                final_path = paths.video_partial_path
            else:
                if status == "completed":
                    status = "write_error"
                final_path = paths.video_partial_path.with_name(
                    paths.video_partial_path.name + ".incomplete"
                )
                if paths.video_partial_path.exists():
                    paths.video_partial_path.replace(final_path)
        else:
            final_path = None
            if paths.video_partial_path.exists():
                paths.video_partial_path.unlink()

        sha256_value = compute_sha256(final_path) if final_path and final_path.exists() and readable else None
        record_video_path = paths.video_path if status == "completed" and readable else final_path
        metadata = build_capture_metadata(
            config,
            CaptureResult(
                capture_id=paths.capture_id,
                batch_id=config.batch_id or "fixed_test",
                status=status,
                video_path=record_video_path,
                metadata_path=paths.metadata_path,
                frames_csv_path=paths.frames_csv_path,
                log_path=paths.log_path,
                capture_start_time=segment_start_time,
                capture_end_time=segment_end_time,
                duration_seconds=segment_elapsed,
                width=profile.actual_width,
                height=profile.actual_height,
                fps=profile.actual_fps,
                codec=config.codec,
                requested_fps=profile.requested_fps,
                statistics=stats,
                scene_tags=config.scene_tags,
                fixed_test=config.fixed_test,
                planned_split=config.planned_split,
            ),
            device,
            profile,
            _software_versions(),
            paths.frames_csv_path,
            status=status,
            video_readable=readable,
            sha256_value=sha256_value,
        )
        write_json(paths.metadata_path, metadata)
        if stats.frames_written > 0:
            manifest_row = {
                "capture_id": paths.capture_id,
                "batch_id": config.batch_id or "fixed_test",
                "video_path": _to_repo_relative(record_video_path),
                "metadata_path": _to_repo_relative(paths.metadata_path),
                "frames_csv_path": _to_repo_relative(paths.frames_csv_path),
                "capture_date": segment_start_time.date().isoformat(),
                "start_time": segment_start_time.isoformat(),
                "end_time": segment_end_time.isoformat(),
                "duration_seconds": f"{segment_elapsed:.6f}",
                "width": profile.actual_width,
                "height": profile.actual_height,
                "fps": profile.actual_fps,
                "requested_fps": profile.requested_fps or profile.actual_fps,
                "measured_fps": f"{stats.measured_fps(elapsed_for_fps):.6f}",
                "codec": config.codec,
                "frame_count": stats.frames_written,
                "device_model": device.model,
                "device_serial": device.serial_number,
                "lighting": config.lighting,
                "scene": config.scene_tags or "",
                "tray_count": config.tray_count,
                "camera_moved": config.camera_moved,
                "conveyor_running": config.conveyor_running,
                "planned_split": config.planned_split,
                "fixed_test": "true" if config.fixed_test else "false",
                "status": status,
                "video_sha256": sha256_value or "",
                "notes": config.notes,
            }
            append_capture_manifest(manifest_path, manifest_row)
        # A completed video is published only after its metadata and manifest
        # entry are safely written. Interrupted but readable recordings remain
        # as *.partial.<container> and are explicitly marked in their JSON.
        published_path = record_video_path
        if status == "completed" and readable and final_path is not None:
            paths.video_partial_path.replace(paths.video_path)
            published_path = paths.video_path
        return CaptureResult(
            capture_id=paths.capture_id,
            batch_id=config.batch_id or "fixed_test",
            status=status,
            video_path=published_path,
            metadata_path=paths.metadata_path,
            frames_csv_path=paths.frames_csv_path,
            log_path=paths.log_path,
            capture_start_time=segment_start_time,
            capture_end_time=segment_end_time,
            duration_seconds=segment_elapsed,
            width=profile.actual_width,
            height=profile.actual_height,
            fps=profile.actual_fps,
            codec=config.codec,
            requested_fps=profile.requested_fps,
            statistics=stats,
            scene_tags=config.scene_tags,
            fixed_test=config.fixed_test,
            planned_split=config.planned_split,
            error=error,
        )

    try:
        source.start()
        started = True
        profile = source.get_active_profile()
        try:
            device = source.get_device_info()
        except (AttributeError, NotImplementedError):
            device = DeviceInfo(model="Gemini 435Le", serial_number=config.serial_number or "unknown")
        serial = device.serial_number
        supported_input_formats = {"BGR", "BGR8", "RGB", "RGB8", "MJPG", "MJPEG", "YUYV", "YUY2", "YUV422", "YUV"}
        if sanitize_token(profile.actual_format).upper() not in supported_input_formats:
            raise FrameSourceError(
                f"当前活动彩色流格式不受支持: {profile.actual_format}；请用 --list-profiles 查看 RGB/BGR/MJPG/YUYV 配置。",
                exit_code=EXIT_UNSUPPORTED_PROFILE,
            )
        warmup_end = time.monotonic() + config.warmup_seconds
        warmup_misses = 0
        while time.monotonic() < warmup_end:
            warmup_frame = source.read_frame(config.frame_timeout_ms)
            if warmup_frame is None:
                warmup_misses += 1
            else:
                warmup_misses = 0
            if warmup_misses >= 5:
                raise FrameSourceError(
                    "预热期间连续超时，可能设备未就绪。",
                    exit_code=EXIT_CAPTURE_STREAM_FAIL,
                )
        overall_start_ns = time.monotonic()
        capture_time = _capture_timestamp()
        while True:
            if config.duration_seconds > 0 and (time.monotonic() - overall_start_ns) >= config.duration_seconds:
                break
            paths, segment_index = resolve_next_capture_paths(
                output_root,
                config,
                device.model,
                serial,
                scene,
                capture_time,
                start_index=segment_index,
            )
            segment_start = datetime.now().astimezone()
            segment_logger = logging.getLogger(f"segment-{paths.capture_id}")
            segment_logger.setLevel(getattr(logging, config.log_level))
            handler = logging.FileHandler(paths.log_path, encoding="utf-8")
            segment_logger.addHandler(handler)
            segment_logger.info("selected device model=%s serial=%s firmware=%s", device.model, serial, device.firmware_version)
            segment_logger.info("color profile=%dx%d@%s format=%s", profile.actual_width, profile.actual_height, profile.actual_fps, profile.actual_format)
            segment_logger.info("partial_video=%s final_video=%s", paths.video_partial_path, paths.video_path)
            status = "completed"
            error: Optional[str] = None
            next_segment_requested = False
            stop_requested = False
            writer: Optional[cv2.VideoWriter] = None
            stats = CaptureStatistics()
            csv_file = None
            pending_error: Optional[BaseException] = None
            segment_start_ns = time.monotonic()
            try:
                writer = video_writer_factory(
                    paths.video_partial_path, profile.actual_width, profile.actual_height,
                    profile.actual_fps, config.codec, config.container,
                )
                if not writer.isOpened():
                    raise VideoWriterError(f"VideoWriter 打开失败：{paths.video_partial_path}", exit_code=EXIT_ENCODER_FAIL)
                csv_file = paths.frames_csv_path.open("w", encoding="utf-8-sig", newline="")
                csv_writer = csv.DictWriter(csv_file, fieldnames=[
                    "video_frame_index", "sdk_frame_index", "device_timestamp_ms", "host_monotonic_ns",
                    "host_time_iso", "inter_frame_delta_ms", "conversion_ok", "write_ok",
                ])
                csv_writer.writeheader()
                last_host_ns: Optional[int] = None
                source_frame_index = 0
                consecutive_timeout = 0
                while True:
                    elapsed_total = time.monotonic() - overall_start_ns
                    elapsed_segment = time.monotonic() - segment_start_ns
                    if config.duration_seconds > 0 and elapsed_total >= config.duration_seconds:
                        break
                    if config.segment_seconds > 0 and elapsed_segment >= config.segment_seconds:
                        break
                    source_frame = source.read_frame(config.frame_timeout_ms)
                    if source_frame is None:
                        stats.timeouts += 1
                        consecutive_timeout += 1
                        if consecutive_timeout >= 5:
                            status = "camera_error"
                            error = "连续取帧超时"
                            raise FrameSourceError(
                                "连续取帧超时",
                                exit_code=EXIT_CAPTURE_STREAM_FAIL,
                            )
                        continue
                    if isinstance(source_frame, BaseException):
                        raise source_frame
                    consecutive_timeout = 0
                    host_ns = time.monotonic_ns()
                    source_frame_index += 1
                    stats.frames_received += 1
                    conversion_ok = True
                    write_ok = False
                    bgr_frame: Optional[np.ndarray] = None
                    try:
                        bgr_frame = convert_color_frame_to_bgr(source_frame)
                    except CaptureError as exc:
                        conversion_ok = False
                        write_ok = False
                        stats.conversion_failures += 1
                        segment_logger.warning("source frame %d conversion failed: %s", source_frame_index, exc)

                    if conversion_ok and bgr_frame is not None:
                        try:
                            if bgr_frame.shape[:2] != (profile.actual_height, profile.actual_width):
                                raise VideoWriterError(
                                    f"转换后帧尺寸 {bgr_frame.shape[1]}x{bgr_frame.shape[0]} 与活动流不一致 "
                                    f"{profile.actual_width}x{profile.actual_height}",
                                    exit_code=EXIT_CAPTURE_STREAM_FAIL,
                                )
                            writer.write(bgr_frame)
                            write_ok = True
                            stats.frames_written += 1
                        except Exception as exc:
                            stats.write_failures += 1
                            status = "write_error"
                            error = str(exc)
                            raise VideoWriterError(
                                f"写盘失败: {exc}", exit_code=EXIT_ENCODER_FAIL
                            )

                    inter_delta_ms = (
                        "" if last_host_ns is None else f"{(host_ns - last_host_ns) / 1_000_000:.3f}"
                    )
                    csv_writer.writerow(
                        {
                            "video_frame_index": stats.frames_written if write_ok else "",
                            "sdk_frame_index": source_frame.frame_index if source_frame.frame_index is not None else "",
                            "device_timestamp_ms": source_frame.device_timestamp_ms if source_frame.device_timestamp_ms is not None else "",
                            "host_monotonic_ns": host_ns,
                            "host_time_iso": datetime.now().astimezone().isoformat(),
                            "inter_frame_delta_ms": inter_delta_ms,
                            "conversion_ok": str(conversion_ok).lower(),
                            "write_ok": str(write_ok).lower(),
                        }
                    )
                    last_host_ns = host_ns
                    segment_logger.info("source_frame=%d conversion_ok=%s write_ok=%s", source_frame_index, conversion_ok, write_ok)
                    if not config.no_preview:
                        preview_frame = bgr_frame
                        if preview_frame is None:
                            preview_frame = np.zeros(
                                (profile.actual_height, profile.actual_width, 3), dtype=np.uint8
                            )
                        action = _preview_status_line(
                            preview_frame,
                            device.model,
                            serial,
                            profile,
                            segment_index,
                            time.monotonic() - segment_start_ns,
                            stats,
                        )
                        if action == "next":
                            next_segment_requested = True
                            status = "completed"
                            break
                        if action == "stop":
                            stop_requested = True
                            status = "completed"
                            break
            except KeyboardInterrupt as exc:
                status = "interrupted"
                error = "KeyboardInterrupt"
                pending_error = exc
                segment_logger.warning("capture interrupted by keyboard")
            except VideoWriterError as exc:
                status = "write_error"
                error = str(exc)
                pending_error = exc
                segment_logger.exception("video writer failure")
            except FrameSourceError as exc:
                status = "camera_error"
                error = str(exc)
                pending_error = exc
                segment_logger.exception("camera stream failure")
            except Exception as exc:
                status = "interrupted"
                error = f"{type(exc).__name__}: {exc}"
                pending_error = exc
                segment_logger.exception("unexpected capture failure")
            finally:
                segment_end = datetime.now().astimezone()
                if csv_file is not None:
                    csv_file.close()
                if writer is not None:
                    writer.release()
                for h in list(segment_logger.handlers):
                    h.flush()
                    h.close()
                    segment_logger.removeHandler(h)

            if not config.no_preview:
                _safe_close_preview_window("tray_capture")
            result = _finalize_segment(
                segment_start,
                segment_end,
                paths,
                stats,
                profile,
                device,
                status,
                error=error,
            )
            results.append(result)
            logger.info("segment finalized status=%s readable=%s manifest=%s", result.status, result.video_path is not None, manifest_path)
            segment_index += 1
            if pending_error is not None:
                raise pending_error
            if stop_requested:
                break
            if status != "completed" or stop_requested:
                break
            if next_segment_requested:
                if config.duration_seconds == 0 or (time.monotonic() - overall_start_ns) < config.duration_seconds:
                    continue
            if config.segment_seconds > 0 and (config.duration_seconds == 0 or elapsed_total < config.duration_seconds):
                if next_segment_requested or config.duration_seconds == 0:
                    continue
                if config.duration_seconds > 0 and (time.monotonic() - overall_start_ns) < config.duration_seconds:
                    continue
            if status == "completed" and config.segment_seconds <= 0 and config.duration_seconds > 0:
                if (time.monotonic() - overall_start_ns) < config.duration_seconds:
                    continue
            if status == "completed" and config.segment_seconds <= 0 and config.duration_seconds == 0:
                # 无窗口且无限时长，等待用户 Ctrl+C
                if config.no_preview:
                    time.sleep(0.01)
                    continue
            break
    finally:
        if started:
            source.stop()
        if not config.no_preview:
            _safe_close_preview_window("tray_capture")
            cv2.destroyAllWindows()
    return results


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        config = normalize_args_to_config(args)
        _configure_logging(config.log_level)

        if args.list_devices:
            return run_list_devices()
        if args.list_profiles:
            return run_list_profiles(config)

        source = OrbbecFrameSource(config.serial_number, None, logger)
        requested_profile = None
        if args.width is not None:
            requested_profile = ResolvedStreamProfile(
                requested_width=args.width,
                requested_height=args.height,
                requested_fps=args.fps,
                requested_format=args.color_format,
                actual_width=args.width,
                actual_height=args.height or 0,
                actual_fps=args.fps or 0.0,
                actual_format=args.color_format or "UNKNOWN",
            )
        if requested_profile is not None:
            source = OrbbecFrameSource(
                serial_number=config.serial_number,
                requested_profile=requested_profile,
                logger_=logger,
            )

        logger.info("start capture batch_id=%s fixed_test=%s", config.batch_id, config.fixed_test)
        results = run_capture(config, source)
        return EXIT_SUCCESS if all(r.status == "completed" for r in results) else EXIT_VERIFY_FAIL
    except CaptureError as exc:
        logger.error("%s", exc)
        return exc.exit_code
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return EXIT_SUCCESS
    except Exception as exc:
        logger.exception("capture failed: %s", exc)
        return EXIT_UNKNOWN


if __name__ == "__main__":
    raise SystemExit(main())
