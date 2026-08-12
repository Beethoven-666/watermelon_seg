"""不含框架对象的运行结果类型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any


@dataclass(frozen=True)
class WatermelonInstance:
    instance_id: int
    confidence: float
    bbox_xyxy_full: tuple[float, float, float, float]
    centroid_full: tuple[float, float]
    polygon_full: list[tuple[float, float]]
    mask_area_pixels: int
    frame_index: int
    timestamp_ms: float
    centroid_fallback_reason: str | None = None


@dataclass(frozen=True)
class TrayTrack:
    track_id: int
    confidence: float
    bbox_xyxy_full: tuple[float, float, float, float]
    center_full: tuple[float, float]
    frame_index: int
    timestamp_ms: float


@dataclass(frozen=True)
class TrayDetection:
    confidence: float
    bbox_xyxy_full: tuple[float, float, float, float]
    center_full: tuple[float, float]
    frame_index: int
    timestamp_ms: float


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    timestamp_ms: float
    watermelons: list[WatermelonInstance]
    tray_tracks: list[TrayTrack]
    unconfirmed_trays: list[TrayDetection]
    watermelon_result_age_frames: int
    capture_fps: float
    processing_fps: float


def _plain(value: Any) -> Any:
    """Return JSON-native values and reject framework/numeric proxy objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON output contains NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    raise TypeError(f"value is not JSON-native: {type(value).__name__}")


def to_json_dict(result: FrameResult) -> dict[str, Any]:
    return _plain(asdict(result))


def dumps_result(result: FrameResult) -> str:
    return json.dumps(
        to_json_dict(result), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
