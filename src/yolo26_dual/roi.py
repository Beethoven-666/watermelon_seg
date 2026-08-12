"""归一化 ROI、裁剪和完整画面坐标映射。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PixelROI:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def validate_normalized_roi(roi: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in roi)
    if len(values) != 4:
        raise ValueError("ROI must contain exactly four values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ROI values must be finite")
    x1, y1, x2, y2 = values
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("ROI values must be in the range 0..1")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must have positive width and height")
    return values


def normalized_to_pixel_roi(
    roi: Iterable[float], frame_width: int, frame_height: int
) -> PixelROI:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    x1, y1, x2, y2 = validate_normalized_roi(roi)
    pixel = PixelROI(
        x1=min(frame_width, max(0, round(x1 * frame_width))),
        y1=min(frame_height, max(0, round(y1 * frame_height))),
        x2=min(frame_width, max(0, round(x2 * frame_width))),
        y2=min(frame_height, max(0, round(y2 * frame_height))),
    )
    if pixel.width <= 0 or pixel.height <= 0:
        raise ValueError("ROI is empty at the current frame resolution")
    return pixel


def crop_frame(frame_bgr: np.ndarray, roi: PixelROI) -> np.ndarray:
    if frame_bgr.ndim not in (2, 3):
        raise ValueError("frame must be a 2-D or 3-D image array")
    height, width = frame_bgr.shape[:2]
    if roi.x1 < 0 or roi.y1 < 0 or roi.x2 > width or roi.y2 > height:
        raise ValueError("pixel ROI is outside the frame")
    return frame_bgr[roi.y1 : roi.y2, roi.x1 : roi.x2]


def _clip(value: float, maximum: int) -> float:
    return min(float(maximum), max(0.0, float(value)))


def map_box_to_full(
    box_xyxy_local: Iterable[float], roi: PixelROI, frame_width: int, frame_height: int
) -> tuple[float, float, float, float]:
    box = tuple(float(value) for value in box_xyxy_local)
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("box must contain four finite values")
    x1, y1, x2, y2 = box
    return (
        _clip(roi.x1 + x1, frame_width),
        _clip(roi.y1 + y1, frame_height),
        _clip(roi.x1 + x2, frame_width),
        _clip(roi.y1 + y2, frame_height),
    )


def map_points_to_full(
    points_local: Iterable[Iterable[float]],
    roi: PixelROI,
    frame_width: int,
    frame_height: int,
) -> list[tuple[float, float]]:
    mapped: list[tuple[float, float]] = []
    for point in points_local:
        values = tuple(float(value) for value in point)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("each point must contain two finite values")
        mapped.append(
            (
                _clip(roi.x1 + values[0], frame_width),
                _clip(roi.y1 + values[1], frame_height),
            )
        )
    return mapped


def box_center(box_xyxy: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
