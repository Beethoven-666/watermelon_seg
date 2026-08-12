"""YOLO26s-seg 西瓜实例分割与完整画面坐标解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .model_utils import resolve_half, validate_model_contract
from .roi import PixelROI, box_center, crop_frame, map_box_to_full, map_points_to_full
from .types import WatermelonInstance


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _mask_centroid(mask: np.ndarray) -> tuple[tuple[float, float] | None, int]:
    binary = (_array(mask) > 0.5).astype(np.uint8)
    area = int(binary.sum())
    moments = cv2.moments(binary, binaryImage=True)
    if area <= 0 or moments["m00"] <= 0:
        return None, area
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    ), area


def parse_watermelon_result(
    result: Any,
    *,
    roi: PixelROI,
    frame_width: int,
    frame_height: int,
    frame_index: int,
    timestamp_ms: float,
) -> list[WatermelonInstance]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or len(boxes) == 0:
        return []
    if (
        masks is None
        or getattr(masks, "xy", None) is None
        or getattr(masks, "data", None) is None
    ):
        raise ValueError("segmentation result contains boxes but no instance masks")
    xyxy = _array(boxes.xyxy)
    confidences = _array(boxes.conf).reshape(-1)
    classes = _array(boxes.cls).reshape(-1)
    polygons = list(masks.xy)
    mask_data = _array(masks.data)
    count = len(xyxy)
    if len(polygons) != count or len(mask_data) != count:
        raise ValueError("box and mask instance counts differ")
    output: list[WatermelonInstance] = []
    for index in range(count):
        if int(classes[index]) != 0:
            continue
        full_box = map_box_to_full(xyxy[index], roi, frame_width, frame_height)
        local_polygon = _array(polygons[index]).reshape(-1, 2)
        full_polygon = map_points_to_full(local_polygon, roi, frame_width, frame_height)
        centroid_local, area = _mask_centroid(mask_data[index])
        fallback_reason: str | None = None
        if centroid_local is None:
            centroid = box_center(full_box)
            fallback_reason = "empty_or_zero_moment_mask"
        else:
            # masks.data may be lower resolution than the ROI; scale before offset.
            mask_height, mask_width = mask_data[index].shape[-2:]
            centroid_scaled = (
                centroid_local[0] * roi.width / max(mask_width, 1),
                centroid_local[1] * roi.height / max(mask_height, 1),
            )
            centroid = map_points_to_full(
                [centroid_scaled], roi, frame_width, frame_height
            )[0]
            area = round(
                area * roi.width * roi.height / max(mask_width * mask_height, 1)
            )
        output.append(
            WatermelonInstance(
                instance_id=len(output),
                confidence=float(confidences[index]),
                bbox_xyxy_full=full_box,
                centroid_full=centroid,
                polygon_full=full_polygon,
                mask_area_pixels=int(area),
                frame_index=frame_index,
                timestamp_ms=float(timestamp_ms),
                centroid_fallback_reason=fallback_reason,
            )
        )
    return output


class WatermelonSegmenter:
    def __init__(
        self,
        weights: str | Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: Any,
        half: bool,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ):
        weights_path = Path(weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"watermelon weights not found: {weights_path}")
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self.model = model_factory(str(weights_path))
        validate_model_contract(
            self.model, task="segment", class_id=0, class_name="watermelon"
        )
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.half = resolve_half(device, bool(half))

    def predict(
        self,
        frame_bgr: np.ndarray,
        roi: PixelROI,
        frame_index: int,
        timestamp_ms: float,
    ) -> list[WatermelonInstance]:
        crop = crop_frame(frame_bgr, roi)
        results = self.model.predict(
            source=crop,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            classes=[0],
            verbose=False,
        )
        if not results:
            return []
        height, width = frame_bgr.shape[:2]
        return parse_watermelon_result(
            results[0],
            roi=roi,
            frame_width=width,
            frame_height=height,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        self.model = None
