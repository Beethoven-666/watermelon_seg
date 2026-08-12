"""YOLO26s Detect + ByteTrack 连续果托跟踪。"""

from __future__ import annotations
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable
import math
import numpy as np
from .model_utils import resolve_half, validate_model_contract
from .roi import PixelROI, box_center, crop_frame, map_box_to_full
from .types import TrayDetection, TrayTrack
from .watermelon_segmenter import _array


def parse_tray_result(
    result: Any,
    *,
    roi: PixelROI,
    frame_width: int,
    frame_height: int,
    frame_index: int,
    timestamp_ms: float,
) -> tuple[list[TrayTrack], list[TrayDetection]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], []
    xyxy = _array(boxes.xyxy)
    conf = _array(boxes.conf).reshape(-1)
    classes = _array(boxes.cls).reshape(-1)
    raw_ids = getattr(boxes, "id", None)
    ids = None if raw_ids is None else _array(raw_ids).reshape(-1)
    tracks = []
    unconfirmed = []
    for index, local_box in enumerate(xyxy):
        if int(classes[index]) != 0:
            continue
        full_box = map_box_to_full(local_box, roi, frame_width, frame_height)
        center = box_center(full_box)
        common = dict(
            confidence=float(conf[index]),
            bbox_xyxy_full=full_box,
            center_full=center,
            frame_index=frame_index,
            timestamp_ms=float(timestamp_ms),
        )
        track_id = None if ids is None or index >= len(ids) else float(ids[index])
        if (
            track_id is None
            or not math.isfinite(track_id)
            or track_id < 0
            or not track_id.is_integer()
        ):
            unconfirmed.append(TrayDetection(**common))
        else:
            tracks.append(TrayTrack(track_id=int(track_id), **common))
    return tracks, unconfirmed


class TrayTracker:
    def __init__(
        self,
        weights: str | Path,
        tracker_config: str | Path,
        imgsz: int,
        conf: float,
        iou: float,
        device: Any,
        half: bool,
        history_length: int = 30,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ):
        weights_path = Path(weights)
        tracker_path = Path(tracker_config)
        if not weights_path.is_file():
            raise FileNotFoundError(f"tray weights not found: {weights_path}")
        if not tracker_path.is_file():
            raise FileNotFoundError(f"tracker config not found: {tracker_path}")
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self._factory = model_factory
        self._weights = str(weights_path)
        self.model = model_factory(self._weights)
        validate_model_contract(
            self.model, task="detect", class_id=0, class_name="tray"
        )
        self.tracker_config = tracker_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.half = resolve_half(device, bool(half))
        self.history_length = int(history_length)
        self.history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        roi: PixelROI,
        frame_index: int,
        timestamp_ms: float,
    ) -> tuple[list[TrayTrack], list[TrayDetection]]:
        results = self.model.track(
            source=crop_frame(frame_bgr, roi),
            persist=True,
            tracker=str(self.tracker_config),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            classes=[0],
            verbose=False,
        )
        if not results:
            return [], []
        height, width = frame_bgr.shape[:2]
        tracks, unconfirmed = parse_tray_result(
            results[0],
            roi=roi,
            frame_width=width,
            frame_height=height,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        for track in tracks:
            self.history[track.track_id].append(track.center_full)
        return tracks, unconfirmed

    def reset(self) -> None:
        # Ultralytics predictor owns tracker state; a fresh model is required when a stream changes.
        self.model = self._factory(self._weights)
        validate_model_contract(
            self.model, task="detect", class_id=0, class_name="tray"
        )
        self.history.clear()

    def close(self) -> None:
        self.model = None
        self.history.clear()
