"""双 ROI 二维结果可视化。"""

from __future__ import annotations
import cv2
import numpy as np
from .roi import PixelROI
from .types import FrameResult


def draw_frame(
    frame: np.ndarray,
    result: FrameResult,
    watermelon_roi: PixelROI | None,
    tray_roi: PixelROI | None,
    track_history: dict | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    for item in result.watermelons:
        polygon = np.asarray(item.polygon_full, dtype=np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(overlay, [polygon], (0, 210, 80))
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)
    for item in result.watermelons:
        x1, y1, x2, y2 = map(round, item.bbox_xyxy_full)
        cx, cy = map(round, item.centroid_full)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cv2.circle(canvas, (cx, cy), 4, (0, 255, 120), -1)
        cv2.putText(
            canvas,
            f"watermelon #{item.instance_id} {item.confidence:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )
    for item in result.tray_tracks:
        x1, y1, x2, y2 = map(round, item.bbox_xyxy_full)
        cx, cy = map(round, item.center_full)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.circle(canvas, (cx, cy), 4, (255, 160, 0), -1)
        cv2.putText(
            canvas,
            f"tray ID={item.track_id} {item.confidence:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 160, 0),
            1,
            cv2.LINE_AA,
        )
        if track_history and item.track_id in track_history:
            points = np.asarray(track_history[item.track_id], dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(canvas, [points], False, (255, 160, 0), 2)
    for item in result.unconfirmed_trays:
        x1, y1, x2, y2 = map(round, item.bbox_xyxy_full)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 1)
        cv2.putText(
            canvas,
            f"tray unconfirmed {item.confidence:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    if watermelon_roi:
        cv2.rectangle(
            canvas,
            (watermelon_roi.x1, watermelon_roi.y1),
            (watermelon_roi.x2, watermelon_roi.y2),
            (0, 150, 0),
            1,
        )
    if tray_roi:
        cv2.rectangle(
            canvas,
            (tray_roi.x1, tray_roi.y1),
            (tray_roi.x2, tray_roi.y2),
            (180, 80, 0),
            1,
        )
    lines = [
        f"Watermelons: {len(result.watermelons)}",
        f"Tray tracks: {len(result.tray_tracks)}",
        f"Unconfirmed trays: {len(result.unconfirmed_trays)}",
        f"Processing FPS: {result.processing_fps:.1f}",
        f"Frame: {result.frame_index}",
        f"Watermelon result age: {result.watermelon_result_age_frames} frames",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (10, 22 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (10, 22 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return canvas
