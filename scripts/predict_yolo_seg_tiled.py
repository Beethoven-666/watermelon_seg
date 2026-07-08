#!/usr/bin/env python
"""Run tiled YOLO segmentation inference and export YOLO polygon labels.

This is intended for small-object recall experiments. It keeps the original
evaluation target unchanged: predictions are mapped back to full-image
coordinates and can be scored by scripts/analyze_yolo_seg_thresholds.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Prediction:
    points: np.ndarray
    confidence: float
    cls: int
    mask: np.ndarray | None = None
    area: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiled YOLO segmentation predictor.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--raster-size", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument(
        "--discard-tile-edge-ratio",
        type=float,
        default=0.0,
        help=(
            "Discard tile predictions touching an internal tile edge within this ratio of tile size. "
            "Use 0 to keep all tile predictions."
        ),
    )
    return parser.parse_args()


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def prediction_mask(points: np.ndarray, raster_size: int) -> tuple[np.ndarray, int]:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    area = int(mask.sum())
    return mask.astype(bool), area


def mask_iou(a: np.ndarray, area_a: int, b: np.ndarray, area_b: int) -> float:
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection = int(np.logical_and(a, b).sum())
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def extract_predictions(
    model: YOLO,
    image: np.ndarray,
    x_offset: int,
    y_offset: int,
    full_width: int,
    full_height: int,
    args: argparse.Namespace,
) -> list[Prediction]:
    tile_height, tile_width = image.shape[:2]
    result = model.predict(
        image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.masks is None or result.boxes is None:
        return []

    predictions: list[Prediction] = []
    boxes = result.boxes
    polygons = result.masks.xy
    for idx, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.float32)
        if args.discard_tile_edge_ratio > 0:
            margin = args.discard_tile_edge_ratio * min(tile_width, tile_height)
            min_x = float(points[:, 0].min())
            min_y = float(points[:, 1].min())
            max_x = float(points[:, 0].max())
            max_y = float(points[:, 1].max())
            touches_internal_edge = (
                (x_offset > 0 and min_x <= margin)
                or (y_offset > 0 and min_y <= margin)
                or (x_offset + tile_width < full_width and max_x >= tile_width - margin)
                or (y_offset + tile_height < full_height and max_y >= tile_height - margin)
            )
            is_full_image = x_offset == 0 and y_offset == 0 and tile_width == full_width and tile_height == full_height
            if touches_internal_edge and not is_full_image:
                continue
        points[:, 0] = (points[:, 0] + x_offset) / max(full_width - 1, 1)
        points[:, 1] = (points[:, 1] + y_offset) / max(full_height - 1, 1)
        points = np.clip(points, 0.0, 1.0)
        confidence = float(boxes.conf[idx].item())
        cls = int(boxes.cls[idx].item())
        predictions.append(Prediction(points=points, confidence=confidence, cls=cls))
    return predictions


def nms_predictions(predictions: list[Prediction], raster_size: int, nms_iou: float) -> list[Prediction]:
    for prediction in predictions:
        prediction.mask, prediction.area = prediction_mask(prediction.points, raster_size)
    valid = [prediction for prediction in predictions if prediction.area > 0]
    valid.sort(key=lambda prediction: prediction.confidence, reverse=True)

    kept: list[Prediction] = []
    for prediction in valid:
        if any(
            mask_iou(prediction.mask, prediction.area, kept_prediction.mask, kept_prediction.area) > nms_iou
            for kept_prediction in kept
        ):
            continue
        kept.append(prediction)
    return kept


def write_label(path: Path, predictions: list[Prediction]) -> None:
    lines = []
    for prediction in predictions:
        coords = " ".join(f"{value:.8f}" for value in prediction.points.reshape(-1))
        lines.append(f"{prediction.cls} {coords} {prediction.confidence:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    labels_dir = args.output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    image_paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        predictions: list[Prediction] = []

        if args.include_full:
            predictions.extend(extract_predictions(model, image, 0, 0, width, height, args))

        tile_width = min(args.tile_size, width)
        tile_height = min(args.tile_size, height)
        for y in tile_starts(height, tile_height, args.overlap):
            for x in tile_starts(width, tile_width, args.overlap):
                tile = image[y : y + tile_height, x : x + tile_width]
                predictions.extend(extract_predictions(model, tile, x, y, width, height, args))

        kept = nms_predictions(predictions, args.raster_size, args.nms_iou)
        write_label(labels_dir / f"{image_path.stem}.txt", kept)
        print(f"{image_index}/{len(image_paths)} {image_path.name}: raw={len(predictions)} kept={len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
