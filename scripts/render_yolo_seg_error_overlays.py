#!/usr/bin/env python
"""Render YOLO segmentation GT/prediction error overlays for inspection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Instance:
    points: np.ndarray
    confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render segmentation error overlays.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--pred-labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--per-image-tsv", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--images", nargs="*", default=[])
    return parser.parse_args()


def read_polygons(path: Path, has_confidence: bool) -> list[Instance]:
    if not path.exists():
        return []
    instances: list[Instance] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:
            continue
        coord_parts = parts[1:]
        confidence = None
        if has_confidence:
            confidence = float(coord_parts[-1])
            coord_parts = coord_parts[:-1]
        if len(coord_parts) < 6 or len(coord_parts) % 2:
            continue
        points = np.asarray([float(x) for x in coord_parts], dtype=np.float32).reshape(-1, 2)
        points = np.clip(points, 0.0, 1.0)
        instances.append(Instance(points=points, confidence=confidence))
    return instances


def mask_from_points(points: np.ndarray, raster_size: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union else 0.0


def match_instances(
    gt: list[Instance],
    preds: list[Instance],
    raster_size: int,
    iou_threshold: float,
) -> tuple[dict[int, int], set[int]]:
    gt_masks = [mask_from_points(instance.points, raster_size) for instance in gt]
    pred_masks = [mask_from_points(instance.points, raster_size) for instance in preds]
    order = sorted(range(len(preds)), key=lambda idx: preds[idx].confidence or 0.0, reverse=True)
    matched_gt: set[int] = set()
    pred_to_gt: dict[int, int] = {}
    for pred_idx in order:
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt_mask in enumerate(gt_masks):
            if gt_idx in matched_gt:
                continue
            score = iou(pred_masks[pred_idx], gt_mask)
            if score > best_iou:
                best_iou = score
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            pred_to_gt[pred_idx] = best_gt
            matched_gt.add(best_gt)
    return pred_to_gt, matched_gt


def scale_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= width - 1
    scaled[:, 1] *= height - 1
    return np.rint(scaled).astype(np.int32)


def draw_instance(
    overlay: np.ndarray,
    canvas: np.ndarray,
    points: np.ndarray,
    fill_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    width: int,
    height: int,
) -> None:
    polygon = scale_points(points, width, height)
    cv2.fillPoly(overlay, [polygon], fill_color)
    cv2.polylines(canvas, [polygon], isClosed=True, color=line_color, thickness=2, lineType=cv2.LINE_AA)


def choose_images(args: argparse.Namespace) -> list[str]:
    if args.images:
        return args.images
    if not args.per_image_tsv:
        return []
    with args.per_image_tsv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows.sort(key=lambda row: (int(row["fn"]), int(row["fp"]), int(row["gt"])), reverse=True)
    return [row["image"] for row in rows[: args.top]]


def render_image(args: argparse.Namespace, image_name: str) -> Path:
    image_path = args.images_dir / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    gt = read_polygons(args.labels_dir / f"{image_path.stem}.txt", has_confidence=False)
    preds_all = read_polygons(args.pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True)
    preds = [p for p in preds_all if p.confidence is not None and p.confidence >= args.threshold]
    pred_to_gt, matched_gt = match_instances(gt, preds, args.raster_size, args.iou_threshold)
    matched_pred = set(pred_to_gt)

    overlay = image.copy()
    canvas = image.copy()
    for gt_idx, instance in enumerate(gt):
        if gt_idx in matched_gt:
            draw_instance(overlay, canvas, instance.points, (0, 180, 0), (0, 255, 0), width, height)
        else:
            draw_instance(overlay, canvas, instance.points, (0, 0, 255), (0, 0, 255), width, height)
    for pred_idx, instance in enumerate(preds):
        if pred_idx not in matched_pred:
            draw_instance(overlay, canvas, instance.points, (180, 0, 180), (255, 0, 255), width, height)

    blended = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    tp = len(matched_gt)
    fp = len(preds) - len(matched_pred)
    fn = len(gt) - len(matched_gt)
    label = f"conf>={args.threshold:g}  GT={len(gt)} TP={tp} FP={fp} FN={fn}  green=TP red=FN purple=FP"
    cv2.rectangle(blended, (0, 0), (min(width, 1120), 34), (0, 0, 0), thickness=-1)
    cv2.putText(blended, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{image_path.stem}_errors.png"
    cv2.imwrite(str(output_path), blended)
    return output_path


def main() -> int:
    args = parse_args()
    image_names = choose_images(args)
    if not image_names:
        raise SystemExit("No images selected. Pass --images or --per-image-tsv.")
    for image_name in image_names:
        output_path = render_image(args, image_name)
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
