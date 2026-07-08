#!/usr/bin/env python
"""Export final YOLO segmentation labels for a two-model deployment policy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pred:
    model_name: str
    confidence: float
    points: np.ndarray
    mask: np.ndarray
    area: int
    area_ratio: float
    bbox: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--pred-a-dir", type=Path, required=True)
    parser.add_argument("--pred-b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--min-area-ratio", type=float, required=True)
    parser.add_argument("--min-support", type=int, choices=(1, 2), default=1)
    parser.add_argument("--support-iou-threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--raster-size", type=int, default=768)
    return parser.parse_args()


def read_points(path: Path) -> list[tuple[np.ndarray, float]]:
    if not path.exists():
        return []
    rows: list[tuple[np.ndarray, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 8:
            continue
        coord_parts = parts[1:-1]
        if len(coord_parts) < 6 or len(coord_parts) % 2:
            continue
        points = np.asarray([float(value) for value in coord_parts], dtype=np.float32).reshape(-1, 2)
        rows.append((np.clip(points, 0.0, 1.0), float(parts[-1])))
    return rows


def bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def bboxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def mask_from_points(points: np.ndarray, raster_size: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_iou(a: Pred, b: Pred) -> float:
    if a.area <= 0 or b.area <= 0 or not bboxes_overlap(a.bbox, b.bbox):
        return 0.0
    intersection = int(np.logical_and(a.mask, b.mask).sum())
    union = a.area + b.area - intersection
    return float(intersection / union) if union else 0.0


def make_preds(path: Path, model_name: str, raster_size: int) -> list[Pred]:
    preds: list[Pred] = []
    for points, confidence in read_points(path):
        mask = mask_from_points(points, raster_size)
        area = int(mask.sum())
        preds.append(
            Pred(
                model_name=model_name,
                confidence=confidence,
                points=points,
                mask=mask,
                area=area,
                area_ratio=area / float(raster_size * raster_size),
                bbox=bbox(points),
            )
        )
    return preds


def supported_by_other_model(pred: Pred, others: list[Pred], support_iou_threshold: float) -> bool:
    return any(mask_iou(pred, other) >= support_iou_threshold for other in others)


def nms(preds: list[Pred], nms_iou_threshold: float) -> list[Pred]:
    kept: list[Pred] = []
    for pred in sorted(preds, key=lambda item: item.confidence, reverse=True):
        if any(mask_iou(pred, kept_pred) > nms_iou_threshold for kept_pred in kept):
            continue
        kept.append(pred)
    return kept


def format_label(pred: Pred) -> str:
    coords = " ".join(f"{value:.6f}" for point in pred.points for value in point)
    return f"0 {coords} {pred.confidence:.6f}"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    labels_dir = args.output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    total_selected_a = total_selected_b = total_candidates = total_kept = 0
    kept_by_model: dict[str, int] = {args.model_a_name: 0, args.model_b_name: 0}

    image_paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        pred_a = make_preds(args.pred_a_dir / f"{image_path.stem}.txt", args.model_a_name, args.raster_size)
        pred_b = make_preds(args.pred_b_dir / f"{image_path.stem}.txt", args.model_b_name, args.raster_size)
        selected_a = [
            pred
            for pred in pred_a
            if pred.confidence >= args.confidence_threshold and pred.area_ratio >= args.min_area_ratio
        ]
        selected_b = [
            pred
            for pred in pred_b
            if pred.confidence >= args.confidence_threshold and pred.area_ratio >= args.min_area_ratio
        ]
        if args.min_support == 2:
            candidates = [
                pred for pred in selected_a if supported_by_other_model(pred, selected_b, args.support_iou_threshold)
            ] + [
                pred for pred in selected_b if supported_by_other_model(pred, selected_a, args.support_iou_threshold)
            ]
        else:
            candidates = selected_a + selected_b
        kept = nms(candidates, args.nms_iou_threshold)

        label_text = "\n".join(format_label(pred) for pred in kept)
        if label_text:
            label_text += "\n"
        (labels_dir / f"{image_path.stem}.txt").write_text(label_text, encoding="utf-8")

        kept_a = sum(1 for pred in kept if pred.model_name == args.model_a_name)
        kept_b = sum(1 for pred in kept if pred.model_name == args.model_b_name)
        kept_by_model[args.model_a_name] += kept_a
        kept_by_model[args.model_b_name] += kept_b
        total_selected_a += len(selected_a)
        total_selected_b += len(selected_b)
        total_candidates += len(candidates)
        total_kept += len(kept)
        manifest_rows.append(
            {
                "image": image_path.name,
                "selected_a": len(selected_a),
                "selected_b": len(selected_b),
                "candidates": len(candidates),
                "kept": len(kept),
                f"kept_{args.model_a_name}": kept_a,
                f"kept_{args.model_b_name}": kept_b,
            }
        )

    write_tsv(args.output_dir / "export_manifest.tsv", manifest_rows)
    summary = {
        "images_dir": args.images_dir.as_posix(),
        "pred_a_dir": args.pred_a_dir.as_posix(),
        "pred_b_dir": args.pred_b_dir.as_posix(),
        "labels_dir": labels_dir.as_posix(),
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "confidence_threshold": args.confidence_threshold,
        "min_area_ratio": args.min_area_ratio,
        "min_support": args.min_support,
        "support_iou_threshold": args.support_iou_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "raster_size": args.raster_size,
        "images": len(image_paths),
        "selected_a": total_selected_a,
        "selected_b": total_selected_b,
        "candidates": total_candidates,
        "kept": total_kept,
        "kept_by_model": kept_by_model,
        "files": {
            "labels": labels_dir.as_posix(),
            "manifest": (args.output_dir / "export_manifest.tsv").as_posix(),
            "summary": (args.output_dir / "export_summary.json").as_posix(),
        },
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
