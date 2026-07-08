#!/usr/bin/env python
"""Fast two-model YOLO segmentation consensus evaluation."""

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
    model_idx: int
    confidence: float
    points: np.ndarray
    mask: np.ndarray
    area: int
    area_ratio: float
    bbox: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--pred-a-dir", required=True, type=Path)
    parser.add_argument("--pred-b-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-thresholds", default="0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--support-iou-thresholds", default="0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--nms-iou-thresholds", default="0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--min-area-ratios", default="0,0.0005,0.001")
    parser.add_argument("--min-support-values", default="1,2")
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--recall-target", type=float, default=0.90)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return sorted({float(value.strip()) for value in raw.split(",") if value.strip()})


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(value.strip()) for value in raw.split(",") if value.strip()})


def read_points(path: Path, has_confidence: bool) -> list[tuple[np.ndarray, float]]:
    if not path.exists():
        return []
    rows: list[tuple[np.ndarray, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:
            continue
        coord_parts = parts[1:]
        confidence = 1.0
        if has_confidence:
            confidence = float(coord_parts[-1])
            coord_parts = coord_parts[:-1]
        if len(coord_parts) < 6 or len(coord_parts) % 2:
            continue
        points = np.asarray([float(value) for value in coord_parts], dtype=np.float32).reshape(-1, 2)
        rows.append((np.clip(points, 0.0, 1.0), confidence))
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


def gt_iou(pred: Pred, gt_mask: np.ndarray, gt_area: int, gt_bbox: tuple[float, float, float, float]) -> float:
    if pred.area <= 0 or gt_area <= 0 or not bboxes_overlap(pred.bbox, gt_bbox):
        return 0.0
    intersection = int(np.logical_and(pred.mask, gt_mask).sum())
    union = pred.area + gt_area - intersection
    return float(intersection / union) if union else 0.0


def make_preds(path: Path, model_idx: int, raster_size: int, min_confidence: float) -> list[Pred]:
    preds: list[Pred] = []
    for points, confidence in read_points(path, has_confidence=True):
        if confidence < min_confidence:
            continue
        mask = mask_from_points(points, raster_size)
        area = int(mask.sum())
        preds.append(
            Pred(
                model_idx=model_idx,
                confidence=confidence,
                points=points,
                mask=mask,
                area=area,
                area_ratio=area / float(raster_size * raster_size),
                bbox=bbox(points),
            )
        )
    return preds


def nms(preds: list[Pred], nms_iou_threshold: float) -> list[Pred]:
    ordered = sorted(preds, key=lambda pred: pred.confidence, reverse=True)
    kept: list[Pred] = []
    for pred in ordered:
        if any(mask_iou(pred, kept_pred) > nms_iou_threshold for kept_pred in kept):
            continue
        kept.append(pred)
    return kept


def supported_by_other_model(pred: Pred, others: list[Pred], support_iou_threshold: float) -> bool:
    return any(mask_iou(pred, other) >= support_iou_threshold for other in others)


def evaluate(records: dict[str, dict[str, object]], config: dict[str, float | int]) -> dict[str, float | int]:
    confidence_threshold = float(config["confidence_threshold"])
    min_area_ratio = float(config["min_area_ratio"])
    min_support = int(config["min_support"])
    support_iou_threshold = float(config["support_iou_threshold"])
    nms_iou_threshold = float(config["nms_iou_threshold"])
    match_iou_threshold = float(config["match_iou_threshold"])
    total_tp = total_fp = total_fn = total_kept = total_selected = total_gt = 0

    for record in records.values():
        pred_a: list[Pred] = record["pred_a"]  # type: ignore[assignment]
        pred_b: list[Pred] = record["pred_b"]  # type: ignore[assignment]
        gt_masks: list[np.ndarray] = record["gt_masks"]  # type: ignore[assignment]
        gt_areas: list[int] = record["gt_areas"]  # type: ignore[assignment]
        gt_bboxes: list[tuple[float, float, float, float]] = record["gt_bboxes"]  # type: ignore[assignment]

        selected_a = [pred for pred in pred_a if pred.confidence >= confidence_threshold and pred.area_ratio >= min_area_ratio]
        selected_b = [pred for pred in pred_b if pred.confidence >= confidence_threshold and pred.area_ratio >= min_area_ratio]
        total_selected += len(selected_a) + len(selected_b)
        if min_support == 2:
            candidates = [
                pred for pred in selected_a if supported_by_other_model(pred, selected_b, support_iou_threshold)
            ] + [
                pred for pred in selected_b if supported_by_other_model(pred, selected_a, support_iou_threshold)
            ]
        else:
            candidates = selected_a + selected_b
        kept = nms(candidates, nms_iou_threshold)

        matched_gt: set[int] = set()
        tp = fp = 0
        for pred in kept:
            best_gt = -1
            best_iou = 0.0
            for gt_idx, gt_mask in enumerate(gt_masks):
                if gt_idx in matched_gt:
                    continue
                score = gt_iou(pred, gt_mask, gt_areas[gt_idx], gt_bboxes[gt_idx])
                if score > best_iou:
                    best_gt = gt_idx
                    best_iou = score
            if best_gt >= 0 and best_iou >= match_iou_threshold:
                matched_gt.add(best_gt)
                tp += 1
            else:
                fp += 1
        fn = len(gt_masks) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_kept += len(kept)
        total_gt += len(gt_masks)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confidence_threshold": confidence_threshold,
        "min_area_ratio": min_area_ratio,
        "min_support": min_support,
        "support_iou_threshold": support_iou_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "gt": total_gt,
        "selected_predictions": total_selected,
        "kept_predictions": total_kept,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confidence_thresholds = parse_float_list(args.confidence_thresholds)
    support_iou_thresholds = parse_float_list(args.support_iou_thresholds)
    nms_iou_thresholds = parse_float_list(args.nms_iou_thresholds)
    min_area_ratios = parse_float_list(args.min_area_ratios)
    min_support_values = parse_int_list(args.min_support_values)
    min_confidence = min(confidence_thresholds)

    records: dict[str, dict[str, object]] = {}
    for image_path in sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        gt_rows = read_points(args.labels_dir / f"{image_path.stem}.txt", has_confidence=False)
        gt_masks = [mask_from_points(points, args.raster_size) for points, _confidence in gt_rows]
        records[image_path.name] = {
            "gt_masks": gt_masks,
            "gt_areas": [int(mask.sum()) for mask in gt_masks],
            "gt_bboxes": [bbox(points) for points, _confidence in gt_rows],
            "pred_a": make_preds(args.pred_a_dir / f"{image_path.stem}.txt", 0, args.raster_size, min_confidence),
            "pred_b": make_preds(args.pred_b_dir / f"{image_path.stem}.txt", 1, args.raster_size, min_confidence),
        }

    rows: list[dict[str, float | int]] = []
    for confidence_threshold in confidence_thresholds:
        for min_area_ratio in min_area_ratios:
            for min_support in min_support_values:
                for support_iou_threshold in support_iou_thresholds:
                    for nms_iou_threshold in nms_iou_thresholds:
                        rows.append(
                            evaluate(
                                records,
                                {
                                    "confidence_threshold": confidence_threshold,
                                    "min_area_ratio": min_area_ratio,
                                    "min_support": min_support,
                                    "support_iou_threshold": support_iou_threshold,
                                    "nms_iou_threshold": nms_iou_threshold,
                                    "match_iou_threshold": args.match_iou_threshold,
                                },
                            )
                        )

    best_f1 = max(rows, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])))
    target_rows = [
        row
        for row in rows
        if float(row["precision"]) >= args.precision_target and float(row["recall"]) >= args.recall_target
    ]
    precision_floor_rows = [row for row in rows if float(row["precision"]) >= args.precision_target]
    recall_floor_rows = [row for row in rows if float(row["recall"]) >= args.recall_target]
    summary = {
        "inputs": {
            "images_dir": str(args.images_dir),
            "labels_dir": str(args.labels_dir),
            "pred_a_dir": str(args.pred_a_dir),
            "pred_b_dir": str(args.pred_b_dir),
            "raster_size": args.raster_size,
            "match_iou_threshold": args.match_iou_threshold,
            "confidence_thresholds": confidence_thresholds,
            "support_iou_thresholds": support_iou_thresholds,
            "nms_iou_thresholds": nms_iou_thresholds,
            "min_area_ratios": min_area_ratios,
            "min_support_values": min_support_values,
        },
        "targets": {"precision": args.precision_target, "recall": args.recall_target},
        "best_f1": best_f1,
        "best_target": max(target_rows, key=lambda row: (float(row["f1"]), float(row["recall"]))) if target_rows else None,
        "best_recall_with_precision_target": (
            max(precision_floor_rows, key=lambda row: (float(row["recall"]), float(row["f1"])))
            if precision_floor_rows
            else None
        ),
        "best_precision_with_recall_target": (
            max(recall_floor_rows, key=lambda row: (float(row["precision"]), float(row["f1"])))
            if recall_floor_rows
            else None
        ),
        "meets_targets": bool(target_rows),
        "rows": rows,
    }
    (args.output_dir / "two_model_consensus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_tsv(args.output_dir / "two_model_consensus_grid.tsv", rows)
    print(
        json.dumps(
            {
                "best_f1": summary["best_f1"],
                "best_target": summary["best_target"],
                "best_recall_with_precision_target": summary["best_recall_with_precision_target"],
                "best_precision_with_recall_target": summary["best_precision_with_recall_target"],
                "meets_targets": summary["meets_targets"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
