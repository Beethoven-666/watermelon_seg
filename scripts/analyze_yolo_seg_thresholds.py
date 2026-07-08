#!/usr/bin/env python
"""Analyze YOLO segmentation prediction thresholds against YOLO polygon labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Instance:
    image: str
    cls: int
    points: np.ndarray
    confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute instance-level precision/recall across confidence thresholds."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--pred-labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--recall-target", type=float, default=0.90)
    parser.add_argument("--min-gt-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-gt-area-ratio", type=float, default=1.0)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.0)
    parser.add_argument(
        "--ignore-filtered-gt-overlap",
        action="store_true",
        help="Do not count predictions matching filtered-out GT as false positives.",
    )
    parser.add_argument(
        "--min-area-ratios",
        default="0",
        help="Comma-separated minimum predicted mask area ratios for post-processing grid search.",
    )
    parser.add_argument(
        "--max-area-ratios",
        default="1",
        help="Comma-separated maximum predicted mask area ratios for post-processing grid search.",
    )
    parser.add_argument(
        "--thresholds",
        default=(
            "0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,"
            "0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,"
            "0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95"
        ),
        help="Comma-separated confidence thresholds.",
    )
    return parser.parse_args()


def read_yolo_polygons(
    path: Path,
    image_name: str,
    has_confidence: bool,
    skipped_invalid_predictions: list[dict[str, object]] | None = None,
) -> list[Instance]:
    if not path.exists():
        return []

    instances: list[Instance] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            if has_confidence and skipped_invalid_predictions is not None:
                skipped_invalid_predictions.append(
                    {"path": str(path), "line": line_no, "reason": "fewer_than_3_points"}
                )
                continue
            raise ValueError(f"{path}:{line_no} has too few values for a segmentation polygon")

        cls = int(float(parts[0]))
        confidence = None
        coord_parts = parts[1:]
        if has_confidence:
            confidence = float(coord_parts[-1])
            coord_parts = coord_parts[:-1]
        if len(coord_parts) % 2 != 0 or len(coord_parts) < 6:
            if has_confidence and skipped_invalid_predictions is not None:
                skipped_invalid_predictions.append(
                    {"path": str(path), "line": line_no, "reason": "invalid_coordinate_count"}
                )
                continue
            raise ValueError(f"{path}:{line_no} has an invalid coordinate count")

        coords = np.asarray([float(x) for x in coord_parts], dtype=np.float32).reshape(-1, 2)
        coords = np.clip(coords, 0.0, 1.0)
        instances.append(Instance(image=image_name, cls=cls, points=coords, confidence=confidence))
    return instances


def polygon_mask(points: np.ndarray, raster_size: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_iou(mask_a: np.ndarray, area_a: int, mask_b: np.ndarray, area_b: int) -> float:
    if area_a == 0 or area_b == 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def build_image_records(
    images_dir: Path,
    labels_dir: Path,
    pred_labels_dir: Path,
    raster_size: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    records: dict[str, dict[str, object]] = {}
    skipped_invalid_predictions: list[dict[str, object]] = []

    for image_path in image_paths:
        stem = image_path.stem
        gt = read_yolo_polygons(labels_dir / f"{stem}.txt", image_path.name, has_confidence=False)
        preds = read_yolo_polygons(
            pred_labels_dir / f"{stem}.txt",
            image_path.name,
            has_confidence=True,
            skipped_invalid_predictions=skipped_invalid_predictions,
        )

        gt_masks = [polygon_mask(instance.points, raster_size) for instance in gt]
        pred_masks = [polygon_mask(instance.points, raster_size) for instance in preds]
        gt_areas = [int(mask.sum()) for mask in gt_masks]
        pred_areas = [int(mask.sum()) for mask in pred_masks]
        valid_preds: list[Instance] = []
        valid_pred_masks: list[np.ndarray] = []
        valid_pred_areas: list[int] = []
        for pred, pred_mask, pred_area in zip(preds, pred_masks, pred_areas):
            if pred_area <= 0:
                skipped_invalid_predictions.append(
                    {"path": str(pred_labels_dir / f"{stem}.txt"), "line": None, "reason": "zero_area_mask"}
                )
                continue
            valid_preds.append(pred)
            valid_pred_masks.append(pred_mask)
            valid_pred_areas.append(pred_area)
        preds = valid_preds
        pred_masks = valid_pred_masks
        pred_areas = valid_pred_areas

        ious: list[list[float]] = []
        for pred_idx, pred_mask in enumerate(pred_masks):
            row = [
                mask_iou(pred_mask, pred_areas[pred_idx], gt_mask, gt_areas[gt_idx])
                for gt_idx, gt_mask in enumerate(gt_masks)
            ]
            ious.append(row)

        records[image_path.name] = {
            "gt": gt,
            "preds": preds,
            "gt_area_ratios": [area / float(raster_size * raster_size) for area in gt_areas],
            "gt_bboxes": [
                (
                    float(instance.points[:, 0].min()),
                    float(instance.points[:, 1].min()),
                    float(instance.points[:, 0].max()),
                    float(instance.points[:, 1].max()),
                )
                for instance in gt
            ],
            "pred_area_ratios": [area / float(raster_size * raster_size) for area in pred_areas],
            "ious": ious,
        }
    return records, skipped_invalid_predictions


def evaluate_threshold(
    records: dict[str, dict[str, object]],
    confidence_threshold: float,
    iou_threshold: float,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 1.0,
    min_gt_area_ratio: float = 0.0,
    max_gt_area_ratio: float = 1.0,
    edge_margin_ratio: float = 0.0,
    ignore_filtered_gt_overlap: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    total_tp = total_fp = total_fn = 0
    per_image: list[dict[str, float | int | str]] = []

    for image_name, record in records.items():
        gt: list[Instance] = record["gt"]  # type: ignore[assignment]
        preds: list[Instance] = record["preds"]  # type: ignore[assignment]
        gt_area_ratios: list[float] = record["gt_area_ratios"]  # type: ignore[assignment]
        gt_bboxes: list[tuple[float, float, float, float]] = record["gt_bboxes"]  # type: ignore[assignment]
        pred_area_ratios: list[float] = record["pred_area_ratios"]  # type: ignore[assignment]
        ious: list[list[float]] = record["ious"]  # type: ignore[assignment]
        target_gt_indices: list[int] = []
        ignored_gt_indices: set[int] = set()
        for gt_idx in range(len(gt)):
            min_x, min_y, max_x, max_y = gt_bboxes[gt_idx]
            touches_edge = (
                min_x <= edge_margin_ratio
                or min_y <= edge_margin_ratio
                or max_x >= 1.0 - edge_margin_ratio
                or max_y >= 1.0 - edge_margin_ratio
            )
            is_target = (
                min_gt_area_ratio <= gt_area_ratios[gt_idx] <= max_gt_area_ratio
                and not touches_edge
            )
            if is_target:
                target_gt_indices.append(gt_idx)
            else:
                ignored_gt_indices.add(gt_idx)
        selected = [
            idx
            for idx, pred in enumerate(preds)
            if pred.confidence is not None
            and pred.confidence >= confidence_threshold
            and min_area_ratio <= pred_area_ratios[idx] <= max_area_ratio
        ]
        selected.sort(key=lambda idx: preds[idx].confidence or 0.0, reverse=True)

        matched_gt: set[int] = set()
        tp = fp = 0
        for pred_idx in selected:
            best_gt = -1
            best_iou = 0.0
            for gt_idx in target_gt_indices:
                if gt_idx in matched_gt:
                    continue
                iou = ious[pred_idx][gt_idx] if gt_idx < len(ious[pred_idx]) else 0.0
                if iou > best_iou:
                    best_gt = gt_idx
                    best_iou = iou
            if best_gt >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_gt)
                tp += 1
            elif ignore_filtered_gt_overlap and any(
                ious[pred_idx][gt_idx] >= iou_threshold for gt_idx in ignored_gt_indices
            ):
                continue
            else:
                fp += 1

        fn = len(target_gt_indices) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image.append(
            {
                "image": image_name,
                "gt": len(target_gt_indices),
                "ignored_gt": len(ignored_gt_indices),
                "pred": len(selected),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / len(target_gt_indices) if target_gt_indices else 0.0,
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "threshold": confidence_threshold,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "min_gt_area_ratio": min_gt_area_ratio,
        "max_gt_area_ratio": max_gt_area_ratio,
        "edge_margin_ratio": edge_margin_ratio,
        "ignore_filtered_gt_overlap": int(ignore_filtered_gt_overlap),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return summary, per_image


def best_available_overlap(records: dict[str, dict[str, object]], iou_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_name, record in records.items():
        gt: list[Instance] = record["gt"]  # type: ignore[assignment]
        preds: list[Instance] = record["preds"]  # type: ignore[assignment]
        ious: list[list[float]] = record["ious"]  # type: ignore[assignment]
        for gt_idx in range(len(gt)):
            best_iou = 0.0
            best_conf = 0.0
            for pred_idx, pred in enumerate(preds):
                iou = ious[pred_idx][gt_idx] if gt_idx < len(ious[pred_idx]) else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_conf = pred.confidence or 0.0
            rows.append(
                {
                    "image": image_name,
                    "gt_index": gt_idx,
                    "best_iou": best_iou,
                    "best_confidence": best_conf,
                    "reachable_at_iou_threshold": best_iou >= iou_threshold,
                }
            )
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    lines = ["\t".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    thresholds = sorted({float(x.strip()) for x in args.thresholds.split(",") if x.strip()})
    min_area_ratios = sorted({float(x.strip()) for x in args.min_area_ratios.split(",") if x.strip()})
    max_area_ratios = sorted({float(x.strip()) for x in args.max_area_ratios.split(",") if x.strip()})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, skipped_invalid_predictions = build_image_records(
        args.images_dir, args.labels_dir, args.pred_labels_dir, args.raster_size
    )
    summaries: list[dict[str, float | int]] = []
    per_image_by_threshold: dict[str, list[dict[str, float | int | str]]] = {}
    for threshold in thresholds:
        summary, per_image = evaluate_threshold(
            records,
            threshold,
            args.iou_threshold,
            min_gt_area_ratio=args.min_gt_area_ratio,
            max_gt_area_ratio=args.max_gt_area_ratio,
            edge_margin_ratio=args.edge_margin_ratio,
            ignore_filtered_gt_overlap=args.ignore_filtered_gt_overlap,
        )
        summaries.append(summary)
        per_image_by_threshold[f"{threshold:.6f}"] = per_image

    grid_summaries: list[dict[str, float | int]] = []
    for min_area_ratio in min_area_ratios:
        for max_area_ratio in max_area_ratios:
            if min_area_ratio > max_area_ratio:
                continue
            for threshold in thresholds:
                summary, _ = evaluate_threshold(
                    records,
                    threshold,
                    args.iou_threshold,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    min_gt_area_ratio=args.min_gt_area_ratio,
                    max_gt_area_ratio=args.max_gt_area_ratio,
                    edge_margin_ratio=args.edge_margin_ratio,
                    ignore_filtered_gt_overlap=args.ignore_filtered_gt_overlap,
                )
                grid_summaries.append(summary)

    best_f1 = max(summaries, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])))
    precision_floor = [
        row for row in summaries if float(row["precision"]) >= args.precision_target
    ]
    best_recall_with_precision_floor = (
        max(precision_floor, key=lambda row: (float(row["recall"]), float(row["f1"]))) if precision_floor else None
    )
    recall_floor = [row for row in summaries if float(row["recall"]) >= args.recall_target]
    best_precision_with_recall_floor = (
        max(recall_floor, key=lambda row: (float(row["precision"]), float(row["f1"]))) if recall_floor else None
    )
    target_rows = [
        row
        for row in summaries
        if float(row["precision"]) >= args.precision_target and float(row["recall"]) >= args.recall_target
    ]
    best_target = max(target_rows, key=lambda row: (float(row["f1"]), float(row["recall"]))) if target_rows else None
    grid_target_rows = [
        row
        for row in grid_summaries
        if float(row["precision"]) >= args.precision_target and float(row["recall"]) >= args.recall_target
    ]
    best_grid_f1 = max(
        grid_summaries,
        key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])),
    )
    best_grid_target = (
        max(grid_target_rows, key=lambda row: (float(row["f1"]), float(row["recall"])))
        if grid_target_rows
        else None
    )
    grid_precision_floor = [
        row for row in grid_summaries if float(row["precision"]) >= args.precision_target
    ]
    best_grid_recall_with_precision_floor = (
        max(grid_precision_floor, key=lambda row: (float(row["recall"]), float(row["f1"])))
        if grid_precision_floor
        else None
    )

    overlap_rows = best_available_overlap(records, args.iou_threshold)
    unreachable = [row for row in overlap_rows if not row["reachable_at_iou_threshold"]]
    gt_total = sum(len(record["gt"]) for record in records.values())  # type: ignore[arg-type]
    pred_total = sum(len(record["preds"]) for record in records.values())  # type: ignore[arg-type]

    analysis = {
        "inputs": {
            "images_dir": str(args.images_dir),
            "labels_dir": str(args.labels_dir),
            "pred_labels_dir": str(args.pred_labels_dir),
            "raster_size": args.raster_size,
            "iou_threshold": args.iou_threshold,
            "min_gt_area_ratio": args.min_gt_area_ratio,
            "max_gt_area_ratio": args.max_gt_area_ratio,
            "edge_margin_ratio": args.edge_margin_ratio,
            "ignore_filtered_gt_overlap": args.ignore_filtered_gt_overlap,
        },
        "targets": {
            "precision": args.precision_target,
            "recall": args.recall_target,
        },
        "dataset": {
            "images": len(records),
            "ground_truth_instances": gt_total,
            "predicted_instances_at_conf_0_001_export": pred_total,
            "skipped_invalid_predictions": len(skipped_invalid_predictions),
        },
        "thresholds": summaries,
        "postprocess_grid": {
            "min_area_ratios": min_area_ratios,
            "max_area_ratios": max_area_ratios,
            "rows": grid_summaries,
        },
        "best_f1": best_f1,
        "best_grid_f1": best_grid_f1,
        "best_recall_with_precision_target": best_recall_with_precision_floor,
        "best_grid_recall_with_precision_target": best_grid_recall_with_precision_floor,
        "best_precision_with_recall_target": best_precision_with_recall_floor,
        "best_threshold_meeting_targets": best_target,
        "best_grid_threshold_meeting_targets": best_grid_target,
        "meets_targets_by_threshold_tuning": best_target is not None,
        "meets_targets_by_confidence_area_grid": best_grid_target is not None,
        "max_reachable_recall_if_all_predictions_kept": (gt_total - len(unreachable)) / gt_total if gt_total else 0.0,
        "ground_truth_instances_with_no_prediction_iou_match": len(unreachable),
        "skipped_invalid_prediction_examples": skipped_invalid_predictions[:20],
    }

    summary_path = args.output_dir / "threshold_pr_summary.json"
    tsv_path = args.output_dir / "threshold_pr_summary.tsv"
    overlap_path = args.output_dir / "gt_best_prediction_overlap.tsv"
    per_image_best_path = args.output_dir / "per_image_best_f1_threshold.tsv"
    grid_path = args.output_dir / "threshold_area_pr_grid.tsv"

    summary_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tsv(tsv_path, summaries)
    write_tsv(grid_path, grid_summaries)
    write_tsv(overlap_path, overlap_rows)
    best_key = f"{float(best_f1['threshold']):.6f}"
    best_per_image = sorted(
        per_image_by_threshold[best_key],
        key=lambda row: (int(row["fn"]), int(row["fp"]), int(row["gt"])),
        reverse=True,
    )
    write_tsv(per_image_best_path, best_per_image)

    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
