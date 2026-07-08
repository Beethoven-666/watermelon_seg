#!/usr/bin/env python
"""Evaluate multi-model YOLO segmentation consensus on instance-level PR."""

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
class Instance:
    cls: int
    points: np.ndarray
    confidence: float | None = None
    model_idx: int = -1


@dataclass(frozen=True)
class Candidate:
    instance: Instance
    pred_idx: int
    support: int
    score: float
    members: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--pred-labels-dir", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-thresholds", default="0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--support-iou-thresholds", default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6")
    parser.add_argument("--nms-iou-thresholds", default="0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--min-area-ratios", default="0,0.0005,0.001,0.0015,0.002,0.003")
    parser.add_argument("--min-support-values", default="1,2")
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--recall-target", type=float, default=0.90)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return sorted({float(value.strip()) for value in raw.split(",") if value.strip()})


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(value.strip()) for value in raw.split(",") if value.strip()})


def read_polygons(path: Path, has_confidence: bool, model_idx: int = -1) -> list[Instance]:
    if not path.exists():
        return []
    instances: list[Instance] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coord_parts = parts[1:]
        confidence = None
        if has_confidence:
            confidence = float(coord_parts[-1])
            coord_parts = coord_parts[:-1]
        if len(coord_parts) < 6 or len(coord_parts) % 2:
            continue
        points = np.asarray([float(value) for value in coord_parts], dtype=np.float32).reshape(-1, 2)
        instances.append(
            Instance(cls=cls, points=np.clip(points, 0.0, 1.0), confidence=confidence, model_idx=model_idx)
        )
    return instances


def mask_from_points(points: np.ndarray, raster_size: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_iou(mask_a: np.ndarray, area_a: int, mask_b: np.ndarray, area_b: int) -> float:
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def pairwise_ious(masks_a: list[np.ndarray], areas_a: list[int], masks_b: list[np.ndarray], areas_b: list[int]) -> list[list[float]]:
    return [
        [mask_iou(mask_a, areas_a[row_idx], mask_b, areas_b[col_idx]) for col_idx, mask_b in enumerate(masks_b)]
        for row_idx, mask_a in enumerate(masks_a)
    ]


def build_consensus_candidates(
    instances: list[Instance],
    pred_pred_ious: list[list[float]],
    selected: list[int],
    min_support: int,
    support_iou_threshold: float,
) -> list[Candidate]:
    selected_set = set(selected)
    candidates: list[Candidate] = []
    for seed_idx in selected:
        member_indices = [
            other_idx
            for other_idx in selected_set
            if pred_pred_ious[seed_idx][other_idx] >= support_iou_threshold
        ]
        support_models = {instances[idx].model_idx for idx in member_indices}
        if len(support_models) < min_support:
            continue
        representative_idx = max(member_indices, key=lambda idx: instances[idx].confidence or 0.0)
        score = max(instances[idx].confidence or 0.0 for idx in member_indices)
        candidates.append(
            Candidate(
                instance=instances[representative_idx],
                pred_idx=representative_idx,
                support=len(support_models),
                score=score,
                members=tuple(sorted(member_indices)),
            )
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def nms_candidates(
    candidates: list[Candidate],
    pred_masks: list[np.ndarray],
    pred_areas: list[int],
    nms_iou_threshold: float,
) -> list[int]:
    kept: list[int] = []
    for candidate_idx in range(len(candidates)):
        pred_idx = candidates[candidate_idx].pred_idx
        if any(
            mask_iou(
                pred_masks[pred_idx],
                pred_areas[pred_idx],
                pred_masks[candidates[kept_idx].pred_idx],
                pred_areas[candidates[kept_idx].pred_idx],
            )
            > nms_iou_threshold
            for kept_idx in kept
        ):
            continue
        kept.append(candidate_idx)
    return kept


def evaluate(records: dict[str, dict[str, object]], config: dict[str, float | int]) -> dict[str, float | int]:
    total_tp = total_fp = total_fn = total_selected = total_kept = total_gt = 0
    confidence_threshold = float(config["confidence_threshold"])
    min_area_ratio = float(config["min_area_ratio"])
    min_support = int(config["min_support"])
    support_iou_threshold = float(config["support_iou_threshold"])
    nms_iou_threshold = float(config["nms_iou_threshold"])
    match_iou_threshold = float(config["match_iou_threshold"])

    for record in records.values():
        gt_masks: list[np.ndarray] = record["gt_masks"]  # type: ignore[assignment]
        gt_areas: list[int] = record["gt_areas"]  # type: ignore[assignment]
        instances: list[Instance] = record["preds"]  # type: ignore[assignment]
        pred_masks: list[np.ndarray] = record["pred_masks"]  # type: ignore[assignment]
        pred_areas: list[int] = record["pred_areas"]  # type: ignore[assignment]
        pred_area_ratios: list[float] = record["pred_area_ratios"]  # type: ignore[assignment]
        pred_pred_ious: list[list[float]] = record["pred_pred_ious"]  # type: ignore[assignment]

        selected = [
            idx
            for idx, instance in enumerate(instances)
            if instance.confidence is not None
            and instance.confidence >= confidence_threshold
            and pred_area_ratios[idx] >= min_area_ratio
        ]
        selected.sort(key=lambda idx: instances[idx].confidence or 0.0, reverse=True)
        candidates = build_consensus_candidates(instances, pred_pred_ious, selected, min_support, support_iou_threshold)
        kept_candidate_indices = nms_candidates(candidates, pred_masks, pred_areas, nms_iou_threshold)

        matched_gt: set[int] = set()
        tp = fp = 0
        for candidate_idx in kept_candidate_indices:
            pred_idx = candidates[candidate_idx].pred_idx
            best_gt = -1
            best_iou = 0.0
            for gt_idx, gt_mask in enumerate(gt_masks):
                if gt_idx in matched_gt:
                    continue
                score = mask_iou(pred_masks[pred_idx], pred_areas[pred_idx], gt_mask, gt_areas[gt_idx])
                if score > best_iou:
                    best_iou = score
                    best_gt = gt_idx
            if best_gt >= 0 and best_iou >= match_iou_threshold:
                matched_gt.add(best_gt)
                tp += 1
            else:
                fp += 1

        fn = len(gt_masks) - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt_masks)
        total_selected += len(selected)
        total_kept += len(kept_candidate_indices)

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
        gt = read_polygons(args.labels_dir / f"{image_path.stem}.txt", has_confidence=False)
        gt_masks = [mask_from_points(instance.points, args.raster_size) for instance in gt]
        gt_areas = [int(mask.sum()) for mask in gt_masks]
        preds: list[Instance] = []
        for model_idx, pred_labels_dir in enumerate(args.pred_labels_dir):
            preds.extend(
                instance
                for instance in read_polygons(pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True, model_idx=model_idx)
                if instance.confidence is not None and instance.confidence >= min_confidence
            )
        pred_masks = [mask_from_points(instance.points, args.raster_size) for instance in preds]
        pred_areas = [int(mask.sum()) for mask in pred_masks]
        records[image_path.name] = {
            "gt_masks": gt_masks,
            "gt_areas": gt_areas,
            "preds": preds,
            "pred_masks": pred_masks,
            "pred_areas": pred_areas,
            "pred_area_ratios": [area / float(args.raster_size * args.raster_size) for area in pred_areas],
            "pred_pred_ious": pairwise_ious(pred_masks, pred_areas, pred_masks, pred_areas),
        }

    rows: list[dict[str, float | int]] = []
    for confidence_threshold in confidence_thresholds:
        for min_area_ratio in min_area_ratios:
            for min_support in min_support_values:
                if min_support > len(args.pred_labels_dir):
                    continue
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
                                    "raster_size": args.raster_size,
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
            "pred_labels_dirs": [str(path) for path in args.pred_labels_dir],
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
    (args.output_dir / "ensemble_consensus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_tsv(args.output_dir / "ensemble_consensus_grid.tsv", rows)
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
