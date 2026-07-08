#!/usr/bin/env python
"""Train a lightweight second-stage filter for YOLO segmentation candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_NAMES = ["local_test_images", "my_labelme", "roboflow_drago_v1", "roboflow_team128_v5", "unknown"]


@dataclass(frozen=True)
class Instance:
    points: np.ndarray
    confidence: float | None = None


@dataclass
class SplitRecord:
    image: str
    source: str
    gt_count: int
    features: np.ndarray
    confidences: np.ndarray
    area_ratios: np.ndarray
    pred_gt_ious: list[list[float]]
    pred_pred_ious: list[list[float]]
    labels: np.ndarray


class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images-dir", type=Path, required=True)
    parser.add_argument("--train-labels-dir", type=Path, required=True)
    parser.add_argument("--train-pred-labels-dir", type=Path, required=True)
    parser.add_argument("--val-images-dir", type=Path, required=True)
    parser.add_argument("--val-labels-dir", type=Path, required=True)
    parser.add_argument("--val-pred-labels-dir", type=Path, required=True)
    parser.add_argument("--test-images-dir", type=Path, required=True)
    parser.add_argument("--test-labels-dir", type=Path, required=True)
    parser.add_argument("--test-pred-labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--candidate-min-confidence", type=float, default=0.03)
    parser.add_argument("--max-candidates-per-image", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument(
        "--prob-thresholds",
        default="0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--confidence-thresholds", default="0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.05,0.075,0.1")
    parser.add_argument("--nms-iou-thresholds", default="0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7")
    parser.add_argument("--min-area-ratios", default="0,0.0005,0.001,0.002,0.003,0.005")
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--recall-target", type=float, default=0.90)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return sorted({float(value.strip()) for value in raw.split(",") if value.strip()})


def source_hint(image_name: str) -> str:
    for source in SOURCE_NAMES[:-1]:
        if image_name.startswith(source):
            return source
    return "unknown"


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
        points = np.asarray([float(value) for value in coord_parts], dtype=np.float32).reshape(-1, 2)
        instances.append(Instance(points=np.clip(points, 0.0, 1.0), confidence=confidence))
    return instances


def mask_from_points(points: np.ndarray, raster_size: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def actual_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= width - 1
    scaled[:, 1] *= height - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_iou(mask_a: np.ndarray, area_a: int, mask_b: np.ndarray, area_b: int) -> float:
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def pairwise_iou_matrix(
    masks_a: list[np.ndarray],
    areas_a: list[int],
    masks_b: list[np.ndarray],
    areas_b: list[int],
) -> list[list[float]]:
    return [
        [mask_iou(mask_a, areas_a[row_idx], mask_b, areas_b[col_idx]) for col_idx, mask_b in enumerate(masks_b)]
        for row_idx, mask_a in enumerate(masks_a)
    ]


def polygon_features(instance: Instance, area_ratio: float, pred_pred_ious: list[list[float]], idx: int, confidences: list[float]) -> list[float]:
    points = instance.points
    min_x = float(points[:, 0].min())
    min_y = float(points[:, 1].min())
    max_x = float(points[:, 0].max())
    max_y = float(points[:, 1].max())
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    bbox_area = width * height
    aspect = width / height
    extent = area_ratio / max(bbox_area, 1e-6)
    centroid_x = float(points[:, 0].mean())
    centroid_y = float(points[:, 1].mean())
    edge_distance = min(min_x, min_y, 1.0 - max_x, 1.0 - max_y)

    closed = np.vstack([points, points[0]])
    perimeter = float(np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1)).sum())
    circularity = 4.0 * math.pi * area_ratio / max(perimeter * perimeter, 1e-9)

    conf = float(instance.confidence or 0.0)
    higher = [j for j, other_conf in enumerate(confidences) if other_conf > conf and j != idx]
    all_overlaps = [pred_pred_ious[idx][j] for j in range(len(confidences)) if j != idx]
    higher_overlaps = [pred_pred_ious[idx][j] for j in higher]
    max_overlap = max(all_overlaps) if all_overlaps else 0.0
    max_higher_overlap = max(higher_overlaps) if higher_overlaps else 0.0
    count_overlap_01 = sum(1 for value in all_overlaps if value >= 0.1)
    count_overlap_03 = sum(1 for value in all_overlaps if value >= 0.3)
    count_higher_overlap_01 = sum(1 for value in higher_overlaps if value >= 0.1)
    count_higher_overlap_03 = sum(1 for value in higher_overlaps if value >= 0.3)

    return [
        conf,
        math.log(max(conf, 1e-6)),
        area_ratio,
        math.log(max(area_ratio, 1e-8)),
        width,
        height,
        bbox_area,
        aspect,
        extent,
        centroid_x,
        centroid_y,
        edge_distance,
        perimeter,
        circularity,
        float(len(points)),
        max_overlap,
        max_higher_overlap,
        float(count_overlap_01),
        float(count_overlap_03),
        float(count_higher_overlap_01),
        float(count_higher_overlap_03),
        float(len(confidences)),
        float(sum(1 for value in confidences if value >= conf)),
    ]


def color_features(image_bgr: np.ndarray, instance: Instance) -> list[float]:
    height, width = image_bgr.shape[:2]
    mask = actual_mask(instance.points, width, height)
    selected = mask.astype(bool)
    if not selected.any():
        return [0.0] * 12

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb_values = rgb[selected].astype(np.float32) / 255.0
    hsv_values = hsv[selected].astype(np.float32)
    hue = hsv_values[:, 0] / 179.0
    sat = hsv_values[:, 1] / 255.0
    val = hsv_values[:, 2] / 255.0
    green_ratio = float(((hsv_values[:, 0] >= 25) & (hsv_values[:, 0] <= 95) & (hsv_values[:, 1] >= 45)).mean())
    dark_ratio = float((val < 0.35).mean())
    bright_ratio = float((val > 0.75).mean())
    red_mean, green_mean, blue_mean = rgb_values.mean(axis=0).tolist()
    red_std, green_std, blue_std = rgb_values.std(axis=0).tolist()
    return [
        float(hue.mean()),
        float(hue.std()),
        float(sat.mean()),
        float(sat.std()),
        float(val.mean()),
        float(val.std()),
        green_ratio,
        dark_ratio,
        bright_ratio,
        float(red_mean),
        float(green_mean),
        float(blue_mean),
    ]


def feature_names() -> list[str]:
    names = [
        "confidence",
        "log_confidence",
        "area_ratio",
        "log_area_ratio",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "aspect_ratio",
        "mask_extent",
        "centroid_x",
        "centroid_y",
        "edge_distance",
        "perimeter",
        "circularity",
        "point_count",
        "max_pred_overlap",
        "max_higher_conf_overlap",
        "count_overlap_0p1",
        "count_overlap_0p3",
        "count_higher_overlap_0p1",
        "count_higher_overlap_0p3",
        "image_prediction_count",
        "confidence_rank",
        "hue_mean",
        "hue_std",
        "saturation_mean",
        "saturation_std",
        "value_mean",
        "value_std",
        "green_ratio",
        "dark_ratio",
        "bright_ratio",
        "red_mean",
        "green_mean",
        "blue_mean",
    ]
    names.extend(f"source_{source}" for source in SOURCE_NAMES)
    return names


def source_one_hot(source: str) -> list[float]:
    return [1.0 if source == item else 0.0 for item in SOURCE_NAMES]


def assign_labels(pred_gt_ious: list[list[float]], confidences: list[float], iou_threshold: float) -> np.ndarray:
    order = sorted(range(len(confidences)), key=lambda idx: confidences[idx], reverse=True)
    labels = np.zeros(len(confidences), dtype=np.float32)
    matched_gt: set[int] = set()
    for pred_idx in order:
        best_gt = -1
        best_iou = 0.0
        for gt_idx, score in enumerate(pred_gt_ious[pred_idx] if pred_idx < len(pred_gt_ious) else []):
            if gt_idx in matched_gt:
                continue
            if score > best_iou:
                best_iou = score
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            labels[pred_idx] = 1.0
            matched_gt.add(best_gt)
    return labels


def build_split_records(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    pred_labels_dir: Path,
    raster_size: int,
    iou_threshold: float,
    candidate_min_confidence: float,
    max_candidates_per_image: int,
) -> tuple[list[SplitRecord], list[dict[str, object]]]:
    records: list[SplitRecord] = []
    skipped: list[dict[str, object]] = []
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    print(f"[{split_name}] building candidate records for {len(image_paths)} images", flush=True)
    for image_idx, image_path in enumerate(image_paths, start=1):
        if image_idx == 1 or image_idx % 50 == 0 or image_idx == len(image_paths):
            print(f"[{split_name}] {image_idx}/{len(image_paths)} {image_path.name}", flush=True)
        gt = read_polygons(labels_dir / f"{image_path.stem}.txt", has_confidence=False)
        preds_raw = read_polygons(pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True)
        preds_raw = [
            pred
            for pred in preds_raw
            if pred.confidence is not None and float(pred.confidence) >= candidate_min_confidence
        ]
        preds_raw.sort(key=lambda pred: float(pred.confidence or 0.0), reverse=True)
        if max_candidates_per_image > 0:
            preds_raw = preds_raw[:max_candidates_per_image]
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append({"image": image_path.name, "reason": "image_read_failed"})
            continue

        gt_masks = [mask_from_points(instance.points, raster_size) for instance in gt]
        gt_areas = [int(mask.sum()) for mask in gt_masks]
        pred_masks: list[np.ndarray] = []
        preds: list[Instance] = []
        pred_areas: list[int] = []
        for pred in preds_raw:
            if pred.confidence is None:
                continue
            mask = mask_from_points(pred.points, raster_size)
            area = int(mask.sum())
            if area <= 0:
                skipped.append({"image": image_path.name, "reason": "zero_area_prediction"})
                continue
            preds.append(pred)
            pred_masks.append(mask)
            pred_areas.append(area)

        if preds:
            pred_gt_ious = pairwise_iou_matrix(pred_masks, pred_areas, gt_masks, gt_areas)
            pred_pred_ious = pairwise_iou_matrix(pred_masks, pred_areas, pred_masks, pred_areas)
        else:
            pred_gt_ious = []
            pred_pred_ious = []

        confidences = [float(pred.confidence or 0.0) for pred in preds]
        area_ratios = [area / float(raster_size * raster_size) for area in pred_areas]
        source = source_hint(image_path.name)
        features: list[list[float]] = []
        for idx, pred in enumerate(preds):
            row = polygon_features(pred, area_ratios[idx], pred_pred_ious, idx, confidences)
            row.extend(color_features(image, pred))
            row.extend(source_one_hot(source))
            features.append(row)
        labels = assign_labels(pred_gt_ious, confidences, iou_threshold)
        feature_array = (
            np.asarray(features, dtype=np.float32)
            if features
            else np.empty((0, len(feature_names())), dtype=np.float32)
        )
        records.append(
            SplitRecord(
                image=image_path.name,
                source=source,
                gt_count=len(gt),
                features=feature_array,
                confidences=np.asarray(confidences, dtype=np.float32),
                area_ratios=np.asarray(area_ratios, dtype=np.float32),
                pred_gt_ious=pred_gt_ious,
                pred_pred_ious=pred_pred_ious,
                labels=labels,
            )
        )
    print(f"[{split_name}] candidate records complete", flush=True)
    return records, skipped


def flatten_features(records: list[SplitRecord]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [record.features for record in records if len(record.features)]
    labels = [record.labels for record in records if len(record.labels)]
    if not arrays:
        raise ValueError("No candidate features found")
    return np.vstack(arrays), np.concatenate(labels)


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[CandidateMLP, np.ndarray, np.ndarray, list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    x = torch.from_numpy(((features - mean) / std).astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))

    model = CandidateMLP(x.shape[1])
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    pos_weight = torch.tensor([min(max(negatives / max(positives, 1.0), 1.0), 80.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            with torch.no_grad():
                probabilities = torch.sigmoid(model(x))
                predicted = probabilities >= 0.5
                tp = float(((predicted == 1) & (y == 1)).sum().item())
                fp = float(((predicted == 1) & (y == 0)).sum().item())
                fn = float(((predicted == 0) & (y == 1)).sum().item())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.item()),
                    "precision_at_0p5": precision,
                    "recall_at_0p5": recall,
                }
            )
    return model, mean.astype(np.float32), std.astype(np.float32), history


def predict_records(records: list[SplitRecord], model: CandidateMLP, mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for record in records:
            if len(record.features) == 0:
                outputs.append(np.zeros(0, dtype=np.float32))
                continue
            x = torch.from_numpy(((record.features - mean) / std).astype(np.float32))
            outputs.append(torch.sigmoid(model(x)).cpu().numpy().astype(np.float32))
    return outputs


def mask_nms(selected: list[int], pred_pred_ious: list[list[float]], scores: np.ndarray, nms_iou: float) -> list[int]:
    ordered = sorted(selected, key=lambda idx: float(scores[idx]), reverse=True)
    kept: list[int] = []
    for pred_idx in ordered:
        if any(pred_pred_ious[pred_idx][kept_idx] > nms_iou for kept_idx in kept):
            continue
        kept.append(pred_idx)
    return kept


def evaluate_records(
    records: list[SplitRecord],
    probabilities: list[np.ndarray],
    prob_threshold: float,
    confidence_threshold: float,
    nms_iou_threshold: float,
    min_area_ratio: float,
    match_iou_threshold: float,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    total_tp = total_fp = total_fn = total_gt = total_selected = total_kept = 0
    per_image: list[dict[str, object]] = []
    for record, probs in zip(records, probabilities):
        selected = [
            idx
            for idx in range(len(record.confidences))
            if record.confidences[idx] >= confidence_threshold
            and record.area_ratios[idx] >= min_area_ratio
            and probs[idx] >= prob_threshold
        ]
        kept = mask_nms(selected, record.pred_pred_ious, probs, nms_iou_threshold)
        matched_gt: set[int] = set()
        tp = fp = 0
        for pred_idx in kept:
            best_gt = -1
            best_iou = 0.0
            row = record.pred_gt_ious[pred_idx] if pred_idx < len(record.pred_gt_ious) else []
            for gt_idx, score in enumerate(row):
                if gt_idx in matched_gt:
                    continue
                if score > best_iou:
                    best_iou = score
                    best_gt = gt_idx
            if best_gt >= 0 and best_iou >= match_iou_threshold:
                matched_gt.add(best_gt)
                tp += 1
            else:
                fp += 1
        fn = record.gt_count - len(matched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += record.gt_count
        total_selected += len(selected)
        total_kept += len(kept)
        per_image.append(
            {
                "image": record.image,
                "source": record.source,
                "gt": record.gt_count,
                "selected_predictions": len(selected),
                "kept_predictions": len(kept),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        {
            "probability_threshold": prob_threshold,
            "confidence_threshold": confidence_threshold,
            "nms_iou_threshold": nms_iou_threshold,
            "min_area_ratio": min_area_ratio,
            "gt": total_gt,
            "selected_predictions": total_selected,
            "kept_predictions": total_kept,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        per_image,
    )


def scan_grid(
    records: list[SplitRecord],
    probabilities: list[np.ndarray],
    prob_thresholds: list[float],
    confidence_thresholds: list[float],
    nms_iou_thresholds: list[float],
    min_area_ratios: list[float],
    match_iou_threshold: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for prob_threshold in prob_thresholds:
        for confidence_threshold in confidence_thresholds:
            for nms_iou_threshold in nms_iou_thresholds:
                for min_area_ratio in min_area_ratios:
                    metrics, _ = evaluate_records(
                        records,
                        probabilities,
                        prob_threshold,
                        confidence_threshold,
                        nms_iou_threshold,
                        min_area_ratio,
                        match_iou_threshold,
                    )
                    rows.append(metrics)
    return rows


def choose_configs(rows: list[dict[str, float | int]], precision_target: float, recall_target: float) -> dict[str, dict[str, float | int] | None]:
    target_rows = [row for row in rows if float(row["precision"]) >= precision_target and float(row["recall"]) >= recall_target]
    precision_rows = [row for row in rows if float(row["precision"]) >= precision_target]
    recall_rows = [row for row in rows if float(row["recall"]) >= recall_target]
    return {
        "best_f1": max(rows, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"]))),
        "best_target": max(target_rows, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"]))) if target_rows else None,
        "best_recall_with_precision_target": max(precision_rows, key=lambda row: (float(row["recall"]), float(row["f1"]))) if precision_rows else None,
        "best_precision_with_recall_target": max(recall_rows, key=lambda row: (float(row["precision"]), float(row["f1"]))) if recall_rows else None,
    }


def evaluate_config_on_split(
    config: dict[str, float | int] | None,
    records: list[SplitRecord],
    probabilities: list[np.ndarray],
    match_iou_threshold: float,
) -> tuple[dict[str, float | int] | None, list[dict[str, object]]]:
    if config is None:
        return None, []
    return evaluate_records(
        records,
        probabilities,
        float(config["probability_threshold"]),
        float(config["confidence_threshold"]),
        float(config["nms_iou_threshold"]),
        float(config["min_area_ratio"]),
        match_iou_threshold,
    )


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def split_candidate_summary(records: list[SplitRecord]) -> dict[str, object]:
    gt = sum(record.gt_count for record in records)
    candidates = sum(len(record.confidences) for record in records)
    positives = int(sum(int(record.labels.sum()) for record in records))
    by_source: dict[str, dict[str, int]] = {}
    for record in records:
        row = by_source.setdefault(record.source, {"images": 0, "gt": 0, "candidates": 0, "positives": 0})
        row["images"] += 1
        row["gt"] += record.gt_count
        row["candidates"] += len(record.confidences)
        row["positives"] += int(record.labels.sum())
    return {
        "images": len(records),
        "gt": gt,
        "candidates": candidates,
        "candidate_positives": positives,
        "candidate_negatives": candidates - positives,
        "by_source": by_source,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_records, train_skipped = build_split_records(
        "train",
        args.train_images_dir,
        args.train_labels_dir,
        args.train_pred_labels_dir,
        args.raster_size,
        args.match_iou_threshold,
        args.candidate_min_confidence,
        args.max_candidates_per_image,
    )
    val_records, val_skipped = build_split_records(
        "val",
        args.val_images_dir,
        args.val_labels_dir,
        args.val_pred_labels_dir,
        args.raster_size,
        args.match_iou_threshold,
        args.candidate_min_confidence,
        args.max_candidates_per_image,
    )
    test_records, test_skipped = build_split_records(
        "test",
        args.test_images_dir,
        args.test_labels_dir,
        args.test_pred_labels_dir,
        args.raster_size,
        args.match_iou_threshold,
        args.candidate_min_confidence,
        args.max_candidates_per_image,
    )

    train_features, train_labels = flatten_features(train_records)
    model, mean, std, history = train_model(
        train_features,
        train_labels,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.seed,
    )

    val_probabilities = predict_records(val_records, model, mean, std)
    test_probabilities = predict_records(test_records, model, mean, std)

    prob_thresholds = parse_float_list(args.prob_thresholds)
    confidence_thresholds = parse_float_list(args.confidence_thresholds)
    nms_iou_thresholds = parse_float_list(args.nms_iou_thresholds)
    min_area_ratios = parse_float_list(args.min_area_ratios)

    val_rows = scan_grid(
        val_records,
        val_probabilities,
        prob_thresholds,
        confidence_thresholds,
        nms_iou_thresholds,
        min_area_ratios,
        args.match_iou_threshold,
    )
    val_choices = choose_configs(val_rows, args.precision_target, args.recall_target)

    eval_from_val: dict[str, object] = {}
    per_image_outputs: dict[str, list[dict[str, object]]] = {}
    for name, config in val_choices.items():
        metrics, per_image = evaluate_config_on_split(config, test_records, test_probabilities, args.match_iou_threshold)
        eval_from_val[name] = {"val": config, "test": metrics}
        per_image_outputs[name] = per_image

    test_rows = scan_grid(
        test_records,
        test_probabilities,
        prob_thresholds,
        confidence_thresholds,
        nms_iou_thresholds,
        min_area_ratios,
        args.match_iou_threshold,
    )
    test_oracle = choose_configs(test_rows, args.precision_target, args.recall_target)

    summary = {
        "inputs": {
            "train_images_dir": str(args.train_images_dir),
            "train_labels_dir": str(args.train_labels_dir),
            "train_pred_labels_dir": str(args.train_pred_labels_dir),
            "val_images_dir": str(args.val_images_dir),
            "val_labels_dir": str(args.val_labels_dir),
            "val_pred_labels_dir": str(args.val_pred_labels_dir),
            "test_images_dir": str(args.test_images_dir),
            "test_labels_dir": str(args.test_labels_dir),
            "test_pred_labels_dir": str(args.test_pred_labels_dir),
            "raster_size": args.raster_size,
            "match_iou_threshold": args.match_iou_threshold,
            "candidate_min_confidence": args.candidate_min_confidence,
            "max_candidates_per_image": args.max_candidates_per_image,
        },
        "targets": {"precision": args.precision_target, "recall": args.recall_target},
        "features": feature_names(),
        "train_history": history,
        "candidate_summaries": {
            "train": split_candidate_summary(train_records),
            "val": split_candidate_summary(val_records),
            "test": split_candidate_summary(test_records),
        },
        "skipped": {
            "train": train_skipped[:20],
            "val": val_skipped[:20],
            "test": test_skipped[:20],
            "counts": {"train": len(train_skipped), "val": len(val_skipped), "test": len(test_skipped)},
        },
        "val_selected_configs_test_eval": eval_from_val,
        "test_oracle_grid": test_oracle,
        "meets_targets_by_val_selected_policy": any(
            item.get("test")
            and float(item["test"]["precision"]) >= args.precision_target  # type: ignore[index]
            and float(item["test"]["recall"]) >= args.recall_target  # type: ignore[index]
            for item in eval_from_val.values()
            if isinstance(item, dict)
        ),
        "meets_targets_by_test_oracle_grid": test_oracle["best_target"] is not None,
    }

    write_tsv(args.output_dir / "val_candidate_filter_grid.tsv", [dict(row) for row in val_rows])
    write_tsv(args.output_dir / "test_candidate_filter_oracle_grid.tsv", [dict(row) for row in test_rows])
    for name, rows in per_image_outputs.items():
        write_tsv(args.output_dir / f"{name}_test_per_image.tsv", rows)
    write_tsv(args.output_dir / "train_history.tsv", [dict(row) for row in history])

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "feature_names": feature_names(),
            "args": vars(args),
        },
        args.output_dir / "candidate_filter.pt",
    )
    (args.output_dir / "candidate_filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "val_selected_configs_test_eval": eval_from_val,
                "test_oracle_grid": test_oracle,
                "meets_targets_by_val_selected_policy": summary["meets_targets_by_val_selected_policy"],
                "meets_targets_by_test_oracle_grid": summary["meets_targets_by_test_oracle_grid"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
