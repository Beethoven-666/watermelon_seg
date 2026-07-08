#!/usr/bin/env python
"""Train a small CNN second-stage filter for YOLO segmentation candidates."""

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
    offset: int
    count: int
    confidences: np.ndarray
    area_ratios: np.ndarray
    pred_gt_ious: list[list[float]]
    pred_pred_ious: list[list[float]]
    labels: np.ndarray


class CandidateCNN(nn.Module):
    def __init__(self, scalar_dim: int) -> None:
        super().__init__()
        self.image_net = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        image_features = self.image_net(image).flatten(1)
        scalar_features = self.scalar_net(scalars)
        return self.head(torch.cat([image_features, scalar_features], dim=1)).squeeze(1)


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
    parser.add_argument("--raster-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--crop-expand", type=float, default=1.8)
    parser.add_argument("--candidate-min-confidence", type=float, default=0.03)
    parser.add_argument("--max-candidates-per-image", type=int, default=300)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument(
        "--prob-thresholds",
        default="0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--confidence-thresholds", default="0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--nms-iou-thresholds", default="0.025,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7")
    parser.add_argument("--min-area-ratios", default="0,0.00025,0.0005,0.00075,0.001,0.0015,0.002,0.003,0.004,0.005")
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


def source_one_hot(source: str) -> list[float]:
    return [1.0 if source == item else 0.0 for item in SOURCE_NAMES]


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
    cv2.fillPoly(mask, [polygon], 255)
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


def scalar_features(instance: Instance, area_ratio: float, source: str) -> list[float]:
    points = instance.points
    min_x = float(points[:, 0].min())
    min_y = float(points[:, 1].min())
    max_x = float(points[:, 0].max())
    max_y = float(points[:, 1].max())
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    bbox_area = width * height
    conf = float(instance.confidence or 0.0)
    aspect = width / height
    extent = area_ratio / max(bbox_area, 1e-6)
    edge_distance = min(min_x, min_y, 1.0 - max_x, 1.0 - max_y)
    features = [
        conf,
        math.log(max(conf, 1e-6)),
        area_ratio,
        math.log(max(area_ratio, 1e-8)),
        width,
        height,
        bbox_area,
        aspect,
        extent,
        float(points[:, 0].mean()),
        float(points[:, 1].mean()),
        edge_distance,
        float(len(points)),
    ]
    features.extend(source_one_hot(source))
    return features


def crop_candidate(image_bgr: np.ndarray, instance: Instance, crop_size: int, expand: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    points = instance.points
    min_x = int(np.floor(float(points[:, 0].min()) * (width - 1)))
    min_y = int(np.floor(float(points[:, 1].min()) * (height - 1)))
    max_x = int(np.ceil(float(points[:, 0].max()) * (width - 1)))
    max_y = int(np.ceil(float(points[:, 1].max()) * (height - 1)))
    box_w = max(1, max_x - min_x + 1)
    box_h = max(1, max_y - min_y + 1)
    side = max(box_w, box_h) * expand
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    x2 = int(round(center_x + side / 2.0))
    y2 = int(round(center_y + side / 2.0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = actual_mask(instance.points, width, height)
    crop_rgb = rgb[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    crop_rgb = cv2.resize(crop_rgb, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([crop_rgb, crop_mask[:, :, None]], axis=2).transpose(2, 0, 1).astype(np.uint8)


def build_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    pred_labels_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[SplitRecord], np.ndarray, np.ndarray, np.ndarray]:
    records: list[SplitRecord] = []
    crops: list[np.ndarray] = []
    scalars: list[list[float]] = []
    labels_all: list[np.ndarray] = []
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    print(f"[{split_name}] building CNN candidates for {len(image_paths)} images", flush=True)

    for image_idx, image_path in enumerate(image_paths, start=1):
        if image_idx == 1 or image_idx % 50 == 0 or image_idx == len(image_paths):
            print(f"[{split_name}] {image_idx}/{len(image_paths)} {image_path.name}", flush=True)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gt = read_polygons(labels_dir / f"{image_path.stem}.txt", has_confidence=False)
        preds_raw = [
            pred
            for pred in read_polygons(pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True)
            if pred.confidence is not None and float(pred.confidence) >= args.candidate_min_confidence
        ]
        preds_raw.sort(key=lambda pred: float(pred.confidence or 0.0), reverse=True)
        if args.max_candidates_per_image > 0:
            preds_raw = preds_raw[: args.max_candidates_per_image]

        gt_masks = [mask_from_points(instance.points, args.raster_size) for instance in gt]
        gt_areas = [int(mask.sum()) for mask in gt_masks]
        pred_masks: list[np.ndarray] = []
        preds: list[Instance] = []
        pred_areas: list[int] = []
        for pred in preds_raw:
            mask = mask_from_points(pred.points, args.raster_size)
            area = int(mask.sum())
            if area <= 0:
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
        area_ratios = [area / float(args.raster_size * args.raster_size) for area in pred_areas]
        labels = assign_labels(pred_gt_ious, confidences, args.match_iou_threshold)
        source = source_hint(image_path.name)
        offset = len(crops)
        for pred, area_ratio in zip(preds, area_ratios):
            crops.append(crop_candidate(image, pred, args.crop_size, args.crop_expand))
            scalars.append(scalar_features(pred, area_ratio, source))
        labels_all.append(labels)
        records.append(
            SplitRecord(
                image=image_path.name,
                source=source,
                gt_count=len(gt),
                offset=offset,
                count=len(preds),
                confidences=np.asarray(confidences, dtype=np.float32),
                area_ratios=np.asarray(area_ratios, dtype=np.float32),
                pred_gt_ious=pred_gt_ious,
                pred_pred_ious=pred_pred_ious,
                labels=labels,
            )
        )
    if crops:
        crop_array = np.stack(crops)
        scalar_array = np.asarray(scalars, dtype=np.float32)
        label_array = np.concatenate(labels_all).astype(np.float32)
    else:
        crop_array = np.empty((0, 4, args.crop_size, args.crop_size), dtype=np.uint8)
        scalar_array = np.empty((0, len(scalar_features(Instance(np.zeros((3, 2), dtype=np.float32), 0.0), 0.0, "unknown"))), dtype=np.float32)
        label_array = np.empty((0,), dtype=np.float32)
    print(f"[{split_name}] candidates={len(crop_array)} positives={int(label_array.sum())}", flush=True)
    return records, crop_array, scalar_array, label_array


def normalize_scalars(train_scalars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_scalars.mean(axis=0).astype(np.float32)
    std = train_scalars.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def image_batch(crops: np.ndarray, indices: np.ndarray, device: torch.device, augment: bool) -> torch.Tensor:
    batch = torch.from_numpy(crops[indices].astype(np.float32) / 255.0)
    if augment:
        if random.random() < 0.5:
            batch = torch.flip(batch, dims=[3])
        if random.random() < 0.5:
            batch = torch.flip(batch, dims=[2])
    return batch.to(device)


def scalar_batch(scalars: np.ndarray, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    values = ((scalars[indices] - mean) / std).astype(np.float32)
    return torch.from_numpy(values).to(device)


def train_model(
    train_crops: np.ndarray,
    train_scalars: np.ndarray,
    train_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[CandidateCNN, np.ndarray, np.ndarray, list[dict[str, float]]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = normalize_scalars(train_scalars)
    model = CandidateCNN(train_scalars.shape[1]).to(device)
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = torch.tensor([min(max(negatives / max(positives, 1.0), 1.0), 80.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_all = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    indices_all = np.arange(len(train_labels))
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(indices_all)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(indices_all), args.batch_size):
            batch_indices_np = indices_all[start : start + args.batch_size]
            batch_indices = torch.from_numpy(batch_indices_np).to(device)
            images = image_batch(train_crops, batch_indices_np, device, augment=True)
            scalars = scalar_batch(train_scalars, batch_indices_np, mean, std, device)
            y = y_all[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, scalars)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices_np)
            seen += len(batch_indices_np)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            probabilities = predict_array(model, train_crops, train_scalars, mean, std, args.batch_size)
            predicted = probabilities >= 0.5
            tp = float(((predicted == 1) & (train_labels == 1)).sum())
            fp = float(((predicted == 1) & (train_labels == 0)).sum())
            fn = float(((predicted == 0) & (train_labels == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            row = {
                "epoch": float(epoch),
                "loss": total_loss / max(seen, 1),
                "precision_at_0p5": precision,
                "recall_at_0p5": recall,
            }
            history.append(row)
            print(f"[train] epoch={epoch} loss={row['loss']:.5f} p={precision:.4f} r={recall:.4f}", flush=True)
    return model, mean, std, history


def predict_array(
    model: CandidateCNN,
    crops: np.ndarray,
    scalars: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(crops) == 0:
        return np.empty((0,), dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            indices = np.arange(start, min(start + batch_size, len(crops)))
            logits = model(
                image_batch(crops, indices, device, augment=False),
                scalar_batch(scalars, indices, mean, std, device),
            )
            outputs.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


def split_probabilities(records: list[SplitRecord], probabilities: np.ndarray) -> list[np.ndarray]:
    return [probabilities[record.offset : record.offset + record.count] for record in records]


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
        scores = probs * record.confidences
        selected = [
            idx
            for idx in range(len(record.confidences))
            if record.confidences[idx] >= confidence_threshold
            and record.area_ratios[idx] >= min_area_ratio
            and probs[idx] >= prob_threshold
        ]
        kept = mask_nms(selected, record.pred_pred_ious, scores, nms_iou_threshold)
        matched_gt: set[int] = set()
        tp = fp = 0
        for pred_idx in kept:
            best_gt = -1
            best_iou = 0.0
            for gt_idx, score in enumerate(record.pred_gt_ious[pred_idx] if pred_idx < len(record.pred_gt_ious) else []):
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


def split_summary(records: list[SplitRecord]) -> dict[str, object]:
    by_source: dict[str, dict[str, int]] = {}
    gt = candidates = positives = 0
    for record in records:
        row = by_source.setdefault(record.source, {"images": 0, "gt": 0, "candidates": 0, "positives": 0})
        row["images"] += 1
        row["gt"] += record.gt_count
        row["candidates"] += record.count
        row["positives"] += int(record.labels.sum())
        gt += record.gt_count
        candidates += record.count
        positives += int(record.labels.sum())
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

    train_records, train_crops, train_scalars, train_labels = build_split(
        "train", args.train_images_dir, args.train_labels_dir, args.train_pred_labels_dir, args
    )
    val_records, val_crops, val_scalars, _val_labels = build_split(
        "val", args.val_images_dir, args.val_labels_dir, args.val_pred_labels_dir, args
    )
    test_records, test_crops, test_scalars, _test_labels = build_split(
        "test", args.test_images_dir, args.test_labels_dir, args.test_pred_labels_dir, args
    )
    if len(train_labels) == 0:
        raise ValueError("No train candidates found")

    model, mean, std, history = train_model(train_crops, train_scalars, train_labels, args)
    val_probs_flat = predict_array(model, val_crops, val_scalars, mean, std, args.batch_size)
    test_probs_flat = predict_array(model, test_crops, test_scalars, mean, std, args.batch_size)
    val_probabilities = split_probabilities(val_records, val_probs_flat)
    test_probabilities = split_probabilities(test_records, test_probs_flat)

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
            "crop_size": args.crop_size,
            "crop_expand": args.crop_expand,
            "candidate_min_confidence": args.candidate_min_confidence,
            "match_iou_threshold": args.match_iou_threshold,
        },
        "targets": {"precision": args.precision_target, "recall": args.recall_target},
        "train_history": history,
        "candidate_summaries": {
            "train": split_summary(train_records),
            "val": split_summary(val_records),
            "test": split_summary(test_records),
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

    write_tsv(args.output_dir / "val_candidate_cnn_grid.tsv", [dict(row) for row in val_rows])
    write_tsv(args.output_dir / "test_candidate_cnn_oracle_grid.tsv", [dict(row) for row in test_rows])
    for name, rows in per_image_outputs.items():
        write_tsv(args.output_dir / f"{name}_test_per_image.tsv", rows)
    write_tsv(args.output_dir / "train_history.tsv", [dict(row) for row in history])

    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
        },
        args.output_dir / "candidate_cnn_filter.pt",
    )
    (args.output_dir / "candidate_cnn_filter_summary.json").write_text(
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
