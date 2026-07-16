#!/usr/bin/env python
"""Benchmark SAM 3 watermelon instance segmentation on val then test."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import predict_sam3_zero_shot_fruits as sam3_deploy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "runs27" / "sam3" / "watermelon_zero_shot_benchmark"


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path
    source: str


@dataclass
class EvalRecord:
    image_id: int
    file_name: str
    source: str
    width: int
    height: int
    inference_width: int
    inference_height: int
    gt_rles: list[dict[str, Any]]
    pred_rles: list[dict[str, Any]]
    scores: np.ndarray
    ious: np.ndarray
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-images", type=Path, default=PROJECT_ROOT / "images" / "val")
    parser.add_argument("--val-labels", type=Path, default=PROJECT_ROOT / "labels" / "val")
    parser.add_argument("--test-images", type=Path, default=PROJECT_ROOT / "images" / "test")
    parser.add_argument("--test-labels", type=Path, default=PROJECT_ROOT / "labels" / "test")
    parser.add_argument(
        "--val-coco",
        type=Path,
        help="Optional COCO JSON whose image list filters val while YOLO labels remain the GT source.",
    )
    parser.add_argument(
        "--test-coco",
        type=Path,
        help="Optional COCO JSON whose image list filters test while YOLO labels remain the GT source.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, help="Local sam3.pt; otherwise use the HF cache/download.")
    parser.add_argument("--prompt", default="watermelon", help="Frozen English concept prompt.")
    parser.add_argument("--target-class-id", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Limit the image passed to SAM, then resize masks back to original size for COCO evaluation.",
    )
    parser.add_argument(
        "--candidate-floor",
        type=float,
        default=0.0,
        help="Low score floor used to build the AP/threshold candidate pool.",
    )
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "none"), default="bfloat16")
    parser.add_argument("--fixed-threshold", type=float, default=0.50)
    parser.add_argument(
        "--model-kind",
        choices=("zero_shot", "finetuned"),
        default="zero_shot",
        help="Controls report wording; it does not alter inference.",
    )
    parser.add_argument(
        "--model-label",
        default="facebook/sam3",
        help="Auditable human-readable label written to the report.",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset counts without loading SAM 3.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def source_name(file_name: str) -> str:
    prefixes = (
        "local_test_images",
        "my_labelme",
        "roboflow_drago_v1",
        "roboflow_team128_v5",
    )
    for prefix in prefixes:
        if file_name.startswith(prefix):
            return prefix
    return "unknown"


def dataset_items(images_dir: Path, labels_dir: Path) -> list[DatasetItem]:
    images_dir = images_dir.expanduser().resolve()
    labels_dir = labels_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    images = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {images_dir}")
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate image stems are not supported: {images_dir}")

    rows: list[DatasetItem] = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        rows.append(DatasetItem(image_path, label_path, source_name(image_path.name)))
    extra_labels = sorted(path.name for path in labels_dir.glob("*.txt") if path.stem not in set(stems))
    if extra_labels:
        raise ValueError(f"Labels without matching images in {labels_dir}: {extra_labels[:5]}")
    return rows


def dataset_items_from_coco(coco_path: Path) -> list[DatasetItem]:
    """Resolve a derived COCO image list back to the immutable project YOLO files."""
    coco_path = coco_path.expanduser().resolve()
    payload = json.loads(coco_path.read_text(encoding="utf-8-sig"))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"COCO JSON has no images: {coco_path}")
    rows: list[DatasetItem] = []
    seen: set[Path] = set()
    for image in sorted(images, key=lambda row: str(row.get("file_name", ""))):
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"Invalid COCO image file_name in {coco_path}")
        relative = Path(file_name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe COCO image path: {file_name}")
        image_path = (PROJECT_ROOT / relative).resolve()
        try:
            image_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"COCO image escapes the project root: {file_name}") from exc
        parts = relative.parts
        if len(parts) < 3 or parts[0] != "images" or parts[1] not in {"train", "val", "test"}:
            raise ValueError(
                f"Expected COCO file_name images/<split>/<name>, got: {file_name}"
            )
        label_path = PROJECT_ROOT / "labels" / parts[1] / f"{image_path.stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image/label for COCO entry: {file_name}")
        if image_path in seen:
            raise ValueError(f"Duplicate COCO image path: {file_name}")
        seen.add(image_path)
        rows.append(DatasetItem(image_path, label_path, source_name(image_path.name)))
    return rows


def parse_yolo_polygons(label_path: Path, target_class_id: int) -> list[list[float]]:
    polygons: list[list[float]] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid YOLO row: {label_path}:{line_number}") from exc
        if class_id != target_class_id:
            continue
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"Invalid polygon coordinate count: {label_path}:{line_number}")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"Non-finite/out-of-range coordinate: {label_path}:{line_number}")
        polygons.append(coords)
    return polygons


def gt_rles(label_path: Path, width: int, height: int, target_class_id: int) -> list[dict[str, Any]]:
    rles: list[dict[str, Any]] = []
    for coords in parse_yolo_polygons(label_path, target_class_id):
        pixel_polygon: list[float] = []
        for index in range(0, len(coords), 2):
            pixel_polygon.extend((coords[index] * width, coords[index + 1] * height))
        encoded = mask_utils.frPyObjects([pixel_polygon], height, width)
        rle = mask_utils.merge(encoded)
        if float(mask_utils.area(rle)) <= 0:
            raise ValueError(f"Zero-area ground-truth polygon: {label_path}")
        rles.append(rle)
    return rles


def dataset_summary(items: list[DatasetItem], target_class_id: int) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    instances_by_source: Counter[str] = Counter()
    instances = 0
    background_images = 0
    max_size = (0, 0, 0)
    for item in items:
        polygons = parse_yolo_polygons(item.label_path, target_class_id)
        count = len(polygons)
        instances += count
        by_source[item.source] += 1
        instances_by_source[item.source] += count
        background_images += int(count == 0)
        with Image.open(item.image_path) as image:
            pixels = image.width * image.height
            if pixels > max_size[0]:
                max_size = (pixels, image.width, image.height)
    return {
        "images": len(items),
        "instances": instances,
        "background_images": background_images,
        "images_by_source": dict(sorted(by_source.items())),
        "instances_by_source": dict(sorted(instances_by_source.items())),
        "max_image_size": {"width": max_size[1], "height": max_size[2]},
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(items: list[DatasetItem]) -> str:
    digest = hashlib.sha256()
    for item in items:
        for role, path in (("image", item.image_path), ("label", item.label_path)):
            digest.update(role.encode("ascii"))
            digest.update(b"\0")
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha256(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _tensor_outputs(output: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    masks = output["masks"].detach().to("cpu").numpy()
    scores = output["scores"].detach().float().to("cpu").numpy().reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected SAM 3 mask shape: {masks.shape}")
    if len(masks) != len(scores):
        raise ValueError(f"SAM 3 returned {len(masks)} masks but {len(scores)} scores")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("SAM 3 returned a non-finite or out-of-range score")
    return masks.astype(bool, copy=False), scores


def prediction_rles(
    masks: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    min_area_ratio: float,
    max_detections: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    kept_rles: list[dict[str, Any]] = []
    kept_scores: list[float] = []
    for index in order.tolist():
        mask = masks[index]
        area_ratio = float(mask.mean())
        if not mask.any() or area_ratio < min_area_ratio:
            continue
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.shape != (height, width):
            mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
        rle = mask_utils.encode(np.asfortranarray(mask_u8))
        kept_rles.append(rle)
        kept_scores.append(float(scores[index]))
        if len(kept_rles) >= max_detections:
            break
    return kept_rles, np.asarray(kept_scores, dtype=np.float64)


def pairwise_iou(pred_rles: list[dict[str, Any]], true_rles: list[dict[str, Any]]) -> np.ndarray:
    if not pred_rles or not true_rles:
        return np.zeros((len(pred_rles), len(true_rles)), dtype=np.float64)
    return np.asarray(mask_utils.iou(pred_rles, true_rles, [0] * len(true_rles)), dtype=np.float64)


def infer_split(
    split_name: str,
    items: list[DatasetItem],
    processor: Any,
    torch_module: Any,
    args: argparse.Namespace,
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for image_id, item in enumerate(items, start=1):
        started = time.perf_counter()
        if not getattr(args, "quiet", False):
            print(f"[{split_name} {image_id}/{len(items)}] {item.image_path.name}", flush=True)
        with Image.open(item.image_path) as opened:
            original = opened.convert("RGB")
        width, height = original.size
        inference_image = sam3_deploy.resize_for_inference(original, args.max_image_side)
        inference_width, inference_height = inference_image.size
        true_rles = gt_rles(item.label_path, width, height, args.target_class_id)

        with sam3_deploy._autocast(torch_module, args.device, args.amp_dtype):
            state = processor.set_image(inference_image)
            output = processor.set_text_prompt(state=state, prompt=args.prompt)
        masks, scores = _tensor_outputs(output)
        pred_rles, kept_scores = prediction_rles(
            masks,
            scores,
            width,
            height,
            args.min_area_ratio,
            args.max_detections,
        )
        ious = pairwise_iou(pred_rles, true_rles)
        records.append(
            EvalRecord(
                image_id=image_id,
                file_name=item.image_path.name,
                source=item.source,
                width=width,
                height=height,
                inference_width=inference_width,
                inference_height=inference_height,
                gt_rles=true_rles,
                pred_rles=pred_rles,
                scores=kept_scores,
                ious=ious,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        del state, output, masks, scores, pred_rles, true_rles, original, inference_image
        if args.device == "cuda":
            torch_module.cuda.empty_cache()
    return records


def maximum_matches(ious: np.ndarray, selected: Iterable[int], iou_threshold: float) -> int:
    gt_count = ious.shape[1]
    gt_match = [-1] * gt_count

    def augment(pred_index: int, seen_gt: set[int]) -> bool:
        neighbors = np.flatnonzero(ious[pred_index] >= iou_threshold)
        if neighbors.size > 1:
            neighbors = neighbors[np.argsort(-ious[pred_index, neighbors], kind="stable")]
        for gt_index_raw in neighbors.tolist():
            gt_index = int(gt_index_raw)
            if gt_index in seen_gt:
                continue
            seen_gt.add(gt_index)
            previous_pred = gt_match[gt_index]
            if previous_pred < 0 or augment(previous_pred, seen_gt):
                gt_match[gt_index] = pred_index
                return True
        return False

    matches = 0
    for pred_index in selected:
        matches += int(augment(int(pred_index), set()))
    return matches


def metric_row(threshold: float, tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    instance_accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "instance_accuracy": float(instance_accuracy),
    }


def fixed_metrics(
    records: list[EvalRecord],
    threshold: float,
    iou_threshold: float,
    only_source: str | None = None,
) -> dict[str, Any]:
    total_tp = 0
    total_pred = 0
    total_gt = 0
    for record in records:
        if only_source is not None and record.source != only_source:
            continue
        selected = np.flatnonzero(record.scores >= threshold).tolist()
        total_tp += maximum_matches(record.ious, selected, iou_threshold)
        total_pred += len(selected)
        total_gt += len(record.gt_rles)
    return metric_row(threshold, total_tp, total_pred - total_tp, total_gt - total_tp)


def threshold_curve(records: list[EvalRecord], iou_threshold: float) -> list[dict[str, Any]]:
    total_gt = sum(len(record.gt_rles) for record in records)
    events: list[tuple[float, int, int]] = []
    for record_index, record in enumerate(records):
        events.extend((float(score), record_index, pred_index) for pred_index, score in enumerate(record.scores))
    events.sort(key=lambda row: (-row[0], records[row[1]].file_name, row[2]))

    gt_matches = [[-1] * len(record.gt_rles) for record in records]

    def augment(record_index: int, pred_index: int, seen_gt: set[int]) -> bool:
        record = records[record_index]
        neighbors = np.flatnonzero(record.ious[pred_index] >= iou_threshold)
        if neighbors.size > 1:
            neighbors = neighbors[np.argsort(-record.ious[pred_index, neighbors], kind="stable")]
        for gt_index_raw in neighbors.tolist():
            gt_index = int(gt_index_raw)
            if gt_index in seen_gt:
                continue
            seen_gt.add(gt_index)
            previous_pred = gt_matches[record_index][gt_index]
            if previous_pred < 0 or augment(record_index, previous_pred, seen_gt):
                gt_matches[record_index][gt_index] = pred_index
                return True
        return False

    curve: list[dict[str, Any]] = []
    selected_count = 0
    tp = 0
    cursor = 0
    while cursor < len(events):
        score = events[cursor][0]
        group_end = cursor
        while group_end < len(events) and events[group_end][0] == score:
            _, record_index, pred_index = events[group_end]
            tp += int(augment(record_index, pred_index, set()))
            selected_count += 1
            group_end += 1
        curve.append(metric_row(score, tp, selected_count - tp, total_gt - tp))
        cursor = group_end
    if not curve:
        curve.append(metric_row(1.0, 0, 0, total_gt))
    return curve


def choose_best_f1(curve: list[dict[str, Any]]) -> dict[str, Any]:
    return max(curve, key=lambda row: (row["f1"], row["recall"], row["threshold"]))


def choose_p95_r90(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in curve if row["precision"] >= 0.95 and row["recall"] >= 0.90]
    return max(eligible, key=lambda row: (row["f1"], row["recall"], row["threshold"])) if eligible else None


def choose_p95_max_recall(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in curve if row["precision"] >= 0.95]
    return max(eligible, key=lambda row: (row["recall"], row["f1"], row["threshold"])) if eligible else None


def coco_metrics(records: list[EvalRecord]) -> dict[str, Any]:
    gt = COCO()
    annotations: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotation_id = 1
    results: list[dict[str, Any]] = []
    for record in records:
        images.append(
            {
                "id": record.image_id,
                "file_name": record.file_name,
                "width": record.width,
                "height": record.height,
            }
        )
        for rle in record.gt_rles:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": record.image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "bbox": [float(value) for value in mask_utils.toBbox(rle).tolist()],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for rle, score in zip(record.pred_rles, record.scores.tolist()):
            results.append(
                {
                    "image_id": record.image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": float(score),
                }
            )
    gt.dataset = {
        "info": {"description": "Watermelon SAM 3 evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "watermelon", "supercategory": "fruit"}],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    if not results:
        return {
            "mask_map_50_95": 0.0,
            "mask_ap50": 0.0,
            "mask_ap75": 0.0,
            "mask_ar100": 0.0,
            "cocoeval_output": "No predictions; metrics set to zero.",
        }
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
    evaluator = COCOeval(gt, dt, "segm")
    evaluator.params.imgIds = [record.image_id for record in records]
    evaluator.params.catIds = [1]
    evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
    evaluator.params.maxDets = [1, 10, 100]
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "mask_map_50_95": float(evaluator.stats[0]),
        "mask_ap50": float(evaluator.stats[1]),
        "mask_ap75": float(evaluator.stats[2]),
        "mask_ar100": float(evaluator.stats[8]),
        "cocoeval_output": stream.getvalue(),
    }


def per_image_rows(
    split: str,
    records: list[EvalRecord],
    threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        selected = np.flatnonzero(record.scores >= threshold).tolist()
        tp = maximum_matches(record.ious, selected, iou_threshold)
        metrics = metric_row(threshold, tp, len(selected) - tp, len(record.gt_rles) - tp)
        rows.append(
            {
                "split": split,
                "image": record.file_name,
                "source": record.source,
                "width": record.width,
                "height": record.height,
                "inference_width": record.inference_width,
                "inference_height": record.inference_height,
                "gt_instances": len(record.gt_rles),
                "candidate_pool": len(record.pred_rles),
                "elapsed_seconds": f"{record.elapsed_seconds:.6f}",
                **metrics,
            }
        )
    return rows


def per_source_metrics(records: list[EvalRecord], threshold: float, iou_threshold: float) -> dict[str, Any]:
    return {
        source: fixed_metrics(records, threshold, iou_threshold, only_source=source)
        for source in sorted({record.source for record in records})
    }


def raw_cache(records: list[EvalRecord]) -> list[dict[str, Any]]:
    return [
        {
            "image_id": record.image_id,
            "image": record.file_name,
            "source": record.source,
            "width": record.width,
            "height": record.height,
            "inference_width": record.inference_width,
            "inference_height": record.inference_height,
            "scores": record.scores.tolist(),
            "ious": record.ious.tolist(),
            "gt_count": len(record.gt_rles),
            "elapsed_seconds": record.elapsed_seconds,
        }
        for record in records
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def policy_table_row(name: str, threshold: float, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {threshold:.6f} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
        f"{format_percent(metrics['precision'])} | {format_percent(metrics['recall'])} | "
        f"{format_percent(metrics['f1'])} | {format_percent(metrics['instance_accuracy'])} |"
    )


def markdown_report(summary: dict[str, Any]) -> str:
    val_best = summary["policies"]["val_best_f1"]
    test_best = summary["policies"]["test_at_val_best_f1"]
    test_fixed = summary["policies"]["test_at_fixed_threshold"]
    test_oracle = summary["diagnostics"]["test_oracle_best_f1"]
    coco = summary["test_coco_mask_metrics"]
    test_backgrounds = summary["datasets"]["test"]["background_images"]
    source_rows = [
        (
            f"| {source} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{format_percent(metrics['precision'])} | {format_percent(metrics['recall'])} | "
            f"{format_percent(metrics['f1'])} | {format_percent(metrics['instance_accuracy'])} |"
        )
        for source, metrics in summary["test_by_source_at_val_best_f1"].items()
    ]
    overlap_audit = summary.get("post_run_audit", {}).get("dataset_split_overlap")
    overlap_audit_lines: list[str] = []
    finetuned = summary["settings"].get("model_kind") == "finetuned"
    report_title = (
        "# SAM 3 西瓜微调实例分割评测"
        if finetuned
        else "# SAM 3 西瓜零样本实例分割评测"
    )
    tuning_note = (
        "已使用本项目 clean train 微调；checkpoint/阈值仅由 clean val 选择"
        if finetuned
        else "未使用本项目西瓜数据微调"
    )
    if overlap_audit:
        deduplicated = overlap_audit["source_independent_deduplicated_test"]
        overlap_audit_lines.append(
            "- 后验拆分审计发现 val/test 有 "
            f"{overlap_audit['val_test_source_scene_groups']} 个同源旋转/重编码场景，test 内另有 "
            f"{overlap_audit['test_exact_duplicate_groups']} 组重复图；剔除后 test F1 约 "
            f"{format_percent(deduplicated['f1'])}，正式 F1 可能高估约 "
            f"{overlap_audit['formal_f1_overestimate_percentage_points']:.2f} 个百分点，但未达 P90/R90 的结论不变。"
            f"详见 `{overlap_audit['report']}`。"
        )
    lines = [
        report_title,
        "",
        f"- 生成时间：`{summary['created_at_utc']}`",
        f"- 模型：`{summary['settings']['model_label']}`，提交：`{summary['model'].get('git_commit')}`",
        f"- 文本提示：`{summary['settings']['prompt']}`（{tuning_note}）",
        f"- 推理分辨率：`{summary['settings']['resolution']}`；传入图最长边：`{summary['settings']['max_image_side']}`",
        f"- 候选分数下限：`{summary['settings']['candidate_floor']}`；每图最多：`{summary['settings']['max_detections']}`",
        f"- 固定阈值匹配：mask IoU `>= {summary['settings']['match_iou']}` 的最大基数一对一匹配",
        "",
        "## 数据与评测纪律",
        "",
        f"- val：{summary['datasets']['val']['images']} 张 / {summary['datasets']['val']['instances']} 个实例，用于选择部署置信度。",
        f"- test：{summary['datasets']['test']['images']} 张 / {summary['datasets']['test']['instances']} 个实例，只应用 val 已冻结策略。",
        "- 提示词、分辨率、候选池和 mask 后处理均在查看 test 指标前冻结；test oracle 只作诊断，不作为部署结果。",
        "",
        "## Test 标准 Mask AP",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| COCO Mask mAP50-95 | {format_percent(coco['mask_map_50_95'])} |",
        f"| COCO Mask AP50 | {format_percent(coco['mask_ap50'])} |",
        f"| COCO Mask AP75 | {format_percent(coco['mask_ap75'])} |",
        f"| COCO Mask AR100 | {format_percent(coco['mask_ar100'])} |",
        "",
        "## 固定操作点",
        "",
        "| 操作点 | conf | TP | FP | FN | Precision | Recall | F1 | Instance Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        policy_table_row("val 选择的最佳 F1（val）", val_best["threshold"], val_best),
        policy_table_row("val 阈值冻结到 test（主结果）", val_best["threshold"], test_best),
        policy_table_row(
            f"固定 conf={summary['settings']['fixed_threshold']:.2f}（test）",
            summary["settings"]["fixed_threshold"],
            test_fixed,
        ),
        policy_table_row("test oracle（仅诊断，禁止部署选参）", test_oracle["threshold"], test_oracle),
        "",
        "## Test 按来源（val 冻结阈值）",
        "",
        "| 来源 | TP | FP | FN | Precision | Recall | F1 | Instance Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *source_rows,
        "",
        "## 验收判断",
        "",
        f"- val 阈值冻结到 test 后 P90/R90：`{'通过' if summary['acceptance']['test_p90_r90'] else '未通过'}`。",
        f"- val 阈值冻结到 test 后 P95/R90：`{'通过' if summary['acceptance']['test_p95_r90'] else '未通过'}`。",
        "- 本结论只覆盖当前西瓜正样本实例分割，不等于多水果分拣系统已经验收。",
        "",
        "## 关键限制",
        "",
        (
            f"- 当前 test 有 {test_backgrounds} 张无目标图片；仍不是专门采集的空传送带/其他水果负样本集，"
            "不能据此证明现场特异度和跨水果拒识能力。"
        ),
        *overlap_audit_lines,
        "- 当前官方 image checkpoint 固定使用 resolution=1008；本机以 max_image_side=1024 控制 mask 上采样显存。",
        "- `instance_accuracy=TP/(TP+FP+FN)` 仅作补充；实例分割主指标应看 AP、Precision、Recall 和 F1。",
        "- 图像内抓取点还需深度、相机标定、遮挡与碰撞检查，不能直接下发机械臂。",
        "",
    ]
    return "\n".join(lines)


def prepare_output(path: Path, allow_existing: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("candidate_floor", "min_area_ratio", "match_iou", "fixed_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.resolution != 1008:
        raise ValueError(
            "facebook/sam3 image inference requires --resolution 1008 with the current official checkpoint"
        )
    if args.max_image_side < 224 or args.max_detections < 1:
        raise ValueError("max-image-side must be >=224 and max-detections must be positive")
    if not args.prompt.strip():
        raise ValueError("--prompt cannot be empty")

    val_items = (
        dataset_items_from_coco(args.val_coco)
        if args.val_coco
        else dataset_items(args.val_images, args.val_labels)
    )
    test_items = (
        dataset_items_from_coco(args.test_coco)
        if args.test_coco
        else dataset_items(args.test_images, args.test_labels)
    )
    datasets = {
        "val": dataset_summary(val_items, args.target_class_id),
        "test": dataset_summary(test_items, args.target_class_id),
    }
    if datasets["val"]["instances"] == 0 or datasets["test"]["instances"] == 0:
        raise ValueError("Both val and test must contain at least one target-class ground-truth instance")
    if args.dry_run:
        return {"dry_run": True, "datasets": datasets, "weights_loaded": False}

    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    model_settings = {
        "repo_id": "facebook/sam3",
        "checkpoint": checkpoint,
        "device": args.device,
        "resolution": args.resolution,
        "amp_dtype": args.amp_dtype,
        "confidence_threshold": args.candidate_floor,
    }
    model_load_started = time.perf_counter()
    model, processor, torch_module, model_info = sam3_deploy._load_model(model_settings)
    model_load_seconds = time.perf_counter() - model_load_started
    checkpoint_path = checkpoint or (
        Path(cached) if (cached := sam3_deploy.cached_checkpoint("facebook/sam3")) else None
    )
    model_info["checkpoint_sha256"] = file_sha256(checkpoint_path) if checkpoint_path else None
    output_dir = prepare_output(args.output_dir, args.allow_existing_output)

    started_at = datetime.now(timezone.utc)
    inference_started = time.perf_counter()
    val_records = infer_split("val", val_items, processor, torch_module, args)
    val_curve = threshold_curve(val_records, args.match_iou)
    val_best = choose_best_f1(val_curve)
    val_p95r90 = choose_p95_r90(val_curve)
    val_p95max = choose_p95_max_recall(val_curve)

    test_records = infer_split("test", test_items, processor, torch_module, args)
    inference_seconds = time.perf_counter() - inference_started
    test_curve = threshold_curve(test_records, args.match_iou)
    test_oracle = choose_best_f1(test_curve)

    test_at_val_best = fixed_metrics(test_records, val_best["threshold"], args.match_iou)
    test_at_fixed = fixed_metrics(test_records, args.fixed_threshold, args.match_iou)
    test_at_val_p95r90 = (
        fixed_metrics(test_records, val_p95r90["threshold"], args.match_iou) if val_p95r90 else None
    )
    test_at_val_p95max = (
        fixed_metrics(test_records, val_p95max["threshold"], args.match_iou) if val_p95max else None
    )
    val_coco = coco_metrics(val_records)
    test_coco = coco_metrics(test_records)
    cuda_memory = None
    if args.device == "cuda":
        cuda_memory = {
            "peak_allocated_mib": round(torch_module.cuda.max_memory_allocated() / 1024**2, 1),
            "peak_reserved_mib": round(torch_module.cuda.max_memory_reserved() / 1024**2, 1),
        }

    summary: dict[str, Any] = {
        "status": "complete",
        "created_at_utc": started_at.isoformat(),
        "model": model_info,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "settings": {
            "prompt": args.prompt,
            "target_class_id": args.target_class_id,
            "device": args.device,
            "resolution": args.resolution,
            "max_image_side": args.max_image_side,
            "candidate_floor": args.candidate_floor,
            "min_area_ratio": args.min_area_ratio,
            "max_detections": args.max_detections,
            "match_iou": args.match_iou,
            "amp_dtype": args.amp_dtype,
            "fixed_threshold": args.fixed_threshold,
            "model_kind": args.model_kind,
            "model_label": args.model_label,
            "matching_method": "maximum-cardinality bipartite matching",
            "ap_method": "pycocotools COCOeval segm, maxDets=100",
        },
        "inputs": {
            "val_images": str(args.val_images.expanduser().resolve()),
            "val_labels": str(args.val_labels.expanduser().resolve()),
            "test_images": str(args.test_images.expanduser().resolve()),
            "test_labels": str(args.test_labels.expanduser().resolve()),
            "val_coco": str(args.val_coco.expanduser().resolve()) if args.val_coco else None,
            "test_coco": str(args.test_coco.expanduser().resolve()) if args.test_coco else None,
            "val_dataset_sha256": dataset_fingerprint(val_items),
            "test_dataset_sha256": dataset_fingerprint(test_items),
        },
        "datasets": datasets,
        "val_coco_mask_metrics": val_coco,
        "test_coco_mask_metrics": test_coco,
        "policies": {
            "val_best_f1": val_best,
            "test_at_val_best_f1": test_at_val_best,
            "val_p95_r90": val_p95r90,
            "test_at_val_p95_r90": test_at_val_p95r90,
            "val_p95_max_recall": val_p95max,
            "test_at_val_p95_max_recall": test_at_val_p95max,
            "test_at_fixed_threshold": test_at_fixed,
        },
        "test_by_source_at_val_best_f1": per_source_metrics(
            test_records, val_best["threshold"], args.match_iou
        ),
        "diagnostics": {
            "test_oracle_best_f1": test_oracle,
            "test_oracle_is_not_a_deployment_policy": True,
            "candidate_pool_truncated_below_floor": args.candidate_floor > 0,
        },
        "acceptance": {
            "test_p90_r90": test_at_val_best["precision"] >= 0.90 and test_at_val_best["recall"] >= 0.90,
            "test_p95_r90": test_at_val_best["precision"] >= 0.95 and test_at_val_best["recall"] >= 0.90,
        },
        "runtime": {
            "cuda_memory": cuda_memory,
            "val_average_seconds_per_image": float(np.mean([row.elapsed_seconds for row in val_records])),
            "test_average_seconds_per_image": float(np.mean([row.elapsed_seconds for row in test_records])),
        },
        "limitations": [
            (
                f"The test split contains {datasets['test']['background_images']} target-free images, but no dedicated "
                "on-site negative/background benchmark."
            ),
            (
                "This benchmark measures watermelon only; it does not verify whether fine-tuning "
                "preserved zero-shot accuracy for other fruit classes."
                if args.model_kind == "finetuned"
                else "This benchmark measures watermelon only, not zero-shot accuracy for other fruit classes."
            ),
            "Resolution/prompt/threshold changes must be selected on val before another frozen test evaluation.",
        ],
    }

    cache = {"val": raw_cache(val_records), "test": raw_cache(test_records)}
    (output_dir / "raw_score_iou_cache.json").write_text(
        json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "val_threshold_curve.csv", val_curve)
    write_csv(output_dir / "test_threshold_curve_oracle_only.csv", test_curve)
    image_rows = per_image_rows("val", val_records, val_best["threshold"], args.match_iou)
    image_rows += per_image_rows("test", test_records, val_best["threshold"], args.match_iou)
    write_csv(output_dir / "per_image_metrics_at_val_threshold.csv", image_rows)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "SAM3_WATERMELON_EVALUATION.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )

    del processor, model
    if args.device == "cuda":
        torch_module.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    try:
        result = run_benchmark(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        if args.debug:
            raise
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
