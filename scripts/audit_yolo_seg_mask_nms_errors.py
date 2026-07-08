#!/usr/bin/env python
"""Audit FP/FN instances for a fixed YOLO segmentation mask-NMS policy."""

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
    points: np.ndarray
    confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--pred-labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--nms-iou-threshold", type=float, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--max-overlays", type=int, default=80)
    return parser.parse_args()


def source_hint(image_name: str) -> str:
    if image_name.startswith("local_test_images"):
        return "local_test_images"
    if image_name.startswith("my_labelme"):
        return "my_labelme"
    if image_name.startswith("roboflow_team128"):
        return "roboflow_team128_v5"
    if image_name.startswith("roboflow_drago"):
        return "roboflow_drago_v1"
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


def mask_nms(selected: list[int], pred_pred_ious: list[list[float]], preds: list[Instance], nms_iou: float) -> list[int]:
    ordered = sorted(selected, key=lambda idx: float(preds[idx].confidence or 0.0), reverse=True)
    kept: list[int] = []
    for pred_idx in ordered:
        if any(pred_pred_ious[pred_idx][kept_idx] > nms_iou for kept_idx in kept):
            continue
        kept.append(pred_idx)
    return kept


def bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def best_iou_for_pred(pred_idx: int, pred_gt_ious: list[list[float]]) -> tuple[int, float]:
    best_gt = -1
    best_iou = 0.0
    for gt_idx, score in enumerate(pred_gt_ious[pred_idx] if pred_idx < len(pred_gt_ious) else []):
        if score > best_iou:
            best_gt = gt_idx
            best_iou = score
    return best_gt, best_iou


def best_iou_for_gt(gt_idx: int, pred_gt_ious: list[list[float]]) -> tuple[int, float]:
    best_pred = -1
    best_iou = 0.0
    for pred_idx, row in enumerate(pred_gt_ious):
        score = row[gt_idx] if gt_idx < len(row) else 0.0
        if score > best_iou:
            best_pred = pred_idx
            best_iou = score
    return best_pred, best_iou


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


def render_overlay(
    image_path: Path,
    gt: list[Instance],
    preds: list[Instance],
    kept: list[int],
    matched_gt: set[int],
    pred_to_gt: dict[int, int],
    output_dir: Path,
    label: str,
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    overlay = image.copy()
    canvas = image.copy()

    for gt_idx, instance in enumerate(gt):
        if gt_idx in matched_gt:
            draw_instance(overlay, canvas, instance.points, (0, 180, 0), (0, 255, 0), width, height)
        else:
            draw_instance(overlay, canvas, instance.points, (0, 0, 255), (0, 0, 255), width, height)

    matched_pred = set(pred_to_gt)
    for pred_idx in kept:
        if pred_idx not in matched_pred:
            draw_instance(overlay, canvas, preds[pred_idx].points, (180, 0, 180), (255, 0, 255), width, height)

    blended = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    cv2.rectangle(blended, (0, 0), (min(width, 1400), 36), (0, 0, 0), thickness=-1)
    cv2.putText(blended, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_mask_nms_errors.png"
    cv2.imwrite(str(output_path), blended)
    return output_path


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_image(image_path: Path, args: argparse.Namespace, overlays_dir: Path, should_render: bool) -> tuple[dict[str, object], list[dict[str, object]], Path | None]:
    gt = read_polygons(args.labels_dir / f"{image_path.stem}.txt", has_confidence=False)
    preds = read_polygons(args.pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True)
    gt_masks = [mask_from_points(instance.points, args.raster_size) for instance in gt]
    pred_masks = [mask_from_points(instance.points, args.raster_size) for instance in preds]
    gt_areas = [int(mask.sum()) for mask in gt_masks]
    pred_areas = [int(mask.sum()) for mask in pred_masks]
    pred_area_ratios = [area / float(args.raster_size * args.raster_size) for area in pred_areas]
    pred_gt_ious = pairwise_iou_matrix(pred_masks, pred_areas, gt_masks, gt_areas)
    pred_pred_ious = pairwise_iou_matrix(pred_masks, pred_areas, pred_masks, pred_areas)

    selected = [
        idx
        for idx, pred in enumerate(preds)
        if pred.confidence is not None
        and pred.confidence >= args.confidence_threshold
        and pred_area_ratios[idx] >= args.min_area_ratio
        and pred_areas[idx] > 0
    ]
    kept = mask_nms(selected, pred_pred_ious, preds, args.nms_iou_threshold)

    matched_gt: set[int] = set()
    pred_to_gt: dict[int, int] = {}
    pred_match_iou: dict[int, float] = {}
    for pred_idx in kept:
        best_gt = -1
        best_iou = 0.0
        for gt_idx, score in enumerate(pred_gt_ious[pred_idx] if pred_idx < len(pred_gt_ious) else []):
            if gt_idx in matched_gt:
                continue
            if score > best_iou:
                best_gt = gt_idx
                best_iou = score
        if best_gt >= 0 and best_iou >= args.match_iou_threshold:
            matched_gt.add(best_gt)
            pred_to_gt[pred_idx] = best_gt
            pred_match_iou[pred_idx] = best_iou

    rows: list[dict[str, object]] = []
    source = source_hint(image_path.name)
    for pred_idx in kept:
        if pred_idx in pred_to_gt:
            continue
        best_gt, best_iou = best_iou_for_pred(pred_idx, pred_gt_ious)
        min_x, min_y, max_x, max_y = bbox(preds[pred_idx].points)
        rows.append(
            {
                "type": "FP",
                "image": image_path.name,
                "source": source,
                "gt_index": "",
                "pred_index": pred_idx,
                "confidence": f"{float(preds[pred_idx].confidence or 0.0):.6f}",
                "area_ratio": f"{pred_area_ratios[pred_idx]:.6f}",
                "best_iou": f"{best_iou:.6f}",
                "best_gt_or_pred": best_gt,
                "bbox_min_x": f"{min_x:.6f}",
                "bbox_min_y": f"{min_y:.6f}",
                "bbox_max_x": f"{max_x:.6f}",
                "bbox_max_y": f"{max_y:.6f}",
            }
        )

    for gt_idx in range(len(gt)):
        if gt_idx in matched_gt:
            continue
        best_pred, best_iou = best_iou_for_gt(gt_idx, pred_gt_ious)
        confidence = float(preds[best_pred].confidence or 0.0) if best_pred >= 0 else 0.0
        area_ratio = gt_areas[gt_idx] / float(args.raster_size * args.raster_size) if gt_idx < len(gt_areas) else 0.0
        min_x, min_y, max_x, max_y = bbox(gt[gt_idx].points)
        rows.append(
            {
                "type": "FN",
                "image": image_path.name,
                "source": source,
                "gt_index": gt_idx,
                "pred_index": best_pred,
                "confidence": f"{confidence:.6f}",
                "area_ratio": f"{area_ratio:.6f}",
                "best_iou": f"{best_iou:.6f}",
                "best_gt_or_pred": best_pred,
                "bbox_min_x": f"{min_x:.6f}",
                "bbox_min_y": f"{min_y:.6f}",
                "bbox_max_x": f"{max_x:.6f}",
                "bbox_max_y": f"{max_y:.6f}",
            }
        )

    tp = len(matched_gt)
    fp = len([row for row in rows if row["type"] == "FP"])
    fn = len([row for row in rows if row["type"] == "FN"])
    per_image = {
        "image": image_path.name,
        "source": source,
        "gt": len(gt),
        "selected_predictions": len(selected),
        "kept_predictions": len(kept),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": f"{tp / (tp + fp):.6f}" if tp + fp else "0.000000",
        "recall": f"{tp / len(gt):.6f}" if gt else "0.000000",
    }

    overlay_path = None
    if should_render and (fp or fn):
        label = (
            f"conf>={args.confidence_threshold:g} nms<={args.nms_iou_threshold:g} "
            f"area>={args.min_area_ratio:g} GT={len(gt)} TP={tp} FP={fp} FN={fn} "
            "green=TP red=FN purple=FP"
        )
        overlay_path = render_overlay(image_path, gt, preds, kept, matched_gt, pred_to_gt, overlays_dir, label)
    return per_image, rows, overlay_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = args.output_dir / "overlays"
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    per_image_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    for image_path in image_paths:
        per_image, rows, overlay_path = evaluate_image(
            image_path,
            args,
            overlays_dir,
            len(overlay_paths) < args.max_overlays,
        )
        per_image_rows.append(per_image)
        error_rows.extend(rows)
        if overlay_path is not None:
            overlay_paths.append(overlay_path)

    totals = {"gt": 0, "selected_predictions": 0, "kept_predictions": 0, "tp": 0, "fp": 0, "fn": 0}
    by_source: dict[str, dict[str, int]] = {}
    for row in per_image_rows:
        source = str(row["source"])
        source_totals = by_source.setdefault(source, {"images": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0})
        source_totals["images"] += 1
        for key in totals:
            totals[key] += int(row[key])
        for key in ("gt", "tp", "fp", "fn"):
            source_totals[key] += int(row[key])

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    for source_totals in by_source.values():
        source_totals["precision_ppm"] = int(round(1_000_000 * source_totals["tp"] / max(source_totals["tp"] + source_totals["fp"], 1)))
        source_totals["recall_ppm"] = int(round(1_000_000 * source_totals["tp"] / max(source_totals["tp"] + source_totals["fn"], 1)))

    summary = {
        "inputs": {
            "images_dir": str(args.images_dir),
            "labels_dir": str(args.labels_dir),
            "pred_labels_dir": str(args.pred_labels_dir),
            "confidence_threshold": args.confidence_threshold,
            "nms_iou_threshold": args.nms_iou_threshold,
            "min_area_ratio": args.min_area_ratio,
            "match_iou_threshold": args.match_iou_threshold,
            "raster_size": args.raster_size,
        },
        "totals": {
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "by_source": by_source,
        "files": {
            "per_image": str(args.output_dir / "mask_nms_policy_per_image.tsv"),
            "errors": str(args.output_dir / "mask_nms_error_instances.tsv"),
            "overlays": str(overlays_dir),
        },
    }
    write_tsv(args.output_dir / "mask_nms_policy_per_image.tsv", per_image_rows)
    write_tsv(args.output_dir / "mask_nms_error_instances.tsv", error_rows)
    (args.output_dir / "mask_nms_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# mask-NMS 错误审计",
        "",
        "## 策略",
        "",
        f"- confidence_threshold: `{args.confidence_threshold}`",
        f"- nms_iou_threshold: `{args.nms_iou_threshold}`",
        f"- min_area_ratio: `{args.min_area_ratio}`",
        f"- match_iou_threshold: `{args.match_iou_threshold}`",
        "",
        "## 汇总",
        "",
        f"- TP / FP / FN: {totals['tp']} / {totals['fp']} / {totals['fn']}",
        f"- Precision: {precision:.6f}",
        f"- Recall: {recall:.6f}",
        f"- F1: {f1:.6f}",
        "",
        "## 按来源",
        "",
        "| source | images | GT | TP | FP | FN | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, values in sorted(by_source.items()):
        src_precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        src_recall = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 0.0
        lines.append(
            f"| {source} | {values['images']} | {values['gt']} | {values['tp']} | {values['fp']} | {values['fn']} | "
            f"{src_precision:.4f} | {src_recall:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "```text",
            str(args.output_dir / "mask_nms_error_summary.json"),
            str(args.output_dir / "mask_nms_policy_per_image.tsv"),
            str(args.output_dir / "mask_nms_error_instances.tsv"),
            str(overlays_dir),
            "```",
            "",
        ]
    )
    (args.output_dir / "MASK_NMS_ERROR_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"totals": summary["totals"], "by_source": by_source}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
