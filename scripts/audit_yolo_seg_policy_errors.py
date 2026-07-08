#!/usr/bin/env python
"""Audit FP/FN instances for a concrete YOLO segmentation deployment policy."""

from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--pred-labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument("--exclude-edge-gt", action="store_true")
    parser.add_argument("--edge-margin-ratio", type=float, default=0.0)
    parser.add_argument(
        "--ignore-excluded-gt-overlap",
        action="store_true",
        help="Do not count predictions matching excluded GT as false positives.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--max-overlays", type=int, default=50)
    return parser.parse_args()


def read_polygons(path: Path, has_confidence: bool) -> list[Instance]:
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
        instances.append(Instance(cls=cls, points=np.clip(points, 0.0, 1.0), confidence=confidence))
    return instances


def mask_from_points(points: np.ndarray, raster_size: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= raster_size - 1
    scaled[:, 1] *= raster_size - 1
    polygon = np.rint(scaled).astype(np.int32)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return float(intersection / union) if union else 0.0


def area_ratio(mask: np.ndarray) -> float:
    return float(mask.sum() / mask.size)


def bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def touches_edge(points: np.ndarray, margin: float) -> bool:
    min_x, min_y, max_x, max_y = bbox(points)
    return bool(min_x <= margin or min_y <= margin or max_x >= 1.0 - margin or max_y >= 1.0 - margin)


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


def best_iou_for_gt(gt_idx: int, pred_masks: list[np.ndarray], ious: list[list[float]]) -> tuple[int, float]:
    best_pred = -1
    best_iou = 0.0
    for pred_idx in range(len(pred_masks)):
        score = ious[pred_idx][gt_idx] if gt_idx < len(ious[pred_idx]) else 0.0
        if score > best_iou:
            best_iou = score
            best_pred = pred_idx
    return best_pred, best_iou


def best_iou_for_pred(pred_idx: int, gt_masks: list[np.ndarray], ious: list[list[float]]) -> tuple[int, float]:
    best_gt = -1
    best_iou = 0.0
    for gt_idx in range(len(gt_masks)):
        score = ious[pred_idx][gt_idx] if gt_idx < len(ious[pred_idx]) else 0.0
        if score > best_iou:
            best_iou = score
            best_gt = gt_idx
    return best_gt, best_iou


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def render_overlay(
    image_path: Path,
    gt: list[Instance],
    selected_preds: list[tuple[int, Instance]],
    matched_gt: set[int],
    pred_to_gt: dict[int, int],
    output_dir: Path,
    threshold: float,
    min_area: float,
    excluded_gt: set[int],
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    overlay = image.copy()
    canvas = image.copy()
    for gt_idx, instance in enumerate(gt):
        if gt_idx in excluded_gt:
            draw_instance(overlay, canvas, instance.points, (0, 160, 180), (0, 220, 255), width, height)
        elif gt_idx in matched_gt:
            draw_instance(overlay, canvas, instance.points, (0, 180, 0), (0, 255, 0), width, height)
        else:
            draw_instance(overlay, canvas, instance.points, (0, 0, 255), (0, 0, 255), width, height)

    matched_pred = set(pred_to_gt)
    for original_pred_idx, instance in selected_preds:
        if original_pred_idx not in matched_pred:
            draw_instance(overlay, canvas, instance.points, (180, 0, 180), (255, 0, 255), width, height)

    blended = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    visible_gt = [idx for idx in range(len(gt)) if idx not in excluded_gt]
    fp = len(selected_preds) - len(matched_pred)
    fn = len(visible_gt) - len(matched_gt)
    label = (
        f"conf>={threshold:g} area>={min_area:g} GT={len(visible_gt)} "
        f"TP={len(matched_gt)} FP={fp} FN={fn} green=TP red=FN purple=FP yellow=excluded"
    )
    cv2.rectangle(blended, (0, 0), (min(width, 1250), 34), (0, 0, 0), thickness=-1)
    cv2.putText(blended, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_policy_errors.png"
    cv2.imwrite(str(output_path), blended)
    return output_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = args.output_dir / "overlays"
    per_image_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    overlay_paths: list[Path] = []

    total_tp = total_fp = total_fn = total_gt = total_pred = 0
    total_excluded_gt = 0
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        gt = read_polygons(args.labels_dir / f"{image_path.stem}.txt", has_confidence=False)
        preds = read_polygons(args.pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True)
        gt_masks = [mask_from_points(instance.points, args.raster_size) for instance in gt]
        pred_masks = [mask_from_points(instance.points, args.raster_size) for instance in preds]
        gt_areas = [area_ratio(mask) for mask in gt_masks]
        pred_areas = [area_ratio(mask) for mask in pred_masks]
        ious = [[mask_iou(pred_mask, gt_mask) for gt_mask in gt_masks] for pred_mask in pred_masks]

        selected_pred_indices = [
            idx
            for idx, pred in enumerate(preds)
            if pred.confidence is not None
            and pred.confidence >= args.threshold
            and args.min_area_ratio <= pred_areas[idx] <= args.max_area_ratio
        ]
        selected_pred_indices.sort(key=lambda idx: preds[idx].confidence or 0.0, reverse=True)
        target_gt_indices = [
            gt_idx
            for gt_idx, instance in enumerate(gt)
            if not (args.exclude_edge_gt and touches_edge(instance.points, args.edge_margin_ratio))
        ]
        excluded_gt = set(range(len(gt))) - set(target_gt_indices)

        matched_gt: set[int] = set()
        pred_to_gt: dict[int, int] = {}
        pred_match_iou: dict[int, float] = {}
        for pred_idx in selected_pred_indices:
            best_gt = -1
            best_score = 0.0
            for gt_idx in target_gt_indices:
                if gt_idx in matched_gt:
                    continue
                score = ious[pred_idx][gt_idx] if gt_idx < len(ious[pred_idx]) else 0.0
                if score > best_score:
                    best_score = score
                    best_gt = gt_idx
            if best_gt >= 0 and best_score >= args.iou_threshold:
                matched_gt.add(best_gt)
                pred_to_gt[pred_idx] = best_gt
                pred_match_iou[pred_idx] = best_score

        fp = 0
        for pred_idx in selected_pred_indices:
            if pred_idx in pred_to_gt:
                continue
            fp += 1
            best_gt, best_score = best_iou_for_pred(pred_idx, gt_masks, ious)
            matched_excluded = best_gt in excluded_gt and best_score >= args.iou_threshold
            if matched_excluded and args.ignore_excluded_gt_overlap:
                fp -= 1
                continue
            min_x, min_y, max_x, max_y = bbox(preds[pred_idx].points)
            error_rows.append(
                {
                    "type": "FP",
                    "image": image_path.name,
                    "source": source_hint(image_path.name),
                    "gt_index": "",
                    "pred_index": pred_idx,
                    "confidence": f"{preds[pred_idx].confidence or 0.0:.6f}",
                    "area_ratio": f"{pred_areas[pred_idx]:.6f}",
                    "best_iou": f"{best_score:.6f}",
                    "best_gt_or_pred": best_gt,
                    "matched_excluded_gt": int(matched_excluded),
                    "bbox_min_x": f"{min_x:.6f}",
                    "bbox_min_y": f"{min_y:.6f}",
                    "bbox_max_x": f"{max_x:.6f}",
                    "bbox_max_y": f"{max_y:.6f}",
                    "recommendation": "复核误报：确认是否为未标注可抓取目标、叶片/背景误检或应补充硬负样本",
                }
            )

        fn = 0
        for gt_idx in target_gt_indices:
            if gt_idx in matched_gt:
                continue
            fn += 1
            best_pred, best_score = best_iou_for_gt(gt_idx, pred_masks, ious)
            best_conf = preds[best_pred].confidence if best_pred >= 0 else 0.0
            best_area = pred_areas[best_pred] if best_pred >= 0 else 0.0
            min_x, min_y, max_x, max_y = bbox(gt[gt_idx].points)
            if best_score >= args.iou_threshold and best_conf < args.threshold:
                recommendation = "漏检来自置信度不足：补充相似样本或考虑局部降低阈值"
            elif best_score >= args.iou_threshold and best_area < args.min_area_ratio:
                recommendation = "漏检来自面积门控过滤：复核目标面积阈值或标注尺寸"
            else:
                recommendation = "真实漏检/形状不匹配：优先补充同类遮挡、密集或视角样本"
            error_rows.append(
                {
                    "type": "FN",
                    "image": image_path.name,
                    "source": source_hint(image_path.name),
                    "gt_index": gt_idx,
                    "pred_index": best_pred,
                    "confidence": f"{best_conf or 0.0:.6f}",
                    "area_ratio": f"{gt_areas[gt_idx]:.6f}",
                    "best_iou": f"{best_score:.6f}",
                    "best_gt_or_pred": best_pred,
                    "matched_excluded_gt": 0,
                    "bbox_min_x": f"{min_x:.6f}",
                    "bbox_min_y": f"{min_y:.6f}",
                    "bbox_max_x": f"{max_x:.6f}",
                    "bbox_max_y": f"{max_y:.6f}",
                    "recommendation": recommendation,
                }
            )

        tp = len(matched_gt)
        selected_preds = [(idx, preds[idx]) for idx in selected_pred_indices]
        if (fp or fn) and len(overlay_paths) < args.max_overlays:
            overlay_paths.append(
                render_overlay(
                    image_path,
                    gt,
                    selected_preds,
                    matched_gt,
                    pred_to_gt,
                    overlays_dir,
                    args.threshold,
                    args.min_area_ratio,
                    excluded_gt,
                )
            )

        total_gt += len(target_gt_indices)
        total_excluded_gt += len(excluded_gt)
        total_pred += len(selected_pred_indices)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image_rows.append(
            {
                "image": image_path.name,
                "source": source_hint(image_path.name),
                "gt": len(target_gt_indices),
                "excluded_gt": len(excluded_gt),
                "pred": len(selected_pred_indices),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": f"{tp / (tp + fp):.6f}" if tp + fp else "0.000000",
                "recall": f"{tp / len(target_gt_indices):.6f}" if target_gt_indices else "0.000000",
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    write_tsv(args.output_dir / "policy_per_image.tsv", per_image_rows)
    write_tsv(args.output_dir / "policy_error_instances.tsv", error_rows)

    error_by_source: dict[str, dict[str, int]] = {}
    for row in error_rows:
        source = str(row["source"])
        error_by_source.setdefault(source, {"FP": 0, "FN": 0})
        error_by_source[source][str(row["type"])] += 1

    lines = [
        "# 可抓取候选部署策略误差审计",
        "",
        "## 策略",
        "",
        f"- `conf >= {args.threshold}`",
        f"- `predicted_mask_area_ratio >= {args.min_area_ratio}`",
        f"- IoU 阈值：`{args.iou_threshold}`",
        f"- 排除贴边 GT：`{bool(args.exclude_edge_gt)}`",
        f"- 贴边边距：`{args.edge_margin_ratio}`",
        "",
        "## 汇总",
        "",
        f"- GT 实例：{total_gt}",
        f"- 排除 GT 实例：{total_excluded_gt}",
        f"- 选中预测：{total_pred}",
        f"- TP / FP / FN：{total_tp} / {total_fp} / {total_fn}",
        f"- Precision：{precision:.6f}",
        f"- Recall：{recall:.6f}",
        f"- F1：{f1:.6f}",
        "",
        "## 按来源统计错误",
        "",
        "| source | FP | FN |",
        "| --- | ---: | ---: |",
    ]
    for source, counts in sorted(error_by_source.items()):
        lines.append(f"| {source} | {counts.get('FP', 0)} | {counts.get('FN', 0)} |")
    lines.extend(
        [
            "",
            "## 高优先级错误实例",
            "",
            "| type | image | index | confidence | area | best IoU | 建议 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in error_rows[:30]:
        index_value = row["gt_index"] if row["type"] == "FN" else row["pred_index"]
        lines.append(
            f"| {row['type']} | `{row['image']}` | {index_value} | {row['confidence']} | "
            f"{row['area_ratio']} | {row['best_iou']} | {row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "```text",
            str(args.output_dir / "policy_error_instances.tsv"),
            str(args.output_dir / "policy_per_image.tsv"),
            str(overlays_dir),
            "```",
            "",
        ]
    )
    (args.output_dir / "POLICY_ERROR_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:18]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
