#!/usr/bin/env python
"""Run two YOLO segmentation models and export graspable watermelon candidates."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan source directories for images.")
    parser.add_argument("--include-glob", default="*", help="Only include image filenames matching this glob.")
    parser.add_argument("--exclude-glob", default="", help="Exclude image filenames matching this glob.")
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-a-name", default="runs6")
    parser.add_argument("--model-b-name", default="runs13")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    parser.add_argument("--inference-conf", type=float, default=0.001)
    parser.add_argument("--inference-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--confidence-threshold", type=float, default=0.15)
    parser.add_argument("--min-area-ratio", type=float, default=0.02)
    parser.add_argument("--min-support", type=int, choices=(1, 2), default=1)
    parser.add_argument("--support-iou-threshold", type=float, default=0.1)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--save-empty-labels", action="store_true", default=True)
    parser.add_argument("--save-jsonl", action="store_true", default=True)
    parser.add_argument("--save-overlays", action="store_true", help="Render candidate overlays for visual QA.")
    return parser.parse_args()


def image_paths(source: Path, recursive: bool, include_glob: str, exclude_glob: str) -> list[Path]:
    def is_selected(path: Path) -> bool:
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return False
        if include_glob and not fnmatch.fnmatch(path.name, include_glob):
            return False
        if exclude_glob and fnmatch.fnmatch(path.name, exclude_glob):
            return False
        return True

    if source.is_file():
        if not is_selected(source):
            raise ValueError(f"Unsupported image suffix: {source}")
        return [source.resolve()]
    if not source.is_dir():
        raise FileNotFoundError(source)
    iterator = source.rglob("*") if recursive else source.iterdir()
    return sorted(path.resolve() for path in iterator if path.is_file() and is_selected(path))


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


def preds_from_result(result: object, model_name: str, raster_size: int) -> list[Pred]:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return []
    points_rows = getattr(masks, "xyn", [])
    confidences = boxes.conf.detach().cpu().numpy().tolist()
    preds: list[Pred] = []
    for points_raw, confidence in zip(points_rows, confidences):
        points = np.asarray(points_raw, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        points = np.clip(points, 0.0, 1.0)
        mask = mask_from_points(points, raster_size)
        area = int(mask.sum())
        preds.append(
            Pred(
                model_name=model_name,
                confidence=float(confidence),
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


def apply_policy(pred_a: list[Pred], pred_b: list[Pred], args: argparse.Namespace) -> tuple[list[Pred], dict[str, int]]:
    selected_a = [
        pred for pred in pred_a if pred.confidence >= args.confidence_threshold and pred.area_ratio >= args.min_area_ratio
    ]
    selected_b = [
        pred for pred in pred_b if pred.confidence >= args.confidence_threshold and pred.area_ratio >= args.min_area_ratio
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
    stats = {
        "selected_a": len(selected_a),
        "selected_b": len(selected_b),
        "candidates": len(candidates),
        "kept": len(kept),
    }
    return kept, stats


def format_label(pred: Pred) -> str:
    coords = " ".join(f"{value:.6f}" for point in pred.points for value in point)
    return f"0 {coords} {pred.confidence:.6f}"


def polygon_centroid(points: np.ndarray) -> tuple[float, float]:
    x = points[:, 0]
    y = points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area_twice = float(cross.sum())
    if abs(area_twice) < 1e-9:
        return float(x.mean()), float(y.mean())
    cx = float(((x + np.roll(x, -1)) * cross).sum() / (3.0 * area_twice))
    cy = float(((y + np.roll(y, -1)) * cross).sum() / (3.0 * area_twice))
    return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))


def pixel_bbox(pred: Pred, width: int, height: int) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = pred.bbox
    return (
        int(round(min_x * (width - 1))),
        int(round(min_y * (height - 1))),
        int(round(max_x * (width - 1))),
        int(round(max_y * (height - 1))),
    )


def candidate_flat_row(image_path: Path, pred_idx: int, pred: Pred, width: int, height: int) -> dict[str, object]:
    min_x, min_y, max_x, max_y = pred.bbox
    cx, cy = polygon_centroid(pred.points)
    bbox_min_x_px, bbox_min_y_px, bbox_max_x_px, bbox_max_y_px = pixel_bbox(pred, width, height)
    return {
        "image": image_path.name,
        "candidate_index": pred_idx,
        "model": pred.model_name,
        "confidence": f"{pred.confidence:.6f}",
        "area_ratio": f"{pred.area_ratio:.6f}",
        "centroid_x": f"{cx:.6f}",
        "centroid_y": f"{cy:.6f}",
        "centroid_x_px": int(round(cx * (width - 1))),
        "centroid_y_px": int(round(cy * (height - 1))),
        "bbox_min_x": f"{min_x:.6f}",
        "bbox_min_y": f"{min_y:.6f}",
        "bbox_max_x": f"{max_x:.6f}",
        "bbox_max_y": f"{max_y:.6f}",
        "bbox_min_x_px": bbox_min_x_px,
        "bbox_min_y_px": bbox_min_y_px,
        "bbox_max_x_px": bbox_max_x_px,
        "bbox_max_y_px": bbox_max_y_px,
        "polygon_points": len(pred.points),
    }


def json_record(image_path: Path, pred_idx: int, pred: Pred, width: int, height: int) -> dict[str, object]:
    min_x, min_y, max_x, max_y = pred.bbox
    cx, cy = polygon_centroid(pred.points)
    bbox_min_x_px, bbox_min_y_px, bbox_max_x_px, bbox_max_y_px = pixel_bbox(pred, width, height)
    return {
        "image": image_path.name,
        "candidate_index": pred_idx,
        "model": pred.model_name,
        "confidence": pred.confidence,
        "area_ratio": pred.area_ratio,
        "centroid": {
            "x": cx,
            "y": cy,
            "x_px": int(round(cx * (width - 1))),
            "y_px": int(round(cy * (height - 1))),
        },
        "bbox": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "min_x_px": bbox_min_x_px,
            "min_y_px": bbox_min_y_px,
            "max_x_px": bbox_max_x_px,
            "max_y_px": bbox_max_y_px,
        },
        "polygon": [[float(x), float(y)] for x, y in pred.points.tolist()],
    }


def scale_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= width - 1
    scaled[:, 1] *= height - 1
    return np.rint(scaled).astype(np.int32)


def render_overlay(image_path: Path, kept: list[Pred], output_dir: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    overlay = image.copy()
    canvas = image.copy()
    colors = {
        "runs6": ((0, 180, 0), (0, 255, 0)),
        "runs13": ((180, 0, 180), (255, 0, 255)),
    }
    for idx, pred in enumerate(kept):
        fill_color, line_color = colors.get(pred.model_name, ((180, 120, 0), (255, 180, 0)))
        polygon = scale_points(pred.points, width, height)
        cv2.fillPoly(overlay, [polygon], fill_color)
        cv2.polylines(canvas, [polygon], isClosed=True, color=line_color, thickness=2, lineType=cv2.LINE_AA)
        cx, cy = polygon_centroid(pred.points)
        center = (int(round(cx * (width - 1))), int(round(cy * (height - 1))))
        cv2.circle(canvas, center, 5, line_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{idx}:{pred.confidence:.2f}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_color,
            2,
            cv2.LINE_AA,
        )
    blended = cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0)
    cv2.rectangle(blended, (0, 0), (min(width, 900), 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        blended,
        f"graspable candidates={len(kept)} green=runs6 magenta=runs13",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_candidates.jpg"), blended)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    labels_dir = output_dir / "labels"
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = image_paths(args.source, args.recursive, args.include_glob, args.exclude_glob)
    model_a = YOLO(str(args.model_a))
    model_b = YOLO(str(args.model_b))

    manifest_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    jsonl_rows: list[dict[str, object]] = []
    totals = {
        "raw_a": 0,
        "raw_b": 0,
        "selected_a": 0,
        "selected_b": 0,
        "candidates": 0,
        "kept": 0,
    }
    kept_by_model = {args.model_a_name: 0, args.model_b_name: 0}

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        result_a = model_a.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.inference_conf,
            iou=args.inference_iou,
            max_det=args.max_det,
            device=args.device,
            save=False,
            verbose=False,
        )[0]
        result_b = model_b.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.inference_conf,
            iou=args.inference_iou,
            max_det=args.max_det,
            device=args.device,
            save=False,
            verbose=False,
        )[0]
        pred_a = preds_from_result(result_a, args.model_a_name, args.raster_size)
        pred_b = preds_from_result(result_b, args.model_b_name, args.raster_size)
        kept, stats = apply_policy(pred_a, pred_b, args)

        label_text = "\n".join(format_label(pred) for pred in kept)
        if label_text:
            label_text += "\n"
        if label_text or args.save_empty_labels:
            (labels_dir / f"{image_path.stem}.txt").write_text(label_text, encoding="utf-8")

        kept_a = sum(1 for pred in kept if pred.model_name == args.model_a_name)
        kept_b = sum(1 for pred in kept if pred.model_name == args.model_b_name)
        kept_by_model[args.model_a_name] += kept_a
        kept_by_model[args.model_b_name] += kept_b
        totals["raw_a"] += len(pred_a)
        totals["raw_b"] += len(pred_b)
        for key in ("selected_a", "selected_b", "candidates", "kept"):
            totals[key] += stats[key]
        manifest_rows.append(
            {
                "image": image_path.name,
                "raw_a": len(pred_a),
                "raw_b": len(pred_b),
                "selected_a": stats["selected_a"],
                "selected_b": stats["selected_b"],
                "candidates": stats["candidates"],
                "kept": stats["kept"],
                f"kept_{args.model_a_name}": kept_a,
                f"kept_{args.model_b_name}": kept_b,
            }
        )
        for pred_idx, pred in enumerate(kept):
            candidate_rows.append(candidate_flat_row(image_path, pred_idx, pred, width, height))
            jsonl_rows.append(json_record(image_path, pred_idx, pred, width, height))
        if args.save_overlays:
            render_overlay(image_path, kept, overlays_dir)

    write_tsv(output_dir / "prediction_manifest.tsv", manifest_rows)
    write_csv(output_dir / "robot_candidates.csv", candidate_rows)
    if args.save_jsonl:
        with (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": args.source.as_posix(),
        "recursive": args.recursive,
        "include_glob": args.include_glob,
        "exclude_glob": args.exclude_glob,
        "model_a": args.model_a.as_posix(),
        "model_b": args.model_b.as_posix(),
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "images": len(images),
        "imgsz": args.imgsz,
        "device": args.device,
        "inference_conf": args.inference_conf,
        "inference_iou": args.inference_iou,
        "max_det": args.max_det,
        "confidence_threshold": args.confidence_threshold,
        "min_area_ratio": args.min_area_ratio,
        "min_support": args.min_support,
        "support_iou_threshold": args.support_iou_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "raster_size": args.raster_size,
        "totals": totals,
        "kept_by_model": kept_by_model,
        "files": {
            "labels": labels_dir.as_posix(),
            "manifest": (output_dir / "prediction_manifest.tsv").as_posix(),
            "robot_candidates_csv": (output_dir / "robot_candidates.csv").as_posix(),
            "candidates_jsonl": (output_dir / "candidates.jsonl").as_posix(),
            "overlays": overlays_dir.as_posix() if args.save_overlays else None,
            "summary": (output_dir / "prediction_summary.json").as_posix(),
        },
    }
    (output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
