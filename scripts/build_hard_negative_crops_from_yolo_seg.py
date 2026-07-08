#!/usr/bin/env python
"""Mine background crops from false-positive YOLO segmentation predictions."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--images-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--pred-labels-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.075)
    parser.add_argument("--max-best-gt-iou", type=float, default=0.1)
    parser.add_argument("--max-gt-crop-overlap", type=float, default=0.01)
    parser.add_argument("--min-pred-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-pred-area-ratio", type=float, default=0.08)
    parser.add_argument("--crop-expand", type=float, default=1.8)
    parser.add_argument("--min-crop-size", type=int, default=96)
    parser.add_argument("--max-crop-size", type=int, default=384)
    parser.add_argument("--max-crops-per-image", type=int, default=3)
    parser.add_argument("--max-crops-total", type=int, default=900)
    parser.add_argument("--raster-size", type=int, default=512)
    parser.add_argument(
        "--include-sources",
        default="",
        help="Comma-separated source names to include, e.g. roboflow_team128_v5. Empty means all sources.",
    )
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


def label_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    relative = image_path.resolve().relative_to(dataset_root.resolve())
    parts = list(relative.parts)
    if not parts or parts[0] != "images":
        raise ValueError(f"Expected image under images/: {image_path}")
    parts[0] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")


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


def bbox_pixels(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    min_x = int(np.floor(float(points[:, 0].min()) * (width - 1)))
    min_y = int(np.floor(float(points[:, 1].min()) * (height - 1)))
    max_x = int(np.ceil(float(points[:, 0].max()) * (width - 1)))
    max_y = int(np.ceil(float(points[:, 1].max()) * (height - 1)))
    return min_x, min_y, max_x, max_y


def expanded_square_crop(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    expand: float,
    min_size: int,
    max_size: int,
) -> tuple[int, int, int, int] | None:
    min_x, min_y, max_x, max_y = bbox
    box_w = max(1, max_x - min_x + 1)
    box_h = max(1, max_y - min_y + 1)
    side = int(round(max(box_w, box_h) * expand))
    side = max(min_size, min(max_size, side))
    if side > width or side > height:
        side = min(width, height)
    if side < min_size:
        return None
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    x1 = max(0, min(width - side, center_x - side // 2))
    y1 = max(0, min(height - side, center_y - side // 2))
    x2 = x1 + side
    y2 = y1 + side
    return x1, y1, x2, y2


def crop_gt_overlap(gt: list[Instance], crop: tuple[int, int, int, int], width: int, height: int, raster_size: int) -> float:
    x1, y1, x2, y2 = crop
    crop_mask = np.zeros((raster_size, raster_size), dtype=np.uint8)
    rx1 = int(round(x1 / width * raster_size))
    ry1 = int(round(y1 / height * raster_size))
    rx2 = int(round(x2 / width * raster_size))
    ry2 = int(round(y2 / height * raster_size))
    crop_mask[max(0, ry1) : min(raster_size, ry2), max(0, rx1) : min(raster_size, rx2)] = 1
    crop_area = int(crop_mask.sum())
    if crop_area <= 0:
        return 1.0
    max_overlap = 0.0
    for instance in gt:
        gt_mask = mask_from_points(instance.points, raster_size)
        overlap = int(np.logical_and(gt_mask, crop_mask).sum()) / crop_area
        max_overlap = max(max_overlap, overlap)
    return max_overlap


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
    if len(args.images_dirs) != len(args.pred_labels_dirs):
        raise ValueError("--images-dirs and --pred-labels-dirs must have the same length")
    include_sources = {item.strip() for item in args.include_sources.split(",") if item.strip()}

    images_out = args.output_dir / "images" / "train"
    labels_out = args.output_dir / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    image_entries: list[str] = []
    selected_total = 0
    candidates_total = 0
    skipped_gt_overlap = 0
    skipped_existing_match = 0

    for images_dir, pred_labels_dir in zip(args.images_dirs, args.pred_labels_dirs):
        image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        split = images_dir.name
        for image_path in image_paths:
            source = source_hint(image_path.name)
            if include_sources and source not in include_sources:
                continue
            if selected_total >= args.max_crops_total:
                break
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            gt = read_polygons(label_path_for_image(args.dataset_root, image_path), has_confidence=False)
            preds = read_polygons(pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True)
            gt_masks = [mask_from_points(instance.points, args.raster_size) for instance in gt]
            pred_masks = [mask_from_points(instance.points, args.raster_size) for instance in preds]
            pred_areas = [int(mask.sum()) for mask in pred_masks]
            pred_rows: list[tuple[int, Instance, float, float]] = []
            for pred_idx, pred in enumerate(preds):
                if pred.confidence is None or pred.confidence < args.min_confidence:
                    continue
                area_ratio = pred_areas[pred_idx] / float(args.raster_size * args.raster_size)
                if area_ratio < args.min_pred_area_ratio or area_ratio > args.max_pred_area_ratio:
                    continue
                best_iou = max((mask_iou(pred_masks[pred_idx], gt_mask) for gt_mask in gt_masks), default=0.0)
                candidates_total += 1
                if best_iou > args.max_best_gt_iou:
                    skipped_existing_match += 1
                    continue
                pred_rows.append((pred_idx, pred, best_iou, area_ratio))

            pred_rows.sort(key=lambda item: item[1].confidence or 0.0, reverse=True)
            kept_for_image = 0
            used_crops: list[tuple[int, int, int, int]] = []
            for pred_idx, pred, best_iou, area_ratio in pred_rows:
                if kept_for_image >= args.max_crops_per_image or selected_total >= args.max_crops_total:
                    break
                crop = expanded_square_crop(
                    bbox_pixels(pred.points, width, height),
                    width,
                    height,
                    args.crop_expand,
                    args.min_crop_size,
                    args.max_crop_size,
                )
                if crop is None:
                    continue
                if any(
                    max(0, min(crop[2], other[2]) - max(crop[0], other[0]))
                    * max(0, min(crop[3], other[3]) - max(crop[1], other[1]))
                    > 0.5 * (crop[2] - crop[0]) * (crop[3] - crop[1])
                    for other in used_crops
                ):
                    continue
                overlap = crop_gt_overlap(gt, crop, width, height, args.raster_size)
                if overlap > args.max_gt_crop_overlap:
                    skipped_gt_overlap += 1
                    continue
                x1, y1, x2, y2 = crop
                crop_image = image[y1:y2, x1:x2]
                if crop_image.size == 0:
                    continue
                out_stem = f"hardneg_{split}_{selected_total:05d}_{image_path.stem}"
                out_image_path = images_out / f"{out_stem}.jpg"
                out_label_path = labels_out / f"{out_stem}.txt"
                cv2.imwrite(str(out_image_path), crop_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                out_label_path.write_text("", encoding="utf-8")
                image_entries.append(out_image_path.resolve().as_posix())
                rows.append(
                    {
                        "hard_negative_image": out_image_path.resolve().as_posix(),
                        "source_image": image_path.resolve().as_posix(),
                        "split": split,
                        "source": source,
                        "prediction_index": pred_idx,
                        "confidence": f"{pred.confidence or 0.0:.6f}",
                        "pred_area_ratio": f"{area_ratio:.6f}",
                        "best_gt_iou": f"{best_iou:.6f}",
                        "max_gt_crop_overlap": f"{overlap:.6f}",
                        "crop_x1": x1,
                        "crop_y1": y1,
                        "crop_x2": x2,
                        "crop_y2": y2,
                    }
                )
                used_crops.append(crop)
                selected_total += 1
                kept_for_image += 1

    list_path = args.output_dir / "hard_negative_images.txt"
    list_path.write_text("\n".join(image_entries) + ("\n" if image_entries else ""), encoding="utf-8")
    write_tsv(args.output_dir / "hard_negative_manifest.tsv", rows)

    by_source: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        by_source[str(row["source"])] = by_source.get(str(row["source"]), 0) + 1
        by_split[str(row["split"])] = by_split.get(str(row["split"]), 0) + 1
    summary = {
        "dataset_root": args.dataset_root.resolve().as_posix(),
        "images_dirs": [path.resolve().as_posix() for path in args.images_dirs],
        "pred_labels_dirs": [path.resolve().as_posix() for path in args.pred_labels_dirs],
        "hard_negative_images": len(rows),
        "candidates_total": candidates_total,
        "skipped_existing_match": skipped_existing_match,
        "skipped_gt_crop_overlap": skipped_gt_overlap,
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "parameters": {
            "min_confidence": args.min_confidence,
            "max_best_gt_iou": args.max_best_gt_iou,
            "max_gt_crop_overlap": args.max_gt_crop_overlap,
            "min_pred_area_ratio": args.min_pred_area_ratio,
            "max_pred_area_ratio": args.max_pred_area_ratio,
            "crop_expand": args.crop_expand,
            "min_crop_size": args.min_crop_size,
            "max_crop_size": args.max_crop_size,
            "max_crops_per_image": args.max_crops_per_image,
            "max_crops_total": args.max_crops_total,
            "include_sources": sorted(include_sources),
        },
        "files": {
            "image_list": list_path.as_posix(),
            "manifest": (args.output_dir / "hard_negative_manifest.tsv").as_posix(),
            "images_dir": images_out.as_posix(),
            "labels_dir": labels_out.as_posix(),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
