#!/usr/bin/env python
"""Export YOLO segmentation labels with source-aware two-model policy settings."""

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
class Policy:
    confidence_threshold: float
    min_area_ratio: float
    min_support: int
    support_iou_threshold: float
    nms_iou_threshold: float


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
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--pred-a-dir", required=True, type=Path)
    parser.add_argument("--pred-b-dir", required=True, type=Path)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument("--raster-size", type=int, default=768)
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
    return "default"


def load_policies(path: Path) -> tuple[dict[str, Policy], Policy]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_policies = raw.get("policies", raw)
    policies = {
        source: Policy(
            confidence_threshold=float(config["confidence_threshold"]),
            min_area_ratio=float(config["min_area_ratio"]),
            min_support=int(config["min_support"]),
            support_iou_threshold=float(config["support_iou_threshold"]),
            nms_iou_threshold=float(config["nms_iou_threshold"]),
        )
        for source, config in raw_policies.items()
        if source != "default"
    }
    default_raw = raw_policies.get(
        "default",
        {
            "confidence_threshold": 0.4,
            "min_area_ratio": 0.0004,
            "min_support": 1,
            "support_iou_threshold": 0.35,
            "nms_iou_threshold": 0.3,
        },
    )
    default_policy = Policy(
        confidence_threshold=float(default_raw["confidence_threshold"]),
        min_area_ratio=float(default_raw["min_area_ratio"]),
        min_support=int(default_raw["min_support"]),
        support_iou_threshold=float(default_raw["support_iou_threshold"]),
        nms_iou_threshold=float(default_raw["nms_iou_threshold"]),
    )
    return policies, default_policy


def read_points(path: Path) -> list[tuple[np.ndarray, float]]:
    if not path.exists():
        return []
    rows: list[tuple[np.ndarray, float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 8:
            continue
        coord_parts = parts[1:-1]
        if len(coord_parts) < 6 or len(coord_parts) % 2:
            continue
        points = np.asarray([float(value) for value in coord_parts], dtype=np.float32).reshape(-1, 2)
        rows.append((np.clip(points, 0.0, 1.0), float(parts[-1])))
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


def make_preds(path: Path, model_name: str, raster_size: int) -> list[Pred]:
    preds: list[Pred] = []
    for points, confidence in read_points(path):
        mask = mask_from_points(points, raster_size)
        area = int(mask.sum())
        if area <= 0:
            continue
        preds.append(
            Pred(
                model_name=model_name,
                confidence=confidence,
                points=points,
                mask=mask,
                area=area,
                area_ratio=area / float(raster_size * raster_size),
                bbox=bbox(points),
            )
        )
    return preds


def mask_iou(a: Pred, b: Pred) -> float:
    if a.area <= 0 or b.area <= 0 or not bboxes_overlap(a.bbox, b.bbox):
        return 0.0
    intersection = int(np.logical_and(a.mask, b.mask).sum())
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


def supported_by_other_model(pred: Pred, others: list[Pred], support_iou_threshold: float) -> bool:
    return any(mask_iou(pred, other) >= support_iou_threshold for other in others)


def nms(preds: list[Pred], nms_iou_threshold: float) -> list[Pred]:
    kept: list[Pred] = []
    for pred in sorted(preds, key=lambda item: item.confidence, reverse=True):
        if any(mask_iou(pred, kept_pred) > nms_iou_threshold for kept_pred in kept):
            continue
        kept.append(pred)
    return kept


def format_label(pred: Pred) -> str:
    coords = " ".join(f"{value:.6f}" for point in pred.points for value in point)
    return f"0 {coords} {pred.confidence:.6f}"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    policies, default_policy = load_policies(args.policy_json)
    labels_dir = args.output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total_kept = total_candidates = total_selected_a = total_selected_b = 0
    kept_by_model = {args.model_a_name: 0, args.model_b_name: 0}
    kept_by_source: dict[str, int] = {}

    image_paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        source = source_hint(image_path.name)
        policy = policies.get(source, default_policy)
        pred_a = make_preds(args.pred_a_dir / f"{image_path.stem}.txt", args.model_a_name, args.raster_size)
        pred_b = make_preds(args.pred_b_dir / f"{image_path.stem}.txt", args.model_b_name, args.raster_size)
        selected_a = [
            pred
            for pred in pred_a
            if pred.confidence >= policy.confidence_threshold and pred.area_ratio >= policy.min_area_ratio
        ]
        selected_b = [
            pred
            for pred in pred_b
            if pred.confidence >= policy.confidence_threshold and pred.area_ratio >= policy.min_area_ratio
        ]
        if policy.min_support == 2:
            candidates = [
                pred for pred in selected_a if supported_by_other_model(pred, selected_b, policy.support_iou_threshold)
            ] + [
                pred for pred in selected_b if supported_by_other_model(pred, selected_a, policy.support_iou_threshold)
            ]
        else:
            candidates = selected_a + selected_b
        kept = nms(candidates, policy.nms_iou_threshold)

        text = "\n".join(format_label(pred) for pred in kept)
        if text:
            text += "\n"
        (labels_dir / f"{image_path.stem}.txt").write_text(text, encoding="utf-8")

        kept_a = sum(1 for pred in kept if pred.model_name == args.model_a_name)
        kept_b = sum(1 for pred in kept if pred.model_name == args.model_b_name)
        kept_by_model[args.model_a_name] += kept_a
        kept_by_model[args.model_b_name] += kept_b
        kept_by_source[source] = kept_by_source.get(source, 0) + len(kept)
        total_selected_a += len(selected_a)
        total_selected_b += len(selected_b)
        total_candidates += len(candidates)
        total_kept += len(kept)
        rows.append(
            {
                "image": image_path.name,
                "source": source,
                "selected_a": len(selected_a),
                "selected_b": len(selected_b),
                "candidates": len(candidates),
                "kept": len(kept),
                f"kept_{args.model_a_name}": kept_a,
                f"kept_{args.model_b_name}": kept_b,
                "confidence_threshold": policy.confidence_threshold,
                "min_area_ratio": policy.min_area_ratio,
                "min_support": policy.min_support,
                "support_iou_threshold": policy.support_iou_threshold,
                "nms_iou_threshold": policy.nms_iou_threshold,
            }
        )

    write_tsv(args.output_dir / "export_manifest.tsv", rows)
    summary = {
        "images_dir": args.images_dir.as_posix(),
        "pred_a_dir": args.pred_a_dir.as_posix(),
        "pred_b_dir": args.pred_b_dir.as_posix(),
        "policy_json": args.policy_json.as_posix(),
        "labels_dir": labels_dir.as_posix(),
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "images": len(image_paths),
        "selected_a": total_selected_a,
        "selected_b": total_selected_b,
        "candidates": total_candidates,
        "kept": total_kept,
        "kept_by_model": kept_by_model,
        "kept_by_source": kept_by_source,
        "policies": {
            source: {
                "confidence_threshold": policy.confidence_threshold,
                "min_area_ratio": policy.min_area_ratio,
                "min_support": policy.min_support,
                "support_iou_threshold": policy.support_iou_threshold,
                "nms_iou_threshold": policy.nms_iou_threshold,
            }
            for source, policy in sorted(policies.items())
        },
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
