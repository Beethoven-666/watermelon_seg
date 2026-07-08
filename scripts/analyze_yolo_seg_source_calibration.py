#!/usr/bin/env python
"""Tune source-aware YOLO segmentation post-processing on one split and evaluate another."""

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


@dataclass(frozen=True)
class Config:
    confidence_threshold: float
    nms_iou_threshold: float
    min_area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-images-dir", required=True, type=Path)
    parser.add_argument("--tune-labels-dir", required=True, type=Path)
    parser.add_argument("--tune-pred-labels-dir", required=True, type=Path)
    parser.add_argument("--eval-images-dir", required=True, type=Path)
    parser.add_argument("--eval-labels-dir", required=True, type=Path)
    parser.add_argument("--eval-pred-labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raster-size", type=int, default=768)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--recall-target", type=float, default=0.90)
    parser.add_argument(
        "--confidence-thresholds",
        default="0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7",
    )
    parser.add_argument("--nms-iou-thresholds", default="0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5")
    parser.add_argument("--min-area-ratios", default="0,0.0005,0.001,0.002,0.003,0.005")
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


def parse_float_list(raw: str) -> list[float]:
    return sorted({float(value.strip()) for value in raw.split(",") if value.strip()})


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


def build_records(
    images_dir: Path,
    labels_dir: Path,
    pred_labels_dir: Path,
    raster_size: int,
    min_confidence: float,
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        gt = read_polygons(labels_dir / f"{image_path.stem}.txt", has_confidence=False)
        preds = [
            pred
            for pred in read_polygons(pred_labels_dir / f"{image_path.stem}.txt", has_confidence=True)
            if pred.confidence is not None and pred.confidence >= min_confidence
        ]
        gt_masks = [mask_from_points(instance.points, raster_size) for instance in gt]
        pred_masks = [mask_from_points(instance.points, raster_size) for instance in preds]
        gt_areas = [int(mask.sum()) for mask in gt_masks]
        pred_areas = [int(mask.sum()) for mask in pred_masks]

        valid_preds: list[Instance] = []
        valid_pred_masks: list[np.ndarray] = []
        valid_pred_areas: list[int] = []
        for pred, pred_mask, pred_area in zip(preds, pred_masks, pred_areas):
            if pred_area <= 0:
                continue
            valid_preds.append(pred)
            valid_pred_masks.append(pred_mask)
            valid_pred_areas.append(pred_area)

        records[image_path.name] = {
            "source": source_hint(image_path.name),
            "gt_count": len(gt_masks),
            "gt_masks": gt_masks,
            "gt_areas": gt_areas,
            "preds": valid_preds,
            "pred_areas": valid_pred_areas,
            "pred_area_ratios": [area / float(raster_size * raster_size) for area in valid_pred_areas],
            "pred_gt_ious": pairwise_iou_matrix(valid_pred_masks, valid_pred_areas, gt_masks, gt_areas),
            "pred_pred_ious": pairwise_iou_matrix(valid_pred_masks, valid_pred_areas, valid_pred_masks, valid_pred_areas),
        }
    return records


def mask_nms(pred_indices: list[int], pred_pred_ious: list[list[float]], nms_iou: float) -> list[int]:
    kept: list[int] = []
    for pred_idx in pred_indices:
        if all(pred_pred_ious[pred_idx][kept_idx] <= nms_iou for kept_idx in kept):
            kept.append(pred_idx)
    return kept


def evaluate_records(
    records: dict[str, dict[str, object]],
    config_by_source: dict[str, Config],
    default_config: Config,
    match_iou_threshold: float,
) -> dict[str, object]:
    totals = {"gt": 0, "selected_predictions": 0, "kept_predictions": 0, "tp": 0, "fp": 0, "fn": 0}
    by_source: dict[str, dict[str, int]] = {}
    per_image: list[dict[str, object]] = []

    for image_name, record in records.items():
        source = str(record["source"])
        config = config_by_source.get(source, default_config)
        gt_count = int(record["gt_count"])
        preds: list[Instance] = record["preds"]  # type: ignore[assignment]
        pred_areas: list[int] = record["pred_areas"]  # type: ignore[assignment]
        pred_area_ratios: list[float] = record["pred_area_ratios"]  # type: ignore[assignment]
        pred_gt_ious: list[list[float]] = record["pred_gt_ious"]  # type: ignore[assignment]
        pred_pred_ious: list[list[float]] = record["pred_pred_ious"]  # type: ignore[assignment]

        selected = [
            idx
            for idx, pred in enumerate(preds)
            if pred.confidence is not None
            and pred.confidence >= config.confidence_threshold
            and pred_areas[idx] > 0
            and pred_area_ratios[idx] >= config.min_area_ratio
        ]
        selected.sort(key=lambda idx: preds[idx].confidence or 0.0, reverse=True)
        kept = mask_nms(selected, pred_pred_ious, config.nms_iou_threshold)

        matched_gt: set[int] = set()
        tp = fp = 0
        for pred_idx in kept:
            best_gt = -1
            best_iou = 0.0
            for gt_idx in range(gt_count):
                if gt_idx in matched_gt:
                    continue
                score = pred_gt_ious[pred_idx][gt_idx] if gt_idx < len(pred_gt_ious[pred_idx]) else 0.0
                if score > best_iou:
                    best_iou = score
                    best_gt = gt_idx
            if best_gt >= 0 and best_iou >= match_iou_threshold:
                matched_gt.add(best_gt)
                tp += 1
            else:
                fp += 1
        fn = gt_count - len(matched_gt)

        for key, value in {
            "gt": gt_count,
            "selected_predictions": len(selected),
            "kept_predictions": len(kept),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }.items():
            totals[key] += value
        source_totals = by_source.setdefault(source, {"gt": 0, "selected_predictions": 0, "kept_predictions": 0, "tp": 0, "fp": 0, "fn": 0})
        for key, value in {
            "gt": gt_count,
            "selected_predictions": len(selected),
            "kept_predictions": len(kept),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }.items():
            source_totals[key] += value
        per_image.append(
            {
                "image": image_name,
                "source": source,
                "gt": gt_count,
                "selected_predictions": len(selected),
                "kept_predictions": len(kept),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return {"totals": add_rates(totals), "by_source": {k: add_rates(v) for k, v in by_source.items()}, "per_image": per_image}


def add_rates(row: dict[str, int]) -> dict[str, float | int]:
    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    enriched: dict[str, float | int] = dict(row)
    enriched.update({"precision": precision, "recall": recall, "f1": f1})
    return enriched


def evaluate_source_grid(
    records: dict[str, dict[str, object]],
    configs: list[Config],
    match_iou_threshold: float,
) -> dict[str, list[dict[str, object]]]:
    sources = sorted({str(record["source"]) for record in records.values()})
    rows_by_source: dict[str, list[dict[str, object]]] = {}
    for source in sources:
        source_records = {name: record for name, record in records.items() if record["source"] == source}
        rows: list[dict[str, object]] = []
        for config in configs:
            metrics = evaluate_records(source_records, {source: config}, config, match_iou_threshold)["totals"]
            rows.append(
                {
                    "source": source,
                    "confidence_threshold": config.confidence_threshold,
                    "nms_iou_threshold": config.nms_iou_threshold,
                    "min_area_ratio": config.min_area_ratio,
                    **metrics,
                }
            )
        rows_by_source[source] = rows
    return rows_by_source


def config_from_row(row: dict[str, object]) -> Config:
    return Config(
        confidence_threshold=float(row["confidence_threshold"]),
        nms_iou_threshold=float(row["nms_iou_threshold"]),
        min_area_ratio=float(row["min_area_ratio"]),
    )


def pareto_prune(states: dict[tuple[int, int], dict[str, Config]]) -> dict[tuple[int, int], dict[str, Config]]:
    items = sorted(states.items(), key=lambda item: (-item[0][0], item[0][1]))
    pruned: dict[tuple[int, int], dict[str, Config]] = {}
    best_fp = 10**9
    for (tp, fp), config_by_source in items:
        if fp < best_fp:
            pruned[(tp, fp)] = config_by_source
            best_fp = fp
    return pruned


def search_source_policies(
    rows_by_source: dict[str, list[dict[str, object]]],
    total_gt: int,
    precision_target: float,
    recall_target: float,
) -> dict[str, object]:
    states: dict[tuple[int, int], dict[str, Config]] = {(0, 0): {}}
    for source, rows in sorted(rows_by_source.items()):
        next_states: dict[tuple[int, int], dict[str, Config]] = {}
        for (base_tp, base_fp), base_configs in states.items():
            for row in rows:
                key = (base_tp + int(row["tp"]), base_fp + int(row["fp"]))
                if key not in next_states:
                    configs = dict(base_configs)
                    configs[source] = config_from_row(row)
                    next_states[key] = configs
        states = pareto_prune(next_states)

    def state_metrics(item: tuple[tuple[int, int], dict[str, Config]]) -> dict[str, object]:
        (tp, fp), config_by_source = item
        fn = total_gt - tp
        metrics = add_rates({"gt": total_gt, "selected_predictions": 0, "kept_predictions": 0, "tp": tp, "fp": fp, "fn": fn})
        return {"metrics": metrics, "config_by_source": config_by_source}

    candidates = [state_metrics(item) for item in states.items()]
    best_f1 = max(candidates, key=lambda row: (float(row["metrics"]["f1"]), float(row["metrics"]["recall"]), float(row["metrics"]["precision"])))
    target_rows = [
        row
        for row in candidates
        if float(row["metrics"]["precision"]) >= precision_target and float(row["metrics"]["recall"]) >= recall_target
    ]
    precision_rows = [row for row in candidates if float(row["metrics"]["precision"]) >= precision_target]
    recall_rows = [row for row in candidates if float(row["metrics"]["recall"]) >= recall_target]
    return {
        "state_count": len(states),
        "best_f1": best_f1,
        "best_target": (
            max(target_rows, key=lambda row: (float(row["metrics"]["f1"]), float(row["metrics"]["recall"])))
            if target_rows
            else None
        ),
        "best_recall_with_precision_target": (
            max(precision_rows, key=lambda row: (float(row["metrics"]["recall"]), float(row["metrics"]["f1"])))
            if precision_rows
            else None
        ),
        "best_precision_with_recall_target": (
            max(recall_rows, key=lambda row: (float(row["metrics"]["precision"]), float(row["metrics"]["f1"])))
            if recall_rows
            else None
        ),
    }


def serialize_config_map(config_by_source: dict[str, Config]) -> dict[str, dict[str, float]]:
    return {
        source: {
            "confidence_threshold": config.confidence_threshold,
            "nms_iou_threshold": config.nms_iou_threshold,
            "min_area_ratio": config.min_area_ratio,
        }
        for source, config in sorted(config_by_source.items())
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
    configs = [
        Config(confidence, nms_iou, min_area)
        for confidence in parse_float_list(args.confidence_thresholds)
        for nms_iou in parse_float_list(args.nms_iou_thresholds)
        for min_area in parse_float_list(args.min_area_ratios)
    ]

    min_confidence_for_records = min(parse_float_list(args.confidence_thresholds))
    tune_records = build_records(
        args.tune_images_dir,
        args.tune_labels_dir,
        args.tune_pred_labels_dir,
        args.raster_size,
        min_confidence_for_records,
    )
    eval_records = build_records(
        args.eval_images_dir,
        args.eval_labels_dir,
        args.eval_pred_labels_dir,
        args.raster_size,
        min_confidence_for_records,
    )
    default_config = Config(confidence_threshold=0.5, nms_iou_threshold=0.075, min_area_ratio=0.0)
    rows_by_source = evaluate_source_grid(tune_records, configs, args.match_iou_threshold)
    all_grid_rows = [row for rows in rows_by_source.values() for row in rows]
    write_tsv(args.output_dir / "source_config_grid.tsv", all_grid_rows)

    total_tune_gt = sum(int(record["gt_count"]) for record in tune_records.values())
    search = search_source_policies(rows_by_source, total_tune_gt, args.precision_target, args.recall_target)

    evaluated_policies: dict[str, object] = {}
    for policy_name in (
        "best_f1",
        "best_target",
        "best_recall_with_precision_target",
        "best_precision_with_recall_target",
    ):
        selected = search.get(policy_name)
        if selected is None:
            evaluated_policies[policy_name] = None
            continue
        config_by_source: dict[str, Config] = selected["config_by_source"]  # type: ignore[assignment]
        eval_result = evaluate_records(eval_records, config_by_source, default_config, args.match_iou_threshold)
        evaluated_policies[policy_name] = {
            "tune": {
                "metrics": selected["metrics"],
                "config_by_source": serialize_config_map(config_by_source),
            },
            "eval": eval_result,
        }
        write_tsv(args.output_dir / f"{policy_name}_eval_per_image.tsv", eval_result["per_image"])  # type: ignore[arg-type]

    summary = {
        "inputs": {
            "tune_images_dir": str(args.tune_images_dir),
            "tune_labels_dir": str(args.tune_labels_dir),
            "tune_pred_labels_dir": str(args.tune_pred_labels_dir),
            "eval_images_dir": str(args.eval_images_dir),
            "eval_labels_dir": str(args.eval_labels_dir),
            "eval_pred_labels_dir": str(args.eval_pred_labels_dir),
            "raster_size": args.raster_size,
            "match_iou_threshold": args.match_iou_threshold,
            "confidence_thresholds": parse_float_list(args.confidence_thresholds),
            "min_confidence_for_records": min_confidence_for_records,
            "nms_iou_thresholds": parse_float_list(args.nms_iou_thresholds),
            "min_area_ratios": parse_float_list(args.min_area_ratios),
        },
        "targets": {
            "precision": args.precision_target,
            "recall": args.recall_target,
        },
        "tune_dataset": {
            "images": len(tune_records),
            "ground_truth_instances": total_tune_gt,
            "sources": sorted(rows_by_source),
            "pareto_state_count": search["state_count"],
        },
        "policies": evaluated_policies,
    }
    (args.output_dir / "source_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=lambda value: value.__dict__) + "\n",
        encoding="utf-8",
    )
    compact = {
        name: None if policy is None else {"tune": policy["tune"]["metrics"], "eval": policy["eval"]["totals"]}
        for name, policy in evaluated_policies.items()
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
