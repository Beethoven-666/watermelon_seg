#!/usr/bin/env python
"""Audit YOLO segmentation label instances for size, edge, and reachability."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO segmentation instances.")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--overlap-tsv", type=Path)
    parser.add_argument("--review-queue-tsv", type=Path)
    parser.add_argument("--small-area", type=float, default=0.0005)
    parser.add_argument("--tiny-area", type=float, default=0.00015)
    parser.add_argument("--edge-margin", type=float, default=0.01)
    return parser.parse_args()


def read_tsv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_yolo_label(path: Path) -> list[np.ndarray]:
    if not path.exists():
        return []
    instances: list[np.ndarray] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(f"{path}:{line_no} has invalid YOLO segmentation format")
        points = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        instances.append(np.clip(points, 0.0, 1.0))
    return instances


def polygon_area_ratio(points: np.ndarray, width: int, height: int) -> float:
    scaled = points.copy()
    scaled[:, 0] *= width - 1
    scaled[:, 1] *= height - 1
    return float(cv2.contourArea(scaled.astype(np.float32)) / max(width * height, 1))


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


def recommendation(
    area_ratio: float,
    bbox_width: float,
    bbox_height: float,
    touches_edge: bool,
    reachable: str,
    in_review_queue: bool,
    args: argparse.Namespace,
) -> str:
    if reachable == "False" and area_ratio <= args.small_area:
        return "优先复核：不可达且极小，可能是碎片/非目标/口径不一致"
    if reachable == "False" and touches_edge:
        return "优先复核：不可达且贴边，可能是截断目标或边界口径问题"
    if reachable == "False":
        return "优先复核：模型无可匹配预测，需确认标注边界或补充同类样本"
    if area_ratio <= args.tiny_area:
        return "抽检：极小实例，确认是否应进入全实例标注"
    if touches_edge and in_review_queue:
        return "抽检：错误高发图中的贴边实例"
    if min(bbox_width, bbox_height) <= 0.015 and in_review_queue:
        return "抽检：错误高发图中的细长/局部实例"
    return "保留观察"


def main() -> int:
    args = parse_args()
    overlap_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in read_tsv(args.overlap_tsv):
        overlap_by_key[(row["image"], int(row["gt_index"]))] = row

    review_images = {row["image"] for row in read_tsv(args.review_queue_tsv)}
    rows: list[dict[str, object]] = []
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        height, width = image.shape[:2]
        instances = read_yolo_label(args.labels_dir / f"{image_path.stem}.txt")
        for idx, points in enumerate(instances):
            min_x = float(points[:, 0].min())
            min_y = float(points[:, 1].min())
            max_x = float(points[:, 0].max())
            max_y = float(points[:, 1].max())
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            touches_edge = (
                min_x <= args.edge_margin
                or min_y <= args.edge_margin
                or max_x >= 1.0 - args.edge_margin
                or max_y >= 1.0 - args.edge_margin
            )
            overlap = overlap_by_key.get((image_path.name, idx), {})
            reachable = overlap.get("reachable_at_iou_threshold", "")
            best_iou = float(overlap["best_iou"]) if "best_iou" in overlap else -1.0
            area_ratio = polygon_area_ratio(points, width, height)
            in_review_queue = image_path.name in review_images
            rec = recommendation(
                area_ratio,
                bbox_width,
                bbox_height,
                touches_edge,
                reachable,
                in_review_queue,
                args,
            )
            priority = 0
            if reachable == "False":
                priority += 100
            if area_ratio <= args.tiny_area:
                priority += 40
            elif area_ratio <= args.small_area:
                priority += 20
            if touches_edge:
                priority += 15
            if in_review_queue:
                priority += 10
            rows.append(
                {
                    "image": image_path.name,
                    "source": source_hint(image_path.name),
                    "gt_index": idx,
                    "area_ratio": area_ratio,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "bbox_min_x": min_x,
                    "bbox_min_y": min_y,
                    "bbox_max_x": max_x,
                    "bbox_max_y": max_y,
                    "touches_edge": int(touches_edge),
                    "in_review_queue": int(in_review_queue),
                    "best_iou": best_iou,
                    "reachable_at_iou_0p5": reachable,
                    "priority": priority,
                    "recommendation": rec,
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["priority"]),
            -float(row["area_ratio"]),
            str(row["image"]),
            int(row["gt_index"]),
        ),
        reverse=True,
    )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "source",
        "gt_index",
        "area_ratio",
        "bbox_width",
        "bbox_height",
        "bbox_min_x",
        "bbox_min_y",
        "bbox_max_x",
        "bbox_max_y",
        "touches_edge",
        "in_review_queue",
        "best_iou",
        "reachable_at_iou_0p5",
        "priority",
        "recommendation",
    ]
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    tiny = sum(1 for row in rows if float(row["area_ratio"]) <= args.tiny_area)
    small = sum(1 for row in rows if float(row["area_ratio"]) <= args.small_area)
    edge = sum(1 for row in rows if int(row["touches_edge"]))
    unreachable = sum(1 for row in rows if row["reachable_at_iou_0p5"] == "False")
    priority_rows = [row for row in rows if int(row["priority"]) >= 100]

    lines = [
        "# 实例级标签审计",
        "",
        "该报告用于辅助复标，不会自动删除标签。",
        "",
        "## 统计",
        "",
        f"- 实例总数：{total}",
        f"- tiny 实例（area <= {args.tiny_area}）：{tiny}",
        f"- small 实例（area <= {args.small_area}）：{small}",
        f"- 贴边实例（edge margin <= {args.edge_margin}）：{edge}",
        f"- 当前模型不可达 GT（best IoU < 0.5）：{unreachable}",
        f"- 高优先级复核实例：{len(priority_rows)}",
        "",
        "## 高优先级实例 Top 30",
        "",
        "| 排名 | 图片 | GT | area | edge | best IoU | 建议 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(priority_rows[:30], start=1):
        lines.append(
            f"| {rank} | `{row['image']}` | {row['gt_index']} | "
            f"{float(row['area_ratio']):.6f} | {row['touches_edge']} | "
            f"{float(row['best_iou']):.3f} | {row['recommendation']} |"
        )
    lines.extend(["", "完整 TSV：", "", f"```text\n{args.output_tsv}\n```"])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_tsv)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
