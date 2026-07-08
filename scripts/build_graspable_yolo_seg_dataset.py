#!/usr/bin/env python
"""Build a YOLO segmentation dataset for large, graspable watermelon targets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, default=Path("."))
    parser.add_argument("--dst-root", type=Path, default=Path("exports/graspable_yolo_seg"))
    parser.add_argument("--min-area-ratio", type=float, default=0.02)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01)
    parser.add_argument("--drop-edge", action="store_true")
    parser.add_argument("--copy-images", action="store_true", default=True)
    return parser.parse_args()


def read_instances(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def polygon_area_ratio(label_line: str, width: int, height: int) -> float:
    parts = label_line.split()
    points = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
    scaled = points.copy()
    scaled[:, 0] *= width - 1
    scaled[:, 1] *= height - 1
    return float(cv2.contourArea(scaled.astype(np.float32)) / max(width * height, 1))


def touches_edge(label_line: str, margin: float) -> bool:
    parts = label_line.split()
    points = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
    return bool(
        points[:, 0].min() <= margin
        or points[:, 1].min() <= margin
        or points[:, 0].max() >= 1.0 - margin
        or points[:, 1].max() >= 1.0 - margin
    )


def filter_label_lines(lines: list[str], width: int, height: int, args: argparse.Namespace) -> tuple[list[str], int]:
    kept: list[str] = []
    removed = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            removed += 1
            continue
        area_ratio = polygon_area_ratio(line, width, height)
        if area_ratio < args.min_area_ratio:
            removed += 1
            continue
        if args.drop_edge and touches_edge(line, args.edge_margin_ratio):
            removed += 1
            continue
        kept.append(line)
    return kept, removed


def write_data_yaml(dst_root: Path) -> None:
    text = "\n".join(
        [
            f"path: {dst_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: watermelon",
            "",
        ]
    )
    (dst_root / "data.yaml").write_text(text, encoding="utf-8")


def write_readme(dst_root: Path, summary: dict[str, object]) -> None:
    text = "\n".join(
        [
            "# 可抓取目标 YOLO 分割数据集",
            "",
            "该数据集从主西瓜实例分割数据集派生生成。",
            "它只保留画面中足够大的可见西瓜实例，用于机械臂抓取候选检测。",
            "生成过程不会修改主数据集。",
            "",
            f"- 最小多边形面积占比：`{summary['min_area_ratio']}`",
            f"- 是否删除贴边实例：`{summary['drop_edge']}`",
            "",
            "## 拆分统计",
            "",
            "| split | 图片 | 原始实例 | 保留实例 | 移除实例 | 空标签图片 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, row in summary["splits"].items():  # type: ignore[index, union-attr]
        text += (
            f"\n| {split} | {row['images']} | {row['source_instances']} | "
            f"{row['kept_instances']} | {row['removed_instances']} | {row['empty_labels']} |"
        )
    text += "\n"
    (dst_root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    summary: dict[str, object] = {
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "min_area_ratio": args.min_area_ratio,
        "drop_edge": bool(args.drop_edge),
        "edge_margin_ratio": args.edge_margin_ratio,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        src_images = src_root / "images" / split
        src_labels = src_root / "labels" / split
        dst_images = dst_root / "images" / split
        dst_labels = dst_root / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        split_summary = {
            "images": 0,
            "source_instances": 0,
            "kept_instances": 0,
            "removed_instances": 0,
            "empty_labels": 0,
        }
        for image_path in sorted(path for path in src_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            height, width = image.shape[:2]
            label_path = src_labels / f"{image_path.stem}.txt"
            lines = read_instances(label_path)
            kept, removed = filter_label_lines(lines, width, height, args)

            if args.copy_images:
                shutil.copy2(image_path, dst_images / image_path.name)
            label_text = "\n".join(kept)
            if label_text:
                label_text += "\n"
            (dst_labels / f"{image_path.stem}.txt").write_text(label_text, encoding="utf-8")

            split_summary["images"] += 1
            split_summary["source_instances"] += len(lines)
            split_summary["kept_instances"] += len(kept)
            split_summary["removed_instances"] += removed
            if not kept:
                split_summary["empty_labels"] += 1

        summary["splits"][split] = split_summary  # type: ignore[index]

    (dst_root / "classes.txt").write_text("watermelon\n", encoding="utf-8")
    write_data_yaml(dst_root)
    write_readme(dst_root, summary)
    (dst_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
