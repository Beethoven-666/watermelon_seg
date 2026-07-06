#!/usr/bin/env python
"""Convert LabelMe polygon annotations to a YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe polygon JSON files to YOLO segmentation labels."
    )
    parser.add_argument("--labelme-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-name", default="watermelon")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    return parser.parse_args()


def find_image(image_dir: Path, json_path: Path, image_path_value: str | None) -> Path:
    if image_path_value:
        candidate = image_dir / Path(image_path_value).name
        if candidate.exists():
            return candidate

    candidates = [
        path
        for path in image_dir.glob(json_path.stem + ".*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing image for {json_path.name}")


def yolo_rows(data: dict, json_path: Path, class_name: str) -> list[str]:
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if not width or not height:
        raise ValueError(f"Missing imageWidth/imageHeight in {json_path}")

    rows: list[str] = []
    for index, shape in enumerate(data.get("shapes") or []):
        label = str(shape.get("label", "")).strip().lower()
        shape_type = shape.get("shape_type")
        points = shape.get("points") or []
        if label != class_name.lower():
            raise ValueError(f"Unexpected label {label!r} in {json_path}, shape {index}")
        if shape_type != "polygon":
            raise ValueError(f"Unsupported shape_type {shape_type!r} in {json_path}, shape {index}")
        if len(points) < 3:
            raise ValueError(f"Polygon has fewer than 3 points in {json_path}, shape {index}")

        coords: list[float] = []
        for x, y in points:
            coords.append(min(max(float(x) / float(width), 0.0), 1.0))
            coords.append(min(max(float(y) / float(height), 0.0), 1.0))
        rows.append("0 " + " ".join(f"{value:.6f}" for value in coords))
    return rows


def split_items(items: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_count = int(count * train_ratio)
    val_count = int(count * val_ratio)
    train_count = max(1, min(train_count, count - 2))
    val_count = max(1, min(val_count, count - train_count - 1))
    return {
        "train": sorted(shuffled[:train_count], key=lambda item: item["json"].name),
        "val": sorted(shuffled[train_count : train_count + val_count], key=lambda item: item["json"].name),
        "test": sorted(shuffled[train_count + val_count :], key=lambda item: item["json"].name),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    json_files = sorted(args.labelme_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No LabelMe JSON files found in {args.labelme_dir}")

    items: list[dict] = []
    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image = find_image(args.image_dir, json_path, data.get("imagePath"))
        items.append(
            {
                "json": json_path,
                "image": image,
                "rows": yolo_rows(data, json_path, args.class_name),
            }
        )

    splits = split_items(items, args.train_ratio, args.val_ratio, args.seed)
    for parent in ("images", "labels"):
        for split in ("train", "val", "test"):
            target = output_dir / parent / split
            target.mkdir(parents=True, exist_ok=True)
            for existing in target.glob("*"):
                if existing.is_file():
                    existing.unlink()

    manifest = ["split\timage\tlabelme_json\tinstances"]
    for split, split_items_for_name in splits.items():
        for item in split_items_for_name:
            target_image = output_dir / "images" / split / item["image"].name
            target_label = output_dir / "labels" / split / f"{item['image'].stem}.txt"
            shutil.copy2(item["image"], target_image)
            target_label.write_text("\n".join(item["rows"]) + "\n", encoding="utf-8")
            manifest.append(f"{split}\t{item['image'].name}\t{item['json'].name}\t{len(item['rows'])}")

    (output_dir / "classes.txt").write_text(f"{args.class_name}\n", encoding="utf-8")
    (output_dir / "split_manifest.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"  0: {args.class_name}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
