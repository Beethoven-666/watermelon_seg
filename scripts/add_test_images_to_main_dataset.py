#!/usr/bin/env python
"""Add the converted test_images LabelMe set into the main YOLO segmentation dataset."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROOT = Path("D:/MelonDataset/watermelon_seg")
SOURCE_ROOT = ROOT / "exports" / "test_images_yolo_seg"
SUMMARY_PATH = ROOT / "exports" / "fused_watermelon_yolo_seg" / "summary.json"
MANIFEST_PATH = ROOT / "exports" / "fused_watermelon_yolo_seg" / "split_manifest.tsv"
OUTPUT_REPORT_DIR = SOURCE_ROOT

SOURCE_NAME = "local_test_images_20260707"
SOURCE_SPLIT = "manual_labelme"
SEED = 20260707
TRAIN_COUNT = 8
VAL_COUNT = 1
TEST_COUNT = 1

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SourceItem:
    index: int
    image: str
    label: str
    source_json: str
    instances: int
    width: int
    height: int

    @property
    def new_stem(self) -> str:
        return f"{SOURCE_NAME}_{self.index:04d}"

    @property
    def new_image(self) -> str:
        return f"{self.new_stem}.jpg"

    @property
    def new_label(self) -> str:
        return f"{self.new_stem}.txt"


def read_source_items() -> list[SourceItem]:
    summary = json.loads((SOURCE_ROOT / "summary.json").read_text(encoding="utf-8"))
    files = sorted(summary["files"], key=lambda item: item["image"])
    items: list[SourceItem] = []
    for index, item in enumerate(files, start=1):
        source_image = SOURCE_ROOT / "images" / "test" / item["image"]
        source_label = SOURCE_ROOT / "labels" / "test" / item["label"]
        if not source_image.exists():
            raise FileNotFoundError(source_image)
        if not source_label.exists():
            raise FileNotFoundError(source_label)
        rows = [line for line in source_label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != int(item["instances"]):
            raise ValueError(f"Instance count mismatch for {item['label']}")
        with Image.open(source_image) as image:
            width, height = image.size
            image.verify()
        if width != int(item["width"]) or height != int(item["height"]):
            raise ValueError(f"Image size mismatch for {item['image']}")
        items.append(
            SourceItem(
                index=index,
                image=item["image"],
                label=item["label"],
                source_json=item["labelme_json"],
                instances=int(item["instances"]),
                width=width,
                height=height,
            )
        )
    expected_total = TRAIN_COUNT + VAL_COUNT + TEST_COUNT
    if len(items) != expected_total:
        raise ValueError(f"Expected {expected_total} source items, got {len(items)}")
    return items


def assign_splits(items: list[SourceItem]) -> dict[str, list[SourceItem]]:
    shuffled = list(items)
    random.Random(SEED).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:TRAIN_COUNT], key=lambda item: item.index),
        "val": sorted(shuffled[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT], key=lambda item: item.index),
        "test": sorted(shuffled[TRAIN_COUNT + VAL_COUNT :], key=lambda item: item.index),
    }


def write_main_files(splits: dict[str, list[SourceItem]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for split, items in splits.items():
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            src_image = SOURCE_ROOT / "images" / "test" / item.image
            src_label = SOURCE_ROOT / "labels" / "test" / item.label
            dst_image = image_dir / item.new_image
            dst_label = label_dir / item.new_label
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_label, dst_label)
            rows.append(
                {
                    "split": split,
                    "source": SOURCE_NAME,
                    "source_split": SOURCE_SPLIT,
                    "image": item.new_image,
                    "label": item.new_label,
                    "instances": item.instances,
                    "source_image": f"test_images/{item.image}",
                    "source_json": f"test_images/label/{item.source_json}",
                    "width": item.width,
                    "height": item.height,
                }
            )
    return rows


def read_manifest_without_existing_source() -> tuple[list[str], list[dict[str, str]]]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("source") != SOURCE_NAME]
    return fieldnames, rows


def update_manifest(new_rows: list[dict[str, str | int]]) -> None:
    fieldnames, rows = read_manifest_without_existing_source()
    required = ["split", "source", "source_split", "image", "label", "instances", "source_image"]
    if fieldnames != required:
        raise ValueError(f"Unexpected manifest columns: {fieldnames}")
    for row in new_rows:
        rows.append({name: str(row[name]) for name in fieldnames})
    split_order = {"train": 0, "val": 1, "test": 2}
    rows.sort(key=lambda row: (split_order.get(row["split"], 99), row["source"], row["image"]))
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    split_summary: dict[str, dict] = {}
    source_totals: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        sources: dict[str, int] = {}
        for row in split_rows:
            sources[row["source"]] = sources.get(row["source"], 0) + 1
            source_totals[row["source"]] = source_totals.get(row["source"], 0) + 1
        split_summary[split] = {
            "images": len(split_rows),
            "labels": len(split_rows),
            "instances": sum(int(row["instances"]) for row in split_rows),
            "sources": dict(sorted(sources.items())),
        }
    summary["total_images"] = len(rows)
    summary["total_instances"] = sum(int(row["instances"]) for row in rows)
    summary["summary"] = split_summary
    summary["source_totals"] = dict(sorted(source_totals.items()))
    summary["local_test_images_20260707"] = {
        "source": "test_images",
        "converted_dataset": "exports/test_images_yolo_seg",
        "added_to_main_dataset": "2026-07-07",
        "seed": SEED,
        "split_counts": {
            "train": TRAIN_COUNT,
            "val": VAL_COUNT,
            "test": TEST_COUNT,
        },
        "images": TRAIN_COUNT + VAL_COUNT + TEST_COUNT,
        "instances": sum(
            int(row["instances"]) for row in rows if row["source"] == SOURCE_NAME
        ),
        "naming": f"{SOURCE_NAME}_0001.jpg ...",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def remove_label_caches() -> list[str]:
    removed: list[str] = []
    for cache in (ROOT / "labels").glob("*.cache"):
        cache.unlink()
        removed.append(cache.as_posix())
    for cache in (ROOT / "labels").glob("*/*.cache"):
        cache.unlink()
        removed.append(cache.as_posix())
    return removed


def write_integration_report(new_rows: list[dict[str, str | int]], summary: dict, removed_caches: list[str]) -> None:
    report_rows = [
        {
            "split": row["split"],
            "image": row["image"],
            "label": row["label"],
            "instances": row["instances"],
            "source_image": row["source_image"],
            "source_json": row["source_json"],
            "width": row["width"],
            "height": row["height"],
        }
        for row in sorted(new_rows, key=lambda row: (row["split"], row["image"]))
    ]
    (OUTPUT_REPORT_DIR / "integration_summary.json").write_text(
        json.dumps(
            {
                "source_name": SOURCE_NAME,
                "seed": SEED,
                "new_rows": report_rows,
                "main_dataset": {
                    "total_images": summary["total_images"],
                    "total_instances": summary["total_instances"],
                    "summary": summary["summary"],
                    "source_totals": summary["source_totals"],
                },
                "removed_label_caches": removed_caches,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_REPORT_DIR / "integration_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "image", "label", "instances", "source_image", "source_json", "width", "height"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    items = read_source_items()
    splits = assign_splits(items)
    new_rows = write_main_files(splits)
    update_manifest(new_rows)
    summary = summarize_manifest()
    removed_caches = remove_label_caches()
    write_integration_report(new_rows, summary, removed_caches)
    print(f"source_added={SOURCE_NAME}")
    print(f"added_images={len(new_rows)}")
    print(f"added_instances={sum(int(row['instances']) for row in new_rows)}")
    for split in ("train", "val", "test"):
        split_rows = [row for row in new_rows if row["split"] == split]
        print(
            f"{split}: images={len(split_rows)} "
            f"instances={sum(int(row['instances']) for row in split_rows)}"
        )
    print(f"total_images={summary['total_images']}")
    print(f"total_instances={summary['total_instances']}")
    print(f"removed_label_caches={len(removed_caches)}")


if __name__ == "__main__":
    main()
