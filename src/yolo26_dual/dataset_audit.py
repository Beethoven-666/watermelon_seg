"""YOLO segmentation/detection 数据集只读审计。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return (
        orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0
    )


def _self_intersects(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def audit_dataset(
    root: Path, task: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if task not in {"segment", "detect"}:
        raise ValueError("task must be segment or detect")
    issues: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    hashes: dict[str, list[tuple[str, str]]] = {}
    split_rows: dict[str, Any] = {}
    source_splits: dict[str, set[str]] = {}
    for split in SPLITS:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        images = (
            sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            if image_dir.exists()
            else []
        )
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        by_stem = {p.stem: p for p in labels}
        instance_count = empty_count = 0
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                issues.append(
                    {
                        "split": split,
                        "file": str(image_path),
                        "line": 0,
                        "code": "invalid_image",
                        "message": str(exc),
                    }
                )
            digest = _sha256(image_path)
            hashes.setdefault(digest, []).append((split, str(image_path)))
            label_path = by_stem.get(image_path.stem)
            if label_path is None:
                issues.append(
                    {
                        "split": split,
                        "file": str(image_path),
                        "line": 0,
                        "code": "missing_label",
                        "message": "matching label is missing",
                    }
                )
                continue
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_count += 1
            for number, line in enumerate(lines, 1):
                values = line.split()
                try:
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                except (ValueError, IndexError) as exc:
                    issues.append(
                        {
                            "split": split,
                            "file": str(label_path),
                            "line": number,
                            "code": "parse_error",
                            "message": str(exc),
                        }
                    )
                    continue
                instance_count += 1
                expected = (
                    len(coordinates) >= 6 and len(coordinates) % 2 == 0
                    if task == "segment"
                    else len(coordinates) == 4
                )
                if (
                    class_id != 0
                    or not expected
                    or not all(
                        math.isfinite(v) and 0.0 <= v <= 1.0 for v in coordinates
                    )
                ):
                    issues.append(
                        {
                            "split": split,
                            "file": str(label_path),
                            "line": number,
                            "code": "invalid_label",
                            "message": "class, field count, or coordinate range is invalid",
                        }
                    )
                    continue
                if task == "detect" and (coordinates[2] <= 0 or coordinates[3] <= 0):
                    issues.append(
                        {
                            "split": split,
                            "file": str(label_path),
                            "line": number,
                            "code": "non_positive_box",
                            "message": "box width and height must be positive",
                        }
                    )
                    continue
                if task == "segment":
                    points = list(zip(coordinates[0::2], coordinates[1::2]))
                    if len(set(points)) < 3 or _self_intersects(points):
                        issues.append(
                            {
                                "split": split,
                                "file": str(label_path),
                                "line": number,
                                "code": "invalid_polygon",
                                "message": "polygon has fewer than 3 unique points or self-intersects",
                            }
                        )
                        continue
        image_stems = {p.stem for p in images}
        for label_path in labels:
            if label_path.stem not in image_stems:
                issues.append(
                    {
                        "split": split,
                        "file": str(label_path),
                        "line": 0,
                        "code": "orphan_label",
                        "message": "matching image is missing",
                    }
                )
        split_rows[split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": instance_count,
            "empty_labels": empty_count,
        }
        manifest = root / "manifests" / f"{split}.csv"
        if manifest.exists():
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    source = row.get("source_video", "").strip()
                    if source:
                        source_splits.setdefault(source, set()).add(split)
    for digest, paths in hashes.items():
        if len(paths) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "files": paths,
                    "cross_split": len({s for s, _ in paths}) > 1,
                }
            )
    for source, splits in source_splits.items():
        if len(splits) > 1:
            issues.append(
                {
                    "split": ",".join(sorted(splits)),
                    "file": source,
                    "line": 0,
                    "code": "source_video_cross_split",
                    "message": "one source video appears in multiple splits",
                }
            )
    summary = {
        "schema_version": 1,
        "task": task,
        "root": str(root.resolve()),
        "splits": split_rows,
        "images": sum(row["images"] for row in split_rows.values()),
        "labels": sum(row["labels"] for row in split_rows.values()),
        "instances": sum(row["instances"] for row in split_rows.values()),
        "invalid_count": len(issues),
        "duplicate_groups": len(duplicates),
        "cross_split_duplicate_groups": sum(
            bool(row["cross_split"]) for row in duplicates
        ),
    }
    return summary, issues, duplicates


def write_audit(
    output: Path,
    summary: dict[str, Any],
    issues: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "invalid_labels.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "file", "line", "code", "message"]
        )
        writer.writeheader()
        writer.writerows(issues)
    with (output / "duplicate_images.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "cross_split", "split", "file"])
        for row in duplicates:
            for split, file in row["files"]:
                writer.writerow([row["sha256"], row["cross_split"], split, file])
    lines = [
        "# 数据集审计报告",
        "",
        f"- 任务：`{summary['task']}`",
        f"- 图片：{summary['images']}",
        f"- 标签：{summary['labels']}",
        f"- 实例：{summary['instances']}",
        f"- 无效项：{summary['invalid_count']}",
        f"- 重复哈希组：{summary['duplicate_groups']}",
        f"- 跨集合重复组：{summary['cross_split_duplicate_groups']}",
        "",
        "| split | images | labels | instances | empty labels |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, row in summary["splits"].items():
        lines.append(
            f"| {split} | {row['images']} | {row['labels']} | {row['instances']} | {row['empty_labels']} |"
        )
    (output / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
