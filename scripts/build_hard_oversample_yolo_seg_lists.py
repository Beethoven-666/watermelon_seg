#!/usr/bin/env python
"""Build YOLO segmentation train lists that oversample hard-looking images.

The script does not copy or modify images/labels. It writes a text file with
absolute image paths, optionally repeated, plus a small summary that explains
why each image was repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LabelStats:
    instances: int
    min_area: float
    small_instances: int
    tiny_instances: int
    edge_instances: int
    repeat: int
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--train-images-dir", type=Path, required=True)
    parser.add_argument("--val-images-dir", type=Path, required=True)
    parser.add_argument("--test-images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--small-area", type=float, default=0.005)
    parser.add_argument("--tiny-area", type=float, default=0.001)
    parser.add_argument("--edge-margin", type=float, default=0.02)
    parser.add_argument("--max-repeat", type=int, default=5)
    parser.add_argument("--include-val-in-train", action="store_true")
    return parser.parse_args()


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def read_instances(label_path: Path) -> list[list[tuple[float, float]]]:
    if not label_path.exists():
        return []
    instances: list[list[tuple[float, float]]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 7:
            continue
        coords = [float(value) for value in parts[1:]]
        if len(coords) < 6 or len(coords) % 2:
            continue
        points = [(max(0.0, min(1.0, coords[i])), max(0.0, min(1.0, coords[i + 1]))) for i in range(0, len(coords), 2)]
        instances.append(points)
    return instances


def label_path_for_image(dataset_root: Path, image_path: Path) -> Path:
    image_path = image_path.resolve()
    root = dataset_root.resolve()
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    if not parts or parts[0] != "images":
        raise ValueError(f"Image path is not under images/: {image_path}")
    parts[0] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def touches_edge(points: list[tuple[float, float]], margin: float) -> bool:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs) <= margin or min(ys) <= margin or max(xs) >= 1.0 - margin or max(ys) >= 1.0 - margin


def stats_for_image(dataset_root: Path, image_path: Path, args: argparse.Namespace) -> LabelStats:
    instances = read_instances(label_path_for_image(dataset_root, image_path))
    areas = [polygon_area(points) for points in instances]
    small_instances = sum(1 for area in areas if area < args.small_area)
    tiny_instances = sum(1 for area in areas if area < args.tiny_area)
    edge_instances = sum(1 for points in instances if touches_edge(points, args.edge_margin))
    min_area = min(areas) if areas else 0.0

    repeat = 1
    reasons: list[str] = []
    if len(instances) >= 8:
        repeat += 2
        reasons.append("dense_instances>=8")
    elif len(instances) >= 5:
        repeat += 1
        reasons.append("multi_instances>=5")
    if small_instances >= 3:
        repeat += 2
        reasons.append("small_instances>=3")
    elif small_instances >= 1:
        repeat += 1
        reasons.append("small_instances>=1")
    if tiny_instances >= 1:
        repeat += 1
        reasons.append("tiny_instances>=1")
    if edge_instances >= 2:
        repeat += 1
        reasons.append("edge_instances>=2")
    if image_path.name.startswith("roboflow_team128") and (small_instances or len(instances) >= 4):
        repeat += 1
        reasons.append("team128_hard_source")

    repeat = max(1, min(args.max_repeat, repeat))
    if repeat == 1:
        reasons.append("base")
    return LabelStats(
        instances=len(instances),
        min_area=min_area,
        small_instances=small_instances,
        tiny_instances=tiny_instances,
        edge_instances=edge_instances,
        repeat=repeat,
        reasons=tuple(reasons),
    )


def image_paths(directory: Path) -> list[Path]:
    return sorted(path.resolve() for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


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
    dataset_root = args.dataset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_sources = image_paths(args.train_images_dir)
    if args.include_val_in_train:
        train_sources += image_paths(args.val_images_dir)

    rows: list[dict[str, object]] = []
    oversampled: list[str] = []
    repeat_histogram: dict[int, int] = {}
    for image_path in train_sources:
        stats = stats_for_image(dataset_root, image_path, args)
        repeat_histogram[stats.repeat] = repeat_histogram.get(stats.repeat, 0) + 1
        oversampled.extend(str(image_path).replace("\\", "/") for _ in range(stats.repeat))
        rows.append(
            {
                "image": str(image_path).replace("\\", "/"),
                "instances": stats.instances,
                "min_area": f"{stats.min_area:.8f}",
                "small_instances": stats.small_instances,
                "tiny_instances": stats.tiny_instances,
                "edge_instances": stats.edge_instances,
                "repeat": stats.repeat,
                "reasons": ",".join(stats.reasons),
            }
        )

    val_list = [str(path).replace("\\", "/") for path in image_paths(args.val_images_dir)]
    test_list = [str(path).replace("\\", "/") for path in image_paths(args.test_images_dir)]

    train_list_path = args.output_dir / "train_images_hard_oversample.txt"
    val_list_path = args.output_dir / "val_images.txt"
    test_list_path = args.output_dir / "test_images.txt"
    train_list_path.write_text("\n".join(oversampled) + "\n", encoding="utf-8")
    val_list_path.write_text("\n".join(val_list) + "\n", encoding="utf-8")
    test_list_path.write_text("\n".join(test_list) + "\n", encoding="utf-8")
    write_tsv(args.output_dir / "oversample_manifest.tsv", rows)

    data_yaml = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            f"train: {train_list_path.as_posix()}",
            f"val: {val_list_path.as_posix()}",
            f"test: {test_list_path.as_posix()}",
            "",
            "names:",
            "  0: watermelon",
            "",
        ]
    )
    (args.output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "include_val_in_train": bool(args.include_val_in_train),
        "source_train_images": len(train_sources),
        "oversampled_train_entries": len(oversampled),
        "val_images": len(val_list),
        "test_images": len(test_list),
        "small_area": args.small_area,
        "tiny_area": args.tiny_area,
        "edge_margin": args.edge_margin,
        "max_repeat": args.max_repeat,
        "repeat_histogram": dict(sorted(repeat_histogram.items())),
        "files": {
            "data_yaml": (args.output_dir / "data.yaml").as_posix(),
            "train_list": train_list_path.as_posix(),
            "val_list": val_list_path.as_posix(),
            "test_list": test_list_path.as_posix(),
            "manifest": (args.output_dir / "oversample_manifest.tsv").as_posix(),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
