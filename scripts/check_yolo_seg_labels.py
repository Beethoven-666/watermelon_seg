#!/usr/bin/env python
"""Validate and visualize a YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SPLITS = ("train", "val", "test")
VALID_CLASS_IDS = {0}


@dataclass
class Issue:
    severity: str
    split: str
    image: str
    label: str
    line: int
    code: str
    message: str


@dataclass
class Instance:
    class_id: int
    points: list[tuple[float, float]]
    line: int
    area_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan images/train,val,test and labels/train,val,test, validate YOLO "
            "segmentation labels, and write polygon visualization images."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Dataset root directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/yolo_seg_label_check"),
        help="Directory for reports and visualization images.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to scan.",
    )
    parser.add_argument("--class-id", type=int, action="append", dest="class_ids")
    parser.add_argument("--no-visuals", action="store_true", help="Skip visualization output.")
    parser.add_argument("--max-visuals", type=int, default=0, help="Limit visuals; 0 means all.")
    parser.add_argument(
        "--max-side",
        type=int,
        default=1400,
        help="Resize visualization so its longest side is at most this value.",
    )
    parser.add_argument(
        "--area-warn-threshold",
        type=float,
        default=1e-6,
        help="Warn when normalized polygon area is smaller than this value.",
    )
    return parser.parse_args()


def dataset_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def as_posix(path: Path) -> str:
    return path.as_posix()


def find_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def find_labels(labels_dir: Path) -> list[Path]:
    if not labels_dir.exists():
        return []
    return sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def add_issue(
    issues: list[Issue],
    severity: str,
    split: str,
    image: Path | str,
    label: Path | str,
    line: int,
    code: str,
    message: str,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            split=split,
            image=as_posix(image) if isinstance(image, Path) else image,
            label=as_posix(label) if isinstance(label, Path) else label,
            line=line,
            code=code,
            message=message,
        )
    )


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def parse_label(
    label_path: Path,
    split: str,
    image_path: Path,
    valid_class_ids: set[int],
    area_warn_threshold: float,
    issues: list[Issue],
) -> list[Instance]:
    instances: list[Instance] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        add_issue(
            issues,
            "error",
            split,
            image_path,
            label_path,
            0,
            "label_decode_failed",
            "Label file is not valid UTF-8.",
        )
        return instances

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            add_issue(
                issues,
                "warning",
                split,
                image_path,
                label_path,
                line_number,
                "blank_line",
                "Blank label line.",
            )
            continue

        parts = line.split()
        if len(parts) < 7:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "too_few_values",
                "YOLO segmentation row needs class_id plus at least 3 points.",
            )
            continue
        if (len(parts) - 1) % 2 != 0:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "odd_coordinate_count",
                "Coordinate count is odd; x/y pairs are incomplete.",
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "invalid_class_id",
                f"Class id {parts[0]!r} is not an integer.",
            )
            continue

        if class_id not in valid_class_ids:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "unexpected_class_id",
                f"Class id {class_id} is not in {sorted(valid_class_ids)}.",
            )

        try:
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "invalid_coordinate",
                f"Could not parse all coordinates as floats: {exc}.",
            )
            continue

        if not all(math.isfinite(value) for value in values):
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "non_finite_coordinate",
                "Coordinate contains NaN or infinity.",
            )
            continue

        points = list(zip(values[0::2], values[1::2]))
        out_of_range = [
            (index + 1, x, y)
            for index, (x, y) in enumerate(points)
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0
        ]
        if out_of_range:
            first_index, x, y = out_of_range[0]
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "coordinate_out_of_range",
                f"Point {first_index} is outside 0..1: ({x}, {y}).",
            )

        unique_points = {(round(x, 10), round(y, 10)) for x, y in points}
        if len(unique_points) < 3:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "too_few_unique_points",
                "Polygon has fewer than 3 unique points.",
            )

        area = polygon_area(points)
        if area <= 0.0:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path,
                line_number,
                "zero_area_polygon",
                "Polygon area is zero.",
            )
        elif area < area_warn_threshold:
            add_issue(
                issues,
                "warning",
                split,
                image_path,
                label_path,
                line_number,
                "tiny_polygon",
                f"Normalized polygon area {area:.8f} is very small.",
            )

        instances.append(Instance(class_id=class_id, points=points, line=line_number, area_norm=area))

    return instances


def scale_image(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0:
        return image.copy(), 1.0
    width, height = image.size
    scale = min(1.0, float(max_side) / float(max(width, height)))
    if scale >= 1.0:
        return image.copy(), 1.0
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_label_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def render_visual(
    image_path: Path,
    output_path: Path,
    instances: list[Instance],
    image_issues: list[Issue],
    max_side: int,
) -> bool:
    try:
        with Image.open(image_path) as source:
            source.load()
            image, _scale = scale_image(source.convert("RGB"), max_side)
    except Exception:
        return False

    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(max(width, height) / 300))
    font = safe_font(max(12, round(max(width, height) / 55)))
    palette = [
        (0, 255, 120),
        (255, 210, 0),
        (0, 200, 255),
        (255, 80, 120),
        (190, 140, 255),
    ]

    for index, instance in enumerate(instances, start=1):
        color = palette[(index - 1) % len(palette)]
        pixels = [
            (
                int(round(min(max(x, 0.0), 1.0) * (width - 1))),
                int(round(min(max(y, 0.0), 1.0) * (height - 1))),
            )
            for x, y in instance.points
        ]
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=(*color, 55))
            draw.line(pixels + [pixels[0]], fill=(*color, 255), width=line_width)
            first_x, first_y = pixels[0]
            draw_label_box(
                draw,
                (max(0, first_x + line_width), max(0, first_y + line_width)),
                f"#{index} c{instance.class_id} L{instance.line}",
                font,
            )

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw_rgb = ImageDraw.Draw(image)
    header = f"{image_path.name} | instances: {len(instances)}"
    if image_issues:
        header += f" | issues: {len(image_issues)}"
    draw_label_box(draw_rgb, (8, 8), header, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)
    return True


def write_tsv(path: Path, issues: list[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "split", "image", "label", "line", "code", "message"],
            delimiter="\t",
        )
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_issues(issues: Iterable[Issue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        key = f"{issue.severity}:{issue.code}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_markdown(path: Path, summary: dict, issues: list[Issue]) -> None:
    lines = [
        "# YOLO segmentation label check",
        "",
        "## Summary",
        "",
        f"- Images: {summary['images']}",
        f"- Labels: {summary['labels']}",
        f"- Instances: {summary['instances']}",
        f"- Errors: {summary['errors']}",
        f"- Warnings: {summary['warnings']}",
        f"- Visualizations: {summary['visualizations']}",
        "",
        "## Split counts",
        "",
        "| split | images | labels | instances | errors | warnings | visuals |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, row in summary["splits"].items():
        lines.append(
            f"| {split} | {row['images']} | {row['labels']} | {row['instances']} | "
            f"{row['errors']} | {row['warnings']} | {row['visualizations']} |"
        )

    lines.extend(["", "## Issue counts", ""])
    issue_counts = summary["issue_counts"]
    if issue_counts:
        for key, count in issue_counts.items():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- No machine-checkable issues found.")

    lines.extend(["", "## First issues", ""])
    if issues:
        lines.append("| severity | split | line | code | image | message |")
        lines.append("| --- | --- | ---: | --- | --- | --- |")
        for issue in issues[:200]:
            image_name = Path(issue.image).name if issue.image else ""
            safe_message = issue.message.replace("|", "\\|")
            lines.append(
                f"| {issue.severity} | {issue.split} | {issue.line} | {issue.code} | "
                f"{image_name} | {safe_message} |"
            )
        if len(issues) > 200:
            lines.append("")
            lines.append(f"Only the first 200 issues are shown here; see errors.tsv for all {len(issues)} issues.")
    else:
        lines.append("No machine-checkable issues found.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_split(
    root: Path,
    output_dir: Path,
    split: str,
    valid_class_ids: set[int],
    area_warn_threshold: float,
    max_side: int,
    write_visuals: bool,
    max_visuals: int,
    visual_counter: int,
) -> tuple[dict, list[Issue], int]:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    issues: list[Issue] = []
    images = find_images(images_dir)
    labels = find_labels(labels_dir)
    instances_total = 0
    visuals_total = 0

    if not images_dir.exists():
        add_issue(issues, "error", split, "", "", 0, "missing_images_dir", f"Missing {images_dir}.")
    if not labels_dir.exists():
        add_issue(issues, "error", split, "", "", 0, "missing_labels_dir", f"Missing {labels_dir}.")

    labels_by_stem = {label.stem: label for label in labels}
    images_by_stem: dict[str, list[Path]] = {}
    for image in images:
        images_by_stem.setdefault(image.stem, []).append(image)
    for stem, stem_images in images_by_stem.items():
        if len(stem_images) > 1:
            add_issue(
                issues,
                "error",
                split,
                stem_images[0],
                labels_by_stem.get(stem, ""),
                0,
                "duplicate_image_stem",
                "Multiple images share the same stem, so one YOLO label path is ambiguous.",
            )

    for label in labels:
        if label.stem not in images_by_stem:
            add_issue(
                issues,
                "error",
                split,
                "",
                label,
                0,
                "orphan_label",
                "Label file has no matching image with the same stem.",
            )

    for image_index, image_path in enumerate(images, start=1):
        label_path = labels_by_stem.get(image_path.stem)
        instances: list[Instance] = []
        before_issue_count = len(issues)

        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                label_path or "",
                0,
                "image_open_failed",
                f"Could not open image: {exc}.",
            )

        if label_path is None:
            add_issue(
                issues,
                "error",
                split,
                image_path,
                "",
                0,
                "missing_label",
                "Image has no matching label file.",
            )
        else:
            instances = parse_label(
                label_path,
                split,
                image_path,
                valid_class_ids,
                area_warn_threshold,
                issues,
            )
            instances_total += len(instances)

        image_issues = [
            issue
            for issue in issues[before_issue_count:]
            if issue.image == as_posix(image_path) or issue.label == as_posix(label_path or Path(""))
        ]
        can_write_visual = write_visuals and (max_visuals <= 0 or visual_counter < max_visuals)
        if can_write_visual:
            visual_counter += 1
            visual_name = f"{image_index:06d}_{image_path.stem}.jpg"
            visual_path = output_dir / "visualizations" / split / visual_name
            if render_visual(image_path, visual_path, instances, image_issues, max_side):
                visuals_total += 1
            else:
                add_issue(
                    issues,
                    "error",
                    split,
                    image_path,
                    label_path or "",
                    0,
                    "visualization_failed",
                    "Could not render visualization image.",
                )

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    row = {
        "images": len(images),
        "labels": len(labels),
        "instances": instances_total,
        "errors": errors,
        "warnings": warnings,
        "visualizations": visuals_total,
    }
    return row, issues, visual_counter


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = dataset_path(root, args.output_dir).resolve()
    valid_class_ids = set(args.class_ids) if args.class_ids else set(VALID_CLASS_IDS)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_issues: list[Issue] = []
    split_summary: dict[str, dict] = {}
    visual_counter = 0
    for split in args.splits:
        row, issues, visual_counter = validate_split(
            root=root,
            output_dir=output_dir,
            split=split,
            valid_class_ids=valid_class_ids,
            area_warn_threshold=args.area_warn_threshold,
            max_side=args.max_side,
            write_visuals=not args.no_visuals,
            max_visuals=args.max_visuals,
            visual_counter=visual_counter,
        )
        split_summary[split] = row
        all_issues.extend(issues)

    summary = {
        "root": as_posix(root),
        "output_dir": as_posix(output_dir),
        "valid_class_ids": sorted(valid_class_ids),
        "splits": split_summary,
        "images": sum(row["images"] for row in split_summary.values()),
        "labels": sum(row["labels"] for row in split_summary.values()),
        "instances": sum(row["instances"] for row in split_summary.values()),
        "errors": sum(1 for issue in all_issues if issue.severity == "error"),
        "warnings": sum(1 for issue in all_issues if issue.severity == "warning"),
        "visualizations": sum(row["visualizations"] for row in split_summary.values()),
        "issue_counts": summarize_issues(all_issues),
    }

    write_tsv(output_dir / "errors.tsv", all_issues)
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "report.md", summary, all_issues)

    print(f"Root: {root}")
    print(f"Output: {output_dir}")
    print(
        "Checked {images} images, {labels} labels, {instances} instances. "
        "Errors: {errors}, warnings: {warnings}, visuals: {visualizations}.".format(**summary)
    )
    print(f"Report: {output_dir / 'report.md'}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
