#!/usr/bin/env python
"""Build a review queue from YOLO segmentation error-analysis outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a relabel/re-sampling review queue.")
    parser.add_argument("--per-image-tsv", required=True, type=Path)
    parser.add_argument("--overlap-tsv", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


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


def recommended_action(row: dict[str, object]) -> str:
    unreachable = int(row["unreachable_gt"])
    fn = int(row["fn"])
    fp = int(row["fp"])
    gt = int(row["gt"])
    if unreachable >= 2:
        return "复核不可达 GT：优先确认是否为极小碎片、边缘截断或非抓取目标"
    if fn >= 4 and gt >= 8:
        return "补充同类密集/遮挡场景，并复核漏标或实例边界"
    if fp >= 4:
        return "复核误检来源：叶片、藤蔓、阴影、局部纹理或半目标"
    if fn >= 2:
        return "复核小目标和遮挡目标是否符合全实例标注口径"
    return "抽检确认标注口径一致"


def main() -> int:
    args = parse_args()
    per_image_rows = read_tsv(args.per_image_tsv)
    overlap_rows = read_tsv(args.overlap_tsv)

    unreachable_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    min_iou_by_image: dict[str, float] = defaultdict(lambda: 1.0)
    for row in overlap_rows:
        image = row["image"]
        best_iou = float(row["best_iou"])
        min_iou_by_image[image] = min(min_iou_by_image[image], best_iou)
        if row["reachable_at_iou_threshold"].lower() == "false":
            unreachable_by_image[image].append(row)

    queue: list[dict[str, object]] = []
    for row in per_image_rows:
        image = row["image"]
        stem = Path(image).stem
        overlay = ""
        if args.overlay_dir:
            overlay_path = args.overlay_dir / f"{stem}_errors.png"
            if overlay_path.exists():
                overlay = str(overlay_path)
        item = {
            "image": image,
            "source": source_hint(image),
            "image_path": str(args.images_dir / image),
            "overlay_path": overlay,
            "gt": int(row["gt"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "unreachable_gt": len(unreachable_by_image[image]),
            "min_best_iou": min_iou_by_image[image] if image in min_iou_by_image else 1.0,
            "unreachable_gt_indices": ",".join(r["gt_index"] for r in unreachable_by_image[image]),
        }
        item["priority_score"] = (
            int(item["fn"]) * 5
            + int(item["fp"]) * 2
            + int(item["unreachable_gt"]) * 8
            + int(item["gt"])
        )
        item["recommended_action"] = recommended_action(item)
        queue.append(item)

    queue.sort(
        key=lambda item: (
            int(item["priority_score"]),
            int(item["unreachable_gt"]),
            int(item["fn"]),
            int(item["fp"]),
        ),
        reverse=True,
    )
    selected = queue[: args.top]

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "source",
        "image_path",
        "overlay_path",
        "gt",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "unreachable_gt",
        "min_best_iou",
        "unreachable_gt_indices",
        "priority_score",
        "recommended_action",
    ]
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)

    source_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        source_counts[str(item["source"])] += 1

    lines = [
        "# 复标与补样优先队列",
        "",
        "该清单基于 runs6 当前最佳模型在 test 集的实例级错误分析生成，目标是继续提升全实例 Precision/Recall。",
        "",
        "## 摘要",
        "",
        f"- 候选图片数：{len(queue)}",
        f"- 输出 Top：{len(selected)}",
        "- 来源分布："
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"  - {source}: {count}")
    lines.extend(
        [
            "",
            "## Top 复核项",
            "",
            "| 排名 | 图片 | 来源 | GT | FP | FN | 不可达 GT | 建议 |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, item in enumerate(selected[:20], start=1):
        image = str(item["image"])
        lines.append(
            f"| {idx} | `{image}` | {item['source']} | {item['gt']} | {item['fp']} | "
            f"{item['fn']} | {item['unreachable_gt']} | {item['recommended_action']} |"
        )
    lines.extend(
        [
            "",
            "完整 TSV：",
            "",
            f"```text\n{args.output_tsv}\n```",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_tsv)
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
