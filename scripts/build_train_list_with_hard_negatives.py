#!/usr/bin/env python
"""Build a YOLO data config that appends hard-negative images to a train list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-train-list", type=Path, required=True)
    parser.add_argument("--hard-negative-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_train = read_lines(args.base_train_list)
    hard_negative = read_lines(args.hard_negative_list)
    combined = base_train + hard_negative

    train_list = output_dir / "train_images_hard_oversample_with_negatives.txt"
    data_yaml = output_dir / "data.yaml"
    summary_json = output_dir / "summary.json"

    write_lines(train_list, combined)
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.dataset_root.as_posix()}",
                f"train: {train_list.as_posix()}",
                f"val: {args.val_list.as_posix()}",
                f"test: {args.test_list.as_posix()}",
                "",
                "names:",
                "  0: watermelon",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "dataset_root": args.dataset_root.as_posix(),
        "base_train_entries": len(base_train),
        "hard_negative_entries": len(hard_negative),
        "combined_train_entries": len(combined),
        "files": {
            "train": train_list.as_posix(),
            "val": args.val_list.as_posix(),
            "test": args.test_list.as_posix(),
            "data_yaml": data_yaml.as_posix(),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
