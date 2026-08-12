#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo26_dual.dataset_audit import audit_dataset, write_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="只读审计西瓜 YOLO 分割数据集")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/yolo26_watermelon_seg/dataset_audit")
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    summary, issues, duplicates = audit_dataset(args.root, "segment")
    write_audit(output, summary, issues, duplicates)
    print(
        f"images={summary['images']} labels={summary['labels']} instances={summary['instances']} invalid={summary['invalid_count']} cross_split_duplicates={summary['cross_split_duplicate_groups']}"
    )
    print(f"report={output / 'audit_report.md'}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
