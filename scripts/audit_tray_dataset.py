#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo26_dual.dataset_audit import audit_dataset, write_audit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/tray_detect"))
    p.add_argument(
        "--output", type=Path, default=Path("runs/yolo26_tray_detect/dataset_audit")
    )
    a = p.parse_args()
    summary, issues, duplicates = audit_dataset(a.root, "detect")
    write_audit(a.output, summary, issues, duplicates)
    print(
        f"images={summary['images']} labels={summary['labels']} instances={summary['instances']} invalid={summary['invalid_count']}"
    )
    print(f"report={a.output / 'audit_report.md'}")
    return (
        1
        if issues or any(summary["splits"][s]["images"] == 0 for s in summary["splits"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
