#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo26_dual.training import validate_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("datasets/tray_detect/data.yaml"))
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--output", type=Path, default=Path("runs/yolo26_tray_detect/validation")
    )
    a = p.parse_args()
    validate_model(
        weights=a.weights,
        data=a.data,
        task="detect",
        split=a.split,
        imgsz=a.imgsz,
        conf=a.conf,
        iou=a.iou,
        device=a.device,
        output=a.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
