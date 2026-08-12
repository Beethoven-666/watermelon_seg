#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo26_dual.training import train_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("datasets/tray_detect/data.yaml"))
    p.add_argument("--model", default="yolo26s.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--project", type=Path, default=Path("runs/yolo26_tray_detect"))
    p.add_argument("--name", default="train")
    a = p.parse_args()
    train_model(
        model_path=a.model,
        data=a.data,
        task="detect",
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        patience=a.patience,
        seed=a.seed,
        project=a.project,
        name=a.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
