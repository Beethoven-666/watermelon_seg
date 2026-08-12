#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cv2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--output", type=Path, default=Path("datasets/tray_detect/unlabeled")
    )
    p.add_argument("--every-n-frames", type=int, default=10)
    a = p.parse_args()
    if not a.source.is_file():
        raise FileNotFoundError(a.source)
    if a.every_n_frames <= 0:
        raise ValueError("every-n-frames must be positive")
    a.output.mkdir(parents=True, exist_ok=True)
    images = a.output / "images"
    images.mkdir(exist_ok=True)
    manifest = a.output / "manifest.csv"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite {manifest}")
    cap = cv2.VideoCapture(str(a.source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {a.source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % a.every_n_frames == 0:
                name = f"{a.source.stem}_{index:08d}.jpg"
                path = images / name
                if path.exists() or not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"failed to write {path}")
                rows.append(
                    {
                        "image": name,
                        "source_video": str(a.source.resolve()),
                        "frame_index": index,
                        "timestamp_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                        "width": width,
                        "height": height,
                        "every_n_frames": a.every_n_frames,
                    }
                )
            index += 1
    finally:
        cap.release()
    with manifest.open("x", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(
            h,
            fieldnames=[
                "image",
                "source_video",
                "frame_index",
                "timestamp_ms",
                "width",
                "height",
                "every_n_frames",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"extracted={len(rows)} manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
