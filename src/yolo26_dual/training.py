"""YOLO26 训练与验证命令的共享实现。"""

from __future__ import annotations
import json
from pathlib import Path
import csv
import time
from typing import Any
from .model_utils import sha256_file


def train_model(
    *,
    model_path: str,
    data: Path,
    task: str,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    workers: int,
    patience: int,
    seed: int,
    project: Path,
    name: str,
) -> Any:
    from ultralytics import YOLO

    model = YOLO(model_path)
    if model.task != task:
        raise ValueError(f"training model task must be {task}, got {model.task}")
    return model.train(
        data=str(data.resolve()),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        seed=seed,
        project=str(project.resolve()),
        name=name,
        exist_ok=False,
    )


def environment_versions() -> dict[str, Any]:
    import cv2
    import torch
    import ultralytics
    import platform
    import sys

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "opencv": cv2.__version__,
    }


def validate_model(
    *,
    weights: Path,
    data: Path,
    task: str,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    output: Path,
) -> dict[str, Any]:
    from ultralytics import YOLO
    import torch

    model = YOLO(str(weights))
    if model.task != task:
        raise ValueError(f"validation model task must be {task}, got {model.task}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    metrics = model.val(
        data=str(data.resolve()),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        project=str(output.parent.resolve()),
        name=output.name,
        exist_ok=False,
    )
    speed = {str(k): float(v) for k, v in getattr(metrics, "speed", {}).items()}
    import yaml

    data_payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    dataset_root = Path(data_payload.get("path", data.parent))
    dataset_root = (
        dataset_root if dataset_root.is_absolute() else data.parent / dataset_root
    )
    image_dir = dataset_root / str(data_payload[split])
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(
        path for path in image_dir.glob("*") if path.suffix.lower() in suffixes
    )
    per_image = []
    for image_path in images:
        tick = time.perf_counter()
        prediction = model.predict(
            str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - tick) * 1000
        boxes = getattr(prediction, "boxes", None)
        per_image.append(
            {
                "image": str(image_path),
                "inference_ms": elapsed_ms,
                "predictions": len(boxes) if boxes is not None else 0,
            }
        )
    ordered = sorted(row["inference_ms"] for row in per_image)
    p95 = (
        ordered[min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))]
        if ordered
        else 0.0
    )
    label_dir = dataset_root / "labels" / split
    instance_count = sum(
        len(
            [
                line
                for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        )
        for path in label_dir.glob("*.txt")
    )
    empty_count = sum(
        not path.read_text(encoding="utf-8-sig").strip()
        for path in label_dir.glob("*.txt")
    )
    payload = {
        "task": task,
        "split": split,
        "weights": str(weights.resolve()),
        "weights_sha256": sha256_file(weights),
        "data": str(data.resolve()),
        "data_sha256": sha256_file(data),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "image_count": len(images),
        "instance_count": instance_count,
        "empty_background_image_count": empty_count,
        "speed_ms_per_image": speed,
        "measured_mean_inference_ms": sum(ordered) / len(ordered) if ordered else 0.0,
        "measured_p95_inference_ms": p95,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "environment": environment_versions(),
    }
    box = getattr(metrics, "box", None)
    seg = getattr(metrics, "seg", None)
    if box is not None:
        payload["box"] = {
            key: float(getattr(box, key)) for key in ("mp", "mr", "map50", "map")
        }
    if seg is not None:
        payload["mask"] = {
            key: float(getattr(seg, key)) for key in ("mp", "mr", "map50", "map")
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "per_image_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image", "inference_ms", "predictions"]
        )
        writer.writeheader()
        writer.writerows(per_image)
    (output / "validation_report.md").write_text(
        "# YOLO26 验证报告\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return payload
