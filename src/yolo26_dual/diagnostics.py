"""环境、数据、模型与视频源就绪性诊断。"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from .training import environment_versions


def dataset_ready(root: Path) -> bool:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return all(
        any(
            path.is_file() and path.suffix.lower() in suffixes
            for path in (root / "images" / split).glob("*")
        )
        and any(
            path.is_file() and path.suffix.lower() == ".txt"
            for path in (root / "labels" / split).glob("*")
        )
        for split in ("train", "val", "test")
    )


def diagnose(
    repo: Path, config: dict[str, Any], source: Any | None = None
) -> dict[str, Any]:
    wm_weights = Path(config["models"]["watermelon"]["weights"])
    tray_weights = Path(config["models"]["tray"]["weights"])
    wm_data = dataset_ready(repo)
    tray_data = dataset_ready(repo / "datasets" / "tray_detect")
    source_ok = None
    if source is not None:
        import cv2

        source_text = str(source)
        if Path(source_text).is_file() and cv2.imread(source_text) is not None:
            source_ok = True
        else:
            cap = cv2.VideoCapture(
                int(source) if source_text.isdecimal() else source_text
            )
            source_ok = cap.isOpened()
            cap.release()
    result = {
        "environment": environment_versions(),
        "official_yolo26_detect_exists": (repo / "yolo26s.pt").is_file(),
        "official_yolo26_segment_exists": (repo / "yolo26s-seg.pt").is_file(),
        "watermelon_weights_exists": wm_weights.is_file(),
        "tray_weights_exists": tray_weights.is_file(),
        "video_source_openable": source_ok,
        "ready_for_watermelon_training": wm_data,
        "ready_for_tray_training": tray_data,
        "ready_for_watermelon_inference": wm_weights.is_file(),
        "ready_for_tray_tracking": tray_weights.is_file(),
        "ready_for_dual_runtime": wm_weights.is_file() and tray_weights.is_file(),
    }
    return result
