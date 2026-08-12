"""双模型运行配置加载与验证。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .roi import validate_normalized_roi


VALID_MODES = {"watermelon", "tray", "dual"}


def _resolve(repo_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (repo_root / path).resolve())


def load_runtime_config(path: Path, *, mode: str | None = None) -> dict[str, Any]:
    config_path = path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("runtime config must be a version 1 mapping")
    config = deepcopy(payload)
    repo_root = config_path.parent.parent
    for roi_name in ("watermelon", "tray"):
        config["rois"][roi_name] = list(
            validate_normalized_roi(config["rois"][roi_name])
        )
    selected_mode = mode or config["runtime"].get("mode", "dual")
    if selected_mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
    config["runtime"]["mode"] = selected_mode
    interval = int(config["runtime"]["watermelon_every_n_frames"])
    if interval < 0:
        raise ValueError("watermelon_every_n_frames must be >= 0")
    if int(config["runtime"]["tray_every_n_frames"]) != 1:
        raise ValueError("tray_every_n_frames must remain 1 for continuous tracking")
    for model_name in ("watermelon", "tray"):
        model = config["models"][model_name]
        model["weights"] = _resolve(repo_root, model["weights"])
        if (
            not 0.0 <= float(model["conf"]) <= 1.0
            or not 0.0 <= float(model["iou"]) <= 1.0
        ):
            raise ValueError(f"invalid confidence/IoU for {model_name}")
    config["tracker"]["config"] = _resolve(repo_root, config["tracker"]["config"])
    config["output"]["root"] = _resolve(repo_root, config["output"]["root"])
    return config


class WatermelonScheduler:
    def __init__(self, every_n_frames: int):
        if every_n_frames < 0:
            raise ValueError("every_n_frames must be >= 0")
        self.every_n_frames = every_n_frames
        self._has_run = False
        self._manual_refresh = False

    def request_refresh(self) -> None:
        self._manual_refresh = True

    def should_run(self, frame_index: int) -> bool:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        should = self._manual_refresh or (not self._has_run)
        if self.every_n_frames > 0 and frame_index % self.every_n_frames == 0:
            should = True
        if should:
            self._has_run = True
            self._manual_refresh = False
        return should
