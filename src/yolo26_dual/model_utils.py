"""Ultralytics 模型与运行环境的共享检查。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_half(device: Any, requested: bool) -> bool:
    if not requested:
        return False
    text = str(device).lower()
    if text == "cpu":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise ValueError("model class names are unavailable")


def validate_model_contract(
    model: Any, *, task: str, class_id: int, class_name: str
) -> None:
    actual_task = getattr(model, "task", None)
    if actual_task != task:
        raise ValueError(f"model task must be {task!r}, got {actual_task!r}")
    names = normalize_names(getattr(model, "names", None))
    if names != {class_id: class_name}:
        raise ValueError(
            f"custom model must contain exactly {{{class_id}: {class_name!r}}}, got {names!r}"
        )
