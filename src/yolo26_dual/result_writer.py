"""原子运行目录、JSONL 与汇总输出。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any, TextIO

from .types import FrameResult, dumps_result


def create_run_directory(root: Path, prefix: str = "run") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(exist_ok=False)
    (path / "logs").mkdir()
    return path


class JsonlResultWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle: TextIO = path.open("x", encoding="utf-8", newline="\n")

    def write(self, result: FrameResult) -> None:
        self._handle.write(dumps_result(result) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "JsonlResultWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(encoded, encoding="utf-8")
    temp.replace(path)
