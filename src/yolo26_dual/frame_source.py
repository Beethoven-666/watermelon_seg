"""OpenCV 视频/摄像头帧源。文件逐帧读取，摄像头只保留最新帧。"""

from __future__ import annotations
from dataclasses import dataclass
import queue
import threading
from pathlib import Path
from typing import Any
import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    is_live: bool


def parse_source(value: Any) -> int | str:
    text = str(value)
    return int(text) if text.isdecimal() else text


class FrameSource:
    def __init__(
        self,
        value: Any,
        requested_width: int = 0,
        requested_height: int = 0,
        requested_fps: float = 0,
    ):
        self.value = parse_source(value)
        self._image = None
        self._image_read = False
        if isinstance(self.value, str) and Path(self.value).suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }:
            self._image = cv2.imread(self.value)
            if self._image is None:
                raise RuntimeError(f"cannot open image source: {value}")
            self.cap = None
            self.is_live = False
            height, width = self._image.shape[:2]
            self.info = VideoInfo(width, height, 1.0, False)
            self._queue = queue.Queue(maxsize=1)
            self._stop = threading.Event()
            self._thread = None
            self.skipped_frames = 0
            return
        self.cap = cv2.VideoCapture(self.value)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source: {value}")
        self.is_live = isinstance(self.value, int)
        if self.is_live:
            if requested_width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
            if requested_height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
            if requested_fps:
                self.cap.set(cv2.CAP_PROP_FPS, requested_fps)
        self.info = VideoInfo(
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(self.cap.get(cv2.CAP_PROP_FPS)),
            self.is_live,
        )
        if self.info.width <= 0 or self.info.height <= 0:
            raise RuntimeError("video source has invalid dimensions")
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self.skipped_frames = 0
        if self.is_live:
            self._thread = threading.Thread(target=self._capture_latest, daemon=True)
            self._thread.start()

    def _capture_latest(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._stop.set()
                break
            item = (frame, float(self.cap.get(cv2.CAP_PROP_POS_MSEC)))
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.skipped_frames += 1
                except queue.Empty:
                    pass
                self._queue.put_nowait(item)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self._image is not None:
            if self._image_read:
                return False, None, 0.0
            self._image_read = True
            return True, self._image.copy(), 0.0
        if self.is_live:
            try:
                frame, stamp = self._queue.get(timeout=2.0)
                return True, frame, stamp
            except queue.Empty:
                return False, None, 0.0
        ok, frame = self.cap.read()
        return ok, frame if ok else None, float(self.cap.get(cv2.CAP_PROP_POS_MSEC))

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
