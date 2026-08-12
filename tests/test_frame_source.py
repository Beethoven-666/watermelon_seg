from pathlib import Path
import cv2
import numpy as np
from yolo26_dual.frame_source import FrameSource


def test_image_source_is_read_exactly_once(tmp_path: Path):
    path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(path), np.zeros((12, 20, 3), np.uint8))
    source = FrameSource(path)
    try:
        ok, frame, stamp = source.read()
        assert ok and frame.shape == (12, 20, 3) and stamp == 0.0
        assert source.info.width == 20 and source.info.height == 12
        assert source.read()[0] is False
    finally:
        source.close()
