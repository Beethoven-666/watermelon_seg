import numpy as np
import pytest

from yolo26_dual.roi import (
    PixelROI,
    crop_frame,
    map_box_to_full,
    map_points_to_full,
    normalized_to_pixel_roi,
)


def test_normalized_roi_to_pixels_and_crop() -> None:
    roi = normalized_to_pixel_roi([0.25, 0.20, 0.75, 0.80], 100, 50)
    assert roi == PixelROI(25, 10, 75, 40)
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    assert crop_frame(frame, roi).shape == (30, 50, 3)


@pytest.mark.parametrize(
    "value", [[-0.1, 0, 1, 1], [0, 0, 0, 1], [0, 0, 1.1, 1], [0, 0, 1]]
)
def test_invalid_roi_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        normalized_to_pixel_roi(value, 100, 50)


def test_box_and_polygon_map_to_full_and_clip() -> None:
    roi = PixelROI(40, 10, 90, 50)
    assert map_box_to_full([-5, 2, 70, 60], roi, 100, 60) == (35.0, 12.0, 100.0, 60.0)
    assert map_points_to_full([[-50, -20], [10, 5], [100, 100]], roi, 100, 60) == [
        (0.0, 0.0),
        (50.0, 15.0),
        (100.0, 60.0),
    ]
