from types import SimpleNamespace
import numpy as np
import pytest
from yolo26_dual.roi import PixelROI
from yolo26_dual.watermelon_segmenter import parse_watermelon_result


def result(mask=True, zero=False):
    boxes = SimpleNamespace(
        xyxy=np.array([[1, 2, 8, 9]], dtype=float),
        conf=np.array([0.9]),
        cls=np.array([0.0]),
        __len__=lambda self: 1,
    )
    # SimpleNamespace special methods are class-defined, use tiny class below.
    boxes = type(
        "Boxes",
        (),
        {
            "xyxy": boxes.xyxy,
            "conf": boxes.conf,
            "cls": boxes.cls,
            "__len__": lambda self: 1,
        },
    )()
    if not mask:
        return SimpleNamespace(boxes=boxes, masks=None)
    data = np.zeros((1, 10, 10), dtype=float)
    data[0, 2:6, 3:7] = 0 if zero else 1
    masks = SimpleNamespace(xy=[np.array([[1, 2], [8, 2], [8, 9]])], data=data)
    return SimpleNamespace(boxes=boxes, masks=masks)


def test_mask_polygon_centroid_and_area_are_full_frame_coordinates():
    items = parse_watermelon_result(
        result(),
        roi=PixelROI(20, 10, 40, 30),
        frame_width=50,
        frame_height=40,
        frame_index=2,
        timestamp_ms=4.0,
    )
    assert len(items) == 1
    item = items[0]
    assert item.bbox_xyxy_full == (21.0, 12.0, 28.0, 19.0)
    assert item.polygon_full[0] == (21.0, 12.0)
    assert item.centroid_full == (29.0, 17.0)
    assert item.mask_area_pixels == 64


def test_missing_masks_are_rejected_and_zero_mask_falls_back():
    with pytest.raises(ValueError, match="no instance masks"):
        parse_watermelon_result(
            result(False),
            roi=PixelROI(0, 0, 10, 10),
            frame_width=10,
            frame_height=10,
            frame_index=0,
            timestamp_ms=0,
        )
    item = parse_watermelon_result(
        result(zero=True),
        roi=PixelROI(0, 0, 10, 10),
        frame_width=10,
        frame_height=10,
        frame_index=0,
        timestamp_ms=0,
    )[0]
    assert item.centroid_full == (4.5, 5.5)
    assert item.centroid_fallback_reason
