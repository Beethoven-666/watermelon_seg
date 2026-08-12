from types import SimpleNamespace
import numpy as np
from yolo26_dual.roi import PixelROI
from yolo26_dual.tray_tracker import parse_tray_result


class Boxes:
    def __init__(self, ids):
        self.xyxy = np.array([[1, 2, 5, 6], [10, 10, 20, 20]], float)
        self.conf = np.array([0.8, 0.6])
        self.cls = np.array([0, 0])
        self.id = ids

    def __len__(self):
        return 2


def test_track_ids_and_unconfirmed_are_separated_with_roi_offset():
    tracks, unconfirmed = parse_tray_result(
        SimpleNamespace(boxes=Boxes(np.array([7, float("nan")]))),
        roi=PixelROI(30, 10, 80, 50),
        frame_width=100,
        frame_height=60,
        frame_index=3,
        timestamp_ms=5,
    )
    assert tracks[0].track_id == 7 and tracks[0].bbox_xyxy_full == (
        31.0,
        12.0,
        35.0,
        16.0,
    )
    assert tracks[0].center_full == (33.0, 14.0)
    assert len(unconfirmed) == 1


def test_none_ids_make_unconfirmed_and_empty_result_is_safe():
    tracks, detections = parse_tray_result(
        SimpleNamespace(boxes=Boxes(None)),
        roi=PixelROI(0, 0, 50, 50),
        frame_width=50,
        frame_height=50,
        frame_index=0,
        timestamp_ms=0,
    )
    assert tracks == [] and len(detections) == 2
    assert parse_tray_result(
        SimpleNamespace(boxes=None),
        roi=PixelROI(0, 0, 50, 50),
        frame_width=50,
        frame_height=50,
        frame_index=0,
        timestamp_ms=0,
    ) == ([], [])
