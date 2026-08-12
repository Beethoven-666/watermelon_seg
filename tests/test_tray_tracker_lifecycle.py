from pathlib import Path
from types import SimpleNamespace
import numpy as np
from yolo26_dual.roi import PixelROI
from yolo26_dual.tray_tracker import TrayTracker


class EmptyBoxes:
    def __len__(self):
        return 0


class Model:
    task = "detect"
    names = {0: "tray"}

    def __init__(self):
        self.calls = []

    def track(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=EmptyBoxes())]


def test_same_model_and_persist_true_are_used_for_continuous_frames(tmp_path: Path):
    weight = tmp_path / "tray.pt"
    weight.write_bytes(b"x")
    tracker = tmp_path / "tracker.yaml"
    tracker.write_text("tracker_type: bytetrack\n")
    created = []

    def factory(_):
        m = Model()
        created.append(m)
        return m

    subject = TrayTracker(
        weight, tracker, 640, 0.2, 0.7, "cpu", True, model_factory=factory
    )
    frame = np.zeros((20, 20, 3), np.uint8)
    roi = PixelROI(0, 0, 20, 20)
    subject.update(frame, roi, 0, 0)
    subject.update(frame, roi, 1, 1)
    assert len(created) == 1 and len(created[0].calls) == 2
    assert all(call["persist"] is True for call in created[0].calls)
    assert all(call["half"] is False for call in created[0].calls)
    subject.reset()
    assert len(created) == 2
