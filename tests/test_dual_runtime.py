import numpy as np
from yolo26_dual.runtime import DualRuntime
from yolo26_dual.types import TrayTrack, WatermelonInstance


class Wm:
    def __init__(self):
        self.calls = []

    def predict(self, frame, roi, index, stamp):
        self.calls.append(index)
        return [
            WatermelonInstance(
                0, 0.9, (0.0, 0.0, 1.0, 1.0), (0.5, 0.5), [(0.0, 0.0)], 1, index, stamp
            )
        ]


class Tray:
    def __init__(self):
        self.calls = []

    def update(self, frame, roi, index, stamp):
        self.calls.append(index)
        return [TrayTrack(4, 0.8, (0.0, 0.0, 2.0, 2.0), (1.0, 1.0), index, stamp)], []


def config(mode="dual"):
    return {
        "runtime": {"mode": mode, "watermelon_every_n_frames": 10},
        "rois": {"watermelon": [0, 0, 0.5, 1], "tray": [0.5, 0, 1, 1]},
    }


def test_dual_prioritizes_continuous_tray_and_schedules_watermelon():
    w, t = Wm(), Tray()
    runtime = DualRuntime(config(), watermelon_segmenter=w, tray_tracker=t)
    frame = np.zeros((20, 40, 3), np.uint8)
    results = [runtime.process_frame(frame, i, i * 10, 30, 0.1) for i in range(12)]
    assert t.calls == list(range(12))
    assert w.calls == [0, 10]
    assert results[9].watermelon_result_age_frames == 9
    assert results[-1].tray_tracks[0].track_id == 4


def test_single_modes_do_not_require_other_model():
    frame = np.zeros((20, 40, 3), np.uint8)
    w = Wm()
    DualRuntime(config("watermelon"), watermelon_segmenter=w).process_frame(
        frame, 0, 0, 30, 0.1
    )
    t = Tray()
    DualRuntime(config("tray"), tray_tracker=t).process_frame(frame, 0, 0, 30, 0.1)
    assert w.calls == [0] and t.calls == [0]
