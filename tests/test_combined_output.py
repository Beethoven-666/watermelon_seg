import json
from pathlib import Path
import cv2
import numpy as np
from yolo26_dual.runtime import run_video
from yolo26_dual.types import TrayTrack, WatermelonInstance


class Wm:
    def predict(self, frame, roi, index, stamp):
        return [
            WatermelonInstance(
                0,
                0.9,
                (2.0, 2.0, 10.0, 10.0),
                (6.0, 6.0),
                [(2.0, 2.0), (10.0, 2.0), (10.0, 10.0)],
                32,
                index,
                stamp,
            )
        ]

    def close(self):
        pass


class Tray:
    history = {4: [(12.0, 12.0)]}

    def update(self, frame, roi, index, stamp):
        return [
            TrayTrack(4, 0.8, (10.0, 10.0, 18.0, 18.0), (14.0, 14.0), index, stamp)
        ], []

    def close(self):
        pass


def test_three_frame_dual_run_writes_video_jsonl_and_summary(tmp_path: Path):
    video_path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24)
    )
    assert writer.isOpened()
    for value in (10, 20, 30):
        writer.write(np.full((24, 32, 3), value, np.uint8))
    writer.release()
    config = {
        "source": {"requested_width": 0, "requested_height": 0, "requested_fps": 0},
        "rois": {"watermelon": [0, 0, 0.6, 1], "tray": [0.4, 0, 1, 1]},
        "models": {
            "watermelon": {"weights": str(tmp_path / "missing-wm.pt")},
            "tray": {"weights": str(tmp_path / "missing-tray.pt")},
        },
        "tracker": {"history_length": 30},
        "runtime": {"mode": "dual", "watermelon_every_n_frames": 10},
        "visualization": {"show_window": False},
        "output": {
            "root": str(tmp_path / "runs"),
            "save_jsonl": True,
            "save_annotated_video": True,
        },
    }
    output = run_video(config, str(video_path), Wm(), Tray())
    lines = (output / "events.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["tray_tracks"][0]["track_id"] == 4 for line in lines)
    summary = json.loads((output / "runtime_summary.json").read_text())
    assert summary["processed_frames"] == 3 and summary["tray_inference_count"] == 3
    assert (output / "annotated.mp4").stat().st_size > 0
