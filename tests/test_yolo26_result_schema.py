import json
import math

import pytest

from yolo26_dual.types import (
    FrameResult,
    TrayTrack,
    WatermelonInstance,
    dumps_result,
    to_json_dict,
)
from yolo26_dual.result_writer import JsonlResultWriter


def make_result(processing_fps: float = 30.0) -> FrameResult:
    return FrameResult(
        frame_index=1,
        timestamp_ms=33.3,
        watermelons=[
            WatermelonInstance(
                0, 0.9, (1.0, 2.0, 3.0, 4.0), (2.0, 3.0), [(1.0, 2.0)], 10, 1, 33.3
            )
        ],
        tray_tracks=[TrayTrack(7, 0.8, (5.0, 6.0, 7.0, 8.0), (6.0, 7.0), 1, 33.3)],
        unconfirmed_trays=[],
        watermelon_result_age_frames=0,
        capture_fps=60.0,
        processing_fps=processing_fps,
    )


def test_result_is_standard_json_and_empty_arrays_remain_arrays() -> None:
    payload = json.loads(dumps_result(make_result()))
    assert payload["unconfirmed_trays"] == []
    assert isinstance(payload["watermelons"][0]["confidence"], float)
    assert to_json_dict(make_result())["tray_tracks"][0]["track_id"] == 7


def test_non_finite_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="NaN"):
        dumps_result(make_result(math.nan))


def test_numpy_scalar_is_rejected() -> None:
    import numpy as np

    result = make_result()
    object.__setattr__(result, "capture_fps", np.float32(30.0))
    with pytest.raises(TypeError, match="JSON-native"):
        dumps_result(result)


def test_jsonl_writer_produces_independently_parseable_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlResultWriter(path) as writer:
        writer.write(make_result())
        writer.write(make_result())
    assert [
        json.loads(line)["frame_index"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ] == [1, 1]
