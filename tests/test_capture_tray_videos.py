from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pytest

from scripts import capture_tray_videos as capture


def _make_frame(index: int, width: int = 64, height: int = 48) -> np.ndarray:
    return np.full((height, width, 3), index % 256, dtype=np.uint8)


def _base_profile(width: int = 64, height: int = 48, fps: float = 30.0, fmt: str = "BGR"):
    return capture.ResolvedStreamProfile(
        requested_width=None,
        requested_height=None,
        requested_fps=None,
        requested_format=None,
        actual_width=width,
        actual_height=height,
        actual_fps=fps,
        actual_format=fmt,
    )


def _make_config(
    output_root: Path,
    **overrides,
) -> capture.CaptureConfig:
    base = dict(
        output_root=output_root,
        batch_id="20260819_test",
        serial_number="SN123456",
        list_devices=False,
        list_profiles=False,
        width=None,
        height=None,
        fps=None,
        color_format=None,
        duration_seconds=0.5,
        segment_seconds=0.0,
        warmup_seconds=0.0,
        no_preview=False,
        fixed_test=False,
        planned_split="unassigned",
        lighting="normal",
        scene_tags="test_scene",
        tray_count="0-3",
        camera_moved="no",
        conveyor_running="yes",
        notes="",
        codec="mp4v",
        container="mp4",
        frame_timeout_ms=100,
        log_level="INFO",
    )
    base.update(overrides)
    return capture.CaptureConfig(**base)


class FakeSource:
    def __init__(
        self,
        frames,
        active_profile: capture.ResolvedStreamProfile,
        on_read: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.frames = list(frames)
        self.active_profile = active_profile
        self.on_read = on_read
        self.started = False
        self.stopped = False
        self.read_calls = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def read_frame(self, timeout_ms: int):
        del timeout_ms
        self.read_calls += 1
        if self.on_read is not None:
            self.on_read(self.read_calls)
        if not self.frames:
            return None
        item = self.frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get_active_profile(self) -> capture.ResolvedStreamProfile:
        return self.active_profile


class FakeWriter:
    def __init__(self, opened: bool = True) -> None:
        self._opened = opened
        self.released = False
        self.written = 0

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame) -> None:
        del frame
        if not self._opened:
            raise RuntimeError("writer closed")
        self.written += 1

    def release(self) -> None:
        self.released = True


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_parse_args_validate_batch_id():
    args = capture.parse_args(["--output-root", "tmp", "--duration-seconds", "1"])
    with pytest.raises(capture.CaptureError):
        capture.normalize_args_to_config(args)


def test_fixed_test_forces_fixed_split(tmp_path: Path):
    args = capture.parse_args(
        [
            "--fixed-test",
            "--output-root",
            str(tmp_path / "raw/tray_detect"),
            "--planned-split",
            "train",
            "--duration-seconds",
            "1",
            "--no-preview",
        ]
    )
    config = capture.normalize_args_to_config(args)
    assert config.fixed_test is True
    assert config.planned_split == "fixed_test"


def test_sanitize_token_and_capture_paths_normal_and_fixed(tmp_path: Path):
    assert capture.sanitize_token("a/b:c d?e") == "a_b_c_d_e"
    cfg = _make_config(tmp_path / "raw/tray_detect")
    ts = capture._capture_timestamp()
    paths, _ = capture.resolve_next_capture_paths(
        cfg.output_root, cfg, "Gemini 435Le", "SN001", "normal", ts, start_index=1
    )
    assert paths.video_path.parent == cfg.output_root / cfg.batch_id
    assert paths.video_path.suffix == ".mp4"

    fixed_cfg = _make_config(tmp_path / "raw/tray_detect", fixed_test=True, batch_id=None, planned_split="fixed_test")
    fixed_paths, _ = capture.resolve_next_capture_paths(
        fixed_cfg.output_root,
        fixed_cfg,
        "Gemini 435Le",
        "SN001",
        "normal",
        ts,
        start_index=1,
    )
    assert fixed_paths.video_path.parent == fixed_cfg.output_root / "fixed_test_videos"


def test_unique_indexing_avoids_overwrite(tmp_path: Path):
    cfg = _make_config(tmp_path / "raw/tray_detect")
    ts = capture._capture_timestamp()
    p1, _ = capture.resolve_next_capture_paths(
        cfg.output_root, cfg, "Gemini 435Le", "SN001", "normal", ts, start_index=1
    )
    p1.video_path.touch()
    p2, _ = capture.resolve_next_capture_paths(
        cfg.output_root, cfg, "Gemini 435Le", "SN001", "normal", ts, start_index=1
    )
    assert p2.capture_id != p1.capture_id
    assert not p2.video_path.exists()


def test_append_manifest_creates_then_appends(tmp_path: Path):
    manifest = tmp_path / "capture_manifest.csv"
    row1 = {"capture_id": "a", "batch_id": "b", "video_path": "v", "metadata_path": "m", "frames_csv_path": "f",
            "capture_date": "2026-08-19", "start_time": "s", "end_time": "e", "duration_seconds": "1",
            "width": "1", "height": "1", "requested_fps": "30", "measured_fps": "30", "codec": "mp4v",
            "frame_count": "10", "device_model": "m", "device_serial": "x", "lighting": "normal", "scene": "",
            "tray_count": "1", "camera_moved": "no", "conveyor_running": "yes", "planned_split": "unassigned",
            "fixed_test": "false", "status": "completed", "video_sha256": "aa", "notes": ""}
    row2 = row1.copy()
    row2["capture_id"] = "b"
    capture.append_capture_manifest(manifest, row1)
    capture.append_capture_manifest(manifest, row2)
    rows = _read_csv_rows(manifest)
    assert len(rows) == 2
    assert rows[0]["capture_id"] == "a"
    assert rows[1]["capture_id"] == "b"


def test_no_frame_timeout_conversion_fail_stats(tmp_path: Path):
    width, height = 16, 12
    cfg = _make_config(tmp_path / "raw/tray_detect", duration_seconds=2.0)
    profile = _base_profile(width=width, height=height)

    # 先超时一次再成功一帧
    frames = [None, capture.SourceFrame(_make_frame(0, width, height), width, height, "BGR", frame_index=0)]
    source = FakeSource(frames, profile)

    actions = {"n": 0}

    def stop_after_write(*_):
        actions["n"] += 1
        return "stop" if actions["n"] >= 1 else None

    # 覆盖预览分支以避免窗口
    with pytest.MonkeyPatch.context() as m:
        m.setattr(capture, "_preview_status_line", stop_after_write)
        results = capture.run_capture(cfg, source)
    assert len(results) == 1
    assert results[0].statistics.timeouts >= 1
    assert results[0].statistics.frames_written == 1
    assert source.stopped


def test_conversion_failures_reported(tmp_path: Path):
    width, height = 16, 12
    cfg = _make_config(tmp_path / "raw/tray_detect", duration_seconds=1.0, segment_seconds=0.0)
    profile = _base_profile(width=width, height=height)
    bad = capture.SourceFrame(np.zeros((height, width, 3), dtype=np.uint8), width, height, "UNSUPPORTED", frame_index=1)
    good = capture.SourceFrame(_make_frame(2, width, height), width, height, "BGR", frame_index=2)
    source = FakeSource([bad, good], profile)

    actions = {"n": 0}

    def stop_after_second(frame_bgr, *_args, **kwargs):
        del frame_bgr, _args, kwargs
        actions["n"] += 1
        return "stop" if actions["n"] >= 2 else None

    with pytest.MonkeyPatch.context() as m:
        m.setattr(capture, "_preview_status_line", stop_after_second)
        results = capture.run_capture(cfg, source)
    assert len(results) == 1
    assert results[0].statistics.conversion_failures == 1
    assert results[0].statistics.frames_written == 1


def test_auto_segment_generates_independent_metadata(tmp_path: Path):
    width, height = 16, 12
    cfg = _make_config(tmp_path / "raw/tray_detect", segment_seconds=1.0, duration_seconds=0.0)
    profile = _base_profile(width=width, height=height)
    source = FakeSource(
        [
            capture.SourceFrame(_make_frame(1, width, height), width, height, "BGR", frame_index=1),
            capture.SourceFrame(_make_frame(2, width, height), width, height, "BGR", frame_index=2),
            capture.SourceFrame(_make_frame(3, width, height), width, height, "BGR", frame_index=3),
        ],
        profile,
    )
    calls = {"n": 0}

    def segment_plan(*_):
        calls["n"] += 1
        if calls["n"] == 1:
            return "next"
        return "stop"

    with pytest.MonkeyPatch.context() as m:
        m.setattr(capture, "_preview_status_line", segment_plan)
        results = capture.run_capture(cfg, source)
    assert len(results) == 2
    assert results[0].status == "completed"
    assert results[1].status == "completed"
    for result in results:
        assert result.video_path is not None and result.video_path.exists()
        assert result.metadata_path.exists()
        assert result.frames_csv_path.exists()
    manifest_rows = _read_csv_rows(cfg.output_root / "capture_manifest.csv")
    assert len(manifest_rows) >= 2
    ids = {r["capture_id"] for r in manifest_rows[:2]}
    assert len(ids) == 2


def test_video_writer_open_failure_is_explicit_error(tmp_path: Path):
    cfg = _make_config(tmp_path / "raw/tray_detect", duration_seconds=0.2, segment_seconds=0.0)
    profile = _base_profile()
    source = FakeSource(
        [capture.SourceFrame(_make_frame(1), 64, 48, "BGR", frame_index=1)],
        profile,
    )

    def bad_writer(*_):
        return FakeWriter(opened=False)

    with pytest.raises(capture.VideoWriterError):
        capture.run_capture(cfg, source, video_writer_factory=bad_writer)
    assert source.stopped


def test_keyboard_interrupt_releases_resources(tmp_path: Path):
    cfg = _make_config(tmp_path / "raw/tray_detect", duration_seconds=2.0)
    profile = _base_profile()
    source = FakeSource(
        [KeyboardInterrupt()],
        profile,
    )
    writer = FakeWriter(opened=True)
    with pytest.raises(KeyboardInterrupt):
        capture.run_capture(cfg, source, video_writer_factory=lambda *_args: writer)
    assert source.stopped
    assert writer.released


def test_synthetic_video_smoke_record(tmp_path: Path, monkeypatch):
    width, height = 64, 48
    cfg = _make_config(
        tmp_path / "raw/tray_detect",
        duration_seconds=10.0,
        segment_seconds=0.0,
        warmup_seconds=0.0,
    )
    profile = _base_profile(width=width, height=height, fps=30.0)
    frames = [
        capture.SourceFrame(_make_frame(i, width, height), width, height, "BGR", frame_index=i, device_timestamp_ms=1000 + i)
        for i in range(45)
    ]
    source = FakeSource(frames, profile)
    calls = {"n": 0}

    def stop_at_end(*_):
        calls["n"] += 1
        return "stop" if calls["n"] >= 45 else None

    with monkeypatch.context() as m:
        m.setattr(capture, "_preview_status_line", stop_at_end)
        results = capture.run_capture(cfg, source)

    assert len(results) == 1
    assert results[0].video_path is not None
    assert results[0].video_path.exists()
    assert not str(results[0].video_path).endswith(".partial")

    cap = cv2.VideoCapture(str(results[0].video_path))
    assert cap.isOpened()
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame is not None and frame.shape[1] == width and frame.shape[0] == height

    assert results[0].metadata_path.exists()
    csv_path = results[0].frames_csv_path
    assert csv_path.exists()
    assert cfg.output_root.joinpath("capture_manifest.csv").exists()
    payload = json.loads(results[0].metadata_path.read_text(encoding="utf-8"))
    assert "NaN" not in json.dumps(payload)
    assert "Infinity" not in json.dumps(payload)
    assert payload["status"] == "completed"
    assert payload["scene"]["scene_tags"] == ["test_scene"]
    assert payload["integrity"]["video_sha256"]
    assert payload["video_path"].endswith(".mp4")
    assert capture.compute_sha256(results[0].video_path) == payload["integrity"]["video_sha256"]


def test_manifest_required_fields_after_synthetic_capture(tmp_path: Path, monkeypatch):
    width, height = 32, 24
    cfg = _make_config(
        tmp_path / "raw/tray_detect",
        duration_seconds=5.0,
        segment_seconds=0.0,
        warmup_seconds=0.0,
    )
    profile = _base_profile(width=width, height=height, fps=20.0)
    frames = [
        capture.SourceFrame(_make_frame(i, width, height), width, height, "BGR", frame_index=i, device_timestamp_ms=1000 + i)
        for i in range(20)
    ]
    source = FakeSource(frames, profile)

    calls = {"n": 0}

    def stop_after_ten(*_):
        calls["n"] += 1
        return "stop" if calls["n"] >= 20 else None

    with monkeypatch.context() as m:
        m.setattr(capture, "_preview_status_line", stop_after_ten)
        capture.run_capture(cfg, source)

    manifest_rows = _read_csv_rows(cfg.output_root / "capture_manifest.csv")
    assert len(manifest_rows) >= 1
    first = manifest_rows[0]
    required_fields = [
        "batch_id",
        "video_path",
        "capture_date",
        "width",
        "height",
        "fps",
        "lighting",
        "scene",
        "tray_count",
        "planned_split",
        "notes",
    ]
    for field in required_fields:
        assert field in first


def test_active_unsupported_format_is_rejected_before_recording(tmp_path: Path):
    cfg = _make_config(tmp_path / "raw/tray_detect", no_preview=True)
    source = FakeSource([], _base_profile(fmt="NV12"))
    with pytest.raises(capture.FrameSourceError) as exc_info:
        capture.run_capture(cfg, source)
    assert exc_info.value.exit_code == capture.EXIT_UNSUPPORTED_PROFILE
    assert source.stopped
    assert not (cfg.output_root / "capture_manifest.csv").exists()


def test_write_failure_keeps_recoverable_partial_with_metadata(tmp_path: Path, monkeypatch):
    width, height = 64, 48
    cfg = _make_config(tmp_path / "raw/tray_detect", duration_seconds=5.0, warmup_seconds=0.0)
    source = FakeSource(
        [
            capture.SourceFrame(_make_frame(i, width, height), width, height, "BGR", frame_index=i)
            for i in range(3)
        ],
        _base_profile(width=width, height=height),
    )

    class FailSecondWrite:
        def __init__(self, path: Path, *args) -> None:
            self.writer = capture.create_video_writer(path, *args)
            self.writes = 0

        def isOpened(self) -> bool:
            return self.writer.isOpened()

        def write(self, frame) -> None:
            self.writes += 1
            if self.writes == 2:
                raise RuntimeError("simulated disk failure")
            self.writer.write(frame)

        def release(self) -> None:
            self.writer.release()

    with pytest.raises(capture.VideoWriterError):
        capture.run_capture(cfg, source, video_writer_factory=FailSecondWrite)

    metadata_files = list((cfg.output_root / cfg.batch_id).glob("*.json"))
    assert len(metadata_files) == 1
    payload = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "write_error"
    assert payload["integrity"]["video_readable_by_opencv"] is True
    assert ".partial.mp4" in payload["video_path"]
    assert list((cfg.output_root / cfg.batch_id).glob("*.partial.mp4"))
    assert _read_csv_rows(cfg.output_root / "capture_manifest.csv")[0]["status"] == "write_error"
