"""果托优先、双 ROI、三模式统一运行循环。"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml
from .config import WatermelonScheduler
from .frame_source import FrameSource
from .model_utils import sha256_file
from .result_writer import JsonlResultWriter, create_run_directory, write_json_atomic
from .roi import normalized_to_pixel_roi
from .training import environment_versions
from .types import FrameResult
from .visualization import draw_frame


class DualRuntime:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        watermelon_segmenter: Any = None,
        tray_tracker: Any = None,
    ):
        self.config = config
        self.mode = config["runtime"]["mode"]
        self.watermelon = watermelon_segmenter
        self.tray = tray_tracker
        self.scheduler = WatermelonScheduler(
            int(config["runtime"]["watermelon_every_n_frames"])
        )
        self.latest_watermelons = []
        self.latest_wm_frame = -1

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
        capture_fps: float,
        elapsed_seconds: float,
    ) -> FrameResult:
        h, w = frame.shape[:2]
        wm_roi = normalized_to_pixel_roi(self.config["rois"]["watermelon"], w, h)
        tray_roi = normalized_to_pixel_roi(self.config["rois"]["tray"], w, h)
        tracks = []
        unconfirmed = []
        if self.mode in {"tray", "dual"}:
            if self.tray is None:
                raise RuntimeError("tray tracker is not loaded")
            tracks, unconfirmed = self.tray.update(
                frame, tray_roi, frame_index, timestamp_ms
            )
        if self.mode in {"watermelon", "dual"} and self.scheduler.should_run(
            frame_index
        ):
            if self.watermelon is None:
                raise RuntimeError("watermelon segmenter is not loaded")
            self.latest_watermelons = self.watermelon.predict(
                frame, wm_roi, frame_index, timestamp_ms
            )
            self.latest_wm_frame = frame_index
        age = 0 if self.latest_wm_frame < 0 else frame_index - self.latest_wm_frame
        return FrameResult(
            frame_index,
            float(timestamp_ms),
            list(self.latest_watermelons),
            tracks,
            unconfirmed,
            age,
            float(capture_fps),
            1 / max(elapsed_seconds, 1e-9),
        )


def run_video(
    config: dict[str, Any],
    source_value: Any,
    watermelon: Any,
    tray: Any,
    max_frames: int | None = None,
) -> Path:
    source_cfg = config["source"]
    source = FrameSource(
        source_value,
        source_cfg.get("requested_width", 0),
        source_cfg.get("requested_height", 0),
        source_cfg.get("requested_fps", 0),
    )
    runtime = DualRuntime(config, watermelon_segmenter=watermelon, tray_tracker=tray)
    run_dir = create_run_directory(
        Path(config["output"]["root"]), config["runtime"]["mode"]
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json_atomic(run_dir / "runtime_versions.json", environment_versions())
    writer = (
        JsonlResultWriter(run_dir / "events.jsonl")
        if config["output"].get("save_jsonl", True)
        else None
    )
    video = None
    times = []
    track_ids = set()
    max_tracks = wm_calls = tray_calls = frames = warnings = 0
    started = time.perf_counter()
    try:
        while max_frames is None or frames < max_frames:
            ok, frame, stamp = source.read()
            if not ok:
                break
            tick = time.perf_counter()
            before = runtime.latest_wm_frame
            result = runtime.process_frame(
                frame,
                frames,
                stamp,
                source.info.fps,
                max(time.perf_counter() - tick, 1e-9),
            )
            elapsed = time.perf_counter() - tick
            times.append(elapsed)
            wm_calls += int(runtime.latest_wm_frame != before)
            tray_calls += int(runtime.mode in {"tray", "dual"})
            track_ids.update(t.track_id for t in result.tray_tracks)
            max_tracks = max(max_tracks, len(result.tray_tracks))
            result = FrameResult(
                **{**result.__dict__, "processing_fps": 1 / max(elapsed, 1e-9)}
            )
            annotated = draw_frame(
                frame,
                result,
                normalized_to_pixel_roi(
                    config["rois"]["watermelon"], source.info.width, source.info.height
                )
                if runtime.mode in {"watermelon", "dual"}
                else None,
                normalized_to_pixel_roi(
                    config["rois"]["tray"], source.info.width, source.info.height
                )
                if runtime.mode in {"tray", "dual"}
                else None,
                getattr(tray, "history", None),
            )
            if frames == 0 and not cv2.imwrite(
                str(run_dir / "annotated_preview.jpg"), annotated
            ):
                raise RuntimeError("failed to write annotated preview")
            if writer:
                writer.write(result)
            if frames == 0 and runtime.mode == "watermelon" and result.watermelons:
                mask_dir = run_dir / "watermelon_masks"
                mask_dir.mkdir()
                for item in result.watermelons:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    polygon = np.asarray(item.polygon_full, dtype=np.int32)
                    if len(polygon) >= 3:
                        cv2.fillPoly(mask, [polygon], 255)
                    if not cv2.imwrite(
                        str(
                            mask_dir
                            / f"frame_{frames:08d}_instance_{item.instance_id:03d}.png"
                        ),
                        mask,
                    ):
                        raise RuntimeError("failed to write watermelon mask")
            if config["output"].get("save_annotated_video", True):
                if video is None:
                    video = cv2.VideoWriter(
                        str(run_dir / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source.info.fps or 30,
                        (source.info.width, source.info.height),
                    )
                    if not video.isOpened():
                        raise RuntimeError("video encoder creation failed")
                video.write(annotated)
            if config["visualization"].get("show_window", False):
                cv2.imshow("YOLO26 dual", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    runtime.scheduler.request_refresh()
                if key == ord("p"):
                    while True:
                        k = cv2.waitKey(50) & 0xFF
                        if k in (ord("p"), ord("q")):
                            break
                    if k == ord("q"):
                        break
            frames += 1
    finally:
        if writer:
            writer.close()
        if video:
            video.release()
        source.close()
        cv2.destroyAllWindows()
        if watermelon:
            watermelon.close()
        if tray:
            tray.close()
    total = time.perf_counter() - started
    ordered = sorted(times)
    p95 = (
        ordered[min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))]
        if ordered
        else 0
    )

    def weight_hash(name):
        path = Path(config["models"][name]["weights"])
        return sha256_file(path) if path.is_file() else None

    summary = {
        "source": str(source_value),
        "mode": runtime.mode,
        "processed_frames": frames,
        "skipped_frames": source.skipped_frames,
        "runtime_seconds": total,
        "average_fps": frames / total if total else 0,
        "p95_frame_seconds": p95,
        "watermelon_inference_count": wm_calls,
        "tray_inference_count": tray_calls,
        "maximum_simultaneous_trays": max_tracks,
        "unique_track_id_count": len(track_ids),
        "watermelon_weights_sha256": weight_hash("watermelon"),
        "tray_weights_sha256": weight_hash("tray"),
        "environment": environment_versions(),
        "exception_count": 0,
        "warning_count": warnings,
    }
    write_json_atomic(run_dir / "runtime_summary.json", summary)
    return run_dir
