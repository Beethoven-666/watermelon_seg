"""不保存媒体的实际运行性能基准。"""

from __future__ import annotations
import statistics
import time
from pathlib import Path
from typing import Any
import psutil
from .frame_source import FrameSource
from .result_writer import create_run_directory, write_json_atomic
from .runtime import DualRuntime


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return (
        ordered[min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))]
        if ordered
        else 0.0
    )


def benchmark_runtime(
    config: dict[str, Any],
    source_value: Any,
    watermelon: Any,
    tray: Any,
    warmup_frames: int = 20,
    measure_frames: int = 300,
) -> Path:
    if warmup_frames < 0 or measure_frames <= 0:
        raise ValueError("invalid benchmark frame counts")
    cfg = config["source"]
    source = FrameSource(
        source_value,
        cfg.get("requested_width", 0),
        cfg.get("requested_height", 0),
        cfg.get("requested_fps", 0),
    )
    runtime = DualRuntime(config, watermelon_segmenter=watermelon, tray_tracker=tray)
    times = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        frame_index = 0
        while frame_index < warmup_frames + measure_frames:
            ok, frame, stamp = source.read()
            if not ok:
                break
            tick = time.perf_counter()
            runtime.process_frame(frame, frame_index, stamp, source.info.fps, 1.0)
            elapsed = time.perf_counter() - tick
            if frame_index >= warmup_frames:
                times.append(elapsed)
            peak_rss = max(peak_rss, process.memory_info().rss)
            frame_index += 1
        cuda_peak = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
    finally:
        source.close()
        if watermelon:
            watermelon.close()
        if tray:
            tray.close()
    output = create_run_directory(
        Path(config["output"]["root"]), f"benchmark_{runtime.mode}"
    )
    mean = statistics.fmean(times) if times else 0.0
    payload = {
        "mode": runtime.mode,
        "source": str(source_value),
        "warmup_frames_requested": warmup_frames,
        "measured_frames_requested": measure_frames,
        "measured_frames": len(times),
        "mean_frame_ms": mean * 1000,
        "p50_frame_ms": _percentile(times, 0.50) * 1000,
        "p95_frame_ms": _percentile(times, 0.95) * 1000,
        "p99_frame_ms": _percentile(times, 0.99) * 1000,
        "average_fps": 1 / mean if mean else 0.0,
        "peak_cuda_memory_bytes": cuda_peak,
        "peak_process_rss_bytes": peak_rss,
        "source_skipped_frames": source.skipped_frames,
    }
    write_json_atomic(output / "benchmark.json", payload)
    return output
