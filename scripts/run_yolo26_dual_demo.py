#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yolo26_dual.config import load_runtime_config
from yolo26_dual.diagnostics import diagnose
from yolo26_dual.runtime import run_video
from yolo26_dual.tray_tracker import TrayTracker
from yolo26_dual.watermelon_segmenter import WatermelonSegmenter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/yolo26_dual_runtime.yaml"
    )
    p.add_argument("--source", default=None)
    p.add_argument("--mode", choices=["watermelon", "tray", "dual"], default=None)
    p.add_argument("--weights", type=Path)
    p.add_argument("--roi", type=float, nargs=4)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--conf", type=float)
    p.add_argument("--device")
    p.add_argument("--tracker", type=Path)
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--warmup-frames", type=int, default=20)
    p.add_argument("--benchmark-frames", type=int, default=300)
    p.add_argument("--watermelon-every-n-frames", type=int)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-window", action="store_true")
    a = p.parse_args()
    c = load_runtime_config(a.config, mode=a.mode)
    if a.watermelon_every_n_frames is not None:
        if a.watermelon_every_n_frames < 0:
            p.error("--watermelon-every-n-frames must be >= 0")
        c["runtime"]["watermelon_every_n_frames"] = a.watermelon_every_n_frames
    source = a.source if a.source is not None else c["source"]["value"]
    selected = c["runtime"]["mode"]
    if a.weights:
        if selected == "dual":
            p.error("--weights is only valid in a single-model mode")
        c["models"][selected]["weights"] = str(a.weights.resolve())
    if a.roi:
        if selected == "dual":
            p.error("--roi is only valid in a single-model mode")
        from yolo26_dual.roi import validate_normalized_roi

        c["rois"][selected] = list(validate_normalized_roi(a.roi))
    for key in ("imgsz", "conf", "device"):
        value = getattr(a, key)
        if value is not None:
            if selected == "dual":
                p.error(f"--{key} is only valid in a single-model mode")
            c["models"][selected][key] = value
    if a.tracker:
        c["tracker"]["config"] = str(a.tracker.resolve())
    if a.no_window:
        c["visualization"]["show_window"] = False
    if a.diagnose:
        print(json.dumps(diagnose(ROOT, c, source), ensure_ascii=False, indent=2))
        return 0
    wm = None
    tray = None
    if c["runtime"]["mode"] in {"watermelon", "dual"}:
        m = c["models"]["watermelon"]
        wm = WatermelonSegmenter(
            m["weights"], m["imgsz"], m["conf"], m["iou"], m["device"], m["half"]
        )
    if c["runtime"]["mode"] in {"tray", "dual"}:
        m = c["models"]["tray"]
        tray = TrayTracker(
            m["weights"],
            c["tracker"]["config"],
            m["imgsz"],
            m["conf"],
            m["iou"],
            m["device"],
            m["half"],
            c["tracker"]["history_length"],
        )
    try:
        if a.benchmark:
            from yolo26_dual.benchmark import benchmark_runtime

            print(
                f"output={benchmark_runtime(c, source, wm, tray, a.warmup_frames, a.benchmark_frames)}"
            )
            return 0
        print(f"output={run_video(c, source, wm, tray, a.max_frames)}")
        return 0
    except KeyboardInterrupt:
        print("interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        try:
            import torch

            is_oom = isinstance(
                exc, torch.cuda.OutOfMemoryError
            ) or "CUDA out of memory" in str(exc)
            if is_oom:
                torch.cuda.empty_cache()
                print(
                    "CUDA OOM: no automatic CPU fallback. Increase watermelon interval, keep tray every frame, or reduce watermelon imgsz to 640.",
                    file=sys.stderr,
                )
                return 2
        except ImportError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
