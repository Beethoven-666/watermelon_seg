#!/usr/bin/env python
"""Use SAM 3 text prompts to export zero-shot fruit masks and sorting candidates."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "deploy" / "sam3" / "fruit_prompts.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "runs27" / "sam3" / "zero_shot_fruit_predict"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CANDIDATE_COLUMNS = [
    "image",
    "image_relpath",
    "image_id",
    "image_width",
    "image_height",
    "mask_width",
    "mask_height",
    "candidate_index",
    "model",
    "class_id",
    "class_name",
    "prompt",
    "confidence",
    "area_pixels",
    "area_ratio",
    "centroid_x",
    "centroid_y",
    "centroid_x_px",
    "centroid_y_px",
    "grasp_x",
    "grasp_y",
    "grasp_x_px",
    "grasp_y_px",
    "grasp_clearance_px",
    "bbox_min_x",
    "bbox_min_y",
    "bbox_max_x",
    "bbox_max_y",
    "bbox_min_x_px",
    "bbox_min_y_px",
    "bbox_max_x_px",
    "bbox_max_y_px",
    "polygon_points",
    "mask_components",
    "mask_holes",
    "mask_path",
]

MANIFEST_COLUMNS = [
    "image",
    "image_relpath",
    "image_id",
    "status",
    "width",
    "height",
    "inference_width",
    "inference_height",
    "prompt_count",
    "after_confidence",
    "truncated_by_limit",
    "filtered_by_area",
    "before_global_mask_nms",
    "kept",
    "elapsed_seconds",
    "error",
]


@dataclass(frozen=True)
class PromptSpec:
    class_id: int
    class_name: str
    text: str
    priority: int
    confidence_threshold: float | None = None


@dataclass
class Candidate:
    prompt: PromptSpec
    confidence: float
    mask: Any
    area_pixels: int
    area_ratio: float
    bbox_px: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    grasp_px: tuple[int, int]
    grasp_clearance_px: float
    polygon_px: Any
    component_count: int
    hole_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Image file or directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="Local sam3.pt; avoids a Hugging Face download.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan source directories.")
    parser.add_argument("--include-glob", default="*", help="Include image filenames matching this glob.")
    parser.add_argument("--exclude-glob", default="", help="Exclude image filenames matching this glob.")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="CLASS_NAME=TEXT",
        help="Replace configured prompts; repeat for multiple fruit concepts.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Override runtime.device.")
    parser.add_argument("--resolution", type=int, help="Override runtime.resolution.")
    parser.add_argument(
        "--max-image-side",
        type=int,
        help="Downscale large source images before SAM mask upsampling; 0 disables the limit.",
    )
    parser.add_argument("--confidence-threshold", type=float, help="Override SAM 3 confidence threshold.")
    parser.add_argument("--min-area-ratio", type=float, help="Reject masks smaller than this image fraction.")
    parser.add_argument(
        "--mask-nms-iou",
        "--cross-prompt-nms-iou",
        dest="mask_nms_iou",
        type=float,
        help="Global class-agnostic mask IoU dedup threshold (old cross-prompt name is an alias).",
    )
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "none"), help="CUDA autocast dtype.")
    parser.add_argument("--max-detections-per-prompt", type=int)
    parser.add_argument("--max-candidates-per-image", type=int)
    parser.add_argument(
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save lossless binary PNG masks.",
    )
    parser.add_argument(
        "--save-overlays",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save visual QA overlays.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Record bad images and continue.")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into a non-empty output directory; stale files are not removed.",
    )
    parser.add_argument("--diagnose", action="store_true", help="Print a no-secret environment readiness report.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and enumerate images without a model.")
    parser.add_argument("--debug", action="store_true", help="Show a full traceback on failure.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - actionable runtime guard
        raise RuntimeError("PyYAML is missing; install deploy/sam3/requirements-windows.txt") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt config must contain a YAML mapping: {path}")
    return data


def _inline_prompts(values: list[str]) -> list[PromptSpec]:
    prompts: list[PromptSpec] = []
    for class_id, value in enumerate(values):
        if "=" not in value:
            raise ValueError(f"--prompt must use CLASS_NAME=TEXT: {value!r}")
        class_name, text = (part.strip() for part in value.split("=", 1))
        if not class_name or not text:
            raise ValueError(f"--prompt must use non-empty CLASS_NAME=TEXT: {value!r}")
        prompts.append(PromptSpec(class_id, class_name, text, 100, None))
    return prompts


def _configured_prompts(rows: Any) -> list[PromptSpec]:
    if not isinstance(rows, list):
        raise ValueError("prompts must be a YAML list")
    prompts: list[PromptSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"prompts[{index}] must be a mapping")
        if not bool(row.get("enabled", True)):
            continue
        try:
            threshold_raw = row.get("confidence_threshold")
            threshold = None if threshold_raw is None else _number(
                threshold_raw, f"prompts[{index}].confidence_threshold", 0.0, 1.0
            )
            spec = PromptSpec(
                class_id=int(row["class_id"]),
                class_name=str(row["class_name"]).strip(),
                text=str(row["text"]).strip(),
                priority=int(row.get("priority", 100)),
                confidence_threshold=threshold,
            )
        except KeyError as exc:
            raise ValueError(f"prompts[{index}] is missing {exc.args[0]!r}") from exc
        if not spec.class_name or not spec.text:
            raise ValueError(f"prompts[{index}] has an empty class_name or text")
        prompts.append(spec)
    return prompts


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {result}")
    return result


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    raw = _load_yaml(config_path)
    model = dict(raw.get("model") or {})
    runtime = dict(raw.get("runtime") or {})
    output = dict(raw.get("output") or {})

    prompts = _inline_prompts(args.prompt) if args.prompt else _configured_prompts(raw.get("prompts"))
    if not prompts:
        raise ValueError("At least one enabled prompt is required")
    ids = [prompt.class_id for prompt in prompts]
    names = [prompt.class_name for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("Enabled prompt class_id values must be unique")
    if len(names) != len(set(names)):
        raise ValueError("Enabled prompt class_name values must be unique")

    repo_id = str(model.get("repo_id", "facebook/sam3"))
    if repo_id != "facebook/sam3":
        raise ValueError("This native deployment entry currently supports model.repo_id=facebook/sam3 only")

    checkpoint_raw: Any = args.checkpoint if args.checkpoint is not None else model.get("checkpoint")
    checkpoint: Path | None = None
    if checkpoint_raw:
        checkpoint = Path(str(checkpoint_raw)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = config_path.parent / checkpoint
        checkpoint = checkpoint.resolve()

    device = args.device or str(runtime.get("device", "cuda"))
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda' or 'cpu'")
    resolution = int(args.resolution if args.resolution is not None else runtime.get("resolution", 1008))
    if resolution != 1008:
        raise ValueError(
            "facebook/sam3 image inference requires resolution=1008 with the current official "
            "checkpoint; use max_image_side to control output-mask memory"
        )
    max_image_side = int(
        args.max_image_side if args.max_image_side is not None else runtime.get("max_image_side", 1280)
    )
    if max_image_side != 0 and max_image_side < 224:
        raise ValueError("max_image_side must be 0 or at least 224")
    amp_dtype = args.amp_dtype or str(runtime.get("amp_dtype", "bfloat16"))
    if amp_dtype not in {"bfloat16", "float16", "none"}:
        raise ValueError("amp_dtype must be bfloat16, float16, or none")

    max_per_prompt = int(
        args.max_detections_per_prompt
        if args.max_detections_per_prompt is not None
        else runtime.get("max_detections_per_prompt", 50)
    )
    max_per_image = int(
        args.max_candidates_per_image
        if args.max_candidates_per_image is not None
        else runtime.get("max_candidates_per_image", 100)
    )
    if max_per_prompt < 1 or max_per_image < 1:
        raise ValueError("max detection limits must be positive")

    return {
        "config_path": config_path,
        "repo_id": repo_id,
        "checkpoint": checkpoint,
        "device": device,
        "resolution": resolution,
        "max_image_side": max_image_side,
        "amp_dtype": amp_dtype,
        "confidence_threshold": _number(
            args.confidence_threshold
            if args.confidence_threshold is not None
            else runtime.get("confidence_threshold", 0.5),
            "confidence_threshold",
            0.0,
            1.0,
        ),
        "min_area_ratio": _number(
            args.min_area_ratio if args.min_area_ratio is not None else runtime.get("min_area_ratio", 0.0),
            "min_area_ratio",
            0.0,
            1.0,
        ),
        "mask_nms_iou": _number(
            args.mask_nms_iou
            if args.mask_nms_iou is not None
            else runtime.get("mask_nms_iou", runtime.get("cross_prompt_nms_iou", 0.7)),
            "mask_nms_iou",
            0.0,
            1.0,
        ),
        "max_detections_per_prompt": max_per_prompt,
        "max_candidates_per_image": max_per_image,
        "save_masks": bool(output.get("save_masks", True)) if args.save_masks is None else args.save_masks,
        "save_overlays": bool(output.get("save_overlays", True)) if args.save_overlays is None else args.save_overlays,
        "prompts": prompts,
    }


def image_paths(source: Path, recursive: bool, include_glob: str, exclude_glob: str) -> list[Path]:
    source = source.expanduser().resolve()

    def selected(path: Path) -> bool:
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return False
        if include_glob and not fnmatch.fnmatch(path.name, include_glob):
            return False
        if exclude_glob and fnmatch.fnmatch(path.name, exclude_glob):
            return False
        return True

    if source.is_file():
        if not selected(source):
            raise ValueError(f"Unsupported or filtered image: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")
    iterator: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    return sorted(path.resolve() for path in iterator if path.is_file() and selected(path))


def relative_image_path(path: Path, source: Path) -> Path:
    source = source.expanduser().resolve()
    return path.relative_to(source) if source.is_dir() else Path(path.name)


def stable_image_id(relative_path: Path) -> str:
    rel = relative_path.as_posix()
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", relative_path.stem).strip("_") or "image"
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:60]}_{digest}"


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def cached_checkpoint(repo_id: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(repo_id, "sam3.pt")
        return str(result) if isinstance(result, str) and Path(result).is_file() else None
    except Exception:
        return None


def cached_config(repo_id: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(repo_id, "config.json")
        return str(result) if isinstance(result, str) and Path(result).is_file() else None
    except Exception:
        return None


def probe_hf_config(repo_id: str) -> tuple[str | None, str | None]:
    if not hf_token_present():
        return cached_config(repo_id), None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename="config.json"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:1000]


def hf_token_present() -> bool:
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def diagnostics(settings: dict[str, Any]) -> dict[str, Any]:
    config_cache, hf_probe_error = probe_hf_config(settings["repo_id"])
    distributions = [
        "sam3",
        "torch",
        "torchvision",
        "numpy",
        "setuptools",
        "timm",
        "huggingface_hub",
        "einops",
        "triton-windows",
        "pycocotools",
        "psutil",
        "opencv-python-headless",
        "PyYAML",
        "Pillow",
    ]
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "supported": sys.version_info >= (3, 12),
        },
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in distributions},
        "config": {
            "path": str(settings["config_path"]),
            "device": settings["device"],
            "resolution": settings["resolution"],
            "max_image_side": settings["max_image_side"],
            "amp_dtype": settings["amp_dtype"],
            "enabled_prompts": [prompt.text for prompt in settings["prompts"]],
        },
        "weights": {
            "repo_id": settings["repo_id"],
            "local_checkpoint": str(settings["checkpoint"]) if settings["checkpoint"] else None,
            "local_checkpoint_exists": bool(settings["checkpoint"] and settings["checkpoint"].is_file()),
            "cached_checkpoint": cached_checkpoint(settings["repo_id"]),
            "cached_config": config_cache,
            "hf_credential_present": hf_token_present(),
            "hf_gated_access_probed": bool(config_cache),
            "hf_probe_error": hf_probe_error,
        },
        "cuda": {},
        "sam3_import": {},
        "blocking_issues": [],
        "notes": [],
    }

    if not report["python"]["supported"]:
        report["blocking_issues"].append("SAM 3 official deployment requires Python >= 3.12")

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        report["cuda"] = {
            "available": cuda_available,
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
            "bfloat16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else None,
        }
        if cuda_available:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            report["cuda"]["memory_free_mib"] = round(free_bytes / 1024**2, 1)
            report["cuda"]["memory_total_mib"] = round(total_bytes / 1024**2, 1)
        if settings["device"] == "cuda" and not cuda_available:
            report["blocking_issues"].append("Configured device is cuda, but torch.cuda.is_available() is false")
        if settings["amp_dtype"] == "bfloat16" and cuda_available and not torch.cuda.is_bf16_supported():
            report["blocking_issues"].append("Configured bfloat16 autocast is not supported by this GPU")
    except Exception as exc:
        report["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        report["blocking_issues"].append("PyTorch/CUDA import failed")

    try:
        import sam3

        report["sam3_import"] = {
            "ok": True,
            "version": getattr(sam3, "__version__", None),
            "path": str(Path(sam3.__file__).resolve()),
        }
    except Exception as exc:
        report["sam3_import"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report["blocking_issues"].append("SAM 3 import failed")

    weight_ready = bool(
        report["weights"]["local_checkpoint_exists"]
        or (report["weights"]["cached_checkpoint"] and report["weights"]["cached_config"])
        or (report["weights"]["hf_credential_present"] and report["weights"]["hf_gated_access_probed"])
    )
    if not weight_ready:
        if report["weights"]["hf_credential_present"]:
            report["blocking_issues"].append(
                "Hugging Face login exists, but facebook/sam3 gated access is not approved or the access probe failed"
            )
        else:
            report["blocking_issues"].append(
                "No local/cached sam3.pt and no Hugging Face credential; request gated access and run `hf auth login`"
            )

    if platform.system() == "Windows":
        report["notes"].append(
            "Native Windows uses community triton-windows; validate a real SAM 3 forward pass before production"
        )
    report["notes"].append(
        "The current official image checkpoint uses fixed resolution=1008; control output-mask memory with max_image_side"
    )
    report["ready_for_inference"] = not report["blocking_issues"]
    return report


def dry_run(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, Any]:
    if args.source is None:
        raise ValueError("--source is required with --dry-run")
    source = args.source.expanduser().resolve()
    images = image_paths(args.source, args.recursive, args.include_glob, args.exclude_glob)
    if not images:
        raise ValueError(f"No selected images found under: {source}")
    return {
        "dry_run": True,
        "source": str(source),
        "images": len(images),
        "sample_images": [relative_image_path(path, source).as_posix() for path in images[:20]],
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "device": settings["device"],
        "resolution": settings["resolution"],
        "max_image_side": settings["max_image_side"],
        "confidence_threshold": settings["confidence_threshold"],
        "min_area_ratio": settings["min_area_ratio"],
        "prompts": [
            {
                "class_id": prompt.class_id,
                "class_name": prompt.class_name,
                "text": prompt.text,
                "priority": prompt.priority,
                "confidence_threshold": prompt.confidence_threshold,
            }
            for prompt in settings["prompts"]
        ],
    }


def _mask_geometry(mask: Any) -> tuple[Any, ...]:
    import cv2
    import numpy as np

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        raise ValueError("Empty mask")
    bbox_px = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    centroid_px = (float(xs.mean()), float(ys.mean()))

    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    _, max_distance, _, max_location = cv2.minMaxLoc(distance)
    grasp_px = (int(max_location[0]), int(max_location[1]))

    contours, hierarchy = cv2.findContours(mask_u8.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        x0, y0, x1, y1 = bbox_px
        polygon = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        return bbox_px, centroid_px, grasp_px, float(max_distance), polygon, 1, 0

    parents = hierarchy[0, :, 3] if hierarchy is not None else np.full(len(contours), -1, dtype=np.int32)
    outer_indices = [index for index, parent in enumerate(parents.tolist()) if parent == -1]
    largest_index = max(outer_indices or list(range(len(contours))), key=lambda index: cv2.contourArea(contours[index]))
    largest = contours[largest_index]
    epsilon = max(0.5, 0.002 * cv2.arcLength(largest, True))
    polygon = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
    if polygon.shape[0] < 3:
        polygon = largest.reshape(-1, 2)
    component_count = len(outer_indices)
    hole_count = int(sum(1 for parent in parents.tolist() if parent != -1))
    return bbox_px, centroid_px, grasp_px, float(max_distance), polygon, component_count, hole_count


def _candidates_from_output(
    output: dict[str, Any],
    prompt: PromptSpec,
    min_area_ratio: float,
    max_detections: int,
) -> tuple[list[Candidate], int, int]:
    import cv2
    import numpy as np

    masks_tensor = output["masks"]
    scores_tensor = output["scores"]
    masks = masks_tensor.detach().to("cpu").numpy()
    scores = scores_tensor.detach().float().to("cpu").numpy().reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected SAM 3 mask shape: {masks.shape}")
    if len(masks) != len(scores):
        raise ValueError(f"SAM 3 returned {len(masks)} masks but {len(scores)} scores")

    count = len(scores)
    order = np.argsort(-scores[:count])[:max_detections]
    candidates: list[Candidate] = []
    filtered_by_area = 0
    for index in order.tolist():
        mask = np.asarray(masks[index], dtype=bool)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if component_count > 2:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = labels == largest_label
        area_pixels = int(mask.sum())
        if area_pixels <= 0:
            continue
        area_ratio = area_pixels / float(mask.shape[0] * mask.shape[1])
        if area_ratio < min_area_ratio:
            filtered_by_area += 1
            continue
        bbox_px, centroid_px, grasp_px, clearance, polygon, components, holes = _mask_geometry(mask)
        candidates.append(
            Candidate(
                prompt=prompt,
                confidence=float(scores[index]),
                mask=mask,
                area_pixels=area_pixels,
                area_ratio=area_ratio,
                bbox_px=bbox_px,
                centroid_px=centroid_px,
                grasp_px=grasp_px,
                grasp_clearance_px=clearance,
                polygon_px=polygon,
                component_count=components,
                hole_count=holes,
            )
        )
    return candidates, count, filtered_by_area


def _bbox_overlap(a: Candidate, b: Candidate) -> bool:
    ax0, ay0, ax1, ay1 = a.bbox_px
    bx0, by0, bx1, by1 = b.bbox_px
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def mask_iou(a: Candidate, b: Candidate) -> float:
    import numpy as np

    if not _bbox_overlap(a, b):
        return 0.0
    intersection = int(np.logical_and(a.mask, b.mask).sum())
    union = a.area_pixels + b.area_pixels - intersection
    return float(intersection / union) if union > 0 else 0.0


def global_mask_nms(candidates: list[Candidate], threshold: float, limit: int) -> list[Candidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.prompt.priority, item.confidence, item.area_pixels),
        reverse=True,
    )
    kept: list[Candidate] = []
    for candidate in ordered:
        if any(mask_iou(candidate, other) > threshold for other in kept):
            continue
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


def _autocast(torch_module: Any, device: str, amp_dtype: str) -> Any:
    if device != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch_module.bfloat16 if amp_dtype == "bfloat16" else torch_module.float16
    return torch_module.autocast(device_type="cuda", dtype=dtype)


def predict_image(
    image: Any,
    processor: Any,
    prompts: list[PromptSpec],
    settings: dict[str, Any],
    torch_module: Any,
) -> tuple[list[Candidate], dict[str, int]]:
    with _autocast(torch_module, settings["device"], settings["amp_dtype"]):
        state = processor.set_image(image)

    all_candidates: list[Candidate] = []
    after_confidence = 0
    truncated_by_limit = 0
    filtered_by_area = 0
    for prompt in prompts:
        prompt_threshold = (
            prompt.confidence_threshold
            if prompt.confidence_threshold is not None
            else settings["confidence_threshold"]
        )
        processor.set_confidence_threshold(prompt_threshold)
        with _autocast(torch_module, settings["device"], settings["amp_dtype"]):
            output = processor.set_text_prompt(state=state, prompt=prompt.text)
        candidates, count, area_filtered = _candidates_from_output(
            output,
            prompt,
            settings["min_area_ratio"],
            settings["max_detections_per_prompt"],
        )
        all_candidates.extend(candidates)
        after_confidence += count
        truncated_by_limit += max(0, count - settings["max_detections_per_prompt"])
        filtered_by_area += area_filtered

    kept = global_mask_nms(
        all_candidates,
        settings["mask_nms_iou"],
        settings["max_candidates_per_image"],
    )
    stats = {
        "after_confidence": after_confidence,
        "truncated_by_limit": truncated_by_limit,
        "filtered_by_area": filtered_by_area,
        "before_global_mask_nms": len(all_candidates),
        "kept": len(kept),
    }
    return kept, stats


def _normalized(value: float, size: int) -> float:
    return max(0.0, min(1.0, float(value) / max(size - 1, 1)))


def _map_pixel(value: float, source_size: int, target_size: int) -> float:
    return _normalized(value, source_size) * max(target_size - 1, 1)


def resize_for_inference(image: Any, max_image_side: int) -> Any:
    if max_image_side <= 0 or max(image.size) <= max_image_side:
        return image
    scale = max_image_side / float(max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    from PIL import Image

    return image.resize(size, Image.Resampling.LANCZOS)


def candidate_row(
    image_path: Path,
    image_relpath: Path,
    image_id: str,
    candidate_index: int,
    candidate: Candidate,
    width: int,
    height: int,
    mask_path: str,
) -> dict[str, Any]:
    mask_height, mask_width = candidate.mask.shape
    x0, y0, x1, y1 = candidate.bbox_px
    cx, cy = candidate.centroid_px
    gx, gy = candidate.grasp_px
    x_scale = max(width - 1, 1) / max(mask_width - 1, 1)
    y_scale = max(height - 1, 1) / max(mask_height - 1, 1)
    return {
        "image": image_path.name,
        "image_relpath": image_relpath.as_posix(),
        "image_id": image_id,
        "image_width": width,
        "image_height": height,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "candidate_index": candidate_index,
        "model": "facebook/sam3",
        "class_id": candidate.prompt.class_id,
        "class_name": candidate.prompt.class_name,
        "prompt": candidate.prompt.text,
        "confidence": f"{candidate.confidence:.6f}",
        "area_pixels": candidate.area_pixels,
        "area_ratio": f"{candidate.area_ratio:.6f}",
        "centroid_x": f"{_normalized(cx, mask_width):.6f}",
        "centroid_y": f"{_normalized(cy, mask_height):.6f}",
        "centroid_x_px": f"{_map_pixel(cx, mask_width, width):.3f}",
        "centroid_y_px": f"{_map_pixel(cy, mask_height, height):.3f}",
        "grasp_x": f"{_normalized(gx, mask_width):.6f}",
        "grasp_y": f"{_normalized(gy, mask_height):.6f}",
        "grasp_x_px": int(round(_map_pixel(gx, mask_width, width))),
        "grasp_y_px": int(round(_map_pixel(gy, mask_height, height))),
        "grasp_clearance_px": f"{candidate.grasp_clearance_px * (x_scale + y_scale) / 2.0:.3f}",
        "bbox_min_x": f"{_normalized(x0, mask_width):.6f}",
        "bbox_min_y": f"{_normalized(y0, mask_height):.6f}",
        "bbox_max_x": f"{_normalized(x1, mask_width):.6f}",
        "bbox_max_y": f"{_normalized(y1, mask_height):.6f}",
        "bbox_min_x_px": int(round(_map_pixel(x0, mask_width, width))),
        "bbox_min_y_px": int(round(_map_pixel(y0, mask_height, height))),
        "bbox_max_x_px": int(round(_map_pixel(x1, mask_width, width))),
        "bbox_max_y_px": int(round(_map_pixel(y1, mask_height, height))),
        "polygon_points": int(len(candidate.polygon_px)),
        "mask_components": candidate.component_count,
        "mask_holes": candidate.hole_count,
        "mask_path": mask_path,
    }


def candidate_json(row: dict[str, Any], candidate: Candidate, width: int, height: int) -> dict[str, Any]:
    mask_height, mask_width = candidate.mask.shape
    x0, y0, x1, y1 = candidate.bbox_px
    cx, cy = candidate.centroid_px
    gx, gy = candidate.grasp_px
    polygon = [
        [_normalized(float(point[0]), mask_width), _normalized(float(point[1]), mask_height)]
        for point in candidate.polygon_px.tolist()
    ]
    return {
        "image": row["image"],
        "image_relpath": row["image_relpath"],
        "image_id": row["image_id"],
        "image_size": {"width": width, "height": height},
        "candidate_index": row["candidate_index"],
        "model": row["model"],
        "class_id": candidate.prompt.class_id,
        "class_name": candidate.prompt.class_name,
        "prompt": candidate.prompt.text,
        "confidence": candidate.confidence,
        "area_pixels": candidate.area_pixels,
        "area_ratio": candidate.area_ratio,
        "centroid": {
            "x": _normalized(cx, mask_width),
            "y": _normalized(cy, mask_height),
            "x_px": _map_pixel(cx, mask_width, width),
            "y_px": _map_pixel(cy, mask_height, height),
        },
        "grasp_point": {
            "x": _normalized(gx, mask_width),
            "y": _normalized(gy, mask_height),
            "x_px": int(round(_map_pixel(gx, mask_width, width))),
            "y_px": int(round(_map_pixel(gy, mask_height, height))),
            "mask_clearance_px": float(row["grasp_clearance_px"]),
            "inside_mask": bool(candidate.mask[gy, gx]),
        },
        "bbox": {
            "min_x": _normalized(x0, mask_width),
            "min_y": _normalized(y0, mask_height),
            "max_x": _normalized(x1, mask_width),
            "max_y": _normalized(y1, mask_height),
            "min_x_px": row["bbox_min_x_px"],
            "min_y_px": row["bbox_min_y_px"],
            "max_x_px": row["bbox_max_x_px"],
            "max_y_px": row["bbox_max_y_px"],
        },
        "polygon": polygon,
        "polygon_is_lossy": candidate.component_count > 1 or candidate.hole_count > 0,
        "mask": {
            "path": row["mask_path"] or None,
            "format": "8-bit PNG; foreground=255" if row["mask_path"] else None,
            "width": mask_width,
            "height": mask_height,
            "components": candidate.component_count,
            "holes": candidate.hole_count,
        },
    }


def _class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (46, 204, 113),
        (231, 76, 60),
        (243, 156, 18),
        (241, 196, 15),
        (52, 152, 219),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
        (233, 30, 99),
    ]
    return palette[class_id % len(palette)]


def render_overlay(image: Any, candidates: list[Candidate], output_path: Path) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tint = base.copy()
    for candidate in candidates:
        mask = candidate.mask
        if mask.shape != base.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        tint[mask] = _class_color(candidate.prompt.class_id)
    canvas = cv2.addWeighted(tint, 0.35, base, 0.65, 0)

    for index, candidate in enumerate(candidates):
        color = _class_color(candidate.prompt.class_id)
        mask_height, mask_width = candidate.mask.shape
        polygon = np.asarray(
            [
                [
                    round(_map_pixel(point[0], mask_width, base.shape[1])),
                    round(_map_pixel(point[1], mask_height, base.shape[0])),
                ]
                for point in candidate.polygon_px.tolist()
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        gx, gy = candidate.grasp_px
        gx = round(_map_pixel(gx, mask_width, base.shape[1]))
        gy = round(_map_pixel(gy, mask_height, base.shape[0]))
        cv2.drawMarker(canvas, (gx, gy), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        x0, y0, _, _ = candidate.bbox_px
        x0 = round(_map_pixel(x0, mask_width, base.shape[1]))
        y0 = round(_map_pixel(y0, mask_height, base.shape[0]))
        label = f"{index}:{candidate.prompt.class_name} {candidate.confidence:.2f}"
        text_y = max(18, y0 - 5)
        cv2.putText(canvas, label, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(output_path, quality=92)


def write_delimited(path: Path, rows: list[dict[str, Any]], columns: list[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sam3_commit(sam3_file: str | None) -> str | None:
    if not sam3_file:
        return None
    path = Path(sam3_file).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    [
                        "git",
                        "-c",
                        f"safe.directory={parent}",
                        "-C",
                        str(parent),
                        "rev-parse",
                        "HEAD",
                    ],
                    text=True,
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                return None
    return None


def _prepare_output(path: Path, allow_existing: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new directory or pass --allow-existing-output."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_full_checkpoint(checkpoint: Path, torch_module: Any) -> dict[str, Any]:
    """Reject small training adapters that the official loader would silently ignore."""
    size_bytes = checkpoint.stat().st_size
    if size_bytes < 1_000_000_000:
        raise ValueError(
            "Checkpoint is too small to be the released full SAM 3 image model "
            f"({size_bytes:,} bytes). If this is best_adapter.pt, export and pass "
            "best_merged.pt instead."
        )
    payload = torch_module.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"SAM 3 checkpoint must contain a mapping: {checkpoint}")
    if payload.get("format") == "sam3-watermelon-adapter-v1":
        raise ValueError(
            "This is an adapter-only training checkpoint. Pass the exported "
            "best_merged.pt deployment checkpoint instead."
        )
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise ValueError(f"SAM 3 checkpoint has no model state mapping: {checkpoint}")
    detector_keys = [str(key) for key in state if str(key).startswith("detector.")]
    required_key = "detector.backbone.vision_backbone.trunk.pos_embed"
    if required_key not in state or len(detector_keys) < 100:
        raise ValueError(
            "Checkpoint is not a complete SAM 3 image deployment checkpoint: "
            f"found {len(detector_keys)} detector.* keys and required key is "
            f"{'present' if required_key in state else 'missing'}."
        )
    return {
        "size_bytes": size_bytes,
        "detector_key_count": len(detector_keys),
        "has_wrapped_model_state": state is not payload,
    }


def _load_model(settings: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        import torch
        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise RuntimeError(
            "SAM 3 runtime import failed. Activate the sam3 environment and install deploy/sam3/requirements-windows.txt."
        ) from exc

    checkpoint = settings["checkpoint"]
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    cache_ready = bool(cached_checkpoint(settings["repo_id"]) and cached_config(settings["repo_id"]))
    if checkpoint is None and not cache_ready and not hf_token_present():
        raise RuntimeError(
            "SAM 3 weights are gated and no credential/cache was found. Request access at "
            "https://huggingface.co/facebook/sam3 and run `hf auth login`, or pass --checkpoint."
        )
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    checkpoint_validation = None
    if checkpoint is not None:
        checkpoint_validation = _validate_full_checkpoint(checkpoint, torch)

    torch.set_float32_matmul_precision("high")
    if settings["device"] == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        model = build_sam3_image_model(
            device=settings["device"],
            checkpoint_path=str(checkpoint) if checkpoint is not None else None,
            load_from_HF=checkpoint is None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "gated" in message or "401" in message or "403" in message or "authorized" in message:
            raise RuntimeError(
                "Hugging Face rejected the gated SAM 3 download. Confirm approval with "
                "`hf download facebook/sam3 config.json` after `hf auth login`."
            ) from exc
        raise

    processor = Sam3Processor(
        model,
        resolution=settings["resolution"],
        device=settings["device"],
        confidence_threshold=settings["confidence_threshold"],
    )
    model_info = {
        "package_version": getattr(sam3, "__version__", None),
        "package_path": str(Path(sam3.__file__).resolve()),
        "git_commit": _sam3_commit(sam3.__file__),
        "checkpoint": str(checkpoint) if checkpoint is not None else "hf://facebook/sam3/sam3.pt",
        "checkpoint_sha256": _sha256_file(checkpoint) if checkpoint is not None else None,
        "checkpoint_validation": checkpoint_validation,
    }
    return model, processor, torch, model_info


def run_inference(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, Any]:
    if args.source is None:
        raise ValueError("--source is required for inference")
    source = args.source.expanduser().resolve()
    images = image_paths(source, args.recursive, args.include_glob, args.exclude_glob)
    if not images:
        raise ValueError(f"No selected images found under: {source}")

    output_dir = _prepare_output(args.output_dir, args.allow_existing_output)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    model, processor, torch_module, model_info = _load_model(settings)

    from PIL import Image

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    candidate_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    total_after_confidence = 0
    total_before_nms = 0
    success_count = 0
    error_count = 0
    stopped_early = False

    classes_payload = {
        "model": settings["repo_id"],
        "config": str(settings["config_path"]),
        "classes": [
            {
                "class_id": prompt.class_id,
                "class_name": prompt.class_name,
                "prompt": prompt.text,
                "priority": prompt.priority,
                "confidence_threshold": prompt.confidence_threshold,
            }
            for prompt in settings["prompts"]
        ],
    }
    (output_dir / "classes.json").write_text(
        json.dumps(classes_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for image_index, image_path in enumerate(images, start=1):
        image_started = time.perf_counter()
        image_relpath = relative_image_path(image_path, source)
        image_id = stable_image_id(image_relpath)
        written_files: list[Path] = []
        print(f"[{image_index}/{len(images)}] {image_relpath.as_posix()}", flush=True)
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            inference_image = resize_for_inference(image, settings["max_image_side"])
            inference_width, inference_height = inference_image.size
            candidates, stats = predict_image(
                inference_image,
                processor,
                settings["prompts"],
                settings,
                torch_module,
            )
            image_candidate_rows: list[dict[str, Any]] = []
            image_jsonl_rows: list[dict[str, Any]] = []
            image_class_counts: Counter[str] = Counter()
            for candidate_index, candidate in enumerate(candidates):
                mask_relpath = ""
                if settings["save_masks"]:
                    safe_class = re.sub(r"[^0-9A-Za-z_-]+", "_", candidate.prompt.class_name)
                    mask_rel = Path("masks") / image_id / f"{candidate_index:03d}_{safe_class}.png"
                    mask_path = output_dir / mask_rel
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((candidate.mask.astype("uint8") * 255), mode="L").save(mask_path)
                    written_files.append(mask_path)
                    mask_relpath = mask_rel.as_posix()

                row = candidate_row(
                    image_path,
                    image_relpath,
                    image_id,
                    candidate_index,
                    candidate,
                    width,
                    height,
                    mask_relpath,
                )
                image_candidate_rows.append(row)
                image_jsonl_rows.append(candidate_json(row, candidate, width, height))
                image_class_counts[candidate.prompt.class_name] += 1

            if settings["save_overlays"]:
                overlay_path = overlays_dir / f"{image_id}.jpg"
                render_overlay(image, candidates, overlay_path)
                written_files.append(overlay_path)

            candidate_rows.extend(image_candidate_rows)
            jsonl_rows.extend(image_jsonl_rows)
            class_counts.update(image_class_counts)
            total_after_confidence += stats["after_confidence"]
            total_before_nms += stats["before_global_mask_nms"]

            manifest_rows.append(
                {
                    "image": image_path.name,
                    "image_relpath": image_relpath.as_posix(),
                    "image_id": image_id,
                    "status": "ok",
                    "width": width,
                    "height": height,
                    "inference_width": inference_width,
                    "inference_height": inference_height,
                    "prompt_count": len(settings["prompts"]),
                    **stats,
                    "elapsed_seconds": f"{time.perf_counter() - image_started:.4f}",
                    "error": "",
                }
            )
            success_count += 1
        except Exception as exc:
            error_count += 1
            if args.debug:
                traceback.print_exc()
            for written_file in written_files:
                try:
                    written_file.unlink(missing_ok=True)
                except OSError:
                    pass
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            manifest_rows.append(
                {
                    "image": image_path.name,
                    "image_relpath": image_relpath.as_posix(),
                    "image_id": image_id,
                    "status": "error",
                    "width": "",
                    "height": "",
                    "inference_width": "",
                    "inference_height": "",
                    "prompt_count": len(settings["prompts"]),
                    "after_confidence": "",
                    "truncated_by_limit": "",
                    "filtered_by_area": "",
                    "before_global_mask_nms": "",
                    "kept": "",
                    "elapsed_seconds": f"{time.perf_counter() - image_started:.4f}",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
            if not args.continue_on_error:
                stopped_early = True
                break

    write_delimited(output_dir / "prediction_manifest.tsv", manifest_rows, MANIFEST_COLUMNS, "\t")
    write_delimited(output_dir / "robot_candidates.csv", candidate_rows, CANDIDATE_COLUMNS, ",")
    with (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started
    cuda_memory: dict[str, float] | None = None
    if settings["device"] == "cuda" and torch_module.cuda.is_available():
        cuda_memory = {
            "peak_allocated_mib": round(torch_module.cuda.max_memory_allocated() / 1024**2, 1),
            "peak_reserved_mib": round(torch_module.cuda.max_memory_reserved() / 1024**2, 1),
        }
    summary = {
        "status": "complete" if error_count == 0 else ("partial" if success_count else "failed"),
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": elapsed,
        "source": str(source),
        "output_dir": str(output_dir),
        "model": model_info,
        "runtime": {
            "device": settings["device"],
            "resolution": settings["resolution"],
            "max_image_side": settings["max_image_side"],
            "amp_dtype": settings["amp_dtype"],
            "confidence_threshold": settings["confidence_threshold"],
            "min_area_ratio": settings["min_area_ratio"],
            "mask_nms_iou": settings["mask_nms_iou"],
            "max_detections_per_prompt": settings["max_detections_per_prompt"],
            "max_candidates_per_image": settings["max_candidates_per_image"],
            "cuda_memory": cuda_memory,
        },
        "images": {
            "selected": len(images),
            "succeeded": success_count,
            "failed": error_count,
            "unprocessed": len(images) - success_count - error_count,
            "stopped_early": stopped_early,
            "average_seconds_per_success": elapsed / success_count if success_count else None,
        },
        "candidates": {
            "after_confidence": total_after_confidence,
            "before_global_mask_nms": total_before_nms,
            "kept": len(candidate_rows),
            "by_class": dict(sorted(class_counts.items())),
        },
        "files": {
            "classes": str(output_dir / "classes.json"),
            "manifest": str(output_dir / "prediction_manifest.tsv"),
            "robot_candidates_csv": str(output_dir / "robot_candidates.csv"),
            "candidates_jsonl": str(output_dir / "candidates.jsonl"),
            "masks": str(masks_dir) if settings["save_masks"] else None,
            "overlays": str(overlays_dir) if settings["save_overlays"] else None,
            "summary": str(output_dir / "prediction_summary.json"),
        },
        "safety": {
            "grasp_point_is_image_space_only": True,
            "depth_calibration_and_collision_check_required": True,
            "zero_shot_accuracy_requires_per_fruit_site_validation": True,
        },
    }
    (output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    del processor, model
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings(args)
        if args.diagnose:
            print(json.dumps(diagnostics(settings), indent=2, ensure_ascii=False))
            return 0
        if args.dry_run:
            print(json.dumps(dry_run(args, settings), indent=2, ensure_ascii=False))
            return 0
        summary = run_inference(args, settings)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["status"] == "complete" else 2
    except Exception as exc:
        if args.debug:
            raise
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
