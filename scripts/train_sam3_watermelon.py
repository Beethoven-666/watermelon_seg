#!/usr/bin/env python
"""Parameter-efficient SAM 3 watermelon instance-segmentation fine-tuning.

This launcher intentionally avoids the official distributed Trainer so that a
single CUDA GPU can be used from native Windows.  It still reuses the official
SAM 3 model, COCO dataset, transforms, collator, matcher, and loss functions.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.transforms.basic import (
    get_random_resize_max_size,
    get_random_resize_scales,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    PadToSizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.transforms.filter_query_transforms import (
    FilterCrowds,
    FilterEmptyTargets,
    FilterFindQueriesWithTooManyOut,
    FlexibleFilterFindGetQueries,
)
from sam3.train.transforms.point_sampling import RandomizeInputBbox
from sam3.train.transforms.segmentation import DecodeRle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "exports" / "sam3_watermelon_finetune_v2"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "runs" / "runs28" / "sam3" / "watermelon_head_finetune"
)
CORE_LOSS_KEY = "core_loss"
SUPPORTED_RESOLUTION = 1008
FREEZE_PREFIXES = {
    "non_backbone": (
        "geometry_encoder",
        "transformer",
        "segmentation_head",
        "dot_prod_scoring",
    ),
    "decoder_heads": (
        "transformer.decoder",
        "segmentation_head",
        "dot_prod_scoring",
    ),
    "heads": ("segmentation_head", "dot_prod_scoring"),
    "mask_head": ("segmentation_head",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--train-ann",
        type=Path,
        help="Defaults to <dataset-root>/annotations/instances_train_clean.json.",
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        help="Defaults to <dataset-root>/annotations/instances_val_clean.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"D:\MelonDataset\sam3\sam3.pt"),
        help="Approved local facebook/sam3 base checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--cooldown-steps", type=int, default=20)
    parser.add_argument("--scheduler-timescale", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pin collated CPU batches. Disabled by default on the native-Windows "
            "single-process path; SAM 3's BatchedDatapoint receives no pinning benefit."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--freeze-policy",
        choices=tuple(FREEZE_PREFIXES),
        default="non_backbone",
    )
    parser.add_argument("--resolution", type=int, default=SUPPORTED_RESOLUTION)
    parser.add_argument("--min-resize", type=int, default=480)
    parser.add_argument("--mask-sample-points", type=int, default=12544)
    parser.add_argument("--mask-oversample-ratio", type=float, default=3.0)
    parser.add_argument("--mask-importance-ratio", type=float, default=0.75)
    parser.add_argument("--gradient-clip", type=float, default=0.1)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Per-epoch cap; 0 means the full training split.",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=0,
        help="Per-epoch cap; 0 means the full validation split.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--val-max-image-side",
        type=int,
        default=1024,
        help="Image-side cap for exact frozen-val SAM processor evaluation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("mask_map_50_95", "mask_ap50", "f1"),
        default="mask_map_50_95",
        help="Metric maximized on clean val when --max-val-steps is 0.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--no-merged-checkpoint", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset/collator and write metadata without loading SAM 3.",
    )
    parser.add_argument("--val-only-subprocess", action="store_true")
    parser.add_argument("--val-epoch", type=int)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--val-output-json", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def match_hash(path: Path, expected: str) -> bool:
    actual = sha256_file(path)
    if actual == expected:
        return True
    if path.suffix in (".json", ".tsv", ".csv", ".txt"):
        try:
            content = path.read_bytes()
            actual_lf = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
            if actual_lf == expected:
                return True
            actual_crlf = hashlib.sha256(content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
            if actual_crlf == expected:
                return True
        except OSError:
            pass
    return False


def host_memory_snapshot() -> dict[str, float]:
    """Return process and system memory telemetry in GiB."""
    process_info = psutil.Process().memory_info()
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    private_bytes = getattr(process_info, "private", process_info.rss)
    return {
        "process_rss_gib": process_info.rss / 2**30,
        "process_private_gib": private_bytes / 2**30,
        "system_available_gib": virtual_memory.available / 2**30,
        "system_memory_percent": float(virtual_memory.percent),
        "pagefile_free_gib": swap_memory.free / 2**30,
    }


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def verify_dataset_integrity(project_root: Path, dataset_root: Path) -> dict[str, Any]:
    integrity_path = dataset_root / "integrity.json"
    manifest_path = dataset_root / "manifests" / "images.tsv"
    if not integrity_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"SAM 3 derived dataset integrity files are missing under {dataset_root}"
        )
    integrity = json.loads(integrity_path.read_text(encoding="utf-8-sig"))
    if integrity.get("status") != "pass":
        raise ValueError(f"Derived dataset integrity status is not pass: {integrity_path}")
    checked_generated = 0
    for relative, expected in integrity.get("generated_files_sha256", {}).items():
        path = dataset_root / relative
        if not path.is_file() or not match_hash(path, expected):
            raise ValueError(f"Derived dataset file hash mismatch: {path}")
        checked_generated += 1

    checked_rows = 0
    seen_pairs: set[tuple[str, str]] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            image_relative = row["relative_image_path"]
            label_relative = row["relative_label_path"]
            pair = (image_relative, label_relative)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate image/label row in {manifest_path}: {pair}")
            seen_pairs.add(pair)
            image_path = project_root / image_relative
            label_path = project_root / label_relative
            if not match_hash(image_path, row["image_sha256"]):
                raise ValueError(f"Source image drift detected: {image_path}")
            if not match_hash(label_path, row["label_sha256"]):
                raise ValueError(f"Source label drift detected: {label_path}")
            checked_rows += 1
    return {
        "status": "pass",
        "integrity_json": str(integrity_path),
        "integrity_json_sha256": sha256_file(integrity_path),
        "images_manifest": str(manifest_path),
        "images_manifest_sha256": sha256_file(manifest_path),
        "generated_files_checked": checked_generated,
        "source_image_label_pairs_checked": checked_rows,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.train_ann = (
        args.train_ann
        or args.dataset_root / "annotations" / "instances_train_clean.json"
    ).expanduser().resolve()
    args.val_ann = (
        args.val_ann
        or args.dataset_root / "annotations" / "instances_val_clean.json"
    ).expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
    if args.adapter_path is not None:
        args.adapter_path = args.adapter_path.expanduser().resolve()
    if args.val_output_json is not None:
        args.val_output_json = args.val_output_json.expanduser().resolve()
    if args.resolution != SUPPORTED_RESOLUTION:
        raise ValueError(
            "The released SAM 3 image checkpoint is wired for resolution=1008; "
            "arbitrary lower resolutions fail its RoPE shape assertions."
        )
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if args.min_resize < 32 or args.min_resize > args.resolution:
        raise ValueError("min-resize must be in [32, resolution]")
    if args.max_train_steps < 0 or args.max_val_steps < 0:
        raise ValueError("step caps cannot be negative")
    if args.val_max_image_side < 224:
        raise ValueError("val-max-image-side must be at least 224")
    if args.mask_sample_points < 1:
        raise ValueError("mask-sample-points must be positive")
    for path in (args.project_root, args.dataset_root, args.train_ann, args.val_ann):
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.dry_run and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    return args


def prepare_output(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        permitted = args.allow_existing_output or args.resume is not None
        if not permitted:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use a new directory, --resume, or --allow-existing-output."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def build_transforms(resolution: int, min_resize: int, training: bool):
    if training:
        transforms = [
            FlexibleFilterFindGetQueries(FilterCrowds()),
            RandomizeInputBbox(box_noise_std=0.1, box_noise_max=20),
            DecodeRle(),
            RandomResizeAPI(
                sizes=get_random_resize_scales(
                    size=resolution, min_size=min_resize, rounded=False
                ),
                max_size=get_random_resize_max_size(size=resolution),
                square=True,
                consistent_transform=False,
            ),
            PadToSizeAPI(size=resolution, consistent_transform=False),
            ToTensorAPI(),
            FlexibleFilterFindGetQueries(FilterEmptyTargets()),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            FlexibleFilterFindGetQueries(FilterEmptyTargets()),
        ]
        return [
            ComposeAPI(transforms),
            FlexibleFilterFindGetQueries(
                FilterFindQueriesWithTooManyOut(max_num_objects=200)
            ),
        ]
    return [
        ComposeAPI(
            [
                DecodeRle(),
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=get_random_resize_max_size(size=resolution),
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    ]


def build_dataset(args: argparse.Namespace, training: bool) -> Sam3ImageDataset:
    ann_file = args.train_ann if training else args.val_ann
    coco_loader = functools.partial(
        COCO_FROM_JSON,
        include_negatives=True,
        category_chunk_size=1,
    )
    return Sam3ImageDataset(
        img_folder=str(args.project_root),
        ann_file=str(ann_file),
        transforms=build_transforms(args.resolution, args.min_resize, training),
        max_ann_per_img=200,
        multiplier=1,
        training=training,
        load_segmentation=True,
        max_train_queries=200,
        max_val_queries=200,
        use_caching=False,
        coco_json_loader=coco_loader,
    )


def make_loader(
    dataset: Sam3ImageDataset,
    args: argparse.Namespace,
    training: bool,
    epoch: int,
) -> DataLoader:
    dataset.set_curr_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch + (0 if training else 1_000_000))
    collate = functools.partial(
        collate_fn_api,
        dict_key="all",
        with_seg_masks=True,
        repeats=1,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=training,
        collate_fn=collate,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )


def inspect_collated_batch(batch: dict[str, Any], resolution: int) -> dict[str, Any]:
    if set(batch) != {"all"}:
        raise ValueError(f"Unexpected collator keys: {sorted(batch)}")
    data = batch["all"]
    if tuple(data.img_batch.shape[-2:]) != (resolution, resolution):
        raise ValueError(f"Unexpected image tensor shape: {tuple(data.img_batch.shape)}")
    if not data.find_targets or data.find_targets[0].segments is None:
        raise ValueError("Segmentation targets were not loaded/collated")
    return {
        "batch_shape": list(data.img_batch.shape),
        "text_prompts": list(data.find_text_batch),
        "stages": len(data.find_targets),
        "instances": int(sum(target.num_boxes.sum().item() for target in data.find_targets)),
        "mask_shape": list(data.find_targets[0].segments.shape),
    }


def build_base_model_memory_efficient(
    checkpoint: Path,
    use_mmap: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build SAM 3 and copy checkpoint tensors from a read-only memory map.

    The upstream loader receives a file object, so ``torch.load`` materializes
    the complete 3.45 GB checkpoint in committed CPU memory alongside the CPU
    model. That peak can leave too little Windows commit for decoded masks on
    a 16 GB workstation. Loading from a path with ``mmap=True`` preserves the
    upstream key mapping while avoiding the second committed tensor copy.
    """
    print("build_base_model: entering build_sam3_image_model...", flush=True)
    model = build_sam3_image_model(
        device="cpu",
        eval_mode=False,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    print("build_base_model: entering torch.load...", flush=True)
    payload = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=True,
        mmap=use_mmap,
    )
    print("build_base_model: entering state_dict filtering...", flush=True)
    if "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported SAM 3 checkpoint payload: {checkpoint}")
    detector_state = {
        key.replace("detector.", ""): tensor
        for key, tensor in payload.items()
        if "detector" in key
    }
    if not detector_state:
        raise ValueError(f"No detector.* tensors found in SAM 3 checkpoint: {checkpoint}")
    print("build_base_model: entering model.load_state_dict...", flush=True)
    missing, unexpected = model.load_state_dict(detector_state, strict=False)
    print("build_base_model: finished loading state dict...", flush=True)
    load_info = {
        "mode": f"torch.load(weights_only=True, mmap={use_mmap})",
        "detector_tensor_count": len(detector_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    del detector_state, payload
    gc.collect()
    return model, load_info


def is_trainable_name(name: str, policy: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FREEZE_PREFIXES[policy])


def apply_freeze_policy(model: torch.nn.Module, policy: str) -> list[str]:
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_trainable_name(name, policy))
        if parameter.requires_grad:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError(f"Freeze policy {policy!r} selected no parameters")
    return trainable_names


def set_model_phase(model: torch.nn.Module, training: bool) -> None:
    model.train(training)
    if training:
        # Frozen modules should not update dropout/statistical state.
        for module in model.modules():
            parameters = list(module.parameters(recurse=True))
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()


def install_frozen_backbone_no_grad(model: torch.nn.Module) -> None:
    """Run a fully frozen backbone without autograd while the heads train.

    The released ViT uses an inference-only fused addmm kernel whenever its
    frozen backbone is in eval mode.  That kernel explicitly requires global
    grad to be disabled for the backbone call.
    """
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        return
    original_forward_image = model.backbone.forward_image
    original_forward_text = model.backbone.forward_text

    def forward_image_no_grad(*args, **kwargs):
        with torch.no_grad():
            return original_forward_image(*args, **kwargs)

    def forward_text_no_grad(*args, **kwargs):
        with torch.no_grad():
            return original_forward_text(*args, **kwargs)

    model.backbone.forward_image = forward_image_no_grad
    model.backbone.forward_text = forward_text_no_grad


def parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    by_top_level: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "trainable": 0}
    )
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        group = name.split(".", 1)[0]
        total += count
        by_top_level[group]["total"] += count
        if parameter.requires_grad:
            trainable += count
            by_top_level[group]["trainable"] += count
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_fraction": trainable / total,
        "by_top_level": dict(sorted(by_top_level.items())),
    }


def write_parameter_manifest(model: torch.nn.Module, output_dir: Path) -> None:
    with (output_dir / "parameters.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("name", "shape", "numel", "trainable", "dtype"))
        for name, parameter in model.named_parameters():
            writer.writerow(
                (
                    name,
                    "x".join(map(str, parameter.shape)),
                    parameter.numel(),
                    int(parameter.requires_grad),
                    str(parameter.dtype),
                )
            )


def build_loss(args: argparse.Namespace) -> Sam3LossWrapper:
    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2.0,
        stable=False,
    )
    return Sam3LossWrapper(
        matcher=matcher,
        o2m_weight=2.0,
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
        use_o2m_matcher_on_o2m_aux=False,
        normalization="local",
        scale_by_find_batch_size=True,
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                weak_loss=False,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                pos_weight=10.0,
                alpha=0.25,
                gamma=2.0,
                use_presence=True,
                pos_focal=False,
                pad_n_queries=200,
                pad_scale_pos=1.0,
            ),
            Masks(
                focal_alpha=0.25,
                focal_gamma=2.0,
                weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                compute_aux=False,
                num_sample_points=args.mask_sample_points,
                oversample_ratio=args.mask_oversample_ratio,
                importance_sample_ratio=args.mask_importance_ratio,
            ),
        ],
    )


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    return optimizer


def scheduled_lr(args: argparse.Namespace, step: int, total_steps: int) -> float:
    lr = args.learning_rate
    if step > args.warmup_steps:
        shift = args.scheduler_timescale - args.warmup_steps
        lr /= math.sqrt((step + shift) / args.scheduler_timescale)
    if args.warmup_steps:
        lr *= min(1.0, step / args.warmup_steps)
    if args.cooldown_steps:
        lr *= min(1.0, max(total_steps - step, 0) / args.cooldown_steps)
    return lr


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def scalar_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().float().item()) for key, value in losses.items()}


def forward_loss(
    model: torch.nn.Module,
    criterion: Sam3LossWrapper,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    if set(batch) != {"all"}:
        raise ValueError(f"Unexpected batch keys: {sorted(batch)}")
    data = copy_data_to_device(batch["all"], device, non_blocking=True)
    outputs = model(data)
    targets = [model.back_convert(target) for target in data.find_targets]
    # The official collator represents a target-free query's segments as a
    # one-dimensional empty tensor.  Sampled mask loss expects N x H x W even
    # when N=0, so preserve the negative query with an explicit empty mask.
    height, width = data.img_batch.shape[-2:]
    for target in targets:
        masks = target["masks"]
        if masks is not None and masks.numel() == 0 and masks.ndim != 3:
            target["masks"] = masks.reshape(0, height, width)
    losses = criterion(outputs, targets)
    core_loss = losses[CORE_LOSS_KEY]
    if not torch.isfinite(core_loss):
        raise FloatingPointError(f"Non-finite loss: {float(core_loss.detach())}")
    return core_loss, losses, len(data.img_batch)


def cap_steps(loader: DataLoader, cap: int) -> int:
    return min(len(loader), cap) if cap else len(loader)


def batch_identity(batch: dict[str, Any]) -> dict[str, Any]:
    data = batch.get("all")
    if data is None:
        return {"keys": sorted(batch)}
    identities: list[dict[str, list[int]]] = []
    for metadata in data.find_metadatas:
        identities.append(
            {
                "original_image_id": metadata.original_image_id.reshape(-1).tolist(),
                "original_category_id": metadata.original_category_id.reshape(-1).tolist(),
                "coco_image_id": metadata.coco_image_id.reshape(-1).tolist(),
            }
        )
    return {"stages": identities}


def train_epoch(
    model: torch.nn.Module,
    criterion: Sam3LossWrapper,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    global_step: int,
    total_steps: int,
) -> tuple[dict[str, Any], int]:
    set_model_phase(model, training=True)
    limit = cap_steps(loader, args.max_train_steps)
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    for batch_index, batch in enumerate(loader):
        if batch_index >= limit:
            break
        global_step += 1
        lr = scheduled_lr(args, global_step, total_steps)
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
                core_loss, losses, batch_size = forward_loss(
                    model, criterion, batch, device
                )
            scaler.scale(core_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=args.gradient_clip,
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm: {float(grad_norm)}")
            scaler.step(optimizer)
            scaler.update()
        except Exception as exc:
            json_dump(
                args.output_dir / "failure_context.json",
                {
                    "timestamp": utc_now(),
                    "epoch": epoch + 1,
                    "epoch_step": batch_index + 1,
                    "global_step_attempted": global_step,
                    "batch": batch_identity(batch),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            raise

        values = scalar_losses(losses)
        for key, value in values.items():
            sums[key] += value * batch_size
        samples += batch_size
        if global_step == 1 or global_step % args.log_every == 0 or batch_index + 1 == limit:
            allocated = torch.cuda.max_memory_allocated(device) / 2**30
            reserved = torch.cuda.max_memory_reserved(device) / 2**30
            progress_row = {
                "timestamp": utc_now(),
                "phase": "train_progress",
                "epoch": epoch + 1,
                "epoch_step": batch_index + 1,
                "epoch_steps": limit,
                "global_step": global_step,
                "core_loss": values[CORE_LOSS_KEY],
                "learning_rate": lr,
                "gradient_norm_before_clip": float(grad_norm),
                "peak_allocated_gib": allocated,
                "peak_reserved_gib": reserved,
                **host_memory_snapshot(),
            }
            append_jsonl(args.output_dir / "progress.jsonl", progress_row)
            print(
                f"train epoch={epoch + 1}/{args.epochs} step={batch_index + 1}/{limit} "
                f"global={global_step} loss={values[CORE_LOSS_KEY]:.5f} "
                f"lr={lr:.3e} grad={float(grad_norm):.4f} "
                f"peak_alloc={allocated:.2f}GiB peak_reserved={reserved:.2f}GiB",
                flush=True,
            )
        del batch, core_loss, losses, grad_norm

    elapsed = time.perf_counter() - started
    if not samples:
        raise RuntimeError("Training loader yielded no samples")
    return (
        {
            "phase": "train",
            "epoch": epoch + 1,
            "steps": limit,
            "samples": samples,
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / limit,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "losses": {key: value / samples for key, value in sorted(sums.items())},
        },
        global_step,
    )


@torch.inference_mode()
def validate_epoch(
    model: torch.nn.Module,
    criterion: Sam3LossWrapper,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    set_model_phase(model, training=False)
    # Sam3Image only attaches Hungarian match indices when its root training
    # flag is true.  Keep every child module in eval mode, but enable that
    # matching branch so validation loss can be used as a selection proxy.
    model.training = True
    limit = cap_steps(loader, args.max_val_steps)
    sums: dict[str, float] = defaultdict(float)
    samples = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    for batch_index, batch in enumerate(loader):
        if batch_index >= limit:
            break
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
            core_loss, losses, batch_size = forward_loss(model, criterion, batch, device)
        for key, value in scalar_losses(losses).items():
            sums[key] += value * batch_size
        samples += batch_size
        del batch, core_loss, losses
    elapsed = time.perf_counter() - started
    if not samples:
        raise RuntimeError("Validation loader yielded no samples")
    metrics = {
        "phase": "val",
        "epoch": epoch + 1,
        "steps": limit,
        "samples": samples,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / limit,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "losses": {key: value / samples for key, value in sorted(sums.items())},
        "selection_metric_note": (
            "proxy validation loss with child modules in eval mode; final quality is "
            "measured separately with frozen-val COCO mask metrics"
        ),
    }
    print(
        f"val epoch={epoch + 1}/{args.epochs} loss={metrics['losses'][CORE_LOSS_KEY]:.5f} "
        f"steps={limit} seconds={elapsed:.1f}",
        flush=True,
    )
    return metrics


@torch.inference_mode()
def validate_coco_epoch(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    """Run the same frozen-val image processor and COCO metrics as deployment."""
    import benchmark_sam3_watermelon as benchmark
    from sam3.model.sam3_image_processor import Sam3Processor

    set_model_phase(model, training=False)
    torch.cuda.reset_peak_memory_stats(device)
    items = benchmark.dataset_items_from_coco(args.val_ann)
    eval_args = argparse.Namespace(
        max_image_side=args.val_max_image_side,
        target_class_id=0,
        device="cuda",
        amp_dtype=args.amp_dtype,
        prompt="watermelon",
        min_area_ratio=0.0,
        max_detections=100,
        quiet=True,
    )
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device="cuda",
        confidence_threshold=0.0,
    )
    started = time.perf_counter()
    records = benchmark.infer_split(
        f"val_epoch_{epoch + 1}", items, processor, torch, eval_args
    )
    curve = benchmark.threshold_curve(records, iou_threshold=0.50)
    best_f1 = benchmark.choose_best_f1(curve)
    coco = benchmark.coco_metrics(records)
    elapsed = time.perf_counter() - started
    selection_score = (
        best_f1["f1"]
        if args.selection_metric == "f1"
        else coco[args.selection_metric]
    )
    metrics = {
        "phase": "val_coco",
        "epoch": epoch + 1,
        "images": len(items),
        "instances": sum(len(record.gt_rles) for record in records),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(items),
        "selection_metric": args.selection_metric,
        "selection_score": float(selection_score),
        "coco_mask_metrics": {
            key: value for key, value in coco.items() if key != "cocoeval_output"
        },
        "best_f1_policy": best_f1,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    print(
        f"val_coco epoch={epoch + 1}/{args.epochs} "
        f"mAP50-95={coco['mask_map_50_95']:.5f} AP50={coco['mask_ap50']:.5f} "
        f"F1={best_f1['f1']:.5f} selected={selection_score:.5f} "
        f"seconds={elapsed:.1f}",
        flush=True,
    )
    del records, processor
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def run_val_subprocess(args: argparse.Namespace) -> int:
    if not args.adapter_path or not args.val_output_json:
        print("Error: --adapter-path and --val-output-json are required for val-only-subprocess", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM 3 validation")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    print(f"Subprocess: loading SAM 3 base checkpoint {args.checkpoint}...", flush=True)
    model, _ = build_base_model_memory_efficient(args.checkpoint, use_mmap=False)
    model._sam3_freeze_policy = args.freeze_policy
    apply_freeze_policy(model, args.freeze_policy)
    
    print(f"Subprocess: restoring adapter state from {args.adapter_path}...", flush=True)
    payload = torch.load(args.adapter_path, map_location="cpu", weights_only=False)
    load_adapter_state(model, payload["adapter"])
    
    model.to(device)
    
    epoch = args.val_epoch if args.val_epoch is not None else 0
    print(f"Subprocess: starting COCO validation for epoch {epoch + 1}...", flush=True)
    
    metrics = validate_coco_epoch(model, args, device, epoch)
    
    json_dump(args.val_output_json, metrics)
    print("Subprocess: validation completed successfully.", flush=True)
    return 0


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if is_trainable_name(name, model._sam3_freeze_policy)  # type: ignore[attr-defined]
    }


def load_adapter_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {
        name
        for name in model.state_dict()
        if is_trainable_name(name, model._sam3_freeze_policy)  # type: ignore[attr-defined]
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(f"Adapter key mismatch: missing={missing[:5]}, extra={extra[:5]}")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {incompatible.unexpected_keys}")


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_selection_score: float,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "format": "sam3-watermelon-adapter-v1",
        "adapter": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_selection_score": best_selection_score,
        "metadata": metadata,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_adapter(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    selection_metric: str,
    selection_score: float,
    val_metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "sam3-watermelon-adapter-v1",
            "adapter": trainable_state_dict(model),
            "epoch": epoch,
            "global_step": global_step,
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            "val_metrics": val_metrics,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def restore_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    expected_metadata: dict[str, Any],
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "sam3-watermelon-adapter-v1":
        raise ValueError(f"Unsupported resume checkpoint: {path}")
    metadata = payload["metadata"]
    if metadata["freeze_policy"] != model._sam3_freeze_policy:  # type: ignore[attr-defined]
        raise ValueError("Resume freeze policy differs from the current run")
    # 容错换行符差异引起的签名哈希冲突
    payload_sig = dict(payload["metadata"].get("resume_signature", {}))
    exp_sig = dict(expected_metadata.get("resume_signature", {}))
    for key, path_str in [("train_ann_sha256", expected_metadata.get("train_ann")), 
                          ("val_ann_sha256", expected_metadata.get("val_ann"))]:
        if key in payload_sig and key in exp_sig and path_str:
            try:
                path = Path(path_str)
                if path.is_file():
                    content = path.read_bytes()
                    h_lf = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
                    h_crlf = hashlib.sha256(content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
                    if payload_sig[key] in (h_lf, h_crlf) and exp_sig[key] in (h_lf, h_crlf):
                        payload_sig[key] = exp_sig[key]
            except Exception:
                pass

    if payload_sig != exp_sig:
        raise ValueError("Resume checkpoint data/base-weight/hyperparameter signature differs")
    load_adapter_state(model, payload["adapter"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return (
        int(payload["epoch"]),
        int(payload["global_step"]),
        float(payload["best_selection_score"]),
    )


def export_merged_checkpoint(
    model: torch.nn.Module,
    adapter_path: Path,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    load_adapter_state(model, payload["adapter"])
    model.cpu()
    torch.cuda.empty_cache()
    state = {f"detector.{name}": tensor for name, tensor in model.state_dict().items()}
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "model": state,
            "meta": {
                **metadata,
                "selected_epoch": payload["epoch"],
                "selection_metric": payload["selection_metric"],
                "selection_score": payload["selection_score"],
                "exported_at": utc_now(),
            },
        },
        temporary,
    )
    os.replace(temporary, output_path)


def initialize_training_objects(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.amp.GradScaler, Sam3LossWrapper]:
    print("loading SAM 3 base checkpoint...", flush=True)
    model, _ = build_base_model_memory_efficient(args.checkpoint)
    model._sam3_freeze_policy = args.freeze_policy  # type: ignore[attr-defined]
    apply_freeze_policy(model, args.freeze_policy)
    install_frozen_backbone_no_grad(model)
    model.to(device)
    criterion = build_loss(args).to(device)
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return model, optimizer, scaler, criterion


def main() -> int:
    args = resolve_args(parse_args())
    if args.val_only_subprocess:
        return run_val_subprocess(args)
    prepare_output(args)
    triton_cache = args.output_dir / "triton_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    seed_everything(args.seed)
    dataset_integrity = verify_dataset_integrity(args.project_root, args.dataset_root)
    train_dataset = build_dataset(args, training=True)
    val_dataset = build_dataset(args, training=False)
    train_loader = make_loader(train_dataset, args, training=True, epoch=0)
    val_loader = make_loader(val_dataset, args, training=False, epoch=0)
    train_batch_audit = inspect_collated_batch(next(iter(train_loader)), args.resolution)
    val_batch_audit = inspect_collated_batch(next(iter(val_loader)), args.resolution)

    script_path = Path(__file__).resolve()
    sam3_source = Path(__import__("sam3").__file__).resolve().parents[1]
    base_metadata: dict[str, Any] = {
        "created_at": utc_now(),
        "command": sys.argv,
        "project_root": str(args.project_root),
        "dataset_root": str(args.dataset_root),
        "train_ann": str(args.train_ann),
        "val_ann": str(args.val_ann),
        "train_ann_sha256": sha256_file(args.train_ann),
        "val_ann_sha256": sha256_file(args.val_ann),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "train_batch_audit": train_batch_audit,
        "val_batch_audit": val_batch_audit,
        "freeze_policy": args.freeze_policy,
        "resolution": args.resolution,
        "seed": args.seed,
        "training_script": str(script_path),
        "training_script_sha256": sha256_file(script_path),
        "python_executable": sys.executable,
        "project_commit": git_commit(args.project_root),
        "dataset_integrity": dataset_integrity,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sam3_source": str(sam3_source),
        "sam3_commit": git_commit(sam3_source),
    }
    json_dump(args.output_dir / "run_config.json", {**vars(args), **base_metadata})
    print(
        f"dataset ready: train={len(train_dataset)} val={len(val_dataset)} "
        f"train_batch={train_batch_audit} val_batch={val_batch_audit}",
        flush=True,
    )
    if args.dry_run:
        json_dump(args.output_dir / "dry_run.json", {"ok": True, **base_metadata})
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM 3 fine-tuning")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    base_metadata.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "cuda_device": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
        }
    )
    base_metadata["resume_signature"] = {
        key: base_metadata[key]
        for key in (
            "checkpoint_sha256",
            "train_ann_sha256",
            "val_ann_sha256",
            "freeze_policy",
            "resolution",
            "seed",
        )
    }
    base_metadata["resume_signature"].update(
        {
            key: getattr(args, key)
            for key in (
                "epochs",
                "learning_rate",
                "weight_decay",
                "warmup_steps",
                "cooldown_steps",
                "scheduler_timescale",
                "batch_size",
                "pin_memory",
                "min_resize",
                "mask_sample_points",
                "mask_oversample_ratio",
                "mask_importance_ratio",
                "gradient_clip",
                "amp_dtype",
                "selection_metric",
                "val_max_image_side",
            )
        }
    )

    model, optimizer, scaler, criterion = initialize_training_objects(args, device)
    params = parameter_summary(model)
    base_metadata["parameters"] = params
    write_parameter_manifest(model, args.output_dir)
    print(f"parameters: {params}", flush=True)

    gc.collect()
    base_metadata["host_memory_after_cuda_move"] = host_memory_snapshot()
    json_dump(args.output_dir / "run_config.json", {**vars(args), **base_metadata})
    start_epoch = 0
    global_step = 0
    best_selection_score = -math.inf
    metrics_path = args.output_dir / "metrics.jsonl"
    best_path = args.output_dir / "best_adapter.pt"
    latest_path = args.output_dir / "latest_train_state.pt"
    last_path = args.output_dir / "last_adapter.pt"
    if args.resume is not None:
        start_epoch, global_step, best_selection_score = restore_training_state(
            args.resume, model, optimizer, scaler, base_metadata
        )
        prior_best = args.resume.parent / "best_adapter.pt"
        if prior_best.resolve() != best_path.resolve():
            if not prior_best.is_file():
                raise FileNotFoundError(
                    f"Resume state refers to a prior best score, but {prior_best} is missing"
                )
            shutil.copy2(prior_best, best_path)
        print(
            f"resumed epoch={start_epoch} global_step={global_step} "
            f"best_selection_score={best_selection_score:.6f}",
            flush=True,
        )

    steps_per_epoch = cap_steps(train_loader, args.max_train_steps)
    total_steps = args.epochs * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        train_loader = make_loader(train_dataset, args, training=True, epoch=epoch)
        train_metrics, global_step = train_epoch(
            model,
            criterion,
            optimizer,
            scaler,
            train_loader,
            args,
            device,
            epoch,
            global_step,
            total_steps,
        )
        append_jsonl(metrics_path, train_metrics)
        # 1. 验证开始前保存恢复点 (latest_train_state.pt)
        # 注意此时保存的 best_selection_score 还是上一轮的最优得分
        save_training_state(
            latest_path,
            model,
            optimizer,
            scaler,
            epoch=epoch + 1,
            global_step=global_step,
            best_selection_score=best_selection_score,
            metadata=base_metadata,
        )
        print(f"Pre-validation recovery point saved to {latest_path}", flush=True)

        # 2. 保存临时的 epoch adapter，供评估子进程加载
        epoch_path = args.output_dir / f"epoch_{epoch + 1:03d}_adapter.pt"
        save_adapter(
            epoch_path,
            model,
            epoch=epoch + 1,
            global_step=global_step,
            selection_metric="pre_val_checkpoint",
            selection_score=0.0,
            val_metrics={},
            metadata=base_metadata,
        )

        # 3. 运行验证
        if args.max_val_steps:
            val_loader = make_loader(val_dataset, args, training=False, epoch=epoch)
            val_metrics = validate_epoch(
                model, criterion, val_loader, args, device, epoch
            )
            selection_metric = "negative_proxy_core_loss"
            selection_score = -float(val_metrics["losses"][CORE_LOSS_KEY])
            val_metrics["selection_metric"] = selection_metric
            val_metrics["selection_score"] = selection_score
        else:
            val_output_json = args.output_dir / f"epoch_{epoch + 1:03d}_val_metrics.json"
            if val_output_json.is_file():
                try:
                    val_output_json.unlink()
                except OSError:
                    pass
            
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--val-only-subprocess",
                "--val-epoch", str(epoch),
                "--adapter-path", str(epoch_path),
                "--val-output-json", str(val_output_json),
                "--checkpoint", str(args.checkpoint),
                "--dataset-root", str(args.dataset_root),
                "--train-ann", str(args.train_ann),
                "--val-ann", str(args.val_ann),
                "--freeze-policy", args.freeze_policy,
                "--resolution", str(args.resolution),
                "--seed", str(args.seed),
                "--amp-dtype", args.amp_dtype,
                "--selection-metric", args.selection_metric,
                "--val-max-image-side", str(args.val_max_image_side),
            ]
            
            print(f"Launching COCO validation subprocess: {' '.join(cmd)}", flush=True)
            # Delete model and optimizer to free both GPU and CPU memory
            del model, optimizer, scaler, criterion
            gc.collect()
            torch.cuda.empty_cache()

            val_log_path = args.output_dir / f"epoch_{epoch + 1:03d}_val_subprocess.log"
            try:
                with val_log_path.open("w", encoding="utf-8") as f_log:
                    res = subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, check=True, timeout=1200)
                if val_output_json.is_file():
                    val_metrics = json.loads(val_output_json.read_text(encoding="utf-8"))
                    try:
                        val_output_json.unlink()
                    except OSError:
                        pass
                else:
                    raise FileNotFoundError(f"Val output json not found: {val_output_json}")
            except Exception as e:
                print(f"WARNING: COCO validation subprocess failed/crashed for epoch {epoch + 1}: {e}", flush=True)
                val_metrics = {
                    "phase": "val_coco",
                    "epoch": epoch + 1,
                    "images": 0,
                    "instances": 0,
                    "elapsed_seconds": 0.0,
                    "seconds_per_image": 0.0,
                    "selection_metric": args.selection_metric,
                    "selection_score": -999.0,
                    "coco_mask_metrics": {
                        "mask_map_50_95": 0.0,
                        "mask_ap50": 0.0,
                        "mask_ap75": 0.0,
                        "mask_ar100": 0.0,
                    },
                    "best_f1_policy": {
                        "f1": 0.0,
                        "threshold": 0.5,
                        "precision": 0.0,
                        "recall": 0.0,
                    },
                    "subprocess_crashed": True,
                    "crash_error": str(e)
                }
            finally:
                # Re-initialize training objects and restore state
                model, optimizer, scaler, criterion = initialize_training_objects(args, device)
                restore_training_state(
                    latest_path,
                    model,
                    optimizer,
                    scaler,
                    base_metadata,
                )
                gc.collect()
                torch.cuda.empty_cache()
            
            selection_metric = str(val_metrics["selection_metric"])
            selection_score = float(val_metrics["selection_score"])
        
        append_jsonl(metrics_path, val_metrics)
        
        # 4. 更新带有最终评测结果的 epoch_path adapter
        save_adapter(
            epoch_path,
            model,
            epoch=epoch + 1,
            global_step=global_step,
            selection_metric=selection_metric,
            selection_score=selection_score,
            val_metrics=val_metrics,
            metadata=base_metadata,
        )
        
        # 5. 更新最佳及最新参数
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            save_adapter(
                best_path,
                model,
                epoch=epoch + 1,
                global_step=global_step,
                selection_metric=selection_metric,
                selection_score=selection_score,
                val_metrics=val_metrics,
                metadata=base_metadata,
            )
            print(
                f"new best adapter: epoch={epoch + 1} "
                f"{selection_metric}={selection_score:.6f}",
                flush=True,
            )
        save_adapter(
            last_path,
            model,
            epoch=epoch + 1,
            global_step=global_step,
            selection_metric=selection_metric,
            selection_score=selection_score,
            val_metrics=val_metrics,
            metadata=base_metadata,
        )
        save_training_state(
            latest_path,
            model,
            optimizer,
            scaler,
            epoch=epoch + 1,
            global_step=global_step,
            best_selection_score=best_selection_score,
            metadata=base_metadata,
        )

    result = {
        "completed_at": utc_now(),
        "epochs": args.epochs,
        "global_steps": global_step,
        "selection_metric": args.selection_metric if not args.max_val_steps else "negative_proxy_core_loss",
        "best_selection_score": best_selection_score,
        "best_adapter": str(best_path),
        "last_adapter": str(last_path),
        "latest_train_state": str(latest_path),
    }
    if not args.no_merged_checkpoint:
        merged_path = args.output_dir / "best_merged.pt"
        print("exporting merged deployment checkpoint...", flush=True)
        export_merged_checkpoint(model, best_path, merged_path, base_metadata)
        result["best_merged"] = str(merged_path)
        result["best_merged_sha256"] = sha256_file(merged_path)
    json_dump(args.output_dir / "training_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
