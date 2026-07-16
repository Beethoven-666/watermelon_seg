#!/usr/bin/env python3
"""Build an auditable, leak-reduced COCO dataset for SAM 3 fine-tuning.

The source YOLO dataset is never modified or copied.  COCO ``file_name`` values
are relative to the project root, so SAM 3 should use the project root as its
``img_folder``.

Split policy:

1. Freeze the existing test split byte-for-byte.
2. Remove val rows whose provenance scene key or image SHA-256 occurs in test.
3. Remove train rows whose scene key or image SHA-256 occurs in test or in the
   cleaned val split.
4. Deduplicate byte-identical images inside the remaining train split, but only
   after proving that their rasterized instance masks are semantically equal.
5. Detect cross-split perceptual duplicates with deterministic grayscale D4
   matching; preserve test, then val, and exclude lower-priority train rows.

Any malformed annotation, conflicting duplicate, frozen-test drift, or residual
cross-split scene/hash intersection aborts the build before publication.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from skimage.metrics import structural_similarity


DATASET_VERSION = "sam3_watermelon_finetune_v2"
TARGET_CLASS_ID = 0
COCO_CATEGORY_ID = 1
COCO_CATEGORY_NAME = "watermelon"
EXPECTED_FROZEN_TEST_FINGERPRINT = (
    "c5fd1505c69860ca359ee3f9affc60e6a9dab08734fc888b4ab05014561b4f9b"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COORD_TOLERANCE = 1e-6
DUPLICATE_MASK_IOU_THRESHOLD = 0.99
PERCEPTUAL_COARSE_SIZE = 16
PERCEPTUAL_CONFIRM_SIZE = 128
PERCEPTUAL_COARSE_CORR_THRESHOLD = 0.995
PERCEPTUAL_CONFIRM_CORR_THRESHOLD = 0.995
PERCEPTUAL_SSIM_THRESHOLD = 0.96
PERCEPTUAL_SPLIT_PRIORITY = {"train_clean": 0, "val_clean": 1, "test_frozen": 2}
D4_NAMES = (
    "rot0",
    "flip_lr_rot0",
    "rot90",
    "flip_lr_rot90",
    "rot180",
    "flip_lr_rot180",
    "rot270",
    "flip_lr_rot270",
)


@dataclass(frozen=True)
class Instance:
    line_number: int
    normalized_polygon: tuple[float, ...]
    pixel_polygon: tuple[float, ...]
    bbox_xywh: tuple[float, float, float, float]
    area: float
    vertices: int
    mask_sha256: str
    polygon_sha256: str


@dataclass(frozen=True)
class SourceImage:
    source_split: str
    source: str
    upstream_split: str
    image_name: str
    label_name: str
    source_image: str
    declared_instances: int
    image_path: Path
    label_path: Path
    relative_image_path: str
    relative_label_path: str
    width: int
    height: int
    scene_key: str
    canonical_asset: str
    image_sha256: str
    label_sha256: str
    instances: tuple[Instance, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.source_split, self.image_name

    @property
    def semantic_mask_signature(self) -> tuple[str, ...]:
        return tuple(sorted(instance.mask_sha256 for instance in self.instances))


@dataclass(frozen=True)
class PerceptualPair:
    left_split: str
    left_key: tuple[str, str]
    right_split: str
    right_key: tuple[str, str]
    d4_index: int
    coarse_corr: float
    confirm_corr: float
    ssim: float


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("exports/fused_watermelon_yolo_seg/split_manifest.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"exports/{DATASET_VERSION}"),
    )
    parser.add_argument(
        "--expected-test-fingerprint",
        default=EXPECTED_FROZEN_TEST_FINGERPRINT,
        help="Expected image+label fingerprint of the frozen source test split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing output directory after a successful build.",
    )
    return parser.parse_args()


def resolve_under(root: Path, path: Path, description: str) -> Path:
    root = root.resolve()
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} must stay inside project root: {resolved}") from error
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_scene(source: str, image_name: str) -> tuple[str, str]:
    stem = Path(image_name).stem
    canonical_asset = stem.split(".rf.", 1)[0] if ".rf." in stem else stem
    normalized_source = unicodedata.normalize("NFC", source).casefold()
    normalized_asset = unicodedata.normalize("NFC", canonical_asset).casefold()
    return f"{normalized_source}::{normalized_asset}", canonical_asset


def mask_signature(rle: dict[str, Any]) -> str:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts_bytes = counts.encode("ascii")
    elif isinstance(counts, bytes):
        counts_bytes = counts
    else:
        counts_bytes = json.dumps(counts, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256()
    digest.update(f"{rle['size'][0]}x{rle['size'][1]}\0".encode("ascii"))
    digest.update(counts_bytes)
    return digest.hexdigest()


def parse_yolo_label(label_path: Path, width: int, height: int) -> tuple[Instance, ...]:
    instances: list[Instance] = []
    raw_text = label_path.read_text(encoding="utf-8-sig")
    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            raise ValueError(
                f"{label_path}:{line_number}: expected class_id plus at least 3 xy pairs"
            )
        try:
            class_value = float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: non-numeric token") from error
        if not math.isfinite(class_value) or int(class_value) != class_value:
            raise ValueError(f"{label_path}:{line_number}: invalid class id {parts[0]!r}")
        if int(class_value) != TARGET_CLASS_ID:
            raise ValueError(
                f"{label_path}:{line_number}: class {int(class_value)} is not target class 0"
            )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: non-finite polygon coordinate")
        if any(value < -COORD_TOLERANCE or value > 1 + COORD_TOLERANCE for value in coordinates):
            raise ValueError(f"{label_path}:{line_number}: normalized coordinate outside [0, 1]")

        normalized = tuple(min(1.0, max(0.0, value)) for value in coordinates)
        points = list(zip(normalized[0::2], normalized[1::2]))
        if len(set(points)) < 3:
            raise ValueError(f"{label_path}:{line_number}: fewer than 3 distinct vertices")

        pixel: list[float] = []
        for index, value in enumerate(normalized):
            scale = width if index % 2 == 0 else height
            pixel.append(round(value * scale, 6))
        rles = mask_utils.frPyObjects([pixel], height, width)
        rle = mask_utils.merge(rles)
        area = float(mask_utils.area(rle))
        if not math.isfinite(area) or area <= 0:
            raise ValueError(f"{label_path}:{line_number}: polygon rasterizes to zero area")
        bbox_raw = np.asarray(mask_utils.toBbox(rle), dtype=np.float64).reshape(4)
        bbox = tuple(round(float(value), 6) for value in bbox_raw.tolist())
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"{label_path}:{line_number}: invalid pycocotools bbox {bbox}")

        polygon_json = json.dumps(pixel, separators=(",", ":"), ensure_ascii=True)
        instances.append(
            Instance(
                line_number=line_number,
                normalized_polygon=normalized,
                pixel_polygon=tuple(pixel),
                bbox_xywh=bbox,
                area=round(area, 6),
                vertices=len(points),
                mask_sha256=mask_signature(rle),
                polygon_sha256=text_sha256(polygon_json),
            )
        )
    return tuple(instances)


def read_source_rows(project_root: Path, manifest_path: Path) -> list[SourceImage]:
    required = {
        "split",
        "source",
        "source_split",
        "image",
        "label",
        "instances",
        "source_image",
    }
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"Split manifest is missing columns: {missing}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"Split manifest is empty: {manifest_path}")

    seen: set[tuple[str, str]] = set()
    rows: list[SourceImage] = []
    for raw in raw_rows:
        split = raw["split"].strip()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r} in manifest")
        image_name = raw["image"].strip()
        label_name = raw["label"].strip()
        key = split, image_name
        if key in seen:
            raise ValueError(f"Duplicate manifest image row: {split}/{image_name}")
        seen.add(key)
        if Path(image_name).name != image_name or Path(label_name).name != label_name:
            raise ValueError(f"Manifest image/label must be basenames: {image_name}, {label_name}")
        if Path(image_name).stem != Path(label_name).stem:
            raise ValueError(f"Image/label stem mismatch: {image_name}, {label_name}")
        try:
            declared_instances = int(raw["instances"])
        except ValueError as error:
            raise ValueError(f"Invalid instance count for {split}/{image_name}") from error

        image_path = (project_root / "images" / split / image_name).resolve()
        label_path = (project_root / "labels" / split / label_name).resolve()
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing source pair: {image_path}, {label_path}")
        with Image.open(image_path) as opened:
            width, height = opened.size
            opened.verify()
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_path}")
        instances = parse_yolo_label(label_path, width, height)
        if len(instances) != declared_instances:
            raise ValueError(
                f"Manifest/label instance mismatch for {split}/{image_name}: "
                f"{declared_instances} != {len(instances)}"
            )
        scene_key, canonical_asset = canonical_scene(raw["source"].strip(), image_name)
        rows.append(
            SourceImage(
                source_split=split,
                source=raw["source"].strip(),
                upstream_split=raw["source_split"].strip(),
                image_name=image_name,
                label_name=label_name,
                source_image=raw["source_image"].strip(),
                declared_instances=declared_instances,
                image_path=image_path,
                label_path=label_path,
                relative_image_path=(Path("images") / split / image_name).as_posix(),
                relative_label_path=(Path("labels") / split / label_name).as_posix(),
                width=width,
                height=height,
                scene_key=scene_key,
                canonical_asset=canonical_asset,
                image_sha256=file_sha256(image_path),
                label_sha256=file_sha256(label_path),
                instances=instances,
            )
        )

    rows.sort(key=lambda row: (row.source_split, row.source.casefold(), row.image_name.casefold(), row.image_name))
    validate_manifest_covers_source_dirs(project_root, rows)
    return rows


def validate_manifest_covers_source_dirs(project_root: Path, rows: Sequence[SourceImage]) -> None:
    for split in ("train", "val", "test"):
        expected_images = {row.image_name for row in rows if row.source_split == split}
        expected_labels = {row.label_name for row in rows if row.source_split == split}
        actual_images = {
            path.name
            for path in (project_root / "images" / split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        actual_labels = {path.name for path in (project_root / "labels" / split).glob("*.txt")}
        if expected_images != actual_images:
            raise ValueError(
                f"Manifest does not exactly cover images/{split}; "
                f"missing={sorted(actual_images - expected_images)[:5]}, "
                f"extra={sorted(expected_images - actual_images)[:5]}"
            )
        if expected_labels != actual_labels:
            raise ValueError(
                f"Manifest does not exactly cover labels/{split}; "
                f"missing={sorted(actual_labels - expected_labels)[:5]}, "
                f"extra={sorted(expected_labels - actual_labels)[:5]}"
            )


def dataset_fingerprint(project_root: Path, split: str) -> str:
    images_dir = project_root / "images" / split
    labels_dir = project_root / "labels" / split
    images = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    digest = hashlib.sha256()
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for fingerprint: {label_path}")
        for role, path in (("image", image_path), ("label", label_path)):
            digest.update(role.encode("ascii"))
            digest.update(b"\0")
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha256(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def instance_rles(row: SourceImage) -> list[dict[str, Any]]:
    return [
        mask_utils.merge(
            mask_utils.frPyObjects([list(instance.pixel_polygon)], row.height, row.width)
        )
        for instance in row.instances
    ]


def perfect_mask_matching_ious(
    left: SourceImage, right: SourceImage, threshold: float
) -> list[float] | None:
    if len(left.instances) != len(right.instances):
        return None
    if not left.instances:
        return []
    left_rles = instance_rles(left)
    right_rles = instance_rles(right)
    ious = np.asarray(
        mask_utils.iou(left_rles, right_rles, [0] * len(right_rles)), dtype=np.float64
    )
    right_match = [-1] * len(right.instances)

    def augment(left_index: int, seen_right: set[int]) -> bool:
        candidates = np.flatnonzero(ious[left_index] >= threshold)
        if candidates.size > 1:
            candidates = candidates[
                np.argsort(-ious[left_index, candidates], kind="stable")
            ]
        for right_index_raw in candidates.tolist():
            right_index = int(right_index_raw)
            if right_index in seen_right:
                continue
            seen_right.add(right_index)
            if right_match[right_index] < 0 or augment(
                right_match[right_index], seen_right
            ):
                right_match[right_index] = left_index
                return True
        return False

    if not all(augment(index, set()) for index in range(len(left.instances))):
        return None
    return [float(ious[left_index, right_index]) for right_index, left_index in enumerate(right_match)]


def validate_exact_duplicate_semantics(rows: Sequence[SourceImage]) -> dict[str, Any]:
    groups: dict[str, list[SourceImage]] = defaultdict(list)
    for row in rows:
        groups[row.image_sha256].append(row)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    duplicate_rows = 0
    exact_raster_groups = 0
    equivalent_raster_groups = 0
    minimum_matched_iou = 1.0
    for group in duplicate_groups:
        duplicate_rows += len(group)
        dimensions = {(row.width, row.height) for row in group}
        if len(dimensions) != 1:
            names = [f"{row.source_split}/{row.image_name}" for row in group]
            raise ValueError(f"Dimension conflict for byte-identical image: {names}")
        signatures = {row.semantic_mask_signature for row in group}
        if len(signatures) == 1:
            exact_raster_groups += 1
            continue
        reference = group[0]
        group_minimum = 1.0
        for candidate in group[1:]:
            matched_ious = perfect_mask_matching_ious(
                reference, candidate, DUPLICATE_MASK_IOU_THRESHOLD
            )
            if matched_ious is None:
                names = [f"{row.source_split}/{row.image_name}" for row in group]
                raise ValueError(
                    "Conflicting labels for byte-identical image at mask IoU "
                    f"{DUPLICATE_MASK_IOU_THRESHOLD}: {names}"
                )
            if matched_ious:
                group_minimum = min(group_minimum, min(matched_ious))
        equivalent_raster_groups += 1
        minimum_matched_iou = min(minimum_matched_iou, group_minimum)
    return {
        "groups": len(duplicate_groups),
        "rows": duplicate_rows,
        "exact_raster_groups": exact_raster_groups,
        "equivalent_raster_groups_at_iou_threshold": equivalent_raster_groups,
        "mask_iou_threshold": DUPLICATE_MASK_IOU_THRESHOLD,
        "minimum_matched_iou": minimum_matched_iou,
    }


def conflict_reason(prefix: str, scene_conflict: bool, sha_conflict: bool) -> str:
    if scene_conflict and sha_conflict:
        return f"{prefix}_scene_and_sha256"
    if scene_conflict:
        return f"{prefix}_scene"
    if sha_conflict:
        return f"{prefix}_sha256"
    raise AssertionError("conflict_reason called without a conflict")


def exclusion_row(
    row: SourceImage,
    scope: str,
    reason: str,
    conflicts_with: str = "",
    representative: str = "",
) -> dict[str, Any]:
    return {
        "scope": scope,
        "source_split": row.source_split,
        "source": row.source,
        "source_image": row.source_image,
        "image": row.image_name,
        "label": row.label_name,
        "instances": len(row.instances),
        "scene_key": row.scene_key,
        "image_sha256": row.image_sha256,
        "reason": reason,
        "conflicts_with": conflicts_with,
        "representative": representative,
    }


def select_splits(
    rows: Sequence[SourceImage],
) -> tuple[dict[str, list[SourceImage]], list[dict[str, Any]], list[SourceImage]]:
    by_split = {split: [row for row in rows if row.source_split == split] for split in ("train", "val", "test")}
    test = sorted(by_split["test"], key=lambda row: (row.image_name.casefold(), row.image_name))
    test_scenes = {row.scene_key for row in test}
    test_hashes = {row.image_sha256 for row in test}
    exclusions: list[dict[str, Any]] = []

    val: list[SourceImage] = []
    for row in sorted(by_split["val"], key=lambda item: (item.image_name.casefold(), item.image_name)):
        scene_conflict = row.scene_key in test_scenes
        sha_conflict = row.image_sha256 in test_hashes
        if scene_conflict or sha_conflict:
            exclusions.append(
                exclusion_row(
                    row,
                    "base_dataset",
                    conflict_reason("protected_test", scene_conflict, sha_conflict),
                    "test",
                )
            )
        else:
            val.append(row)

    val_scenes = {row.scene_key for row in val}
    val_hashes = {row.image_sha256 for row in val}
    train_candidates: list[SourceImage] = []
    for row in sorted(by_split["train"], key=lambda item: (item.image_name.casefold(), item.image_name)):
        test_scene = row.scene_key in test_scenes
        test_sha = row.image_sha256 in test_hashes
        if test_scene or test_sha:
            exclusions.append(
                exclusion_row(
                    row,
                    "base_dataset",
                    conflict_reason("protected_test", test_scene, test_sha),
                    "test",
                )
            )
            continue
        val_scene = row.scene_key in val_scenes
        val_sha = row.image_sha256 in val_hashes
        if val_scene or val_sha:
            exclusions.append(
                exclusion_row(
                    row,
                    "base_dataset",
                    conflict_reason("clean_val", val_scene, val_sha),
                    "val_clean",
                )
            )
            continue
        train_candidates.append(row)

    train: list[SourceImage] = []
    by_hash: dict[str, list[SourceImage]] = defaultdict(list)
    for row in train_candidates:
        by_hash[row.image_sha256].append(row)
    for image_hash in sorted(by_hash):
        group = sorted(by_hash[image_hash], key=lambda item: (item.image_name.casefold(), item.image_name))
        representative = group[0]
        train.append(representative)
        for duplicate in group[1:]:
            exclusions.append(
                exclusion_row(
                    duplicate,
                    "base_dataset",
                    "exact_duplicate_within_train",
                    "train_clean",
                    representative.image_name,
                )
            )

    train.sort(key=lambda row: (row.image_name.casefold(), row.image_name))
    val.sort(key=lambda row: (row.image_name.casefold(), row.image_name))

    test_unique: list[SourceImage] = []
    test_by_scene: dict[str, list[SourceImage]] = defaultdict(list)
    for row in test:
        test_by_scene[row.scene_key].append(row)
    for scene_key in sorted(test_by_scene):
        group = sorted(test_by_scene[scene_key], key=lambda item: (item.image_name.casefold(), item.image_name))
        representative = group[0]
        test_unique.append(representative)
        for duplicate in group[1:]:
            exclusions.append(
                exclusion_row(
                    duplicate,
                    "test_scene_unique_view",
                    "scene_duplicate_within_test",
                    "test_frozen",
                    representative.image_name,
                )
            )
    test_unique.sort(key=lambda row: (row.image_name.casefold(), row.image_name))
    exclusions.sort(key=lambda item: (item["scope"], item["source_split"], item["image"], item["reason"]))
    return {"train_clean": train, "val_clean": val, "test_frozen": test}, exclusions, test_unique


def pairwise_intersections(selected: dict[str, list[SourceImage]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = ("train_clean", "val_clean", "test_frozen")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            scene_overlap = sorted(
                {row.scene_key for row in selected[left]} & {row.scene_key for row in selected[right]}
            )
            hash_overlap = sorted(
                {row.image_sha256 for row in selected[left]}
                & {row.image_sha256 for row in selected[right]}
            )
            key = f"{left}__{right}"
            result[key] = {
                "scene_key_count": len(scene_overlap),
                "image_sha256_count": len(hash_overlap),
                "scene_keys": scene_overlap,
                "image_sha256": hash_overlap,
            }
            if scene_overlap or hash_overlap:
                raise ValueError(f"Residual cross-split leak in {key}: {result[key]}")
    return result


def d4_variants(array: np.ndarray) -> tuple[np.ndarray, ...]:
    variants: list[np.ndarray] = []
    for rotations in range(4):
        rotated = np.rot90(array, rotations)
        variants.append(np.ascontiguousarray(rotated))
        variants.append(np.ascontiguousarray(np.fliplr(rotated)))
    return tuple(variants)


def zero_mean_l2(array: np.ndarray) -> np.ndarray:
    flattened = np.asarray(array, dtype=np.float64).reshape(-1)
    flattened = flattened - flattened.mean()
    norm = float(np.linalg.norm(flattened))
    if norm <= np.finfo(np.float64).eps:
        return np.zeros_like(flattened)
    return flattened / norm


def grayscale_d4(row: SourceImage, size: int) -> tuple[np.ndarray, ...]:
    with Image.open(row.image_path) as opened:
        resized = opened.convert("L").resize(
            (size, size), resample=Image.Resampling.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float64)
    return d4_variants(array)


def find_perceptual_pairs(
    selected: dict[str, list[SourceImage]],
) -> tuple[list[PerceptualPair], dict[str, Any]]:
    split_pairs = (
        ("train_clean", "val_clean"),
        ("train_clean", "test_frozen"),
        ("val_clean", "test_frozen"),
    )
    coarse_features: dict[tuple[str, str], np.ndarray] = {}

    def coarse(row: SourceImage) -> np.ndarray:
        if row.key not in coarse_features:
            variants = grayscale_d4(row, PERCEPTUAL_COARSE_SIZE)
            coarse_features[row.key] = np.stack(
                [zero_mean_l2(variant) for variant in variants], axis=0
            )
        return coarse_features[row.key]

    scanned_pairs = 0
    coarse_candidates: list[
        tuple[str, SourceImage, str, SourceImage, int, float]
    ] = []
    for left_split, right_split in split_pairs:
        left_rows = selected[left_split]
        right_rows = selected[right_split]
        scanned_pairs += len(left_rows) * len(right_rows)
        if not left_rows or not right_rows:
            continue
        left_matrix = np.stack([coarse(row)[0] for row in left_rows], axis=0)
        right_matrix = np.stack([coarse(row) for row in right_rows], axis=0)
        scores = np.einsum("id,jkd->ijk", left_matrix, right_matrix, optimize=True)
        best_d4 = np.argmax(scores, axis=2)
        best_scores = np.take_along_axis(scores, best_d4[:, :, None], axis=2)[:, :, 0]
        for left_index, right_index in np.argwhere(
            best_scores >= PERCEPTUAL_COARSE_CORR_THRESHOLD
        ).tolist():
            coarse_candidates.append(
                (
                    left_split,
                    left_rows[left_index],
                    right_split,
                    right_rows[right_index],
                    int(best_d4[left_index, right_index]),
                    float(best_scores[left_index, right_index]),
                )
            )

    confirm_raw: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}

    def confirm(row: SourceImage) -> tuple[np.ndarray, ...]:
        if row.key not in confirm_raw:
            confirm_raw[row.key] = grayscale_d4(row, PERCEPTUAL_CONFIRM_SIZE)
        return confirm_raw[row.key]

    confirmed: list[PerceptualPair] = []
    for left_split, left, right_split, right, d4_index, coarse_corr in sorted(
        coarse_candidates,
        key=lambda item: (
            item[0],
            item[1].relative_image_path,
            item[2],
            item[3].relative_image_path,
        ),
    ):
        left_array = confirm(left)[0]
        right_array = confirm(right)[d4_index]
        confirm_corr = float(zero_mean_l2(left_array) @ zero_mean_l2(right_array))
        ssim = float(
            structural_similarity(left_array, right_array, data_range=255.0)
        )
        if (
            confirm_corr >= PERCEPTUAL_CONFIRM_CORR_THRESHOLD
            and ssim >= PERCEPTUAL_SSIM_THRESHOLD
        ):
            confirmed.append(
                PerceptualPair(
                    left_split=left_split,
                    left_key=left.key,
                    right_split=right_split,
                    right_key=right.key,
                    d4_index=d4_index,
                    coarse_corr=coarse_corr,
                    confirm_corr=confirm_corr,
                    ssim=ssim,
                )
            )

    by_split_pair = Counter(
        f"{pair.left_split}__{pair.right_split}" for pair in confirmed
    )
    audit = {
        "rule": {
            "color": "Pillow grayscale L",
            "resize": "Pillow Bilinear",
            "normalization": "flatten, zero mean, L2 norm",
            "d4_order": list(D4_NAMES),
            "coarse_size": PERCEPTUAL_COARSE_SIZE,
            "coarse_corr_threshold": PERCEPTUAL_COARSE_CORR_THRESHOLD,
            "confirm_size": PERCEPTUAL_CONFIRM_SIZE,
            "confirm_same_d4": True,
            "confirm_corr_threshold": PERCEPTUAL_CONFIRM_CORR_THRESHOLD,
            "ssim": "skimage.metrics.structural_similarity, data_range=255",
            "ssim_threshold": PERCEPTUAL_SSIM_THRESHOLD,
            "priority": ["test_frozen", "val_clean", "train_clean"],
        },
        "scanned_cross_split_pairs": scanned_pairs,
        "coarse_candidates": len(coarse_candidates),
        "confirmed_pairs": len(confirmed),
        "confirmed_by_split_pair": dict(sorted(by_split_pair.items())),
    }
    return confirmed, audit


def apply_perceptual_priority(
    selected: dict[str, list[SourceImage]],
    exclusions: list[dict[str, Any]],
    pairs: Sequence[PerceptualPair],
    audit: dict[str, Any],
) -> tuple[
    dict[str, list[SourceImage]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], str],
    dict[str, Any],
]:
    lookup: dict[tuple[str, str], tuple[str, SourceImage]] = {
        row.key: (split_name, row)
        for split_name, rows in selected.items()
        for row in rows
    }
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(key: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    for pair in pairs:
        union(pair.left_key, pair.right_key)
    components: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(parent):
        components[find(key)].append(key)
    ordered_components = sorted(
        (sorted(component) for component in components.values()), key=lambda value: value
    )

    group_ids: dict[tuple[str, str], str] = {}
    removed: set[tuple[str, str]] = set()
    perceptual_manifest: list[dict[str, Any]] = []
    additional_exclusions: list[dict[str, Any]] = []
    for group_index, component in enumerate(ordered_components, start=1):
        group_id = f"d4_group_{group_index:04d}"
        for key in component:
            group_ids[key] = group_id
        max_priority = max(PERCEPTUAL_SPLIT_PRIORITY[lookup[key][0]] for key in component)
        winner_split = next(
            split_name
            for split_name, priority in PERCEPTUAL_SPLIT_PRIORITY.items()
            if priority == max_priority
        )
        winners = sorted(
            (lookup[key][1] for key in component if lookup[key][0] == winner_split),
            key=lambda row: row.relative_image_path,
        )
        representative = winners[0]
        losers = [key for key in component if lookup[key][0] != winner_split]

        # Excluding the complete lower-priority provenance scene prevents an
        # unmatched augmentation sibling from surviving the perceptual group.
        loser_scene_pairs = {
            (lookup[key][0], lookup[key][1].scene_key) for key in losers
        }
        group_removed: set[tuple[str, str]] = set()
        for split_name, rows in selected.items():
            for row in rows:
                if (split_name, row.scene_key) in loser_scene_pairs:
                    removed.add(row.key)
                    group_removed.add(row.key)
                    group_ids.setdefault(row.key, group_id)

        component_pairs = [
            pair
            for pair in pairs
            if pair.left_key in component and pair.right_key in component
        ]
        members_text = " | ".join(
            f"{lookup[key][0]}:{lookup[key][1].image_name}" for key in component
        )
        for pair in sorted(
            component_pairs,
            key=lambda item: (
                lookup[item.left_key][1].relative_image_path,
                lookup[item.right_key][1].relative_image_path,
            ),
        ):
            left_row = lookup[pair.left_key][1]
            right_row = lookup[pair.right_key][1]
            perceptual_manifest.append(
                {
                    "group_id": group_id,
                    "winner_split": winner_split,
                    "representative": representative.image_name,
                    "members": members_text,
                    "left_split": pair.left_split,
                    "left_source": left_row.source,
                    "left_image": left_row.image_name,
                    "left_scene_key": left_row.scene_key,
                    "right_split": pair.right_split,
                    "right_source": right_row.source,
                    "right_image": right_row.image_name,
                    "right_scene_key": right_row.scene_key,
                    "d4_index": pair.d4_index,
                    "d4_transform": D4_NAMES[pair.d4_index],
                    "coarse_corr": f"{pair.coarse_corr:.12f}",
                    "confirm_corr": f"{pair.confirm_corr:.12f}",
                    "ssim": f"{pair.ssim:.12f}",
                }
            )

        for key in sorted(group_removed):
            split_name, row = lookup[key]
            incident = [
                pair
                for pair in component_pairs
                if key in {pair.left_key, pair.right_key}
            ]
            best_pair = max(
                incident,
                key=lambda item: (item.confirm_corr, item.ssim, item.coarse_corr),
                default=None,
            )
            exclusion = exclusion_row(
                row,
                "base_dataset",
                "perceptual_d4_lower_priority",
                winner_split,
                representative.image_name,
            )
            exclusion.update(
                {
                    "perceptual_group_id": group_id,
                    "d4_transform": D4_NAMES[best_pair.d4_index] if best_pair else "",
                    "coarse_corr": f"{best_pair.coarse_corr:.12f}" if best_pair else "",
                    "confirm_corr": f"{best_pair.confirm_corr:.12f}" if best_pair else "",
                    "ssim": f"{best_pair.ssim:.12f}" if best_pair else "",
                }
            )
            additional_exclusions.append(exclusion)

    if any(lookup[key][0] == "test_frozen" for key in removed):
        raise ValueError("Perceptual priority attempted to remove a frozen test image")
    filtered = {
        split_name: [row for row in rows if row.key not in removed]
        for split_name, rows in selected.items()
    }
    retained_keys = {row.key for rows in filtered.values() for row in rows}
    residual_pairs = [
        pair
        for pair in pairs
        if pair.left_key in retained_keys and pair.right_key in retained_keys
    ]
    if residual_pairs:
        raise ValueError(f"Residual perceptual cross-split pairs after priority: {residual_pairs}")

    exclusions = sorted(
        [*exclusions, *additional_exclusions],
        key=lambda item: (
            item["scope"], item["source_split"], item["image"], item["reason"]
        ),
    )
    audit = dict(audit)
    audit.update(
        {
            "groups": len(ordered_components),
            "excluded_images": len(removed),
            "excluded_instances": sum(len(lookup[key][1].instances) for key in removed),
            "residual_confirmed_pairs": 0,
        }
    )
    return filtered, exclusions, perceptual_manifest, group_ids, audit


def license_id(source: str) -> int:
    return 1 if source.startswith("roboflow_") else 2


def build_coco(
    rows: Sequence[SourceImage], split_name: str
) -> tuple[dict[str, Any], dict[tuple[str, str], int], dict[tuple[str, str, int], int]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_ids: dict[tuple[str, str], int] = {}
    annotation_ids: dict[tuple[str, str, int], int] = {}
    annotation_id = 1
    for image_id, row in enumerate(rows, start=1):
        image_ids[row.key] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": row.relative_image_path,
                "width": row.width,
                "height": row.height,
                "license": license_id(row.source),
                "extra": {
                    "source": row.source,
                    "source_split": row.source_split,
                    "upstream_split": row.upstream_split,
                    "source_image": row.source_image,
                    "scene_key": row.scene_key,
                    "image_sha256": row.image_sha256,
                    "label_sha256": row.label_sha256,
                },
            }
        )
        for instance in row.instances:
            annotation_ids[(row.source_split, row.image_name, instance.line_number)] = annotation_id
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": COCO_CATEGORY_ID,
                    "bbox": list(instance.bbox_xywh),
                    "area": instance.area,
                    "segmentation": [list(instance.pixel_polygon)],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    coco = {
        "info": {
            "description": "Leak-reduced watermelon instance segmentation for SAM 3 fine-tuning",
            "version": DATASET_VERSION,
            "split": split_name,
            "source_format": "YOLO instance segmentation polygons",
            "prompt": COCO_CATEGORY_NAME,
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            },
            {
                "id": 2,
                "name": "Local project data; see external/SOURCES.md",
                "url": "",
            },
        ],
        "categories": [
            {"id": COCO_CATEGORY_ID, "name": COCO_CATEGORY_NAME, "supercategory": "fruit"}
        ],
        "images": images,
        "annotations": annotations,
    }
    return coco, image_ids, annotation_ids


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        # pycocotools.COCO and the official SAM 3 JSON loader open files without
        # an explicit encoding. ASCII-safe JSON therefore avoids Windows locale
        # failures when provenance paths contain Chinese characters.
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def split_statistics(rows: Sequence[SourceImage]) -> dict[str, Any]:
    by_source: dict[str, dict[str, int]] = {}
    for source in sorted({row.source for row in rows}):
        source_rows = [row for row in rows if row.source == source]
        by_source[source] = {
            "images": len(source_rows),
            "instances": sum(len(row.instances) for row in source_rows),
        }
    return {
        "images": len(rows),
        "instances": sum(len(row.instances) for row in rows),
        "background_images": sum(not row.instances for row in rows),
        "scenes": len({row.scene_key for row in rows}),
        "unique_image_sha256": len({row.image_sha256 for row in rows}),
        "sources": by_source,
    }


def original_statistics(rows: Sequence[SourceImage]) -> dict[str, Any]:
    return {
        split: split_statistics([row for row in rows if row.source_split == split])
        for split in ("train", "val", "test")
    }


def make_image_manifest(
    selected: dict[str, list[SourceImage]],
    ids_by_split: dict[str, dict[tuple[str, str], int]],
    test_unique: Sequence[SourceImage],
    test_unique_ids: dict[tuple[str, str], int],
    perceptual_group_ids: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    unique_keys = {row.key for row in test_unique}
    rows: list[dict[str, Any]] = []
    for derived_split in ("train_clean", "val_clean", "test_frozen"):
        for row in selected[derived_split]:
            rows.append(
                {
                    "derived_split": derived_split,
                    "source_split": row.source_split,
                    "source": row.source,
                    "upstream_split": row.upstream_split,
                    "source_image": row.source_image,
                    "image": row.image_name,
                    "label": row.label_name,
                    "relative_image_path": row.relative_image_path,
                    "relative_label_path": row.relative_label_path,
                    "width": row.width,
                    "height": row.height,
                    "instances": len(row.instances),
                    "scene_key": row.scene_key,
                    "canonical_asset": row.canonical_asset,
                    "image_sha256": row.image_sha256,
                    "label_sha256": row.label_sha256,
                    "perceptual_group_id": perceptual_group_ids.get(row.key, ""),
                    "coco_image_id": ids_by_split[derived_split][row.key],
                    "included_in_test_scene_unique": str(row.key in unique_keys).lower(),
                    "test_scene_unique_coco_image_id": test_unique_ids.get(row.key, ""),
                }
            )
    return rows


def make_instance_manifest(
    selected: dict[str, list[SourceImage]],
    ids_by_split: dict[str, dict[tuple[str, str], int]],
    ann_ids_by_split: dict[str, dict[tuple[str, str, int], int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for derived_split in ("train_clean", "val_clean", "test_frozen"):
        for row in selected[derived_split]:
            for instance in row.instances:
                rows.append(
                    {
                        "derived_split": derived_split,
                        "source_split": row.source_split,
                        "source": row.source,
                        "image": row.image_name,
                        "label": row.label_name,
                        "label_line": instance.line_number,
                        "coco_image_id": ids_by_split[derived_split][row.key],
                        "coco_annotation_id": ann_ids_by_split[derived_split][
                            (row.source_split, row.image_name, instance.line_number)
                        ],
                        "yolo_class_id": TARGET_CLASS_ID,
                        "coco_category_id": COCO_CATEGORY_ID,
                        "vertices": instance.vertices,
                        "bbox_x": instance.bbox_xywh[0],
                        "bbox_y": instance.bbox_xywh[1],
                        "bbox_w": instance.bbox_xywh[2],
                        "bbox_h": instance.bbox_xywh[3],
                        "area": instance.area,
                        "mask_sha256": instance.mask_sha256,
                        "polygon_sha256": instance.polygon_sha256,
                    }
                )
    return rows


def make_scene_manifest(
    all_rows: Sequence[SourceImage], selected: dict[str, list[SourceImage]]
) -> list[dict[str, Any]]:
    selected_lookup = {
        row.key: split_name for split_name, split_rows in selected.items() for row in split_rows
    }
    groups: dict[str, list[SourceImage]] = defaultdict(list)
    for row in all_rows:
        groups[row.scene_key].append(row)
    output: list[dict[str, Any]] = []
    for scene_key in sorted(groups):
        group = sorted(groups[scene_key], key=lambda row: (row.source_split, row.image_name))
        original_splits = sorted({row.source_split for row in group})
        selected_rows = [row for row in group if row.key in selected_lookup]
        output.append(
            {
                "scene_key": scene_key,
                "source": group[0].source,
                "canonical_asset": group[0].canonical_asset,
                "original_splits": ",".join(original_splits),
                "original_images": len(group),
                "original_instances": sum(len(row.instances) for row in group),
                "cross_split_original": str(len(original_splits) > 1).lower(),
                "selected_splits": ",".join(sorted({selected_lookup[row.key] for row in selected_rows})),
                "selected_images": len(selected_rows),
                "selected_instances": sum(len(row.instances) for row in selected_rows),
                "members": " | ".join(f"{row.source_split}:{row.image_name}" for row in group),
            }
        )
    return output


def validate_coco_file(path: Path, expected_images: int, expected_annotations: int) -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        coco = COCO(str(path))
    if len(coco.imgs) != expected_images or len(coco.anns) != expected_annotations:
        raise ValueError(
            f"COCO round-trip count mismatch for {path}: "
            f"{len(coco.imgs)}/{len(coco.anns)} != {expected_images}/{expected_annotations}"
        )
    if set(coco.cats) != {COCO_CATEGORY_ID} or coco.cats[COCO_CATEGORY_ID]["name"] != COCO_CATEGORY_NAME:
        raise ValueError(f"Unexpected COCO categories in {path}")


def verify_source_unchanged(rows: Sequence[SourceImage]) -> None:
    for row in rows:
        if file_sha256(row.image_path) != row.image_sha256:
            raise RuntimeError(f"Source image changed during build: {row.image_path}")
        if file_sha256(row.label_path) != row.label_sha256:
            raise RuntimeError(f"Source label changed during build: {row.label_path}")


def write_readme(
    path: Path,
    summary: dict[str, Any],
    project_root: Path,
    output_dir: Path,
) -> None:
    train = summary["derived"]["train_clean"]
    val = summary["derived"]["val_clean"]
    test = summary["derived"]["test_frozen"]
    test_unique = summary["derived"]["test_scene_unique"]
    relative_output = output_dir.relative_to(project_root).as_posix()
    text = f"""# SAM 3 西瓜微调数据集（{DATASET_VERSION}）

本目录由 `scripts/build_sam3_finetune_dataset.py` 确定性生成。图片和 YOLO 标签仍保留在项目根目录，未复制、未覆盖；COCO `file_name` 均相对 `D:/MelonDataset/watermelon_seg`。

## 拆分

| 拆分 | 图片 | 实例 | 背景图 | 场景键 |
| --- | ---: | ---: | ---: | ---: |
| train_clean | {train['images']} | {train['instances']} | {train['background_images']} | {train['scenes']} |
| val_clean | {val['images']} | {val['instances']} | {val['background_images']} | {val['scenes']} |
| test_frozen | {test['images']} | {test['instances']} | {test['background_images']} | {test['scenes']} |
| test_scene_unique（敏感性视图） | {test_unique['images']} | {test_unique['instances']} | {test_unique['background_images']} | {test_unique['scenes']} |

类别固定为 COCO `1: watermelon`，文本提示词为 `watermelon`。空标签图片被保留，SAM 3 的 `COCO_FROM_JSON(include_negatives=true)` 会把它们作为负查询。

## 防泄漏策略

1. 源 `images/test` 与 `labels/test` 完全冻结，指纹必须为 `{summary['frozen_test']['actual_fingerprint']}`。
2. scene key 为 `source + Roboflow 文件名 .rf. 前的原始资产名`；非 Roboflow 文件使用完整 stem。
3. val 中与 test scene/SHA 冲突的图片被隔离。
4. train 中与 test 或 clean val scene/SHA 冲突的图片被隔离。
5. clean train 内字节相同的图片仅在实例可一对一匹配且 mask IoU ≥ {DUPLICATE_MASK_IOU_THRESHOLD:.2f} 时去重；冲突时构建直接失败。
6. 对基础 clean 拆分全量扫描灰度 D4 近重复：16×16 Bilinear、零均值 L2、corr ≥ {PERCEPTUAL_COARSE_CORR_THRESHOLD} 粗筛，再以同一 D4 在 128×128 确认 corr ≥ {PERCEPTUAL_CONFIRM_CORR_THRESHOLD} 且 SSIM ≥ {PERCEPTUAL_SSIM_THRESHOLD}；优先保留 test，其次 val，再隔离冲突 train 的完整 provenance scene。

该策略消除了已知的同原图增强与字节重复泄漏，但不能证明采集会话独立。本地 `test_images` 是连续拍摄批次，`my_labelme` 也包含短时间连续帧；最终工业验收仍应使用新采集、独立场景的现场 test。

## 文件

- `annotations/instances_train_clean.json`
- `annotations/instances_val_clean.json`
- `annotations/instances_test_frozen.json`
- `annotations/instances_test_scene_unique.json`
- `manifests/images.tsv`：所有基础派生拆分图片及 SHA-256、场景键、COCO ID。
- `manifests/instances.tsv`：逐实例 bbox/area、源标签行和 mask/polygon SHA-256。
- `manifests/scene_groups.tsv`：原始场景组及派生去向。
- `manifests/perceptual_groups.tsv`：逐对 D4 coarse/confirm corr、SSIM、变换和保留优先级。
- `manifests/exclusions.tsv`：所有隔离或去重记录。
- `summary.json`：数量、来源和生成文件摘要。
- `integrity.json`：冻结 test、重复语义和交叉拆分检查。

## SAM 3 数据路径

Hydra 数据配置应使用：

```yaml
img_folder: D:/MelonDataset/watermelon_seg
ann_file: D:/MelonDataset/watermelon_seg/{relative_output}/annotations/instances_train_clean.json
coco_json_loader:
  _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
  include_negatives: true
  category_chunk_size: 1
  _partial_: true
```

实例分割训练还必须同时启用 `load_segmentation`、`with_seg_masks`、模型 `enable_segmentation`、`DecodeRle` 和 `sam3.train.loss.loss_fns.Masks`。`test_frozen` 不得进入训练、选 epoch、阈值选择或超参数调优。
"""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def publish_directory(temp_dir: Path, output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        temp_dir.replace(output_dir)
        return
    if not force:
        raise FileExistsError(f"Output directory already exists; use --force: {output_dir}")
    backup = output_dir.parent / f".{output_dir.name}.backup-{os.getpid()}"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite temporary backup: {backup}")
    output_dir.replace(backup)
    try:
        temp_dir.replace(output_dir)
    except Exception:
        backup.replace(output_dir)
        raise
    shutil.rmtree(backup)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    manifest_path = resolve_under(project_root, args.split_manifest, "split manifest")
    output_dir = resolve_under(project_root, args.output_dir, "output directory")
    if output_dir == project_root:
        raise ValueError("Output directory cannot be the project root")
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Output directory already exists; use --force: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading and validating source dataset: {manifest_path}", flush=True)
    source_rows = read_source_rows(project_root, manifest_path)
    duplicate_semantics = validate_exact_duplicate_semantics(source_rows)
    actual_test_fingerprint = dataset_fingerprint(project_root, "test")
    if actual_test_fingerprint != args.expected_test_fingerprint:
        raise ValueError(
            "Frozen test fingerprint mismatch: "
            f"expected {args.expected_test_fingerprint}, got {actual_test_fingerprint}"
        )

    selected, exclusions, test_unique = select_splits(source_rows)
    base_intersections = pairwise_intersections(selected)
    print("Scanning deterministic grayscale D4 perceptual duplicates...", flush=True)
    perceptual_pairs, perceptual_audit = find_perceptual_pairs(selected)
    (
        selected,
        exclusions,
        perceptual_manifest,
        perceptual_group_ids,
        perceptual_audit,
    ) = apply_perceptual_priority(
        selected, exclusions, perceptual_pairs, perceptual_audit
    )
    intersections = pairwise_intersections(selected)
    print(
        "Selected: "
        + ", ".join(
            f"{name}={len(rows)} images/{sum(len(row.instances) for row in rows)} instances"
            for name, rows in selected.items()
        ),
        flush=True,
    )

    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temp_dir.exists():
        raise FileExistsError(f"Temporary output already exists: {temp_dir}")
    temp_dir.mkdir(parents=True)
    try:
        annotation_specs = {
            "train_clean": (selected["train_clean"], "instances_train_clean.json"),
            "val_clean": (selected["val_clean"], "instances_val_clean.json"),
            "test_frozen": (selected["test_frozen"], "instances_test_frozen.json"),
            "test_scene_unique": (test_unique, "instances_test_scene_unique.json"),
        }
        ids_by_split: dict[str, dict[tuple[str, str], int]] = {}
        ann_ids_by_split: dict[str, dict[tuple[str, str, int], int]] = {}
        coco_paths: dict[str, Path] = {}
        for split_name, (split_rows, file_name) in annotation_specs.items():
            coco, image_ids, annotation_ids = build_coco(split_rows, split_name)
            path = temp_dir / "annotations" / file_name
            write_json(path, coco)
            validate_coco_file(
                path,
                expected_images=len(split_rows),
                expected_annotations=sum(len(row.instances) for row in split_rows),
            )
            ids_by_split[split_name] = image_ids
            ann_ids_by_split[split_name] = annotation_ids
            coco_paths[split_name] = path

        image_manifest = make_image_manifest(
            selected,
            ids_by_split,
            test_unique,
            ids_by_split["test_scene_unique"],
            perceptual_group_ids,
        )
        instance_manifest = make_instance_manifest(selected, ids_by_split, ann_ids_by_split)
        scene_manifest = make_scene_manifest(source_rows, selected)
        write_tsv(
            temp_dir / "manifests" / "images.tsv",
            [
                "derived_split",
                "source_split",
                "source",
                "upstream_split",
                "source_image",
                "image",
                "label",
                "relative_image_path",
                "relative_label_path",
                "width",
                "height",
                "instances",
                "scene_key",
                "canonical_asset",
                "image_sha256",
                "label_sha256",
                "perceptual_group_id",
                "coco_image_id",
                "included_in_test_scene_unique",
                "test_scene_unique_coco_image_id",
            ],
            image_manifest,
        )
        write_tsv(
            temp_dir / "manifests" / "instances.tsv",
            [
                "derived_split",
                "source_split",
                "source",
                "image",
                "label",
                "label_line",
                "coco_image_id",
                "coco_annotation_id",
                "yolo_class_id",
                "coco_category_id",
                "vertices",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "area",
                "mask_sha256",
                "polygon_sha256",
            ],
            instance_manifest,
        )
        write_tsv(
            temp_dir / "manifests" / "scene_groups.tsv",
            [
                "scene_key",
                "source",
                "canonical_asset",
                "original_splits",
                "original_images",
                "original_instances",
                "cross_split_original",
                "selected_splits",
                "selected_images",
                "selected_instances",
                "members",
            ],
            scene_manifest,
        )
        write_tsv(
            temp_dir / "manifests" / "perceptual_groups.tsv",
            [
                "group_id",
                "winner_split",
                "representative",
                "members",
                "left_split",
                "left_source",
                "left_image",
                "left_scene_key",
                "right_split",
                "right_source",
                "right_image",
                "right_scene_key",
                "d4_index",
                "d4_transform",
                "coarse_corr",
                "confirm_corr",
                "ssim",
            ],
            perceptual_manifest,
        )
        write_tsv(
            temp_dir / "manifests" / "exclusions.tsv",
            [
                "scope",
                "source_split",
                "source",
                "source_image",
                "image",
                "label",
                "instances",
                "scene_key",
                "image_sha256",
                "reason",
                "conflicts_with",
                "representative",
                "perceptual_group_id",
                "d4_transform",
                "coarse_corr",
                "confirm_corr",
                "ssim",
            ],
            exclusions,
        )

        derived_stats = {
            "train_clean": split_statistics(selected["train_clean"]),
            "val_clean": split_statistics(selected["val_clean"]),
            "test_frozen": split_statistics(selected["test_frozen"]),
            "test_scene_unique": split_statistics(test_unique),
        }
        generated_paths = [
            *coco_paths.values(),
            temp_dir / "manifests" / "images.tsv",
            temp_dir / "manifests" / "instances.tsv",
            temp_dir / "manifests" / "scene_groups.tsv",
            temp_dir / "manifests" / "perceptual_groups.tsv",
            temp_dir / "manifests" / "exclusions.tsv",
        ]
        generated_hashes = {
            path.relative_to(temp_dir).as_posix(): file_sha256(path) for path in sorted(generated_paths)
        }
        exclusion_counts = Counter(f"{row['scope']}::{row['reason']}" for row in exclusions)
        summary = {
            "dataset_version": DATASET_VERSION,
            "source": {
                "project_root": project_root.as_posix(),
                "split_manifest": manifest_path.relative_to(project_root).as_posix(),
                "split_manifest_sha256": file_sha256(manifest_path),
                "script": Path(__file__).resolve().relative_to(project_root).as_posix(),
                "script_sha256": file_sha256(Path(__file__).resolve()),
            },
            "category": {
                "source_yolo_id": TARGET_CLASS_ID,
                "coco_id": COCO_CATEGORY_ID,
                "name": COCO_CATEGORY_NAME,
                "supercategory": "fruit",
            },
            "policy": {
                "test": "frozen; never removed from test_frozen",
                "scene_key": "source + image stem before .rf.; otherwise full stem",
                "val": "exclude scene_key or image SHA-256 found in frozen test",
                "train": "exclude scene_key or image SHA-256 found in test or clean val",
                "train_exact_duplicates": (
                    "keep deterministic first only after maximum-cardinality one-to-one "
                    f"raster-mask matching at IoU >= {DUPLICATE_MASK_IOU_THRESHOLD}"
                ),
                "perceptual_d4": perceptual_audit["rule"],
                "images_copied": False,
                "coco_file_name_base": project_root.as_posix(),
            },
            "frozen_test": {
                "expected_fingerprint": args.expected_test_fingerprint,
                "actual_fingerprint": actual_test_fingerprint,
                "matched": True,
            },
            "original": original_statistics(source_rows),
            "derived": derived_stats,
            "perceptual_d4_audit": perceptual_audit,
            "exclusions": {
                "rows": len(exclusions),
                "by_scope_and_reason": dict(sorted(exclusion_counts.items())),
            },
            "generated_files_sha256": generated_hashes,
            "known_limitations": [
                "Explicit original-asset and exact-byte leakage is removed, but acquisition-session independence is not proven.",
                "local_test_images_20260707 is one short capture burst split across the original train/val/test.",
                "my_labelme contains temporally adjacent frames split across the original train/val/test.",
                "The frozen test has no background-only image, so it cannot estimate no-watermelon specificity.",
            ],
        }
        integrity = {
            "status": "pass",
            "checks": {
                "manifest_exactly_covers_source_directories": True,
                "all_yolo_rows_class_0_and_valid_polygons": True,
                "bbox_and_area_generated_by_pycocotools": True,
                "all_exact_duplicate_labels_semantically_equal_at_iou_0p99": True,
                "source_files_unchanged_after_generation": True,
                "frozen_test_fingerprint_matches": True,
                "cross_split_scene_and_sha256_intersections_zero": True,
                "cross_split_perceptual_d4_confirmed_pairs_zero_after_priority": True,
                "all_coco_jsons_round_trip_through_pycocotools_COCO": True,
            },
            "source_files": {
                "images": len(source_rows),
                "labels": len(source_rows),
                "instances": sum(len(row.instances) for row in source_rows),
            },
            "exact_image_duplicates": duplicate_semantics,
            "base_cross_split_intersections_before_perceptual_filter": base_intersections,
            "cross_split_intersections": intersections,
            "perceptual_d4_audit": perceptual_audit,
            "frozen_test": summary["frozen_test"],
            "generated_files_sha256": generated_hashes,
        }

        verify_source_unchanged(source_rows)
        write_json(temp_dir / "summary.json", summary)
        write_json(temp_dir / "integrity.json", integrity)
        write_readme(temp_dir / "README.md", summary, project_root, output_dir)
        publish_directory(temp_dir, output_dir, args.force)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    print(f"Published: {output_dir}")
    print(json.dumps(summary["derived"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
