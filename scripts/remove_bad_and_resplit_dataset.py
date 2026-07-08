#!/usr/bin/env python
"""Remove visually rejected samples and rebuild the main YOLO split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
SEED = 20260707
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

BAD_VISUAL_STEMS = {
    "test": [
        "000035_roboflow_drago_v1_467-Copy_jpg.rf.3ae4fc23861bd3468222d4ecd078b5c5",
        "000036_roboflow_drago_v1_467-Copy_jpg.rf.b07a0ec07204fc12edb0736ce311d9b2",
        "000037_roboflow_drago_v1_506-Copy_jpg.rf.13c41445558d89c5d48278a09a60f740",
        "000103_roboflow_team128_v5_IMG_1970_jpg.rf.b8a8be1c1634aa768b3b766951d507d0",
        "000104_roboflow_team128_v5_IMG_1974_jpg.rf.148e2ced2adf6bb65910673ae15b175b",
        "000105_roboflow_team128_v5_IMG_1983_jpg.rf.639c1326031e83138e47b59dc62c62ef",
        "000106_roboflow_team128_v5_IMG_1992_jpg.rf.55e6484d663f7d48b0ea38cbc42894ef",
        "000107_roboflow_team128_v5_IMG_2002_jpg.rf.8f3a2480235c1f8716bca35fc16b2a9c",
    ],
    "train": [
        "000097_roboflow_drago_v1_209_jpg.rf.b0a674cad423a1bd2497101061244d2d",
        "000437_roboflow_team128_v5_IMG_0345_jpg.rf.76fa801514fc08ff8289d0d429ba710d",
        "000456_roboflow_team128_v5_IMG_0367_jpg.rf.a308c36a0026e2d5d06b1a9d8fce4d67",
        "000468_roboflow_team128_v5_IMG_0381_jpg.rf.35562055fa1d2e5a5687cc89edfd0e10",
        "000556_roboflow_team128_v5_IMG_0493_jpg.rf.b122be104e0269593056d274321eb9f5",
        "000582_roboflow_team128_v5_IMG_0532_jpg.rf.01a31a39d1ce7fdaa092f3bce3d53caf",
        "000602_roboflow_team128_v5_IMG_0558_jpg.rf.2fe5851f8a71a8b26e17c8b83ad446f4",
        "000609_roboflow_team128_v5_IMG_0572_jpg.rf.5ce632021766d664bdff4de1d626df53",
        "000610_roboflow_team128_v5_IMG_0573_jpg.rf.e94350893da0f79e970fac45ce5863c9",
        "000611_roboflow_team128_v5_IMG_0574_jpg.rf.a9dcfbea14ded30f79b770ea3e14ee19",
        "000615_roboflow_team128_v5_IMG_0578_jpg.rf.72f11d31c44ca177e44c4e0b9150732b",
        "000616_roboflow_team128_v5_IMG_0579_jpg.rf.81b247631edf2206d69b2063d7fde01c",
        "000622_roboflow_team128_v5_IMG_0586_jpg.rf.d829a4c0b3df1e0a9495a2d1536dc8a4",
        "000630_roboflow_team128_v5_IMG_0594_jpg.rf.c01621a3134f7c2400e6a4419497a135",
        "000635_roboflow_team128_v5_IMG_0599_jpg.rf.c214bfbc03c53517711382b4eecdb1ca",
        "000636_roboflow_team128_v5_IMG_0600_jpg.rf.d907225c00156700561452accfb7a6e7",
        "000664_roboflow_team128_v5_IMG_0633_jpg.rf.a677d2658d64afc0402f88ed69839c3d",
        "000691_roboflow_team128_v5_IMG_0666_jpg.rf.4666877c7d5db73cb7b1fc88d82b59f2",
        "000717_roboflow_team128_v5_IMG_0698_jpg.rf.f0aa57ebbd479324f001f1db2c57f40d",
        "000814_roboflow_team128_v5_IMG_1966_jpg.rf.964648f92662130667312bf24da80260",
        "000815_roboflow_team128_v5_IMG_1967_jpg.rf.7c1ef6fb24feceb53f2d4db3b7c1398a",
    ],
    "val": [
        "000090_roboflow_team128_v5_IMG_0658_jpg.rf.bb3a013316d335158f0c900f4b8fd166",
        "000102_roboflow_team128_v5_IMG_1965_jpg.rf.08d856cb208ec65ca7b9c0d5f11c271b",
        "000103_roboflow_team128_v5_IMG_1977_jpg.rf.8e71c2b6affcafed1050eddb86b02b0d",
        "000104_roboflow_team128_v5_IMG_1987_jpg.rf.66b611db2dc7bc0ac6d361c3e5d67504",
        "000105_roboflow_team128_v5_IMG_2004_jpg.rf.eb73940d8552660b56ea5ebfa81bcb34",
    ],
}

BAD_VISUAL_RANGES = {
    "train": [(816, 845)],
}


@dataclass
class Sample:
    old_split: str
    new_split: str
    source: str
    source_split: str
    image: str
    label: str
    instances: int
    source_image: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def assert_inside(root: Path, path: Path) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise ValueError(f"Refusing to touch path outside workspace: {path_resolved}")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows[row["image"]] = row
    return rows


def visual_prefix(visual_stem: str) -> int:
    head, sep, _tail = visual_stem.partition("_")
    if not sep or len(head) != 6 or not head.isdigit():
        raise ValueError(f"Unexpected visualization stem: {visual_stem}")
    return int(head)


def image_stem_from_visual(visual_stem: str) -> str:
    visual_prefix(visual_stem)
    return visual_stem[7:]


def resolve_bad_visuals(root: Path) -> dict[str, set[str]]:
    bad: dict[str, set[str]] = {split: set(values) for split, values in BAD_VISUAL_STEMS.items()}
    visual_root = root / "exports" / "yolo_seg_label_check" / "visualizations"

    for split, ranges in BAD_VISUAL_RANGES.items():
        split_dir = visual_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing visualization directory for range lookup: {split_dir}")
        for visual_path in split_dir.glob("*.jpg"):
            prefix = visual_prefix(visual_path.stem)
            if any(start <= prefix <= end for start, end in ranges):
                bad.setdefault(split, set()).add(visual_path.stem)

    for split, stems in bad.items():
        split_dir = visual_root / split
        missing_visuals = [stem for stem in sorted(stems) if not (split_dir / f"{stem}.jpg").exists()]
        if missing_visuals:
            joined = "\n".join(missing_visuals)
            raise FileNotFoundError(f"Missing listed visualization files for {split}:\n{joined}")
    return bad


def image_files(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def count_instances(label_path: Path) -> int:
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def infer_source(row: dict[str, str] | None, image_name: str) -> tuple[str, str, str]:
    if row:
        return row["source"], row["source_split"], row["source_image"]
    stem = Path(image_name).stem
    if stem.startswith("my_labelme_"):
        raw_name = stem.removeprefix("my_labelme_") + Path(image_name).suffix
        return "my_labelme", "manual", f"raw\\{raw_name}"
    if stem.startswith("roboflow_drago_v1_"):
        return "roboflow_drago_v1", "unknown", ""
    if stem.startswith("roboflow_team128_v5_"):
        return "roboflow_team128_v5", "unknown", ""
    return "unknown", "unknown", ""


def load_samples(root: Path, manifest: dict[str, dict[str, str]]) -> list[Sample]:
    samples: list[Sample] = []
    seen_images: set[str] = set()
    for split in SPLITS:
        for image_path in image_files(root / "images" / split):
            if image_path.name in seen_images:
                raise ValueError(f"Duplicate image filename across splits: {image_path.name}")
            seen_images.add(image_path.name)

            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}")

            row = manifest.get(image_path.name)
            source, source_split, source_image = infer_source(row, image_path.name)
            samples.append(
                Sample(
                    old_split=split,
                    new_split=split,
                    source=source,
                    source_split=source_split,
                    image=image_path.name,
                    label=label_path.name,
                    instances=count_instances(label_path),
                    source_image=source_image,
                    image_path=image_path,
                    label_path=label_path,
                )
            )
    return samples


def assign_splits(samples: list[Sample]) -> None:
    ordered = sorted(samples, key=lambda item: (item.source, item.image))
    shuffled = list(ordered)
    random.Random(SEED).shuffle(shuffled)

    train_count = int(len(shuffled) * RATIOS["train"])
    val_count = int(len(shuffled) * RATIOS["val"])
    for index, sample in enumerate(shuffled):
        if index < train_count:
            sample.new_split = "train"
        elif index < train_count + val_count:
            sample.new_split = "val"
        else:
            sample.new_split = "test"


def write_removed(path: Path, removed: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "old_split",
                "source",
                "source_split",
                "image",
                "label",
                "instances",
                "source_image",
            ],
        )
        writer.writeheader()
        for sample in sorted(removed, key=lambda item: (item.old_split, item.image)):
            writer.writerow(
                {
                    "old_split": sample.old_split,
                    "source": sample.source,
                    "source_split": sample.source_split,
                    "image": sample.image,
                    "label": sample.label,
                    "instances": sample.instances,
                    "source_image": sample.source_image,
                }
            )


def write_manifest(path: Path, kept: list[Sample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["split", "source", "source_split", "image", "label", "instances", "source_image"],
        )
        writer.writeheader()
        for sample in sorted(kept, key=lambda item: (SPLITS.index(item.new_split), item.source, item.image)):
            writer.writerow(
                {
                    "split": sample.new_split,
                    "source": sample.source,
                    "source_split": sample.source_split,
                    "image": sample.image,
                    "label": sample.label,
                    "instances": sample.instances,
                    "source_image": sample.source_image,
                }
            )


def build_summary(kept: list[Sample], removed: list[Sample]) -> dict:
    summary = {
        split: {"images": 0, "labels": 0, "instances": 0, "sources": {}} for split in SPLITS
    }
    source_totals: dict[str, int] = {}
    removed_summary: dict[str, dict[str, int]] = {}

    for sample in kept:
        row = summary[sample.new_split]
        row["images"] += 1
        row["labels"] += 1
        row["instances"] += sample.instances
        row["sources"][sample.source] = row["sources"].get(sample.source, 0) + 1
        source_totals[sample.source] = source_totals.get(sample.source, 0) + 1

    for sample in removed:
        row = removed_summary.setdefault(sample.old_split, {"images": 0, "instances": 0})
        row["images"] += 1
        row["instances"] += sample.instances

    for row in summary.values():
        row["sources"] = dict(sorted(row["sources"].items()))

    return {
        "seed": SEED,
        "ratios": RATIOS,
        "total_images": len(kept),
        "total_instances": sum(sample.instances for sample in kept),
        "removed_images": len(removed),
        "removed_instances": sum(sample.instances for sample in removed),
        "summary": summary,
        "source_totals": dict(sorted(source_totals.items())),
        "removed_summary": removed_summary,
    }


def move_sample(root: Path, sample: Sample) -> None:
    if sample.old_split == sample.new_split:
        return
    target_image = root / "images" / sample.new_split / sample.image
    target_label = root / "labels" / sample.new_split / sample.label
    assert_inside(root, target_image)
    assert_inside(root, target_label)
    if target_image.exists() or target_label.exists():
        raise FileExistsError(f"Refusing to overwrite target for {sample.image}")
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sample.image_path), str(target_image))
    shutil.move(str(sample.label_path), str(target_label))
    sample.image_path = target_image
    sample.label_path = target_label


def delete_sample(root: Path, sample: Sample) -> None:
    assert_inside(root, sample.image_path)
    assert_inside(root, sample.label_path)
    sample.image_path.unlink()
    sample.label_path.unlink()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / "exports" / "fused_watermelon_yolo_seg" / "split_manifest.tsv"
    summary_path = root / "exports" / "fused_watermelon_yolo_seg" / "summary.json"
    removed_path = root / "exports" / "fused_watermelon_yolo_seg" / "removed_bad_samples_20260707.tsv"

    bad_visuals = resolve_bad_visuals(root)
    bad_by_split = {
        split: {image_stem_from_visual(stem) for stem in stems}
        for split, stems in bad_visuals.items()
    }

    manifest = read_manifest(manifest_path)
    samples = load_samples(root, manifest)
    bad_keys = {(split, stem) for split, stems in bad_by_split.items() for stem in stems}
    samples_by_key = {(sample.old_split, Path(sample.image).stem): sample for sample in samples}
    missing = sorted(key for key in bad_keys if key not in samples_by_key)
    if missing:
        joined = "\n".join(f"{split}/{stem}" for split, stem in missing)
        raise FileNotFoundError(f"Listed bad samples are missing from the current dataset:\n{joined}")

    removed = [samples_by_key[key] for key in sorted(bad_keys)]
    removed_set = {id(sample) for sample in removed}
    kept = [sample for sample in samples if id(sample) not in removed_set]
    assign_splits(kept)

    output_summary = build_summary(kept, removed)
    move_count = sum(1 for sample in kept if sample.old_split != sample.new_split)

    print(f"Root: {root}")
    print(f"Current samples: {len(samples)}")
    print(f"Remove samples: {len(removed)}")
    print(f"Keep samples: {len(kept)}")
    print(f"Move kept samples between splits: {move_count}")
    for split in SPLITS:
        row = output_summary["summary"][split]
        print(f"{split}: {row['images']} images, {row['instances']} instances")

    if args.dry_run:
        print("Dry run only; no files were changed.")
        return 0

    write_removed(removed_path, removed)
    for sample in removed:
        delete_sample(root, sample)
    for sample in kept:
        move_sample(root, sample)
    write_manifest(manifest_path, kept)
    summary_path.write_text(json.dumps(output_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Removed list: {removed_path}")
    print(f"Updated manifest: {manifest_path}")
    print(f"Updated summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
