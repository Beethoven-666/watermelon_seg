from pathlib import Path
from yolo26_dual.diagnostics import dataset_ready


def test_gitkeep_only_dataset_is_not_ready(tmp_path: Path):
    for split in ("train", "val", "test"):
        image = tmp_path / "images" / split
        label = tmp_path / "labels" / split
        image.mkdir(parents=True)
        label.mkdir(parents=True)
        (image / ".gitkeep").touch()
        (label / ".gitkeep").touch()
    assert not dataset_ready(tmp_path)


def test_all_splits_need_real_images_and_labels(tmp_path: Path):
    for split in ("train", "val", "test"):
        image = tmp_path / "images" / split
        label = tmp_path / "labels" / split
        image.mkdir(parents=True)
        label.mkdir(parents=True)
        (image / "sample.jpg").write_bytes(b"x")
        (label / "sample.txt").write_text("")
    assert dataset_ready(tmp_path)
