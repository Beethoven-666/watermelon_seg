from pathlib import Path
import pytest

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]


def test_official_yolo26_models_load_with_expected_tasks():
    from ultralytics import YOLO

    detect = ROOT / "yolo26s.pt"
    segment = ROOT / "yolo26s-seg.pt"
    if not detect.is_file() or not segment.is_file():
        pytest.skip("official YOLO26 weights unavailable")
    assert YOLO(str(detect)).task == "detect"
    assert YOLO(str(segment)).task == "segment"


def test_custom_watermelon_weight_produces_instance_mask():
    import torch
    from ultralytics import YOLO

    weight = ROOT / "runs/yolo26_watermelon_seg/train/weights/best.pt"
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not weight.is_file():
        pytest.skip("formal YOLO26s-seg weight unavailable")
    image = next((ROOT / "images/test").glob("*.jpg"))
    result = YOLO(str(weight)).predict(str(image), device=0, imgsz=640, verbose=False)[
        0
    ]
    assert result.masks is not None and len(result.masks.data) > 0


def test_custom_tray_weight_load_and_bytetrack_video():
    weight = ROOT / "runs/yolo26_tray_detect/train/weights/best.pt"
    if not weight.is_file():
        pytest.skip("real audited tray weight unavailable")
    pytest.fail(
        "A fixed real tray test video must be configured before this test is enabled"
    )
