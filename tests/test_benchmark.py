from yolo26_dual.benchmark import _percentile


def test_percentiles_are_deterministic_and_empty_safe():
    assert _percentile([], 0.95) == 0.0
    assert _percentile([0.4, 0.1, 0.3, 0.2], 0.5) == 0.2
    assert _percentile([0.4, 0.1, 0.3, 0.2], 0.95) == 0.3
