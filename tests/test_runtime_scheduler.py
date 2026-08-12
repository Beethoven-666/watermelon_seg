from yolo26_dual.config import WatermelonScheduler


def test_every_ten_frames() -> None:
    scheduler = WatermelonScheduler(10)
    assert [index for index in range(21) if scheduler.should_run(index)] == [0, 10, 20]


def test_zero_means_once_and_one_means_every_frame() -> None:
    once = WatermelonScheduler(0)
    assert [once.should_run(index) for index in range(4)] == [True, False, False, False]
    every = WatermelonScheduler(1)
    assert all(every.should_run(index) for index in range(4))


def test_manual_refresh_is_immediate() -> None:
    scheduler = WatermelonScheduler(10)
    assert scheduler.should_run(0)
    assert not scheduler.should_run(1)
    scheduler.request_refresh()
    assert scheduler.should_run(2)
