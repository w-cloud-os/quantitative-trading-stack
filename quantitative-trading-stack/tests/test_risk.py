import math

import pytest

from quant_stack.risk import DrawdownMonitor, slice_quantity


def test_monitor_trips_at_limit_and_latches():
    m = DrawdownMonitor(0.05)
    assert not m.update(1_000_000)
    assert not m.update(970_000)  # -3%
    assert m.update(949_000)  # -5.1%
    assert m.update(2_000_000)  # recovery does not reset it
    assert m.tripped


def test_monitor_tracks_high_water_mark():
    m = DrawdownMonitor(0.10)
    m.update(100)
    m.update(120)
    assert m.peak == 120
    assert not m.update(110)  # -8.3% from 120
    assert m.update(107)  # -10.8%


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_monitor_rejects_bad_limits(bad):
    with pytest.raises(ValueError):
        DrawdownMonitor(bad)


@pytest.mark.parametrize("equity", [0.0, -5.0, math.nan, math.inf])
def test_monitor_trips_on_corrupt_equity(equity):
    assert DrawdownMonitor(0.1).update(equity)


def test_slice_quantity_respects_limit_and_sums():
    parts = slice_quantity(5000, 100.0, 250_000)
    assert sum(parts) == 5000
    assert all(p * 100.0 <= 250_000 for p in parts)
    assert len(parts) == 2


def test_slice_quantity_edge_cases():
    assert slice_quantity(0, 10.0, 1000) == []
    assert slice_quantity(7, 10.0, 1000) == [7]
    with pytest.raises(ValueError):
        slice_quantity(1, 500.0, 100.0)
