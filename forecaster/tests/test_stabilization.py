"""Unit tests for the scale-down max-hold stabilization."""

from __future__ import annotations

import pytest

from forescale_core.stabilization import MaxHold


def test_single_update_returns_that_value() -> None:
    assert MaxHold(60).update(0.0, 3) == 3


def test_holds_recent_peak_during_a_dip() -> None:
    hold = MaxHold(window_s=60)
    assert hold.update(0.0, 5) == 5
    # A forecast dip should not immediately strip the pre-warmed pods.
    assert hold.update(10.0, 2) == 5
    assert hold.update(30.0, 3) == 5


def test_peak_decays_after_the_window() -> None:
    hold = MaxHold(window_s=60)
    hold.update(0.0, 5)
    # Once the t=0 peak falls outside the trailing window it is forgotten.
    assert hold.update(61.0, 2) == 2


def test_eviction_keeps_boundary_entry() -> None:
    # Entry exactly window_s old is still within the trailing window (>= cutoff).
    hold = MaxHold(window_s=60)
    hold.update(0.0, 7)
    assert hold.update(60.0, 1) == 7
    assert hold.update(60.5, 1) == 1


def test_result_never_below_current_desired() -> None:
    hold = MaxHold(window_s=30)
    hold.update(0.0, 2)
    # Even after a higher peak ages out, the result is at least the new value.
    assert hold.update(100.0, 8) == 8


def test_zero_window_has_no_memory() -> None:
    hold = MaxHold(window_s=0)
    assert hold.update(0.0, 5) == 5
    assert hold.update(1.0, 2) == 2


def test_negative_window_raises() -> None:
    with pytest.raises(ValueError):
        MaxHold(window_s=-1)
