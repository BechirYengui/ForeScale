"""Unit tests for the desired-replicas computation."""

from __future__ import annotations

import math

import pytest

from forescale_core.replica_math import desired_replicas


def test_basic_formula_rounds_up_with_margin() -> None:
    # 120 rps / 50 cap = 2.4; *1.15 = 2.76 -> ceil = 3.
    assert desired_replicas(120, 50, safety_margin=0.15) == 3


def test_zero_load_clamps_to_min() -> None:
    assert desired_replicas(0, 50, min_replicas=2) == 2


def test_clamps_to_max() -> None:
    assert desired_replicas(10_000, 50, max_replicas=20) == 20


def test_exact_capacity_no_margin() -> None:
    # 100 / 50 = 2 exactly, no margin -> 2.
    assert desired_replicas(100, 50, safety_margin=0.0) == 2


def test_never_below_one_even_with_min_zero() -> None:
    # A live service should keep at least one pod when there is any demand.
    assert desired_replicas(1, 50, min_replicas=0) >= 1


@pytest.mark.parametrize(
    "rps,cap,margin",
    [(75, 50, 0.2), (200, 30, 0.1), (45, 60, 0.5)],
)
def test_matches_reference_formula(rps: float, cap: float, margin: float) -> None:
    expected = max(1, math.ceil(rps / cap * (1 + margin)))
    expected = max(1, min(expected, 20))
    assert desired_replicas(rps, cap, safety_margin=margin) == expected


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        desired_replicas(10, 0)


def test_negative_margin_raises() -> None:
    with pytest.raises(ValueError):
        desired_replicas(10, 50, safety_margin=-0.1)


def test_min_above_max_raises() -> None:
    with pytest.raises(ValueError):
        desired_replicas(10, 50, min_replicas=5, max_replicas=3)
