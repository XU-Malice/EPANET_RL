import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epanet_rl.demand_randomization import (  # noqa: E402
    compose_randomized_demands,
    generate_randomized_demands,
    generate_space_multipliers,
    generate_time_multipliers,
    sample_truncated_normal,
)


def test_truncated_normal_range_is_respected() -> None:
    rng = np.random.default_rng(123)
    delta = 0.2
    samples = sample_truncated_normal(size=5000, delta=delta, rng=rng)
    assert np.all(samples >= 1.0 - delta)
    assert np.all(samples <= 1.0 + delta)


def test_truncated_normal_delta_zero_returns_ones() -> None:
    rng = np.random.default_rng(123)
    samples = sample_truncated_normal(size=10, delta=0.0, rng=rng)
    assert np.allclose(samples, np.ones(10))


def test_time_multipliers_shape_and_range() -> None:
    rng = np.random.default_rng(7)
    multipliers = generate_time_multipliers(num_steps=24, delta_time=0.1, rng=rng)
    assert multipliers.shape == (24,)
    assert np.all(multipliers >= 0.9)
    assert np.all(multipliers <= 1.1)


def test_space_multipliers_support_non_randomized_mask() -> None:
    rng = np.random.default_rng(7)
    mask = np.array([True, False, True, False], dtype=bool)
    multipliers = generate_space_multipliers(
        num_nodes=4,
        delta_space=0.15,
        rng=rng,
        randomizable_mask=mask,
    )
    assert multipliers.shape == (4,)
    assert multipliers[1] == pytest.approx(1.0)
    assert multipliers[3] == pytest.approx(1.0)
    assert np.all(multipliers[mask] >= 0.85)
    assert np.all(multipliers[mask] <= 1.15)


def test_compose_randomized_demands_matches_formula() -> None:
    base = np.array([10.0, 20.0], dtype=np.float64)
    pattern = np.array([1.0, 2.0], dtype=np.float64)
    time_mul = np.array([0.9, 1.1], dtype=np.float64)
    space_mul = np.array([1.2, 0.8], dtype=np.float64)

    demands = compose_randomized_demands(
        base_demands=base,
        default_pattern=pattern,
        time_multipliers=time_mul,
        space_multipliers=space_mul,
    )

    expected = np.outer(pattern * time_mul, base * space_mul)
    assert demands.shape == (2, 2)
    assert np.allclose(demands, expected)


def test_generate_randomized_demands_returns_consistent_shapes() -> None:
    rng = np.random.default_rng(1234)
    base = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    pattern = np.array([1.0] * 24, dtype=np.float64)
    mask = np.array([True, False, True], dtype=bool)

    demands, time_mul, space_mul = generate_randomized_demands(
        base_demands=base,
        default_pattern=pattern,
        delta_time=0.1,
        delta_space=0.2,
        rng=rng,
        randomizable_mask=mask,
    )
    assert demands.shape == (24, 3)
    assert time_mul.shape == (24,)
    assert space_mul.shape == (3,)
    assert space_mul[1] == pytest.approx(1.0)
