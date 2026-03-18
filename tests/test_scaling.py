import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epanet_rl.scaling import (  # noqa: E402
    demand_max_min_scale,
    demand_z_score_scale,
    max_min_scale,
    tank_max_min_scale,
    tank_z_score_scale,
    z_score_scale,
)


def test_tank_max_min_scaling_basic_case() -> None:
    tank_levels = np.array([10.0, 20.0, 30.0])
    scaled = tank_max_min_scale(tank_levels, tank_level_min=0.0, tank_level_max=40.0)
    assert np.allclose(scaled, np.array([0.25, 0.5, 0.75]))


def test_demand_max_min_scaling_with_equal_bounds_is_stable() -> None:
    demands = np.array([5.0, 7.0])
    scaled = demand_max_min_scale(demands, demand_min=np.array([2.0, 3.0]), demand_max=np.array([2.0, 3.0]))
    assert np.allclose(scaled, np.array([0.5, 0.5]))


def test_max_min_scaling_clip_behavior() -> None:
    x = np.array([-1.0, 0.5, 2.0])
    clipped = max_min_scale(x, lower_bounds=0.0, upper_bounds=1.0, clip=True)
    unclipped = max_min_scale(x, lower_bounds=0.0, upper_bounds=1.0, clip=False)
    assert np.allclose(clipped, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(unclipped, np.array([-1.0, 0.5, 2.0]))


def test_tank_and_demand_z_score_support_zero_std_safely() -> None:
    tank_z = tank_z_score_scale([1.0, 2.0], tank_mean=[1.0, 2.0], tank_std=[0.0, 0.0])
    demand_z = demand_z_score_scale([3.0, 4.0], demand_mean=[3.0, 4.0], demand_std=[0.0, 0.0])
    assert np.allclose(tank_z, np.array([0.0, 0.0]))
    assert np.allclose(demand_z, np.array([0.0, 0.0]))


def test_z_score_general_formula() -> None:
    z = z_score_scale(values=[10.0, 20.0], means=[15.0, 15.0], stds=[5.0, 5.0])
    assert np.allclose(z, np.array([-1.0, 1.0]))


def test_max_min_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError):
        max_min_scale(values=[1.0], lower_bounds=[2.0], upper_bounds=[1.0])
