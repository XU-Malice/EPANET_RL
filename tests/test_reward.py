import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epanet_rl.reward import (  # noqa: E402
    StepRewardInput,
    TankPenaltyConfig,
    compute_regular_reward,
    compute_tank_penalty,
    compute_total_reward,
)


def test_regular_reward_formula() -> None:
    reward = compute_regular_reward(r_benchmark=240.0, e_pump_t=3.0, horizon_steps=24)
    assert reward == pytest.approx(7.0)


def test_hydraulic_violation_overrides_reward_and_terminates() -> None:
    step = StepRewardInput(
        t=10,
        e_pump_t=3.0,
        r_benchmark=240.0,
        hydraulic_violation=True,
        p_hydraulic=-999.0,
        initial_tank_volume=100.0,
        final_tank_volume=90.0,
    )
    cfg = TankPenaltyConfig(mode="proportional", proportional_coefficient=-1.0)
    result = compute_total_reward(step, cfg)
    assert result.reward == pytest.approx(-999.0)
    assert result.terminated is True
    assert result.base_reward == pytest.approx(0.0)
    assert result.tank_penalty == pytest.approx(0.0)


def test_constant_tank_penalty_mode_is_unchanged() -> None:
    cfg = TankPenaltyConfig(mode="constant", constant_value=-50.0)
    penalty = compute_tank_penalty(
        initial_tank_volume=100.0,
        final_tank_volume=90.0,
        r_benchmark=2000.0,
        config=cfg,
    )
    assert penalty == pytest.approx(-50.0)


def test_proportional_tank_penalty_uses_ratio_and_benchmark() -> None:
    cfg = TankPenaltyConfig(mode="proportional", proportional_coefficient=-1.0)
    penalty = compute_tank_penalty(
        initial_tank_volume=100.0,
        final_tank_volume=90.0,
        r_benchmark=2000.0,
        config=cfg,
    )
    # shortfall_ratio = (100 - 90) / 100 = 0.1, penalty = -1.0 * 0.1 * 2000 = -200
    assert penalty == pytest.approx(-200.0)


def test_tank_penalty_zero_when_final_not_below_initial() -> None:
    cfg = TankPenaltyConfig(mode="proportional", proportional_coefficient=-1.0)
    penalty = compute_tank_penalty(
        initial_tank_volume=100.0,
        final_tank_volume=100.0,
        r_benchmark=2000.0,
        config=cfg,
    )
    assert penalty == pytest.approx(0.0)


def test_tank_penalty_safe_guard_when_initial_volume_non_positive() -> None:
    cfg = TankPenaltyConfig(mode="proportional", proportional_coefficient=-1.0)
    penalty = compute_tank_penalty(
        initial_tank_volume=0.0,
        final_tank_volume=-1.0,
        r_benchmark=2000.0,
        config=cfg,
    )
    assert penalty == pytest.approx(0.0)


def test_final_step_adds_tank_penalty_to_base_reward() -> None:
    step = StepRewardInput(
        t=23,
        e_pump_t=10.0,
        r_benchmark=2400.0,
        hydraulic_violation=False,
        p_hydraulic=-200.0,
        initial_tank_volume=100.0,
        final_tank_volume=90.0,
    )
    cfg = TankPenaltyConfig(mode="proportional", proportional_coefficient=-1.0)
    result = compute_total_reward(step, cfg, horizon_steps=24, final_step=23)

    expected_base = 2400.0 / 24.0 - 10.0
    expected_penalty = -1.0 * ((100.0 - 90.0) / 100.0) * 2400.0
    assert result.base_reward == pytest.approx(expected_base)
    assert result.tank_penalty == pytest.approx(expected_penalty)
    assert result.reward == pytest.approx(expected_base + expected_penalty)
    assert result.terminated is False
