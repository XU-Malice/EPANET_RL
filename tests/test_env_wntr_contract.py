import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("gymnasium")
pytest.importorskip("wntr")

from epanet_rl.action_space import action_id_to_speeds  # noqa: E402
from epanet_rl.env_wntr import Net3WntrEnv  # noqa: E402


def _net3_path() -> Path:
    return ROOT / "networks" / "Net3.inp"


def test_tank_upper_bound_state_does_not_immediately_return_nan_outputs() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    env.reset(seed=123)

    # Contract setup: force a numerically risky state at tank upper bounds.
    env._tank_levels = env._tank_max_levels.copy()  # noqa: SLF001

    _, _, _, _, info = env.step(63)
    pump_flows = np.asarray(info["pump_flows"], dtype=np.float64)
    pump_head_gains = np.asarray(info["pump_head_gains"], dtype=np.float64)
    min_pressure = float(info["min_pressure"])

    assert info["tank_level_clipped_before_sim"] is True
    assert np.isfinite(min_pressure)
    assert np.isfinite(pump_flows).all()
    assert np.isfinite(pump_head_gains).all()

    env.close()


def test_constructor_rejects_non_negative_hydraulic_penalty() -> None:
    with pytest.raises(ValueError, match="p_hydraulic must be negative"):
        Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none", p_hydraulic=0.0)


def test_hydraulic_violation_terminates_with_penalty_and_reason() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none", p_hydraulic=-321.0)
    env.reset(seed=42)

    def fake_simulate_single_step(pump_speeds, current_demands, tank_init_levels):
        _ = pump_speeds, current_demands
        return {
            "tank_levels": np.asarray(tank_init_levels, dtype=np.float64),
            "hydraulic_violation": True,
            "min_pressure": -1.0,
            "pump_flows": np.asarray([0.0, 0.0], dtype=np.float64),
            "pump_head_gains": np.asarray([0.0, 0.0], dtype=np.float64),
            "pump_energy_cost": 123.0,
        }

    env._simulate_single_step = fake_simulate_single_step  # type: ignore[method-assign]  # noqa: SLF001
    _, reward, terminated, truncated, info = env.step(63)

    assert reward == pytest.approx(-321.0)
    assert terminated is True
    assert truncated is False
    assert info["hydraulic_violation"] is True
    assert info["termination_reason"] == "hydraulic_violation"

    env.close()


def test_tank_levels_passed_to_solver_are_below_max_by_epsilon() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none", tank_level_epsilon=1e-4)
    env.reset(seed=1)

    env._tank_levels = env._tank_max_levels.copy()  # noqa: SLF001
    _, _, _, _, info = env.step(63)

    levels_for_sim = np.asarray(info["tank_levels_for_sim"], dtype=np.float64)
    max_allowed = env._tank_max_levels - env.tank_level_epsilon  # noqa: SLF001
    assert np.all(levels_for_sim <= (max_allowed + 1e-12))

    env.close()


def test_normal_case_executed_action_matches_original_action() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    env.reset(seed=42)

    original_action = 63
    _, _, _, _, info = env.step(original_action)

    assert info["original_action"] == original_action
    assert tuple(info["executed_pump_speeds"]) == action_id_to_speeds(original_action)
    assert info["fallback_used"] is False
    assert info["original_action_failed"] is False

    env.close()


def test_fallback_only_used_when_original_action_failed() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    env.reset(seed=5)

    original_action = 63
    original_speeds = action_id_to_speeds(original_action)
    calls: list[tuple[float, float]] = []

    def fake_simulate_single_step(pump_speeds, current_demands, tank_init_levels):
        calls.append(tuple(float(x) for x in pump_speeds))
        if len(calls) == 1:
            # Force "original action failed" path.
            raise RuntimeError("forced original action simulation failure")
        return {
            "tank_levels": np.asarray(tank_init_levels, dtype=np.float64),
            "hydraulic_violation": False,
            "min_pressure": 10.0,
            "pump_flows": np.asarray([0.0, 0.0], dtype=np.float64),
            "pump_head_gains": np.asarray([0.0, 0.0], dtype=np.float64),
            "pump_energy_cost": 0.0,
        }

    env._simulate_single_step = fake_simulate_single_step  # type: ignore[method-assign]  # noqa: SLF001
    _, _, terminated, truncated, info = env.step(original_action)

    assert terminated is False
    assert truncated is False
    assert info["original_action"] == original_action
    assert info["original_action_failed"] is True
    assert info["fallback_used"] is True
    assert tuple(info["executed_pump_speeds"]) == (0.0, 0.0)
    assert calls[0] == original_speeds
    assert calls[1] == (0.0, 0.0)
    assert len(calls) == 2

    env.close()
