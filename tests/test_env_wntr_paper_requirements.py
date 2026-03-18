"""Paper-compliance tests for Net3WntrEnv.

This file checks core requirements for a real WNTR/EPANET single-step
sequential environment and explicitly marks still-pending items.
"""

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

from epanet_rl.action_space import ACTION_COUNT, action_id_to_speeds  # noqa: E402
from epanet_rl.env_wntr import Net3WntrEnv  # noqa: E402


def _net3_path() -> Path:
    return ROOT / "networks" / "Net3.inp"


def test_env_wntr_has_24_steps_and_1hour_step_size() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        assert env.EPISODE_STEPS == 24
        assert env.STEP_HOURS == 1.0
        assert env.STEP_SECONDS == 3600
    finally:
        env.close()


def test_env_wntr_state_is_demand_plus_tank_level() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        obs, _ = env.reset(seed=123)
        # Net3 in this setup: 92 junction demands + 3 tank levels.
        assert len(env._junction_names) == 92  # noqa: SLF001
        assert len(env._tank_names) == 3  # noqa: SLF001
        assert obs.shape == (95,)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_env_wntr_action_space_is_64_discrete_combinations() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        assert env.action_space.n == ACTION_COUNT == 64
        _, _ = env.reset(seed=7)
        _, _, _, _, info = env.step(63)
        assert info["original_action"] == 63
        assert tuple(info["executed_pump_speeds"]) == action_id_to_speeds(63)
    finally:
        env.close()


def test_reset_randomizes_demands_and_tank_initial_levels() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        _, info1 = env.reset(seed=1)
        levels1 = env._tank_levels.copy()  # noqa: SLF001
        _, info2 = env.reset(seed=2)
        levels2 = env._tank_levels.copy()  # noqa: SLF001

        assert info1["time_multipliers"].shape == (24,)
        assert info1["space_multipliers"].shape == (len(env._junction_names),)  # noqa: SLF001
        assert float(np.std(info1["time_multipliers"])) > 0.0
        assert float(np.std(info1["space_multipliers"])) > 0.0

        assert np.all(levels1 >= env._tank_min_levels)  # noqa: SLF001
        assert np.all(levels1 <= env._tank_max_levels)  # noqa: SLF001
        assert np.all(levels2 >= env._tank_min_levels)  # noqa: SLF001
        assert np.all(levels2 <= env._tank_max_levels)  # noqa: SLF001
        assert not np.allclose(levels1, levels2)
    finally:
        env.close()


def test_step_calls_real_epanet_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    import epanet_rl.env_wntr as env_wntr_module

    original_simulator_ctor = env_wntr_module.wntr.sim.EpanetSimulator
    call_count = {"n": 0}

    def tracking_simulator_ctor(*args, **kwargs):
        call_count["n"] += 1
        return original_simulator_ctor(*args, **kwargs)

    monkeypatch.setattr(env_wntr_module.wntr.sim, "EpanetSimulator", tracking_simulator_ctor)

    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        env.reset(seed=42)
        _, _, _, _, info = env.step(63)
        assert call_count["n"] >= 1
        assert "pump_flows" in info
        assert "pump_head_gains" in info
        assert "min_pressure" in info
    finally:
        env.close()


def test_step_info_contains_required_hydraulic_outputs() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        env.reset(seed=42)
        _, _, _, _, info = env.step(63)

        pump_flows = np.asarray(info["pump_flows"], dtype=np.float64)
        pump_head_gains = np.asarray(info["pump_head_gains"], dtype=np.float64)
        min_pressure = float(info["min_pressure"])

        assert pump_flows.shape == (2,)
        assert pump_head_gains.shape == (2,)
        assert isinstance(min_pressure, float)
    finally:
        env.close()


def test_fixed_action_63_rollout_runs_multiple_steps_without_immediate_nan_crash() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        env.reset(seed=42)
        finite_steps = 0
        for _ in range(24):
            _, _, terminated, truncated, info = env.step(63)
            pump_flows = np.asarray(info["pump_flows"], dtype=np.float64)
            pump_head_gains = np.asarray(info["pump_head_gains"], dtype=np.float64)
            min_pressure = float(info["min_pressure"])

            if np.isfinite(min_pressure) and np.isfinite(pump_flows).all() and np.isfinite(pump_head_gains).all():
                finite_steps += 1

            if terminated or truncated:
                break

        assert finite_steps >= 3
    finally:
        env.close()


def test_r_benchmark_is_currently_placeholder_and_allowed() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="none")
    try:
        assert env.r_benchmark == 2000.0, (
            "Current local implementation keeps r_benchmark as hard-coded placeholder; "
            "this is allowed at this stage but not the final paper statistic."
        )
    finally:
        env.close()


@pytest.mark.xfail(
    reason=(
        "ESFC-Z-Score calibration from large-scale statistics (e.g., 10,000 episodes) "
        "is intentionally pending. Current env keeps interface only."
    ),
    strict=False,
)
def test_pending_zscore_10000_episode_statistics_calibration() -> None:
    env = Net3WntrEnv(net3_inp_path=_net3_path(), scaling_mode="z_score")
    try:
        assert hasattr(env, "_zscore_stats_from_10000_episodes")
    finally:
        env.close()
