import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("gymnasium")

from epanet_rl.action_space import ACTION_COUNT  # noqa: E402
from epanet_rl.env import Net3PumpSchedulingEnv  # noqa: E402


def _net3_path() -> Path:
    return ROOT / "networks" / "Net3.inp"


def test_env_reset_and_step_contract_minimal() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    obs, info = env.reset(seed=123)

    assert env.action_space.n == ACTION_COUNT == 64
    assert obs.dtype == np.float32
    assert obs.shape == (env._base_demands.size + len(env._tank_ids),)  # noqa: SLF001
    assert env.observation_space.contains(obs)

    assert "time_multipliers" in info and info["time_multipliers"].shape == (24,)
    assert "space_multipliers" in info and info["space_multipliers"].shape == (env._base_demands.size,)  # noqa: SLF001

    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "tank_levels" in step_info
    assert "base_reward" in step_info
    assert "tank_penalty" in step_info
    assert not any(re.search("pressure", key, flags=re.IGNORECASE) for key in step_info.keys())


def test_env_episode_ends_within_24_steps() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    env.reset(seed=0)
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        assert steps <= 24
    assert steps <= 24


def test_env_reset_random_initial_tank_levels_within_bounds() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    env.reset(seed=100)
    first_levels = env._tank_levels.copy()  # noqa: SLF001
    assert np.all(first_levels >= env._tank_min_levels)  # noqa: SLF001
    assert np.all(first_levels <= env._tank_max_levels)  # noqa: SLF001

    env.reset(seed=101)
    second_levels = env._tank_levels.copy()  # noqa: SLF001
    assert np.all(second_levels >= env._tank_min_levels)  # noqa: SLF001
    assert np.all(second_levels <= env._tank_max_levels)  # noqa: SLF001
    assert not np.allclose(first_levels, second_levels)


def test_env_supports_non_randomized_large_users_interface() -> None:
    env_probe = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    fixed_node_id = env_probe._demand_node_ids[0]  # noqa: SLF001

    env = Net3PumpSchedulingEnv(
        net3_inp_path=_net3_path(),
        non_randomized_node_ids=[fixed_node_id],
    )
    _, info = env.reset(seed=7)
    fixed_idx = env._demand_node_ids.index(fixed_node_id)  # noqa: SLF001
    assert info["space_multipliers"][fixed_idx] == pytest.approx(1.0)


def test_env_step_after_done_raises_runtime_error() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
    with pytest.raises(RuntimeError):
        env.step(0)
