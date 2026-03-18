"""Minimal Gymnasium environment for Net3 pump scheduling.

This is a structural, minimal runnable environment for smoke testing and
integration with later RL training code. It intentionally uses a simplified
internal dynamics model (not a full EPANET hydraulic solver) while preserving
the target interfaces:

- 24-step episode, 1 hour per step
- 64 discrete actions from ``action_space.py``
- state = [node demands, tank levels]
- demand randomization via ``demand_randomization.py``
- reward calculation via ``reward.py``
- scaling interface via ``scaling.py``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .action_space import ACTION_COUNT, action_id_to_speeds
from .demand_randomization import generate_randomized_demands
from .reward import StepRewardInput, TankPenaltyConfig, compute_total_reward
from .scaling import (
    demand_max_min_scale,
    demand_z_score_scale,
    tank_max_min_scale,
    tank_z_score_scale,
)

ScalingMode = Literal["none", "max_min", "z_score"]


@dataclass(frozen=True)
class _ParsedNet3Data:
    demand_node_ids: tuple[str, ...]
    base_demands: NDArray[np.float64]
    default_pattern: NDArray[np.float64]
    tank_ids: tuple[str, ...]
    tank_init_levels: NDArray[np.float64]
    tank_min_levels: NDArray[np.float64]
    tank_max_levels: NDArray[np.float64]
    tank_diameters: NDArray[np.float64]
    tank_min_volumes: NDArray[np.float64]


class Net3PumpSchedulingEnv(gym.Env[NDArray[np.float32], int]):
    """Minimal Net3 pump scheduling environment.

    Notes:
    - This environment is for pipeline bring-up and smoke tests.
    - Internal state transition is simplified and deterministic.
    """

    metadata = {"render_modes": []}

    EPISODE_STEPS = 24
    STEP_HOURS = 1.0

    def __init__(
        self,
        net3_inp_path: str | Path = "networks/Net3.inp",
        *,
        delta_time: float = 0.10,
        delta_space: float = 0.10,
        non_randomized_node_ids: Sequence[str] | None = None,
        scaling_mode: ScalingMode = "max_min",
        r_benchmark: float = 2000.0,
        p_hydraulic: float = -200.0,
        tank_penalty_config: TankPenaltyConfig | None = None,
        pump_flow_coeffs: tuple[float, float] = (1700.0, 1700.0),
        pump_power_coeffs: tuple[float, float] = (120.0, 120.0),
        tank_dynamics_gain: float = 2.5,
        demand_std_time: float | None = None,
        demand_std_space: float | None = None,
    ) -> None:
        super().__init__()

        if not (0.0 <= delta_time < 1.0):
            raise ValueError(f"delta_time must be in [0,1), got {delta_time}.")
        if not (0.0 <= delta_space < 1.0):
            raise ValueError(f"delta_space must be in [0,1), got {delta_space}.")
        if scaling_mode not in ("none", "max_min", "z_score"):
            raise ValueError(f"Unsupported scaling_mode: {scaling_mode}.")
        if len(pump_flow_coeffs) != 2 or len(pump_power_coeffs) != 2:
            raise ValueError("pump_flow_coeffs and pump_power_coeffs must each have length 2.")

        self.net3_inp_path = Path(net3_inp_path)
        self.delta_time = float(delta_time)
        self.delta_space = float(delta_space)
        self.scaling_mode = scaling_mode
        self.r_benchmark = float(r_benchmark)
        self.p_hydraulic = float(p_hydraulic)
        self.tank_penalty_config = tank_penalty_config or TankPenaltyConfig(
            mode="proportional",
            proportional_coefficient=-1.0,
        )
        self.pump_flow_coeffs = np.asarray(pump_flow_coeffs, dtype=np.float64)
        self.pump_power_coeffs = np.asarray(pump_power_coeffs, dtype=np.float64)
        self.tank_dynamics_gain = float(tank_dynamics_gain)
        self.demand_std_time = demand_std_time
        self.demand_std_space = demand_std_space

        parsed = self._parse_net3_inp(self.net3_inp_path, horizon_steps=self.EPISODE_STEPS)
        self._demand_node_ids = parsed.demand_node_ids
        self._base_demands = parsed.base_demands
        self._default_pattern = parsed.default_pattern
        self._tank_ids = parsed.tank_ids
        self._tank_init_levels = parsed.tank_init_levels
        self._tank_min_levels = parsed.tank_min_levels
        self._tank_max_levels = parsed.tank_max_levels
        self._tank_diameters = parsed.tank_diameters
        self._tank_min_volumes = parsed.tank_min_volumes
        self._tank_areas = np.pi * (self._tank_diameters / 2.0) ** 2

        self._randomizable_mask = self._build_randomizable_mask(non_randomized_node_ids)
        self._prepare_scaling_statistics()

        self._obs_dim = self._base_demands.size + self._tank_ids.__len__()
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = self._build_observation_space()

        # Runtime state (set in reset()).
        self._rng: np.random.Generator | None = None
        self._step_index = 0
        self._done = False
        self._current_demands = np.zeros((self.EPISODE_STEPS, self._base_demands.size), dtype=np.float64)
        self._time_multipliers = np.ones(self.EPISODE_STEPS, dtype=np.float64)
        self._space_multipliers = np.ones(self._base_demands.size, dtype=np.float64)
        self._tank_levels = self._tank_init_levels.copy()
        self._initial_tank_volume = self._compute_total_tank_volume(self._tank_levels)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[NDArray[np.float32], dict]:
        """Reset environment and sample a new randomized demand trajectory."""

        super().reset(seed=seed)
        self._rng = self.np_random
        self._step_index = 0
        self._done = False

        demands, time_mul, space_mul = generate_randomized_demands(
            base_demands=self._base_demands,
            default_pattern=self._default_pattern,
            delta_time=self.delta_time,
            delta_space=self.delta_space,
            rng=self._rng,
            std_time=self.demand_std_time,
            std_space=self.demand_std_space,
            randomizable_mask=self._randomizable_mask,
        )
        self._current_demands = demands
        self._time_multipliers = time_mul
        self._space_multipliers = space_mul

        # Requirement: random initial tank levels at each reset.
        self._tank_levels = self._rng.uniform(low=self._tank_min_levels, high=self._tank_max_levels)
        self._initial_tank_volume = self._compute_total_tank_volume(self._tank_levels)

        obs = self._get_obs()
        info = {
            "step_index": self._step_index,
            "time_multipliers": self._time_multipliers.copy(),
            "space_multipliers": self._space_multipliers.copy(),
        }
        return obs, info

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Advance one hourly step with the selected discrete action."""

        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {ACTION_COUNT - 1}].")

        current_demands = self._current_demands[self._step_index]
        pump_speeds = action_id_to_speeds(int(action))
        e_pump_t = self._compute_pump_energy_kwh(pump_speeds)

        self._apply_tank_dynamics(pump_speeds, current_demands)
        final_tank_volume = self._compute_total_tank_volume(self._tank_levels)
        hydraulic_violation = self._check_hydraulic_violation(self._tank_levels)

        reward_result = compute_total_reward(
            StepRewardInput(
                t=self._step_index,
                e_pump_t=e_pump_t,
                r_benchmark=self.r_benchmark,
                hydraulic_violation=hydraulic_violation,
                p_hydraulic=self.p_hydraulic,
                initial_tank_volume=self._initial_tank_volume,
                final_tank_volume=final_tank_volume,
            ),
            tank_penalty_config=self.tank_penalty_config,
            horizon_steps=self.EPISODE_STEPS,
            final_step=self.EPISODE_STEPS - 1,
        )

        terminated = bool(reward_result.terminated)
        self._step_index += 1
        truncated = bool((self._step_index >= self.EPISODE_STEPS) and not terminated)
        self._done = bool(terminated or truncated)

        obs = self._get_obs()
        info = {
            "step_index": self._step_index,
            "pump_speeds": pump_speeds,
            "e_pump_t": e_pump_t,
            "total_demand": float(np.sum(current_demands)),
            "base_reward": reward_result.base_reward,
            "tank_penalty": reward_result.tank_penalty,
            "hydraulic_violation": hydraulic_violation,
            "tank_levels": self._tank_levels.copy(),
        }
        return obs, float(reward_result.reward), terminated, truncated, info

    def _get_obs(self) -> NDArray[np.float32]:
        """Build observation = [node_demands, tank_levels]."""

        if self._step_index >= self.EPISODE_STEPS:
            demands = np.zeros(self._base_demands.size, dtype=np.float64)
        else:
            demands = self._current_demands[self._step_index]

        tank_levels = self._tank_levels

        if self.scaling_mode == "none":
            demand_obs = demands
            tank_obs = tank_levels
        elif self.scaling_mode == "max_min":
            demand_obs = demand_max_min_scale(
                demands,
                self._demand_min_bounds,
                self._demand_max_bounds,
                feature_range=(0.0, 1.0),
                clip=True,
            )
            tank_obs = tank_max_min_scale(
                tank_levels,
                self._tank_min_levels,
                self._tank_max_levels,
                feature_range=(0.0, 1.0),
                clip=True,
            )
        else:  # self.scaling_mode == "z_score"
            demand_obs = demand_z_score_scale(
                demands,
                self._demand_mean,
                self._demand_std,
                zero_std_value=0.0,
            )
            tank_obs = tank_z_score_scale(
                tank_levels,
                self._tank_mean,
                self._tank_std,
                zero_std_value=0.0,
            )

        obs = np.concatenate([demand_obs, tank_obs], axis=0)
        return obs.astype(np.float32, copy=False)

    def _compute_pump_energy_kwh(self, pump_speeds: tuple[float, float]) -> float:
        """Compute one-step pump energy proxy (kWh)."""

        speeds = np.asarray(pump_speeds, dtype=np.float64)
        return float(self.STEP_HOURS * np.dot(self.pump_power_coeffs, np.power(speeds, 3)))

    def _apply_tank_dynamics(
        self,
        pump_speeds: tuple[float, float],
        current_demands: NDArray[np.float64],
    ) -> None:
        """Apply a minimal tank level transition for one hourly step."""

        inflow = float(np.dot(self.pump_flow_coeffs, np.asarray(pump_speeds, dtype=np.float64)))
        outflow = float(np.sum(current_demands))
        net_flow = inflow - outflow

        total_area = float(np.sum(self._tank_areas))
        if total_area <= 0:
            raise ValueError("Invalid tank area configuration: total area must be positive.")

        # Shared level change across tanks; intentionally simple for MVP environment.
        delta_level = self.tank_dynamics_gain * net_flow / total_area
        self._tank_levels = self._tank_levels + delta_level

    def _compute_total_tank_volume(self, tank_levels: NDArray[np.float64]) -> float:
        """Compute total storage volume proxy from tank levels."""

        effective_height = np.maximum(tank_levels - self._tank_min_levels, 0.0)
        volumes = self._tank_min_volumes + self._tank_areas * effective_height
        return float(np.sum(volumes))

    def _check_hydraulic_violation(self, tank_levels: NDArray[np.float64]) -> bool:
        """Check simple hydraulic violation: any tank level outside [min, max]."""

        below = np.any(tank_levels < self._tank_min_levels)
        above = np.any(tank_levels > self._tank_max_levels)
        return bool(below or above)

    def _build_randomizable_mask(
        self,
        non_randomized_node_ids: Sequence[str] | None,
    ) -> NDArray[np.bool_]:
        """Build mask for demand randomization (False => not randomized)."""

        if non_randomized_node_ids is None:
            return np.ones(self._base_demands.size, dtype=bool)

        fixed_ids = {str(node_id) for node_id in non_randomized_node_ids}
        return np.asarray(
            [node_id not in fixed_ids for node_id in self._demand_node_ids],
            dtype=bool,
        )

    def _prepare_scaling_statistics(self) -> None:
        """Pre-compute bounds and moments for observation scaling."""

        pattern_min = float(np.min(self._default_pattern))
        pattern_max = float(np.max(self._default_pattern))

        node_delta = np.where(self._randomizable_mask, self.delta_space, 0.0)
        demand_min_factor = pattern_min * (1.0 - self.delta_time) * (1.0 - node_delta)
        demand_max_factor = pattern_max * (1.0 + self.delta_time) * (1.0 + node_delta)

        self._demand_min_bounds = self._base_demands * demand_min_factor
        self._demand_max_bounds = self._base_demands * demand_max_factor
        self._demand_mean = (self._demand_min_bounds + self._demand_max_bounds) / 2.0
        self._demand_std = (self._demand_max_bounds - self._demand_min_bounds) / 6.0

        self._tank_mean = (self._tank_min_levels + self._tank_max_levels) / 2.0
        self._tank_std = (self._tank_max_levels - self._tank_min_levels) / 6.0

    def _build_observation_space(self) -> spaces.Box:
        """Create observation space according to selected scaling mode."""

        if self.scaling_mode == "max_min":
            low = np.zeros(self._obs_dim, dtype=np.float32)
            high = np.ones(self._obs_dim, dtype=np.float32)
            return spaces.Box(low=low, high=high, dtype=np.float32)

        if self.scaling_mode == "z_score":
            low = np.full(self._obs_dim, -np.inf, dtype=np.float32)
            high = np.full(self._obs_dim, np.inf, dtype=np.float32)
            return spaces.Box(low=low, high=high, dtype=np.float32)

        low = np.concatenate(
            [np.zeros(self._base_demands.size, dtype=np.float64), self._tank_min_levels],
            axis=0,
        ).astype(np.float32)
        high = np.concatenate([self._demand_max_bounds, self._tank_max_levels], axis=0).astype(np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def _parse_net3_inp(inp_path: Path, *, horizon_steps: int) -> _ParsedNet3Data:
        """Parse only data required by this minimal environment from Net3 INP."""

        if not inp_path.exists():
            raise FileNotFoundError(f"Net3 INP file not found: {inp_path}")

        lines = inp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        section_re = re.compile(r"^\s*\[([^\]]+)\]\s*$")
        section = ""

        demand_node_ids: list[str] = []
        base_demands: list[float] = []
        tank_ids: list[str] = []
        tank_init_levels: list[float] = []
        tank_min_levels: list[float] = []
        tank_max_levels: list[float] = []
        tank_diameters: list[float] = []
        tank_min_volumes: list[float] = []

        patterns: dict[str, list[float]] = {}
        default_pattern_id = "1"

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue

            section_match = section_re.match(stripped)
            if section_match:
                section = section_match.group(1).strip().upper()
                continue

            if stripped.startswith(";"):
                continue

            parts = stripped.split()
            if section == "JUNCTIONS":
                if len(parts) >= 3:
                    demand_node_ids.append(parts[0])
                    try:
                        base_demands.append(float(parts[2]))
                    except ValueError:
                        base_demands.append(0.0)
                continue

            if section == "TANKS":
                if len(parts) >= 6:
                    tank_ids.append(parts[0])
                    tank_init_levels.append(float(parts[2]))
                    tank_min_levels.append(float(parts[3]))
                    tank_max_levels.append(float(parts[4]))
                    tank_diameters.append(float(parts[5]))
                    tank_min_volumes.append(float(parts[6]) if len(parts) >= 7 else 0.0)
                continue

            if section == "PATTERNS":
                if len(parts) >= 2:
                    pid = parts[0]
                    values: list[float] = []
                    for token in parts[1:]:
                        try:
                            values.append(float(token))
                        except ValueError:
                            continue
                    patterns.setdefault(pid, []).extend(values)
                continue

            if section == "OPTIONS":
                if len(parts) >= 2 and parts[0].upper() == "PATTERN":
                    default_pattern_id = parts[1]

        if not demand_node_ids:
            raise ValueError("No junction demand data found in Net3 INP.")
        if not tank_ids:
            raise ValueError("No tank data found in Net3 INP.")

        default_pattern = Net3PumpSchedulingEnv._build_default_pattern(
            pattern_values=patterns.get(default_pattern_id, []),
            horizon_steps=horizon_steps,
        )

        return _ParsedNet3Data(
            demand_node_ids=tuple(demand_node_ids),
            base_demands=np.asarray(base_demands, dtype=np.float64),
            default_pattern=default_pattern,
            tank_ids=tuple(tank_ids),
            tank_init_levels=np.asarray(tank_init_levels, dtype=np.float64),
            tank_min_levels=np.asarray(tank_min_levels, dtype=np.float64),
            tank_max_levels=np.asarray(tank_max_levels, dtype=np.float64),
            tank_diameters=np.asarray(tank_diameters, dtype=np.float64),
            tank_min_volumes=np.asarray(tank_min_volumes, dtype=np.float64),
        )

    @staticmethod
    def _build_default_pattern(pattern_values: Sequence[float], *, horizon_steps: int) -> NDArray[np.float64]:
        """Build a horizon-length default demand pattern from INP pattern values."""

        if horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be positive, got {horizon_steps}.")

        if not pattern_values:
            return np.ones(horizon_steps, dtype=np.float64)

        base = np.asarray(pattern_values, dtype=np.float64).reshape(-1)
        if base.size >= horizon_steps:
            return base[:horizon_steps].astype(np.float64, copy=False)

        repeats = int(np.ceil(horizon_steps / base.size))
        return np.tile(base, repeats)[:horizon_steps].astype(np.float64, copy=False)

    def render(self) -> None:
        """No-op render for minimal environment."""

        return None

    def close(self) -> None:
        """No-op close for minimal environment."""

        return None


__all__ = ["Net3PumpSchedulingEnv", "ScalingMode"]
