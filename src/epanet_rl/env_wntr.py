"""WNTR/EPANET single-step environment for Net3 pump scheduling.

This environment is separate from the minimal proxy env in ``env.py``.
It performs a real hydraulic simulation each step using WNTR + EPANET toolkit.

Key behaviors:
- 24-step episode, 1 hour per step.
- Action sets two pump speeds (discrete action id from action_space.py).
- Each step updates current-hour junction demands and tank initial levels.
- Runs one-hour EPANET hydraulic simulation and extracts results.
- Reward is computed via reward.py.
- Observation = [junction demands, tank levels].

Current scope:
- r_benchmark can be hard-coded (development placeholder).
- Z-score scaling interface is provided, but not calibrated by 10k rollouts.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .action_space import ACTION_COUNT, action_id_to_speeds
from .demand_randomization import generate_randomized_demands
from .inp_modifier import OFFPEAK_PRICE_USD_PER_KWH, PEAK_PRICE_USD_PER_KWH, modify_inp_text
from .reward import StepRewardInput, TankPenaltyConfig, compute_total_reward
from .scaling import (
    demand_max_min_scale,
    demand_z_score_scale,
    tank_max_min_scale,
    tank_z_score_scale,
)

try:
    import wntr
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    wntr = None
    _WNTR_IMPORT_ERROR = exc
else:  # pragma: no cover - only typing/runtime helper
    _WNTR_IMPORT_ERROR = None

ScalingMode = Literal["none", "max_min", "z_score"]


@dataclass(frozen=True)
class _Net3Meta:
    junction_names: tuple[str, ...]
    base_demands: NDArray[np.float64]
    default_pattern: NDArray[np.float64]
    tank_names: tuple[str, ...]
    tank_elevations: NDArray[np.float64]
    tank_min_levels: NDArray[np.float64]
    tank_max_levels: NDArray[np.float64]
    tank_init_levels: NDArray[np.float64]
    tank_diameters: NDArray[np.float64]
    tank_min_volumes: NDArray[np.float64]
    pump_names: tuple[str, ...]


class Net3WntrEnv(gym.Env[NDArray[np.float32], int]):
    """Real single-step hydraulic environment backed by WNTR/EPANET."""

    metadata = {"render_modes": []}

    EPISODE_STEPS = 24
    STEP_HOURS = 1.0
    STEP_SECONDS = 3600
    WATER_DENSITY_KG_PER_M3 = 1000.0
    GRAVITY_M_PER_S2 = 9.81
    DEFAULT_TANK_LEVEL_EPSILON = 1e-4

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
        pressure_violation_threshold: float = 0.0,
        pump_efficiency: float = 0.75,
        tank_level_epsilon: float = DEFAULT_TANK_LEVEL_EPSILON,
        enable_nan_fallback: bool = True,
        tank_penalty_config: TankPenaltyConfig | None = None,
        demand_std_time: float | None = None,
        demand_std_space: float | None = None,
    ) -> None:
        super().__init__()

        if wntr is None:
            raise ModuleNotFoundError(
                "wntr is required for Net3WntrEnv but is not installed in the current environment."
            ) from _WNTR_IMPORT_ERROR

        if not (0.0 <= delta_time < 1.0):
            raise ValueError(f"delta_time must be in [0,1), got {delta_time}.")
        if not (0.0 <= delta_space < 1.0):
            raise ValueError(f"delta_space must be in [0,1), got {delta_space}.")
        if scaling_mode not in ("none", "max_min", "z_score"):
            raise ValueError(f"Unsupported scaling_mode: {scaling_mode}.")
        if pump_efficiency <= 0.0:
            raise ValueError(f"pump_efficiency must be positive, got {pump_efficiency}.")
        if tank_level_epsilon < 0.0:
            raise ValueError(f"tank_level_epsilon must be >= 0, got {tank_level_epsilon}.")

        self.net3_inp_path = Path(net3_inp_path)
        self.delta_time = float(delta_time)
        self.delta_space = float(delta_space)
        self.scaling_mode = scaling_mode
        self.r_benchmark = float(r_benchmark)
        self.p_hydraulic = float(p_hydraulic)
        self.pressure_violation_threshold = float(pressure_violation_threshold)
        self.pump_efficiency = float(pump_efficiency)
        self.tank_level_epsilon = float(tank_level_epsilon)
        self.enable_nan_fallback = bool(enable_nan_fallback)
        self.tank_penalty_config = tank_penalty_config or TankPenaltyConfig(
            mode="proportional",
            proportional_coefficient=-1.0,
        )
        self.demand_std_time = demand_std_time
        self.demand_std_space = demand_std_space

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="epanet_rl_wntr_"))
        self._modified_inp_path = self._tmp_dir / "net3_modified_for_wntr.inp"
        self._prepare_modified_inp()

        base_wn = wntr.network.WaterNetworkModel(str(self._modified_inp_path))
        self._meta = self._extract_meta(base_wn, horizon_steps=self.EPISODE_STEPS)

        if len(self._meta.pump_names) != 2:
            raise ValueError(
                f"Net3WntrEnv expects exactly 2 pumps, got {len(self._meta.pump_names)}: {self._meta.pump_names}"
            )

        self._junction_names = self._meta.junction_names
        self._base_demands = self._meta.base_demands
        self._default_pattern = self._meta.default_pattern
        self._tank_names = self._meta.tank_names
        self._tank_elevations = self._meta.tank_elevations
        self._tank_min_levels = self._meta.tank_min_levels
        self._tank_max_levels = self._meta.tank_max_levels
        self._tank_init_levels = self._meta.tank_init_levels
        self._tank_diameters = self._meta.tank_diameters
        self._tank_min_volumes = self._meta.tank_min_volumes
        self._pump_names = self._meta.pump_names
        self._tank_areas = np.pi * (self._tank_diameters / 2.0) ** 2

        self._randomizable_mask = self._build_randomizable_mask(non_randomized_node_ids)
        self._prepare_scaling_statistics()

        self._obs_dim = self._base_demands.size + self._tank_names.__len__()
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
        """Reset episode and sample randomized hourly demand trajectory."""

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

        # Initial tank levels are uniformly sampled in [min_level, max_level].
        self._tank_levels = self._rng.uniform(low=self._tank_min_levels, high=self._tank_max_levels)
        self._initial_tank_volume = self._compute_total_tank_volume(self._tank_levels)

        obs = self._get_obs()
        info = {
            "step_index": self._step_index,
            "time_multipliers": self._time_multipliers.copy(),
            "space_multipliers": self._space_multipliers.copy(),
            "pump_names": self._pump_names,
            "tank_names": self._tank_names,
        }
        return obs, info

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Run one-hour WNTR/EPANET simulation for the current action."""

        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {ACTION_COUNT - 1}].")

        original_action = int(action)
        current_demands = self._current_demands[self._step_index]
        commanded_pump_speeds = action_id_to_speeds(original_action)
        sim_tank_levels, tank_level_clipped = self._sanitize_tank_levels_for_simulation(self._tank_levels)

        executed_pump_speeds = commanded_pump_speeds
        fallback_used = False
        original_action_failed = False
        sim_failure_caught = False

        try:
            sim_output = self._simulate_single_step(
                executed_pump_speeds,
                current_demands,
                sim_tank_levels,
            )
            if self._sim_output_has_nan(sim_output):
                original_action_failed = True
                if self.enable_nan_fallback:
                    fallback_used = True
                    executed_pump_speeds = (0.0, 0.0)
                    sim_output = self._simulate_single_step(
                        executed_pump_speeds,
                        current_demands,
                        sim_tank_levels,
                    )

            self._tank_levels = sim_output["tank_levels"]
            final_tank_volume = self._compute_total_tank_volume(self._tank_levels)
            hydraulic_violation = bool(sim_output["hydraulic_violation"])
            e_pump_t = float(sim_output["pump_energy_cost"])
            min_pressure = float(sim_output["min_pressure"])
            pump_flows = sim_output["pump_flows"]
            pump_heads = sim_output["pump_head_gains"]
        except Exception:
            original_action_failed = True
            if self.enable_nan_fallback:
                fallback_used = True
                executed_pump_speeds = (0.0, 0.0)
                try:
                    sim_output = self._simulate_single_step(
                        executed_pump_speeds,
                        current_demands,
                        sim_tank_levels,
                    )
                    if self._sim_output_has_nan(sim_output):
                        raise RuntimeError("Fallback simulation produced NaN outputs.")

                    self._tank_levels = sim_output["tank_levels"]
                    final_tank_volume = self._compute_total_tank_volume(self._tank_levels)
                    hydraulic_violation = bool(sim_output["hydraulic_violation"])
                    e_pump_t = float(sim_output["pump_energy_cost"])
                    min_pressure = float(sim_output["min_pressure"])
                    pump_flows = sim_output["pump_flows"]
                    pump_heads = sim_output["pump_head_gains"]
                except Exception:
                    sim_failure_caught = True
                    self._tank_levels = sim_tank_levels.copy()
                    final_tank_volume = self._compute_total_tank_volume(self._tank_levels)
                    hydraulic_violation = True
                    e_pump_t = 0.0
                    min_pressure = float("nan")
                    pump_flows = np.full(2, np.nan, dtype=np.float64)
                    pump_heads = np.full(2, np.nan, dtype=np.float64)
            else:
                # Any simulation/runtime failure is treated as a hydraulic violation for RL step.
                sim_failure_caught = True
                self._tank_levels = sim_tank_levels.copy()
                final_tank_volume = self._compute_total_tank_volume(self._tank_levels)
                hydraulic_violation = True
                e_pump_t = 0.0
                min_pressure = float("nan")
                pump_flows = np.full(2, np.nan, dtype=np.float64)
                pump_heads = np.full(2, np.nan, dtype=np.float64)

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
            "pump_names": self._pump_names,
            "pump_speeds": executed_pump_speeds,
            "pump_speeds_commanded": commanded_pump_speeds,
            "pump_speeds_applied": executed_pump_speeds,
            "original_action": original_action,
            "executed_pump_speeds": executed_pump_speeds,
            "fallback_used": fallback_used,
            "original_action_failed": original_action_failed,
            "pump_flows": pump_flows.copy(),
            "pump_head_gains": pump_heads.copy(),
            "e_pump_t": e_pump_t,
            "base_reward": reward_result.base_reward,
            "tank_penalty": reward_result.tank_penalty,
            "hydraulic_violation": hydraulic_violation,
            "min_pressure": min_pressure,
            "tank_levels": self._tank_levels.copy(),
            "tank_levels_for_sim": sim_tank_levels.copy(),
            "tank_level_clipped_before_sim": tank_level_clipped,
            "sim_failure_caught": sim_failure_caught,
            "initial_tank_volume": self._initial_tank_volume,
            "final_tank_volume": final_tank_volume,
        }
        return obs, float(reward_result.reward), terminated, truncated, info

    def _simulate_single_step(
        self,
        pump_speeds: tuple[float, float],
        current_demands: NDArray[np.float64],
        tank_init_levels: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64] | float | bool]:
        """Simulate one hydraulic hour and return extracted outputs."""

        wn = wntr.network.WaterNetworkModel(str(self._modified_inp_path))
        self._configure_one_hour_options(wn)
        self._apply_tank_initial_levels(wn, tank_init_levels)
        self._apply_hourly_demands(wn, current_demands)
        self._apply_pump_speeds(wn, pump_speeds)

        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        end_time = int(results.node["head"].index.max())

        tank_heads = results.node["head"].loc[end_time, list(self._tank_names)].to_numpy(dtype=np.float64)
        tank_levels = tank_heads - self._tank_elevations

        junction_pressures = results.node["pressure"].loc[end_time, list(self._junction_names)].to_numpy(dtype=np.float64)
        min_pressure = float(np.nanmin(junction_pressures))
        hydraulic_violation = bool(np.any(np.isnan(junction_pressures)) or (min_pressure < self.pressure_violation_threshold))

        pump_flows = np.abs(results.link["flowrate"].loc[end_time, list(self._pump_names)].to_numpy(dtype=np.float64))
        pump_headloss = results.link["headloss"].loc[end_time, list(self._pump_names)].to_numpy(dtype=np.float64)
        pump_head_gains = np.maximum(0.0, -pump_headloss)
        pump_energy_cost = self._compute_pump_energy_cost(pump_flows, pump_head_gains, hour=self._step_index)

        return {
            "tank_levels": tank_levels,
            "hydraulic_violation": hydraulic_violation,
            "min_pressure": min_pressure,
            "pump_flows": pump_flows,
            "pump_head_gains": pump_head_gains,
            "pump_energy_cost": pump_energy_cost,
        }

    def _sanitize_tank_levels_for_simulation(
        self,
        tank_levels: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], bool]:
        """Clip tank levels away from exact bounds for solver stability."""

        lower = self._tank_min_levels + self.tank_level_epsilon
        upper = self._tank_max_levels - self.tank_level_epsilon

        # If a tank has very narrow valid range, use the midpoint as stable value.
        invalid = upper <= lower
        if np.any(invalid):
            midpoint = (self._tank_min_levels + self._tank_max_levels) / 2.0
            lower = np.where(invalid, midpoint, lower)
            upper = np.where(invalid, midpoint, upper)

        clipped = np.clip(tank_levels, lower, upper)
        changed = bool(np.any(np.abs(clipped - tank_levels) > 1e-12))
        return clipped.astype(np.float64, copy=False), changed

    @staticmethod
    def _sim_output_has_nan(sim_output: dict[str, NDArray[np.float64] | float | bool]) -> bool:
        tank_levels = np.asarray(sim_output["tank_levels"], dtype=np.float64)
        pump_flows = np.asarray(sim_output["pump_flows"], dtype=np.float64)
        pump_heads = np.asarray(sim_output["pump_head_gains"], dtype=np.float64)
        min_pressure = float(sim_output["min_pressure"])
        return bool(
            np.isnan(tank_levels).any()
            or np.isnan(pump_flows).any()
            or np.isnan(pump_heads).any()
            or np.isnan(min_pressure)
        )

    def _configure_one_hour_options(self, wn) -> None:
        wn.options.time.duration = self.STEP_SECONDS
        wn.options.time.hydraulic_timestep = self.STEP_SECONDS
        wn.options.time.report_timestep = self.STEP_SECONDS
        wn.options.time.pattern_timestep = self.STEP_SECONDS

    def _apply_tank_initial_levels(self, wn, tank_levels: NDArray[np.float64]) -> None:
        for tank_name, level in zip(self._tank_names, tank_levels):
            tank = wn.get_node(tank_name)
            tank.init_level = float(level)

    def _apply_hourly_demands(self, wn, current_demands: NDArray[np.float64]) -> None:
        for junction_name, demand_value in zip(self._junction_names, current_demands):
            junction = wn.get_node(junction_name)
            if len(junction.demand_timeseries_list) == 0:
                continue
            junction.demand_timeseries_list[0].base_value = float(demand_value)
            junction.demand_timeseries_list[0].pattern_name = None
            for extra in junction.demand_timeseries_list[1:]:
                extra.base_value = 0.0
                extra.pattern_name = None

    def _apply_pump_speeds(self, wn, pump_speeds: tuple[float, float]) -> None:
        for pump_name, speed in zip(self._pump_names, pump_speeds):
            pump = wn.get_link(pump_name)
            pump.base_speed = float(speed)
            pump.speed_pattern_name = None
            pump.initial_status = "OPEN" if speed > 0.0 else "CLOSED"

    def _compute_pump_energy_cost(
        self,
        pump_flows_m3s: NDArray[np.float64],
        pump_head_gains_m: NDArray[np.float64],
        *,
        hour: int,
    ) -> float:
        """Compute one-step pump electricity cost from simulated hydraulic outputs."""

        power_w = (
            self.WATER_DENSITY_KG_PER_M3
            * self.GRAVITY_M_PER_S2
            * pump_flows_m3s
            * pump_head_gains_m
            / self.pump_efficiency
        )
        energy_kwh = power_w * self.STEP_SECONDS / 3_600_000.0
        unit_price = self._electricity_price_for_hour(hour)
        return float(np.sum(energy_kwh) * unit_price)

    @staticmethod
    def _electricity_price_for_hour(hour: int) -> float:
        return PEAK_PRICE_USD_PER_KWH if 7 <= hour < 23 else OFFPEAK_PRICE_USD_PER_KWH

    def _get_obs(self) -> NDArray[np.float32]:
        """Build observation = [current-hour demands, current tank levels]."""

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

    def _compute_total_tank_volume(self, tank_levels: NDArray[np.float64]) -> float:
        effective_height = np.maximum(tank_levels - self._tank_min_levels, 0.0)
        volumes = self._tank_min_volumes + self._tank_areas * effective_height
        return float(np.sum(volumes))

    def _build_randomizable_mask(self, non_randomized_node_ids: Sequence[str] | None) -> NDArray[np.bool_]:
        if non_randomized_node_ids is None:
            return np.ones(self._base_demands.size, dtype=bool)
        fixed_ids = {str(node_id) for node_id in non_randomized_node_ids}
        return np.asarray([node_id not in fixed_ids for node_id in self._junction_names], dtype=bool)

    def _prepare_scaling_statistics(self) -> None:
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

    def _prepare_modified_inp(self) -> None:
        if not self.net3_inp_path.exists():
            raise FileNotFoundError(f"Net3 INP file not found: {self.net3_inp_path}")
        original_text = self.net3_inp_path.read_text(encoding="utf-8")
        modified_text = modify_inp_text(original_text)
        self._modified_inp_path.write_text(modified_text, encoding="utf-8")

    @staticmethod
    def _extract_meta(base_wn, *, horizon_steps: int) -> _Net3Meta:
        junction_names = tuple(base_wn.junction_name_list)
        base_demands = np.asarray([base_wn.get_node(j).base_demand for j in junction_names], dtype=np.float64)

        pattern_name = base_wn.options.hydraulic.pattern
        if pattern_name is None:
            default_pattern = np.ones(horizon_steps, dtype=np.float64)
        else:
            pattern_obj = base_wn.get_pattern(pattern_name)
            multipliers = np.asarray(pattern_obj.multipliers, dtype=np.float64).reshape(-1)
            if multipliers.size == 0:
                default_pattern = np.ones(horizon_steps, dtype=np.float64)
            elif multipliers.size >= horizon_steps:
                default_pattern = multipliers[:horizon_steps]
            else:
                repeats = int(np.ceil(horizon_steps / multipliers.size))
                default_pattern = np.tile(multipliers, repeats)[:horizon_steps]

        tank_names = tuple(base_wn.tank_name_list)
        tank_elevations = np.asarray([base_wn.get_node(t).elevation for t in tank_names], dtype=np.float64)
        tank_min_levels = np.asarray([base_wn.get_node(t).min_level for t in tank_names], dtype=np.float64)
        tank_max_levels = np.asarray([base_wn.get_node(t).max_level for t in tank_names], dtype=np.float64)
        tank_init_levels = np.asarray([base_wn.get_node(t).init_level for t in tank_names], dtype=np.float64)
        tank_diameters = np.asarray([base_wn.get_node(t).diameter for t in tank_names], dtype=np.float64)
        tank_min_volumes = np.asarray([base_wn.get_node(t).min_vol for t in tank_names], dtype=np.float64)

        pump_names = tuple(base_wn.pump_name_list)

        return _Net3Meta(
            junction_names=junction_names,
            base_demands=base_demands,
            default_pattern=default_pattern.astype(np.float64, copy=False),
            tank_names=tank_names,
            tank_elevations=tank_elevations,
            tank_min_levels=tank_min_levels,
            tank_max_levels=tank_max_levels,
            tank_init_levels=tank_init_levels,
            tank_diameters=tank_diameters,
            tank_min_volumes=tank_min_volumes,
            pump_names=pump_names,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        if hasattr(self, "_tmp_dir") and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


__all__ = ["Net3WntrEnv", "ScalingMode"]
