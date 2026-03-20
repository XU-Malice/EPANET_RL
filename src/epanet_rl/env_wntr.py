"""Net3 的 WNTR/EPANET 主线环境（论文复现核心）。

教学导读（先看这里）：
1. 论文明确给出的：
   - 一个 episode = 24 步，每步 1 小时；
   - 状态由 demand + tank levels 组成；
   - 水力违例需要给大惩罚并提前终止。
2. 当前仓库实现：
   - 使用 WNTR/EPANET 做真实单步仿真；
   - 在 `step()` 的 `info` 里保留较完整诊断字段；
   - 奖励逻辑统一委托给 `reward.py`。
3. 工程可运行近似：
   - 为提升稳定性，提供 NaN fallback、tank level 裁剪等数值保护分支；
   - 这些保护逻辑用于“训练可跑、易排障”，不改变论文主线语义。
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
    """从 Net3/修改版 INP 中提取出的静态元数据。

    这些字段在一个训练 run 内通常视为“网络常量”：
    - junction/tank/pump 名称用于建立稳定索引；
    - demand/tank 几何参数用于 reset、obs 构造与奖励结算；
    - 之所以集中到 dataclass，是为了把“网络结构信息”和“episode 运行态”清晰分开。
    """
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
    """基于 WNTR/EPANET 的单步真实水力环境。

    输入：
    - 离散动作 `action_id`（0..63），由两台泵速度组合映射得到。

    输出：
    - Gymnasium 标准五元组 `(obs, reward, terminated, truncated, info)`。

    复现关系：
    - 这是论文复现主线环境，不建议用 `env.py` 替代训练主流程。
    """

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
        if p_hydraulic >= 0.0:
            raise ValueError(
                f"p_hydraulic must be negative to represent a large penalty, got {p_hydraulic}."
            )
        if pump_efficiency <= 0.0:
            raise ValueError(f"pump_efficiency must be positive, got {pump_efficiency}.")
        if tank_level_epsilon < 0.0:
            raise ValueError(f"tank_level_epsilon must be >= 0, got {tank_level_epsilon}.")

        # 下面这些参数大多直接对应“论文主线设定”或“当前仓库的工程开关”。
        # 设计上统一在构造函数保存，原因是：训练脚本、评估脚本、诊断脚本都需要显式记录它们，
        # 这样实验日志才能回答“这次 run 到底用了哪套环境口径”。
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

        # 不直接改原始 Net3.inp：先生成临时“修改版 INP”供仿真使用。
        # 这样做的原因：
        # 1) 保持原始网络文件可追溯；
        # 2) 训练期间每次实例化环境都能得到同样的实验底座；
        # 3) 若你要核查论文复现差异，只需比较修改前后 INP diff。
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

        # 运行态（每次 reset 会重置）。
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
        """重置 episode 并采样 24 小时 demand 轨迹。

        关键点：
        - 每次 reset 都会重新采样 demand 时间/空间乘子；
        - 初始 tank level 在 `[min, max]` 内均匀随机；
        - 返回的 `info` 包含乘子与设备名称，便于离线审计。
        """

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

        # 论文明确给出的：初始 tank level 在 [min, max] 内均匀随机。
        # 教学理解：这一步决定了 episode 的“初始库存”，会直接影响后续 24 小时调度难度。
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
        """执行当前动作对应的 1 小时仿真，并返回 Gym step 五元组。

        教学提示：
        - 先做动作解码与 tank level 数值保护，再跑仿真；
        - 奖励与终止语义由 `reward.compute_total_reward` 决定；
        - `info` 字段是后续 benchmark/评估/排障的主要数据来源。
        """

        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {ACTION_COUNT - 1}].")

        # 输入：离散动作编号。这里先保留 `original_action`，因为后续可能因 fallback 改成别的实际执行动作。
        # 这样日志里能同时看到“agent 想做什么”和“环境最后执行了什么”。
        original_action = int(action)
        current_demands = self._current_demands[self._step_index]
        commanded_pump_speeds = action_id_to_speeds(original_action)
        # 在求解前先对 tank level 做轻微裁剪，减少“卡在边界”导致的数值不稳定。
        sim_tank_levels, tank_level_clipped = self._sanitize_tank_levels_for_simulation(self._tank_levels)

        executed_pump_speeds = commanded_pump_speeds
        fallback_used = False
        original_action_failed = False
        sim_failure_caught = False

        try:
            # 主路径：按 agent 给出的动作做 1 小时真实仿真。
            # 若出现 NaN/异常，当前仓库实现可选地退回到 (0, 0) 速度动作。
            # 这是工程近似，不是论文明确逐句写出的策略，因此日志里要保留 `fallback_used`。
            sim_output = self._simulate_single_step(
                executed_pump_speeds,
                current_demands,
                sim_tank_levels,
            )
            # 若原动作仿真输出出现 NaN，可按配置走 fallback（0,0）动作。
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
            energy_cost_mode = str(sim_output.get("energy_cost_mode", "single_point"))
            energy_time_point_count = int(sim_output.get("energy_time_point_count", 1))
            energy_time_span_seconds = sim_output.get("energy_time_span_seconds")
            energy_raw_time_point_count = int(sim_output.get("energy_raw_time_point_count", 0))
            energy_raw_time_index_type = sim_output.get("energy_raw_time_index_type")
            energy_integration_attempted = bool(sim_output.get("energy_integration_attempted", False))
            energy_integration_failure_reason = sim_output.get("energy_integration_failure_reason")
            min_pressure = float(sim_output["min_pressure"])
            min_pressure_end_time = float(sim_output.get("min_pressure_end_time", min_pressure))
            min_pressure_over_step = float(
                sim_output.get("min_pressure_over_step", min_pressure_end_time)
            )
            pressure_time_point_count = int(sim_output.get("pressure_time_point_count", 1))
            hydraulic_violation_end_time_rule = bool(
                sim_output.get("hydraulic_violation_end_time_rule", hydraulic_violation)
            )
            hydraulic_violation_full_step_rule = bool(
                sim_output.get("hydraulic_violation_full_step_rule", hydraulic_violation_end_time_rule)
            )
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
                    energy_cost_mode = str(sim_output.get("energy_cost_mode", "single_point"))
                    energy_time_point_count = int(sim_output.get("energy_time_point_count", 1))
                    energy_time_span_seconds = sim_output.get("energy_time_span_seconds")
                    energy_raw_time_point_count = int(sim_output.get("energy_raw_time_point_count", 0))
                    energy_raw_time_index_type = sim_output.get("energy_raw_time_index_type")
                    energy_integration_attempted = bool(sim_output.get("energy_integration_attempted", False))
                    energy_integration_failure_reason = sim_output.get("energy_integration_failure_reason")
                    min_pressure = float(sim_output["min_pressure"])
                    min_pressure_end_time = float(sim_output.get("min_pressure_end_time", min_pressure))
                    min_pressure_over_step = float(
                        sim_output.get("min_pressure_over_step", min_pressure_end_time)
                    )
                    pressure_time_point_count = int(sim_output.get("pressure_time_point_count", 1))
                    hydraulic_violation_end_time_rule = bool(
                        sim_output.get("hydraulic_violation_end_time_rule", hydraulic_violation)
                    )
                    hydraulic_violation_full_step_rule = bool(
                        sim_output.get("hydraulic_violation_full_step_rule", hydraulic_violation_end_time_rule)
                    )
                    pump_flows = sim_output["pump_flows"]
                    pump_heads = sim_output["pump_head_gains"]
                except Exception:
                    sim_failure_caught = True
                    self._tank_levels = sim_tank_levels.copy()
                    final_tank_volume = self._compute_total_tank_volume(self._tank_levels)
                    hydraulic_violation = True
                    e_pump_t = 0.0
                    energy_cost_mode = "single_point"
                    energy_time_point_count = 1
                    energy_time_span_seconds = None
                    energy_raw_time_point_count = 0
                    energy_raw_time_index_type = None
                    energy_integration_attempted = False
                    energy_integration_failure_reason = "exception:simulate_single_step_failed"
                    min_pressure = float("nan")
                    min_pressure_end_time = float("nan")
                    min_pressure_over_step = float("nan")
                    pressure_time_point_count = 0
                    hydraulic_violation_end_time_rule = True
                    hydraulic_violation_full_step_rule = True
                    pump_flows = np.full(2, np.nan, dtype=np.float64)
                    pump_heads = np.full(2, np.nan, dtype=np.float64)
            else:
                # Any simulation/runtime failure is treated as a hydraulic violation for RL step.
                sim_failure_caught = True
                self._tank_levels = sim_tank_levels.copy()
                final_tank_volume = self._compute_total_tank_volume(self._tank_levels)
                hydraulic_violation = True
                e_pump_t = 0.0
                energy_cost_mode = "single_point"
                energy_time_point_count = 1
                energy_time_span_seconds = None
                energy_raw_time_point_count = 0
                energy_raw_time_index_type = None
                energy_integration_attempted = False
                energy_integration_failure_reason = "exception:simulate_single_step_failed"
                min_pressure = float("nan")
                min_pressure_end_time = float("nan")
                min_pressure_over_step = float("nan")
                pressure_time_point_count = 0
                hydraulic_violation_end_time_rule = True
                hydraulic_violation_full_step_rule = True
                pump_flows = np.full(2, np.nan, dtype=np.float64)
                pump_heads = np.full(2, np.nan, dtype=np.float64)

        # reward 与终止语义完全交给 reward.py，环境这里只负责提供输入。
        # 这样做的教学价值很高：
        # - 你可以把环境理解成“物理仿真器 + 状态机”；
        # - 把奖励理解成“独立结算器”；
        # 两者边界清楚后，复现实验时更容易逐层排错。
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
        # 统一终止原因，便于训练日志与离线统计脚本做归因分析。
        if terminated and hydraulic_violation:
            termination_reason: str | None = "hydraulic_violation"
        elif truncated:
            termination_reason = "horizon_reached"
        else:
            termination_reason = None

        obs = self._get_obs()
        # info 保留较完整诊断信息，方便 benchmark / diagnose / 训练排障复用。
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
            "energy_cost_mode": energy_cost_mode,
            "energy_time_point_count": energy_time_point_count,
            "energy_time_span_seconds": (
                None if energy_time_span_seconds is None else float(energy_time_span_seconds)
            ),
            "energy_raw_time_point_count": energy_raw_time_point_count,
            "energy_raw_time_index_type": energy_raw_time_index_type,
            "energy_integration_attempted": energy_integration_attempted,
            "energy_integration_failure_reason": energy_integration_failure_reason,
            "base_reward": reward_result.base_reward,
            "tank_penalty": reward_result.tank_penalty,
            "hydraulic_violation": hydraulic_violation,
            "min_pressure": min_pressure,
            "min_pressure_end_time": min_pressure_end_time,
            "min_pressure_over_step": min_pressure_over_step,
            "pressure_time_point_count": pressure_time_point_count,
            "hydraulic_violation_end_time_rule": hydraulic_violation_end_time_rule,
            "hydraulic_violation_full_step_rule": hydraulic_violation_full_step_rule,
            "tank_levels": self._tank_levels.copy(),
            "tank_levels_for_sim": sim_tank_levels.copy(),
            "tank_level_clipped_before_sim": tank_level_clipped,
            "sim_failure_caught": sim_failure_caught,
            "initial_tank_volume": self._initial_tank_volume,
            "final_tank_volume": final_tank_volume,
            "termination_reason": termination_reason,
        }
        return obs, float(reward_result.reward), terminated, truncated, info

    def _simulate_single_step(
        self,
        pump_speeds: tuple[float, float],
        current_demands: NDArray[np.float64],
        tank_init_levels: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64] | float | bool]:
        """仿真 1 小时并提取本 step 所需的水力结果。

        输入：
        - `pump_speeds`: 两台泵本步执行速度；
        - `current_demands`: 当前小时各节点 demand；
        - `tank_init_levels`: 本步起始 tank levels。

        返回字典（节选）：
        - `tank_levels`, `hydraulic_violation`, `pump_energy_cost`
        - `pump_flows`, `pump_head_gains`, `min_pressure`
        - 若有诊断逻辑，还会附带 energy/pressure 相关诊断字段。
        """

        # 这里每个 step 都重新从“修改版 INP”构建 WNTR 网络，然后灌入本步状态。
        # 当前仓库实现选择这种方式，是为了让单步仿真边界非常清晰：
        # 输入就是“本小时 demand + 起始 tank level + 泵速度”，输出就是“1 小时后的结果”。
        # 代价是仿真开销更高，但更利于论文复现审计。
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

        pressure_frame = results.node["pressure"].loc[:, list(self._junction_names)]
        pressure_time_point_count = int(len(pressure_frame.index))
        junction_pressures = pressure_frame.loc[end_time].to_numpy(dtype=np.float64)
        all_pressures = pressure_frame.to_numpy(dtype=np.float64)
        min_pressure_end_time = float(np.nanmin(junction_pressures))
        if np.isnan(all_pressures).all():
            min_pressure_over_step = float("nan")
        else:
            min_pressure_over_step = float(np.nanmin(all_pressures))
        hydraulic_violation_end_time_rule = bool(
            np.any(np.isnan(junction_pressures))
            or (min_pressure_end_time < self.pressure_violation_threshold)
        )
        hydraulic_violation_full_step_rule = bool(
            np.any(np.isnan(all_pressures))
            or (min_pressure_over_step < self.pressure_violation_threshold)
        )

        # 当前主逻辑保持不变：违例判定仍以“末时刻压力规则”为准。
        # full-step 压力信息仅作为诊断输出，不参与终止决策。
        hydraulic_violation = hydraulic_violation_end_time_rule
        min_pressure = min_pressure_end_time

        pump_flow_frame = results.link["flowrate"].loc[:, list(self._pump_names)]
        pump_headloss_frame = results.link["headloss"].loc[:, list(self._pump_names)]
        raw_time_index = pump_flow_frame.index
        energy_raw_time_point_count = int(len(raw_time_index))
        raw_time_index_dtype = getattr(raw_time_index, "dtype", None)
        if raw_time_index_dtype is None:
            energy_raw_time_index_type = type(raw_time_index).__name__
        else:
            energy_raw_time_index_type = f"{type(raw_time_index).__name__}:{raw_time_index_dtype}"

        # 诊断输出仍以 step 末时刻的泵工况为主。
        pump_flows = np.abs(pump_flow_frame.loc[end_time].to_numpy(dtype=np.float64))
        pump_headloss = pump_headloss_frame.loc[end_time].to_numpy(dtype=np.float64)
        pump_head_gains = np.maximum(0.0, -pump_headloss)

        # 当前能耗口径：single_point（沿用稳健的一点估算，避免引入额外求积风险）。
        pump_energy_cost = self._compute_pump_energy_cost(
            pump_flows,
            pump_head_gains,
            hour=self._step_index,
        )
        energy_cost_mode = "single_point"
        energy_time_point_count = 1
        energy_time_span_seconds: float | None = None
        energy_integration_attempted = False
        energy_integration_failure_reason: str | None = "none"

        return {
            "tank_levels": tank_levels,
            "hydraulic_violation": hydraulic_violation,
            "min_pressure": min_pressure,
            "min_pressure_end_time": min_pressure_end_time,
            "min_pressure_over_step": min_pressure_over_step,
            "pressure_time_point_count": pressure_time_point_count,
            "hydraulic_violation_end_time_rule": hydraulic_violation_end_time_rule,
            "hydraulic_violation_full_step_rule": hydraulic_violation_full_step_rule,
            "pump_flows": pump_flows,
            "pump_head_gains": pump_head_gains,
            "pump_energy_cost": pump_energy_cost,
            "energy_cost_mode": energy_cost_mode,
            "energy_time_point_count": energy_time_point_count,
            "energy_time_span_seconds": energy_time_span_seconds,
            "energy_raw_time_point_count": energy_raw_time_point_count,
            "energy_raw_time_index_type": energy_raw_time_index_type,
            "energy_integration_attempted": energy_integration_attempted,
            "energy_integration_failure_reason": energy_integration_failure_reason,
        }

    def _sanitize_tank_levels_for_simulation(
        self,
        tank_levels: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], bool]:
        """将 tank level 与边界拉开极小距离，提升求解稳定性。"""

        lower = self._tank_min_levels + self.tank_level_epsilon
        upper = self._tank_max_levels - self.tank_level_epsilon

        # 极端情况下若上下界几乎重合，用中点替代可降低求解器异常概率。
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
        """配置“外部决策步长=1 小时”的仿真时间参数。

        注意：
        - 这里约束的是环境主语义（每次 step 代表 1 小时）；
        - 如需改变内部更细时间分辨率，应在不破坏主语义前提下单独设计。
        """

        wn.options.time.duration = self.STEP_SECONDS
        wn.options.time.hydraulic_timestep = self.STEP_SECONDS
        wn.options.time.report_timestep = self.STEP_SECONDS
        wn.options.time.pattern_timestep = self.STEP_SECONDS

    def _apply_tank_initial_levels(self, wn, tank_levels: NDArray[np.float64]) -> None:
        for tank_name, level in zip(self._tank_names, tank_levels):
            tank = wn.get_node(tank_name)
            tank.init_level = float(level)

    def _apply_hourly_demands(self, wn, current_demands: NDArray[np.float64]) -> None:
        """把当前 step 的 demand 写入 WNTR 网络对象。"""

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
        """把动作对应的两泵速度写入网络模型。"""

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
        """按 `Q/H/η` 估算单步泵电费（美元）。

        公式口径：
        - 先估算每台泵功率，再折算到本步时长对应能耗；
        - 最后按时段电价（峰/谷）换算为美元成本。
        """

        power_w = (
            self.WATER_DENSITY_KG_PER_M3
            * self.GRAVITY_M_PER_S2
            * pump_flows_m3s
            * pump_head_gains_m
            / self.pump_efficiency
        )
        energy_kwh = power_w * self.STEP_SECONDS / 3_600_000.0
        # 电价按 step 所在小时选择峰/谷价。
        unit_price = self._electricity_price_for_hour(hour)
        return float(np.sum(energy_kwh) * unit_price)

    @staticmethod
    def _electricity_price_for_hour(hour: int) -> float:
        return PEAK_PRICE_USD_PER_KWH if 7 <= hour < 23 else OFFPEAK_PRICE_USD_PER_KWH

    def _get_obs(self) -> NDArray[np.float32]:
        """构建观测向量：`[当前小时 demand, 当前 tank levels]`。

        这也是论文复现中的关键状态定义来源之一：
        - 不把 pressure 等结果变量放入状态；
        - 按 `scaling_mode` 对 demand/tank 分别缩放。
        """

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
        """准备 max-min / z-score 所需的固定统计量。"""

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
        """读取原始 INP 并写出临时修改版 INP。"""

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
