"""Reward utilities for EPANET RL environments.

Implemented rules:
1) Regular step reward:
   reward = r_benchmark / 24 - E_pump_t
2) Hydraulic violation:
   reward = P_hydraulic, and episode terminates
3) At t == 23, if final tank volume < initial tank volume:
   total reward = regular reward + P_tank
4) P_tank supports two modes:
   - proportional: C_tank * ((initial - final) / initial) * r_benchmark
   - constant: fixed value

All functions are pure and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TankPenaltyMode = Literal["proportional", "constant"]


@dataclass(frozen=True)
class TankPenaltyConfig:
    """Configuration for final-step tank penalty."""

    mode: TankPenaltyMode
    constant_value: float = 0.0
    proportional_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("proportional", "constant"):
            raise ValueError("mode must be 'proportional' or 'constant'.")


@dataclass(frozen=True)
class StepRewardInput:
    """Inputs required to compute reward for one environment step."""

    t: int
    e_pump_t: float
    r_benchmark: float
    hydraulic_violation: bool
    p_hydraulic: float
    initial_tank_volume: float
    final_tank_volume: float


@dataclass(frozen=True)
class RewardResult:
    """Reward output for one step."""

    reward: float
    terminated: bool
    base_reward: float
    tank_penalty: float
    hydraulic_violation: bool


def compute_regular_reward(r_benchmark: float, e_pump_t: float, horizon_steps: int = 24) -> float:
    """Compute regular step reward: r_benchmark / horizon_steps - e_pump_t."""

    if horizon_steps <= 0:
        raise ValueError(f"horizon_steps must be positive, got {horizon_steps}.")
    return r_benchmark / horizon_steps - e_pump_t


def compute_tank_penalty(
    initial_tank_volume: float,
    final_tank_volume: float,
    r_benchmark: float,
    config: TankPenaltyConfig,
) -> float:
    """Compute P_tank when final tank volume is below initial volume.

    Returns 0.0 when final_tank_volume >= initial_tank_volume.
    In proportional mode, returns:
        C_tank * ((initial - final) / initial) * r_benchmark
    with safety guard for initial_tank_volume <= 0.
    """

    if final_tank_volume >= initial_tank_volume:
        return 0.0

    if config.mode == "constant":
        return config.constant_value

    if initial_tank_volume <= 0.0:
        return 0.0

    shortfall_ratio = (initial_tank_volume - final_tank_volume) / initial_tank_volume
    return config.proportional_coefficient * shortfall_ratio * r_benchmark


def compute_total_reward(
    step: StepRewardInput,
    tank_penalty_config: TankPenaltyConfig,
    *,
    horizon_steps: int = 24,
    final_step: int = 23,
) -> RewardResult:
    """Compute total reward and termination flag for one step.

    Priority:
    1) If hydraulic violation occurs, return P_hydraulic and terminate.
    2) Otherwise use regular reward.
    3) If this is final_step and final tank is below initial tank, add P_tank.
    """

    if step.hydraulic_violation:
        return RewardResult(
            reward=step.p_hydraulic,
            terminated=True,
            base_reward=0.0,
            tank_penalty=0.0,
            hydraulic_violation=True,
        )

    base_reward = compute_regular_reward(
        r_benchmark=step.r_benchmark,
        e_pump_t=step.e_pump_t,
        horizon_steps=horizon_steps,
    )

    tank_penalty = 0.0
    if step.t == final_step:
        tank_penalty = compute_tank_penalty(
            initial_tank_volume=step.initial_tank_volume,
            final_tank_volume=step.final_tank_volume,
            r_benchmark=step.r_benchmark,
            config=tank_penalty_config,
        )

    return RewardResult(
        reward=base_reward + tank_penalty,
        terminated=False,
        base_reward=base_reward,
        tank_penalty=tank_penalty,
        hydraulic_violation=False,
    )


__all__ = [
    "TankPenaltyMode",
    "TankPenaltyConfig",
    "StepRewardInput",
    "RewardResult",
    "compute_regular_reward",
    "compute_tank_penalty",
    "compute_total_reward",
]
