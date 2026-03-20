"""EPANET RL 奖励函数工具（纯函数版）。

教学导读：
1. 论文明确给出的：
   - 常规步奖励：`r_benchmark / 24 - E_pump_t`
   - 水力违例：给大负惩罚并提前终止
   - tank penalty：仅在最后一步检查
2. 当前仓库实现：
   - 用纯函数组织，便于单元测试与环境复用
   - 通过 `compute_total_reward` 明确“违例优先级高于一切”
3. 工程近似说明：
   - `TankPenaltyConfig` 提供 proportional/constant 两种形式，便于实验切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TankPenaltyMode = Literal["proportional", "constant"]


@dataclass(frozen=True)
class TankPenaltyConfig:
    """末步水箱惩罚配置。"""

    mode: TankPenaltyMode
    constant_value: float = 0.0
    proportional_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("proportional", "constant"):
            raise ValueError("mode must be 'proportional' or 'constant'.")


@dataclass(frozen=True)
class StepRewardInput:
    """单步奖励计算所需输入。"""

    t: int
    e_pump_t: float
    r_benchmark: float
    hydraulic_violation: bool
    p_hydraulic: float
    initial_tank_volume: float
    final_tank_volume: float


@dataclass(frozen=True)
class RewardResult:
    """单步奖励计算结果。"""

    reward: float
    terminated: bool
    base_reward: float
    tank_penalty: float
    hydraulic_violation: bool


def compute_regular_reward(r_benchmark: float, e_pump_t: float, horizon_steps: int = 24) -> float:
    """常规奖励：`r_benchmark / horizon_steps - e_pump_t`。"""

    if horizon_steps <= 0:
        raise ValueError(f"horizon_steps must be positive, got {horizon_steps}.")
    return r_benchmark / horizon_steps - e_pump_t


def compute_tank_penalty(
    initial_tank_volume: float,
    final_tank_volume: float,
    r_benchmark: float,
    config: TankPenaltyConfig,
) -> float:
    """计算末步水箱惩罚 P_tank。

    设计意图：
    - 若 final >= initial，则无惩罚（返回 0）。
    - proportional 模式下，按短缺比例与 r_benchmark 成比例惩罚。
    - 对 initial<=0 做保护，避免除零与非物理放大。
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
    """计算单步总奖励并给出是否终止。

    优先级（非常关键）：
    1) 先看 hydraulic_violation：若违例，直接返回 P_hydraulic 并 terminated=True。
    2) 未违例时，计算常规奖励。
    3) 若是 final_step，再按配置追加 tank penalty。

    这个优先级会直接影响训练行为：
    - 一旦出现违例，不再“奖励抵扣”，而是立即按惩罚结束。
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
