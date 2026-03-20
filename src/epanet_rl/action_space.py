"""两泵离散动作空间工具。

教学导读：
1. 论文明确给出的：
   - 两台泵，每台 8 档速度，合计 64 个离散动作。
2. 当前仓库实现：
   - 用 `action_id_to_speeds` / `speeds_to_action_id` 提供可逆映射；
   - 映射顺序固定，便于训练日志与测试对齐。
3. 工程细节：
   - 速度反查时使用微小浮点容差，降低数值噪声带来的误判风险。
"""

from __future__ import annotations

from typing import Final

# 每台泵的离散速度档位（与论文动作定义保持一致）。
ACTION_LEVELS: Final[tuple[float, ...]] = (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
NUM_PUMPS: Final[int] = 2
ACTION_COUNT: Final[int] = len(ACTION_LEVELS) ** NUM_PUMPS


def _validate_action_id(action_id: int) -> None:
    """校验动作 id 是否在合法区间。"""

    if not isinstance(action_id, int):
        raise TypeError(f"action_id must be int, got {type(action_id).__name__}.")
    if not 0 <= action_id < ACTION_COUNT:
        raise ValueError(f"action_id must be in [0, {ACTION_COUNT - 1}], got {action_id}.")


def _speed_to_index(speed: float) -> int:
    # 接受极小浮点误差，避免 0.9 与 0.9000000001 这类噪声导致误判。
    tolerance = 1e-9
    for idx, level in enumerate(ACTION_LEVELS):
        if abs(speed - level) <= tolerance:
            return idx
    raise ValueError(f"Invalid pump speed {speed}. Valid levels: {ACTION_LEVELS}.")


def action_id_to_speeds(action_id: int) -> tuple[float, float]:
    """将动作 id（0..63）映射到两泵速度。

    映射规则（建议记住这条，便于排查动作日志）：
    - pump1 是高位索引
    - pump2 是低位索引
    - action_id = pump1_index * 8 + pump2_index
    """

    _validate_action_id(action_id)
    levels_per_pump = len(ACTION_LEVELS)
    pump1_index = action_id // levels_per_pump
    pump2_index = action_id % levels_per_pump
    return ACTION_LEVELS[pump1_index], ACTION_LEVELS[pump2_index]


def speeds_to_action_id(pump1_speed: float, pump2_speed: float) -> int:
    """将两泵速度映射回动作 id（0..63）。"""

    levels_per_pump = len(ACTION_LEVELS)
    pump1_index = _speed_to_index(pump1_speed)
    pump2_index = _speed_to_index(pump2_speed)
    return pump1_index * levels_per_pump + pump2_index


def all_action_combinations() -> list[tuple[float, float]]:
    """返回按 action_id 顺序排列的全部 64 个速度组合。"""

    return [action_id_to_speeds(action_id) for action_id in range(ACTION_COUNT)]


__all__ = [
    "ACTION_LEVELS",
    "ACTION_COUNT",
    "NUM_PUMPS",
    "action_id_to_speeds",
    "speeds_to_action_id",
    "all_action_combinations",
]
