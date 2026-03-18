"""Discrete action space utilities for two-pump speed control.

Action design:
- 2 pumps
- each pump has 8 discrete speed levels
- total actions = 8 * 8 = 64
"""

from __future__ import annotations

from typing import Final

# Discrete speed levels for each pump.
ACTION_LEVELS: Final[tuple[float, ...]] = (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
NUM_PUMPS: Final[int] = 2
ACTION_COUNT: Final[int] = len(ACTION_LEVELS) ** NUM_PUMPS


def _validate_action_id(action_id: int) -> None:
    if not isinstance(action_id, int):
        raise TypeError(f"action_id must be int, got {type(action_id).__name__}.")
    if not 0 <= action_id < ACTION_COUNT:
        raise ValueError(f"action_id must be in [0, {ACTION_COUNT - 1}], got {action_id}.")


def _speed_to_index(speed: float) -> int:
    # Accept tiny floating-point noise and snap to the nearest valid level.
    tolerance = 1e-9
    for idx, level in enumerate(ACTION_LEVELS):
        if abs(speed - level) <= tolerance:
            return idx
    raise ValueError(f"Invalid pump speed {speed}. Valid levels: {ACTION_LEVELS}.")


def action_id_to_speeds(action_id: int) -> tuple[float, float]:
    """Convert action id (0..63) to two-pump speeds.

    Mapping rule:
    - pump1 index is the high-order index
    - pump2 index is the low-order index
    - action_id = pump1_index * 8 + pump2_index
    """

    _validate_action_id(action_id)
    levels_per_pump = len(ACTION_LEVELS)
    pump1_index = action_id // levels_per_pump
    pump2_index = action_id % levels_per_pump
    return ACTION_LEVELS[pump1_index], ACTION_LEVELS[pump2_index]


def speeds_to_action_id(pump1_speed: float, pump2_speed: float) -> int:
    """Convert two-pump speeds to action id (0..63)."""

    levels_per_pump = len(ACTION_LEVELS)
    pump1_index = _speed_to_index(pump1_speed)
    pump2_index = _speed_to_index(pump2_speed)
    return pump1_index * levels_per_pump + pump2_index


def all_action_combinations() -> list[tuple[float, float]]:
    """Return all 64 speed combinations ordered by action_id."""

    return [action_id_to_speeds(action_id) for action_id in range(ACTION_COUNT)]


__all__ = [
    "ACTION_LEVELS",
    "ACTION_COUNT",
    "NUM_PUMPS",
    "action_id_to_speeds",
    "speeds_to_action_id",
    "all_action_combinations",
]
