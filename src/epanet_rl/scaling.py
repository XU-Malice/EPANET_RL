"""State scaling utilities for EPANET RL.

This module provides pure scaling functions for two state categories:
- tank level
- demand

Implemented scaling methods:
1) ESFC-Max-Min scaling
2) ESFC-Z-Score scaling

Design goals:
- simple APIs for environment usage
- stable behavior for edge cases (equal bounds, zero variance)
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

_DEFAULT_EPS: Final[float] = 1e-12


def _as_float_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert input to float64 numpy array."""

    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} cannot be converted to float array.") from exc


def _validate_feature_range(feature_range: tuple[float, float]) -> tuple[float, float]:
    """Validate feature range for min-max scaling."""

    if len(feature_range) != 2:
        raise ValueError(f"feature_range must have length 2, got {feature_range}.")
    out_min, out_max = float(feature_range[0]), float(feature_range[1])
    if out_max <= out_min:
        raise ValueError(
            f"feature_range must satisfy max > min, got ({out_min}, {out_max})."
        )
    return out_min, out_max


def max_min_scale(
    values: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = True,
    equal_bounds_value: float | None = None,
    eps: float = _DEFAULT_EPS,
) -> NDArray[np.float64]:
    """Apply stable max-min scaling.

    Formula:
        x_norm = (x - min) / (max - min)
        x_scaled = out_min + x_norm * (out_max - out_min)

    Edge cases:
    - If max ~= min (|max-min| <= eps), output ``equal_bounds_value``.
    - If ``equal_bounds_value`` is None, use midpoint of ``feature_range``.
    """

    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}.")

    out_min, out_max = _validate_feature_range(feature_range)
    x = _as_float_array(values, name="values")
    lower = _as_float_array(lower_bounds, name="lower_bounds")
    upper = _as_float_array(upper_bounds, name="upper_bounds")

    width = upper - lower
    if np.any(width < -eps):
        raise ValueError("Found lower_bounds > upper_bounds, which is invalid.")

    stable_width = np.where(np.abs(width) > eps, width, 1.0)
    x_norm = (x - lower) / stable_width
    if clip:
        x_norm = np.clip(x_norm, 0.0, 1.0)

    scaled = out_min + x_norm * (out_max - out_min)

    degenerate = np.abs(width) <= eps
    if np.any(degenerate):
        fill_value = (out_min + out_max) / 2.0 if equal_bounds_value is None else float(equal_bounds_value)
        scaled = np.where(degenerate, fill_value, scaled)

    return scaled.astype(np.float64, copy=False)


def z_score_scale(
    values: ArrayLike,
    means: ArrayLike,
    stds: ArrayLike,
    *,
    zero_std_value: float = 0.0,
    eps: float = _DEFAULT_EPS,
) -> NDArray[np.float64]:
    """Apply stable z-score scaling.

    Formula:
        z = (x - mean) / std

    Edge cases:
    - If std ~= 0 (|std| <= eps), output ``zero_std_value``.
    """

    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}.")

    x = _as_float_array(values, name="values")
    mean = _as_float_array(means, name="means")
    std = _as_float_array(stds, name="stds")

    stable_std = np.where(np.abs(std) > eps, std, 1.0)
    z = (x - mean) / stable_std

    degenerate = np.abs(std) <= eps
    if np.any(degenerate):
        z = np.where(degenerate, float(zero_std_value), z)

    return z.astype(np.float64, copy=False)


def tank_max_min_scale(
    tank_levels: ArrayLike,
    tank_level_min: ArrayLike,
    tank_level_max: ArrayLike,
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = True,
    equal_bounds_value: float | None = None,
    eps: float = _DEFAULT_EPS,
) -> NDArray[np.float64]:
    """Max-min scaling interface for tank levels."""

    return max_min_scale(
        values=tank_levels,
        lower_bounds=tank_level_min,
        upper_bounds=tank_level_max,
        feature_range=feature_range,
        clip=clip,
        equal_bounds_value=equal_bounds_value,
        eps=eps,
    )


def demand_max_min_scale(
    demands: ArrayLike,
    demand_min: ArrayLike,
    demand_max: ArrayLike,
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
    clip: bool = True,
    equal_bounds_value: float | None = None,
    eps: float = _DEFAULT_EPS,
) -> NDArray[np.float64]:
    """Max-min scaling interface for demands."""

    return max_min_scale(
        values=demands,
        lower_bounds=demand_min,
        upper_bounds=demand_max,
        feature_range=feature_range,
        clip=clip,
        equal_bounds_value=equal_bounds_value,
        eps=eps,
    )


def tank_z_score_scale(
    tank_levels: ArrayLike,
    tank_mean: ArrayLike,
    tank_std: ArrayLike,
    *,
    zero_std_value: float = 0.0,
    eps: float = _DEFAULT_EPS,
) -> NDArray[np.float64]:
    """Z-score scaling interface for tank levels."""

    return z_score_scale(
        values=tank_levels,
        means=tank_mean,
        stds=tank_std,
        zero_std_value=zero_std_value,
        eps=eps,
    )


def demand_z_score_scale(
    demands: ArrayLike,
    demand_mean: ArrayLike,
    demand_std: ArrayLike,
    *,
    zero_std_value: float = 0.0,
    eps: float = _DEFAULT_EPS,
) -> NDArray[np.float64]:
    """Z-score scaling interface for demands."""

    return z_score_scale(
        values=demands,
        means=demand_mean,
        stds=demand_std,
        zero_std_value=zero_std_value,
        eps=eps,
    )


__all__ = [
    "max_min_scale",
    "z_score_scale",
    "tank_max_min_scale",
    "demand_max_min_scale",
    "tank_z_score_scale",
    "demand_z_score_scale",
]
