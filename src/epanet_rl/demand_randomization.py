"""Demand randomization utilities for EPANET RL environments.

This module provides pure functions to:
1) sample truncated-normal multipliers in (1 - delta, 1 + delta),
2) generate time multipliers,
3) generate space multipliers,
4) combine default demand pattern and base demands into randomized demands.

Design notes:
- All randomness is driven by an explicit ``numpy.random.Generator``.
- No global state is used.
- A mask interface is reserved for "large users are not randomized".
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_delta(delta: float, name: str) -> None:
    if delta < 0:
        raise ValueError(f"{name} must be >= 0, got {delta}.")
    if delta >= 1:
        raise ValueError(f"{name} must be < 1 to keep multipliers positive, got {delta}.")


def _as_bool_mask(mask: ArrayLike, expected_size: int) -> NDArray[np.bool_]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"Mask size mismatch: expected {expected_size}, got {arr.size}.")
    return arr


def sample_truncated_normal(
    *,
    size: int | tuple[int, ...],
    delta: float,
    rng: np.random.Generator,
    mean: float = 1.0,
    std: float | None = None,
    max_rounds: int = 100,
) -> NDArray[np.float64]:
    """Sample from a truncated normal distribution in [mean-delta, mean+delta].

    Sampling method: rejection sampling.
    If ``std`` is None, ``std = delta / 3`` is used.
    """

    _validate_delta(delta, "delta")

    if isinstance(size, int):
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}.")
        out_shape = (size,)
    else:
        out_shape = tuple(size)
        if any(dim < 0 for dim in out_shape):
            raise ValueError(f"size dimensions must be >= 0, got {out_shape}.")

    if np.prod(out_shape, dtype=np.int64) == 0:
        return np.empty(out_shape, dtype=np.float64)

    if delta == 0:
        return np.full(out_shape, mean, dtype=np.float64)

    if std is None:
        std = delta / 3.0
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}.")
    if max_rounds <= 0:
        raise ValueError(f"max_rounds must be positive, got {max_rounds}.")

    low = mean - delta
    high = mean + delta

    result = np.empty(out_shape, dtype=np.float64)
    filled = np.zeros(out_shape, dtype=bool)

    for _ in range(max_rounds):
        need = ~filled
        if not np.any(need):
            return result

        candidates = rng.normal(loc=mean, scale=std, size=out_shape)
        accepted = (candidates >= low) & (candidates <= high) & need
        result[accepted] = candidates[accepted]
        filled[accepted] = True

    # Fallback to nearest bound if rejection sampling did not finish.
    # This keeps function total and avoids hidden infinite loops.
    if np.any(~filled):
        candidates = rng.normal(loc=mean, scale=std, size=out_shape)
        candidates = np.clip(candidates, low, high)
        result[~filled] = candidates[~filled]

    return result


def generate_time_multipliers(
    *,
    num_steps: int,
    delta_time: float,
    rng: np.random.Generator,
    std_time: float | None = None,
) -> NDArray[np.float64]:
    """Generate per-timestep multipliers with truncated normal in (1-delta, 1+delta)."""

    if num_steps < 0:
        raise ValueError(f"num_steps must be >= 0, got {num_steps}.")

    return sample_truncated_normal(
        size=num_steps,
        delta=delta_time,
        rng=rng,
        mean=1.0,
        std=std_time,
    )


def generate_space_multipliers(
    *,
    num_nodes: int,
    delta_space: float,
    rng: np.random.Generator,
    std_space: float | None = None,
    randomizable_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Generate per-node multipliers with optional mask for non-randomized users.

    ``randomizable_mask``:
    - True  -> node is randomized
    - False -> node multiplier is fixed at 1.0
    This is the reserved interface for "large users are not randomized".
    """

    if num_nodes < 0:
        raise ValueError(f"num_nodes must be >= 0, got {num_nodes}.")

    if randomizable_mask is None:
        active_mask = np.ones(num_nodes, dtype=bool)
    else:
        active_mask = _as_bool_mask(randomizable_mask, expected_size=num_nodes)

    multipliers = np.ones(num_nodes, dtype=np.float64)
    active_count = int(active_mask.sum())
    if active_count > 0:
        multipliers[active_mask] = sample_truncated_normal(
            size=active_count,
            delta=delta_space,
            rng=rng,
            mean=1.0,
            std=std_space,
        )
    return multipliers


def compose_randomized_demands(
    *,
    base_demands: ArrayLike,
    default_pattern: ArrayLike,
    time_multipliers: ArrayLike,
    space_multipliers: ArrayLike,
) -> NDArray[np.float64]:
    """Compose randomized demand matrix from pattern, base demands, and multipliers.

    Formula:
        demand[t, i] =
            base_demands[i]
            * default_pattern[t]
            * time_multipliers[t]
            * space_multipliers[i]

    Returns:
        2D array with shape (T, N), where:
        - T is the number of time steps
        - N is the number of demand nodes
    """

    base = np.asarray(base_demands, dtype=np.float64).reshape(-1)
    pattern = np.asarray(default_pattern, dtype=np.float64).reshape(-1)
    t_mul = np.asarray(time_multipliers, dtype=np.float64).reshape(-1)
    s_mul = np.asarray(space_multipliers, dtype=np.float64).reshape(-1)

    if pattern.size != t_mul.size:
        raise ValueError(
            f"default_pattern and time_multipliers length mismatch: "
            f"{pattern.size} vs {t_mul.size}."
        )
    if base.size != s_mul.size:
        raise ValueError(
            f"base_demands and space_multipliers length mismatch: "
            f"{base.size} vs {s_mul.size}."
        )

    time_factor = pattern * t_mul
    space_factor = base * s_mul
    return np.outer(time_factor, space_factor)


def generate_randomized_demands(
    *,
    base_demands: ArrayLike,
    default_pattern: ArrayLike,
    delta_time: float,
    delta_space: float,
    rng: np.random.Generator,
    std_time: float | None = None,
    std_space: float | None = None,
    randomizable_mask: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convenience pure function for environment usage.

    Returns:
    - randomized_demands: shape (T, N)
    - time_multipliers: shape (T,)
    - space_multipliers: shape (N,)
    """

    base = np.asarray(base_demands, dtype=np.float64).reshape(-1)
    pattern = np.asarray(default_pattern, dtype=np.float64).reshape(-1)

    time_multipliers = generate_time_multipliers(
        num_steps=pattern.size,
        delta_time=delta_time,
        rng=rng,
        std_time=std_time,
    )
    space_multipliers = generate_space_multipliers(
        num_nodes=base.size,
        delta_space=delta_space,
        rng=rng,
        std_space=std_space,
        randomizable_mask=randomizable_mask,
    )
    randomized_demands = compose_randomized_demands(
        base_demands=base,
        default_pattern=pattern,
        time_multipliers=time_multipliers,
        space_multipliers=space_multipliers,
    )

    return randomized_demands, time_multipliers, space_multipliers


__all__ = [
    "sample_truncated_normal",
    "generate_time_multipliers",
    "generate_space_multipliers",
    "compose_randomized_demands",
    "generate_randomized_demands",
]
