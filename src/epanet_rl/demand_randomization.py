"""需水随机化工具（论文两阶段口径）。

教学导读：
1. 论文明确给出的：
   - 时间乘子与空间乘子均来自截断正态，范围 `(1-Δ, 1+Δ)`；
   - 最终 demand 由默认模式、时间乘子、空间乘子和 base demand 共同决定。
2. 当前仓库实现：
   - 随机性全部通过显式 `numpy.random.Generator` 传入，保证可复现；
   - 提供一站式 `generate_randomized_demands`，也保留分步函数方便单测。
3. 工程细节：
   - 采样失败时有兜底策略（clip 补齐），优先保证批处理稳定返回。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_delta(delta: float, name: str) -> None:
    """统一校验 delta 边界，保证乘子保持正值。"""

    if delta < 0:
        raise ValueError(f"{name} must be >= 0, got {delta}.")
    if delta >= 1:
        raise ValueError(f"{name} must be < 1 to keep multipliers positive, got {delta}.")


def _as_bool_mask(mask: ArrayLike, expected_size: int) -> NDArray[np.bool_]:
    """将输入转换为布尔 mask，并校验长度一致。"""

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
    """在 [mean-delta, mean+delta] 上采样截断正态。

    采样方法：
    - 先做 rejection sampling；
    - 超过 max_rounds 仍未填满时，对剩余位置做 clip 兜底，
      以保证函数总能返回结果，避免隐式死循环。
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

    # 兜底：若拒绝采样回合数耗尽，使用 clip 补齐剩余样本。
    # 这样做的目标是“稳定可返回”，而不是追求严格的理论截断分布抽样效率最优。
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
    """生成按时间步变化的随机乘子。"""

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
    """生成按节点变化的随机乘子（支持固定节点）。

    `randomizable_mask` 语义：
    - True：该节点参与随机化
    - False：该节点乘子固定为 1.0
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
    """组合得到最终随机需水矩阵（本文件最关键公式）。

    公式：
        demand[t, i] =
            base_demands[i]
            * default_pattern[t]
            * time_multipliers[t]
            * space_multipliers[i]

    输入语义：
    - `base_demands`：各节点基础需水；
    - `default_pattern`：原始时序模式；
    - `time_multipliers`：时间随机乘子；
    - `space_multipliers`：空间随机乘子。

    返回：
    - 形状 `(T, N)` 的二维数组；
    - `T` 为时间步数，`N` 为需求节点数。
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
    """环境侧常用的一站式随机化接口。

    返回三元组：
    - randomized_demands: `(T, N)`
    - time_multipliers: `(T,)`
    - space_multipliers: `(N,)`
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
