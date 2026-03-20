"""基于随机策略统计 r_benchmark 参考值（Net3WntrEnv）。

教学导读：
1. 论文明确给出的：
   - `r_benchmark` 应来自大量 rollout 的统计，而不是手写常数。
2. 当前仓库实现：
   - 逐 episode 累计 `info["e_pump_t"]` 作为总电费；
   - 输出 all/successful/full_horizon/fallback 等多口径统计。
3. 工程近似说明：
   - “论文口径最接近哪一组”需要结合你的实验设定判断，本脚本提供多组候选值供对照。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Make `src/` importable when running from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:
    import numpy as np
    from epanet_rl.env_wntr import Net3WntrEnv

ScalingMode = Literal["none", "max_min", "z_score"]


@dataclass(frozen=True)
class EpisodeRecord:
    """单个 episode 的最小记录单元。

    输出字段全部是“后处理统计真正需要的最小闭包”：
    - 总能耗成本；
    - 实际走了多少步；
    - 是否出现水力违例；
    - 是否触发 fallback。

    为什么不把完整 step 日志都存下来：
    - benchmark 脚本的目标是统计口径，不是做逐步诊断；
    - 只保留聚合字段可以降低内存占用并简化 JSON 输出。
    """

    total_energy_cost: float
    episode_length: int
    hydraulic_violation: bool
    fallback_used: bool


@dataclass(frozen=True)
class BenchmarkStats:
    """脚本最终输出的聚合统计结构。"""

    episode_count: int
    mean_total_energy_cost: float
    std_total_energy_cost: float
    min_total_energy_cost: float
    max_total_energy_cost: float
    mean_episode_length: float
    hydraulic_violation_count: int
    hydraulic_violation_rate: float
    successful_episode_count: int
    successful_episode_rate: float
    successful_mean_total_energy_cost: float | None
    successful_std_total_energy_cost: float | None
    successful_mean_episode_length: float | None
    violated_episode_count: int
    violated_episode_rate: float
    violated_mean_total_energy_cost: float | None
    violated_std_total_energy_cost: float | None
    violated_mean_episode_length: float | None
    fallback_episode_count: int
    fallback_episode_rate: float
    non_fallback_episode_count: int
    non_fallback_episode_rate: float
    fallback_mean_total_energy_cost: float | None
    fallback_std_total_energy_cost: float | None
    fallback_mean_episode_length: float | None
    non_fallback_mean_total_energy_cost: float | None
    non_fallback_std_total_energy_cost: float | None
    non_fallback_mean_episode_length: float | None
    full_horizon_count: int
    full_horizon_rate: float
    candidate_benchmark_all: float
    candidate_benchmark_successful: float | None
    candidate_benchmark_full_horizon: float | None
    candidate_benchmark_successful_non_fallback: float | None
    candidate_benchmark_full_horizon_non_fallback: float | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate r_benchmark statistics via random-policy rollouts on Net3WntrEnv."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to roll out.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility. Episode i uses seed + i.",
    )
    parser.add_argument(
        "--net3-inp",
        type=Path,
        default=PROJECT_ROOT / "networks" / "Net3.inp",
        help="Path to Net3 INP file.",
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        default="none",
        choices=["none", "max_min", "z_score"],
        help="Scaling mode passed to Net3WntrEnv.",
    )
    parser.add_argument(
        "--delta-time",
        type=float,
        default=0.10,
        help="Demand randomization delta_time passed to Net3WntrEnv.",
    )
    parser.add_argument(
        "--delta-space",
        type=float,
        default=0.10,
        help="Demand randomization delta_space passed to Net3WntrEnv.",
    )
    parser.add_argument(
        "--p-hydraulic",
        type=float,
        default=-200.0,
        help="Hydraulic violation penalty passed to Net3WntrEnv (must be negative).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path to save full JSON summary.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print periodic rollout progress.",
    )
    return parser


def _episode_seed(base_seed: int | None, episode_index: int) -> int | None:
    """让每个 episode 使用可复现但不同的 seed。"""

    if base_seed is None:
        return None
    return base_seed + episode_index


def _run_single_episode(
    env: "Net3WntrEnv",
    *,
    reset_seed: int | None,
    policy_rng: "np.random.Generator",
) -> EpisodeRecord:
    """滚动一个 episode，并累计能耗/长度/约束标记。

    输入：一个已构造好的环境、reset seed，以及用于随机策略采样动作的 RNG。
    输出：`EpisodeRecord`。

    教学理解：
    - 这里的策略不是训练好的策略，而是“随机离散动作”；
    - 因此算出来的 benchmark 更像“随机基准线”，用于给 `r_benchmark` 选一个可解释的参考值。
    """

    _, _ = env.reset(seed=reset_seed)

    total_energy_cost = 0.0
    episode_length = 0
    hydraulic_violation = False
    fallback_used = False

    while True:
        action = int(policy_rng.integers(0, env.action_space.n))
        _, _, terminated, truncated, info = env.step(action)

        # 该字段是 benchmark 的核心口径，缺失时直接报错，避免默默算错。
        if "e_pump_t" not in info:
            raise KeyError("Net3WntrEnv info does not contain 'e_pump_t'; cannot compute energy benchmark.")

        total_energy_cost += float(info["e_pump_t"])
        episode_length += 1
        hydraulic_violation = hydraulic_violation or bool(info.get("hydraulic_violation", False))
        fallback_used = fallback_used or bool(info.get("fallback_used", False))

        if terminated or truncated:
            break

    return EpisodeRecord(
        total_energy_cost=total_energy_cost,
        episode_length=episode_length,
        hydraulic_violation=hydraulic_violation,
        fallback_used=fallback_used,
    )


def _compute_stats(records: list[EpisodeRecord]) -> BenchmarkStats:
    """把 episode 记录聚合为多口径 benchmark 统计。"""

    import numpy as np

    if not records:
        raise ValueError("No episode records provided.")

    costs = np.asarray([r.total_energy_cost for r in records], dtype=np.float64)
    lengths = np.asarray([r.episode_length for r in records], dtype=np.float64)
    violations = np.asarray([r.hydraulic_violation for r in records], dtype=bool)
    fallbacks = np.asarray([r.fallback_used for r in records], dtype=bool)
    episode_count = len(records)

    # 口径定义（与脚本文案保持一致）：
    # - successful: 无 hydraulic violation 且完整 24 步
    # - violated:   出现 hydraulic violation
    # - full_horizon: 走满 24 步（不论是否 violation）
    successful_mask = (~violations) & (lengths == 24.0)
    violated_mask = violations
    full_horizon_mask = lengths == 24.0
    fallback_mask = fallbacks
    non_fallback_mask = ~fallback_mask
    successful_non_fallback_mask = successful_mask & non_fallback_mask
    full_horizon_non_fallback_mask = full_horizon_mask & non_fallback_mask

    # 为了处理“某一组为空”的情况，下面统一通过 _safe_mean/_safe_std 输出 None。
    successful_costs = costs[successful_mask]
    successful_lengths = lengths[successful_mask]
    violated_costs = costs[violated_mask]
    violated_lengths = lengths[violated_mask]
    full_horizon_costs = costs[full_horizon_mask]
    fallback_costs = costs[fallback_mask]
    fallback_lengths = lengths[fallback_mask]
    non_fallback_costs = costs[non_fallback_mask]
    non_fallback_lengths = lengths[non_fallback_mask]
    successful_non_fallback_costs = costs[successful_non_fallback_mask]
    full_horizon_non_fallback_costs = costs[full_horizon_non_fallback_mask]

    successful_count = int(np.sum(successful_mask))
    violated_count = int(np.sum(violated_mask))
    full_horizon_count = int(np.sum(full_horizon_mask))
    fallback_count = int(np.sum(fallback_mask))
    non_fallback_count = int(np.sum(non_fallback_mask))

    def _safe_mean(values: np.ndarray) -> float | None:
        if values.size == 0:
            return None
        return float(np.mean(values))

    def _safe_std(values: np.ndarray) -> float | None:
        if values.size == 0:
            return None
        return float(np.std(values))

    # candidate benchmark 本质是不同口径下的均值能耗，便于和论文值对齐比较。
    # 若你在论文复现报告里引用某个值，请一定写明使用的是哪一种口径，
    # 因为“all / successful / full_horizon / non_fallback”并不完全等价。
    mean_total_energy_cost = float(np.mean(costs))
    successful_mean_total_energy_cost = _safe_mean(successful_costs)
    full_horizon_mean_total_energy_cost = _safe_mean(full_horizon_costs)
    successful_non_fallback_mean_total_energy_cost = _safe_mean(successful_non_fallback_costs)
    full_horizon_non_fallback_mean_total_energy_cost = _safe_mean(full_horizon_non_fallback_costs)

    return BenchmarkStats(
        episode_count=episode_count,
        mean_total_energy_cost=mean_total_energy_cost,
        std_total_energy_cost=float(np.std(costs)),
        min_total_energy_cost=float(np.min(costs)),
        max_total_energy_cost=float(np.max(costs)),
        mean_episode_length=float(np.mean(lengths)),
        hydraulic_violation_count=violated_count,
        hydraulic_violation_rate=float(violated_count / episode_count),
        successful_episode_count=successful_count,
        successful_episode_rate=float(successful_count / episode_count),
        successful_mean_total_energy_cost=_safe_mean(successful_costs),
        successful_std_total_energy_cost=_safe_std(successful_costs),
        successful_mean_episode_length=_safe_mean(successful_lengths),
        violated_episode_count=violated_count,
        violated_episode_rate=float(violated_count / episode_count),
        violated_mean_total_energy_cost=_safe_mean(violated_costs),
        violated_std_total_energy_cost=_safe_std(violated_costs),
        violated_mean_episode_length=_safe_mean(violated_lengths),
        fallback_episode_count=fallback_count,
        fallback_episode_rate=float(fallback_count / episode_count),
        non_fallback_episode_count=non_fallback_count,
        non_fallback_episode_rate=float(non_fallback_count / episode_count),
        fallback_mean_total_energy_cost=_safe_mean(fallback_costs),
        fallback_std_total_energy_cost=_safe_std(fallback_costs),
        fallback_mean_episode_length=_safe_mean(fallback_lengths),
        non_fallback_mean_total_energy_cost=_safe_mean(non_fallback_costs),
        non_fallback_std_total_energy_cost=_safe_std(non_fallback_costs),
        non_fallback_mean_episode_length=_safe_mean(non_fallback_lengths),
        full_horizon_count=full_horizon_count,
        full_horizon_rate=float(full_horizon_count / episode_count),
        candidate_benchmark_all=mean_total_energy_cost,
        candidate_benchmark_successful=successful_mean_total_energy_cost,
        candidate_benchmark_full_horizon=full_horizon_mean_total_energy_cost,
        candidate_benchmark_successful_non_fallback=successful_non_fallback_mean_total_energy_cost,
        candidate_benchmark_full_horizon_non_fallback=full_horizon_non_fallback_mean_total_energy_cost,
    )


def _should_report_progress(index: int, total: int) -> bool:
    """在大样本时稀疏打印进度，避免刷屏。"""

    if total <= 20:
        return True
    # Print around 20 progress lines total.
    stride = max(1, total // 20)
    return (index + 1) % stride == 0 or (index + 1) == total


def _print_summary(*, args: argparse.Namespace, stats: BenchmarkStats, elapsed_sec: float) -> None:
    """终端摘要输出：结构与 JSON 字段尽量一一对应。"""

    def _fmt(value: float | None) -> str:
        if value is None:
            return "null"
        return f"{value:.6f}"

    print("[r_benchmark rollout summary]")
    print(
        "config: "
        f"episodes={args.episodes}, seed={args.seed}, scaling_mode={args.scaling_mode}, "
        f"delta_time={args.delta_time}, delta_space={args.delta_space}, p_hydraulic={args.p_hydraulic}"
    )
    print(f"net3_inp={Path(args.net3_inp).resolve()}")
    print("[all]")
    print(f"episode_count={stats.episode_count}")
    print(f"mean_total_energy_cost={stats.mean_total_energy_cost:.6f}")
    print(f"std_total_energy_cost={stats.std_total_energy_cost:.6f}")
    print(f"min_total_energy_cost={stats.min_total_energy_cost:.6f}")
    print(f"max_total_energy_cost={stats.max_total_energy_cost:.6f}")
    print(f"mean_episode_length={stats.mean_episode_length:.6f}")
    print(f"hydraulic_violation_count={stats.hydraulic_violation_count}")
    print(f"hydraulic_violation_rate={stats.hydraulic_violation_rate:.6%}")
    print(f"full_horizon_count={stats.full_horizon_count}")
    print(f"full_horizon_rate={stats.full_horizon_rate:.6%}")
    print("[successful]")
    print(f"successful_episode_count={stats.successful_episode_count}")
    print(f"successful_episode_rate={stats.successful_episode_rate:.6%}")
    print(f"successful_mean_total_energy_cost={_fmt(stats.successful_mean_total_energy_cost)}")
    print(f"successful_std_total_energy_cost={_fmt(stats.successful_std_total_energy_cost)}")
    print(f"successful_mean_episode_length={_fmt(stats.successful_mean_episode_length)}")
    print("[violated]")
    print(f"violated_episode_count={stats.violated_episode_count}")
    print(f"violated_episode_rate={stats.violated_episode_rate:.6%}")
    print(f"violated_mean_total_energy_cost={_fmt(stats.violated_mean_total_energy_cost)}")
    print(f"violated_std_total_energy_cost={_fmt(stats.violated_std_total_energy_cost)}")
    print(f"violated_mean_episode_length={_fmt(stats.violated_mean_episode_length)}")
    print("[fallback]")
    print(f"fallback_episode_count={stats.fallback_episode_count}")
    print(f"fallback_episode_rate={stats.fallback_episode_rate:.6%}")
    print(f"fallback_mean_total_energy_cost={_fmt(stats.fallback_mean_total_energy_cost)}")
    print(f"fallback_std_total_energy_cost={_fmt(stats.fallback_std_total_energy_cost)}")
    print(f"fallback_mean_episode_length={_fmt(stats.fallback_mean_episode_length)}")
    print("[non_fallback]")
    print(f"non_fallback_episode_count={stats.non_fallback_episode_count}")
    print(f"non_fallback_episode_rate={stats.non_fallback_episode_rate:.6%}")
    print(f"non_fallback_mean_total_energy_cost={_fmt(stats.non_fallback_mean_total_energy_cost)}")
    print(f"non_fallback_std_total_energy_cost={_fmt(stats.non_fallback_std_total_energy_cost)}")
    print(f"non_fallback_mean_episode_length={_fmt(stats.non_fallback_mean_episode_length)}")
    print("[candidate benchmark]")
    print(f"candidate_benchmark_all={_fmt(stats.candidate_benchmark_all)}")
    print(f"candidate_benchmark_successful={_fmt(stats.candidate_benchmark_successful)}")
    print(f"candidate_benchmark_full_horizon={_fmt(stats.candidate_benchmark_full_horizon)}")
    print(
        "candidate_benchmark_successful_non_fallback="
        f"{_fmt(stats.candidate_benchmark_successful_non_fallback)}"
    )
    print(
        "candidate_benchmark_full_horizon_non_fallback="
        f"{_fmt(stats.candidate_benchmark_full_horizon_non_fallback)}"
    )
    print(f"elapsed_seconds={elapsed_sec:.3f}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON（自动创建目录）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError(f"--episodes must be positive, got {args.episodes}.")

    if args.p_hydraulic >= 0.0:
        raise ValueError(f"--p-hydraulic must be negative, got {args.p_hydraulic}.")

    try:
        import numpy as np
        from epanet_rl.env_wntr import Net3WntrEnv
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ModuleNotFoundError(
            f"Missing required dependency '{missing}'. "
            "Install project dependencies (e.g., from requirements.txt/environment.yml) and retry."
        ) from exc

    # 行为策略与 reset seed 分离：
    # - reset seed 控制环境随机化
    # - policy_rng 控制动作采样
    policy_rng = np.random.default_rng(args.seed)
    records: list[EpisodeRecord] = []
    t0 = time.perf_counter()

    env = Net3WntrEnv(
        net3_inp_path=args.net3_inp,
        scaling_mode=args.scaling_mode,
        delta_time=args.delta_time,
        delta_space=args.delta_space,
        p_hydraulic=args.p_hydraulic,
    )
    try:
        for ep_idx in range(args.episodes):
            record = _run_single_episode(
                env,
                reset_seed=_episode_seed(args.seed, ep_idx),
                policy_rng=policy_rng,
            )
            records.append(record)

            if args.progress and _should_report_progress(ep_idx, args.episodes):
                print(
                    f"[progress] {ep_idx + 1}/{args.episodes} episodes "
                    f"(last_total_energy_cost={record.total_energy_cost:.6f}, "
                    f"last_length={record.episode_length}, "
                    f"last_hydraulic_violation={record.hydraulic_violation})"
                )
    finally:
        env.close()

    stats = _compute_stats(records)
    elapsed_sec = time.perf_counter() - t0
    _print_summary(args=args, stats=stats, elapsed_sec=elapsed_sec)

    if args.output_json is not None:
        payload = {
            "config": {
                "episodes": int(args.episodes),
                "seed": None if args.seed is None else int(args.seed),
                "net3_inp": str(Path(args.net3_inp).resolve()),
                "scaling_mode": str(args.scaling_mode),
                "delta_time": float(args.delta_time),
                "delta_space": float(args.delta_space),
                "p_hydraulic": float(args.p_hydraulic),
            },
            "stats": asdict(stats),
            "elapsed_seconds": float(elapsed_sec),
        }
        _write_json(Path(args.output_json), payload)
        print(f"Saved JSON summary to: {Path(args.output_json).resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
