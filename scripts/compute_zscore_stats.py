"""基于随机策略统计 ESFC-Z-Score 所需的状态均值/方差（Net3WntrEnv）。

教学导读：
1. 论文明确给出的：
   - Z-Score 需要先统计状态特征的 `mean/std`。
2. 当前仓库实现：
   - demand 与 tank 分开统计，并输出 JSON；
   - 统计时固定 `scaling_mode=none`，避免“先缩放再统计”的口径错误。
3. 工程近似说明：
   - 统计精度取决于 rollout 样本规模，样本越大通常越稳定。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Make `src/` importable when running from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:
    import numpy as np
    from epanet_rl.env_wntr import Net3WntrEnv


@dataclass(frozen=True)
class ZScoreStats:
    """Z-Score 统计结果结构。"""

    episode_count: int
    state_sample_count: int
    demand_feature_dim: int
    tank_feature_dim: int
    mean_episode_length: float
    hydraulic_violation_episode_count: int
    hydraulic_violation_episode_rate: float
    demand_mean: list[float]
    demand_std: list[float]
    tank_mean: list[float]
    tank_std: list[float]


class _RunningFeatureStats:
    """数值稳定的在线均值/方差累积器（向量版）。

    作用：
    - 避免把所有状态样本一次性堆在内存中；
    - 支持长 rollout 的流式统计，降低内存压力。
    """

    def __init__(self, feature_dim: int) -> None:
        import numpy as np

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        self._np = np
        self.n = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.m2 = np.zeros(feature_dim, dtype=np.float64)

    def update_batch(self, batch: "np.ndarray") -> None:
        """按批次更新统计量，避免一次性堆全部样本占用内存。"""

        np = self._np
        x = np.asarray(batch, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"batch must be 2D, got shape {x.shape}.")
        if x.shape[1] != self.mean.size:
            raise ValueError(
                f"feature dim mismatch: expected {self.mean.size}, got {x.shape[1]}."
            )
        if x.shape[0] == 0:
            return

        batch_n = int(x.shape[0])
        batch_mean = np.mean(x, axis=0)
        batch_m2 = np.sum((x - batch_mean) ** 2, axis=0)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        new_n = self.n + batch_n
        self.mean = self.mean + delta * (batch_n / new_n)
        self.m2 = self.m2 + batch_m2 + (delta**2) * (self.n * batch_n / new_n)
        self.n = new_n

    def std(self) -> "np.ndarray":
        if self.n == 0:
            return self._np.full_like(self.mean, self._np.nan, dtype=self._np.float64)
        # 使用总体标准差（ddof=0），与 np.std 默认口径一致。
        return self._np.sqrt(self.m2 / self.n)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute demand/tank Z-score statistics via random-policy rollouts on Net3WntrEnv."
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
    """每个 episode 派生独立 seed，保证可复现与多样性并存。"""

    if base_seed is None:
        return None
    return base_seed + episode_index


def _should_report_progress(index: int, total: int) -> bool:
    """控制进度打印频率，避免长跑输出过多。"""

    if total <= 20:
        return True
    stride = max(1, total // 20)
    return (index + 1) % stride == 0 or (index + 1) == total


def _append_state_sample(
    *,
    obs: "np.ndarray",
    demand_dim: int,
    demand_stats: _RunningFeatureStats,
    tank_stats: _RunningFeatureStats,
) -> None:
    """将单个观测拆成 demand/tank 两段后写入在线统计器。"""

    obs64 = obs.astype("float64", copy=False)
    demand_stats.update_batch(obs64[:demand_dim].reshape(1, -1))
    tank_stats.update_batch(obs64[demand_dim:].reshape(1, -1))


def _to_float_list(arr: "np.ndarray") -> list[float]:
    """将 numpy 数组转换为 JSON 友好的 float 列表。"""

    return [float(v) for v in arr.tolist()]


def _print_summary(*, args: argparse.Namespace, stats: ZScoreStats, elapsed_sec: float) -> None:
    """终端摘要输出。"""

    print("[zscore stats summary]")
    print(
        "config: "
        f"episodes={args.episodes}, seed={args.seed}, "
        f"delta_time={args.delta_time}, delta_space={args.delta_space}"
    )
    print(f"net3_inp={Path(args.net3_inp).resolve()}")
    print(f"episode_count={stats.episode_count}")
    print(f"state_sample_count={stats.state_sample_count}")
    print(f"demand_feature_dim={stats.demand_feature_dim}")
    print(f"tank_feature_dim={stats.tank_feature_dim}")
    print(f"mean_episode_length={stats.mean_episode_length:.6f}")
    print(f"hydraulic_violation_episode_count={stats.hydraulic_violation_episode_count}")
    print(f"hydraulic_violation_episode_rate={stats.hydraulic_violation_episode_rate:.6%}")
    print(f"demand_mean={json.dumps(stats.demand_mean, ensure_ascii=False)}")
    print(f"demand_std={json.dumps(stats.demand_std, ensure_ascii=False)}")
    print(f"tank_mean={json.dumps(stats.tank_mean, ensure_ascii=False)}")
    print(f"tank_std={json.dumps(stats.tank_std, ensure_ascii=False)}")
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

    try:
        import numpy as np
        from epanet_rl.env_wntr import Net3WntrEnv
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ModuleNotFoundError(
            f"Missing required dependency '{missing}'. "
            "Install project dependencies (e.g., from requirements.txt/environment.yml) and retry."
        ) from exc

    # 随机动作策略的 RNG；与环境 reset seed 共同决定可复现实验轨迹。
    policy_rng = np.random.default_rng(args.seed)
    episode_lengths: list[int] = []
    violation_count = 0
    t0 = time.perf_counter()

    env = Net3WntrEnv(
        net3_inp_path=args.net3_inp,
        # 统计 z-score 时必须读取未缩放状态，避免“缩放后的统计再缩放”。
        scaling_mode="none",
        delta_time=args.delta_time,
        delta_space=args.delta_space,
    )
    try:
        demand_dim = int(env._base_demands.size)  # noqa: SLF001
        tank_dim = int(len(env._tank_names))  # noqa: SLF001
        demand_stats = _RunningFeatureStats(feature_dim=demand_dim)
        tank_stats = _RunningFeatureStats(feature_dim=tank_dim)

        for ep_idx in range(args.episodes):
            obs, _ = env.reset(seed=_episode_seed(args.seed, ep_idx))
            _append_state_sample(
                obs=obs,
                demand_dim=demand_dim,
                demand_stats=demand_stats,
                tank_stats=tank_stats,
            )

            ep_len = 0
            ep_violation = False

            while True:
                action = int(policy_rng.integers(0, env.action_space.n))
                obs, _, terminated, truncated, info = env.step(action)
                _append_state_sample(
                    obs=obs,
                    demand_dim=demand_dim,
                    demand_stats=demand_stats,
                    tank_stats=tank_stats,
                )

                ep_len += 1
                ep_violation = ep_violation or bool(info.get("hydraulic_violation", False))
                if terminated or truncated:
                    break

            episode_lengths.append(ep_len)
            if ep_violation:
                violation_count += 1

            if args.progress and _should_report_progress(ep_idx, args.episodes):
                print(
                    f"[progress] {ep_idx + 1}/{args.episodes} episodes "
                    f"(last_length={ep_len}, last_hydraulic_violation={ep_violation})"
                )
    finally:
        env.close()

    elapsed_sec = time.perf_counter() - t0
    episode_count = int(args.episodes)
    stats = ZScoreStats(
        episode_count=episode_count,
        state_sample_count=int(demand_stats.n),
        demand_feature_dim=demand_dim,
        tank_feature_dim=tank_dim,
        mean_episode_length=float(np.mean(np.asarray(episode_lengths, dtype=np.float64))),
        hydraulic_violation_episode_count=int(violation_count),
        hydraulic_violation_episode_rate=float(violation_count / episode_count),
        demand_mean=_to_float_list(demand_stats.mean),
        demand_std=_to_float_list(demand_stats.std()),
        tank_mean=_to_float_list(tank_stats.mean),
        tank_std=_to_float_list(tank_stats.std()),
    )

    _print_summary(args=args, stats=stats, elapsed_sec=elapsed_sec)

    if args.output_json is not None:
        payload = {
            "config": {
                "episodes": int(args.episodes),
                "seed": None if args.seed is None else int(args.seed),
                "net3_inp": str(Path(args.net3_inp).resolve()),
                "delta_time": float(args.delta_time),
                "delta_space": float(args.delta_space),
                "scaling_mode": "none",
                "policy": "random_discrete",
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
