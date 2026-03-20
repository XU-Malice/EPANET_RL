"""统一策略评估脚本：在 Net3WntrEnv 上评估 PPO/E-PPO 模型。

教学导读：
1. 论文明确给出的：
   - 需要比较 PPO 与 E-PPO 在相同环境设定下的行为差异。
2. 当前仓库实现：
   - 统一评估入口，输出 reward/energy/违例/全程率等指标；
   - 增加奖励分解与 tank 末端体积审计字段，便于分析“省电但透支储水”现象。
3. 工程近似说明：
   - 评估脚本负责“观测与统计”，不改变环境或训练逻辑。

这个脚本的定位不是训练，而是“审计一个已经保存的策略”。

教学理解：
- 训练脚本回答“怎么学”；
- 评估脚本回答“学出来以后到底表现怎样”；
- 如果你在论文复现中要汇报最终指标，应优先引用这里产出的聚合统计。

口径提示：
- 论文明确给出的：需要看 reward、能耗、约束满足情况；
- 当前仓库实现：额外记录 tank volume 变化、full horizon 比例等工程审计指标。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# 允许从仓库根目录直接运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:
    import numpy as np
    from stable_baselines3.common.base_class import BaseAlgorithm

AlgoName = Literal["ppo", "eppo"]
ScalingMode = Literal["none", "max_min", "z_score"]


@dataclass(frozen=True)
class EpisodeEvalRecord:
    """单个 episode 的评估记录。

    这里保留的是“最终报告最常用”的聚合指标，而不是逐 step 全日志。
    如果你要进一步定位某次 episode 为何失败，应改用诊断脚本而不是继续堆字段到这里。
    """

    total_reward: float
    total_energy_cost: float
    total_base_reward: float
    total_tank_penalty: float
    episode_length: int
    hydraulic_violation: bool
    full_horizon: bool
    initial_tank_volume: float
    final_tank_volume: float
    volume_change_ratio: float


@dataclass(frozen=True)
class EvalStats:
    """聚合后的评估统计结果。"""

    episode_count: int
    mean_total_reward: float
    std_total_reward: float
    mean_total_energy_cost: float
    std_total_energy_cost: float
    mean_total_base_reward: float
    std_total_base_reward: float
    mean_total_tank_penalty: float
    std_total_tank_penalty: float
    mean_episode_length: float
    mean_initial_tank_volume: float
    mean_final_tank_volume: float
    mean_volume_change_ratio: float
    std_volume_change_ratio: float
    hydraulic_violation_count: int
    hydraulic_violation_rate: float
    full_horizon_count: int
    full_horizon_rate: float
    successful_episode_count: int
    successful_episode_rate: float
    successful_mean_total_energy_cost: float | None
    successful_mean_total_base_reward: float | None
    successful_mean_total_tank_penalty: float | None
    successful_mean_final_tank_volume: float | None
    successful_mean_volume_change_ratio: float | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO/E-PPO policy on Net3WntrEnv."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to saved model (.zip).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "eppo"],
        help="Algorithm label for logging (loader uses SB3 PPO-compatible format).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Episode i uses seed+i.",
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
        "--scaling-mode",
        type=str,
        default="max_min",
        choices=["none", "max_min", "z_score"],
        help="Observation scaling mode passed to Net3WntrEnv.",
    )
    parser.add_argument(
        "--r-benchmark",
        type=float,
        default=406.54,
        help="r_benchmark passed to Net3WntrEnv.",
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
        help="Optional JSON output file path.",
    )
    return parser


def _episode_seed(base_seed: int | None, episode_index: int) -> int | None:
    """为每个 episode 生成稳定可复现的种子。"""

    if base_seed is None:
        return None
    return base_seed + episode_index


def _resolve_model_path(path: Path) -> Path:
    """兼容传入不带 .zip 后缀的模型路径。"""

    if path.exists():
        return path
    if path.suffix != ".zip":
        candidate = path.with_suffix(".zip")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Model file not found: {path}")


def _load_model(model_path: Path) -> "BaseAlgorithm":
    """加载 SB3 PPO 兼容模型。"""

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ModuleNotFoundError(
            f"Missing required dependency '{missing}'. Please install stable-baselines3."
        ) from exc

    # 当前训练入口的 PPO/E-PPO 都是 PPO-compatible 存档格式。
    return PPO.load(str(model_path), device="cpu")


def _run_single_episode(
    *,
    env,
    model: "BaseAlgorithm",
    reset_seed: int | None,
) -> EpisodeEvalRecord:
    """执行一个 episode，并返回审计版记录。

    输入：环境、模型、reset seed。
    输出：单个 episode 的聚合评估记录。

    审计重点：
    - `total_reward` 与 `total_energy_cost` 同时保留，避免只看单一指标；
    - 同步记录 `base_reward/tank_penalty` 分解；
    - 记录初末 tank volume 与变化比例，辅助解释策略行为。

    为什么采用“聚合后返回”：
    - 评估阶段重点通常是最终均值/方差/成功率；
    - 逐步日志太大，不适合默认写入评估 JSON。
    """

    obs, _ = env.reset(seed=reset_seed)

    total_reward = 0.0
    total_energy_cost = 0.0
    total_base_reward = 0.0
    total_tank_penalty = 0.0
    episode_length = 0
    hydraulic_violation = False

    # initial/final tank volume 来自 env.step() 的 info 字段。
    initial_tank_volume: float | None = None
    final_tank_volume: float | None = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        if "e_pump_t" not in info:
            raise KeyError("Net3WntrEnv info does not contain 'e_pump_t'; cannot compute energy metrics.")

        total_reward += float(reward)
        total_energy_cost += float(info["e_pump_t"])
        total_base_reward += float(info.get("base_reward", 0.0))
        total_tank_penalty += float(info.get("tank_penalty", 0.0))
        episode_length += 1
        hydraulic_violation = hydraulic_violation or bool(info.get("hydraulic_violation", False))

        # 记录 tank volume 端点：
        # - initial 理论上整集不变，取首次可用值；
        # - final 每步更新，最后保留末步值。
        if initial_tank_volume is None and ("initial_tank_volume" in info):
            initial_tank_volume = float(info["initial_tank_volume"])
        if "final_tank_volume" in info:
            final_tank_volume = float(info["final_tank_volume"])

        if terminated or truncated:
            break

    # 在当前环境合同下这两个值应始终可得；这里做防御式兜底避免脚本异常退出。
    if initial_tank_volume is None:
        initial_tank_volume = 0.0
    if final_tank_volume is None:
        final_tank_volume = initial_tank_volume

    # 需求定义：volume_change_ratio = (final - initial) / initial
    # 若 initial<=0（极端防御场景）则置 0，避免除零。
    if initial_tank_volume > 0.0:
        volume_change_ratio = (final_tank_volume - initial_tank_volume) / initial_tank_volume
    else:
        volume_change_ratio = 0.0

    full_horizon = episode_length == 24
    return EpisodeEvalRecord(
        total_reward=total_reward,
        total_energy_cost=total_energy_cost,
        total_base_reward=total_base_reward,
        total_tank_penalty=total_tank_penalty,
        episode_length=episode_length,
        hydraulic_violation=hydraulic_violation,
        full_horizon=full_horizon,
        initial_tank_volume=initial_tank_volume,
        final_tank_volume=final_tank_volume,
        volume_change_ratio=volume_change_ratio,
    )


def _compute_stats(records: list[EpisodeEvalRecord]) -> EvalStats:
    import numpy as np

    if not records:
        raise ValueError("No episode records provided.")

    rewards = np.asarray([r.total_reward for r in records], dtype=np.float64)
    costs = np.asarray([r.total_energy_cost for r in records], dtype=np.float64)
    base_rewards = np.asarray([r.total_base_reward for r in records], dtype=np.float64)
    tank_penalties = np.asarray([r.total_tank_penalty for r in records], dtype=np.float64)
    lengths = np.asarray([r.episode_length for r in records], dtype=np.float64)
    initial_volumes = np.asarray([r.initial_tank_volume for r in records], dtype=np.float64)
    final_volumes = np.asarray([r.final_tank_volume for r in records], dtype=np.float64)
    volume_change_ratios = np.asarray([r.volume_change_ratio for r in records], dtype=np.float64)
    violations = np.asarray([r.hydraulic_violation for r in records], dtype=bool)
    full_horizon = np.asarray([r.full_horizon for r in records], dtype=bool)

    successful_mask = (~violations) & (lengths == 24.0)
    successful_costs = costs[successful_mask]
    successful_base_rewards = base_rewards[successful_mask]
    successful_tank_penalties = tank_penalties[successful_mask]
    successful_final_volumes = final_volumes[successful_mask]
    successful_volume_change_ratios = volume_change_ratios[successful_mask]

    def _safe_mean(values: np.ndarray) -> float | None:
        if values.size == 0:
            return None
        return float(np.mean(values))

    episode_count = len(records)
    violation_count = int(np.sum(violations))
    full_horizon_count = int(np.sum(full_horizon))
    successful_count = int(np.sum(successful_mask))

    return EvalStats(
        episode_count=episode_count,
        mean_total_reward=float(np.mean(rewards)),
        std_total_reward=float(np.std(rewards)),
        mean_total_energy_cost=float(np.mean(costs)),
        std_total_energy_cost=float(np.std(costs)),
        mean_total_base_reward=float(np.mean(base_rewards)),
        std_total_base_reward=float(np.std(base_rewards)),
        mean_total_tank_penalty=float(np.mean(tank_penalties)),
        std_total_tank_penalty=float(np.std(tank_penalties)),
        mean_episode_length=float(np.mean(lengths)),
        mean_initial_tank_volume=float(np.mean(initial_volumes)),
        mean_final_tank_volume=float(np.mean(final_volumes)),
        mean_volume_change_ratio=float(np.mean(volume_change_ratios)),
        std_volume_change_ratio=float(np.std(volume_change_ratios)),
        hydraulic_violation_count=violation_count,
        hydraulic_violation_rate=float(violation_count / episode_count),
        full_horizon_count=full_horizon_count,
        full_horizon_rate=float(full_horizon_count / episode_count),
        successful_episode_count=successful_count,
        successful_episode_rate=float(successful_count / episode_count),
        successful_mean_total_energy_cost=_safe_mean(successful_costs),
        successful_mean_total_base_reward=_safe_mean(successful_base_rewards),
        successful_mean_total_tank_penalty=_safe_mean(successful_tank_penalties),
        successful_mean_final_tank_volume=_safe_mean(successful_final_volumes),
        successful_mean_volume_change_ratio=_safe_mean(successful_volume_change_ratios),
    )


def _print_summary(*, args: argparse.Namespace, stats: EvalStats, elapsed_sec: float) -> None:
    """终端摘要输出（风格对齐 benchmark 脚本）。"""

    def _fmt(value: float | None) -> str:
        if value is None:
            return "null"
        return f"{value:.6f}"

    print("[policy evaluation summary]")
    print(
        "config: "
        f"algo={args.algo}, episodes={args.episodes}, seed={args.seed}, "
        f"scaling_mode={args.scaling_mode}, delta_time={args.delta_time}, "
        f"delta_space={args.delta_space}, r_benchmark={args.r_benchmark}, p_hydraulic={args.p_hydraulic}"
    )
    print(f"model_path={Path(args.model_path).resolve()}")
    print(f"net3_inp={Path(args.net3_inp).resolve()}")
    print("[all]")
    print(f"episode_count={stats.episode_count}")
    print(f"mean_total_reward={stats.mean_total_reward:.6f}")
    print(f"std_total_reward={stats.std_total_reward:.6f}")
    print(f"mean_total_energy_cost={stats.mean_total_energy_cost:.6f}")
    print(f"std_total_energy_cost={stats.std_total_energy_cost:.6f}")
    print(f"mean_total_base_reward={stats.mean_total_base_reward:.6f}")
    print(f"std_total_base_reward={stats.std_total_base_reward:.6f}")
    print(f"mean_total_tank_penalty={stats.mean_total_tank_penalty:.6f}")
    print(f"std_total_tank_penalty={stats.std_total_tank_penalty:.6f}")
    print(f"mean_episode_length={stats.mean_episode_length:.6f}")
    print(f"mean_initial_tank_volume={stats.mean_initial_tank_volume:.6f}")
    print(f"mean_final_tank_volume={stats.mean_final_tank_volume:.6f}")
    print(f"mean_volume_change_ratio={stats.mean_volume_change_ratio:.6f}")
    print(f"std_volume_change_ratio={stats.std_volume_change_ratio:.6f}")
    print(f"hydraulic_violation_count={stats.hydraulic_violation_count}")
    print(f"hydraulic_violation_rate={stats.hydraulic_violation_rate:.6%}")
    print(f"full_horizon_count={stats.full_horizon_count}")
    print(f"full_horizon_rate={stats.full_horizon_rate:.6%}")
    print("[successful]")
    print(f"successful_episode_count={stats.successful_episode_count}")
    print(f"successful_episode_rate={stats.successful_episode_rate:.6%}")
    print(f"successful_mean_total_energy_cost={_fmt(stats.successful_mean_total_energy_cost)}")
    print(f"successful_mean_total_base_reward={_fmt(stats.successful_mean_total_base_reward)}")
    print(f"successful_mean_total_tank_penalty={_fmt(stats.successful_mean_total_tank_penalty)}")
    print(f"successful_mean_final_tank_volume={_fmt(stats.successful_mean_final_tank_volume)}")
    print(f"successful_mean_volume_change_ratio={_fmt(stats.successful_mean_volume_change_ratio)}")
    print(f"elapsed_seconds={elapsed_sec:.3f}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON 并自动创建目录。"""

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
        from tqdm import tqdm
        from epanet_rl.env_wntr import Net3WntrEnv
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ModuleNotFoundError(
            f"Missing required dependency '{missing}'. "
            "Install project dependencies and retry."
        ) from exc

    model_file = _resolve_model_path(Path(args.model_path))
    model = _load_model(model_file)

    records: list[EpisodeEvalRecord] = []
    t0 = time.perf_counter()

    env = Net3WntrEnv(
        net3_inp_path=args.net3_inp,
        delta_time=float(args.delta_time),
        delta_space=float(args.delta_space),
        scaling_mode=str(args.scaling_mode),
        r_benchmark=float(args.r_benchmark),
        p_hydraulic=float(args.p_hydraulic),
    )

    try:
        # 评估过程显示 tqdm 进度条，方便长评估任务观察完成度。
        for ep_idx in tqdm(range(args.episodes), desc="Evaluating", unit="ep"):
            record = _run_single_episode(
                env=env,
                model=model,
                reset_seed=_episode_seed(args.seed, ep_idx),
            )
            records.append(record)
    finally:
        env.close()

    stats = _compute_stats(records)
    elapsed_sec = time.perf_counter() - t0
    _print_summary(args=args, stats=stats, elapsed_sec=elapsed_sec)

    if args.output_json is not None:
        payload = {
            "config": {
                "model_path": str(model_file.resolve()),
                "algo": str(args.algo),
                "episodes": int(args.episodes),
                "seed": None if args.seed is None else int(args.seed),
                "net3_inp": str(Path(args.net3_inp).resolve()),
                "delta_time": float(args.delta_time),
                "delta_space": float(args.delta_space),
                "scaling_mode": str(args.scaling_mode),
                "r_benchmark": float(args.r_benchmark),
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
