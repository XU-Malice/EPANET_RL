"""论文复现自动化：批量运行 Net3WntrEnv 上的 PPO / E-PPO 训练与评估。

设计目标：
1. 严格遵守当前阶段约束：不修改 `env_wntr.py` 主逻辑，只在 `scripts/` 下新增自动化脚本；
2. 将“论文明确给出的设定”与“当前仓库已有实现”以及“工程可运行近似”显式写入产物；
3. 支持服务器上一条命令拉起整套实验矩阵，并自动汇总结果到 JSON / CSV。

口径说明：
- 论文明确给出的：
  * 环境为 `Net3WntrEnv`；
  * 24 步决策、每步 1 小时；
  * PPO 超参数：actor/critic 网络、lr、gamma、clip、epochs、r_benchmark、p_hydraulic；
  * E-PPO 重点关注 `sigma=0.2`，PPO 可视为 `sigma=0`。
- 当前仓库已有实现：
  * `scripts/train_ppo_net3.py` 可训练 `ppo/eppo`；
  * `scripts/evaluate_policy_net3.py` 可输出 reward / energy / tank penalty / volume_change_ratio；
  * actor/critic 分离学习率已在训练脚本内生效。
- 为工程可运行做的近似：
  * 批量实验通过子进程串行调用现有训练/评估脚本，而不是重写训练框架；
  * E-PPO 的差异仍以仓库现有实现为准，即 `train_ppo_net3.py` 中对 entropy bonus 的参数映射；
  * summary CSV/JSON 中会记录这些说明，避免误把工程近似当成论文原始代码。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_ppo_net3.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_policy_net3.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reproduction_suite"

# 为了让脚本在服务器上直接运行，默认解释器使用当前 Python。
PYTHON_BIN = sys.executable


@dataclass(frozen=True)
class ExperimentSpec:
    """单个实验配置。

    这里把“实验矩阵中的一个格子”抽象成显式对象，
    好处是：后续写清单、落盘 summary、失败定位时都更稳定。
    """

    experiment_id: str
    algo: str
    sigma: float
    delta_time: float
    delta_space: float
    total_timesteps: int
    seed: int


@dataclass(frozen=True)
class ExperimentResult:
    """单个实验的最终汇总记录。"""

    experiment_id: str
    status: str
    algo: str
    sigma: float
    delta_time: float
    delta_space: float
    total_timesteps: int
    seed: int
    run_root: str
    train_run_dir: str | None
    train_summary_json: str | None
    eval_summary_json: str | None
    model_path: str | None
    elapsed_seconds: float
    train_returncode: int
    eval_returncode: int
    mean_total_reward: float | None
    mean_total_energy_cost: float | None
    mean_total_tank_penalty: float | None
    mean_volume_change_ratio: float | None
    hydraulic_violation_rate: float | None
    full_horizon_rate: float | None
    successful_episode_rate: float | None
    successful_mean_total_energy_cost: float | None
    successful_mean_total_tank_penalty: float | None
    notes: list[str]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch reproduction suite for Net3WntrEnv PPO / E-PPO experiments."
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["ppo", "eppo"],
        choices=["ppo", "eppo"],
        help="要跑哪些算法；默认同时跑 ppo 与 eppo。",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.3],
        help="实验矩阵中的 sigma 列表；默认 0 / 0.2 / 0.3。",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.3,
        help="同时作用于 delta_time 与 delta_space；按当前任务要求默认 0.3。",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="训练步数；默认 100000，可改为 200000。",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="至少传入一个 seed；建议多个 seed 形成对照。",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="每个模型训练完成后的评估 episode 数；默认 20。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="训练设备，透传给 train_ppo_net3.py。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="整套批量实验输出目录。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若输出目录已存在则继续写入；默认避免误覆盖。",
    )
    parser.add_argument(
        "--progress-log-interval",
        type=int,
        default=10_000,
        help="透传给训练脚本，便于服务器日志监控。",
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        default="z_score",
        choices=["none", "max_min", "z_score"],
        help=(
            "观测归一化模式。已知信息表明状态归一化有助于提升初始熵，"
            "因此默认用 z_score；这属于工程侧可运行选择，会被写入说明。"
        ),
    )
    parser.add_argument(
        "--net3-inp",
        type=Path,
        default=PROJECT_ROOT / "networks" / "Net3.inp",
        help="Net3 输入文件路径。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若某实验已有 eval_summary.json，则直接读取并跳过重跑。",
    )
    return parser


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_scripts_exist() -> None:
    missing = [str(path) for path in (TRAIN_SCRIPT, EVAL_SCRIPT) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required scripts not found: {missing}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive.")
    if args.delta < 0.0:
        raise ValueError("--delta must be >= 0.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if not args.net3_inp.exists():
        raise FileNotFoundError(f"Net3 INP file not found: {args.net3_inp}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {args.output_dir}. "
            "Use --force to continue writing into it."
        )


def _build_experiment_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    # 明确区分论文重点与实验矩阵：
    # - ppo 本质上应视作 sigma=0；
    # - 但用户要求矩阵支持 sigma=0/0.2/0.3，
    #   因此这里保留所有组合，并在 notes 中提醒 ppo 会忽略非零 sigma。
    for algo, sigma, seed in itertools.product(args.algos, args.sigmas, args.seeds):
        experiment_id = (
            f"algo-{algo}__sigma-{sigma:g}__delta-{args.delta:g}"
            f"__steps-{args.timesteps}__seed-{seed}"
        )
        specs.append(
            ExperimentSpec(
                experiment_id=experiment_id,
                algo=str(algo),
                sigma=float(sigma),
                delta_time=float(args.delta),
                delta_space=float(args.delta),
                total_timesteps=int(args.timesteps),
                seed=int(seed),
            )
        )
    return specs


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_command(*, command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("[command]\n")
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_new_train_run_dir(base_output_dir: Path, started_at: float, seed: int, algo: str) -> Path | None:
    # 训练脚本会在输出目录下创建 run_时间戳_algo_seedX 目录。
    # 这里不依赖训练日志字符串解析，而是从文件系统中找最新目录，稳健性更高。
    candidates: list[Path] = []
    pattern = f"run_*_{algo}_seed{seed}"
    if base_output_dir.exists():
        for candidate in base_output_dir.glob(pattern):
            if candidate.is_dir() and candidate.stat().st_mtime >= started_at - 1.0:
                candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_eval_metrics(eval_payload: dict[str, Any]) -> dict[str, float | None]:
    stats = eval_payload.get("stats", {})
    return {
        "mean_total_reward": stats.get("mean_total_reward"),
        "mean_total_energy_cost": stats.get("mean_total_energy_cost"),
        "mean_total_tank_penalty": stats.get("mean_total_tank_penalty"),
        "mean_volume_change_ratio": stats.get("mean_volume_change_ratio"),
        "hydraulic_violation_rate": stats.get("hydraulic_violation_rate"),
        "full_horizon_rate": stats.get("full_horizon_rate"),
        "successful_episode_rate": stats.get("successful_episode_rate"),
        "successful_mean_total_energy_cost": stats.get("successful_mean_total_energy_cost"),
        "successful_mean_total_tank_penalty": stats.get("successful_mean_total_tank_penalty"),
    }


def _run_one_experiment(
    *,
    spec: ExperimentSpec,
    suite_dir: Path,
    args: argparse.Namespace,
) -> ExperimentResult:
    run_root = suite_dir / spec.experiment_id
    run_root.mkdir(parents=True, exist_ok=True)

    train_output_dir = run_root / "train_outputs"
    train_log = run_root / "train.log"
    eval_log = run_root / "eval.log"
    eval_summary_json = run_root / "eval_summary.json"
    experiment_meta_json = run_root / "experiment_meta.json"

    notes: list[str] = [
        "论文明确给出：Net3WntrEnv、24步决策、每步1小时、PPO超参数、r_benchmark=406.54、p_hydraulic=-200。",
        "当前仓库已有实现：train_ppo_net3.py 负责训练，evaluate_policy_net3.py 负责评估并输出 reward/energy/tank penalty/volume_change_ratio。",
        "工程可运行近似：批量复现通过子进程串行调用现有脚本完成，不改写 env_wntr.py 主逻辑。",
    ]
    if spec.algo == "ppo" and spec.sigma != 0.0:
        notes.append("工程提示：当前仓库中 algo=ppo 时会忽略非零 sigma，并强制 effective_ent_coef=0。")
    if args.scaling_mode == "z_score":
        notes.append("工程提示：根据已知信息“状态归一化有助于提高初始熵”，批量脚本默认 scaling_mode=z_score。")

    _json_dump(
        experiment_meta_json,
        {
            "spec": asdict(spec),
            "paper_given": {
                "environment": "Net3WntrEnv",
                "episode_steps": 24,
                "step_hours": 1,
                "actor_network": [256, 128, 64],
                "critic_network": [256, 128, 1],
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "gamma": 0.9,
                "clip_range": 0.2,
                "epochs": 10,
                "r_benchmark": 406.54,
                "p_hydraulic": -200,
                "eppo_focus_sigma": 0.2,
            },
            "repo_existing": {
                "train_script": str(TRAIN_SCRIPT.relative_to(PROJECT_ROOT)),
                "eval_script": str(EVAL_SCRIPT.relative_to(PROJECT_ROOT)),
                "separate_actor_critic_lr": True,
                "main_environment_fixed": "Net3WntrEnv",
            },
            "engineering_approximation": {
                "suite_runner": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "batch_execution_mode": "sequential subprocess",
                "scaling_mode": args.scaling_mode,
                "sigma_handling": "delegate to existing train_ppo_net3.py semantics",
            },
            "notes": notes,
        },
    )

    # 跳过逻辑：适合服务器断点续跑。
    if args.skip_existing and eval_summary_json.exists():
        eval_payload = _read_json(eval_summary_json)
        metrics = _extract_eval_metrics(eval_payload)
        train_summary_json = run_root / "train_summary_snapshot.json"
        train_summary_payload = _read_json(train_summary_json) if train_summary_json.exists() else {}
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            status="skipped_existing",
            algo=spec.algo,
            sigma=spec.sigma,
            delta_time=spec.delta_time,
            delta_space=spec.delta_space,
            total_timesteps=spec.total_timesteps,
            seed=spec.seed,
            run_root=str(run_root.resolve()),
            train_run_dir=train_summary_payload.get("run_dir"),
            train_summary_json=str(train_summary_json.resolve()) if train_summary_json.exists() else None,
            eval_summary_json=str(eval_summary_json.resolve()),
            model_path=train_summary_payload.get("model_path"),
            elapsed_seconds=0.0,
            train_returncode=0,
            eval_returncode=0,
            notes=notes + ["跳过重跑：检测到既有 eval_summary.json。"],
            **metrics,
        )

    train_started = time.time()
    train_command = [
        PYTHON_BIN,
        str(TRAIN_SCRIPT),
        "--algo",
        spec.algo,
        "--sigma",
        str(spec.sigma),
        "--total-timesteps",
        str(spec.total_timesteps),
        "--seed",
        str(spec.seed),
        "--delta-time",
        str(spec.delta_time),
        "--delta-space",
        str(spec.delta_space),
        "--scaling-mode",
        args.scaling_mode,
        "--r-benchmark",
        "406.54",
        "--p-hydraulic",
        "-200",
        "--actor-lr",
        "1e-4",
        "--critic-lr",
        "1e-3",
        "--gamma",
        "0.9",
        "--clip-range",
        "0.2",
        "--n-epochs",
        "10",
        "--device",
        args.device,
        "--output-dir",
        str(train_output_dir),
        "--progress-log-interval",
        str(args.progress_log_interval),
        "--net3-inp",
        str(args.net3_inp),
    ]
    train_returncode = _run_command(command=train_command, log_path=train_log)

    train_run_dir = _find_new_train_run_dir(train_output_dir, train_started, spec.seed, spec.algo)
    train_summary_json_path = train_run_dir / "train_summary.json" if train_run_dir else None
    train_summary_payload: dict[str, Any] = {}
    model_path: str | None = None

    if train_run_dir and train_summary_json_path and train_summary_json_path.exists():
        train_summary_payload = _read_json(train_summary_json_path)
        model_path = train_summary_payload.get("model_path")
        # 在 run_root 下复制一份摘要快照，方便 suite 层统一查看。
        _json_dump(run_root / "train_summary_snapshot.json", train_summary_payload)

    if train_returncode != 0 or not model_path:
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            status="train_failed",
            algo=spec.algo,
            sigma=spec.sigma,
            delta_time=spec.delta_time,
            delta_space=spec.delta_space,
            total_timesteps=spec.total_timesteps,
            seed=spec.seed,
            run_root=str(run_root.resolve()),
            train_run_dir=str(train_run_dir.resolve()) if train_run_dir else None,
            train_summary_json=str(train_summary_json_path.resolve()) if train_summary_json_path and train_summary_json_path.exists() else None,
            eval_summary_json=None,
            model_path=model_path,
            elapsed_seconds=float(time.time() - train_started),
            train_returncode=int(train_returncode),
            eval_returncode=-1,
            mean_total_reward=None,
            mean_total_energy_cost=None,
            mean_total_tank_penalty=None,
            mean_volume_change_ratio=None,
            hydraulic_violation_rate=None,
            full_horizon_rate=None,
            successful_episode_rate=None,
            successful_mean_total_energy_cost=None,
            successful_mean_total_tank_penalty=None,
            notes=notes + ["训练失败或未找到 model_path，请查看 train.log。"],
        )

    eval_command = [
        PYTHON_BIN,
        str(EVAL_SCRIPT),
        "--model-path",
        model_path,
        "--algo",
        spec.algo,
        "--episodes",
        str(args.eval_episodes),
        "--seed",
        str(spec.seed),
        "--delta-time",
        str(spec.delta_time),
        "--delta-space",
        str(spec.delta_space),
        "--scaling-mode",
        args.scaling_mode,
        "--r-benchmark",
        "406.54",
        "--p-hydraulic",
        "-200",
        "--net3-inp",
        str(args.net3_inp),
        "--output-json",
        str(eval_summary_json),
    ]
    eval_returncode = _run_command(command=eval_command, log_path=eval_log)
    elapsed_seconds = float(time.time() - train_started)

    if eval_returncode != 0 or not eval_summary_json.exists():
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            status="eval_failed",
            algo=spec.algo,
            sigma=spec.sigma,
            delta_time=spec.delta_time,
            delta_space=spec.delta_space,
            total_timesteps=spec.total_timesteps,
            seed=spec.seed,
            run_root=str(run_root.resolve()),
            train_run_dir=str(train_run_dir.resolve()) if train_run_dir else None,
            train_summary_json=str(train_summary_json_path.resolve()) if train_summary_json_path and train_summary_json_path.exists() else None,
            eval_summary_json=str(eval_summary_json.resolve()) if eval_summary_json.exists() else None,
            model_path=model_path,
            elapsed_seconds=elapsed_seconds,
            train_returncode=int(train_returncode),
            eval_returncode=int(eval_returncode),
            mean_total_reward=None,
            mean_total_energy_cost=None,
            mean_total_tank_penalty=None,
            mean_volume_change_ratio=None,
            hydraulic_violation_rate=None,
            full_horizon_rate=None,
            successful_episode_rate=None,
            successful_mean_total_energy_cost=None,
            successful_mean_total_tank_penalty=None,
            notes=notes + ["评估失败，请查看 eval.log。"],
        )

    eval_payload = _read_json(eval_summary_json)
    metrics = _extract_eval_metrics(eval_payload)
    return ExperimentResult(
        experiment_id=spec.experiment_id,
        status="ok",
        algo=spec.algo,
        sigma=spec.sigma,
        delta_time=spec.delta_time,
        delta_space=spec.delta_space,
        total_timesteps=spec.total_timesteps,
        seed=spec.seed,
        run_root=str(run_root.resolve()),
        train_run_dir=str(train_run_dir.resolve()) if train_run_dir else None,
        train_summary_json=str(train_summary_json_path.resolve()) if train_summary_json_path and train_summary_json_path.exists() else None,
        eval_summary_json=str(eval_summary_json.resolve()),
        model_path=model_path,
        elapsed_seconds=elapsed_seconds,
        train_returncode=int(train_returncode),
        eval_returncode=int(eval_returncode),
        notes=notes,
        **metrics,
    )


def _write_summary_csv(path: Path, results: list[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in results]
    if not rows:
        rows = [{}]
    # notes 在 CSV 中转成 JSON 字符串，方便后处理。
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        if "notes" in normalized:
            normalized["notes"] = json.dumps(normalized["notes"], ensure_ascii=False)
        normalized_rows.append(normalized)

    fieldnames = list(normalized_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_summary_json(
    *,
    path: Path,
    suite_dir: Path,
    args: argparse.Namespace,
    specs: list[ExperimentSpec],
    results: list[ExperimentResult],
    started_at: float,
) -> None:
    ok_count = sum(1 for item in results if item.status == "ok")
    payload = {
        "suite_metadata": {
            "suite_dir": str(suite_dir.resolve()),
            "started_at_epoch": started_at,
            "finished_at_epoch": time.time(),
            "python": PYTHON_BIN,
            "project_root": str(PROJECT_ROOT.resolve()),
        },
        "paper_given": {
            "environment": "Net3WntrEnv",
            "episode_steps": 24,
            "step_hours": 1,
            "actor_network": [256, 128, 64],
            "critic_network": [256, 128, 1],
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "gamma": 0.9,
            "clip_range": 0.2,
            "epochs": 10,
            "r_benchmark": 406.54,
            "p_hydraulic": -200,
            "ppo_equivalent_sigma": 0.0,
            "eppo_focus_sigma": 0.2,
        },
        "repo_existing": {
            "train_script": str(TRAIN_SCRIPT.relative_to(PROJECT_ROOT)),
            "eval_script": str(EVAL_SCRIPT.relative_to(PROJECT_ROOT)),
            "main_environment_logic_modified": False,
            "known_behavior": "current models may exploit tank volume to lower energy cost",
            "current_focus": "penalty form and PPO/E-PPO comparison",
        },
        "engineering_approximation": {
            "batch_runner": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "execution_mode": "sequential subprocess",
            "scaling_mode": args.scaling_mode,
            "default_delta_applied_to_both_time_and_space": float(args.delta),
        },
        "suite_config": {
            "algos": list(args.algos),
            "sigmas": [float(x) for x in args.sigmas],
            "delta": float(args.delta),
            "timesteps": int(args.timesteps),
            "seeds": [int(x) for x in args.seeds],
            "eval_episodes": int(args.eval_episodes),
            "device": str(args.device),
            "net3_inp": str(Path(args.net3_inp).resolve()),
            "progress_log_interval": int(args.progress_log_interval),
        },
        "suite_stats": {
            "experiment_count": len(specs),
            "ok_count": ok_count,
            "failed_count": len(results) - ok_count,
        },
        "experiments": [asdict(item) for item in results],
    }
    _json_dump(path, payload)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _ensure_scripts_exist()
    _validate_args(args)

    suite_dir = args.output_dir / f"suite_{_timestamp()}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    specs = _build_experiment_specs(args)
    manifest_json = suite_dir / "suite_manifest.json"
    _json_dump(
        manifest_json,
        {
            "specs": [asdict(item) for item in specs],
            "paper_given": {
                "environment": "Net3WntrEnv",
                "episode_steps": 24,
                "step_hours": 1,
                "note": "不修改 env_wntr.py 主逻辑；训练与评估完全复用现有 scripts。",
            },
            "repo_existing": {
                "train_script": str(TRAIN_SCRIPT.relative_to(PROJECT_ROOT)),
                "eval_script": str(EVAL_SCRIPT.relative_to(PROJECT_ROOT)),
            },
            "engineering_approximation": {
                "scaling_mode": args.scaling_mode,
                "batch_runner": str(Path(__file__).relative_to(PROJECT_ROOT)),
            },
        },
    )

    print(f"[suite] output_dir={suite_dir.resolve()}")
    print(f"[suite] experiment_count={len(specs)}")
    results: list[ExperimentResult] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[suite] ({index}/{len(specs)}) start {spec.experiment_id} "
            f"algo={spec.algo} sigma={spec.sigma} seed={spec.seed} timesteps={spec.total_timesteps}"
        )
        result = _run_one_experiment(spec=spec, suite_dir=suite_dir, args=args)
        results.append(result)
        print(
            f"[suite] ({index}/{len(specs)}) done {spec.experiment_id} "
            f"status={result.status} reward={result.mean_total_reward} energy={result.mean_total_energy_cost}"
        )

    summary_json = suite_dir / "suite_summary.json"
    summary_csv = suite_dir / "suite_summary.csv"
    _write_summary_json(
        path=summary_json,
        suite_dir=suite_dir,
        args=args,
        specs=specs,
        results=results,
        started_at=started_at,
    )
    _write_summary_csv(summary_csv, results)

    ok_count = sum(1 for item in results if item.status == "ok")
    print(f"[suite] finished ok={ok_count}/{len(results)}")
    print(f"[suite] summary_json={summary_json.resolve()}")
    print(f"[suite] summary_csv={summary_csv.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[suite][error] {exc}", file=sys.stderr)
        raise
