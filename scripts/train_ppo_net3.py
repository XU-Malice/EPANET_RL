"""论文复现导向的 PPO / E-PPO 训练入口（Net3WntrEnv）。

教学导读：
1. 论文明确给出的：
   - PPO 关键超参数（网络结构、lr、gamma、clip、epochs 等）；
   - E-PPO 与 PPO 的核心差别可通过 entropy bonus 系数体现。
2. 当前仓库实现：
   - 固定主线环境为 `Net3WntrEnv`；
   - 通过自定义 policy + PPO 子类保证 actor/critic 分离学习率持续生效；
   - 训练过程同时保留 SB3 日志与脚本级 JSON 审计产物。
3. 工程可运行增强：
   - 双模式进度反馈：TTY 使用 tqdm，日志重定向时输出静态 `[progress]` 行。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Make `src/` importable when running from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class TrainSummary:
    """训练完成后写入 JSON 的摘要结构。"""

    total_timesteps: int
    elapsed_seconds: float
    model_path: str
    run_dir: str
    seed: int
    net3_inp: str
    delta_time: float
    delta_space: float
    scaling_mode: str
    r_benchmark: float
    p_hydraulic: float
    device: str
    algo: str
    sigma: float
    effective_ent_coef: float
    actor_lr: float
    critic_lr: float
    gamma: float
    clip_range: float
    n_epochs: int
    n_steps: int
    batch_size: int
    vf_coef: float
    max_grad_norm: float
    optimizer_lrs_json: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train paper-oriented PPO / E-PPO on Net3WntrEnv."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "eppo"],
        help="Training algorithm: ppo (sigma forced to 0) or eppo (entropy bonus enabled by sigma).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.01,
        help="Entropy-bonus coefficient for E-PPO. For PPO, this is ignored and forced to 0.",
    )
    parser.add_argument(
        "--net3-inp",
        type=Path,
        default=PROJECT_ROOT / "networks" / "Net3.inp",
        help="Path to Net3 INP file.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
        help="r_benchmark used by reward shaping (paper default: 406.54).",
    )
    parser.add_argument(
        "--p-hydraulic",
        type=float,
        default=-200.0,
        help="Hydraulic violation penalty passed to Net3WntrEnv (paper default: -200).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for PPO (default: cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ppo_net3",
        help="Directory to store model and logs.",
    )
    parser.add_argument(
        "--progress-log-interval",
        type=int,
        default=10_000,
        help="Static progress log interval in timesteps for non-interactive mode.",
    )

    # 论文对齐优先的默认值（仍保留 CLI 可覆盖能力）。
    parser.add_argument("--actor-lr", type=float, default=1e-4, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="Critic learning rate.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update.")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=192,
        help="Rollout steps per update (single-env).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for PPO updates.",
    )
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    return parser


def _timestamp() -> str:
    """用于 run 目录命名，便于多次实验并存。"""

    return time.strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON 并自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _set_global_seeds(seed: int) -> None:
    """统一设置 Python / NumPy / Torch 随机种子。"""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    """训练主流程入口。

    设计重点：
    - 先做参数校验，再构建环境与模型；
    - 保持 SB3 主训练循环不改写，仅通过 callback 扩展进度显示；
    - 训练前后写入配置和学习率审计文件，便于复现实验追溯。
    """

    parser = _build_parser()
    args = parser.parse_args()

    if args.total_timesteps <= 0:
        raise ValueError(f"--total-timesteps must be positive, got {args.total_timesteps}.")
    if args.n_steps <= 0:
        raise ValueError(f"--n-steps must be positive, got {args.n_steps}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.n_epochs <= 0:
        raise ValueError(f"--n-epochs must be positive, got {args.n_epochs}.")
    if not (0.0 <= args.clip_range <= 1.0):
        raise ValueError(f"--clip-range must be in [0,1], got {args.clip_range}.")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError(f"--gamma must be in (0,1], got {args.gamma}.")
    if args.actor_lr <= 0.0:
        raise ValueError(f"--actor-lr must be positive, got {args.actor_lr}.")
    if args.critic_lr <= 0.0:
        raise ValueError(f"--critic-lr must be positive, got {args.critic_lr}.")
    if args.p_hydraulic >= 0.0:
        raise ValueError(f"--p-hydraulic must be negative, got {args.p_hydraulic}.")
    if args.sigma < 0.0:
        raise ValueError(f"--sigma must be >= 0, got {args.sigma}.")
    if args.progress_log_interval <= 0:
        raise ValueError(
            f"--progress-log-interval must be positive, got {args.progress_log_interval}."
        )

    # PPO 与 E-PPO 的差异在这里显式映射：
    # - PPO：不使用 entropy bonus（ent_coef=0）
    # - E-PPO：ent_coef=sigma
    if args.algo == "ppo":
        effective_ent_coef = 0.0
        if args.sigma != 0.0:
            print(
                f"[train] warning: --algo=ppo ignores --sigma={args.sigma}; using sigma=0.0"
            )
    else:
        effective_ent_coef = float(args.sigma)

    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ModuleNotFoundError(
            f"Missing required dependency '{missing}'. "
            "Please install stable-baselines3 (and torch/tqdm) before running PPO training."
        ) from exc

    try:
        from epanet_rl.env_wntr import Net3WntrEnv
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ModuleNotFoundError(
            f"Missing required dependency '{missing}' for Net3WntrEnv."
        ) from exc

    if not args.net3_inp.exists():
        raise FileNotFoundError(f"Net3 INP file not found: {args.net3_inp}")

    _set_global_seeds(int(args.seed))

    run_dir = args.output_dir / f"run_{_timestamp()}_{args.algo}_seed{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "ppo_net3_model"
    monitor_csv = run_dir / "vec_monitor.csv"
    sb3_log_dir = run_dir / "sb3_logs"
    config_json = run_dir / "train_config.json"
    summary_json = run_dir / "train_summary.json"
    optimizer_lrs_json = run_dir / "optimizer_lrs.json"

    # 论文网络结构对齐：
    # - actor: [256, 128, 64]
    # - critic: [256, 128, 1]
    #   在 SB3 中 value head 是隐式单输出层，因此这里写 vf=[256,128]。
    policy_kwargs = {
        "net_arch": {
            "pi": [256, 128, 64],
            "vf": [256, 128],
        },
        "activation_fn": torch.nn.ReLU,
    }

    class PaperLikeActorCriticPolicy(ActorCriticPolicy):
        """支持 actor/critic 分离学习率的策略类。"""

        def __init__(
            self, *policy_args, actor_lr: float, critic_lr: float, **policy_kwargs_local
        ):
            self._actor_lr = float(actor_lr)
            self._critic_lr = float(critic_lr)
            super().__init__(*policy_args, **policy_kwargs_local)

        @property
        def actor_lr(self) -> float:
            return self._actor_lr

        @property
        def critic_lr(self) -> float:
            return self._critic_lr

        def _build(self, lr_schedule) -> None:  # type: ignore[override]
            # 先让父类构建网络，再接管 optimizer（按参数组分别设置 lr）。
            super()._build(lr_schedule)

            actor_params = list(self.mlp_extractor.policy_net.parameters()) + list(
                self.action_net.parameters()
            )
            critic_params = list(self.mlp_extractor.value_net.parameters()) + list(
                self.value_net.parameters()
            )
            actor_ids = {id(p) for p in actor_params}
            critic_ids = {id(p) for p in critic_params}
            assigned_ids = actor_ids | critic_ids

            # 若存在共享参数（如共享特征提取层），这里跟随 actor_lr。
            shared_params = [p for p in self.parameters() if id(p) not in assigned_ids]

            param_groups: list[dict[str, Any]] = []
            if actor_params:
                param_groups.append(
                    {"params": actor_params, "lr": self._actor_lr, "group_name": "actor"}
                )
            if critic_params:
                param_groups.append(
                    {"params": critic_params, "lr": self._critic_lr, "group_name": "critic"}
                )
            if shared_params:
                param_groups.append(
                    {"params": shared_params, "lr": self._actor_lr, "group_name": "shared"}
                )

            self.optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)
            self.apply_separate_learning_rates()

        def apply_separate_learning_rates(self) -> None:
            """将参数组学习率强制写回目标值，防止外部统一覆盖。"""

            if self.optimizer is None:
                return
            for group in self.optimizer.param_groups:
                group_name = str(group.get("group_name", "actor"))
                if group_name == "critic":
                    group["lr"] = self._critic_lr
                else:
                    group["lr"] = self._actor_lr

        def current_group_lrs(self) -> dict[str, float]:
            """导出当前各参数组 lr，供日志与审计使用。"""

            if self.optimizer is None:
                return {}
            result: dict[str, float] = {}
            for idx, group in enumerate(self.optimizer.param_groups):
                group_name = str(group.get("group_name", f"group_{idx}"))
                result[group_name] = float(group["lr"])
            return result

    class PaperLikePPO(PPO):
        """覆盖 SB3 学习率更新，保证分离 lr 持续生效。"""

        def _update_learning_rate(self, optimizers) -> None:  # type: ignore[override]
            if isinstance(self.policy, PaperLikeActorCriticPolicy):
                # 关键点：每次更新前都把 actor/critic lr 重新写回，避免被统一 lr 改写。
                self.policy.apply_separate_learning_rates()
                lrs = self.policy.current_group_lrs()
                self.logger.record("train/learning_rate", float(self.policy.actor_lr))
                self.logger.record(
                    "train/actor_learning_rate",
                    float(lrs.get("actor", self.policy.actor_lr)),
                )
                self.logger.record(
                    "train/critic_learning_rate",
                    float(lrs.get("critic", self.policy.critic_lr)),
                )
                if "shared" in lrs:
                    self.logger.record("train/shared_learning_rate", float(lrs["shared"]))
                return
            super()._update_learning_rate(optimizers)

    class TqdmProgressCallback(BaseCallback):
        """训练进度反馈 callback（双模式）。

        说明：
        - 交互终端（TTY）：使用 tqdm 动态进度条，显示速度/ETA 更直观；
        - 非交互场景（如 `> log 2>&1`）：动态刷新不友好，
          改为固定步长输出静态 [progress] 行，便于 `tail -f` 实时查看。
        """

        def __init__(
            self,
            total_timesteps: int,
            *,
            use_tqdm: bool,
            progress_log_interval: int,
        ) -> None:
            super().__init__(verbose=0)
            self.total_timesteps = int(total_timesteps)
            self.use_tqdm = bool(use_tqdm)
            self.progress_log_interval = int(progress_log_interval)

            self._pbar: Any | None = None
            self._last_num_timesteps = 0
            self._iterations = 0
            self._start_time = 0.0

            self._next_log_step = self.progress_log_interval
            self._last_logged_step = 0

        def _on_training_start(self) -> None:
            self._start_time = time.perf_counter()
            self._last_num_timesteps = int(self.model.num_timesteps)

            if self.use_tqdm:
                self._pbar = tqdm(
                    total=self.total_timesteps,
                    desc="Training",
                    unit="ts",
                    dynamic_ncols=True,
                    leave=False,
                    disable=False,
                )
                if self._last_num_timesteps > 0:
                    already = min(self._last_num_timesteps, self.total_timesteps)
                    self._pbar.update(already)

            # 兼容从已有步数继续训练：把静态日志阈值推进到当前步数之后。
            if self._last_num_timesteps > 0:
                k = self._last_num_timesteps // self.progress_log_interval
                self._next_log_step = (k + 1) * self.progress_log_interval

        def _safe_ep_metrics(self) -> tuple[float | None, float | None]:
            ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
            if not ep_info_buffer:
                return None, None
            try:
                rewards = [float(item.get("r")) for item in ep_info_buffer if "r" in item]
                lengths = [float(item.get("l")) for item in ep_info_buffer if "l" in item]
                rew_mean = (sum(rewards) / len(rewards)) if rewards else None
                len_mean = (sum(lengths) / len(lengths)) if lengths else None
                return rew_mean, len_mean
            except Exception:
                # 读取不到统计时降级，不影响训练。
                return None, None

        def _format_progress_line(self, current: int) -> str:
            elapsed = max(0.0, time.perf_counter() - self._start_time)
            completed = max(1, current)
            speed = completed / elapsed if elapsed > 1e-9 else 0.0
            remaining = max(0, self.total_timesteps - current)
            eta = (remaining / speed) if speed > 1e-9 else float("inf")
            percent = min(100.0, 100.0 * current / self.total_timesteps)

            rew_mean, len_mean = self._safe_ep_metrics()
            rew_text = f"{rew_mean:.3f}" if rew_mean is not None else "n/a"
            len_text = f"{len_mean:.2f}" if len_mean is not None else "n/a"
            eta_text = f"{eta:.1f}" if eta != float("inf") else "inf"
            return (
                f"[progress] {current}/{self.total_timesteps} ({percent:.2f}%) "
                f"elapsed={elapsed:.1f}s eta={eta_text}s iter={self._iterations} "
                f"ep_rew_mean={rew_text} ep_len_mean={len_text}"
            )

        def _refresh_tqdm_postfix(self) -> None:
            if self._pbar is None:
                return
            rew_mean, len_mean = self._safe_ep_metrics()
            postfix: dict[str, str] = {"iter": str(self._iterations)}
            if rew_mean is not None:
                postfix["ep_rew_mean"] = f"{rew_mean:.3f}"
            if len_mean is not None:
                postfix["ep_len_mean"] = f"{len_mean:.2f}"
            self._pbar.set_postfix(postfix, refresh=False)

        def _maybe_emit_static_progress(self, current: int) -> None:
            if self.use_tqdm:
                return
            while current >= self._next_log_step and self._next_log_step <= self.total_timesteps:
                step_to_log = min(current, self.total_timesteps)
                print(self._format_progress_line(step_to_log))
                self._last_logged_step = step_to_log
                self._next_log_step += self.progress_log_interval

        def _on_step(self) -> bool:
            current = int(self.model.num_timesteps)
            delta = max(0, current - self._last_num_timesteps)

            if self.use_tqdm and self._pbar is not None and delta > 0:
                remaining = self.total_timesteps - int(self._pbar.n)
                if remaining > 0:
                    self._pbar.update(min(delta, remaining))
                self._refresh_tqdm_postfix()

            if delta > 0:
                self._last_num_timesteps = current
                self._maybe_emit_static_progress(current)
            return True

        def _on_rollout_end(self) -> None:
            self._iterations += 1
            if self.use_tqdm:
                self._refresh_tqdm_postfix()

        def _on_training_end(self) -> None:
            final_step = int(self.model.num_timesteps)

            if self.use_tqdm and self._pbar is not None:
                if int(self._pbar.n) < self.total_timesteps:
                    self._pbar.update(self.total_timesteps - int(self._pbar.n))
                self._refresh_tqdm_postfix()
                self._pbar.close()
                self._pbar = None
            else:
                # 非交互模式下，确保最后有一条终态 progress 行。
                final_to_log = min(final_step, self.total_timesteps)
                if final_to_log > self._last_logged_step:
                    print(self._format_progress_line(final_to_log))
                    self._last_logged_step = final_to_log

    config_payload = {
        "net3_inp": str(Path(args.net3_inp).resolve()),
        "total_timesteps": int(args.total_timesteps),
        "seed": int(args.seed),
        "delta_time": float(args.delta_time),
        "delta_space": float(args.delta_space),
        "scaling_mode": str(args.scaling_mode),
        "r_benchmark": float(args.r_benchmark),
        "p_hydraulic": float(args.p_hydraulic),
        "device": str(args.device),
        "output_dir": str(Path(args.output_dir).resolve()),
        "run_dir": str(run_dir.resolve()),
        "env_name": "Net3WntrEnv",
        "algorithm": str(args.algo).upper(),
        "policy": "PaperLikeActorCriticPolicy",
        "ppo_impl": "PaperLikePPO",
        "policy_kwargs": {
            "net_arch": {"pi": [256, 128, 64], "vf": [256, 128]},
            "activation_fn": "ReLU",
        },
        "ppo_hyperparameters": {
            "actor_lr": float(args.actor_lr),
            "critic_lr": float(args.critic_lr),
            "gamma": float(args.gamma),
            "clip_range": float(args.clip_range),
            "n_epochs": int(args.n_epochs),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "sigma": float(args.sigma),
            "effective_ent_coef": float(effective_ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
            # 传给 SB3 的 learning_rate 仅作占位；
            # 真正生效的是 PaperLikePPO + param_groups 中的分离 lr。
            "sb3_learning_rate_argument": float(args.actor_lr),
            "progress_log_interval": int(args.progress_log_interval),
        },
    }
    _write_json(config_json, config_payload)

    print(f"[train] starting {str(args.algo).upper()} training on Net3WntrEnv")
    print(
        "[train] env: "
        f"delta_time={args.delta_time}, delta_space={args.delta_space}, "
        f"scaling_mode={args.scaling_mode}, r_benchmark={args.r_benchmark}, "
        f"p_hydraulic={args.p_hydraulic}"
    )
    print(
        "[train] ppo: "
        f"actor_lr={args.actor_lr}, critic_lr={args.critic_lr}, gamma={args.gamma}, "
        f"clip_range={args.clip_range}, n_epochs={args.n_epochs}, "
        f"n_steps={args.n_steps}, batch_size={args.batch_size}, device={args.device}"
    )
    print(
        f"[train] entropy: algo={args.algo}, sigma={args.sigma}, effective_ent_coef={effective_ent_coef}"
    )
    print("[train] network: actor=[256,128,64], critic=[256,128,1]")
    print(f"[train] net3_inp={Path(args.net3_inp).resolve()}")
    print(f"[train] run_dir={run_dir.resolve()}")

    def _make_env():
        # 训练入口固定使用主线环境 Net3WntrEnv。
        env = Net3WntrEnv(
            net3_inp_path=args.net3_inp,
            delta_time=float(args.delta_time),
            delta_space=float(args.delta_space),
            scaling_mode=str(args.scaling_mode),
            r_benchmark=float(args.r_benchmark),
            p_hydraulic=float(args.p_hydraulic),
        )
        episode_steps = int(getattr(env, "EPISODE_STEPS", -1))
        # 论文语义保护：若环境不是 24 步 horizon，直接报错。
        if episode_steps != 24:
            raise ValueError(
                f"Paper setup expects maximum episode length 24, got EPISODE_STEPS={episode_steps}."
            )
        env.reset(seed=int(args.seed))
        return Monitor(env)

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecMonitor(venv=vec_env, filename=str(monitor_csv))

    model = PaperLikePPO(
        policy=PaperLikeActorCriticPolicy,
        env=vec_env,
        seed=int(args.seed),
        verbose=1,
        device=str(args.device),
        # 占位 learning_rate：真实学习率由分组参数与 _update_learning_rate() 保证。
        learning_rate=float(args.actor_lr),
        gamma=float(args.gamma),
        clip_range=float(args.clip_range),
        n_epochs=int(args.n_epochs),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        ent_coef=float(effective_ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        policy_kwargs={
            **policy_kwargs,
            "actor_lr": float(args.actor_lr),
            "critic_lr": float(args.critic_lr),
        },
    )
    model.set_logger(configure(folder=str(sb3_log_dir), format_strings=["stdout", "csv"]))

    initial_group_lrs: dict[str, float] = {}
    if isinstance(model.policy, PaperLikeActorCriticPolicy):
        initial_group_lrs = model.policy.current_group_lrs()
    print(f"[train] optimizer_lrs_initial={initial_group_lrs}")

    # 双模式进度反馈：
    # - TTY 下：tqdm 动态刷新最直观；
    # - 非 TTY（日志重定向）下：动态光标控制不友好，改为静态 progress 行。
    tqdm_interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())
    print("[train] tqdm progress enabled")
    if not tqdm_interactive:
        print(
            "[train] non-interactive output detected; "
            "will emit static [progress] lines for log-friendly monitoring."
        )
    progress_callback = TqdmProgressCallback(
        total_timesteps=int(args.total_timesteps),
        use_tqdm=tqdm_interactive,
        progress_log_interval=int(args.progress_log_interval),
    )

    t0 = time.perf_counter()
    try:
        # 使用 callback 注入进度显示，不改 SB3 主训练循环。
        model.learn(total_timesteps=int(args.total_timesteps), callback=progress_callback)
        model.save(str(model_path))
    finally:
        vec_env.close()
    elapsed_sec = time.perf_counter() - t0

    final_group_lrs: dict[str, float] = {}
    if isinstance(model.policy, PaperLikeActorCriticPolicy):
        final_group_lrs = model.policy.current_group_lrs()
    _write_json(
        optimizer_lrs_json,
        {
            "initial_group_lrs": initial_group_lrs,
            "final_group_lrs": final_group_lrs,
            "expected_actor_lr": float(args.actor_lr),
            "expected_critic_lr": float(args.critic_lr),
            "algo": str(args.algo),
            "sigma": float(args.sigma),
            "effective_ent_coef": float(effective_ent_coef),
        },
    )
    print(f"[train] optimizer_lrs_final={final_group_lrs}")

    summary = TrainSummary(
        total_timesteps=int(args.total_timesteps),
        elapsed_seconds=float(elapsed_sec),
        model_path=str(model_path.with_suffix(".zip").resolve()),
        run_dir=str(run_dir.resolve()),
        seed=int(args.seed),
        net3_inp=str(Path(args.net3_inp).resolve()),
        delta_time=float(args.delta_time),
        delta_space=float(args.delta_space),
        scaling_mode=str(args.scaling_mode),
        r_benchmark=float(args.r_benchmark),
        p_hydraulic=float(args.p_hydraulic),
        device=str(args.device),
        algo=str(args.algo),
        sigma=float(args.sigma),
        effective_ent_coef=float(effective_ent_coef),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        gamma=float(args.gamma),
        clip_range=float(args.clip_range),
        n_epochs=int(args.n_epochs),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        optimizer_lrs_json=str(optimizer_lrs_json.resolve()),
    )
    _write_json(summary_json, asdict(summary))

    print("[train] finished")
    print(f"[train] model_path={summary.model_path}")
    print(f"[train] summary_json={summary_json.resolve()}")
    print(f"[train] optimizer_lrs_json={optimizer_lrs_json.resolve()}")
    print(f"[train] elapsed_seconds={summary.elapsed_seconds:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
