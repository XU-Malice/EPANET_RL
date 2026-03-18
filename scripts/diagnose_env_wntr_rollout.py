"""Diagnostic rollout script for Net3WntrEnv.

This script is for diagnostics only (not training).
It focuses on early termination analysis in WNTR/EPANET step-by-step rollout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make src importable when running from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epanet_rl.action_space import action_id_to_speeds
from epanet_rl.env_wntr import Net3WntrEnv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Net3WntrEnv rollout and early termination.")
    parser.add_argument(
        "--net3-inp",
        type=Path,
        default=PROJECT_ROOT / "networks" / "Net3.inp",
        help="Path to Net3 INP file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reset seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Maximum rollout steps (clamped to [1, 24]).",
    )
    parser.add_argument(
        "--fixed-action",
        type=int,
        default=None,
        help="If set, always use this action id (0..63); otherwise random action each step.",
    )
    parser.add_argument(
        "--scaling-mode",
        type=str,
        default="max_min",
        choices=["none", "max_min", "z_score"],
        help="Observation scaling mode for env construction.",
    )
    return parser


def _format_array(values: np.ndarray, precision: int = 6) -> str:
    return np.array2string(values, precision=precision, separator=", ", floatmode="fixed")


def _diagnose_nan_pressure_cause(
    *,
    terminated: bool,
    truncated: bool,
    hydraulic_violation: bool,
    min_pressure: float,
    pump_flows: np.ndarray,
    pump_head_gains: np.ndarray,
) -> str | None:
    if not np.isnan(min_pressure):
        return None

    flows_all_nan = np.isnan(pump_flows).all()
    heads_all_nan = np.isnan(pump_head_gains).all()
    result_missing_or_failed = bool(flows_all_nan or heads_all_nan)

    if terminated and hydraulic_violation:
        if result_missing_or_failed:
            return (
                "min_pressure=NaN 且泵结果为 NaN：更可能是仿真结果缺失/求解失败，"
                "环境将其作为 hydraulic violation 提前终止。"
            )
        return (
            "min_pressure=NaN 但泵结果存在：更可能是仿真输出中的压力数据异常，"
            "环境按 hydraulic violation 提前终止。"
        )

    if truncated:
        return "min_pressure=NaN，episode 被截断结束；需要进一步检查仿真输出完整性。"

    if hydraulic_violation:
        return "min_pressure=NaN，已标记 hydraulic violation；若未终止请检查环境终止逻辑。"

    return "min_pressure=NaN，但当前未标记 hydraulic violation；需要检查判定条件。"


def main() -> None:
    args = _build_parser().parse_args()
    max_steps = max(1, min(24, args.max_steps))

    env = Net3WntrEnv(net3_inp_path=args.net3_inp, scaling_mode=args.scaling_mode)
    _, reset_info = env.reset(seed=args.seed)

    print("[WNTR Rollout Start]")
    print(f"net3_inp={args.net3_inp}")
    print(
        f"seed={args.seed}, max_steps={max_steps}, "
        f"action_mode={'fixed' if args.fixed_action is not None else 'random'}"
    )
    print(f"pump_names={reset_info.get('pump_names')}, tank_names={reset_info.get('tank_names')}")

    for t in range(max_steps):
        if args.fixed_action is not None:
            action = int(args.fixed_action)
            if not env.action_space.contains(action):
                raise ValueError(f"--fixed-action must be in [0, {env.action_space.n - 1}], got {action}.")
        else:
            action = int(env.action_space.sample())

        decoded_speeds = action_id_to_speeds(action)
        _, reward, terminated, truncated, info = env.step(action)

        pump_flows = np.asarray(info.get("pump_flows", np.array([])), dtype=np.float64)
        pump_head_gains = np.asarray(info.get("pump_head_gains", np.array([])), dtype=np.float64)
        min_pressure = float(info.get("min_pressure", float("nan")))
        tank_levels = np.asarray(info.get("tank_levels", np.array([])), dtype=np.float64)
        initial_tank_volume = float(info.get("initial_tank_volume", float("nan")))
        final_tank_volume = float(info.get("final_tank_volume", float("nan")))
        base_reward = float(info.get("base_reward", float("nan")))
        tank_penalty = float(info.get("tank_penalty", float("nan")))
        hydraulic_violation = bool(info.get("hydraulic_violation", False))

        print(
            f"t={t:02d} action={action:02d} decoded_pump_speeds={decoded_speeds} "
            f"reward={float(reward):.6f} terminated={bool(terminated)} truncated={bool(truncated)}"
        )
        print(f"  pump_flows={_format_array(pump_flows)}")
        print(f"  pump_head_gains={_format_array(pump_head_gains)}")
        print(f"  min_pressure={min_pressure}")
        print(f"  tank_levels={_format_array(tank_levels)}")
        print(
            f"  initial_tank_volume={initial_tank_volume:.6f} "
            f"final_tank_volume={final_tank_volume:.6f}"
        )
        print(f"  base_reward={base_reward:.6f} tank_penalty={tank_penalty:.6f}")

        nan_reason = _diagnose_nan_pressure_cause(
            terminated=bool(terminated),
            truncated=bool(truncated),
            hydraulic_violation=hydraulic_violation,
            min_pressure=min_pressure,
            pump_flows=pump_flows,
            pump_head_gains=pump_head_gains,
        )
        if nan_reason is not None:
            print(f"  [NaN Diagnostic] {nan_reason}")

        if terminated or truncated:
            print("  [Episode End] Rollout stopped early due to termination/truncation.")
            break

    env.close()
    print("[WNTR Rollout End]")


if __name__ == "__main__":
    main()
