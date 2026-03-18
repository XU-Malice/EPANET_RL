"""Diagnostic rollout script for Net3PumpSchedulingEnv.

This script is for diagnostics only (not training).
It runs at most 24 environment steps and prints per-step details, with
special focus on large reward magnitude at the final step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `src/` importable when running from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epanet_rl.action_space import action_id_to_speeds
from epanet_rl.env import Net3PumpSchedulingEnv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Roll out Net3 env for diagnostics.")
    parser.add_argument(
        "--net3-inp",
        type=Path,
        default=PROJECT_ROOT / "networks" / "Net3.inp",
        help="Path to Net3 INP file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reset and action sampling.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Max rollout steps (clamped to [1, 24]).",
    )
    parser.add_argument(
        "--fixed-action",
        type=int,
        default=None,
        help="If set, always use this action id (0..63). Otherwise sample random actions.",
    )
    return parser


def _compute_final_tank_volume(env: Net3PumpSchedulingEnv) -> float:
    # Diagnostic script intentionally reads env internals for reward debugging.
    return float(env._compute_total_tank_volume(env._tank_levels))  # noqa: SLF001


def _format_array(values: np.ndarray, precision: int = 4) -> str:
    return np.array2string(values, precision=precision, separator=", ", floatmode="fixed")


def _print_last_step_diagnosis(
    *,
    env: Net3PumpSchedulingEnv,
    t: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    hydraulic_violation: bool,
    initial_tank_volume: float,
    final_tank_volume: float,
    base_reward: float,
    tank_penalty: float,
) -> None:
    print("\n[Final-Step Diagnosis]")
    print(f"t={t}, reward={reward:.6f}, base_reward={base_reward:.6f}, tank_penalty={tank_penalty:.6f}")
    print(f"initial_tank_volume={initial_tank_volume:.6f}, final_tank_volume={final_tank_volume:.6f}")

    if terminated and hydraulic_violation:
        print(
            "Reason: terminated by hydraulic violation, so reward is replaced by P_hydraulic "
            f"(configured as {env.p_hydraulic:.6f})."
        )
        return

    if t == env.EPISODE_STEPS - 1:
        if final_tank_volume < initial_tank_volume:
            shortfall = initial_tank_volume - final_tank_volume
            cfg = env.tank_penalty_config
            if cfg.mode == "proportional":
                shortfall_ratio = shortfall / initial_tank_volume if initial_tank_volume > 0.0 else 0.0
                expected_penalty = cfg.proportional_coefficient * shortfall_ratio * env.r_benchmark
                print(
                    "Reason: final-step tank shortfall penalty applied in proportional mode. "
                    f"shortfall={shortfall:.6f}, shortfall_ratio={shortfall_ratio:.6f}, "
                    f"coefficient={cfg.proportional_coefficient:.6f}, r_benchmark={env.r_benchmark:.6f}, "
                    f"expected_penalty={expected_penalty:.6f}. "
                    "With the usual negative coefficient, this yields a negative penalty."
                )
            else:
                expected_penalty = cfg.constant_value
                print(
                    "Reason: final-step tank shortfall penalty applied in constant mode. "
                    f"constant_penalty={expected_penalty:.6f}."
                )
            print(
                "Reward decomposition check: "
                f"base_reward + tank_penalty = {base_reward + tank_penalty:.6f}."
            )
            if abs(tank_penalty) > abs(base_reward):
                print("Observation: tank_penalty magnitude dominates base_reward at final step.")
        else:
            print(
                "Reason: no final-step tank penalty (final tank volume is not below initial), "
                "reward should be regular base_reward."
            )
    elif truncated:
        print("Episode reached horizon before diagnostic final-step branch.")
    else:
        print("Episode ended before final step; final-step penalty rule was not triggered.")


def main() -> None:
    args = _build_parser().parse_args()
    max_steps = min(24, max(1, args.max_steps))

    env = Net3PumpSchedulingEnv(net3_inp_path=args.net3_inp)
    obs, info = env.reset(seed=args.seed)
    _ = obs, info

    print("[Rollout Start]")
    print(f"net3_inp={args.net3_inp}")
    print(f"seed={args.seed}, max_steps={max_steps}, action_mode={'fixed' if args.fixed_action is not None else 'random'}")

    for t in range(max_steps):
        if args.fixed_action is not None:
            action = int(args.fixed_action)
            if not env.action_space.contains(action):
                raise ValueError(f"--fixed-action must be in [0, {env.action_space.n - 1}], got {action}.")
        else:
            action = int(env.action_space.sample())

        decoded_speeds = action_id_to_speeds(action)
        _, reward, terminated, truncated, step_info = env.step(action)

        tank_levels = np.asarray(step_info.get("tank_levels", np.array([])), dtype=np.float64)
        initial_tank_volume = float(env._initial_tank_volume)  # noqa: SLF001
        final_tank_volume = _compute_final_tank_volume(env)
        base_reward = float(step_info.get("base_reward", float("nan")))
        tank_penalty = float(step_info.get("tank_penalty", float("nan")))
        hydraulic_violation = bool(step_info.get("hydraulic_violation", False))

        print(
            f"t={t:02d} action={action:02d} decoded_pump_speeds={decoded_speeds} "
            f"reward={reward:.6f} terminated={terminated} truncated={truncated}"
        )
        print(f"  tank_levels={_format_array(tank_levels)}")
        print(
            f"  initial_tank_volume={initial_tank_volume:.6f} "
            f"final_tank_volume={final_tank_volume:.6f}"
        )
        print(f"  base_reward={base_reward:.6f} tank_penalty={tank_penalty:.6f}")

        if t == 23 or terminated or truncated:
            _print_last_step_diagnosis(
                env=env,
                t=t,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                hydraulic_violation=hydraulic_violation,
                initial_tank_volume=initial_tank_volume,
                final_tank_volume=final_tank_volume,
                base_reward=base_reward,
                tank_penalty=tank_penalty,
            )

        if terminated or truncated:
            break

    env.close()
    print("[Rollout End]")


if __name__ == "__main__":
    main()
