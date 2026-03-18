"""Paper-compliance oriented checks.

This file intentionally separates:
- requirements already satisfied by current local implementation, and
- requirements that are intentionally pending for full paper reproduction.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("gymnasium")

from epanet_rl.env import Net3PumpSchedulingEnv  # noqa: E402


def _net3_path() -> Path:
    return ROOT / "networks" / "Net3.inp"


def _count_section_entries(inp_text: str, section_name: str) -> int:
    lines = inp_text.splitlines()
    in_section = False
    count = 0
    target = f"[{section_name.upper()}]"
    for line in lines:
        stripped = line.strip()
        if stripped.upper() == target:
            in_section = True
            continue
        if in_section and stripped.startswith("[") and stripped.endswith("]"):
            break
        if in_section and stripped and not stripped.startswith(";"):
            count += 1
    return count


def test_paper_requirement_24_steps_with_hourly_discretization() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    assert env.EPISODE_STEPS == 24
    assert env.STEP_HOURS == 1.0


def test_paper_requirement_net3_counts_in_dataset_file() -> None:
    text = _net3_path().read_text(encoding="utf-8")
    assert _count_section_entries(text, "JUNCTIONS") == 92
    assert _count_section_entries(text, "TANKS") == 3
    assert _count_section_entries(text, "RESERVOIRS") == 2
    assert _count_section_entries(text, "PUMPS") == 2


def test_paper_requirement_state_structure_is_demand_plus_tank_level_only() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    obs, _ = env.reset(seed=11)
    assert obs.shape[0] == env._base_demands.size + len(env._tank_ids)  # noqa: SLF001


def test_paper_requirement_r_benchmark_is_currently_placeholder_not_final_statistic() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    assert env.r_benchmark == 2000.0, (
        "Current project uses hard-coded placeholder r_benchmark for local development. "
        "This is not the paper's final statistical benchmark value."
    )


@pytest.mark.xfail(
    reason=(
        "Current env.py is a minimal runnable skeleton with simplified tank dynamics. "
        "It is not yet a true WNTR/EPANET single-step hydraulic simulation environment."
    ),
    strict=False,
)
def test_pending_full_hydraulic_step_simulation_backend() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    assert hasattr(env, "wntr_step_simulator")


@pytest.mark.xfail(
    reason=(
        "ESFC-Z-Score in paper uses statistics from large rollout batches (e.g., 10,000 episodes). "
        "Current local implementation provides the interface but not this full statistical calibration."
    ),
    strict=False,
)
def test_pending_esfc_zscore_large_rollout_statistics() -> None:
    env = Net3PumpSchedulingEnv(net3_inp_path=_net3_path())
    assert hasattr(env, "_zscore_stats_from_10000_episodes")
