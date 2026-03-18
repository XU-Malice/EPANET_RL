import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epanet_rl.action_space import (  # noqa: E402
    ACTION_COUNT,
    ACTION_LEVELS,
    NUM_PUMPS,
    action_id_to_speeds,
    all_action_combinations,
    speeds_to_action_id,
)


def test_action_levels_and_count_match_paper_definition() -> None:
    assert NUM_PUMPS == 2
    assert ACTION_LEVELS == (0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
    assert ACTION_COUNT == 64


def test_action_id_and_speed_roundtrip_for_all_actions() -> None:
    for action_id in range(ACTION_COUNT):
        speeds = action_id_to_speeds(action_id)
        reconstructed = speeds_to_action_id(*speeds)
        assert reconstructed == action_id


def test_all_action_combinations_order_and_size() -> None:
    combos = all_action_combinations()
    assert len(combos) == 64
    assert combos[0] == (0.0, 0.0)
    assert combos[-1] == (1.0, 1.0)


@pytest.mark.parametrize("invalid_action", [-1, 64, 999])
def test_action_id_to_speeds_rejects_out_of_range_id(invalid_action: int) -> None:
    with pytest.raises(ValueError):
        action_id_to_speeds(invalid_action)


def test_speeds_to_action_id_rejects_invalid_speed() -> None:
    with pytest.raises(ValueError):
        speeds_to_action_id(0.65, 1.0)
