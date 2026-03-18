import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epanet_rl.inp_modifier import (  # noqa: E402
    OFFPEAK_PRICE_USD_PER_KWH,
    PEAK_PRICE_USD_PER_KWH,
    TOU_PATTERN_ID,
    modify_inp_text,
)


def _section(text: str, name: str) -> str:
    lines = text.splitlines()
    start = None
    section_header = f"[{name.upper()}]"
    for i, line in enumerate(lines):
        if line.strip().upper() == section_header:
            start = i + 1
            break
    if start is None:
        raise AssertionError(f"Section {section_header} not found.")

    end = len(lines)
    for i in range(start, len(lines)):
        if re.match(r"^\s*\[[^\]]+\]\s*$", lines[i]):
            end = i
            break
    return "\n".join(lines[start:end])


def _collect_pattern_values(pattern_section: str, pattern_id: str) -> list[float]:
    values: list[float] = []
    for line in pattern_section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        parts = stripped.split()
        if parts[0].upper() != pattern_id.upper():
            continue
        values.extend(float(x) for x in parts[1:])
    return values


def test_net3_modifier_removes_target_controls_and_closes_pipe_330() -> None:
    net3_text = (ROOT / "networks" / "Net3.inp").read_text(encoding="utf-8")
    modified = modify_inp_text(net3_text)

    controls = _section(modified, "CONTROLS").upper()
    assert "LINK 10 " not in controls
    assert "LINK 335 " not in controls
    assert "LINK 330 " not in controls
    assert "LAKE SOURCE" not in controls

    status = _section(modified, "STATUS")
    matches = re.findall(r"^\s*330\s+Closed\s*$", status, flags=re.MULTILINE)
    assert len(matches) == 1


def test_net3_modifier_sets_energy_and_tou_pricing() -> None:
    net3_text = (ROOT / "networks" / "Net3.inp").read_text(encoding="utf-8")
    modified = modify_inp_text(net3_text)

    energy = _section(modified, "ENERGY")
    assert re.search(r"^\s*Global\s+Efficiency\s+75(?:\.0+)?\s*$", energy, flags=re.IGNORECASE | re.MULTILINE)
    assert re.search(
        rf"^\s*Global\s+Price\s+{OFFPEAK_PRICE_USD_PER_KWH}\s*$",
        energy,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    assert re.search(
        rf"^\s*Global\s+Pattern\s+{TOU_PATTERN_ID}\s*$",
        energy,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    patterns = _section(modified, "PATTERNS")
    tou_values = _collect_pattern_values(patterns, TOU_PATTERN_ID)
    assert len(tou_values) == 24

    peak_ratio = PEAK_PRICE_USD_PER_KWH / OFFPEAK_PRICE_USD_PER_KWH
    for hour, value in enumerate(tou_values):
        expected = peak_ratio if 7 <= hour < 23 else 1.0
        assert value == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_net3_modifier_keeps_default_demand_pattern_reference() -> None:
    net3_text = (ROOT / "networks" / "Net3.inp").read_text(encoding="utf-8")
    modified = modify_inp_text(net3_text)

    options = _section(modified, "OPTIONS")
    assert re.search(r"^\s*Pattern\s+1\s*$", options, flags=re.IGNORECASE | re.MULTILINE)

    patterns = _section(modified, "PATTERNS")
    pattern_1_values = _collect_pattern_values(patterns, "1")
    assert len(pattern_1_values) >= 24
