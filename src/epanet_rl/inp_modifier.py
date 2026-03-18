"""Utilities for modifying EPANET INP files for the Net3 RL setup."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

PEAK_PRICE_USD_PER_KWH = 0.1194
OFFPEAK_PRICE_USD_PER_KWH = 0.0244
TARGET_PUMP_EFFICIENCY = 0.75
TOU_PATTERN_ID = "EPANET_RL_TOU"

_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_TARGET_ENTITY_RE = re.compile(r"\b(?:LINK|PUMP|PIPE)\s+(?:10|330|335)\b", re.IGNORECASE)
_CONTROL_COMMENT_RE = re.compile(
    r"lake source|pump\s*335|pump\s*10|pipe\s*330|link\s*10|link\s*330|link\s*335|bypass pipe",
    re.IGNORECASE,
)
_GLOBAL_ENERGY_RE = re.compile(r"^\s*Global\s+(?:Efficiency|Price|Pattern)\b", re.IGNORECASE)
_DEMAND_CHARGE_RE = re.compile(r"^\s*Demand\s+Charge\b", re.IGNORECASE)


def _find_section_bounds(lines: list[str], section_name: str) -> tuple[int, int]:
    target = section_name.strip().upper()
    for idx, line in enumerate(lines):
        match = _SECTION_RE.match(line)
        if not match or match.group(1).strip().upper() != target:
            continue

        end = len(lines)
        for next_idx in range(idx + 1, len(lines)):
            if _SECTION_RE.match(lines[next_idx]):
                end = next_idx
                break
        return idx, end

    raise ValueError(f"Missing [{section_name}] section in INP file.")


def _try_find_section_bounds(lines: list[str], section_name: str) -> tuple[int, int] | None:
    try:
        return _find_section_bounds(lines, section_name)
    except ValueError:
        return None


def _replace_section_body(
    lines: list[str],
    section_start: int,
    section_end: int,
    new_body: list[str],
) -> list[str]:
    return lines[: section_start + 1] + new_body + lines[section_end:]


def _squeeze_blank_lines(lines: Iterable[str]) -> list[str]:
    result: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line.strip() == ""
        if is_blank and previous_blank:
            continue
        result.append(line)
        previous_blank = is_blank
    return result


def _is_related_control_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(";"):
        return bool(_CONTROL_COMMENT_RE.search(stripped))
    return bool(_TARGET_ENTITY_RE.search(stripped))


def _remove_related_controls(lines: list[str]) -> list[str]:
    for section_name in ("CONTROLS", "RULES"):
        bounds = _try_find_section_bounds(lines, section_name)
        if bounds is None:
            continue
        section_start, section_end = bounds
        body = lines[section_start + 1 : section_end]
        filtered_body = [line for line in body if not _is_related_control_line(line)]
        lines = _replace_section_body(lines, section_start, section_end, _squeeze_blank_lines(filtered_body))
    return lines


def _set_pipe_330_closed(lines: list[str]) -> list[str]:
    section_start, section_end = _find_section_bounds(lines, "STATUS")
    body = lines[section_start + 1 : section_end]

    updated: list[str] = []
    has_pipe_330 = False
    for line in body:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            updated.append(line)
            continue

        link_id = stripped.split()[0]
        if link_id == "330":
            if not has_pipe_330:
                updated.append(" 330\tClosed")
                has_pipe_330 = True
            continue

        updated.append(line)

    if not has_pipe_330:
        insert_at = len(updated)
        while insert_at > 0 and updated[insert_at - 1].strip() == "":
            insert_at -= 1
        updated.insert(insert_at, " 330\tClosed")

    return _replace_section_body(lines, section_start, section_end, updated)


def _format_number(value: float, decimals: int = 10) -> str:
    rounded_int = round(value)
    if abs(value - rounded_int) < 1e-12:
        return str(int(rounded_int))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _set_energy_section(lines: list[str]) -> list[str]:
    section_start, section_end = _find_section_bounds(lines, "ENERGY")
    body = lines[section_start + 1 : section_end]

    demand_charge_value = "0.0"
    preserved_lines: list[str] = []
    for line in body:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            preserved_lines.append(line)
            continue

        if _GLOBAL_ENERGY_RE.match(line):
            continue

        if _DEMAND_CHARGE_RE.match(line):
            parts = stripped.split()
            if parts:
                demand_charge_value = parts[-1]
            continue

        preserved_lines.append(line)

    efficiency_percent = TARGET_PUMP_EFFICIENCY * 100.0
    energy_settings = [
        f" Global Efficiency\t{_format_number(efficiency_percent)}",
        f" Global Price\t{_format_number(OFFPEAK_PRICE_USD_PER_KWH, decimals=4)}",
        f" Global Pattern\t{TOU_PATTERN_ID}",
        f" Demand Charge\t{demand_charge_value}",
    ]

    leading_comments: list[str] = []
    trailing = preserved_lines[:]
    while trailing and (trailing[0].strip() == "" or trailing[0].lstrip().startswith(";")):
        leading_comments.append(trailing.pop(0))

    new_body = leading_comments + energy_settings
    if trailing:
        if new_body and new_body[-1].strip() != "":
            new_body.append("")
        new_body.extend(trailing)

    return _replace_section_body(lines, section_start, section_end, new_body)


def _build_tou_pattern_values() -> list[str]:
    peak_ratio = PEAK_PRICE_USD_PER_KWH / OFFPEAK_PRICE_USD_PER_KWH
    values = []
    for hour in range(24):
        if 7 <= hour < 23:
            values.append(_format_number(peak_ratio, decimals=9))
        else:
            values.append("1")
    return values


def _upsert_tou_pattern(lines: list[str]) -> list[str]:
    section_start, section_end = _find_section_bounds(lines, "PATTERNS")
    body = lines[section_start + 1 : section_end]

    filtered: list[str] = []
    for line in body:
        stripped = line.strip()
        if stripped.startswith(";") and TOU_PATTERN_ID.lower() in stripped.lower():
            continue
        if stripped and not stripped.startswith(";"):
            first_token = stripped.split()[0]
            if first_token.upper() == TOU_PATTERN_ID.upper():
                continue
        filtered.append(line)

    while filtered and filtered[-1].strip() == "":
        filtered.pop()

    pattern_values = _build_tou_pattern_values()
    tou_block = [
        ";TOU electricity tariff multipliers (07:00-23:00 peak, others off-peak)",
        f" {TOU_PATTERN_ID}\t" + "\t".join(pattern_values[:12]),
        f" {TOU_PATTERN_ID}\t" + "\t".join(pattern_values[12:]),
    ]

    if filtered and filtered[-1].strip() != "":
        filtered.append("")
    filtered.extend(tou_block)
    filtered.append("")

    return _replace_section_body(lines, section_start, section_end, filtered)


def modify_inp_text(inp_text: str) -> str:
    """Return modified INP text matching the Net3 RL preprocessing targets."""

    newline = "\r\n" if "\r\n" in inp_text else "\n"
    had_trailing_newline = inp_text.endswith(("\r\n", "\n"))
    lines = inp_text.splitlines()

    lines = _remove_related_controls(lines)
    lines = _set_pipe_330_closed(lines)
    lines = _set_energy_section(lines)
    lines = _upsert_tou_pattern(lines)

    output = newline.join(lines)
    if had_trailing_newline:
        output += newline
    return output


def modify_inp_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    in_place: bool = False,
) -> Path:
    """Modify an INP file and return the output path."""

    input_file = Path(input_path)
    if in_place and output_path is not None:
        raise ValueError("Use either output_path or in_place=True, not both.")

    if in_place:
        target_file = input_file
    elif output_path is not None:
        target_file = Path(output_path)
    else:
        target_file = input_file.with_name(f"{input_file.stem}_modified{input_file.suffix}")

    inp_text = input_file.read_text(encoding="utf-8")
    modified_text = modify_inp_text(inp_text)
    target_file.write_text(modified_text, encoding="utf-8")
    return target_file


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modify EPANET INP file for Net3 RL setup.")
    parser.add_argument("--input", type=Path, required=True, help="Input INP file path.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path, help="Output INP file path.")
    group.add_argument("--in-place", action="store_true", help="Overwrite input file in place.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output = modify_inp_file(args.input, args.output, in_place=args.in_place)
    print(f"Modified INP written to: {output}")


if __name__ == "__main__":
    main()
