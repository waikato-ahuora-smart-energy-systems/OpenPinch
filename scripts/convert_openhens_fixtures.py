"""Convert selected OpenHENS CSV problems into OpenPinch-owned JSON assets.

This is a development helper for the OpenHENS migration task set. It is not a
runtime synthesis API and is intentionally kept outside the ``OpenPinch``
package namespace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)

CASE_IDS = (
    "Four-stream-Escobar-and-Trierweiler-2013-1",
    "Four-stream-Yee-and-Grossmann-1990-1",
    "Five-stream-Bogataj-and-Kravanja-2012-1",
    "Five-stream-Kim-et-al-2017-1",
    "Six-stream-Spray-Dryer-2025-1",
    "Six-stream-Yee-and-Grossmann-1990-1",
    "Nine-stream-Linnhoff-and-Ahmad-1999-1",
    "Ten-stream-Ahmad-1985-1",
    "Ten-stream-Chakraborty-and-Ghoshb-1999-1",
    "Ten-stream-Escobar-and-Grossmann-2010-1",
    "Eleven-stream-Castillo-et-al-1998-1",
    "Pinch-Problem",
    "Thirteen-stream-Kim-et-al-2017-1",
)
PACKAGE_SAMPLE_CASE_ID = "Four-stream-Yee-and-Grossmann-1990-1"
OPENHENS_DT_GRID = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
OPENHENS_DQDA_GRID = [0.5, 0.9, 1.3, 1.7, 2.1, 2.4, 2.8, 3.2, 3.6, 4.0]


class ConversionError(ValueError):
    """Raised when a source CSV row cannot be converted with row/field context."""


@dataclass(frozen=True)
class ProcessStreamRow:
    row_number: int
    number: str
    subsystem: str
    name: str
    designation: str
    supply_temperature: float
    target_temperature: float
    heat_capacity_flowrate: float
    heat_transfer_coefficient: float
    cost: float
    temperature_contribution: float | None = None


@dataclass(frozen=True)
class UtilityRow:
    row_number: int
    number: str
    subsystem: str
    name: str
    designation: str
    supply_temperature: float
    target_temperature: float
    heat_transfer_coefficient: float
    cost: float


@dataclass(frozen=True)
class EconomicsRow:
    row_number: int
    number: str
    subsystem: str
    name: str
    kind: str
    unit_cost: float
    area_coefficient: float
    area_exponent: float


@dataclass(frozen=True)
class ParsedOpenHENSCase:
    case_id: str
    source_csv: Path
    hot_streams: tuple[ProcessStreamRow, ...]
    cold_streams: tuple[ProcessStreamRow, ...]
    hot_utilities: tuple[UtilityRow, ...]
    cold_utilities: tuple[UtilityRow, ...]
    exchanger_economics: tuple[EconomicsRow, ...]

    def economics(self, kind: str) -> EconomicsRow:
        matches = tuple(row for row in self.exchanger_economics if row.kind == kind)
        if not matches:
            raise ConversionError(
                f"exchanger economics section: missing required {kind.title()} row"
            )
        return matches[0]

    def to_legacy_arrays(self, dTmin: float) -> dict[str, list[float] | list[str]]:
        default_temperature_contribution = float(dTmin) / 2.0
        exchange = self.economics("exchange")
        heating = self.economics("heating")
        cooling = self.economics("cooling")

        return {
            "A_coeff": [exchange.area_coefficient],
            "A_exp": [exchange.area_exponent],
            "T_c_cont": [
                (
                    row.temperature_contribution
                    if row.temperature_contribution is not None
                    else default_temperature_contribution
                )
                for row in self.cold_streams
            ],
            "T_c_in": [row.supply_temperature for row in self.cold_streams],
            "T_c_out": [row.target_temperature for row in self.cold_streams],
            "T_cu_in": [row.supply_temperature for row in self.cold_utilities],
            "T_cu_out": [row.target_temperature for row in self.cold_utilities],
            "T_h_cont": [
                (
                    row.temperature_contribution
                    if row.temperature_contribution is not None
                    else default_temperature_contribution
                )
                for row in self.hot_streams
            ],
            "T_h_in": [row.supply_temperature for row in self.hot_streams],
            "T_h_out": [row.target_temperature for row in self.hot_streams],
            "T_hu_in": [row.supply_temperature for row in self.hot_utilities],
            "T_hu_out": [row.target_temperature for row in self.hot_utilities],
            "c_cost": [row.cost for row in self.cold_streams],
            "cold_names": [row.name for row in self.cold_streams],
            "cu_coeff": [cooling.area_coefficient],
            "cu_cost": [row.cost for row in self.cold_utilities],
            "cu_exp": [cooling.area_exponent],
            "cu_unit_cost": [cooling.unit_cost],
            "f_c": [row.heat_capacity_flowrate for row in self.cold_streams],
            "f_h": [row.heat_capacity_flowrate for row in self.hot_streams],
            "h_cost": [row.cost for row in self.hot_streams],
            "hot_names": [row.name for row in self.hot_streams],
            "htc_c": [row.heat_transfer_coefficient for row in self.cold_streams],
            "htc_cu": [row.heat_transfer_coefficient for row in self.cold_utilities],
            "htc_h": [row.heat_transfer_coefficient for row in self.hot_streams],
            "htc_hu": [row.heat_transfer_coefficient for row in self.hot_utilities],
            "hu_coeff": [heating.area_coefficient],
            "hu_cost": [row.cost for row in self.hot_utilities],
            "hu_exp": [heating.area_exponent],
            "hu_unit_cost": [heating.unit_cost],
            "unit_cost": [exchange.unit_cost],
        }


def parse_openhens_csv(
    source_csv: Path, *, case_id: str | None = None
) -> ParsedOpenHENSCase:
    """Parse one OpenHENS workbook-export CSV with row and field context."""
    rows = _read_rows(source_csv)
    process_header = _find_header_row(
        rows,
        ("number", "subsystem", "description", "designation", "supply temp"),
    )
    _find_header_row(
        rows,
        ("number", "subsystem", "description", "designation", "hx unit cost"),
    )
    t_cont_column = _temperature_contribution_column(process_header)
    hot_streams: list[ProcessStreamRow] = []
    cold_streams: list[ProcessStreamRow] = []
    hot_utilities: list[UtilityRow] = []
    cold_utilities: list[UtilityRow] = []
    exchanger_economics: list[EconomicsRow] = []

    for row_number, row in rows:
        designation = _normalise(_cell(row, 3))
        if not designation or designation == "designation":
            continue
        if designation == "hot":
            hot_streams.append(
                _parse_process_stream(row_number, row, "Hot", t_cont_column)
            )
        elif designation == "cold":
            cold_streams.append(
                _parse_process_stream(row_number, row, "Cold", t_cont_column)
            )
        elif designation == "hot utility":
            hot_utilities.append(_parse_utility(row_number, row, "Hot Utility"))
        elif designation == "cold utility":
            cold_utilities.append(_parse_utility(row_number, row, "Cold Utility"))
        elif designation in {"exchange", "heating", "cooling"}:
            exchanger_economics.append(_parse_economics(row_number, row, designation))
        elif designation == "electricity":
            continue
        else:
            raise ConversionError(
                f"row {row_number}: unknown Designation {designation!r}"
            )

    _require_rows("process streams", "Hot", hot_streams)
    _require_rows("process streams", "Cold", cold_streams)
    _require_rows("utilities", "Hot Utility", hot_utilities)
    _require_rows("utilities", "Cold Utility", cold_utilities)
    for kind in ("exchange", "heating", "cooling"):
        if not any(row.kind == kind for row in exchanger_economics):
            raise ConversionError(
                f"exchanger economics section: missing required {kind.title()} row"
            )

    parsed = ParsedOpenHENSCase(
        case_id=case_id or source_csv.stem,
        source_csv=source_csv,
        hot_streams=tuple(hot_streams),
        cold_streams=tuple(cold_streams),
        hot_utilities=tuple(hot_utilities),
        cold_utilities=tuple(cold_utilities),
        exchanger_economics=tuple(exchanger_economics),
    )
    _validate_costing_can_use_openpinch_configuration(parsed)
    return parsed


def convert_case_to_target_input(
    parsed: ParsedOpenHENSCase,
    *,
    output_folder: str | None = None,
    output_formats: list[str] | None = None,
) -> dict[str, Any]:
    """Return standard OpenPinch ``TargetInput`` JSON data."""
    exchange = parsed.economics("exchange")
    streams = [
        _stream_record(row, parsed)
        for row in [*parsed.hot_streams, *parsed.cold_streams]
    ]
    utilities = [
        _utility_record(row) for row in [*parsed.hot_utilities, *parsed.cold_utilities]
    ]
    return {
        "streams": streams,
        "utilities": utilities,
        "options": {
            "COSTING_HX_AREA_EXP": exchange.area_exponent,
            "COSTING_HX_UNIT_COST": exchange.unit_cost,
            "HENS_APPROACH_TEMPERATURES": OPENHENS_DT_GRID,
            "HENS_BEST_SOLUTIONS_TO_SAVE": 10,
            "HENS_DERIVATIVE_THRESHOLDS": OPENHENS_DQDA_GRID,
            "HENS_SOLVER_EVM": "ipopt-pyomo",
            "HENS_SOLVER_OPTIONS_EVM": {},
            "HENS_LOG_LEVEL": "WARNING",
            "HENS_MAX_PARALLEL": 10,
            "HENS_OUTPUT_FOLDER": (
                output_folder
                if output_folder is not None
                else f"tests/fixtures/openhens/solver_baselines/{parsed.case_id}"
            ),
            "HENS_OUTPUT_FORMATS": (
                output_formats if output_formats is not None else ["json", "csv"]
            ),
            "HENS_SOLVER_PDM": "couenne",
            "HENS_SOLVER_OPTIONS_PDM": {},
            "HENS_RUN_ID": parsed.case_id,
            "HENS_SOLVE_TOLERANCE": 1e-3,
            "HENS_STAGE_SELECTION": [1, 2, 3, 4],
            "HENS_SOLVER_TDM": "couenne",
            "HENS_SOLVER_OPTIONS_TDM": {},
            "COSTING_HX_AREA_COEFF": exchange.area_coefficient,
        },
        "zone_tree": {
            "name": "Site",
            "type": "Site",
            "children": [
                {
                    "name": "Process A",
                    "type": "Process Zone",
                    "children": None,
                }
            ],
        },
    }


def write_migration_fixtures(openhens_root: Path, repo_root: Path) -> None:
    """Generate converted JSON fixtures, package samples, and adapter snapshots."""
    for case_id in CASE_IDS:
        parsed = parse_openhens_csv(
            openhens_root / "examples" / "cases" / f"{case_id}.csv",
            case_id=case_id,
        )
        target_input = convert_case_to_target_input(parsed)
        fixture_path = repo_root / "tests" / "fixtures" / "openhens" / f"{case_id}.json"
        _write_json(fixture_path, target_input)
        reordered_path = fixture_path.with_name(f"{case_id}.reordered.json")
        reordered = dict(target_input)
        reordered["streams"] = list(reversed(target_input["streams"]))
        _write_json(reordered_path, reordered)

        if case_id == PACKAGE_SAMPLE_CASE_ID:
            sample_case = convert_case_to_target_input(
                parsed,
                output_folder="",
                output_formats=[],
            )
            _write_json(
                repo_root / "OpenPinch" / "data" / "sample_cases" / f"{case_id}.json",
                sample_case,
            )

    _write_adapter_snapshot(
        parse_openhens_csv(
            openhens_root / "examples" / "cases" / f"{PACKAGE_SAMPLE_CASE_ID}.csv",
            case_id=PACKAGE_SAMPLE_CASE_ID,
        ),
        repo_root,
        openhens_root=openhens_root,
        dTmin=14.0,
    )


def _write_adapter_snapshot(
    parsed: ParsedOpenHENSCase,
    repo_root: Path,
    *,
    openhens_root: Path,
    dTmin: float,
) -> None:
    fixture_path = (
        repo_root / "tests" / "fixtures" / "openhens" / f"{parsed.case_id}.json"
    )
    target_input = json.loads(fixture_path.read_text())
    problem = PinchProblem(source=target_input)
    adapter = problem_to_solver_arrays(problem, dTmin)
    snapshot = adapter.to_json_dict() | {
        "schema_version": "openpinch-openhens-adapter-snapshot/v1",
        "case_id": parsed.case_id,
        "active_dTmin": dTmin,
        "source_csv_path": _display_path(parsed.source_csv, openhens_root),
        "source_csv_sha256": _sha256(parsed.source_csv),
        "source_fixture_path": _display_path(fixture_path, repo_root),
        "source_fixture_sha256": _sha256(fixture_path),
        "source_openhens_arrays": parsed.to_legacy_arrays(dTmin),
        "extraction_command": (
            "uv run python scripts/convert_openhens_fixtures.py "
            "--openhens-root ../OpenHENS --write"
        ),
        "preparation": adapter.preparation
        | {
            "pinch_problem_load": "passed",
            "prepare_problem": "created Zone, StreamCollection, and Stream objects",
        },
    }
    _write_json(
        repo_root
        / "tests"
        / "fixtures"
        / "openhens"
        / "regression_artifacts"
        / "adapter_array_snapshots"
        / parsed.case_id
        / "dTmin-14.json",
        snapshot,
    )


def _stream_record(row: ProcessStreamRow, parsed: ParsedOpenHENSCase) -> dict[str, Any]:
    name = _unique_process_stream_name(row, parsed)
    delta_t = abs(row.supply_temperature - row.target_temperature)
    stream_record = {
        "zone": "Site/Process A",
        "name": name,
        "t_supply": {"value": row.supply_temperature, "unit": "K"},
        "t_target": {"value": row.target_temperature, "unit": "K"},
        "heat_flow": {
            "value": row.heat_capacity_flowrate * delta_t,
            "unit": "kW",
        },
        "heat_capacity_flowrate": {
            "value": row.heat_capacity_flowrate,
            "unit": "kW/delta_degC",
        },
        "htc": {
            "value": row.heat_transfer_coefficient,
            "unit": "kW/m^2/K",
        },
    }
    if row.temperature_contribution is not None:
        stream_record["dt_cont"] = {"value": row.temperature_contribution, "unit": "K"}
    return stream_record


def _utility_record(row: UtilityRow) -> dict[str, Any]:
    utility_type = "Hot" if row.designation == "Hot Utility" else "Cold"
    return {
        "name": row.name.strip(),
        "type": utility_type,
        "t_supply": {"value": row.supply_temperature, "unit": "K"},
        "t_target": {"value": row.target_temperature, "unit": "K"},
        "heat_flow": None,
        "htc": {"value": row.heat_transfer_coefficient, "unit": "kW/m^2/K"},
        "price": {"value": row.cost, "unit": "$/MWh"},
    }


def _unique_process_stream_name(
    row: ProcessStreamRow,
    parsed: ParsedOpenHENSCase,
) -> str:
    row_name = row.name.strip()
    matching_names = [
        item.name.strip()
        for item in [*parsed.hot_streams, *parsed.cold_streams]
        if item.name.strip() == row_name
    ]
    if len(matching_names) == 1:
        return row_name
    prefix = "Hot" if row.designation == "Hot" else "Cold"
    return f"{prefix} {row.number} {row_name}".strip()


def _validate_costing_can_use_openpinch_configuration(
    parsed: ParsedOpenHENSCase,
) -> None:
    rows = [parsed.economics(kind) for kind in ("exchange", "heating", "cooling")]
    first = rows[0]
    if any(
        row.unit_cost != first.unit_cost
        or row.area_coefficient != first.area_coefficient
        or row.area_exponent != first.area_exponent
        for row in rows
    ):
        row_numbers = ", ".join(str(row.row_number) for row in rows)
        raise ConversionError(
            "exchanger economics rows "
            f"{row_numbers}: per-kind coefficients require a general "
            "OpenPinch costing schema extension before conversion"
        )


def _parse_process_stream(
    row_number: int,
    row: list[str],
    designation: str,
    t_cont_column: int | None,
) -> ProcessStreamRow:
    parsed = ProcessStreamRow(
        row_number=row_number,
        number=_cell(row, 0),
        subsystem=_cell(row, 1),
        name=_raw_cell(row, 2),
        designation=designation,
        supply_temperature=_numeric(
            row_number, row, 4, "Supply Temp", "process streams"
        ),
        target_temperature=_numeric(
            row_number, row, 5, "Target Temp", "process streams"
        ),
        heat_capacity_flowrate=_numeric(
            row_number,
            row,
            6,
            "Flow heat capacity",
            "process streams",
        ),
        heat_transfer_coefficient=_numeric(
            row_number, row, 7, "HTC", "process streams"
        ),
        cost=_numeric(row_number, row, 8, "Stream cost", "process streams"),
        temperature_contribution=_optional_numeric(
            row_number,
            row,
            t_cont_column,
            "T cont",
            "process streams",
        ),
    )
    if designation == "Hot" and parsed.supply_temperature <= parsed.target_temperature:
        raise ConversionError(
            f"process streams row {row_number}: hot stream must cool down "
            "(Supply Temp must be greater than Target Temp)"
        )
    if designation == "Cold" and parsed.target_temperature <= parsed.supply_temperature:
        raise ConversionError(
            f"process streams row {row_number}: cold stream must heat up "
            "(Target Temp must be greater than Supply Temp)"
        )
    if parsed.heat_capacity_flowrate <= 0:
        raise ConversionError(
            f"process streams row {row_number}: Flow heat capacity must be positive"
        )
    if parsed.heat_transfer_coefficient <= 0:
        raise ConversionError(f"process streams row {row_number}: HTC must be positive")
    if parsed.cost < 0:
        raise ConversionError(
            f"process streams row {row_number}: Stream cost must be non-negative"
        )
    if (
        parsed.temperature_contribution is not None
        and parsed.temperature_contribution < 0
    ):
        raise ConversionError(
            f"process streams row {row_number}: T cont must be non-negative"
        )
    return parsed


def _parse_utility(row_number: int, row: list[str], designation: str) -> UtilityRow:
    parsed = UtilityRow(
        row_number=row_number,
        number=_cell(row, 0),
        subsystem=_cell(row, 1),
        name=_raw_cell(row, 2),
        designation=designation,
        supply_temperature=_numeric(row_number, row, 4, "Supply Temp", "utilities"),
        target_temperature=_numeric(row_number, row, 5, "Target Temp", "utilities"),
        heat_transfer_coefficient=_numeric(row_number, row, 7, "HTC", "utilities"),
        cost=_numeric(row_number, row, 8, "Stream cost", "utilities"),
    )
    if (
        designation == "Hot Utility"
        and parsed.supply_temperature < parsed.target_temperature
    ):
        raise ConversionError(
            f"utilities row {row_number}: hot utility Supply Temp must be greater "
            "than or equal to Target Temp"
        )
    if (
        designation == "Cold Utility"
        and parsed.target_temperature < parsed.supply_temperature
    ):
        raise ConversionError(
            f"utilities row {row_number}: cold utility Target Temp must be greater "
            "than or equal to Supply Temp"
        )
    if parsed.heat_transfer_coefficient <= 0:
        raise ConversionError(f"utilities row {row_number}: HTC must be positive")
    if parsed.cost < 0:
        raise ConversionError(
            f"utilities row {row_number}: Stream cost must be non-negative"
        )
    return parsed


def _parse_economics(row_number: int, row: list[str], kind: str) -> EconomicsRow:
    parsed = EconomicsRow(
        row_number=row_number,
        number=_cell(row, 0),
        subsystem=_cell(row, 1),
        name=_raw_cell(row, 2),
        kind=kind,
        unit_cost=_numeric(row_number, row, 4, "HX unit cost", "exchanger economics"),
        area_coefficient=_numeric(
            row_number,
            row,
            5,
            "HX area coefficient",
            "exchanger economics",
        ),
        area_exponent=_numeric(
            row_number,
            row,
            6,
            "HX area exponent",
            "exchanger economics",
        ),
    )
    if parsed.unit_cost < 0:
        raise ConversionError(
            f"exchanger economics row {row_number}: HX unit cost must be non-negative"
        )
    if parsed.area_coefficient < 0:
        raise ConversionError(
            f"exchanger economics row {row_number}: "
            "HX area coefficient must be non-negative"
        )
    if parsed.area_exponent < 0:
        raise ConversionError(
            f"exchanger economics row {row_number}: "
            "HX area exponent must be non-negative"
        )
    return parsed


def _read_rows(path: Path) -> list[tuple[int, list[str]]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=";,")
    except csv.Error:
        dialect = csv.excel()
        dialect.delimiter = ";" if text.count(";") >= text.count(",") else ","
    reader = csv.reader(text.splitlines(), dialect)
    return [(index, row) for index, row in enumerate(reader, start=1)]


def _find_header_row(
    rows: list[tuple[int, list[str]]],
    required_prefix: tuple[str, ...],
) -> list[str]:
    required = tuple(_normalise_header(value) for value in required_prefix)
    for _, row in rows:
        normalised = tuple(_normalise_header(cell) for cell in row[: len(required)])
        if normalised == required:
            return row
    raise ConversionError(
        "CSV does not match the expected workbook-export schema; "
        f"missing header {required_prefix!r}"
    )


def _temperature_contribution_column(process_header: list[str]) -> int | None:
    for index, label in enumerate(process_header):
        normalised = _normalise_header(label)
        if normalised in {"t cont", "t contribution", "temperature contribution"}:
            return index
    return None


def _optional_numeric(
    row_number: int,
    row: list[str],
    index: int | None,
    field_name: str,
    section: str,
) -> float | None:
    if index is None or _cell(row, index) == "":
        return None
    return _numeric(row_number, row, index, field_name, section)


def _numeric(
    row_number: int,
    row: list[str],
    index: int,
    field_name: str,
    section: str,
) -> float:
    raw_value = _cell(row, index)
    if raw_value == "":
        raise ConversionError(f"{section} row {row_number}: {field_name} is required")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConversionError(
            f"{section} row {row_number}: {field_name} must be numeric; "
            f"got {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ConversionError(
            f"{section} row {row_number}: {field_name} must be finite"
        )
    return value


def _require_rows(section: str, designation: str, rows: list[Any]) -> None:
    if not rows:
        raise ConversionError(f"{section} section: missing required {designation} row")


def _normalise(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalise_header(value: str) -> str:
    return _normalise(value).replace("_", " ")


def _cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return row[index].strip()


def _raw_cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return row[index]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, json_data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openhens-root",
        type=Path,
        default=Path("../OpenHENS"),
        help="Path to the source OpenHENS checkout.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the OpenPinch checkout.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write converted fixtures and snapshot artifacts.",
    )
    args = parser.parse_args()
    if not args.write:
        parser.error("pass --write to update migration fixture artifacts")
    write_migration_fixtures(args.openhens_root.resolve(), args.repo_root.resolve())


if __name__ == "__main__":
    main()
