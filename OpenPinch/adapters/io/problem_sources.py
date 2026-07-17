"""Filesystem adapters for supported problem source formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...resources import list_sample_cases, read_sample_case
from ...utils.csv_to_json import get_problem_from_csv
from ...utils.wkbook_to_json import get_problem_from_excel

JsonDict = dict[str, Any]
PathLike = str | Path


@dataclass(frozen=True)
class ExternalProblemSource:
    """Raw external inputs plus source metadata for application normalization."""

    input_data: JsonDict
    source_kind: str
    problem_filepath: Path | None
    project_name: str | None = None


def load_external_problem_source(
    source: PathLike | tuple[PathLike, PathLike],
    *,
    current_project_name: str | None,
) -> ExternalProblemSource:
    """Read a JSON, workbook, CSV pair, or CSV directory problem source."""
    if isinstance(source, tuple) and len(source) == 2:
        streams_csv, utilities_csv = map(Path, source)
        return ExternalProblemSource(
            input_data=get_problem_from_csv(
                streams_csv,
                utilities_csv,
                output_json=None,
            ),
            source_kind="csv",
            problem_filepath=None,
        )

    source_path = Path(source)
    project_name = (
        source_path.stem if current_project_name == "Untitled" else current_project_name
    )
    suffix = source_path.suffix.lower()
    if suffix == ".json":
        return ExternalProblemSource(
            input_data=_load_json_inputs(source_path, source=source),
            source_kind="json",
            problem_filepath=source_path,
            project_name=project_name,
        )
    if suffix in {".xlsx", ".xls", ".xlsb", ".xlsm"}:
        return ExternalProblemSource(
            input_data=get_problem_from_excel(source_path, output_json=None),
            source_kind="excel",
            problem_filepath=source_path,
            project_name=project_name,
        )
    if source_path.is_dir():
        streams_csv = source_path / "streams.csv"
        utilities_csv = source_path / "utilities.csv"
        if not streams_csv.exists() or not utilities_csv.exists():
            raise FileNotFoundError(
                f"CSV directory '{source_path}' must contain "
                "'streams.csv' and 'utilities.csv'."
            )
        return ExternalProblemSource(
            input_data=get_problem_from_csv(
                streams_csv,
                utilities_csv,
                output_json=None,
            ),
            source_kind="csv",
            problem_filepath=source_path,
            project_name=project_name,
        )
    raise ValueError(
        f"Unrecognized source '{source_path}'. Provide a JSON/Excel file, "
        "a directory with 'streams.csv' and 'utilities.csv', "
        "or a (streams, utilities) tuple."
    )


def _load_json_inputs(source_path: Path, *, source: PathLike) -> JsonDict:
    try:
        sample_case_name = _packaged_sample_case_name(source, source_path)
        if sample_case_name is not None:
            input_data = json.loads(read_sample_case(sample_case_name))
        else:
            input_data = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse JSON from {source_path}: {exc}") from exc
    if not isinstance(input_data, dict):
        raise ValueError(f"JSON inputs in {source_path} must be an object.")
    return input_data


def _packaged_sample_case_name(
    source: PathLike,
    source_path: Path,
) -> str | None:
    source_text = str(source)
    if source_path.exists() or source_path.name != source_text:
        return None
    if source_path.suffix.lower() != ".json":
        return None
    return source_path.name if source_path.name in set(list_sample_cases()) else None


__all__ = ["ExternalProblemSource", "PathLike", "load_external_problem_source"]
