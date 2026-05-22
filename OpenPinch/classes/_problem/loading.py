"""Problem-source loading helpers for :class:`OpenPinch.classes.PinchProblem`."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pydantic import ValidationError

from ...lib.schemas.io import TargetInput
from ...utils.input_validation import validate_stream_data, validate_utility_data
from .validation import build_validation_context

JsonDict = Dict[str, Any]
PathLike = str | Path


@dataclass(frozen=True)
class ProblemSourceAdapters:
    """External readers and resource lookups used during source loading."""

    get_problem_from_excel: Callable[[PathLike, Optional[str]], JsonDict]
    get_problem_from_csv: Callable[[PathLike, PathLike, Optional[str]], JsonDict]
    list_sample_cases: Callable[[], list[str]]
    read_sample_case: Callable[[str], str]


@dataclass(frozen=True)
class LoadedProblemSource:
    """Normalized payload and metadata produced by source loading."""

    payload: TargetInput | JsonDict
    source_kind: str
    validation_context: dict[str, list[dict[str, Any]]]
    problem_filepath: Optional[Path]
    project_name: Optional[str] = None


def normalize_problem_mapping(payload: JsonDict) -> JsonDict:
    """Return a defensive copy of one in-memory payload with cleaned records."""
    normalized = deepcopy(payload)
    if isinstance(normalized.get("streams"), list):
        normalized["streams"] = validate_stream_data(normalized["streams"])
    if isinstance(normalized.get("utilities"), list):
        normalized["utilities"] = validate_utility_data(normalized["utilities"])
    return normalized


def prepare_in_memory_problem_source(
    source: TargetInput | JsonDict,
    *,
    source_kind: str,
) -> LoadedProblemSource:
    """Normalize one in-memory payload without reading from the filesystem."""
    if isinstance(source, TargetInput):
        payload: TargetInput | JsonDict = source
        context_payload = source.model_dump(mode="python")
    else:
        payload = normalize_problem_mapping(source)
        context_payload = payload

    return LoadedProblemSource(
        payload=payload,
        source_kind=source_kind,
        validation_context=build_validation_context(
            context_payload,
            source_kind=source_kind,
        ),
        problem_filepath=None,
    )


def load_problem_source(
    source: TargetInput | JsonDict | PathLike | tuple[PathLike, PathLike],
    *,
    current_project_name: Optional[str],
    adapters: ProblemSourceAdapters,
) -> LoadedProblemSource:
    """Load one supported problem source into a normalized payload bundle."""
    source = _coerce_target_input(source)

    if isinstance(source, TargetInput):
        return prepare_in_memory_problem_source(source, source_kind="target_input")

    if isinstance(source, dict):
        return prepare_in_memory_problem_source(source, source_kind="target_input")

    if isinstance(source, tuple) and len(source) == 2:
        streams_csv, utilities_csv = map(Path, source)
        payload = adapters.get_problem_from_csv(
            streams_csv,
            utilities_csv,
            output_json=None,
        )
        return LoadedProblemSource(
            payload=payload,
            source_kind="csv",
            validation_context=build_validation_context(payload, source_kind="csv"),
            problem_filepath=None,
        )

    src_path = Path(source)
    resolved_project_name = (
        src_path.stem if current_project_name == "Untitled" else current_project_name
    )

    if src_path.suffix.lower() == ".json":
        payload = _load_json_payload(src_path, source=source, adapters=adapters)
        return LoadedProblemSource(
            payload=payload,
            source_kind="json",
            validation_context=build_validation_context(payload, source_kind="json"),
            problem_filepath=src_path,
            project_name=resolved_project_name,
        )

    if src_path.suffix.lower() in {".xlsx", ".xls", ".xlsb", ".xlsm"}:
        payload = adapters.get_problem_from_excel(src_path, output_json=None)
        return LoadedProblemSource(
            payload=payload,
            source_kind="excel",
            validation_context=build_validation_context(payload, source_kind="excel"),
            problem_filepath=src_path,
            project_name=resolved_project_name,
        )

    if src_path.is_dir():
        streams_csv = src_path / "streams.csv"
        utilities_csv = src_path / "utilities.csv"
        if not streams_csv.exists() or not utilities_csv.exists():
            raise FileNotFoundError(
                f"CSV directory '{src_path}' must contain "
                "'streams.csv' and 'utilities.csv'."
            )
        payload = adapters.get_problem_from_csv(
            streams_csv,
            utilities_csv,
            output_json=None,
        )
        return LoadedProblemSource(
            payload=payload,
            source_kind="csv",
            validation_context=build_validation_context(payload, source_kind="csv"),
            problem_filepath=src_path,
            project_name=resolved_project_name,
        )

    raise ValueError(
        f"Unrecognized source '{src_path}'. Provide a JSON/Excel file, "
        "a directory with 'streams.csv' and 'utilities.csv', "
        "or a (streams, utilities) tuple."
    )


def find_zone_tree_node(
    zone_tree: dict[str, Any],
    zone_name: str,
) -> dict[str, Any]:
    """Return one zone-tree node addressed by absolute or relative name."""
    root_name = str(zone_tree.get("name") or "")
    path_parts = [part.strip() for part in str(zone_name).split("/") if part.strip()]
    if not path_parts:
        raise ValueError("zone_name must identify a zone in the zone_tree.")

    if path_parts[0] == root_name:
        path_parts = path_parts[1:]

    node = zone_tree
    if not path_parts:
        return node

    for part in path_parts:
        children = node.get("children") or []
        next_node = next(
            (child for child in children if str(child.get("name")) == part),
            None,
        )
        if next_node is None:
            raise ValueError(f"Zone {zone_name!r} was not found in the zone_tree.")
        node = next_node
    return node


def _coerce_target_input(
    source: TargetInput | JsonDict | PathLike | tuple[PathLike, PathLike],
) -> TargetInput | JsonDict | PathLike | tuple[PathLike, PathLike]:
    if isinstance(source, (TargetInput, dict)):
        return source
    try:
        return TargetInput.model_validate(source)
    except ValidationError:
        return source


def _load_json_payload(
    src_path: Path,
    *,
    source: PathLike,
    adapters: ProblemSourceAdapters,
) -> JsonDict:
    try:
        sample_case_name = _packaged_sample_case_name(
            source,
            src_path,
            sample_case_names=adapters.list_sample_cases(),
        )
        if sample_case_name is not None:
            payload = json.loads(adapters.read_sample_case(sample_case_name))
        else:
            with src_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse JSON from {src_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload in {src_path} must be an object.")
    return normalize_problem_mapping(payload)


def _packaged_sample_case_name(
    source: PathLike,
    src_path: Path,
    *,
    sample_case_names: list[str],
) -> Optional[str]:
    source_text = str(source)
    if src_path.exists():
        return None
    if src_path.name != source_text:
        return None
    if src_path.suffix.lower() != ".json":
        return None
    if src_path.name not in set(sample_case_names):
        return None
    return src_path.name
