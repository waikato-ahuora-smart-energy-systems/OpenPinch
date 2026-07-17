"""Source normalization and state replacement for application problems."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from pydantic import ValidationError

from ....adapters.io.problem_sources import (
    PathLike,
    load_external_problem_source,
)
from ....adapters.io.records import validate_stream_data, validate_utility_data
from ....contracts.input import TargetInput
from .validation import build_validation_context

if TYPE_CHECKING:
    from ....domain.zone import Zone
    from ...problem import PinchProblem

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class _LoadedProblemSource:
    """Normalized problem inputs and metadata produced by source loading."""

    input_data: TargetInput | JsonDict
    source_kind: str
    validation_context: dict[str, list[dict[str, Any]]]
    problem_filepath: Any | None
    project_name: Optional[str] = None


def apply_loaded_source(
    problem: "PinchProblem",
    loaded_source: _LoadedProblemSource,
) -> None:
    """Apply one normalized source bundle to a parent problem."""
    problem._problem_data = loaded_source.input_data
    problem._input_source_kind = loaded_source.source_kind
    problem._validation_context = loaded_source.validation_context
    problem._problem_filepath = loaded_source.problem_filepath
    if loaded_source.project_name is not None:
        problem._project_name = loaded_source.project_name


def rebuild_problem_state(
    problem: "PinchProblem",
    *,
    preprocessing: Callable[[], "Zone"],
) -> "Zone":
    """Revalidate, reconstruct domain state, and clear dependent caches."""
    problem._validated_data = problem.validate()
    problem._master_zone = preprocessing()
    problem._process_components = {}
    problem._results = None
    problem._last_target_run_spec = None
    return problem._master_zone


def replace_problem_inputs(
    problem: "PinchProblem",
    problem_inputs: JsonDict,
) -> "Zone":
    """Replace canonical inputs while preserving external source identity."""
    current_filepath = problem._problem_filepath
    loaded_source = prepare_in_memory_problem_source(
        problem_inputs,
        source_kind=problem._input_source_kind or "target_input",
    )
    apply_loaded_source(problem, loaded_source)
    problem._problem_filepath = current_filepath
    return problem._rebuild_problem_state()


def normalize_problem_mapping(input_data: JsonDict) -> JsonDict:
    """Return a defensive copy of one in-memory problem definition."""
    normalized = deepcopy(input_data)
    if isinstance(normalized.get("streams"), list):
        normalized["streams"] = validate_stream_data(normalized["streams"])
    if isinstance(normalized.get("utilities"), list):
        normalized["utilities"] = validate_utility_data(normalized["utilities"])
    return normalized


def prepare_in_memory_problem_source(
    source: TargetInput | JsonDict,
    *,
    source_kind: str,
) -> _LoadedProblemSource:
    """Normalize one in-memory problem definition without filesystem access."""
    if isinstance(source, TargetInput):
        input_data: TargetInput | JsonDict = source
        context_data = source.model_dump(mode="python")
    else:
        input_data = normalize_problem_mapping(source)
        context_data = input_data

    return _LoadedProblemSource(
        input_data=input_data,
        source_kind=source_kind,
        validation_context=build_validation_context(
            context_data,
            source_kind=source_kind,
        ),
        problem_filepath=None,
    )


def load_problem_source(
    source: TargetInput | JsonDict | PathLike | tuple[PathLike, PathLike],
    *,
    current_project_name: Optional[str],
) -> _LoadedProblemSource:
    """Load one supported problem source into a normalized input-data bundle."""
    source = _coerce_target_input(source)

    if isinstance(source, TargetInput):
        return prepare_in_memory_problem_source(source, source_kind="target_input")

    if isinstance(source, dict):
        return prepare_in_memory_problem_source(source, source_kind="target_input")

    loaded = load_external_problem_source(
        source,
        current_project_name=current_project_name,
    )
    input_data = normalize_problem_mapping(loaded.input_data)
    return _LoadedProblemSource(
        input_data=input_data,
        source_kind=loaded.source_kind,
        validation_context=build_validation_context(
            input_data,
            source_kind=loaded.source_kind,
        ),
        problem_filepath=loaded.problem_filepath,
        project_name=loaded.project_name,
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
