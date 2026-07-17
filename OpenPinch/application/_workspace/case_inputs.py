"""Case-input normalization for the application workspace."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from ...adapters.io.problem_sources import PathLike
from ...contracts.input import TargetInput
from ..problem import PinchProblem

JsonDict = Dict[str, Any]


def normalise_case_input(case_input: TargetInput | JsonDict) -> JsonDict:
    """Return a defensive JSON-style case input copy for workspace storage."""
    if isinstance(case_input, TargetInput):
        return case_input.model_dump(mode="python")
    if not isinstance(case_input, dict):
        raise TypeError("Workspace case inputs must be a dict or TargetInput instance.")
    return deepcopy(case_input)


def project_name_from_case_input(case_input: JsonDict) -> Optional[str]:
    """Extract the root project name from a canonical zone tree when present."""
    zone_tree = case_input.get("zone_tree")
    if isinstance(zone_tree, dict):
        name = zone_tree.get("name")
        if name not in (None, ""):
            return str(name)
    return None


def merge_case_inputs(base: JsonDict, overlay: JsonDict) -> JsonDict:
    """Deep-merge two case input fragments for case scenario workflows."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_case_inputs(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def canonical_case_input_from_source(
    source: (
        TargetInput | JsonDict | PathLike | tuple[PathLike, PathLike] | PinchProblem
    ),
    *,
    project_name: Optional[str],
    workspace_project_name: Optional[str],
) -> tuple[JsonDict, str]:
    """Normalize one workspace source into a canonical stored case input."""
    if isinstance(source, PinchProblem):
        case_input = source.to_problem_json()
        resolved_project_name = (
            project_name
            or source.project_name
            or project_name_from_case_input(case_input)
            or workspace_project_name
            or "Site"
        )
        return case_input, resolved_project_name

    if isinstance(source, TargetInput):
        case_input = normalise_case_input(source)
        resolved_project_name = (
            project_name
            or project_name_from_case_input(case_input)
            or workspace_project_name
            or "Site"
        )
        return case_input, resolved_project_name

    normalized = normalise_case_input(source) if isinstance(source, dict) else source
    seed_project_name = (
        project_name
        or workspace_project_name
        or (
            project_name_from_case_input(normalized)
            if isinstance(normalized, dict)
            else None
        )
        or "Site"
    )
    try:
        problem = PinchProblem(
            source=deepcopy(normalized),
            project_name=seed_project_name,
        )
    except ValueError:
        if isinstance(normalized, dict):
            return normalized, seed_project_name
        raise

    case_input = problem.to_problem_json()
    resolved_project_name = (
        project_name
        or project_name_from_case_input(case_input)
        or problem.project_name
        or workspace_project_name
        or "Site"
    )
    return case_input, resolved_project_name
