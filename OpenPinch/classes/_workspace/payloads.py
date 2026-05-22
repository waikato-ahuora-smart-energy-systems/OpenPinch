"""Payload normalization helpers for :class:`OpenPinch.classes.PinchWorkspace`."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from ...lib.schemas.io import TargetInput
from ..pinch_problem import PinchProblem

JsonDict = Dict[str, Any]
PathLike = str | Path


def normalise_payload(payload: TargetInput | JsonDict) -> JsonDict:
    """Return a defensive JSON-style payload copy for workspace storage."""
    if isinstance(payload, TargetInput):
        return payload.model_dump(mode="python")
    if not isinstance(payload, dict):
        raise TypeError("Workspace payloads must be a dict or TargetInput instance.")
    return deepcopy(payload)


def project_name_from_payload(payload: JsonDict) -> Optional[str]:
    """Extract the root project name from a canonical zone tree when present."""
    zone_tree = payload.get("zone_tree")
    if isinstance(zone_tree, dict):
        name = zone_tree.get("name")
        if name not in (None, ""):
            return str(name)
    return None


def merge_payloads(base: JsonDict, overlay: JsonDict) -> JsonDict:
    """Deep-merge two payload fragments for variant editing workflows."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_payloads(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def canonical_payload_from_source(
    source: (
        TargetInput
        | JsonDict
        | PathLike
        | tuple[PathLike, PathLike]
        | PinchProblem
    ),
    *,
    project_name: Optional[str],
    workspace_project_name: Optional[str],
) -> tuple[JsonDict, str]:
    """Normalize one workspace source into a canonical stored payload."""
    if isinstance(source, PinchProblem):
        payload = source.canonical_problem_json()
        resolved_project_name = (
            project_name
            or source.project_name
            or project_name_from_payload(payload)
            or workspace_project_name
            or "Site"
        )
        return payload, resolved_project_name

    if isinstance(source, TargetInput):
        payload = normalise_payload(source)
        resolved_project_name = (
            project_name
            or project_name_from_payload(payload)
            or workspace_project_name
            or "Site"
        )
        return payload, resolved_project_name

    normalized = normalise_payload(source) if isinstance(source, dict) else source
    seed_project_name = (
        project_name
        or workspace_project_name
        or (
            project_name_from_payload(normalized)
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

    payload = problem.canonical_problem_json()
    resolved_project_name = (
        project_name
        or project_name_from_payload(payload)
        or problem.project_name
        or workspace_project_name
        or "Site"
    )
    return payload, resolved_project_name
