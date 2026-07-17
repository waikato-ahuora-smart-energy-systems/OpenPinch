"""Workspace case selection and cache-state operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional


def case_for_name(workspace, name: Optional[str]):
    """Return one cached or newly constructed live problem case."""
    from ..problem import PinchProblem
    from .case_inputs import project_name_from_case_input

    resolved_name = resolve_case_name(workspace, name)
    cached = workspace._case_cache.get(resolved_name)
    if cached is not None:
        if workspace.project_name:
            cached.project_name = workspace.project_name
        return cached

    case_input = deepcopy(workspace._variant_inputs[resolved_name])
    project_name = (
        workspace.project_name or project_name_from_case_input(case_input) or "Site"
    )
    problem = PinchProblem(source=case_input, project_name=project_name)
    if workspace.project_name:
        problem.project_name = workspace.project_name
    workspace._case_cache[resolved_name] = problem
    return problem


def default_case_name(workspace) -> Optional[str]:
    """Return and activate the deterministic default case, when one exists."""
    if workspace._active_case_name in workspace._variant_inputs:
        return workspace._active_case_name
    if workspace.baseline_name in workspace._variant_inputs:
        workspace._active_case_name = workspace.baseline_name
        return workspace._active_case_name
    if workspace._variant_inputs:
        workspace._active_case_name = next(iter(workspace._variant_inputs))
        return workspace._active_case_name
    return None


def resolve_case_name(workspace, name: Optional[str]) -> str:
    """Resolve one optional case name against current workspace state."""
    if name is None:
        resolved = default_case_name(workspace)
        if resolved is None:
            raise KeyError("No cases are loaded in this PinchWorkspace.")
        return resolved
    if name not in workspace._variant_inputs:
        available = ", ".join(workspace.list_cases())
        raise KeyError(f"Unknown case {name!r}. Available cases: {available}")
    return name


def invalidate_variant_state(workspace, name: str) -> None:
    """Drop cached case and view state for one variant input."""
    workspace._cached_views.pop(name, None)
    workspace._case_cache.pop(name, None)


def sync_case_input(workspace, name: str) -> None:
    """Synchronize one live problem's canonical input into workspace storage."""
    problem = workspace._case_cache.get(name)
    if problem is None:
        return
    case_input = problem.canonical_problem_json()
    if workspace._variant_inputs.get(name) != case_input:
        workspace._variant_inputs[name] = case_input
        workspace._cached_views.pop(name, None)
