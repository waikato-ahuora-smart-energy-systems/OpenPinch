"""Workspace case selection and cache-state operations."""

from __future__ import annotations

from typing import Optional


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
