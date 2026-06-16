"""Workflow execution helpers for :class:`OpenPinch.classes.PinchWorkspace`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..pinch_problem import PinchProblem

_WORKFLOW_SUPPORT_LEVELS = {
    "target": "stable",
    "direct_heat_integration": "stable",
    "indirect_heat_integration": "stable",
    "direct_heat_pump": "advanced",
    "indirect_heat_pump": "advanced",
    "direct_refrigeration": "advanced",
    "indirect_refrigeration": "advanced",
    "cogeneration": "advanced",
    "area_cost": "advanced",
    "heat_exchanger_network_synthesis": "advanced",
}

_DESIGN_WORKFLOWS = {"heat_exchanger_network_synthesis"}


@dataclass
class WorkspaceExecutionError(RuntimeError):
    """Structured workflow failure raised by workspace execution helpers."""

    category: str
    message: str

    def __str__(self) -> str:
        return self.message


def workflow_support_level(workflow: str) -> str:
    """Return the declared support level for one workflow name."""
    normalized = normalise_workflow_name(workflow)
    return _WORKFLOW_SUPPORT_LEVELS.get(normalized, "unsupported")


def workflow_warnings(workflow: str, support_level: str) -> list[str]:
    """Return user-facing warnings for advanced or unsupported workflows."""
    if support_level == "advanced":
        return [
            f"Workflow '{workflow}' should be treated as an advanced "
            "PinchWorkspace workflow."
        ]
    if support_level == "unsupported":
        return [f"Workflow '{workflow}' is not a supported PinchWorkspace workflow."]
    return []


def run_problem_workflow(
    problem: PinchProblem,
    workflow: str,
    workflow_options: dict[str, Any],
    *,
    workspace_variant: str | None = None,
) -> None:
    """Execute one named workflow against a live :class:`PinchProblem`."""
    normalized = normalise_workflow_name(workflow)
    if normalized == "target":
        problem.target()
        return

    if normalized in _DESIGN_WORKFLOWS:
        if not hasattr(problem.design, normalized):
            raise WorkspaceExecutionError(
                category="unsupported_workflow",
                message=(
                    f"Unknown design workflow {workflow!r}. Supported workflows "
                    f"include: {', '.join(sorted(_WORKFLOW_SUPPORT_LEVELS))}."
                ),
            )
        try:
            method = getattr(problem.design, normalized)
            method(**workflow_options, workspace_variant=workspace_variant)
            return
        except WorkspaceExecutionError:
            raise
        except Exception as exc:
            raise WorkspaceExecutionError(
                category="workflow_runtime",
                message=str(exc),
            ) from exc

    if not hasattr(problem.target, normalized):
        raise WorkspaceExecutionError(
            category="unsupported_workflow",
            message=(
                f"Unknown workflow {workflow!r}. Supported workflows include: "
                f"target, {', '.join(sorted(_WORKFLOW_SUPPORT_LEVELS))}."
            ),
        )

    try:
        problem.target()
        method = getattr(problem.target, normalized)
        method(**workflow_options)
    except WorkspaceExecutionError:
        raise
    except Exception as exc:
        raise WorkspaceExecutionError(
            category="workflow_runtime",
            message=str(exc),
        ) from exc


def normalise_workflow_name(workflow: str) -> str:
    """Normalize one user-supplied workflow name for attribute lookup."""
    return str(workflow).strip().lower().replace("-", "_").replace(" ", "_")
