"""Compatibility-protected service for running an OpenPinch analysis."""

from typing import Any

from .application.problem import PinchProblem
from .contracts.input import TargetInput
from .contracts.output import TargetOutput

__all__ = ["pinch_analysis_service"]


################################################################################
# Public API
################################################################################


def pinch_analysis_service(
    data: Any,
    project_name: str = "Project",
) -> TargetOutput:
    """Validate input data, run targeting, and return ``TargetOutput``.

    Parameters
    ----------
    data:
        Raw request data matching :class:`OpenPinch.contracts.input.TargetInput`.
        Dictionaries, Pydantic models, and dataclass-like objects are accepted.
    project_name:
        Optional label used in generated graphs and result files.

    Returns
    -------
    TargetOutput
        Validated response data containing solved targets and graph data.
    """
    input_data = TargetInput.model_validate(data)
    problem = PinchProblem(
        project_name=project_name,
        source=input_data,
    )
    problem.target()
    return TargetOutput.model_validate(problem.results)
