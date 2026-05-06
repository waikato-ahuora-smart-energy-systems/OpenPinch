"""High-level orchestration for running an OpenPinch analysis.

The functions in this module wire together data validation, pinch targeting,
and output formatting.  They act as the main entry points used by the
``PinchProblem`` helper class as well as external callers embedding OpenPinch
in larger workflows.
"""

from typing import Any

from .classes.pinch_problem import PinchProblem
from .lib.schema import TargetInput, TargetOutput


__all__ = ["pinch_analysis_service"]


#######################################################################################################
# Public API
#######################################################################################################


def pinch_analysis_service(
    data: Any,
    project_name: str = "Project",
) -> TargetOutput:
    """Validate user data, run the targeting workflow, and return structured results.

    Parameters
    ----------
    data:
        Raw request payload matching :class:`OpenPinch.lib.schema.TargetInput`.
        Dictionaries, Pydantic models, and dataclass-like objects are accepted.
    project_name:
        Optional label used in generated graphs and result files.
    is_return_full_results:
        When ``True``, return both the validated
        :class:`~OpenPinch.lib.schema.TargetOutput` and the solved
        :class:`~OpenPinch.classes.zone.Zone` hierarchy.

    Returns
    -------
    TargetOutput or tuple[TargetOutput, Zone]
        Validated response payload, optionally paired with the in-memory zone
        tree for advanced inspection and post-processing.
    """
    input_data = TargetInput.model_validate(data)
    problem = PinchProblem(project_name=project_name)
    problem.load(input_data)
    problem.target()
    return TargetOutput.model_validate(problem.results)

