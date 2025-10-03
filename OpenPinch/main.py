"""High-level orchestration for running an OpenPinch analysis.

The functions in this module wire together data validation, pinch targeting,
and output formatting.  They act as the main entry points used by the
``PinchProblem`` helper class as well as external callers embedding OpenPinch
in larger workflows.
"""

from typing import Any

from .analysis import prepare_problem
from .lib import *
from .scales import get_targets, extract_results
from .utils import *

__all__ = ["pinch_analysis_service"]


def pinch_analysis_service(data: Any, project_name: str = "Project") -> TargetOutput:
    """Validate user data, run the targeting workflow, and return structured results.

    Parameters
    ----------
    data:
        Raw request payload matching :class:`OpenPinch.lib.schema.TargetInput`.
        Dictionaries, Pydantic models, and dataclass-like objects are accepted.
    project_name:
        Optional label used in generated graphs and result files.

    Returns
    -------
    TargetOutput
        Pydantic model containing site, zone, and utility targets plus summary
        tables ready for serialisation.
    """
    # Validate request data using Pydantic model
    request_data = TargetInput.model_validate(data)

    # Formulate the top level zone with all subzones and approperiate input data
    master_zone = prepare_problem(
        project_name=project_name,
        streams=request_data.streams,
        utilities=request_data.utilities,
        options=request_data.options,
        zone_tree=request_data.zone_tree,
    )

    # Perform advanced targeting analysis on the master zone and all subzones
    master_zone = get_targets(master_zone)

    # Extract the core results from the master zone
    return_data = extract_results(master_zone)

    # Validate response data
    validated_data = TargetOutput.model_validate(return_data)

    # Return data
    return validated_data
