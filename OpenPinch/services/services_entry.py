"""TODO: intro"""

from typing import Any

from ..classes.zone import Zone
from ..lib.schema import TargetInput, TargetOutput

from . import (
    prepare_problem,
    compute_direct_integration_targets,
    compute_indirect_integration_targets,

)

__all__ = [
    "data_preprocessing_service",
    "direct_heat_integration_service",
    "indirect_heat_integration_service",
]

def data_preprocessing_service(
    input_data: Any,        
    project_name: str = "Site",
) -> Zone:
    input_data = TargetInput.model_validate(input_data)
    return prepare_problem(
        project_name=project_name,
        streams=input_data.streams,
        utilities=input_data.utilities,
        options=input_data.options,
        zone_tree=input_data.zone_tree,        
    )


def direct_heat_integration_service(zone: Zone) -> Zone:
    return compute_direct_integration_targets(zone)


def indirect_heat_integration_service(zone: Zone) -> Zone:
    return compute_indirect_integration_targets(zone)



