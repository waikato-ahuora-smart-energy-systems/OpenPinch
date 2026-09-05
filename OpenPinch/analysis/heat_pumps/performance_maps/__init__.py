"""Engine-neutral HPR performance-map generation internals."""

from .context import build_hpr_map_generation_context
from .errors import HprMapGenerationError
from .factory import get_hpr_point_simulator
from .fluids import (
    parse_hpr_working_fluid,
    resolve_hpr_working_fluid,
    to_tespy_fluid_token,
)
from .generation import generate_hpr_performance_map
from .models import (
    HprMapGenerationContext,
    HprOperatingPoint,
    HprPointSimulation,
    HprSimulationDiagnostic,
    HprSimulatorMetadata,
    HprTargetMapBasis,
    HprWorkingFluidSpec,
)
from .points import iter_hpr_operating_points
from .protocols import HprPointSimulator, HprPointSimulatorFactory
from .target_basis import (
    HprPerformanceMapCompatibilityError,
    build_hpr_target_map_basis,
)

__all__ = [
    "HprMapGenerationContext",
    "HprMapGenerationError",
    "HprPerformanceMapCompatibilityError",
    "HprOperatingPoint",
    "HprPointSimulation",
    "HprSimulationDiagnostic",
    "HprSimulatorMetadata",
    "HprTargetMapBasis",
    "HprWorkingFluidSpec",
    "HprPointSimulator",
    "HprPointSimulatorFactory",
    "build_hpr_map_generation_context",
    "build_hpr_target_map_basis",
    "generate_hpr_performance_map",
    "get_hpr_point_simulator",
    "iter_hpr_operating_points",
    "parse_hpr_working_fluid",
    "resolve_hpr_working_fluid",
    "to_tespy_fluid_token",
]
