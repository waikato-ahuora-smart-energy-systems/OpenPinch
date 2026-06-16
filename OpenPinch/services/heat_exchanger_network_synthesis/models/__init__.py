"""Private equation-model boundary for OpenPinch HEN synthesis.

The symbols in this package are implementation details behind the
problem-rooted synthesis service. Importing the package must stay lightweight;
GEKKO, Pyomo, and external solver checks happen only when a concrete backend is
created or configured.
"""

from __future__ import annotations

from .backend import (
    GEKKO_SOLVERS,
    PYOMO_SOLVER_BINARIES,
    PYOMO_SOLVERS,
    SolverRun,
    configure_gekko_solver,
    create_gekko_model,
    require_solver_backend,
    solve_gekko_model,
)
from .base import BaseHeatExchangerNetworkModel
from .extraction import (
    extract_heat_exchanger_network,
    extract_network_synthesis_result,
)
from .problem import (
    InternalHeatExchangerNetworkProblem,
    ModelSliceUnavailableError,
)

__all__ = [
    "BaseHeatExchangerNetworkModel",
    "GEKKO_SOLVERS",
    "InternalHeatExchangerNetworkProblem",
    "ModelSliceUnavailableError",
    "PYOMO_SOLVERS",
    "PYOMO_SOLVER_BINARIES",
    "SolverRun",
    "configure_gekko_solver",
    "create_gekko_model",
    "extract_heat_exchanger_network",
    "extract_network_synthesis_result",
    "require_solver_backend",
    "solve_gekko_model",
]
