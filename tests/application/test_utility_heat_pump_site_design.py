"""The site design exchanges a utility heat pump with its utility levels.

``design.total_site_heat_exchanger_network`` on a case that holds a
utility heat pump's two zones designs them with the site's processes.
The condenser raises MPS and the evaporator is heated from LPS, each at
its whole duty, and the site imports only what the heat pump target
left: none of the 800 kW of MPS it bought.

The heat pump's streams are a converged target's, moved to the ends of
the level bands they sat in. They then stand 1 K, the heat pump's
approach, from the levels, which the design models accept.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.case_derivation import _case_with_utility_heat_pump_zones
from OpenPinch.domain.enums import HeatExchangerKind
from tests.analysis.heat_pumps.two_level_site import (
    SITE_HOT_UTILITY,
    recovering_two_level_site,
)
from tests.application.test_utility_heat_pump_level_bands import (
    CONDENSER_DUTY,
    EVAPORATOR_DUTY,
    _converged_target,
)


def _solvers_available() -> bool:
    try:
        backend.require_solver_backend("highs")
        backend.require_solver_backend("ipopt-pyomo")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _utility_duties(network) -> dict[tuple[str, str], float]:
    """The duty of each heater and cooler, by (source, sink)."""
    return {
        (exchanger.source_stream, exchanger.sink_stream): sum(
            float(state.duty) for state in exchanger.period_states if state.active
        )
        for exchanger in network.exchangers
        if exchanger.kind is not HeatExchangerKind.RECOVERY
    }


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_the_site_design_raises_and_draws_the_levels_the_heat_pump_was_placed_on():
    case = _case_with_utility_heat_pump_zones(
        recovering_two_level_site(), _converged_target(), dt_cont=1.0
    )

    view = case.design.total_site_heat_exchanger_network(width=0, depth=1)

    selected = view.selected_network
    duties = _utility_duties(selected)
    assert duties[("HP condenser.HP_H1", "Cold Utility.MPS")] == pytest.approx(
        CONDENSER_DUTY, abs=1e-3
    )
    assert duties[("Hot Utility.LPS", "HP evaporator.HP_C1")] == pytest.approx(
        EVAPORATOR_DUTY, abs=1e-3
    )
    # MPS used by process B less MPS raised by the condenser: none bought.
    used = sum(d for (source, _), d in duties.items() if source == "Hot Utility.MPS")
    assert used == pytest.approx(SITE_HOT_UTILITY, abs=1e-3)
    assert used - CONDENSER_DUTY == pytest.approx(0.0, abs=1e-3)
    assert verify_network_feasibility(selected) == ()
