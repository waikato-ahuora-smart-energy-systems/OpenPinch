"""A retrofit search can start from a heat pump added as streams.

The heat pump is one machine of a fixed size, the as-piped target's: its
condensers and evaporators are streams of the case, and the search's
first design is the existing network with the matches that serve them.
Solved, the heaters and coolers give up exactly what the machine
delivers and absorbs. On the levels case a full heat pump search finds
186.5 kW of the 240 kW of steam and 129.7 kW of the 240 kW of cooling
water in 64 s; the test stops the search at 200 evaluations (0.4 s),
which gives a machine of three condensers and two evaporators that
delivers 156 kW.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    HeatPumpStep,
)
from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_heat_pump,
)
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
    run_retrofit_search,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.domain.heat_exchanger_network import NetworkHeatPump
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
    with_machine_streams,
)


def _solvers_available() -> bool:
    try:
        backend.require_solver_backend("ipopt-pyomo")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _duty(streams) -> float:
    return sum(float(stream.heat_flow.value) for _key, stream in streams.items())


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs ipopt")
def test_a_search_started_from_a_heat_pump_rates_at_the_machines_duties():
    problem = levels_problem()
    existing = problem.design._network_from_structure(EXISTING)
    settings = RetrofitSettings(max_modifications=1, approach=None, new_unit_cost=0.0)
    base = run_retrofit_search(
        problem, existing, replace(settings, max_modifications=0)
    ).frontier[0]
    machine = as_piped_heat_pump(
        problem, base.network, args={"maximum_evaluations": 200}
    )
    delivered, absorbed = (
        _duty(machine.hpr_hot_streams),
        _duty(machine.hpr_cold_streams),
    )
    assert delivered > 150.0
    step = HeatPumpStep(
        record=NetworkHeatPump(
            machine="HP",
            cycle="carnot",
            condenser_duty=delivered,
            evaporator_duty=absorbed,
            work=float(machine.w_net),
            cop=float(machine.cop_h),
            electricity_cost=0.0,
            capital_cost=0.0,
            annual_capital_cost=0.0,
        ),
        condensers=tuple(s for _key, s in machine.hpr_hot_streams.items()),
        evaporators=tuple(s for _key, s in machine.hpr_cold_streams.items()),
        base=base,
    )

    result = run_retrofit_search(
        with_machine_streams(problem, machine), existing, settings, heat_pump=step
    )

    design = result.frontier[0]
    assert design.hot_utility == pytest.approx(base.hot_utility - delivered, abs=0.01)
    assert design.cold_utility == pytest.approx(base.cold_utility - absorbed, abs=0.01)
