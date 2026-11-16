"""Heat exchanger network retrofit on the Ciric and Floudas (1989) case.

Case data (Example 1 of Hallberg and Scribner, 2009, "Heat Exchanger Network
Retrofit Comparison", University of Oklahoma, adapted from Ciric and Floudas,
1989): three hot and two cold streams, a 500 C hot utility at $0.0113/MJ and
20-40 C cooling water at $0.00238/MJ, 350 days a year. The existing network
has seven exchangers; its hot utility is 17,759 kW and cold utility 15,510 kW.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import REPOSITORY_ROOT

CASE_JSON = (
    REPOSITORY_ROOT / "tests/fixtures/retrofit/Five-stream-Ciric-and-Floudas-1989.json"
)

# Existing exchangers (Figure 1 and Table 2 of the report): E1 H3-C2 and
# E3 H1-C1 in stage 1, E4 H2-C2 and E2 H3-C1 in stage 2, coolers E5 on H1 and
# E6 on H2, heater E7 on C2. Areas in m2.
EXISTING_AREAS = {
    "E1": 609.7,
    "E2": 579.2,
    "E3": 1008.5,
    "E4": 117.96,
    "E5": 787.5,
    "E6": 104.6,
    "E7": 246.75,
}
REPORTED_HOT_UTILITY = 17_759.0
REPORTED_COLD_UTILITY = 15_510.0


def _live_solver_available() -> bool:
    try:
        require_synthesis_dependency("pyomo.environ", purpose="retrofit test")
        require_solver_binary("ipopt", purpose="retrofit test")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
_needs_live_solver = pytest.mark.skipif(
    "not _live_solver_available()",
    reason="needs the ipopt binary and the pyomo package",
)


def _case_problem() -> PinchProblem:
    return PinchProblem(json.loads(CASE_JSON.read_text(encoding="utf-8")))


def _existing_network() -> HeatExchangerNetwork:
    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H3", "C2", 1), ("H1", "C1", 1), ("H2", "C2", 2), ("H3", "C1", 2)],
        heaters=["C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )
    names = {
        "recovery:H3->C2:S1": "E1",
        "recovery:H3->C1:S2": "E2",
        "recovery:H1->C1:S1": "E3",
        "recovery:H2->C2:S2": "E4",
        "cold-utility:H1->CU": "E5",
        "cold-utility:H2->CU": "E6",
        "hot-utility:HU->C2": "E7",
    }
    return structure.model_copy(
        update={
            "exchangers": tuple(
                exchanger.model_copy(
                    update={
                        "exchanger_id": names[exchanger.exchanger_id],
                        "area": EXISTING_AREAS[names[exchanger.exchanger_id]],
                    }
                )
                for exchanger in structure.exchangers
            )
        }
    )


def _retrofit_model(problem, network, **spec_changes):
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
        build_duty_optimisation_request,
        fixed_network_structure,
    )
    from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
        FixedStructureModel,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )

    problem.target.direct_heat_integration()
    arrays = problem_to_solver_arrays(problem, 10.0)
    request = build_duty_optimisation_request(network, objective="cost")
    structure = fixed_network_structure(request, arrays.axis_maps)
    model = FixedStructureModel(
        name="retrofit",
        solver="ipopt-pyomo",
        solver_arrays=arrays,
        spec=replace(structure.spec, **spec_changes),
        minimisation_goal="retrofit",
        default_approach=1.0,
    )
    model.optimise(print_output=False)
    return structure, model


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_goal_rates_the_existing_network_within_the_area_tolerance():
    problem = _case_problem()
    structure, model = _retrofit_model(
        problem, _existing_network(), area_tolerance=0.02
    )

    assert model.mSuccess == 1
    hot = sum(
        result.periods[0].duty
        for (kind, index), result in zip(structure.slots, structure.results(model))
        if kind == "utility" and structure.spec.utilities[index].side == "hot"
    )
    # Reported areas are rounded and the solver uses Chen's LMTD, so the
    # existing network rates within 2 % of the reported hot utility.
    assert hot == pytest.approx(REPORTED_HOT_UTILITY, rel=0.02)
    # Existing units keep their installed areas when none may be enlarged.
    areas = [result.area for result in structure.results(model)]
    assert areas == pytest.approx(
        [EXISTING_AREAS[e.exchanger_id] for e in _existing_network().exchangers]
    )


def _with_new_match(network, hot, cold, stage, exchanger_id):
    extra = HeatExchangerNetwork.from_structure(recovery=[(hot, cold, stage)])
    (new,) = extra.exchangers
    return network.model_copy(
        update={
            "exchangers": network.exchangers
            + (new.model_copy(update={"exchanger_id": exchanger_id}),)
        }
    )


def _hot_utility(structure, model) -> float:
    return sum(
        result.periods[0].duty
        for (kind, index), result in zip(structure.slots, structure.results(model))
        if kind == "utility" and structure.spec.utilities[index].side == "hot"
    )


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_goal_sizes_a_new_match_and_charges_only_its_area():
    problem = _case_problem()
    base_structure, base = _retrofit_model(
        problem, _existing_network(), area_tolerance=0.02
    )
    network = _with_new_match(_existing_network(), "H1", "C2", 3, "N1")
    structure, model = _retrofit_model(problem, network, area_tolerance=0.02)

    assert model.mSuccess == 1
    # H3 has no cooler and E2 stays at its installed area, so the new match
    # recovers only some hundreds of kW.
    assert _hot_utility(structure, model) < _hot_utility(base_structure, base) - 100.0
    results = structure.results(model)
    new = results[-1]
    assert new.area > 0.0
    assert new.added_area == pytest.approx(new.area)
    assert [r.added_area for r in results[:-1]] == [0.0] * (len(results) - 1)
    # Area cost law exponent 1: the retrofit capital is the annualised
    # coefficient times the added area.
    assert model.retrofit_capital_value == pytest.approx(
        float(model.A_coeff[0]) * new.area, rel=1e-6
    )


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_retrofit_goal_lets_an_enlargeable_unit_grow_at_a_cost():
    problem = _case_problem()
    base_structure, base = _retrofit_model(
        problem, _existing_network(), area_tolerance=0.02
    )
    # E1 (H3-C2) is the first recovery match.
    structure, model = _retrofit_model(
        problem,
        _existing_network(),
        area_tolerance=0.02,
        enlargeable_recovery=frozenset({0}),
    )

    assert model.mSuccess == 1
    e1 = structure.results(model)[0]
    assert e1.added_area > 1.0
    assert e1.area == pytest.approx(EXISTING_AREAS["E1"] + e1.added_area)
    assert _hot_utility(structure, model) < _hot_utility(base_structure, base)
    assert model.retrofit_capital_value == pytest.approx(
        float(model.A_coeff[0]) * e1.added_area, rel=1e-6
    )
