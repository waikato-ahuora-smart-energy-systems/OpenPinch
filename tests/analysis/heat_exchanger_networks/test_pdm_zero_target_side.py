"""A pinch side with recovery to do is designed even at zero utility target.

A threshold problem can leave one side of the pinch with no utility, yet its
streams still exchange heat there. The pinch design method must solve that
side and keep its matches in the amalgamated network.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver import (
    pinch_design_decomposition as decomposition_module,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem

D_TMIN = 10.0


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    "not _highs_available()", reason="needs highspy and pyomo"
)


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _stream(name, supply, target, heat_flow):
    return {
        "zone": "Site/Plant",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(D_TMIN / 2, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, supply, target):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(10.0, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
    }


UTILITIES = [
    _utility("LPS", "Hot", 140.0, 139.9),
    _utility("HPS", "Hot", 250.0, 249.9),
    _utility("CW", "Cold", 20.0, 25.0),
]

# No cold utility below the pinch: H2 heats C1 from 60 to 90 C there.
NO_COLD_UTILITY = [
    _stream("H1", 220.0, 150.0, 70.0),
    _stream("H2", 100.0, 70.0, 30.0),
    _stream("C1", 60.0, 200.0, 140.0),
]

# The mirror image: no hot utility above the pinch, where H1 heats C2.
NO_HOT_UTILITY = [
    _stream("H1", 210.0, 70.0, 140.0),
    _stream("C1", 50.0, 120.0, 70.0),
    _stream("C2", 170.0, 200.0, 30.0),
]


def _problem(streams) -> PinchProblem:
    return PinchProblem(
        source={
            "streams": streams,
            "utilities": UTILITIES,
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


@pytest.mark.parametrize("heuristic_network_limit", [0, 3], ids=["milp", "heuristic"])
@pytest.mark.parametrize(
    "streams", [NO_COLD_UTILITY, NO_HOT_UTILITY], ids=["below", "above"]
)
def test_a_side_with_recovery_but_no_utility_target_keeps_its_matches(
    streams, heuristic_network_limit
):
    problem = _problem(streams)
    pdm = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        name="zero-target-side",
        framework="PDM",
        solver="highs",
        dTmin=D_TMIN,
        minimisation_goal="total utility",
        pinch_decompositions={
            location: decomposition_module.build_pinch_design_decomposition(
                problem, D_TMIN, pinch_location=location
            )
            for location in ("above", "below")
        },
        heuristic_network_limit=heuristic_network_limit,
    )

    case = pdm.get_solution(print_output=False)

    assert case is not None, getattr(pdm, "solution_failure_reason", None)
    network = pdm.extract_network(run_id="zero-target-side")
    assert verify_network_feasibility(network) == ()


def test_a_side_whose_streams_carry_no_more_than_tol_is_not_required():
    """H2 carries 0.0003 kW below the pinch, under the 1e-3 kW tolerance.

    The side model places no recovery that small, so requiring the side
    would leave H2's load nowhere to go.
    """
    from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
        PinchDecompModel,
    )

    problem = _problem(
        [
            _stream("H1", 220.0, 150.0, 70.0),
            _stream("H2", 100.0, 70.0, 0.0003),
            _stream("C1", 90.0, 200.0, 110.0),
        ]
    )
    model = PinchDecompModel(
        name="below",
        framework="PDM",
        solver="highs",
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=0,
        minimisation_goal="cold utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="below",
        pinch_decomposition=decomposition_module.build_pinch_design_decomposition(
            problem, D_TMIN, pinch_location="below"
        ),
        stage_selection="automated",
    )

    assert any(model.z_i_active)
    assert model.side_required is False
