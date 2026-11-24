"""The stage-wise model without integers keeps the approach inside a match.

Network evolution solves stage-wise models without integers, so the lazy
cuts that pinch design and the thermal derivative step use (they need
binaries) do not apply. On a condensing stream V that desuperheats and
then condenses, matched in one stage against a cold stream C, the model
held the approach at the match's ends alone: it took all of C's 150 kW
and heated C past V's plateau, and taking the match's areas then failed
("Segmented exchanger slice has a non-positive terminal temperature
difference"). Evolution drops such a candidate rather than serving the
match at a smaller duty. Couenne solves the model here, since Ipopt does
not settle the segment intervals on this case from a cold start.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)


def _couenne_available() -> bool:
    try:
        backend.require_solver_backend("couenne")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _degc(value: float) -> dict:
    return {"value": value, "unit": "degC"}


def _kink_problem() -> PinchProblem:
    contribution = {"value": 1.0, "unit": "delta_degC"}
    return PinchProblem(
        source={
            "streams": [
                {
                    "zone": "Plant",
                    "name": "V",
                    "dt_cont": contribution,
                    "segments": [
                        {"t_supply": 224.75, "t_target": 196.3, "heat_flow": 5.7},
                        {"t_supply": 196.3, "t_target": 174.76, "heat_flow": 4.67},
                        {"t_supply": 174.76, "t_target": 174.75, "heat_flow": 177.1},
                    ],
                },
                {
                    "zone": "Plant",
                    "name": "C",
                    "t_supply": _degc(160.0),
                    "t_target": _degc(190.0),
                    "heat_flow": {"value": 150.0, "unit": "kW"},
                    "dt_cont": contribution,
                },
            ],
            "utilities": [
                {
                    "name": "HU",
                    "type": "Hot",
                    "t_supply": _degc(260.0),
                    "t_target": _degc(259.0),
                    "dt_cont": contribution,
                    "price": {"value": 50.0, "unit": "$/MWh"},
                    "heat_flow": None,
                },
                {
                    "name": "CU",
                    "type": "Cold",
                    "t_supply": _degc(20.0),
                    "t_target": _degc(25.0),
                    "dt_cont": contribution,
                    "price": {"value": 5.0, "unit": "$/MWh"},
                    "heat_flow": None,
                },
            ],
            # At the reference contribution of 1 K, the arrays' approach of
            # 2 K keeps each stream's 1 K: the pair is held to 2 K.
            "options": {"THERMAL_DT_CONT": 1.0},
        },
        project_name="kink-nlp",
    )


@pytest.mark.solver
@pytest.mark.skipif("not _couenne_available()", reason="needs couenne")
def test_a_model_without_integers_holds_the_approach_at_the_kink():
    model = StageWiseModel(
        name="kink-nlp",
        framework="ESM",
        solver="couenne",
        solver_arrays=problem_to_solver_arrays(_kink_problem(), 2.0),
        stages=1,
        dTmin=2.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=False,
        tol=1e-6,
    )

    model.optimise(False)

    # Each segment pair the match spans keeps the pair's 2 K at both ends.
    slices = model.segment_area_contributions_by_period[0][0][0][0]
    assert sum(item.duty for item in slices) > 0.0
    for item in slices:
        assert item.hot_inlet_temperature - item.cold_outlet_temperature >= 2.0 - 1e-3
        assert item.hot_outlet_temperature - item.cold_inlet_temperature >= 2.0 - 1e-3
