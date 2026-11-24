"""A match that can run in one period but not another.

In a multiperiod stage-wise model a match is allowed when it can carry duty
in any period. In a period where its supply temperatures are closer than
its approach it cannot run: there its duty is zero and its approach
temperatures are not variables, so their bounds never cross and the model
is not infeasible before any match is chosen.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.application.problem import PinchProblem
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    _highs_available,
)

pytestmark = pytest.mark.skipif(not _highs_available(), reason="needs pyomo")

D_TMIN = 10.0


def _values(*values, unit="degC"):
    return {"values": list(values), "unit": unit}


def _problem() -> PinchProblem:
    """H1 enters at 200 C in "on" but 55 C in "off"; C1 enters at 50 C."""
    stream = {
        "zone": "Plant",
        "dt_cont": {"value": D_TMIN / 2, "unit": "delta_degC"},
        "htc": {"value": 1.0, "unit": "kW/m^2/delta_degC"},
    }
    return PinchProblem(
        source={
            "streams": [
                stream
                | {
                    "name": "H1",
                    "t_supply": _values(200.0, 55.0),
                    "t_target": _values(50.0, 45.0),
                    "heat_flow": _values(150.0, 10.0, unit="kW"),
                },
                stream
                | {
                    "name": "C1",
                    "t_supply": _values(50.0, 50.0),
                    "t_target": _values(150.0, 150.0),
                    "heat_flow": _values(100.0, 100.0, unit="kW"),
                },
            ],
            "utilities": [
                {
                    "name": name,
                    "type": kind,
                    "t_supply": {"value": supply, "unit": "degC"},
                    "t_target": {"value": target, "unit": "degC"},
                    "dt_cont": {"value": 2.5, "unit": "delta_degC"},
                    "price": {"value": price, "unit": "$/MWh"},
                    "htc": {"value": 1.0, "unit": "kW/m^2/delta_degC"},
                    "heat_flow": None,
                }
                for name, kind, supply, target, price in (
                    ("HPS", "Hot", 250.0, 249.9, 40.0),
                    ("CW", "Cold", 10.0, 15.0, 5.0),
                )
            ],
            "options": {
                "PROBLEM_PERIOD_IDS": ["on", "off"],
                "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
            },
        }
    )


def _model(framework: str = "ESM", min_dqda: float = 0.0) -> StageWiseModel:
    return StageWiseModel(
        name="period-pairs",
        framework=framework,
        solver="ipopt-pyomo",
        solver_arrays=problem_to_solver_arrays(_problem(), D_TMIN),
        stages=2,
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=min_dqda,
        minimisation_goal="total cost",
        non_isothermal_model=True,
        integers=False,
        tol=1e-6,
    )


def _bounds(item):
    component = getattr(item, "component", item)
    return getattr(component, "lb", None), getattr(component, "ub", None)


def test_a_match_that_cannot_run_in_a_period_has_no_crossed_approach_bounds():
    model = _model()

    assert model.N_periods == 2
    for n in range(model.N_periods):
        for grid in (model.theta_1_by_period, model.theta_2_by_period):
            for i in range(model.I):
                for j in range(model.J):
                    for k in range(model.S):
                        lower, upper = _bounds(grid[n][i][j][k])
                        if lower is not None and upper is not None:
                            assert lower <= upper, (n, i, j, k, lower, upper)


def test_a_match_that_cannot_run_in_a_period_has_no_dqda_constraint_there():
    """The topology binary is shared over periods, so a dQ/dA constraint in
    a period where the match cannot run would bind once another period
    chooses it."""
    from OpenPinch.analysis.heat_exchanger_networks.models._stagewise.equations import (
        _runs,
    )

    model = _model(framework="TDM", min_dqda=0.5)

    cells = [
        (n, i, j, k)
        for n in range(model.N_periods)
        for i in range(model.I)
        for j in range(model.J)
        for k in range(model.S)
        if model.z_allowed[i][j][k] > 0
    ]
    idle = [cell for cell in cells if not _runs(model, *cell)]
    assert idle and len(idle) < len(cells)
    for n, i, j, k in cells:
        assert (model.min_dqda_int[n][i][j][k] is None) == (
            not _runs(model, n, i, j, k)
        ), (n, i, j, k)
    equations = [item for item in model.min_dQ_dA_eqn if item is not None]
    assert len(equations) == len(cells) - len(idle)


def test_a_multiperiod_warm_start_leaves_a_disallowed_utility_off():
    """Evolution can remove a heater; warm-started from a parent that used
    it, the child keeps it off rather than paying its unit cost at no
    duty."""
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.models._base.execution import (
        _value,
    )

    child = _model()
    hpsu = 0
    (c1,) = range(child.J)
    child.z_hu_allowed[c1][hpsu] = 0
    source = SimpleNamespace(**vars(_model()))
    source.z_hu = [[1 for _ in row] for row in source.z_hu_allowed]

    child.set_initial_values_for_variables(source)

    assert _value(child.z_hu[c1][hpsu]) == 0
