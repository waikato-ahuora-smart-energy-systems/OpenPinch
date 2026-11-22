"""The minimum approach temperature of one match, from its two contributions.

Pinch design holds a match to the sum of its source and sink contributions,
each already scaled by the task's dt_cont multiplier. The thermal derivative
and evolution steps hold a share of that sum. Either way a match is never
held to less than 0.1 K: that floor is on the pair, so a stream with a zero
contribution adds nothing to a partner that already supplies an approach.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._base.approach import (
    _cold_utility_inlet_approach_temperature,
    _cold_utility_outlet_approach_temperature,
    _hot_utility_inlet_approach_temperature,
    _hot_utility_outlet_approach_temperature,
    _set_minimum_approach_temperatures,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    MINIMUM_PAIR_APPROACH,
    pair_minimum_approach,
)


@pytest.mark.parametrize(
    ("source", "sink", "expected"),
    [
        (5.0, 5.0, 10.0),
        (5.0, 1.0, 6.0),  # a process stream on a utility
        (5.0, 0.0, 5.0),  # a zero contribution adds nothing
        (0.03, 0.02, 0.1),  # a sum below the floor
        (0.0, 0.0, 0.1),
    ],
)
def test_pinch_design_holds_the_sum_of_the_contributions(source, sink, expected):
    assert pair_minimum_approach(source, sink) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("source", "sink", "expected"),
    [
        (5.0, 5.0, 2.5),
        (5.0, 1.0, 1.5),
        (5.0, 0.0, 1.25),
        (0.03, 0.02, 0.1),
        (0.0, 0.0, 0.1),
    ],
)
def test_the_later_steps_hold_a_share_of_the_sum(source, sink, expected):
    assert pair_minimum_approach(source, sink, share=0.25) == pytest.approx(
        expected, abs=1e-12
    )


def test_the_multiplier_scales_the_contributions_and_not_the_floor():
    multiplier = 0.75
    assert pair_minimum_approach(5.0 * multiplier, 0.0 * multiplier) == pytest.approx(
        3.75, abs=1e-12
    )
    assert pair_minimum_approach(0.0 * multiplier, 0.0 * multiplier) == pytest.approx(
        MINIMUM_PAIR_APPROACH, abs=1e-12
    )


def test_the_floor_is_a_tenth_of_a_kelvin():
    assert MINIMUM_PAIR_APPROACH == pytest.approx(0.1, abs=1e-12)


def _model(*, hot, cold, hot_utilities, cold_utilities):
    """The contribution rows a model reads, one period, already scaled."""
    return SimpleNamespace(
        N_periods=1,
        dTmin=10.0,
        T_h_cont_period=np.array([hot], dtype=float),
        T_c_cont_period=np.array([cold], dtype=float),
        T_hu_cont_period=np.array([hot_utilities], dtype=float),
        T_cu_cont_period=np.array([cold_utilities], dtype=float),
    )


def test_a_models_approach_grids_floor_each_pair_and_not_each_stream():
    model = _model(
        hot=[5.0, 0.0], cold=[5.0, 0.0], hot_utilities=[1.0, 0.0], cold_utilities=[0.0]
    )

    _set_minimum_approach_temperatures(model)

    # [hot][cold]: only the pair of two zero contributions meets the floor.
    np.testing.assert_allclose(
        model.dT_r_period[0], [[10.0, 5.0], [5.0, 0.1]], rtol=0, atol=1e-12
    )
    # Heaters, [utility][cold].
    np.testing.assert_allclose(
        model.dT_hu_period[0], [[6.0, 1.0], [5.0, 0.1]], rtol=0, atol=1e-12
    )
    # Coolers, [hot][utility].
    np.testing.assert_allclose(
        model.dT_cu_period[0], [[5.0], [0.1]], rtol=0, atol=1e-12
    )
    assert model.dT_hu_min_period[0][1][1] == pytest.approx(0.1, abs=1e-12)
    assert model.dT_cu_min_period[0][1][0] == pytest.approx(0.1, abs=1e-12)


def test_a_heaters_and_a_coolers_end_approaches_floor_the_pair():
    model = _model(
        hot=[5.0, 0.0], cold=[5.0, 0.0], hot_utilities=[1.0, 0.0], cold_utilities=[0.0]
    )

    assert _hot_utility_inlet_approach_temperature(model, 0, 1, 0) == pytest.approx(
        5.0, abs=1e-12
    )
    assert _hot_utility_inlet_approach_temperature(model, 1, 1, 0) == pytest.approx(
        0.1, abs=1e-12
    )
    assert _hot_utility_outlet_approach_temperature(model, 1, 1, 0) == pytest.approx(
        0.1, abs=1e-12
    )
    assert _cold_utility_inlet_approach_temperature(model, 0, 0, 0) == pytest.approx(
        5.0, abs=1e-12
    )
    assert _cold_utility_inlet_approach_temperature(model, 1, 0, 0) == pytest.approx(
        0.1, abs=1e-12
    )
    assert _cold_utility_outlet_approach_temperature(model, 1, 0, 0) == pytest.approx(
        0.1, abs=1e-12
    )


def _segmented_heater_model(pyo, *, outlet_difference):
    """A heater on a two-segment hot utility whose outlet contribution is a
    variable of the model, as it is at a duty the solver chooses. The utility
    leaves ``outlet_difference`` K above the cold stream it heats; every
    contribution is 0."""
    from OpenPinch.analysis.heat_exchanger_networks.solver.pyomo_model import (
        PyomoModel,
    )

    m = PyomoModel()
    outlet = m.Var(value=300.0 + outlet_difference)
    cold_inlet = m.Var(value=300.0)
    arrays = {
        "hot_utility_parent_segmented": np.array([True]),
        "hot_utility_segment_count": np.array([2]),
        "hot_utility_segment_dt_cont_period": np.zeros((1, 1, 2)),
    }
    model = SimpleNamespace(
        m=m,
        tol=1e-6,
        dTmin=10.0,
        solver_arrays=SimpleNamespace(arrays=arrays),
        T_c_cont_period=np.array([[0.0]]),
        T_hu_cont_period=np.array([[0.0]]),
        T_hu_out_cont_by_period=[[[m.Var(value=0.0)]]],
        T_hu_in_period=np.array([[400.0]]),
        T_hu_out_period=np.array([[300.0]]),
        T_c_in_period=np.array([[300.0]]),
        T_c_out_period=np.array([[350.0]]),
        z_hu=[[m.Param(value=1)]],
        Q_h_by_period=[[[m.Var(value=5.0, lb=0.0, ub=10.0)]]],
        T_c_by_period=[[[cold_inlet]]],
        _utility_solved_outlet_temperature=lambda side, n, j, u: outlet,
    )
    return model


def _broken(pyo, model):
    def is_broken(constraint):
        body = pyo.value(constraint.body)
        return (
            constraint.upper is not None and body > pyo.value(constraint.upper) + 1e-9
        ) or (
            constraint.lower is not None and body < pyo.value(constraint.lower) - 1e-9
        )

    return [
        c for c in model.m.model.component_data_objects(pyo.Constraint) if is_broken(c)
    ]


def test_a_segmented_heaters_outlet_holds_the_pair_floor():
    """A segmented utility's outlet contribution is a variable, so the pair's
    sum cannot be floored as a number; the outlet end must still keep 0.1 K."""
    from OpenPinch.analysis.heat_exchanger_networks.models._base.approach import (
        _set_hot_utility_approach_equations,
    )

    pyo = pytest.importorskip("pyomo.environ")
    model = _segmented_heater_model(pyo, outlet_difference=0.05)

    _set_hot_utility_approach_equations(model, 0, 0, 0)

    assert _broken(pyo, model)


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _two_stream_problem(*, hot_dt_cont, cold_dt_cont):
    """H1 100 to 50 degC against C1 50 to 100 degC, 2 kW/K each, with
    utilities that contribute nothing."""
    from OpenPinch.application.problem import PinchProblem

    def stream(name, t_supply, t_target, dt_cont):
        return {
            "zone": "Site/A",
            "name": name,
            "t_supply": _quantity(t_supply, "degC"),
            "t_target": _quantity(t_target, "degC"),
            "heat_flow": _quantity(100.0, "kW"),
            "dt_cont": _quantity(dt_cont, "delta_degC"),
            "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        }

    def utility(name, kind, t_supply, t_target):
        return {
            "name": name,
            "type": kind,
            "t_supply": _quantity(t_supply, "degC"),
            "t_target": _quantity(t_target, "degC"),
            "dt_cont": _quantity(0.0, "delta_degC"),
            "price": _quantity(10.0, "$/MWh"),
            "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        }

    problem = PinchProblem(
        source={
            "streams": [
                stream("H1", 100.0, 50.0, hot_dt_cont),
                stream("C1", 50.0, 100.0, cold_dt_cont),
            ],
            "utilities": [
                utility("HPS", "Hot", 250.0, 249.0),
                utility("CW", "Cold", 10.0, 15.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "A", "type": "Process Zone"}],
            },
        }
    )
    problem.target.direct_heat_integration()
    return problem


def test_pinch_targets_hold_the_floor_for_a_pair_of_zero_contributions():
    """The models hold a pair of zero contributions 0.1 K apart, so the
    targets they must meet cannot assume the two streams meet at 0 K: at
    2 kW/K the 0.1 K costs 0.2 kW of each utility."""
    from OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
        _calculate_openpinch_targets,
    )

    problem = _two_stream_problem(hot_dt_cont=0.0, cold_dt_cont=0.0)

    (target,) = _calculate_openpinch_targets(problem, dTmin=10.0)

    assert target.hot_utility_target == pytest.approx(0.2, abs=1e-9)
    assert target.cold_utility_target == pytest.approx(0.2, abs=1e-9)
