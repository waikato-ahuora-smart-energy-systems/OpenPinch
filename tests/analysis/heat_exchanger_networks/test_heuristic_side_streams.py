"""Decision data the heuristic pinch design method reads from a pinch side."""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heuristic.side_streams import (
    pinch_side,
)
from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
    build_pinch_design_decomposition,
)
from OpenPinch.application.problem import PinchProblem
from tests.analysis.heat_exchanger_networks.test_segmented_streams import (
    _segmented_problem,
)
from tests.support.paths import REPOSITORY_ROOT

FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)

pytestmark = pytest.mark.synthesis


def _sides(pinch: PinchProblem, dTmin: float):
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(pinch, dTmin),
        framework="PDM",
        solver="highs",
        dTmin=dTmin,
        pinch_decompositions={
            side: build_pinch_design_decomposition(pinch, dTmin, pinch_location=side)
            for side in ("above", "below")
        },
    )
    problem.load_model()
    return pinch_side(problem.above), pinch_side(problem.below)


def _four_stream() -> PinchProblem:
    return PinchProblem(
        source=json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8")),
        project_name="Four-stream",
    )


def test_constant_cp_streams_carry_loads_pinch_cps_and_pinch_status():
    above, below = _sides(_four_stream(), 14.0)

    assert above.location == "above" and above.required
    assert [stream.index for stream in above.hot] == [0]
    assert [stream.index for stream in above.cold] == [0]
    assert above.hot[0].load == pytest.approx(600.0)
    assert above.hot[0].pinch_cp == pytest.approx(10.0)
    assert above.hot[0].touches_pinch
    assert above.cold[0].load == pytest.approx(1110.0)
    assert above.cold[0].pinch_cp == pytest.approx(15.0)
    assert above.heater_allowed[0] == (1,)

    assert [stream.index for stream in below.hot] == [0, 1]
    assert [stream.index for stream in below.cold] == [0, 1]
    assert [stream.touches_pinch for stream in below.cold] == [True, False]
    assert [stream.pinch_cp for stream in below.hot] == pytest.approx([10.0, 20.0])
    assert below.cold[1].load == pytest.approx(1950.0)
    assert below.cooler_allowed[1] == (1,)


def test_stream_temperature_is_read_from_its_pinch_end():
    above, below = _sides(_four_stream(), 14.0)

    # Below the pinch a hot stream cools from its inlet at the pinch.
    assert below.hot[1].temperature(0, 0.0) == pytest.approx(590.0)
    assert below.hot[1].temperature(0, 2490.0) == pytest.approx(465.5)
    # Above the pinch a cold stream heats from its inlet at the pinch.
    assert above.cold[0].temperature(0, 600.0) == pytest.approx(616.0)


def test_segmented_streams_take_the_pinch_end_segment_cp():
    above, below = _sides(_segmented_problem(cold_second_duty=200.0), 10.0)

    assert not below.hot and not below.cold
    hot, cold = above.hot[0], above.cold[0]
    assert hot.load == pytest.approx(150.0)
    assert not hot.touches_pinch
    # The hot stream's pinch end is its outlet, on the 2 kW/K segment.
    assert hot.pinch_cp == pytest.approx(2.0)
    assert hot.temperature(0, 100.0) == pytest.approx(423.15)
    assert cold.touches_pinch
    assert cold.pinch_cp == pytest.approx(1.0)
    assert cold.temperature(0, 50.0) == pytest.approx(373.15)
    assert cold.temperature(0, 150.0) == pytest.approx(398.15)


def test_multiperiod_decisions_use_period_weighted_means_over_active_periods():
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        nominal_approach_temperature,
    )

    pinch = PinchProblem("crude_preheat_train_multiperiod.json")
    pinch.target.all_heat_integration()
    dTmin = nominal_approach_temperature(pinch.master_zone.config, 1.0)

    above, _below = _sides(pinch, dTmin)

    assert above.periods == 3
    weights = above.weights
    # A hot stream absent above the pinch in the first period.
    stream = next(stream for stream in above.hot if stream.loads[0] == 0.0)
    assert stream.at_pinch[0] is False and stream.touches_pinch
    assert stream.load == pytest.approx(
        sum(w * load for w, load in zip(weights, stream.loads)) / sum(weights)
    )
    active = [n for n in range(3) if stream.loads[n] > 0.0]
    assert stream.pinch_cp == pytest.approx(
        sum(weights[n] * stream.pinch_cps[n] for n in active)
        / sum(weights[n] for n in active)
    )


def test_segmented_cp_at_a_match_end_follows_the_segment_there():
    above, _below = _sides(_segmented_problem(cold_second_duty=200.0), 10.0)
    hot = above.hot[0]

    # From the outlet: 100 kW on the 2 kW/K segment, then 50 kW on 1 kW/K.
    assert hot.cp_at(50.0) == pytest.approx(2.0)
    assert hot.cp_at(120.0) == pytest.approx(1.0)
