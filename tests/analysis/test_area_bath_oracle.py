"""Area targets against an independent fine-slice Bath-formula integration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from OpenPinch import PinchProblem
from tests.support.bath_area import bath_area

# The oracle's midpoint rule on 400k slices is converged to about 1e-7
# relative on these cases; 1e-5 leaves room for the problem table's 6 dp
# temperature grid without hiding a percent-level driving-force error.
BATH_RTOL = 1e-5


def _segments(collection, *, by_segment: bool):
    view = (
        collection.segment_numeric_view() if by_segment else collection.numeric_view()
    )
    return list(zip(view.t_min, view.t_max, view.heat_flow, view.cp / view.rcp))


def _area_and_oracle(problem: PinchProblem) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        target = problem.target.heat_exchanger_area_and_cost()
    hot = _segments(problem.hot_streams, by_segment=True)
    hot += _segments(target.hot_utilities, by_segment=False)
    cold = _segments(problem.cold_streams, by_segment=True)
    cold += _segments(target.cold_utilities, by_segment=False)
    return target.area, bath_area(hot, cold)


def test_yee_and_grossmann_area_matches_the_bath_integration():
    # The cold composite steps between the process streams and cooling water,
    # where the old discontinuity rule cut a slice's LMTD from 91.8 to 37 K
    # and reported 341.08 m2.
    area, oracle = _area_and_oracle(
        PinchProblem("Four-stream-Yee-and-Grossmann-1990-1.json", project_name="Site")
    )

    assert oracle == pytest.approx(337.20, abs=5e-3)
    assert area == pytest.approx(oracle, rel=BATH_RTOL)


@pytest.mark.parametrize(
    "case",
    [
        "basic_pinch.json",
        "chocolate_factory.json",
        "crude_preheat_train.json",
        "process_mvr.json",
        "pulp_mill.json",
    ],
)
def test_sample_case_area_matches_the_bath_integration(case):
    area, oracle = _area_and_oracle(PinchProblem(case, project_name="Site"))

    assert area == pytest.approx(oracle, rel=BATH_RTOL)


def _random_payload(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    streams = []
    for kind, count in (("H", rng.integers(1, 5)), ("C", rng.integers(1, 5))):
        for n in range(count):
            t_low, t_high = np.sort(rng.uniform(20.0, 300.0, size=2).round(1))
            t_high = max(t_high, t_low + 5.0)
            supply, target = (t_high, t_low) if kind == "H" else (t_low, t_high)
            streams.append(
                {
                    "zone": "Site",
                    "name": f"{kind}{n}",
                    "t_supply": float(supply),
                    "t_target": float(target),
                    "heat_flow": round(float(rng.uniform(50.0, 2000.0)), 1),
                    "htc": round(float(rng.uniform(0.2, 2.0)), 2),
                }
            )
    return {"streams": streams, "utilities": [], "options": {}}


@pytest.mark.parametrize("seed", range(20))
def test_random_case_area_matches_the_bath_integration(seed):
    area, oracle = _area_and_oracle(
        PinchProblem(_random_payload(seed), project_name="Site")
    )

    assert area == pytest.approx(oracle, rel=BATH_RTOL)
