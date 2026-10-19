"""Minimum unit counts against N = S - 1 per pinch region."""

from __future__ import annotations

import warnings

import pytest

from OpenPinch import PinchProblem
from tests.analysis.test_area_bath_oracle import _random_payload
from tests.support.unit_count import minimum_units


def _segments(collection, prefix: str, *, by_segment: bool):
    view = (
        collection.segment_numeric_view() if by_segment else collection.numeric_view()
    )
    # Stream names repeat across zones (pulp_mill has two "Steam demand"
    # streams), so segments are grouped by their parent stream's position.
    owners = view.parent_index if by_segment else range(len(collection))
    return [
        (f"{prefix}{owner}", t_min, t_max, q)
        for owner, t_min, t_max, q in zip(
            owners, view.t_min_star, view.t_max_star, view.heat_flow
        )
    ]


def _units_and_oracle(problem: PinchProblem) -> tuple[float, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        target = problem.target.heat_exchanger_area_and_cost()
    hot = _segments(problem.hot_streams, "hot:", by_segment=True)
    hot += _segments(target.hot_utilities, "hot utility:", by_segment=False)
    cold = _segments(problem.cold_streams, "cold:", by_segment=True)
    cold += _segments(target.cold_utilities, "cold utility:", by_segment=False)
    return target.num_units, minimum_units(hot, cold)


@pytest.mark.parametrize(
    "case",
    [
        "Four-stream-Yee-and-Grossmann-1990-1.json",
        "basic_pinch.json",
        "chocolate_factory.json",
        "crude_preheat_train.json",
        "process_mvr.json",
        "pulp_mill.json",
        "zonal_site.json",
        "crude_preheat_train_multiperiod.json",
        "dairy_chilling_multiperiod.json",
        "zonal_site_multiperiod.json",
    ],
)
def test_sample_case_unit_count_is_streams_minus_one_per_pinch_region(case):
    # basic_pinch, crude_preheat_train and chocolate_factory reported 0 units:
    # residues of 2e-6 to 3.5e-5 kW at the balanced curve ends failed an
    # absolute 1e-6 kW zero test, so only the pinch counted as a boundary.
    units, oracle = _units_and_oracle(PinchProblem(case, project_name="Site"))

    assert units == oracle


@pytest.mark.parametrize("seed", range(20))
def test_random_case_unit_count_is_streams_minus_one_per_pinch_region(seed):
    units, oracle = _units_and_oracle(
        PinchProblem(_random_payload(seed), project_name="Site")
    )

    assert units == oracle
