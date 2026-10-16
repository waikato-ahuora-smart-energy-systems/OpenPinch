"""The per-utility table joining placed and retargeted utility duties."""

import pytest

from OpenPinch import PinchWorkspace
from OpenPinch.contracts.utility_placement import UtilityPlacementResult

_OPTIONS = {
    "iteration_limit": 3,
    "evaluation_limit": 6,
    "candidate_limit": 1,
    "run_count": 1,
    "seed": 20260715,
}


@pytest.fixture(scope="module")
def placed():
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    return problem.target.utility_placement(
        isothermal=4, sensible=0, period_ids=("0",), options=_OPTIONS
    )


def test_retarget_comparison_has_one_row_per_placed_level(placed):
    result = placed.utility_placement_result
    (period,) = result.best.period_results

    rows = result.retarget_comparison

    assert [(row.period_id, row.side.value, row.utility) for row in rows] == [
        ("0", side, level.template_key.name)
        for side, levels in (("hot", period.hot_levels), ("cold", period.cold_levels))
        for level in levels
    ]
    levels = {
        level.template_key.name: level
        for level in period.hot_levels + period.cold_levels
    }
    for row in rows:
        level = levels[row.utility]
        assert row.optimizer_duty == level.allocated_duty
        assert row.kind == level.kind
        assert row.placement_rank == level.placement_rank
        assert row.supply_temperature == level.supply_temperature
        assert row.target_temperature == level.target_temperature
        assert row.is_fallback == level.is_fallback


def test_retarget_comparison_carries_the_retargeted_utility_duties(placed):
    target = placed.results.targets[-1]
    retargeted = {
        utility.name: utility.heat_flow
        for utility in target.hot_utilities + target.cold_utilities
    }

    rows = placed.utility_placement_result.retarget_comparison

    for row in rows:
        assert row.retargeted_duty.unit == retargeted[row.utility].unit
        assert row.retargeted_duty.value == pytest.approx(
            retargeted[row.utility].value, rel=1e-5, abs=1e-4
        )


def test_zone_placement_rows_cover_the_zone_levels():
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    placed = problem.target.utility_placement(
        zone="Almond", isothermal=4, sensible=0, period_ids=("0",), options=_OPTIONS
    )
    result = placed.utility_placement_result
    (period,) = result.best.period_results

    assert {row.utility for row in result.retarget_comparison} == {
        level.template_key.name for level in period.hot_levels + period.cold_levels
    }


def test_a_result_recorded_without_the_table_still_validates(placed):
    payload = placed.utility_placement_result.model_dump(mode="json")
    rows = payload.pop("retarget_comparison")

    restored = UtilityPlacementResult.model_validate(payload)

    assert rows
    assert restored.retarget_comparison == ()
