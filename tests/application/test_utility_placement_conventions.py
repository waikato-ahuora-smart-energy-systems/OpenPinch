"""Utility placement conventions: bounds, ambient, entropy basis and outputs."""

import math

import pytest

from OpenPinch import PinchProblem, PinchWorkspace

_OPTIONS = {
    "iteration_limit": 3,
    "evaluation_limit": 6,
    "candidate_limit": 1,
    "run_count": 1,
    "seed": 20260715,
}


@pytest.mark.parametrize(("isothermal", "sensible"), [(2, 1), (3, 2), (2, 3)])
def test_yee_grossmann_places_with_sensible_levels(isothermal, sensible):
    # HP steam at 680 K and cooling water at 300 K made the span bound reach
    # below absolute zero from the lowest supply temperature.
    problem = PinchProblem("Four-stream-Yee-and-Grossmann-1990-1.json")

    placed = problem.target.utility_placement(
        isothermal=isothermal,
        sensible=sensible,
        period_ids=("0",),
        options=_OPTIONS,
    )

    (period,) = placed.utility_placement_result.best.period_results
    assert [level.kind.value for level in period.hot_levels].count(
        "sensible"
    ) == sensible
    for level in period.hot_levels + period.cold_levels:
        assert level.target_temperature.unit == "degC"
        assert level.target_temperature.value > -273.15


@pytest.fixture(scope="module")
def placed_basic_pinch():
    problem = PinchProblem("basic_pinch.json")
    problem.update_options({"ENV_TEMPERATURE": 30.0})
    return problem.target.utility_placement(
        isothermal=2, sensible=1, period_ids=("0",), options=_OPTIONS
    )


def test_placement_exergy_uses_the_problem_ambient(placed_basic_pinch):
    (period,) = placed_basic_pinch.utility_placement_result.best.period_results
    thermodynamic = period.thermodynamic

    assert thermodynamic.ambient_temperature.value == pytest.approx(303.15)
    assert thermodynamic.exergy_destruction.value == pytest.approx(
        303.15 * thermodynamic.total_entropy_generation.value
    )


def _level_entropy_change(level) -> float:
    duty = level.allocated_duty.value
    inlet = level.supply_temperature.value + 273.15
    outlet = level.target_temperature.value + 273.15
    # Cooling (hot utility) lowers its entropy, heating raises it.
    return duty / abs(outlet - inlet) * math.log(outlet / inlet)


def test_placement_entropy_breakdown_uses_one_basis(placed_basic_pinch):
    (period,) = placed_basic_pinch.utility_placement_result.best.period_results
    thermodynamic = period.thermodynamic
    levels = [
        level
        for level in period.hot_levels + period.cold_levels
        if level.allocated_duty.value > 0.0
    ]

    utility_entropy = math.fsum(_level_entropy_change(level) for level in levels)

    assert thermodynamic.utility_entropy.value == pytest.approx(
        utility_entropy, rel=1e-6
    )
    assert (
        thermodynamic.process_entropy.value + thermodynamic.utility_entropy.value
    ) == pytest.approx(thermodynamic.total_entropy_generation.value, rel=1e-9)


def test_placement_local_method_is_a_closed_choice():
    from OpenPinch.contracts.utility_placement import (
        UtilityPlacementLocalMethod,
        UtilityPlacementOptions,
    )
    from OpenPinch.optimisation.models import LOCAL_METHODS

    assert tuple(method.value for method in UtilityPlacementLocalMethod) == (
        LOCAL_METHODS
    )
    assert UtilityPlacementOptions(local_method="Powell").local_method == "Powell"
    with pytest.raises(ValueError, match="local_method"):
        UtilityPlacementOptions(local_method="BFGS")


def test_direct_case_keeps_the_nearest_original_utility_properties():
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    original = problem.to_problem_json()["utilities"]

    # A process zone places against its Direct Integration target.
    placed = problem.target.utility_placement(
        isothermal=2, sensible=2, zone="Almond", period_ids=("0",), options=_OPTIONS
    )

    (period,) = placed.utility_placement_result.best.period_results
    levels = {
        level.template_key.name: level
        for level in period.hot_levels + period.cold_levels
    }
    for utility in placed.to_problem_json()["utilities"]:
        level = levels[utility["name"]]
        source = min(
            (item for item in original if item["type"] in (utility["type"], "Both")),
            key=lambda item: (
                item["name"] != utility["name"],
                abs(item["t_supply"]["value"] - level.supply_temperature.value),
            ),
        )
        assert utility["price"] == source["price"]
        assert utility["htc"] == source["htc"]
        assert utility["dt_cont"] == source["dt_cont"]
        # The placed temperatures are shifted; the real ones sit dt_cont
        # outward so the case targets the same shifted levels.
        offset = source["dt_cont"]["value"] * (
            1.0 if utility["type"] == "Hot" else -1.0
        )
        assert utility["t_supply"]["value"] == pytest.approx(
            level.supply_temperature.value + offset
        )
        assert utility["t_target"]["value"] == pytest.approx(
            level.target_temperature.value + offset
        )


def test_total_site_case_keeps_price_and_htc_with_unshifted_levels():
    problem = PinchProblem("basic_pinch.json")
    original = {
        utility["type"]: utility for utility in problem.to_problem_json()["utilities"]
    }

    placed = problem.target.utility_placement(
        isothermal=2, sensible=1, period_ids=("0",), options=_OPTIONS
    )

    (period,) = placed.utility_placement_result.best.period_results
    levels = {
        level.template_key.name: level
        for level in period.hot_levels + period.cold_levels
    }
    for utility in placed.to_problem_json()["utilities"]:
        source = original[utility["type"]]
        assert utility["price"] == source["price"]
        assert utility["htc"] == source["htc"]
        # Total Site targets also depend on each subzone's utility shift, so
        # the placed levels stay unshifted.
        assert utility["dt_cont"]["value"] == 0.0
        level = levels[utility["name"]]
        assert utility["t_supply"]["value"] == pytest.approx(
            level.supply_temperature.value
        )


def test_period_varying_dt_cont_keeps_the_placed_level_unshifted():
    from types import SimpleNamespace

    from OpenPinch.application.utility_placement import _optimized_utility
    from OpenPinch.contracts.utility_placement import (
        QuantityValue,
        UtilityPlacementBaseTarget,
    )

    level = SimpleNamespace(
        template_key=SimpleNamespace(name="hot_iso_1"),
        supply_temperature=QuantityValue(value=150.0, unit="degC"),
        target_temperature=QuantityValue(value=149.99, unit="degC"),
    )
    result = SimpleNamespace(
        request=SimpleNamespace(
            base_target=UtilityPlacementBaseTarget.DIRECT, maximum_duties=()
        ),
        units=SimpleNamespace(heat_flow="kW", temperature_difference="delta_degC"),
    )
    raw = {
        "dt_cont": {"values": [1.0, 3.0], "period_ids": ["a", "b"], "unit": "degC"},
        "price": {"value": 10.0, "unit": "$/MWh"},
        "htc": {"value": 2.0, "unit": "kW/m^2/degC"},
    }

    utility = _optimized_utility(level, "Hot", ("HU", 200.0, 1.0, raw), result)

    # One temperature cannot carry a shift that differs by period.
    assert utility["dt_cont"] == {"value": 0.0, "unit": "delta_degC"}
    assert utility["t_supply"]["value"] == pytest.approx(150.0)
    assert utility["price"] == raw["price"]
    assert utility["htc"] == raw["htc"]
