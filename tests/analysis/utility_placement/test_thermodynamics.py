"""Thermodynamic cost kernel tests."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.utility_placement.allocation import (
    AllocatedUtilityLevel,
    PlacementPeriodAllocation,
)
from OpenPinch.analysis.utility_placement.context import (
    PlacementPeriodInput,
    PlacementTargetSnapshot,
    ProcessEntropySlice,
)
from OpenPinch.analysis.utility_placement.thermodynamics import (
    balanced_composite_entropy_generation,
    evaluate_thermodynamic_cost,
    stream_entropy_change,
)
from OpenPinch.contracts.utility_placement import (
    TemplateKey,
    UtilityLevelKind,
    UtilityPlacementRequest,
    UtilitySide,
)


def _level(side: UtilitySide, supply: float, target: float, duty: float):
    return AllocatedUtilityLevel(
        template_key=TemplateKey(side=side, name=f"{side.value}_1"),
        kind=UtilityLevelKind.SENSIBLE,
        placement_rank=0,
        supply_temperature=supply,
        target_temperature=target,
        allocated_duty=duty,
    )


def test_stream_entropy_change_matches_sensible_and_isothermal_limits() -> None:
    expected = 2.0 * math.log(400.0 / 500.0)
    assert stream_entropy_change(200.0, 500.0, 400.0) == pytest.approx(expected)
    assert stream_entropy_change(100.0, 350.0, 350.0) == pytest.approx(100.0 / 350.0)
    assert stream_entropy_change(0.0, 350.0, 350.0) == 0.0
    near_isothermal_cooling = stream_entropy_change(
        100.0,
        500.0,
        math.nextafter(500.0, 0.0),
    )
    assert near_isothermal_cooling == pytest.approx(-100.0 / 500.0)


def test_balanced_composite_entropy_matches_sensible_logarithmic_oracle() -> None:
    hot, cold, generation = balanced_composite_entropy_generation(
        hot_temperatures_celsius=(226.85, 126.85),
        hot_heat_loads=(100.0, 0.0),
        cold_temperatures_celsius=(76.85, 26.85),
        cold_heat_loads=(100.0, 0.0),
    )

    expected_hot = -math.log(500.0 / 400.0)
    expected_cold = 2.0 * math.log(350.0 / 300.0)
    assert hot == pytest.approx(expected_hot)
    assert cold == pytest.approx(expected_cold)
    assert generation == pytest.approx(expected_hot + expected_cold)


def test_balanced_composite_entropy_uses_isothermal_q_over_t_limit() -> None:
    hot, cold, generation = balanced_composite_entropy_generation(
        hot_temperatures_celsius=(226.85, 226.85),
        hot_heat_loads=(100.0, 0.0),
        cold_temperatures_celsius=(76.85, 26.85),
        cold_heat_loads=(100.0, 0.0),
    )

    assert hot == pytest.approx(-100.0 / 500.0)
    assert cold == pytest.approx(2.0 * math.log(350.0 / 300.0))
    assert generation == pytest.approx(hot + cold)


@given(
    duty=st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False),
    shift=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
)
def test_balanced_composite_entropy_prefers_closer_hot_curve(duty, shift) -> None:
    common = {
        "hot_heat_loads": (duty, 0.0),
        "cold_temperatures_celsius": (76.85, 26.85),
        "cold_heat_loads": (duty, 0.0),
    }
    closer = balanced_composite_entropy_generation(
        hot_temperatures_celsius=(126.85, 76.85),
        **common,
    )[2]
    farther = balanced_composite_entropy_generation(
        hot_temperatures_celsius=(126.85 + shift, 76.85 + shift),
        **common,
    )[2]

    assert 0.0 <= closer <= farther


@given(
    duty=st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False),
    factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
def test_balanced_composite_entropy_scales_with_heat_load(duty, factor) -> None:
    def generation(load: float) -> float:
        return balanced_composite_entropy_generation(
            hot_temperatures_celsius=(226.85, 126.85),
            hot_heat_loads=(load, 0.0),
            cold_temperatures_celsius=(76.85, 26.85),
            cold_heat_loads=(load, 0.0),
        )[2]

    assert generation(duty * factor) == pytest.approx(
        generation(duty) * factor,
        rel=2e-6,
        abs=2e-6,
    )


def test_balanced_composite_entropy_rejects_unbalanced_curves() -> None:
    with pytest.raises(Exception, match="balanced"):
        balanced_composite_entropy_generation(
            hot_temperatures_celsius=(200.0, 100.0),
            hot_heat_loads=(100.0, 0.0),
            cold_temperatures_celsius=(100.0, 50.0),
            cold_heat_loads=(90.0, 0.0),
        )


@pytest.mark.parametrize(
    ("hot_temperatures", "hot_heat", "cold_temperatures", "cold_heat", "match"),
    [
        ((), (), (), (), "cannot be empty"),
        ((200.0, 100.0), (float("nan"), 0.0), (100.0, 50.0), (1.0, 0.0), "finite"),
        (((200.0, 100.0),), ((1.0, 0.0),), (100.0, 50.0), (1.0, 0.0), "one-dimensional"),
    ],
)
def test_balanced_composite_entropy_rejects_malformed_curves(
    hot_temperatures,
    hot_heat,
    cold_temperatures,
    cold_heat,
    match,
) -> None:
    with pytest.raises(Exception, match=match):
        balanced_composite_entropy_generation(
            hot_temperatures_celsius=hot_temperatures,
            hot_heat_loads=hot_heat,
            cold_temperatures_celsius=cold_temperatures,
            cold_heat_loads=cold_heat,
        )


def test_balanced_composite_entropy_closes_only_numerical_heat_balance_drift() -> None:
    hot, cold, total = balanced_composite_entropy_generation(
        hot_temperatures_celsius=(226.85, 126.85),
        hot_heat_loads=(100.0, 0.0),
        cold_temperatures_celsius=(76.85, 26.85),
        cold_heat_loads=(100.0000005, 0.0),
    )

    assert hot < 0.0
    assert cold > 0.0
    assert total == pytest.approx(hot + cold)


def test_balanced_composite_entropy_handles_zero_duty_curves() -> None:
    assert balanced_composite_entropy_generation(
        hot_temperatures_celsius=(200.0, 100.0),
        hot_heat_loads=(0.0, 0.0),
        cold_temperatures_celsius=(100.0, 50.0),
        cold_heat_loads=(0.0, 0.0),
    ) == (0.0, 0.0, 0.0)


def test_balanced_composite_entropy_rejects_negative_generation() -> None:
    with pytest.raises(Exception, match="negative entropy generation"):
        balanced_composite_entropy_generation(
            hot_temperatures_celsius=(100.0, 50.0),
            hot_heat_loads=(100.0, 0.0),
            cold_temperatures_celsius=(200.0, 150.0),
            cold_heat_loads=(100.0, 0.0),
        )


def test_period_thermodynamic_breakdown_uses_balanced_composite_curve() -> None:
    period = PlacementPeriodInput(
        period_id="p1",
        weight=1.0,
        snapshot=PlacementTargetSnapshot(
            shifted_temperatures=(226.85, 126.85, 76.85, 26.85),
            real_temperatures=(226.85, 126.85, 76.85, 26.85),
            hot_load_profile=(100.0, 0.0, 0.0, 0.0),
            cold_load_profile=(0.0, 0.0, 0.0, 0.0),
            real_hot_composite=(0.0, 0.0, 0.0, 0.0),
            real_cold_composite=(100.0, 100.0, 100.0, 0.0),
            hot_pinch_index=1,
            cold_pinch_index=0,
            entropy_slices=(
                ProcessEntropySlice(
                    interval_index=0,
                    side=UtilitySide.COLD,
                    temperature_in_kelvin=300.0,
                    temperature_out_kelvin=350.0,
                    available_duty=100.0,
                    heat_capacity_flow=2.0,
                ),
                ProcessEntropySlice(
                    interval_index=1,
                    side=UtilitySide.HOT,
                    temperature_in_kelvin=500.0,
                    temperature_out_kelvin=500.0,
                    available_duty=50.0,
                    heat_capacity_flow=0.0,
                ),
                ProcessEntropySlice(
                    interval_index=2,
                    side=UtilitySide.COLD,
                    temperature_in_kelvin=300.0,
                    temperature_out_kelvin=300.0,
                    available_duty=30.0,
                    heat_capacity_flow=0.0,
                ),
            ),
        ),
        residual_hot_duty=100.0,
        residual_cold_duty=0.0,
        ambient_temperature_kelvin=300.0,
        coordinate_bounds=(),
    )
    allocation = PlacementPeriodAllocation(
        period_id="p1",
        hot_levels=(_level(UtilitySide.HOT, 226.85, 126.85, 100.0),),
        cold_levels=(),
        allocated_hot_duty=100.0,
        allocated_cold_duty=0.0,
        hot_coverage_residual=0.0,
        cold_coverage_residual=0.0,
        coverage_tolerance_hot=1e-6,
        coverage_tolerance_cold=1e-6,
        feasible=True,
    )

    result = evaluate_thermodynamic_cost(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=period,
        allocation=allocation,
    )

    expected = 2.0 * math.log(350.0 / 300.0) - math.log(500.0 / 400.0)
    assert result.total_entropy_generation.value == pytest.approx(expected)
    assert result.process_entropy.value == pytest.approx(
        2.0 * math.log(350.0 / 300.0)
    )
    assert result.utility_entropy.value == pytest.approx(-math.log(500.0 / 400.0))
    assert result.utility_entropy.value + result.process_entropy.value == (
        pytest.approx(result.total_entropy_generation.value)
    )
    assert result.exergy_destruction.value == pytest.approx(300.0 * expected)


def test_thermodynamic_cost_rejects_nonpositive_kelvin() -> None:
    with pytest.raises(Exception, match="kelvin"):
        stream_entropy_change(10.0, 0.0, 300.0)


def test_stream_entropy_change_rejects_nonfinite_and_negative_duty() -> None:
    with pytest.raises(Exception, match="finite"):
        stream_entropy_change(float("nan"), 300.0, 350.0)
    with pytest.raises(Exception, match="non-negative"):
        stream_entropy_change(-1.0, 300.0, 350.0)
