"""Candidate-local utility allocation and coverage tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from OpenPinch.analysis.utility_placement.allocation import (
    AllocationAdapterResult,
    allocate_placement_period,
)
from OpenPinch.analysis.utility_placement.context import (
    PlacementPeriodInput,
    PlacementTargetSnapshot,
)
from OpenPinch.contracts.utility_placement import (
    CandidateDiagnostic,
    DecodedPlacement,
    DecodedUtilityLevel,
    QuantityValue,
    TemplateKey,
    UtilityLevelKind,
    UtilityPlacementRequest,
    UtilitySide,
)


def _level(name: str, side: UtilitySide, supply: float, target: float, rank: int):
    return DecodedUtilityLevel(
        template_key=TemplateKey(side=side, name=name),
        kind=UtilityLevelKind.ISOTHERMAL,
        placement_rank=rank,
        supply_temperature=QuantityValue(value=supply, unit="degC"),
        target_temperature=QuantityValue(value=target, unit="degC"),
        temperature_span=QuantityValue(value=abs(supply - target), unit="delta_degC"),
    )


def _placement() -> DecodedPlacement:
    return DecodedPlacement(
        hot=(
            _level("steam_high", UtilitySide.HOT, 200.0, 199.99, 0),
            _level("steam_low", UtilitySide.HOT, 130.0, 129.99, 1),
        ),
        cold=(
            _level("cw_low", UtilitySide.COLD, 20.0, 20.01, 0),
            _level("cw_high", UtilitySide.COLD, 50.0, 50.01, 1),
        ),
        coordinates=(200.0, 130.0, 20.0, 50.0),
    )


def _period() -> PlacementPeriodInput:
    return PlacementPeriodInput(
        period_id="p1",
        weight=1.0,
        snapshot=PlacementTargetSnapshot(
            shifted_temperatures=(220.0, 120.0, 10.0),
            real_temperatures=(225.0, 125.0, 15.0),
            hot_load_profile=(100.0, 40.0, 0.0),
            cold_load_profile=(0.0, 20.0, 80.0),
            real_hot_composite=(80.0, 40.0, 0.0),
            real_cold_composite=(100.0, 50.0, 0.0),
            hot_pinch_index=2,
            cold_pinch_index=0,
            entropy_slices=(),
        ),
        residual_hot_duty=100.0,
        residual_cold_duty=80.0,
        ambient_temperature_kelvin=298.15,
        coordinate_bounds=(),
    )


@dataclass
class _RecordingAdapter:
    hot: tuple[float, ...]
    cold: tuple[float, ...]
    calls: int = 0

    def allocate(self, period, placement):
        self.calls += 1
        return AllocationAdapterResult(hot_duties=self.hot, cold_duties=self.cold)


def test_allocation_uses_fresh_adapter_result_and_preserves_stable_keys() -> None:
    adapter = _RecordingAdapter(hot=(60.0, 40.0), cold=(0.0, 80.0))
    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=_period(),
        placement=_placement(),
        adapter=adapter,
    )

    assert result.feasible
    assert adapter.calls == 1
    assert tuple(level.template_key.name for level in result.hot_levels) == (
        "steam_high",
        "steam_low",
    )
    assert tuple(level.allocated_duty for level in result.hot_levels) == (60.0, 40.0)
    assert result.cold_levels[0].allocated_duty == 0.0
    assert result.hot_coverage_residual == pytest.approx(0.0)
    assert result.cold_coverage_residual == pytest.approx(0.0)


def test_allocation_uses_candidate_local_targets_snapshot_and_exact_fallback() -> None:
    period = _period()
    exact_snapshot = period.snapshot.model_copy(
        update={"shifted_temperatures": (230.0, 120.0, 5.0)}
    )
    adapter = _RecordingAdapter(hot=(), cold=())
    adapter.allocate = lambda period, placement: AllocationAdapterResult(
        hot_duties=(70.0, 0.0),
        cold_duties=(60.0, 0.0),
        hot_fallback_duty=10.0,
        required_hot_duty=80.0,
        required_cold_duty=60.0,
        target_snapshot=exact_snapshot,
        hot_fallback_name="exact_HU",
        hot_fallback_supply_temperature=260.0,
        hot_fallback_target_temperature=259.99,
    )

    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=period,
        placement=_placement(),
        adapter=adapter,
    )

    assert result.feasible
    assert result.required_hot_duty == pytest.approx(80.0)
    assert result.required_cold_duty == pytest.approx(60.0)
    assert result.target_snapshot == exact_snapshot
    assert result.hot_levels[-1].template_key.name == "exact_HU"
    assert result.hot_levels[-1].supply_temperature == pytest.approx(260.0)


def test_allocation_uses_combined_tolerance_without_clipping() -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        tolerances={"coverage": 1e-6, "relative": 1e-3},
    )
    adapter = _RecordingAdapter(hot=(50.0, 49.900001), cold=(40.0, 40.0))
    result = allocate_placement_period(
        request=request,
        period=_period(),
        placement=_placement(),
        adapter=adapter,
    )

    assert result.feasible
    assert result.hot_coverage_residual == pytest.approx(0.099999)
    assert result.hot_levels[1].allocated_duty == pytest.approx(49.900001)
    assert result.coverage_tolerance_hot == pytest.approx(0.100001)


def test_allocation_reports_shortfall_and_rejects_bad_adapter_shape() -> None:
    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=_period(),
        placement=_placement(),
        adapter=_RecordingAdapter(hot=(50.0, 40.0), cold=(80.0, 0.0)),
    )
    assert not result.feasible
    assert result.diagnostics[0].code == "hot_coverage_shortfall"

    malformed = _RecordingAdapter(hot=(100.0,), cold=(80.0, 0.0))
    with pytest.raises(Exception, match="duty count"):
        allocate_placement_period(
            request=UtilityPlacementRequest(isothermal_level_count=2),
            period=_period(),
            placement=_placement(),
            adapter=malformed,
        )


def test_candidate_correctable_adapter_diagnostic_is_ordinary_infeasibility() -> None:
    diagnostic = CandidateDiagnostic(
        code="temperature_unreachable",
        constraint="targeting",
        message="No utility can reach an interval.",
    )
    adapter = _RecordingAdapter(hot=(), cold=())
    adapter.allocate = lambda period, placement: AllocationAdapterResult(
        hot_duties=(0.0, 0.0),
        cold_duties=(0.0, 0.0),
        diagnostics=(diagnostic,),
    )

    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=_period(),
        placement=_placement(),
        adapter=adapter,
    )

    assert not result.feasible
    assert result.diagnostics[0] == diagnostic


def test_existing_targeting_caps_named_levels_then_allocates_fallback() -> None:
    period = _period().model_copy(
        update={
            "maximum_duties": (
                ("steam_high", 20.0),
                ("steam_low", 10.0),
                ("cw_low", 15.0),
                ("cw_high", 5.0),
            )
        }
    )
    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=period,
        placement=_placement(),
    )

    assert result.feasible
    named_hot = [level for level in result.hot_levels if not level.is_fallback]
    named_cold = [level for level in result.cold_levels if not level.is_fallback]
    assert all(level.allocated_duty <= level.maximum_duty + 1e-9 for level in named_hot)
    assert all(
        level.allocated_duty <= level.maximum_duty + 1e-9 for level in named_cold
    )
    assert result.hot_levels[-1].template_key.name == "HU"
    assert result.cold_levels[-1].template_key.name == "CU"
    assert result.hot_levels[-1].is_fallback
    assert result.cold_levels[-1].is_fallback
    assert result.allocated_hot_duty == pytest.approx(period.residual_hot_duty)
    assert result.allocated_cold_duty == pytest.approx(period.residual_cold_duty)


def test_zero_maximum_disables_only_its_named_utility() -> None:
    period = _period().model_copy(
        update={"maximum_duties": (("steam_high", 0.0), ("cw_low", 0.0))}
    )

    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=period,
        placement=_placement(),
    )

    by_name = {
        level.template_key.name: level
        for level in result.hot_levels + result.cold_levels
    }
    assert by_name["steam_high"].allocated_duty == 0.0
    assert by_name["cw_low"].allocated_duty == 0.0
    assert by_name["steam_low"].maximum_duty is None
    assert by_name["cw_high"].maximum_duty is None
    assert result.feasible


def test_fallback_uses_context_wide_temperature_support() -> None:
    period = _period().model_copy(
        update={
            "maximum_duties": (("steam_high", 0.0), ("cw_low", 0.0)),
            "fallback_hot_target_temperature": 500.0,
            "fallback_cold_target_temperature": -100.0,
        }
    )

    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=period,
        placement=_placement(),
    )

    fallback_hot = next(level for level in result.hot_levels if level.is_fallback)
    fallback_cold = next(level for level in result.cold_levels if level.is_fallback)
    assert fallback_hot.target_temperature == 500.0
    assert fallback_hot.supply_temperature == pytest.approx(500.01)
    assert fallback_cold.target_temperature == -100.0
    assert fallback_cold.supply_temperature == pytest.approx(-100.01)


def test_adapter_duty_above_maximum_is_infeasible() -> None:
    period = _period().model_copy(update={"maximum_duties": (("steam_high", 10.0),)})
    adapter = _RecordingAdapter(hot=(20.0, 80.0), cold=(100.0, 0.0))

    result = allocate_placement_period(
        request=UtilityPlacementRequest(isothermal_level_count=2),
        period=period,
        placement=_placement(),
        adapter=adapter,
    )

    assert not result.feasible
    assert any(item.code == "maximum_duty_exceeded" for item in result.diagnostics)
