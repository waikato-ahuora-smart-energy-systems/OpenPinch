"""Detached utility-placement context examples."""

from __future__ import annotations

import pickle

import pytest

from OpenPinch.analysis.utility_placement.context import (
    PlacementPeriodInput,
    PlacementTargetSnapshot,
    ProcessEntropySlice,
    build_utility_placement_context,
)
from OpenPinch.analysis.utility_placement.errors import PlacementContextError
from OpenPinch.analysis.utility_placement.normalization import (
    prepare_template_blueprints,
)
from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    QuantityInterval,
    UtilityPlacementBaseTarget,
    UtilityPlacementRequest,
    UtilitySide,
)


def _period_input(
    request: UtilityPlacementRequest,
    period_id: str = "summer",
    weight: float = 2.0,
) -> PlacementPeriodInput:
    blueprints = prepare_template_blueprints(request)
    bounds = tuple(
        PhysicalCoordinateBound(
            coordinate=CoordinateKey(
                template_key=blueprint.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            ),
            bounds=QuantityInterval(
                lower=120.0 if blueprint.key.side is UtilitySide.HOT else 5.0,
                upper=250.0 if blueprint.key.side is UtilitySide.HOT else 80.0,
                unit="degC",
            ),
            reason="fixture profile",
        )
        for blueprint in blueprints.all
    )
    snapshot = PlacementTargetSnapshot(
        shifted_temperatures=(200.0, 100.0, 20.0),
        real_temperatures=(205.0, 105.0, 25.0),
        hot_load_profile=(100.0, 40.0, 0.0),
        cold_load_profile=(0.0, 25.0, 80.0),
        real_hot_composite=(80.0, 40.0, 0.0),
        real_cold_composite=(100.0, 50.0, 0.0),
        hot_pinch_index=2,
        cold_pinch_index=0,
        entropy_slices=(
            ProcessEntropySlice(
                interval_index=0,
                side=UtilitySide.HOT,
                temperature_in_kelvin=378.15,
                temperature_out_kelvin=478.15,
                available_duty=100.0,
                heat_capacity_flow=1.0,
            ),
            ProcessEntropySlice(
                interval_index=1,
                side=UtilitySide.COLD,
                temperature_in_kelvin=298.15,
                temperature_out_kelvin=378.15,
                available_duty=80.0,
                heat_capacity_flow=1.0,
            ),
        ),
    )
    return PlacementPeriodInput(
        period_id=period_id,
        weight=weight,
        snapshot=snapshot,
        residual_hot_duty=100.0,
        residual_cold_duty=80.0,
        ambient_temperature_kelvin=298.15,
        coordinate_bounds=bounds,
    )


@pytest.mark.parametrize(
    "scope",
    [
        UtilityPlacementBaseTarget.DIRECT,
        UtilityPlacementBaseTarget.INDIRECT,
        UtilityPlacementBaseTarget.TOTAL_SITE,
    ],
)
def test_build_context_preserves_resolved_scope_periods_and_envelope(scope) -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        period_ids=("summer", "winter"),
    )
    blueprints = prepare_template_blueprints(request)
    periods = (
        _period_input(request, "summer", 2.0),
        _period_input(request, "winter", 1.0),
    )

    context = build_utility_placement_context(
        request=request,
        blueprints=blueprints,
        scope=scope,
        base_target_id=f"{scope.value}:zone",
        periods=periods,
    )

    assert context.scope is scope
    assert tuple(period.period_id for period in context.periods) == (
        "summer",
        "winter",
    )
    assert tuple(period.weight for period in context.envelope.periods) == (2.0, 1.0)
    assert context.periods[0].snapshot == periods[0].snapshot
    assert pickle.loads(pickle.dumps(context)) == context


def test_context_input_remains_unchanged() -> None:
    request = UtilityPlacementRequest(isothermal_level_count=2)
    blueprints = prepare_template_blueprints(request)
    period = _period_input(request)
    before = pickle.dumps((request, blueprints, period))

    build_utility_placement_context(
        request=request,
        blueprints=blueprints,
        scope=UtilityPlacementBaseTarget.DIRECT,
        base_target_id="direct:zone",
        periods=(period,),
    )

    assert pickle.dumps((request, blueprints, period)) == before


def test_context_rejects_auto_scope_and_period_mismatch() -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        period_ids=("summer",),
    )
    blueprints = prepare_template_blueprints(request)
    period = _period_input(request, "winter")

    with pytest.raises(PlacementContextError, match="resolved"):
        build_utility_placement_context(
            request=request,
            blueprints=blueprints,
            scope=UtilityPlacementBaseTarget.AUTO,
            base_target_id="auto:zone",
            periods=(period,),
        )
    with pytest.raises(PlacementContextError, match="period"):
        build_utility_placement_context(
            request=request,
            blueprints=blueprints,
            scope=UtilityPlacementBaseTarget.DIRECT,
            base_target_id="direct:zone",
            periods=(period,),
        )


def test_context_rejects_nonpositive_ambient_and_incomplete_bounds() -> None:
    request = UtilityPlacementRequest(isothermal_level_count=2)
    blueprints = prepare_template_blueprints(request)
    period = _period_input(request)

    with pytest.raises(PlacementContextError, match="ambient"):
        build_utility_placement_context(
            request=request,
            blueprints=blueprints,
            scope=UtilityPlacementBaseTarget.DIRECT,
            base_target_id="direct:zone",
            periods=(period.model_copy(update={"ambient_temperature_kelvin": 0.0}),),
        )
    with pytest.raises(PlacementContextError, match="coordinate"):
        build_utility_placement_context(
            request=request,
            blueprints=blueprints,
            scope=UtilityPlacementBaseTarget.DIRECT,
            base_target_id="direct:zone",
            periods=(
                period.model_copy(
                    update={"coordinate_bounds": period.coordinate_bounds[:-1]}
                ),
            ),
        )
