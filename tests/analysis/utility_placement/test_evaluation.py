"""Evaluation-session memo and complete thermodynamic replay tests."""

from __future__ import annotations

from dataclasses import dataclass

from OpenPinch.analysis.utility_placement.allocation import AllocationAdapterResult
from OpenPinch.analysis.utility_placement.codec import build_utility_placement_model
from OpenPinch.analysis.utility_placement.context import (
    PlacementPeriodInput,
    PlacementTargetSnapshot,
    ProcessEntropySlice,
    build_utility_placement_context,
)
from OpenPinch.analysis.utility_placement.evaluation import PlacementEvaluationSession
from OpenPinch.analysis.utility_placement.normalization import (
    prepare_template_blueprints,
)
from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    QuantityInterval,
    UtilityLevelKind,
    UtilityLevelTemplate,
    UtilityPlacementBaseTarget,
    UtilityPlacementRequest,
    UtilitySide,
)


@dataclass
class _Adapter:
    calls: int = 0

    def allocate(self, period, placement):
        self.calls += 1
        return AllocationAdapterResult(
            hot_duties=(period.residual_hot_duty, 0.0),
            cold_duties=(period.residual_cold_duty, 0.0),
        )


def _case(request=None):
    if request is None:
        generated_request = UtilityPlacementRequest(isothermal_level_count=2)
        generated_blueprints = prepare_template_blueprints(generated_request)
        request = generated_request.model_copy(
            update={
                "hot_templates": tuple(
                    item.as_template() for item in generated_blueprints.hot
                ),
                "cold_templates": tuple(
                    item.as_template() for item in generated_blueprints.cold
                ),
            }
        )
    blueprints = prepare_template_blueprints(request)
    coordinate_bounds = tuple(
        PhysicalCoordinateBound(
            coordinate=CoordinateKey(
                template_key=blueprint.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            ),
            bounds=QuantityInterval(
                lower=180.0 if blueprint.key.side is UtilitySide.HOT else 5.0,
                upper=260.0 if blueprint.key.side is UtilitySide.HOT else 80.0,
                unit="degC",
            ),
            reason="test target profile",
        )
        for blueprint in blueprints.all
    )
    period = PlacementPeriodInput(
        period_id="p1",
        weight=2.0,
        snapshot=PlacementTargetSnapshot(
            shifted_temperatures=(300.0, 100.0, 0.0),
            real_temperatures=(300.0, 100.0, 0.0),
            hot_load_profile=(100.0, 30.0, 0.0),
            cold_load_profile=(0.0, 20.0, 80.0),
            real_hot_composite=(80.0, 40.0, 0.0),
            real_cold_composite=(100.0, 50.0, 0.0),
            hot_pinch_index=2,
            cold_pinch_index=0,
            entropy_slices=(
                ProcessEntropySlice(
                    interval_index=0,
                    side=UtilitySide.HOT,
                    temperature_in_kelvin=300.0,
                    temperature_out_kelvin=500.0,
                    available_duty=100.0,
                    heat_capacity_flow=0.5,
                ),
                ProcessEntropySlice(
                    interval_index=1,
                    side=UtilitySide.COLD,
                    temperature_in_kelvin=350.0,
                    temperature_out_kelvin=280.0,
                    available_duty=80.0,
                    heat_capacity_flow=80.0 / 70.0,
                ),
            ),
        ),
        residual_hot_duty=100.0,
        residual_cold_duty=80.0,
        ambient_temperature_kelvin=298.15,
        coordinate_bounds=coordinate_bounds,
    )
    context = build_utility_placement_context(
        request=request,
        blueprints=blueprints,
        scope=UtilityPlacementBaseTarget.DIRECT,
        base_target_id="direct:test",
        periods=(period,),
    )
    model = build_utility_placement_model(request, blueprints, context.envelope)
    return request, context, model


def test_evaluation_session_replays_once_for_exact_coordinate_memo() -> None:
    request, context, model = _case()
    adapter = _Adapter()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=adapter,
    )

    first = session.evaluate(model.initial_points[0])
    second = session.evaluate(tuple(model.initial_points[0]))

    assert first == second
    assert first.feasible
    assert 0.0 < first.scalar_objective < 1.0
    assert first.physical_objective is not None
    assert first.period_results[0].thermodynamic is not None
    assert adapter.calls == 1
    assert session.evaluation_count == 1
    assert session.memo_hit_count == 1


def test_evaluation_session_penalizes_default_utility_allocation() -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        hot_templates=(
            UtilityLevelTemplate(
                name="HU",
                side=UtilitySide.HOT,
                kind=UtilityLevelKind.ISOTHERMAL,
            ),
            UtilityLevelTemplate(
                name="declared_hot",
                side=UtilitySide.HOT,
                kind=UtilityLevelKind.ISOTHERMAL,
            ),
        ),
        cold_templates=(
            UtilityLevelTemplate(
                name="declared_cold_1",
                side=UtilitySide.COLD,
                kind=UtilityLevelKind.ISOTHERMAL,
            ),
            UtilityLevelTemplate(
                name="declared_cold_2",
                side=UtilitySide.COLD,
                kind=UtilityLevelKind.ISOTHERMAL,
            ),
        ),
    )
    request, context, model = _case(request)
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    result = session.evaluate(model.initial_points[0])

    assert not result.feasible
    assert result.diagnostics[0].code == "default_utility_forbidden"
    assert 1.0 <= result.scalar_objective < 2.0


def test_evaluation_session_returns_graded_infeasibility_for_bad_candidate() -> None:
    request, context, model = _case()
    session = PlacementEvaluationSession(
        request=request,
        context=context,
        model=model,
        allocation_adapter=_Adapter(),
    )
    bad = tuple(coordinate.bounds.upper + 10.0 for coordinate in model.coordinates)

    result = session.evaluate(bad)

    assert not result.feasible
    assert 1.0 <= result.scalar_objective < 2.0
    assert result.physical_objective is None
    assert result.diagnostics
    assert len(session.diagnostic_representatives) <= 10
