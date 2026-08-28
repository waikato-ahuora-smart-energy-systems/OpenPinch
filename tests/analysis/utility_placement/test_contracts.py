"""Example tests for utility-placement contracts and errors."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from OpenPinch.analysis.utility_placement.errors import (
    PlacementRequestValidationError,
    UtilityPlacementError,
    UtilityPlacementUnitError,
    UtilityTemplateValidationError,
)
from OpenPinch.contracts.utility_placement import (
    CandidateDiagnostic,
    CandidateVerification,
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    PlacementFeasibilityEnvelope,
    PlacementPeriodEnvelope,
    PlacementPeriodResult,
    PlacementTermination,
    PlacementTolerances,
    QuantityInterval,
    QuantityValue,
    TemplateKey,
    ThermodynamicCostBreakdown,
    UtilityLevelKind,
    UtilityLevelPeriodResult,
    UtilityLevelTemplate,
    UtilityPlacementBaseTarget,
    UtilityPlacementCandidate,
    UtilityPlacementOptions,
    UtilityPlacementRequest,
    UtilityPlacementResult,
    UtilitySide,
)


def test_minimum_count_request_defaults_to_thermodynamic() -> None:
    request = UtilityPlacementRequest(isothermal_level_count=2)

    assert request.isothermal_level_count == 2
    assert request.sensible_level_count == 0
    assert request.base_target is UtilityPlacementBaseTarget.AUTO
    assert request.hot_templates is None
    assert request.cold_templates is None


@pytest.mark.parametrize(
    "value",
    [True, False, 1, 1.0, 2.5, "2", math.inf, math.nan],
)
def test_isothermal_count_rejects_non_integer_or_too_small_values(value) -> None:
    with pytest.raises(ValidationError):
        UtilityPlacementRequest(isothermal_level_count=value)


@pytest.mark.parametrize("value", [True, False, -1, 0.0, "0", 1.5])
def test_sensible_count_rejects_invalid_values(value) -> None:
    with pytest.raises(ValidationError):
        UtilityPlacementRequest(
            isothermal_level_count=2,
            sensible_level_count=value,
        )


def test_contracts_are_frozen_and_forbid_extra_fields() -> None:
    request = UtilityPlacementRequest(isothermal_level_count=2)

    with pytest.raises(ValidationError):
        request.isothermal_level_count = 3
    with pytest.raises(ValidationError):
        UtilityPlacementRequest(isothermal_level_count=2, surprise=True)


def test_stable_enum_values_are_lowercase_contract_vocabulary() -> None:
    assert [member.value for member in UtilityPlacementBaseTarget] == [
        "auto",
        "direct",
        "indirect",
        "total_site",
    ]
    assert [member.value for member in UtilitySide] == ["hot", "cold"]
    assert [member.value for member in UtilityLevelKind] == [
        "isothermal",
        "sensible",
    ]
    assert [member.value for member in DecisionField] == [
        "supply_temperature",
        "temperature_span",
    ]


def test_quantities_are_finite_and_normalize_signed_zero() -> None:
    quantity = QuantityValue(value=-0.0, unit="kW")

    assert quantity.value == 0.0
    assert math.copysign(1.0, quantity.value) == 1.0
    with pytest.raises(ValidationError):
        QuantityValue(value=math.inf, unit="kW")
    with pytest.raises(ValidationError):
        QuantityValue(value=1.0, unit="")


def test_equal_interval_is_valid_but_reversed_interval_is_not() -> None:
    fixed = QuantityInterval(lower=120.0, upper=120.0, unit="degC")

    assert fixed.lower == fixed.upper
    with pytest.raises(ValidationError):
        QuantityInterval(lower=121.0, upper=120.0, unit="degC")


def test_request_json_round_trip_preserves_tuple_order() -> None:
    templates = (
        UtilityLevelTemplate(
            name="steam_high",
            side=UtilitySide.HOT,
            kind=UtilityLevelKind.ISOTHERMAL,
            fixed_span=QuantityValue(value=0.01, unit="delta_degC"),
        ),
        UtilityLevelTemplate(
            name="steam_low",
            side=UtilitySide.HOT,
            kind=UtilityLevelKind.ISOTHERMAL,
            fixed_span=QuantityValue(value=0.02, unit="delta_degC"),
        ),
    )
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        hot_templates=templates,
        period_ids=("summer", "winter"),
    )

    restored = UtilityPlacementRequest.model_validate_json(request.model_dump_json())

    assert restored == request
    assert tuple(template.name for template in restored.hot_templates or ()) == (
        "steam_high",
        "steam_low",
    )


def test_options_and_tolerances_validate_bounds() -> None:
    options = UtilityPlacementOptions()
    tolerances = PlacementTolerances()

    assert options.seed == 20260715
    assert options.candidate_limit > 0
    assert tolerances.absolute == pytest.approx(1e-6)
    assert tolerances.relative == pytest.approx(1e-9)
    with pytest.raises(ValidationError):
        UtilityPlacementOptions(candidate_limit=0)
    with pytest.raises(ValidationError):
        PlacementTolerances(ordering=-1.0)


def test_error_taxonomy_has_stable_json_safe_context() -> None:
    error = UtilityTemplateValidationError(
        code="duplicate_template",
        message="Template names must be unique.",
        field_path="hot_templates",
        details=(("name", "steam"),),
    )

    assert isinstance(error, UtilityPlacementError)
    assert isinstance(error, ValueError)
    assert error.code == "duplicate_template"
    assert error.context == {
        "code": "duplicate_template",
        "message": "Template names must be unique.",
        "field_path": "hot_templates",
        "details": [["name", "steam"]],
    }
    assert issubclass(PlacementRequestValidationError, ValueError)
    assert issubclass(UtilityPlacementUnitError, ValueError)


def _nested_result() -> UtilityPlacementResult:
    request = UtilityPlacementRequest(isothermal_level_count=2)
    level = UtilityLevelPeriodResult(
        template_key={"side": "hot", "name": "hot_iso_1"},
        kind="isothermal",
        placement_rank=0,
        supply_temperature=QuantityValue(value=200.0, unit="degC"),
        target_temperature=QuantityValue(value=199.99, unit="degC"),
        temperature_span=QuantityValue(value=0.01, unit="delta_degC"),
        allocated_duty=QuantityValue(value=1_000.0, unit="kW"),
    )
    thermodynamic = ThermodynamicCostBreakdown(
        utility_entropy=QuantityValue(value=1.0, unit="kW/K"),
        process_entropy=QuantityValue(value=2.0, unit="kW/K"),
        total_entropy_generation=QuantityValue(value=3.0, unit="kW/K"),
        ambient_temperature=QuantityValue(value=25.0, unit="degC"),
        exergy_destruction=QuantityValue(value=894.45, unit="kW"),
    )
    period = PlacementPeriodResult(
        period_id="only",
        weight=1.0,
        hot_levels=(level,),
        cold_levels=(),
        allocated_hot_duty=QuantityValue(value=1_000.0, unit="kW"),
        allocated_cold_duty=QuantityValue(value=0.0, unit="kW"),
        residual_hot_duty=QuantityValue(value=0.0, unit="kW"),
        residual_cold_duty=QuantityValue(value=0.0, unit="kW"),
        hot_coverage_residual=QuantityValue(value=0.0, unit="kW"),
        cold_coverage_residual=QuantityValue(value=0.0, unit="kW"),
        coverage_tolerance=QuantityValue(value=1e-6, unit="kW"),
        feasible=True,
        thermodynamic=thermodynamic,
        selected_objective=QuantityValue(value=3.0, unit="kW/K"),
    )
    candidate = UtilityPlacementCandidate(
        coordinates=(200.0, 150.0, 20.0, 30.0),
        hot_levels=(),
        cold_levels=(),
        period_results=(period,),
        aggregate_objective=QuantityValue(value=3.0, unit="kW/K"),
        thermodynamic_total=QuantityValue(value=3.0, unit="kW/K"),
    )
    return UtilityPlacementResult(
        request=request,
        scope="direct",
        base_target_id="fixture",
        period_ids=("only",),
        period_weights=(1.0,),
        best=candidate,
        alternatives=(),
        termination=PlacementTermination(
            method="fixture",
            seed=20260715,
            status="success",
            code="converged",
            message="Fixture converged.",
            iterations=1,
            evaluations=2,
            candidate_count=1,
            feasible_candidate_count=1,
            iteration_limit=500,
            evaluation_limit=5_000,
        ),
    )


def test_nested_result_json_round_trip_preserves_optional_breakdowns() -> None:
    result = _nested_result()

    restored = UtilityPlacementResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.best.feasible


def test_result_contract_rejects_nonfinite_coordinates_and_infeasible_best() -> None:
    candidate = _nested_result().best

    with pytest.raises(ValidationError):
        UtilityPlacementCandidate.model_validate(
            {**candidate.model_dump(), "coordinates": (math.inf,)},
        )
    with pytest.raises(ValidationError):
        UtilityPlacementCandidate.model_validate(
            {**candidate.model_dump(), "feasible": False},
        )


def test_result_schema_exposes_stable_top_level_fields() -> None:
    assert set(UtilityPlacementResult.model_json_schema()["properties"]) == {
        "request",
        "scope",
        "base_target_id",
        "period_ids",
        "period_weights",
        "units",
        "best",
        "alternatives",
        "termination",
        "diagnostics",
    }


def test_diagnostic_json_round_trip_preserves_context() -> None:
    diagnostic = CandidateDiagnostic(
        code="coverage",
        constraint="hot_coverage",
        message="Hot duty was not fully covered.",
        side="hot",
        period_id="only",
        measured=QuantityValue(value=0.1, unit="kW"),
        limit=QuantityValue(value=1e-6, unit="kW"),
        details=(("source", "fixture"),),
    )

    assert (
        CandidateDiagnostic.model_validate_json(diagnostic.model_dump_json())
        == diagnostic
    )


def test_identity_and_envelope_contract_validation_edges() -> None:
    with pytest.raises(ValidationError):
        UtilityLevelTemplate(name=" ", side="hot", kind="isothermal")
    with pytest.raises(ValidationError):
        UtilityLevelTemplate(
            name="hot",
            side="hot",
            kind="isothermal",
            placement_rank=True,
        )
    with pytest.raises(ValidationError):
        UtilityLevelTemplate(
            name="hot",
            side="hot",
            kind="isothermal",
            placement_rank=-1,
        )
    with pytest.raises(ValidationError):
        TemplateKey(side="hot", name=" ")

    key = CoordinateKey(
        template_key=TemplateKey(side="hot", name="hot"),
        field="supply_temperature",
    )
    interval = QuantityInterval(lower=10.0, upper=20.0, unit="degC")
    with pytest.raises(ValidationError):
        PhysicalCoordinateBound(coordinate=key, bounds=interval, reason=" ")
    bound = PhysicalCoordinateBound(
        coordinate=key,
        bounds=interval,
        reason="fixture",
    )
    with pytest.raises(ValidationError):
        PlacementPeriodEnvelope(
            period_id=" ",
            weight=1.0,
            coordinate_bounds=(bound,),
        )
    with pytest.raises(ValidationError):
        PlacementPeriodEnvelope(
            period_id="period",
            weight=-1.0,
            coordinate_bounds=(bound,),
        )


def test_feasibility_envelope_validation_edges() -> None:
    key = CoordinateKey(
        template_key=TemplateKey(side="hot", name="hot"),
        field="supply_temperature",
    )
    bound = PhysicalCoordinateBound(
        coordinate=key,
        bounds=QuantityInterval(lower=10.0, upper=20.0, unit="degC"),
        reason="fixture",
    )
    period = PlacementPeriodEnvelope(
        period_id="period",
        weight=1.0,
        coordinate_bounds=(bound,),
    )
    base = {
        "minimum_separation": QuantityValue(value=1.0, unit="delta_degC"),
        "scope": "direct",
        "base_target_id": "fixture",
    }
    with pytest.raises(ValidationError):
        PlacementFeasibilityEnvelope(periods=(), **base)
    with pytest.raises(ValidationError):
        PlacementFeasibilityEnvelope(periods=(period, period), **base)
    with pytest.raises(ValidationError):
        PlacementFeasibilityEnvelope(
            periods=(period.model_copy(update={"weight": 0.0}),),
            **base,
        )
    with pytest.raises(ValidationError):
        PlacementFeasibilityEnvelope(
            periods=(period,),
            **{
                **base,
                "minimum_separation": QuantityValue(
                    value=0.0,
                    unit="delta_degC",
                ),
            },
        )
    with pytest.raises(ValidationError):
        PlacementFeasibilityEnvelope(
            periods=(period,),
            **{**base, "base_target_id": " "},
        )


def test_candidate_and_result_metadata_validation_edges() -> None:
    diagnostic = CandidateDiagnostic(
        code="code",
        constraint="constraint",
        message="message",
    )
    with pytest.raises(ValidationError):
        CandidateDiagnostic(code=" ", constraint="constraint", message="message")
    with pytest.raises(ValidationError):
        CandidateVerification(feasible=True, diagnostics=(diagnostic,))
    with pytest.raises(ValidationError):
        CandidateVerification(feasible=False, diagnostics=())

    result = _nested_result()
    period_payload = result.best.period_results[0].model_dump()
    with pytest.raises(ValidationError):
        PlacementPeriodResult.model_validate({**period_payload, "period_id": " "})
    with pytest.raises(ValidationError):
        PlacementPeriodResult.model_validate({**period_payload, "weight": -1.0})

    termination_payload = result.termination.model_dump()
    with pytest.raises(ValidationError):
        PlacementTermination.model_validate({**termination_payload, "method": " "})
    with pytest.raises(ValidationError):
        PlacementTermination.model_validate({**termination_payload, "seed": True})
    with pytest.raises(ValidationError):
        PlacementTermination.model_validate({**termination_payload, "iterations": -1})
    with pytest.raises(ValidationError):
        PlacementTermination.model_validate(
            {**termination_payload, "iteration_limit": 0}
        )

    payload = result.model_dump()
    invalid_updates = (
        {"base_target_id": " "},
        {"period_ids": ()},
        {"period_ids": ("same", "same"), "period_weights": (0.5, 0.5)},
        {"period_weights": (-1.0,)},
        {"period_weights": (0.0,)},
        {"scope": "auto"},
        {"period_ids": ("one", "two")},
    )
    for updates in invalid_updates:
        with pytest.raises(ValidationError):
            UtilityPlacementResult.model_validate({**payload, **updates})


def test_request_period_selection_validation_edges() -> None:
    with pytest.raises(ValidationError):
        UtilityPlacementRequest(isothermal_level_count=2, period_ids=())
    with pytest.raises(ValidationError):
        UtilityPlacementRequest(
            isothermal_level_count=2,
            period_ids=("same", "same"),
        )
