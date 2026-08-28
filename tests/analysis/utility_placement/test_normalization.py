"""Examples for request, template, and unit normalization."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.utility_placement import (
    normalize_utility_placement_request,
    prepare_template_blueprints,
)
from OpenPinch.analysis.utility_placement.errors import (
    PlacementRequestValidationError,
    UtilityPlacementUnitError,
    UtilityTemplateValidationError,
)
from OpenPinch.analysis.utility_placement.normalization import (
    convert_placement_value,
)
from OpenPinch.contracts.common import ValueWithUnit
from OpenPinch.contracts.utility_placement import (
    QuantityInterval,
    QuantityValue,
    UtilityLevelKind,
    UtilityLevelTemplate,
)
from OpenPinch.domain.value import Value


def test_count_only_request_generates_complete_symmetric_inventory() -> None:
    request = normalize_utility_placement_request(
        isothermal_level_count=2,
        sensible_level_count=1,
    )

    blueprints = prepare_template_blueprints(request)

    assert [item.key.name for item in blueprints.hot] == [
        "hot_iso_1",
        "hot_iso_2",
        "hot_sensible_1",
    ]
    assert [item.key.name for item in blueprints.cold] == [
        "cold_iso_1",
        "cold_iso_2",
        "cold_sensible_1",
    ]
    assert [item.placement_rank for item in blueprints.hot] == [0, 1, 2]
    assert all(
        item.fixed_span == QuantityValue(value=0.01, unit="delta_degC")
        for item in blueprints.all
        if item.kind is UtilityLevelKind.ISOTHERMAL
    )


def test_explicit_interleaved_templates_preserve_identity_and_side_order() -> None:
    hot = (
        UtilityLevelTemplate(
            name="hot_sensible",
            side="hot",
            kind="sensible",
            span_bounds=QuantityInterval(
                lower=5.0,
                upper=30.0,
                unit="delta_degC",
            ),
        ),
        UtilityLevelTemplate(
            name="hot_high",
            side="hot",
            kind="isothermal",
            fixed_span=QuantityValue(value=1.0, unit="delta_degC"),
        ),
        UtilityLevelTemplate(
            name="hot_low",
            side="hot",
            kind="isothermal",
            fixed_span=QuantityValue(value=2.0, unit="delta_degC"),
        ),
    )
    cold = tuple(
        UtilityLevelTemplate(
            name=f"cold_{index}",
            side="cold",
            kind="isothermal" if index < 2 else "sensible",
            fixed_span=(
                QuantityValue(value=0.01, unit="delta_degC") if index < 2 else None
            ),
            span_bounds=(
                QuantityInterval(lower=2.0, upper=20.0, unit="delta_degC")
                if index == 2
                else None
            ),
        )
        for index in range(3)
    )
    request = normalize_utility_placement_request(
        isothermal_level_count=2,
        sensible_level_count=1,
        hot_templates=hot,
        cold_templates=cold,
    )

    blueprints = prepare_template_blueprints(request)

    assert [item.key.name for item in blueprints.hot] == [
        "hot_sensible",
        "hot_high",
        "hot_low",
    ]
    assert [item.placement_rank for item in blueprints.hot] == [0, 1, 2]


def test_explicit_collection_must_be_complete_and_side_correct() -> None:
    one_hot = (
        UtilityLevelTemplate(
            name="only_one",
            side="hot",
            kind="isothermal",
        ),
    )
    request = normalize_utility_placement_request(
        isothermal_level_count=2,
        hot_templates=one_hot,
    )

    with pytest.raises(UtilityTemplateValidationError) as captured:
        prepare_template_blueprints(request)

    assert captured.value.code == "template_inventory_mismatch"


def test_template_names_must_be_globally_unique() -> None:
    templates = tuple(
        UtilityLevelTemplate(name=name, side=side, kind="isothermal")
        for name, side in (("same", "hot"), ("hot_2", "hot"))
    )
    cold = tuple(
        UtilityLevelTemplate(name=name, side="cold", kind="isothermal")
        for name in ("same", "cold_2")
    )
    request = normalize_utility_placement_request(
        isothermal_level_count=2,
        hot_templates=templates,
        cold_templates=cold,
    )

    with pytest.raises(UtilityTemplateValidationError, match="unique"):
        prepare_template_blueprints(request)


def test_conversion_adapter_matches_existing_value_owner() -> None:
    actual = convert_placement_value(
        ValueWithUnit(value=212.0, unit="degF"),
        canonical_unit="degC",
        default_unit="degC",
        field_path="supply",
    )
    oracle = float(Value(212.0, "degF").to("degC"))

    assert actual == pytest.approx(oracle)


def test_conversion_adapter_uses_default_for_bare_scalar_and_rejects_units() -> None:
    assert convert_placement_value(
        25.0,
        canonical_unit="degC",
        default_unit="degC",
        field_path="supply",
    ) == pytest.approx(25.0)

    with pytest.raises(UtilityPlacementUnitError) as captured:
        convert_placement_value(
            QuantityValue(value=1.0, unit="kW"),
            canonical_unit="degC",
            default_unit="degC",
            field_path="supply",
        )

    assert captured.value.context["field_path"] == "supply"


def test_conversion_adapter_rejects_invalid_scalar_shapes() -> None:
    invalid_values = (
        True,
        object(),
        ValueWithUnit(value=None, unit="degC"),
        Value([10.0, 20.0], "degC"),
    )
    for value in invalid_values:
        with pytest.raises(UtilityPlacementUnitError):
            convert_placement_value(
                value,
                canonical_unit="degC",
                default_unit="degC",
                field_path="temperature",
            )


def test_request_normalizer_rejects_ambiguous_missing_and_invalid_arguments() -> None:
    request = normalize_utility_placement_request(isothermal_level_count=2)
    with pytest.raises(PlacementRequestValidationError) as captured:
        normalize_utility_placement_request(request, isothermal_level_count=2)
    assert captured.value.code == "ambiguous_request"

    with pytest.raises(PlacementRequestValidationError) as captured:
        normalize_utility_placement_request()
    assert captured.value.code == "missing_isothermal_count"

    with pytest.raises(PlacementRequestValidationError) as captured:
        normalize_utility_placement_request(isothermal_level_count=True)
    assert captured.value.code == "invalid_request"


def test_kind_specific_template_validation_edges() -> None:
    cases = (
        UtilityLevelTemplate(
            name="isothermal_with_bounds",
            side="hot",
            kind="isothermal",
            span_bounds=QuantityInterval(
                lower=1.0,
                upper=2.0,
                unit="delta_degC",
            ),
        ),
        UtilityLevelTemplate(
            name="isothermal_zero",
            side="hot",
            kind="isothermal",
            fixed_span=QuantityValue(value=0.0, unit="delta_degC"),
        ),
        UtilityLevelTemplate(
            name="sensible_fixed",
            side="hot",
            kind="sensible",
            fixed_span=QuantityValue(value=1.0, unit="delta_degC"),
        ),
        UtilityLevelTemplate(
            name="sensible_nonpositive",
            side="hot",
            kind="sensible",
            span_bounds=QuantityInterval(
                lower=0.0,
                upper=2.0,
                unit="delta_degC",
            ),
        ),
    )
    for template in cases:
        if template.kind is UtilityLevelKind.ISOTHERMAL:
            hot = (
                template,
                UtilityLevelTemplate(
                    name="companion_iso",
                    side="hot",
                    kind="isothermal",
                ),
                UtilityLevelTemplate(
                    name="companion_sensible",
                    side="hot",
                    kind="sensible",
                ),
            )
        else:
            hot = (
                template,
                UtilityLevelTemplate(
                    name="companion_iso_1",
                    side="hot",
                    kind="isothermal",
                ),
                UtilityLevelTemplate(
                    name="companion_iso_2",
                    side="hot",
                    kind="isothermal",
                ),
            )
        request = normalize_utility_placement_request(
            isothermal_level_count=2,
            sensible_level_count=1,
            hot_templates=hot,
        )
        with pytest.raises(UtilityTemplateValidationError):
            prepare_template_blueprints(request)


def test_explicit_template_side_mismatch_is_rejected() -> None:
    request = normalize_utility_placement_request(
        isothermal_level_count=2,
        hot_templates=tuple(
            UtilityLevelTemplate(
                name=f"wrong_{index}",
                side="cold",
                kind="isothermal",
            )
            for index in range(2)
        ),
    )

    with pytest.raises(UtilityTemplateValidationError) as captured:
        prepare_template_blueprints(request)

    assert captured.value.code == "template_side_mismatch"
