"""Examples for HPR map-generation context and operating points."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_pumps.performance_maps.context import (
    build_hpr_map_generation_context,
)
from OpenPinch.analysis.heat_pumps.performance_maps.errors import (
    HprMapGenerationError,
    HprSimulatorFailure,
    sanitize_details,
)
from OpenPinch.analysis.heat_pumps.performance_maps.fluids import (
    parse_hpr_working_fluid,
    resolve_hpr_working_fluid,
    to_tespy_fluid_token,
)
from OpenPinch.analysis.heat_pumps.performance_maps.generation import (
    generate_hpr_performance_map,
)
from OpenPinch.analysis.heat_pumps.performance_maps.models import (
    HprPointSimulation,
    HprSimulationDiagnostic,
    HprTargetMapBasis,
)
from OpenPinch.analysis.heat_pumps.performance_maps.points import (
    iter_hpr_operating_points,
)
from OpenPinch.contracts.hpr_performance_map import HprPerformanceMapRequest
from tests.analysis.heat_pumps.hpr_map_fakes import FakeHprPointSimulator


def _basis(**overrides) -> HprTargetMapBasis:
    values = {
        "target_id": "target-1",
        "mode": "heat_pump",
        "simulation_backend": "coolprop",
        "cycle_id": "single_stage_vapour_compression",
        "model_id": "openpinch-vapour-compression-v1",
        "refrigerant_spec": "R134a",
        "nominal_evaporating_temperature": 5.0,
        "nominal_condensing_temperature": 55.0,
        "nominal_useful_duty": 1_000.0,
        "source_approach_temperature": 3.0,
        "sink_approach_temperature": 4.0,
        "compressor_isentropic_efficiency": 0.75,
        "superheat": 5.0,
        "subcooling": 2.0,
        "internal_hx_gas_temperature_change": 0.0,
        "source_provenance": {"target_method": "vapour_compression_heat_pump"},
    }
    values.update(overrides)
    return HprTargetMapBasis(**values)


def _request(**overrides) -> HprPerformanceMapRequest:
    values = {
        "map_id": "map-1",
        "source_temperatures": [8.0, 10.0],
        "sink_temperatures": [45.0, 50.0],
        "load_fractions": [0.5, 1.0],
    }
    values.update(overrides)
    return HprPerformanceMapRequest(**values)


def test_context_is_frozen_and_uses_target_capacity_and_external_approaches():
    basis = _basis()
    request = _request()

    context = build_hpr_map_generation_context(basis, request)

    assert context.basis is basis
    assert context.request is request
    assert context.reference_capacity == 1_000.0
    assert context.reference_capacity_basis == "q_sink"
    assert context.cop_convention == "heating"
    assert context.nominal_source_temperature == 8.0
    assert context.nominal_sink_temperature == 51.0
    assert context.energy_balance_tolerance == 1e-6
    assert context.temperature_match_tolerance == 1e-6
    with pytest.raises(FrozenInstanceError):
        context.reference_capacity = 2_000.0


def test_request_capacity_overrides_target_and_refrigeration_uses_source_basis():
    basis = _basis(mode="refrigeration")
    context = build_hpr_map_generation_context(
        basis,
        _request(reference_capacity=750.0),
    )

    assert context.reference_capacity == 750.0
    assert context.reference_capacity_basis == "q_source"
    assert context.cop_convention == "cooling"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"simulation_backend": "unknown"}, "backend"),
        ({"mode": "cogeneration"}, "mode"),
        ({"cycle_id": "carnot"}, "single-stage"),
        ({"nominal_useful_duty": 0.0}, "capacity"),
        ({"compressor_isentropic_efficiency": 1.1}, "efficiency"),
        ({"source_approach_temperature": -1.0}, "approach"),
    ],
)
def test_context_rejects_unsupported_or_invalid_target_basis(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_hpr_map_generation_context(_basis(**overrides), _request())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_id": ""}, "identifiers"),
        ({"nominal_evaporating_temperature": math.nan}, "finite"),
        ({"nominal_evaporating_temperature": -273.15}, "absolute zero"),
        (
            {
                "nominal_evaporating_temperature": 50.0,
                "nominal_condensing_temperature": 50.0,
            },
            "positive temperature lift",
        ),
        ({"source_approach_temperature": math.nan}, "finite"),
        ({"superheat": -1.0}, "nonnegative"),
        ({"compressor_isentropic_efficiency": 0.0}, "interval"),
    ],
)
def test_context_rejects_remaining_nonfinite_or_out_of_domain_values(
    overrides,
    match,
):
    with pytest.raises(ValueError, match=match):
        build_hpr_map_generation_context(_basis(**overrides), _request())


def test_context_rejects_nonpositive_internal_lift():
    basis = _basis(source_approach_temperature=5.0, sink_approach_temperature=5.0)
    request = _request(source_temperatures=[45.0], sink_temperatures=[35.0])

    context = build_hpr_map_generation_context(basis, request)
    points = list(iter_hpr_operating_points(context))

    assert len(points) == 2
    assert all(
        point.condensing_temperature == point.evaporating_temperature
        for point in points
    )
    assert all(point.is_valid is False for point in points)


@pytest.mark.parametrize(
    ("source", "kind", "components", "fractions"),
    [
        ("R134a", "pure", ("R134a",), (1.0,)),
        ("R407C", "registered_blend", (), ()),
        (
            "HEOS::R32[50]&R125[30]&R134a[20]",
            "explicit_molar_mixture",
            ("R32", "R125", "R134a"),
            (0.5, 0.3, 0.2),
        ),
    ],
)
def test_working_fluid_parser_preserves_supported_categories(
    source,
    kind,
    components,
    fractions,
):
    fluid = parse_hpr_working_fluid(source)

    assert fluid.source_spec == source
    assert fluid.property_backend == "HEOS"
    assert fluid.kind == kind
    assert fluid.components == components
    assert fluid.mole_fractions == pytest.approx(fractions)


def test_explicit_mixture_translates_to_one_tespy_molar_wrapper():
    fluid = parse_hpr_working_fluid("PR::R32[2]&R125[3]")

    assert fluid.property_backend == "PR"
    assert fluid.mole_fractions == pytest.approx((0.4, 0.6))
    assert to_tespy_fluid_token(fluid) == "PR::R32[0.4]&R125[0.6]|molar"


def test_refprop_is_rejected_before_property_state_construction(monkeypatch):
    import OpenPinch.analysis.heat_pumps.performance_maps.fluids as fluids

    called = False

    def fail_if_called(value):
        nonlocal called
        called = True
        raise AssertionError(value)

    monkeypatch.setattr(fluids, "build_coolprop_abstract_state", fail_if_called)

    with pytest.raises(ValueError, match="REFPROP"):
        resolve_hpr_working_fluid("REFPROP::R134a", 5.0, 55.0)
    assert called is False


@pytest.mark.parametrize(
    "value",
    [
        None,
        " ",
        "::R134a",
        "HEOS::",
        "HEOS::R32[0.5]&invalid",
        "HEOS::R32[0.5]&R32[0.5]",
        "HEOS::R32[-0.1]&R125[1.1]",
        "HEOS::R32[nan]&R125[1.0]",
        "HEOS::R32[0]&R125[0]",
    ],
)
def test_working_fluid_parser_rejects_malformed_compositions(value):
    with pytest.raises((TypeError, ValueError)):
        parse_hpr_working_fluid(value)


def test_registered_mixture_catalog_ignores_blanks_and_normalizes_mix_suffix(
    monkeypatch,
):
    import OpenPinch.analysis.heat_pumps.performance_maps.fluids as fluids

    fluids._registered_mixture_names.cache_clear()
    monkeypatch.setattr(
        fluids,
        "get_global_param_string",
        lambda name: " ,Custom.MIX,Other",
    )

    assert fluids._registered_mixture_names() == {
        "custom.mix",
        "custom",
        "other",
    }
    fluids._registered_mixture_names.cache_clear()


def test_working_fluid_resolution_translates_backend_state_failure():
    with pytest.raises(ValueError, match="dew/bubble"):
        resolve_hpr_working_fluid("Air", 5.0, 55.0)


def test_real_registered_blend_resolution_proves_dew_and_bubble_states():
    fluid = resolve_hpr_working_fluid("R407C", 0.0, 45.0)

    assert fluid.kind == "registered_blend"
    assert fluid.saturation_anchor == "evaporation_dew_condensation_bubble"


def test_operating_points_follow_canonical_cartesian_order_and_ordinals():
    context = build_hpr_map_generation_context(
        _basis(),
        _request(
            source_temperatures=[10.0, 8.0],
            sink_temperatures=[50.0, 45.0],
            load_fractions=[1.0, 0.5],
        ),
    )

    points = list(iter_hpr_operating_points(context))

    assert len(points) == 8
    assert [
        (point.source_temperature, point.sink_temperature, point.load_fraction)
        for point in points
    ] == [
        (source, sink, load)
        for source in (8.0, 10.0)
        for sink in (45.0, 50.0)
        for load in (0.5, 1.0)
    ]
    assert [point.ordinal for point in points] == list(range(8))
    assert len({point.name for point in points}) == 8
    assert len({point.curve_id for point in points}) == 4
    assert points[0].requested_useful_duty == 500.0
    assert points[0].evaporating_temperature == 5.0
    assert points[0].condensing_temperature == 49.0


def test_operating_point_iterator_is_lazy():
    context = build_hpr_map_generation_context(_basis(), _request())

    points = iter_hpr_operating_points(context)

    assert iter(points) is points


def test_operating_point_below_absolute_zero_is_invalid_before_simulation():
    context = build_hpr_map_generation_context(
        _basis(),
        _request(source_temperatures=[-300.0], sink_temperatures=[45.0]),
    )

    point = next(iter_hpr_operating_points(context))

    assert point.is_valid is False
    assert point.validation_error == "translated temperatures must exceed absolute zero"


def test_adding_highest_load_preserves_existing_point_identities():
    basis = _basis()
    first = build_hpr_map_generation_context(
        basis,
        _request(load_fractions=[0.25, 0.5]),
    )
    extended = build_hpr_map_generation_context(
        basis,
        _request(load_fractions=[0.25, 0.5, 1.0]),
    )

    original = list(iter_hpr_operating_points(first))
    added = list(iter_hpr_operating_points(extended))
    added_by_coordinate = {
        (point.source_temperature, point.sink_temperature, point.load_fraction): point
        for point in added
    }

    for point in original:
        coordinate = (
            point.source_temperature,
            point.sink_temperature,
            point.load_fraction,
        )
        matching = added_by_coordinate[coordinate]
        assert matching.curve_id == point.curve_id
        assert matching.name == point.name


def test_context_construction_does_not_mutate_inputs():
    provenance = {"nested": {"value": 1}}
    basis = _basis(source_provenance=provenance)
    request = _request()
    basis_before = replace(basis)
    request_before = request.model_copy(deep=True)

    build_hpr_map_generation_context(basis, request)

    assert basis == basis_before
    assert request == request_before
    assert provenance == {"nested": {"value": 1}}


def _generate_with_fake(basis=None, request=None, fake=None):
    selected_basis = basis or _basis()
    selected_request = request or _request()
    selected_fake = fake or FakeHprPointSimulator()
    factory_calls = []

    def factory(backend):
        factory_calls.append(backend)
        return selected_fake

    result = generate_hpr_performance_map(
        selected_basis,
        selected_request,
        simulator_factory=factory,
    )
    return result, selected_fake, factory_calls


def test_complete_fake_generation_constructs_one_unit1_map_atomically():
    result, fake, factory_calls = _generate_with_fake()

    assert result.schema_version == "1.0"
    assert result.thermodynamic_backend == "coolprop"
    assert result.reference_capacity == 1_000.0
    assert len(result.points) == 8
    assert [point.name for point in result.points] == [
        f"map-1-s{source}-k{sink}-l{load}"
        for source in range(2)
        for sink in range(2)
        for load in range(2)
    ]
    assert all(point.cop == pytest.approx(4.0) for point in result.points)
    assert factory_calls == ["coolprop"]
    assert fake.events == [
        ("prepare", None),
        *(("simulate", ordinal) for ordinal in range(8)),
        ("close", None),
    ]
    assert result.provenance["power_boundary"] == "compressor_only"
    assert result.provenance["modeled_auxiliaries"] == []


def test_equal_fake_calls_produce_structurally_equal_maps_and_provenance():
    first, _, _ = _generate_with_fake()
    second, _, _ = _generate_with_fake()

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_prepare_failure_closes_once_and_never_calls_a_point():
    cause = RuntimeError("raw engine path /private/tmp/secret")
    fake = FakeHprPointSimulator(
        prepare_failure=HprSimulatorFailure(
            "prepare_failed",
            "selected simulator could not prepare its design state",
            session_fatal=True,
            cause=cause,
        )
    )

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(fake=fake)

    assert [diagnostic.code for diagnostic in raised.value.diagnostics] == [
        "prepare_failed"
    ]
    assert fake.events == [("prepare", None), ("close", None)]
    assert raised.value.__cause__ is cause
    assert "/private/tmp" not in str(raised.value)


def test_point_local_failure_continues_without_retry_and_returns_no_map():
    fake = FakeHprPointSimulator(
        point_failures={
            2: HprSimulatorFailure(
                "non_converged",
                "selected operating point did not converge",
                session_fatal=False,
                details={"iterations": 50},
            )
        }
    )

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(fake=fake)

    assert [diagnostic.point_ordinal for diagnostic in raised.value.diagnostics] == [2]
    assert [event for event in fake.events if event[0] == "simulate"] == [
        ("simulate", ordinal) for ordinal in range(8)
    ]
    assert fake.events[-1] == ("close", None)
    assert not hasattr(raised.value, "partial_map")


def test_fatal_point_failure_marks_remaining_coordinates_unavailable():
    fake = FakeHprPointSimulator(
        point_failures={
            1: HprSimulatorFailure(
                "point_exception",
                "simulator session became unusable",
                session_fatal=True,
            )
        }
    )

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(fake=fake)

    assert [event for event in fake.events if event[0] == "simulate"] == [
        ("simulate", 0),
        ("simulate", 1),
    ]
    assert [
        diagnostic.point_ordinal for diagnostic in raised.value.diagnostics
    ] == list(range(1, 8))
    assert [diagnostic.code for diagnostic in raised.value.diagnostics] == [
        "point_exception",
        *("session_unavailable" for _ in range(6)),
    ]


def test_invalid_internal_lift_is_diagnostic_without_engine_call():
    fake = FakeHprPointSimulator()
    request = _request(
        source_temperatures=[45.0],
        sink_temperatures=[35.0, 60.0],
        load_fractions=[1.0],
    )
    basis = _basis(source_approach_temperature=5.0, sink_approach_temperature=5.0)

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(basis=basis, request=request, fake=fake)

    assert [diagnostic.code for diagnostic in raised.value.diagnostics] == [
        "invalid_operating_point"
    ]
    assert [event for event in fake.events if event[0] == "simulate"] == [
        ("simulate", 1)
    ]


def test_cleanup_failure_is_last_and_invalidates_success():
    fake = FakeHprPointSimulator(close_error=RuntimeError("cleanup path"))

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(fake=fake)

    assert raised.value.diagnostics[-1].code == "cleanup_failed"
    assert raised.value.diagnostics[-1].point_ordinal is None
    assert fake.events[-1] == ("close", None)


def test_untyped_prepare_and_point_exceptions_are_sanitized_and_atomic():
    class PrepareCrash(FakeHprPointSimulator):
        def prepare(self, context):
            self.events.append(("prepare", None))
            raise RuntimeError("raw preparation exception")

    with pytest.raises(HprMapGenerationError) as prepare_error:
        _generate_with_fake(fake=PrepareCrash())
    assert prepare_error.value.diagnostics[0].code == "prepare_failed"
    assert isinstance(prepare_error.value.__cause__, RuntimeError)

    class PointCrash(FakeHprPointSimulator):
        def simulate(self, point):
            self.events.append(("simulate", point.ordinal))
            raise RuntimeError("raw point exception")

    with pytest.raises(HprMapGenerationError) as point_error:
        _generate_with_fake(fake=PointCrash())
    assert point_error.value.diagnostics[0].code == "point_exception"
    assert len(point_error.value.diagnostics) == 8
    assert isinstance(point_error.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "metadata_override",
    [
        {"design_converged": False},
        {"backend": "tespy"},
        {"model_id": "wrong-model"},
    ],
)
def test_invalid_preparation_metadata_is_rejected(metadata_override):
    class InvalidMetadata(FakeHprPointSimulator):
        def prepare(self, context):
            metadata = super().prepare(context)
            return replace(metadata, **metadata_override)

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(fake=InvalidMetadata())

    assert raised.value.diagnostics[0].code == "prepare_failed"


def test_first_failure_cause_wins_when_cleanup_also_fails():
    cause = RuntimeError("first")
    fake = FakeHprPointSimulator(
        point_failures={
            0: HprSimulatorFailure(
                "point_exception",
                "point failed",
                session_fatal=False,
                cause=cause,
            )
        },
        close_error=RuntimeError("later cleanup"),
    )

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(fake=fake)

    assert raised.value.__cause__ is cause
    assert raised.value.diagnostics[-1].code == "cleanup_failed"


@pytest.mark.parametrize(
    ("simulation", "code"),
    [
        (HprPointSimulation(-1.0, 10.0, 11.0, True), "invalid_simulation"),
        (HprPointSimulation(10.0, 5.0, 1.0, True), "energy_balance"),
        (HprPointSimulation(10.0, 12.0, 2.0, True), "useful_duty_mismatch"),
        (HprPointSimulation(10.0, 12.0, 2.0, False), "non_converged"),
    ],
)
def test_corrupt_adapter_results_fail_independent_physical_validation(
    simulation,
    code,
):
    fake = FakeHprPointSimulator(point_results={0: simulation})
    request = _request(
        source_temperatures=[8.0],
        sink_temperatures=[45.0],
        load_fractions=[1.0],
    )

    with pytest.raises(HprMapGenerationError) as raised:
        _generate_with_fake(request=request, fake=fake)

    assert raised.value.diagnostics[0].code == code


def test_high_precision_fake_values_are_not_quantized():
    useful_duty = 123.456789012345
    power = 31.234567890123
    simulation = HprPointSimulation(
        q_source=useful_duty - power,
        q_sink=useful_duty,
        compressor_power=power,
        converged=True,
    )
    fake = FakeHprPointSimulator(point_results={0: simulation})
    request = _request(
        source_temperatures=[8.0],
        sink_temperatures=[45.0],
        load_fractions=[1.0],
        reference_capacity=useful_duty,
    )

    result, _, _ = _generate_with_fake(request=request, fake=fake)

    assert result.points[0].q_sink == useful_duty
    assert result.points[0].electric_power == power


def test_error_values_are_bounded_recursive_json_and_reject_empty_aggregate():
    details = sanitize_details(
        {
            "none": None,
            "boolean": True,
            "integer": 4,
            "finite": 1.5,
            "nonfinite": math.inf,
            "long": "x" * 200,
            "sequence": [1, SimpleNamespace(secret="hidden")],
            "mapping": {"nested": 2},
        }
    )

    assert details["none"] is None
    assert details["boolean"] is True
    assert details["integer"] == 4
    assert details["finite"] == 1.5
    assert details["nonfinite"] == "inf"
    assert len(details["long"]) == 160
    assert details["sequence"] == [1, "SimpleNamespace"]
    assert details["mapping"] == {"nested": 2}
    assert sanitize_details(None) == {}
    with pytest.raises(ValueError, match="must not be empty"):
        HprMapGenerationError(())

    diagnostic = HprSimulationDiagnostic(
        code="failure",
        backend=None,
        model_id=None,
        point_ordinal=None,
        curve_id=None,
        source_temperature=None,
        sink_temperature=None,
        load_fraction=None,
        message="failure",
    )
    assert "unresolved" in str(HprMapGenerationError((diagnostic,)))


def test_factory_rejects_unknown_backend_and_creates_selected_tespy_session():
    from OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy import (
        TespyHprPointSimulator,
    )
    from OpenPinch.analysis.heat_pumps.performance_maps.factory import (
        get_hpr_point_simulator,
    )

    assert isinstance(get_hpr_point_simulator("tespy"), TespyHprPointSimulator)
    with pytest.raises(HprSimulatorFailure) as raised:
        get_hpr_point_simulator("future")
    assert raised.value.code == "unsupported_backend"


def test_provenance_uses_unknown_when_distribution_metadata_is_unavailable(
    monkeypatch,
):
    import OpenPinch.analysis.heat_pumps.performance_maps.provenance as provenance

    monkeypatch.setattr(
        provenance,
        "version",
        lambda distribution: (_ for _ in ()).throw(
            provenance.PackageNotFoundError(distribution)
        ),
    )

    result, _, _ = _generate_with_fake()

    assert result.provenance["openpinch_version"] == "unknown"
