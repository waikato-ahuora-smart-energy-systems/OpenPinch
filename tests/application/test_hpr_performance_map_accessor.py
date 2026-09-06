"""Public contracts for HPR backend selection and explicit map generation."""

from __future__ import annotations

import inspect
import json

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

import OpenPinch.application._problem.accessors.target as target_accessor
from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.performance_maps.generation import (
    generate_hpr_performance_map,
)
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.contracts.hpr_performance_map import (
    HprPerformanceMap,
    HprPerformanceMapRequest,
)
from OpenPinch.resources import load_hpr_performance_map_contract_resource
from tests.analysis.heat_pumps.hpr_map_fakes import FakeHprPointSimulator
from tests.analysis.heat_pumps.test_hpr_target_basis import _target
from tests.contracts.test_hpr_target_simulation_record import _record


@pytest.mark.parametrize(
    "method_name",
    ["vapour_compression_heat_pump", "vapour_compression_refrigeration"],
)
def test_vapour_compression_methods_expose_coolprop_default(method_name: str) -> None:
    parameters = inspect.signature(
        getattr(PinchProblem().target, method_name)
    ).parameters

    assert parameters["simulation_backend"].default == "coolprop"


def test_map_generation_is_an_explicit_keyword_only_target_operation() -> None:
    parameters = inspect.signature(PinchProblem().target.hpr_performance_map).parameters

    assert tuple(parameters) == ("target", "request")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )


@pytest.mark.parametrize("value", ["", "unknown", None, 1, object()])
def test_invalid_backend_fails_before_hpr_execution(monkeypatch, value: object) -> None:
    called = False

    def forbidden_hpr(self, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("HPR execution must not start")

    monkeypatch.setattr(_TargetAccessor, "_hpr", forbidden_hpr)

    with pytest.raises((TypeError, ValueError), match="simulation_backend"):
        PinchProblem().target.vapour_compression_heat_pump(
            simulation_backend=value,
        )

    assert called is False


@pytest.mark.parametrize(
    "method_name", ["vapour_compression_heat_pump", "vapour_compression_refrigeration"]
)
def test_omitted_and_explicit_coolprop_forward_identically(
    monkeypatch, method_name: str
) -> None:
    observed: list[dict[str, object]] = []

    def capture_hpr(self, **kwargs):
        observed.append(kwargs)
        return object()

    monkeypatch.setattr(_TargetAccessor, "_hpr", capture_hpr)
    method = getattr(PinchProblem().target, method_name)

    method()
    method(simulation_backend="  COOLPROP  ")

    assert len(observed) == 2
    assert observed[0] == observed[1]
    assert observed[0]["simulation_backend"] == "coolprop"


@seed(20260715)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    method_name=st.sampled_from(
        ("vapour_compression_heat_pump", "vapour_compression_refrigeration")
    ),
    explicit_backend=st.sampled_from(("coolprop", " COOLPROP ", "CoolProp")),
)
def test_generated_omitted_and_explicit_coolprop_share_the_same_target_oracle(
    monkeypatch,
    method_name: str,
    explicit_backend: str,
) -> None:
    def deterministic_hpr(self, **kwargs):
        useful_duty = 400.0 if kwargs["is_heat_pump"] else 300.0
        compressor_power = useful_duty / 4.0
        return {
            "target_type": "heat_pump" if kwargs["is_heat_pump"] else "refrigeration",
            "simulation_backend": kwargs["simulation_backend"],
            "useful_duty": useful_duty,
            "compressor_power": compressor_power,
            "cop": useful_duty / compressor_power,
        }

    monkeypatch.setattr(_TargetAccessor, "_hpr", deterministic_hpr)
    method = getattr(PinchProblem().target, method_name)

    omitted = method()
    explicit = method(simulation_backend=explicit_backend)

    assert omitted == explicit
    assert omitted["simulation_backend"] == "coolprop"


def test_backend_selection_does_not_generate_a_map(monkeypatch) -> None:
    observed: list[str] = []

    def capture_hpr(self, **kwargs):
        observed.append(str(kwargs["simulation_backend"]))
        return object()

    monkeypatch.setattr(_TargetAccessor, "_hpr", capture_hpr)

    PinchProblem().target.vapour_compression_heat_pump(simulation_backend="tespy")

    assert observed == ["tespy"]


def test_backend_selector_is_runtime_intent_not_cycle_configuration(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_execute(self, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_TargetAccessor, "_execute", capture_execute)

    PinchProblem().target.vapour_compression_heat_pump(
        simulation_backend="  TESPY  ",
        condensers=1,
        evaporators=1,
    )

    assert captured["options"] == {"simulation_backend": "tespy"}
    assert captured["configuration"]["HPR_N_COND"] == 1
    assert captured["configuration"]["HPR_N_EVAP"] == 1
    assert "simulation_backend" not in captured["configuration"]


@pytest.mark.parametrize(
    ("mode", "backend", "fixture_name", "capacity_basis"),
    [
        ("heat_pump", "coolprop", "heat-pump-1.0.json", "q_sink"),
        ("refrigeration", "tespy", "refrigeration-1.0.json", "q_source"),
    ],
)
def test_explicit_map_accessor_returns_complete_target_owned_backend_map(
    monkeypatch,
    mode: str,
    backend: str,
    fixture_name: str,
    capacity_basis: str,
) -> None:
    problem = PinchProblem()
    record = _record(mode=mode, simulation_backend=backend, period_id="winter")
    target = _target(record)
    request = HprPerformanceMapRequest(
        map_id=f"{mode}-map",
        source_temperatures=[10.0],
        sink_temperatures=[45.0],
        load_fractions=[0.5, 1.0],
        reference_capacity=250.0,
    )
    target_before = record.model_dump(mode="json")
    request_before = request.model_dump(mode="json")
    problem_before = (
        problem._problem_data,
        problem._master_zone,
        problem._results,
        problem._last_target_run_spec,
    )
    calls = []

    def generate(basis, received_request):
        calls.append((basis, received_request))
        return generate_hpr_performance_map(
            basis,
            received_request,
            simulator_factory=lambda _backend: FakeHprPointSimulator(),
        )

    monkeypatch.setattr(target_accessor, "generate_hpr_performance_map", generate)

    result = problem.target.hpr_performance_map(target=target, request=request)

    assert isinstance(result, HprPerformanceMap)
    assert len(calls) == 1
    assert calls[0][0].simulation_backend == backend
    assert result.thermodynamic_backend == backend
    assert result.reference_capacity == pytest.approx(250.0)
    assert result.reference_capacity_basis == capacity_basis
    assert result == HprPerformanceMap.model_validate_json(result.model_dump_json())
    golden = load_hpr_performance_map_contract_resource(fixture_name)
    assert set(json.loads(result.model_dump_json())) == set(golden)
    assert result.schema_version == golden["schema_version"]
    assert result.mode == golden["mode"]
    assert result.interpolation_topology == golden["interpolation_topology"]
    assert record.model_dump(mode="json") == target_before
    assert request.model_dump(mode="json") == request_before
    assert (
        problem._problem_data,
        problem._master_zone,
        problem._results,
        problem._last_target_run_spec,
    ) == problem_before


def test_incompatible_target_fails_before_map_generator_or_simulator(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        target_accessor,
        "generate_hpr_performance_map",
        lambda *_args: pytest.fail("map generator must not start"),
    )
    request = HprPerformanceMapRequest(
        map_id="map",
        source_temperatures=[10.0],
        sink_temperatures=[45.0],
        load_fractions=[1.0],
    )

    with pytest.raises(ValueError, match="winning simulation record"):
        PinchProblem().target.hpr_performance_map(
            target=_target(hpr_details=object()),
            request=request,
        )


def test_map_accessor_requires_the_public_request_contract_before_generation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        target_accessor,
        "generate_hpr_performance_map",
        lambda *_args: pytest.fail("map generator must not start"),
    )

    with pytest.raises(TypeError, match="HprPerformanceMapRequest"):
        PinchProblem().target.hpr_performance_map(
            target=_target(),
            request={"map_id": "not-a-contract"},
        )


def test_map_operation_has_no_automatic_all_period_mirror_or_root_export() -> None:
    assert not hasattr(PinchProblem().target.all_periods, "hpr_performance_map")

    import OpenPinch

    assert "HprPerformanceMap" not in OpenPinch.__all__
