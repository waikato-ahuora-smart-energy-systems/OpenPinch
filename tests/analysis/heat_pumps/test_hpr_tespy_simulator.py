"""TESPy HPR adapter, deterministic settings, and owned-resource tests."""

# ruff: noqa: E402 - optional dependency gate must run before concrete imports.

from __future__ import annotations

import builtins
import hashlib
import inspect
import json
import math
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tespy")
pytestmark = pytest.mark.tespy

import OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy as tespy_adapter
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy import (
    TespyHprPointSimulator,
)
from OpenPinch.analysis.heat_pumps.performance_maps.context import (
    build_hpr_map_generation_context,
)
from OpenPinch.analysis.heat_pumps.performance_maps.errors import HprSimulatorFailure
from OpenPinch.analysis.heat_pumps.performance_maps.factory import (
    get_hpr_point_simulator,
)
from OpenPinch.analysis.heat_pumps.performance_maps.models import (
    HprWorkingFluidSpec,
)
from OpenPinch.analysis.heat_pumps.performance_maps.points import (
    iter_hpr_operating_points,
)
from OpenPinch.analysis.heat_pumps.performance_maps.resources import (
    CHARACTERISTIC_RESOURCE_NAME,
    load_tespy_compressor_characteristic,
    parse_tespy_compressor_characteristic,
    read_tespy_compressor_characteristic_bytes,
)
from OpenPinch.analysis.heat_pumps.performance_maps.settings import (
    TESPY_CONVERGENCE_SETTINGS,
)
from tests.analysis.heat_pumps.test_hpr_map_generation import _basis, _request

EXPECTED_CHARACTERISTIC_POINTS = (
    (0.49, 0.78),
    (0.55782, 0.82066),
    (0.62612, 0.86025),
    (0.69418, 0.89742),
    (0.76132, 0.93083),
    (0.82682, 0.95914),
    (0.89, 0.981),
    (0.95016, 0.99507),
    (1.0, 1.0),
    (1.04326, 0.99733),
    (1.07753, 0.98913),
    (1.10916, 0.97496),
    (1.13795, 0.95435),
    (1.16365, 0.92687),
    (1.18604, 0.89205),
    (1.2049, 0.84944),
    (1.22, 0.79859),
)
EXPECTED_CHARACTERISTIC_DIGEST = (
    "f7e1864476a243366aac8721a41aa09e4b909025414428d59b4ae7df9683b8af"
)
EXPECTED_CHARACTERISTIC_SIZE = 458
EXPECTED_SETTINGS = {
    "max_iter": 50,
    "min_iter": 4,
    "init_previous": False,
    "use_cuda": False,
    "print_results": False,
    "robust_relax": False,
    "oscillation_damping": False,
    "skip_postprocess": False,
}


def _characteristic_bytes(**overrides) -> bytes:
    payload = {
        "schema_version": "1.0",
        "characteristic_set_id": "openpinch-single-stage-compressor-v1",
        "abscissa": "relative_mass_flow",
        "ordinate": "relative_isentropic_efficiency",
        "points": [[0.5, 0.8], [1.0, 1.0]],
    }
    payload.update(overrides)
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _tespy_context(
    *,
    mode: str = "heat_pump",
    refrigerant: str = "R134a",
    loads: tuple[float, ...] = (0.7,),
):
    return build_hpr_map_generation_context(
        _basis(
            mode=mode,
            simulation_backend="tespy",
            refrigerant_spec=refrigerant,
        ),
        _request(
            source_temperatures=[8.0],
            sink_temperatures=[51.0],
            load_fractions=loads,
        ),
    )


def test_tespy_convergence_settings_are_exact_frozen_and_schema_private():
    settings = TESPY_CONVERGENCE_SETTINGS

    assert settings.identifier == "openpinch-tespy-hpr-convergence-v1"
    assert settings.solve_arguments == EXPECTED_SETTINGS
    assert settings.as_provenance() == {
        "identifier": settings.identifier,
        "solve_arguments": EXPECTED_SETTINGS,
    }
    with pytest.raises(TypeError):
        settings.solve_arguments["max_iter"] = 20
    with pytest.raises(FrozenInstanceError):
        settings.identifier = "changed"

    assert tuple(inspect.signature(TespyHprPointSimulator.prepare).parameters) == (
        "self",
        "context",
    )
    assert tuple(inspect.signature(TespyHprPointSimulator.simulate).parameters) == (
        "self",
        "point",
    )


def test_owned_compressor_characteristic_has_exact_canonical_bytes_and_digest():
    content = read_tespy_compressor_characteristic_bytes()
    characteristic = load_tespy_compressor_characteristic()

    assert CHARACTERISTIC_RESOURCE_NAME == ("openpinch-single-stage-compressor-v1.json")
    assert len(content) == EXPECTED_CHARACTERISTIC_SIZE
    assert content.endswith(b"\n")
    assert hashlib.sha256(content).hexdigest() == EXPECTED_CHARACTERISTIC_DIGEST
    assert characteristic.schema_version == "1.0"
    assert (
        characteristic.characteristic_set_id == "openpinch-single-stage-compressor-v1"
    )
    assert characteristic.abscissa == "relative_mass_flow"
    assert characteristic.ordinate == "relative_isentropic_efficiency"
    assert characteristic.points == EXPECTED_CHARACTERISTIC_POINTS
    assert characteristic.sha256 == EXPECTED_CHARACTERISTIC_DIGEST
    with pytest.raises(FrozenInstanceError):
        characteristic.sha256 = "changed"


@pytest.mark.parametrize(
    "payload",
    [
        b"{}\n",
        b'{"abscissa":"relative_mass_flow","characteristic_set_id":'
        b'"openpinch-single-stage-compressor-v1","extra":true,'
        b'"ordinate":"relative_isentropic_efficiency","points":'
        b'[[0.5,0.8],[1.0,1.0]],"schema_version":"1.0"}\n',
        b'{"abscissa":"relative_mass_flow","characteristic_set_id":'
        b'"openpinch-single-stage-compressor-v1","ordinate":'
        b'"relative_isentropic_efficiency","points":'
        b'[[1.0,1.0],[0.5,0.8]],"schema_version":"1.0"}\n',
        b'{"abscissa":"relative_mass_flow","characteristic_set_id":'
        b'"openpinch-single-stage-compressor-v1","ordinate":'
        b'"relative_isentropic_efficiency","points":'
        b'[[0.5,0.8],[1.0,-1.0]],"schema_version":"1.0"}\n',
    ],
)
def test_characteristic_parser_rejects_noncanonical_or_invalid_content(payload):
    with pytest.raises(ValueError, match="characteristic"):
        parse_tespy_compressor_characteristic(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"{invalid}\n",
        b'{"abscissa":"relative_mass_flow","abscissa":'
        b'"relative_mass_flow","characteristic_set_id":'
        b'"openpinch-single-stage-compressor-v1","ordinate":'
        b'"relative_isentropic_efficiency","points":'
        b'[[0.5,0.8],[1.0,1.0]],"schema_version":"1.0"}\n',
        _characteristic_bytes(schema_version="2.0"),
        _characteristic_bytes(characteristic_set_id="other"),
        _characteristic_bytes(abscissa="load"),
        _characteristic_bytes(ordinate="cop"),
        _characteristic_bytes(points=[]),
        _characteristic_bytes(points=[[0.5, 0.8]]),
        _characteristic_bytes(points=[[0.5], [1.0, 1.0]]),
        _characteristic_bytes(points=[[False, 0.8], [1.0, 1.0]]),
        _characteristic_bytes(points=[[0.5, 0.8], [float("inf"), 1.0]]),
    ],
)
def test_characteristic_parser_rejects_all_remaining_schema_violations(payload):
    with pytest.raises(ValueError, match="characteristic"):
        parse_tespy_compressor_characteristic(payload)


def test_tespy_compatibility_helpers_cover_old_and_new_public_result_shapes(
    monkeypatch,
):
    assert tespy_adapter._finite_float("3.5") == 3.5
    assert tespy_adapter._finite_float(object()) is None
    assert tespy_adapter._finite_float(math.nan) is None
    assert tespy_adapter._result_value(SimpleNamespace(val_SI=None, val=4.0)) == 4.0
    assert (
        tespy_adapter._iteration_count(
            SimpleNamespace(problem=SimpleNamespace(iter=None), iter=6)
        )
        == 6
    )
    assert (
        tespy_adapter._iteration_count(
            SimpleNamespace(problem=SimpleNamespace(iter=None), iter=None)
        )
        is None
    )
    assert tespy_adapter._network_converged(SimpleNamespace(converged=False)) is False
    assert (
        tespy_adapter._network_converged(
            SimpleNamespace(converged=True, problem=SimpleNamespace(lin_dep=True))
        )
        is False
    )
    assert (
        tespy_adapter._network_converged(
            SimpleNamespace(converged=True, problem=SimpleNamespace(), lin_dep=False)
        )
        is True
    )

    monkeypatch.setattr(
        tespy_adapter,
        "version",
        lambda distribution: (_ for _ in ()).throw(
            tespy_adapter.PackageNotFoundError(distribution)
        ),
    )
    assert tespy_adapter._tespy_version() == "unknown"


def test_tespy_saturation_pressure_validation_rejects_nonfinite_state(
    monkeypatch,
):
    class InvalidState:
        def update(self, *args):
            return None

        def p(self):
            return math.nan

    monkeypatch.setattr(
        tespy_adapter,
        "build_coolprop_abstract_state",
        lambda fluid: InvalidState(),
    )

    with pytest.raises(ValueError, match="pressures"):
        TespyHprPointSimulator._saturation_pressures(_tespy_context(), 5.0, 55.0)


@pytest.mark.tespy
@pytest.mark.parametrize("mode", ["heat_pump", "refrigeration"])
@pytest.mark.parametrize(
    "refrigerant",
    [
        "R134a",
        "R407C",
        "HEOS::R32[0.5]&R125[0.5]",
        "HEOS::R32[0.3]&R125[0.3]&R143a[0.4]",
    ],
)
def test_real_tespy_design_and_offdesign_support_approved_fluid_categories(
    mode,
    refrigerant,
):
    context = _tespy_context(mode=mode, refrigerant=refrigerant)
    point = next(iter_hpr_operating_points(context))
    simulator = TespyHprPointSimulator()

    metadata = simulator.prepare(context)
    result = simulator.simulate(point)
    snapshot = simulator._design_state_path

    assert metadata.backend == "tespy"
    assert metadata.design_converged is True
    assert metadata.characteristic_set_id == context.characteristic_set_id
    assert metadata.power_boundary == "compressor_only"
    assert metadata.modeled_auxiliaries == ()
    assert metadata.design_details["convergence_settings"] == (
        TESPY_CONVERGENCE_SETTINGS.as_provenance()
    )
    assert metadata.design_details["characteristic_sha256"] == (
        EXPECTED_CHARACTERISTIC_DIGEST
    )
    assert all(
        math.isfinite(value) and value > 0.0
        for value in (result.q_source, result.q_sink, result.compressor_power)
    )
    assert result.converged is True
    assert result.q_sink == pytest.approx(
        result.q_source + result.compressor_power,
        abs=context.energy_balance_tolerance,
    )
    expected_useful = result.q_sink if mode == "heat_pump" else result.q_source
    assert expected_useful == pytest.approx(
        point.requested_useful_duty,
        abs=context.energy_balance_tolerance,
    )
    assert isinstance(snapshot, Path)
    assert snapshot.is_file()

    simulator.close()
    assert not snapshot.parent.exists()


@pytest.mark.tespy
def test_every_point_restores_the_design_snapshot_and_is_order_independent():
    context = _tespy_context(loads=(0.55, 0.85))
    first, second = tuple(iter_hpr_operating_points(context))
    simulator = TespyHprPointSimulator()
    simulator.prepare(context)

    first_result = simulator.simulate(first)
    simulator.simulate(second)
    repeated_result = simulator.simulate(first)

    assert repeated_result.q_source == pytest.approx(first_result.q_source, rel=1e-10)
    assert repeated_result.q_sink == pytest.approx(first_result.q_sink, rel=1e-10)
    assert repeated_result.compressor_power == pytest.approx(
        first_result.compressor_power,
        rel=1e-10,
    )
    assert simulator._design_restore_count == 3
    simulator.close()


def test_missing_tespy_dependency_is_typed_and_never_falls_back(monkeypatch):
    module_name = "OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    original_import = builtins.__import__

    def block_tespy(name, *args, **kwargs):
        if name == "tespy" or name.startswith("tespy."):
            raise ModuleNotFoundError("blocked optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_tespy)
    monkeypatch.setattr(
        VapourCompressionCycle,
        "solve",
        lambda *args, **kwargs: pytest.fail("CoolProp fallback was invoked"),
    )

    with pytest.raises(HprSimulatorFailure) as raised:
        get_hpr_point_simulator("tespy")

    assert raised.value.code == "dependency_unavailable"
    assert raised.value.session_fatal is True


@pytest.mark.tespy
def test_real_tespy_wrapper_limitation_is_typed_without_fluid_allowlist():
    context = _tespy_context()
    unsupported = HprWorkingFluidSpec(
        source_spec="NOT_A_BACKEND::NOT_A_FLUID",
        property_backend="NOT_A_BACKEND",
        kind="pure",
        registered_name="NOT_A_FLUID",
        components=("NOT_A_FLUID",),
        mole_fractions=(1.0,),
        composition_basis="not_applicable",
    )
    simulator = TespyHprPointSimulator()

    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.prepare(replace(context, working_fluid=unsupported))

    assert raised.value.code == "unsupported_working_fluid"
    assert raised.value.session_fatal is True
    simulator.close()


@pytest.mark.tespy
def test_unmodeled_internal_heat_exchanger_fails_as_an_unsupported_model():
    context = build_hpr_map_generation_context(
        _basis(
            simulation_backend="tespy",
            internal_hx_gas_temperature_change=3.0,
        ),
        _request(),
    )
    simulator = TespyHprPointSimulator()

    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.prepare(context)

    assert raised.value.code == "unsupported_model"
    assert raised.value.session_fatal is True
    simulator.close()


@pytest.mark.tespy
def test_tespy_adapter_rejects_wrong_lifecycle_and_backend():
    simulator = TespyHprPointSimulator()
    point = next(iter_hpr_operating_points(_tespy_context()))

    with pytest.raises(HprSimulatorFailure, match="prepared state"):
        simulator.simulate(point)
    with pytest.raises(HprSimulatorFailure, match="different backend"):
        simulator.prepare(build_hpr_map_generation_context(_basis(), _request()))
    simulator.close()
    with pytest.raises(HprSimulatorFailure, match="closed more than once"):
        simulator.close()

    prepared = TespyHprPointSimulator()
    prepared.prepare(_tespy_context())
    with pytest.raises(HprSimulatorFailure, match="prepared once"):
        prepared.prepare(_tespy_context())
    prepared.close()


@pytest.mark.tespy
@pytest.mark.parametrize(
    ("failure_stage", "expected_code"),
    [
        ("design_exception", "prepare_failed"),
        ("design_nonconvergence", "non_converged"),
        ("save_exception", "prepare_failed"),
        ("save_missing", "prepare_failed"),
    ],
)
def test_tespy_design_and_snapshot_failures_are_typed_and_cleanable(
    monkeypatch,
    failure_stage,
    expected_code,
):
    simulator = TespyHprPointSimulator()
    if failure_stage == "design_exception":
        monkeypatch.setattr(
            TespyHprPointSimulator,
            "_solve",
            lambda self, mode, **paths: (_ for _ in ()).throw(
                RuntimeError("design solve failed")
            ),
        )
    elif failure_stage == "design_nonconvergence":
        monkeypatch.setattr(tespy_adapter, "_network_converged", lambda network: False)
    elif failure_stage == "save_exception":
        monkeypatch.setattr(
            tespy_adapter.Network,
            "save",
            lambda self, path: (_ for _ in ()).throw(OSError("snapshot write failed")),
        )
    else:
        monkeypatch.setattr(tespy_adapter.Network, "save", lambda self, path: None)

    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.prepare(_tespy_context())

    assert raised.value.code == expected_code
    assert raised.value.session_fatal is True
    simulator.close()


@pytest.mark.tespy
def test_tespy_point_state_and_engine_failures_are_classified(monkeypatch):
    context = _tespy_context()
    point = next(iter_hpr_operating_points(context))

    state_failure = TespyHprPointSimulator()
    state_failure.prepare(context)
    monkeypatch.setattr(
        state_failure,
        "_apply_condition",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("state failed")),
    )
    with pytest.raises(HprSimulatorFailure) as raised:
        state_failure.simulate(point)
    assert raised.value.code == "unsupported_working_fluid"
    assert raised.value.session_fatal is False
    state_failure.close()

    solve_failure = TespyHprPointSimulator()
    solve_failure.prepare(context)
    monkeypatch.setattr(
        solve_failure,
        "_solve",
        lambda mode, **paths: (_ for _ in ()).throw(RuntimeError("offdesign failed")),
    )
    with pytest.raises(HprSimulatorFailure) as raised:
        solve_failure.simulate(point)
    assert raised.value.code == "point_exception"
    assert raised.value.session_fatal is False
    solve_failure.close()


@pytest.mark.tespy
def test_tespy_offdesign_restore_and_nonfinite_result_failures(monkeypatch):
    context = _tespy_context()
    point = next(iter_hpr_operating_points(context))

    restore_failure = TespyHprPointSimulator()
    restore_failure.prepare(context)
    monkeypatch.setattr(
        restore_failure,
        "_solve",
        lambda mode, **paths: (_ for _ in ()).throw(
            FileNotFoundError("snapshot disappeared")
        ),
    )
    with pytest.raises(HprSimulatorFailure) as raised:
        restore_failure.simulate(point)
    assert raised.value.code == "restore_failed"
    assert raised.value.session_fatal is True
    restore_failure.close()

    invalid_result = TespyHprPointSimulator()
    invalid_result.prepare(context)
    monkeypatch.setattr(invalid_result, "_solve", lambda mode, **paths: None)
    monkeypatch.setattr(tespy_adapter, "_network_converged", lambda network: True)
    monkeypatch.setattr(tespy_adapter, "_result_value", lambda container: None)
    with pytest.raises(HprSimulatorFailure) as raised:
        invalid_result.simulate(point)
    assert raised.value.code == "non_converged"
    assert raised.value.session_fatal is False
    invalid_result.close()


@pytest.mark.tespy
def test_offdesign_nonconvergence_is_a_typed_point_failure(monkeypatch):
    import OpenPinch.analysis.heat_pumps.performance_maps.adapters.tespy as adapter

    context = _tespy_context()
    point = next(iter_hpr_operating_points(context))
    simulator = TespyHprPointSimulator()
    simulator.prepare(context)

    monkeypatch.setattr(adapter, "_network_converged", lambda network: False)

    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.simulate(point)

    assert raised.value.code == "non_converged"
    assert raised.value.session_fatal is False
    simulator.close()


@pytest.mark.tespy
def test_design_restoration_failure_is_session_fatal_and_cleanup_still_works(
    monkeypatch,
):
    context = _tespy_context()
    point = next(iter_hpr_operating_points(context))
    simulator = TespyHprPointSimulator()
    simulator.prepare(context)
    snapshot = simulator._design_state_path

    snapshot.unlink()
    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.simulate(point)

    assert raised.value.code == "restore_failed"
    assert raised.value.session_fatal is True
    assert str(snapshot) not in str(raised.value)
    simulator.close()
    assert not snapshot.parent.exists()
