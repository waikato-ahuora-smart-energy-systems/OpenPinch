"""CoolProp HPR point-simulator oracle and compatibility tests."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.performance_maps.adapters.coolprop import (
    CoolPropHprPointSimulator,
)
from OpenPinch.analysis.heat_pumps.performance_maps.context import (
    build_hpr_map_generation_context,
)
from OpenPinch.analysis.heat_pumps.performance_maps.errors import HprSimulatorFailure
from OpenPinch.analysis.heat_pumps.performance_maps.factory import (
    get_hpr_point_simulator,
)
from OpenPinch.analysis.heat_pumps.performance_maps.points import (
    iter_hpr_operating_points,
)
from tests.analysis.heat_pumps.test_hpr_map_generation import _basis, _request


def _context(*, mode="heat_pump", refrigerant="R134a", loads=(1.0,)):
    return build_hpr_map_generation_context(
        _basis(mode=mode, refrigerant_spec=refrigerant),
        _request(
            source_temperatures=[8.0],
            sink_temperatures=[36.0],
            load_fractions=loads,
        ),
    )


def _direct_cycle(context, point):
    cycle = VapourCompressionCycle()
    is_heat_pump = context.basis.mode == "heat_pump"
    cycle.solve(
        T_evap=point.evaporating_temperature,
        T_cond=point.condensing_temperature,
        dtcont=min(
            context.basis.source_approach_temperature,
            context.basis.sink_approach_temperature,
        ),
        dT_superheat=context.basis.superheat,
        dT_subcool=context.basis.subcooling,
        eta_comp=context.basis.compressor_isentropic_efficiency,
        refrigerant=context.working_fluid.source_spec,
        dT_ihx_gas_side=context.basis.internal_hx_gas_temperature_change,
        Q_heat=point.requested_useful_duty * 1_000.0 if is_heat_pump else None,
        Q_cool=None if is_heat_pump else point.requested_useful_duty * 1_000.0,
        is_heat_pump=is_heat_pump,
    )
    return cycle


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
def test_coolprop_point_matches_existing_cycle_oracle(mode, refrigerant):
    context = _context(mode=mode, refrigerant=refrigerant)
    point = next(iter_hpr_operating_points(context))
    simulator = CoolPropHprPointSimulator()

    metadata = simulator.prepare(context)
    result = simulator.simulate(point)
    simulator.close()
    expected = _direct_cycle(context, point)

    assert metadata.backend == "coolprop"
    assert metadata.design_converged is True
    assert metadata.power_boundary == "compressor_only"
    assert metadata.modeled_auxiliaries == ()
    assert result.converged is True
    assert result.q_source == pytest.approx(expected.Q_evap / 1_000.0, rel=1e-10)
    assert result.q_sink == pytest.approx(expected.Q_cond / 1_000.0, rel=1e-10)
    assert result.compressor_power == pytest.approx(
        expected.work / 1_000.0,
        rel=1e-10,
    )


def test_closed_factory_returns_fresh_default_coolprop_sessions():
    first = get_hpr_point_simulator("coolprop")
    second = get_hpr_point_simulator("COOLPROP")

    assert isinstance(first, CoolPropHprPointSimulator)
    assert isinstance(second, CoolPropHprPointSimulator)
    assert first is not second


def test_coolprop_adapter_rejects_wrong_state_backend_and_double_close():
    simulator = CoolPropHprPointSimulator()
    point = next(iter_hpr_operating_points(_context()))

    with pytest.raises(HprSimulatorFailure, match="prepared state"):
        simulator.simulate(point)
    with pytest.raises(HprSimulatorFailure, match="different backend"):
        simulator.prepare(
            build_hpr_map_generation_context(
                _basis(simulation_backend="tespy"),
                _request(),
            )
        )

    prepared = CoolPropHprPointSimulator()
    prepared.prepare(_context())
    with pytest.raises(HprSimulatorFailure, match="prepared once"):
        prepared.prepare(_context())
    prepared.close()
    with pytest.raises(HprSimulatorFailure, match="closed more than once"):
        prepared.close()


def test_coolprop_version_falls_back_when_distribution_metadata_is_missing(
    monkeypatch,
):
    import OpenPinch.analysis.heat_pumps.performance_maps.adapters.coolprop as adapter

    monkeypatch.setattr(
        adapter,
        "version",
        lambda distribution: (_ for _ in ()).throw(
            adapter.PackageNotFoundError(distribution)
        ),
    )

    assert adapter._coolprop_version() == "unknown"


@given(st.sampled_from((0.1, 0.25, 0.5, 0.75, 1.0)))
def test_fixed_temperature_duty_and_power_scale_while_cop_is_constant(load):
    context = _context(loads=(load, 1.0) if load != 1.0 else (1.0,))
    simulator = CoolPropHprPointSimulator()
    simulator.prepare(context)
    results = [
        simulator.simulate(point) for point in iter_hpr_operating_points(context)
    ]
    simulator.close()

    full = results[-1]
    partial = results[0]
    expected_scale = load if load != 1.0 else 1.0
    assert partial.q_sink == pytest.approx(full.q_sink * expected_scale, rel=1e-9)
    assert partial.q_source == pytest.approx(full.q_source * expected_scale, rel=1e-9)
    assert partial.compressor_power == pytest.approx(
        full.compressor_power * expected_scale,
        rel=1e-9,
    )
    assert partial.q_sink / partial.compressor_power == pytest.approx(
        full.q_sink / full.compressor_power,
        rel=1e-9,
    )


def test_cycle_exception_is_a_point_local_typed_failure(monkeypatch):
    context = _context()
    point = next(iter_hpr_operating_points(context))
    simulator = CoolPropHprPointSimulator()
    simulator.prepare(context)

    def fail_solve(self, *args, **kwargs):
        raise ValueError("raw path /private/tmp/engine-state")

    monkeypatch.setattr(VapourCompressionCycle, "solve", fail_solve)

    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.simulate(point)

    assert raised.value.code == "point_exception"
    assert raised.value.session_fatal is False
    assert "/private/tmp" not in str(raised.value)
    simulator.close()


def test_nonfinite_or_unsolved_cycle_is_a_typed_point_failure(monkeypatch):
    context = _context()
    point = next(iter_hpr_operating_points(context))
    simulator = CoolPropHprPointSimulator()
    simulator.prepare(context)

    monkeypatch.setattr(VapourCompressionCycle, "solve", lambda self, *a, **k: math.nan)

    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.simulate(point)

    assert raised.value.code == "non_converged"
    assert raised.value.session_fatal is False
    simulator.close()


@pytest.mark.parametrize("invalid_value", [None, math.inf])
def test_solved_cycle_with_invalid_result_is_nonconverged(
    monkeypatch,
    invalid_value,
):
    context = _context()
    point = next(iter_hpr_operating_points(context))
    simulator = CoolPropHprPointSimulator()
    simulator.prepare(context)

    def invalid_solve(self, *args, **kwargs):
        self._solved = True
        self._Q_evap = invalid_value
        self._Q_cond = 10_000.0
        return 1_000.0

    monkeypatch.setattr(VapourCompressionCycle, "solve", invalid_solve)

    with pytest.raises(HprSimulatorFailure) as raised:
        simulator.simulate(point)

    assert raised.value.code == "non_converged"
    simulator.close()
