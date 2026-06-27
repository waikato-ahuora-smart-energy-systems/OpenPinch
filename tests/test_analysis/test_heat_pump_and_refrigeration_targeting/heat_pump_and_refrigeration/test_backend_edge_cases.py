"""Edge-path tests for simulated HPR backend wrappers."""

import numpy as np

from OpenPinch.lib.schemas.hpr import HPRBackendResult, HPRParsedState
from OpenPinch.services.heat_pump_integration.targeting_services import (
    cascade_carnot,
    cascade_vapour_compression,
    parallel_vapour_compression,
)

from ..helpers import _base_args


def _state() -> HPRParsedState:
    return HPRParsedState(
        T_cond=np.array([80.0]),
        T_evap=np.array([20.0]),
        dT_subcool=np.array([2.0]),
        dT_superheat=np.array([5.0]),
        dT_ihx_gas_side=np.array([5.0]),
        Q_heat_base=10.0,
        Q_cool_base=10.0,
        x_heat_split=np.array([1.0]),
        x_cool_split=np.array([1.0]),
        Q_heat_available=np.array([10.0]),
        Q_cool_available=np.array([10.0]),
        Q_amb_hot=2.0,
        Q_amb_cold=3.0,
    )


def test_cascade_carnot_backend_validates_mapping_state(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        cascade_carnot,
        "_parse_cascade_carnot_cycle_state_variables",
        lambda *_args, **_kwargs: state.model_dump(),
    )

    class FakeCycle:
        work = 1.0
        w_hpr = 1.0
        w_he = 0.0
        heat_recovery = 0.0
        COP_h = 2.0
        Q_cond = 10.0
        Q_evap = 5.0
        Q_cond_he = 0.0
        Q_evap_he = 0.0
        penalty = []

        def solve(self, **_kwargs):
            return None

    monkeypatch.setattr(cascade_carnot, "CascadeCarnotCycle", FakeCycle)
    monkeypatch.setattr(
        cascade_carnot,
        "evaluate_carnot_hpr_result",
        lambda **kwargs: HPRBackendResult.failure(
            reason="evaluated",
            Q_amb_hot=kwargs["state"].Q_amb_hot,
            Q_amb_cold=kwargs["state"].Q_amb_cold,
        ),
    )

    result = cascade_carnot._compute_cascade_carnot_cycle_obj(
        np.array([0.0]),
        _base_args(),
    )

    assert result.success is False
    assert result.Q_amb_hot == 2.0
    assert result.Q_amb_cold == 3.0


def test_cascade_vapour_backend_returns_failure_when_cycle_raises(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        cascade_vapour_compression,
        "_parse_cascade_hp_state_variables",
        lambda *_args, **_kwargs: state,
    )

    class RaisingCycle:
        def solve(self, **_kwargs):
            raise RuntimeError("cycle failed")

    monkeypatch.setattr(
        cascade_vapour_compression,
        "CascadeVapourCompressionCycle",
        RaisingCycle,
    )

    result = cascade_vapour_compression._compute_cascade_hp_system_obj(
        np.array([0.0]),
        _base_args(),
    )

    assert result.success is False
    assert result.failure_reason == "cycle failed"
    assert result.Q_amb_hot == 2.0


def test_parallel_vapour_backend_failure_paths(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        parallel_vapour_compression,
        "_parse_parallel_hp_state_temperatures",
        lambda *_args, **_kwargs: state,
    )

    class UnsolvedCycle:
        solved = False

        def solve(self, **_kwargs):
            return None

    monkeypatch.setattr(
        parallel_vapour_compression,
        "ParallelVapourCompressionCycles",
        UnsolvedCycle,
    )
    result = parallel_vapour_compression._compute_parallel_hp_system_obj(
        np.array([0.0]),
        _base_args(),
    )

    assert result.success is False
    assert "failed to solve" in result.failure_reason

    class RaisingCycle:
        def solve(self, **_kwargs):
            raise RuntimeError("parallel failed")

    monkeypatch.setattr(
        parallel_vapour_compression,
        "ParallelVapourCompressionCycles",
        RaisingCycle,
    )
    result = parallel_vapour_compression._compute_parallel_hp_system_obj(
        np.array([0.0]),
        _base_args(),
    )

    assert result.success is False
    assert result.failure_reason == "parallel failed"
