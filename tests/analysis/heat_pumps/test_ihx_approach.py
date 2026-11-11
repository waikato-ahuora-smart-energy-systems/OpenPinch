"""The internal heat exchanger keeps a minimum approach at its hot end.

The suction gas leaving the IHX can get no warmer than the liquid entering it
less ``dt_ihx``, so the IHX cannot raise the suction to the condensing
temperature through an exchanger of unbounded area.
"""

from __future__ import annotations

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle as cascade_cycle_module
import OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles as parallel_cycle_module
import OpenPinch.analysis.heat_pumps.cycles.vapour_compression_mvr_cascade as vc_mvr_cycle_module
from OpenPinch.analysis.heat_pumps.common import preprocessing as hp_pre
from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
    CascadeVapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_mvr_cascade import (
    VapourCompressionMvrCascade,
)
from OpenPinch.analysis.heat_pumps.targeting import (
    cascade_vapour_compression as cascade_targeting,
)
from OpenPinch.analysis.heat_pumps.targeting import (
    parallel_vapour_compression as parallel_targeting,
)
from OpenPinch.analysis.heat_pumps.targeting import (
    vapour_compression_mvr as vc_mvr_targeting,
)
from OpenPinch.contracts.hpr import HPRParsedState
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS

pytest.importorskip("CoolProp")


def _solve(*, dtcont=0.0, dt_ihx=0.0, dT_subcool=0.0):
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=5.0,
        T_cond=80.0,
        dtcont=dtcont,
        dt_ihx=dt_ihx,
        dT_subcool=dT_subcool,
        refrigerant="R1234ze(E)",
        dT_ihx_gas_side=100.0,
        Q_heat=1000.0,
    )
    assert cycle.solved
    return cycle


def test_ihx_suction_stays_dt_ihx_below_the_liquid_entering_it():
    cycle = _solve(dt_ihx=5.0, dT_subcool=10.0)

    # Liquid enters the IHX at 80 - 10 = 70 degC; the gas may reach 65 degC.
    assert cycle.dT_ihx_gas_side == pytest.approx(70.0 - 5.0 - 5.0)


def test_ihx_approach_holds_after_a_cascade_stage_settles_its_subcooling():
    # A small process duty beside cascade heat cannot take the requested
    # subcooling, so the stage re-solves with less; the approach must hold.
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=0.0,
        T_cond=80.0,
        dtcont=5.0,
        dt_ihx=5.0,
        dT_subcool=78.0,
        refrigerant="R245fa",
        dT_ihx_gas_side=200.0,
        Q_heat=5.0,
        Q_cas_heat=100.0,
    )

    assert cycle.dT_subcool < 78.0
    assert cycle.dT_ihx_gas_side == pytest.approx(80.0 - cycle.dT_subcool - 5.0)


def test_working_fluid_approach_does_not_limit_the_ihx():
    cycle = _solve(dtcont=3.0, dt_ihx=0.0)

    assert cycle.dT_ihx_gas_side == pytest.approx(80.0 - 5.0)


def _recording_stage_class(seen):
    class RecordingStage(VapourCompressionCycle):
        def solve(self, *args, **kwargs):
            seen.append(kwargs.get("dt_ihx"))
            return super().solve(*args, **kwargs)

    return RecordingStage


def _solve_cascade():
    CascadeVapourCompressionCycle().solve(
        T_evap=np.array([5.0]),
        T_cond=np.array([80.0, 45.0]),
        dtcont=0.0,
        dt_ihx=5.0,
        dT_ihx_gas_side=100.0,
        refrigerant="R1234ze(E)",
        dt_cascade_hx=5.0,
        Q_heat=np.array([100.0, 50.0]),
    )


def _solve_parallel():
    ParallelVapourCompressionCycles().solve(
        T_evap=np.array([5.0, 10.0]),
        T_cond=np.array([80.0, 45.0]),
        dtcont=0.0,
        dt_ihx=5.0,
        dT_ihx_gas_side=100.0,
        refrigerant="R1234ze(E)",
        Q_heat=np.array([100.0, 50.0]),
    )


def _solve_vc_mvr():
    VapourCompressionMvrCascade().solve(
        T_evap_vc=np.array([20.0, 30.0]),
        T_cond_vc=np.array([90.0, 80.0]),
        dT_lift_mvr=np.array([15.0, 12.0]),
        Q_heat_vc=np.array([1000.0, 500.0]),
        mvr_source_split=0.25,
        mvr_process_split=np.array([0.4]),
        dT_ihx_gas_side_vc=np.array([100.0, 100.0]),
        dt_ihx=5.0,
        refrigerant=["R134A", "R134A"],
        dt_cascade_hx=5.0,
    )


@pytest.mark.parametrize(
    ("module", "solve"),
    [
        (cascade_cycle_module, _solve_cascade),
        (parallel_cycle_module, _solve_parallel),
        (vc_mvr_cycle_module, _solve_vc_mvr),
    ],
)
def test_multi_stage_cycles_pass_the_ihx_approach_to_every_stage(
    monkeypatch, module, solve
):
    seen = []
    monkeypatch.setattr(module, "VapourCompressionCycle", _recording_stage_class(seen))

    solve()

    assert seen
    assert seen == [5.0] * len(seen)


def _target_inputs(options):
    return hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -50.0, -100.0]),
        H_cold=np.array([100.0, 50.0, 0.0]),
        config=Configuration(options=options),
    )


class _RecordingSolve(Exception):
    pass


def _recording_cycle_class(seen):
    class RecordingCycle:
        def solve(self, **kwargs):
            seen.update(kwargs)
            raise _RecordingSolve

    return RecordingCycle


_STATE = HPRParsedState(
    T_evap=np.array([10.0]),
    T_cond=np.array([60.0]),
    dT_subcool=np.zeros(1),
    dT_ihx_gas_side=np.zeros(1),
)


@pytest.mark.parametrize(
    ("module", "builder", "cycle_class"),
    [
        (
            parallel_targeting,
            "_build_and_solve_parallel_cycles",
            "ParallelVapourCompressionCycles",
        ),
        (
            cascade_targeting,
            "_build_and_solve_cascade_cycle",
            "CascadeVapourCompressionCycle",
        ),
    ],
)
def test_vapour_compression_targeting_passes_the_ihx_approach_to_the_cycles(
    monkeypatch, module, builder, cycle_class
):
    seen = {}
    monkeypatch.setattr(module, cycle_class, _recording_cycle_class(seen))

    with pytest.raises(_RecordingSolve):
        getattr(module, builder)(
            _STATE.model_copy(), _target_inputs({"HPR_DT_IHX": 7.0}), True
        )

    assert seen.get("dt_ihx") == pytest.approx(7.0)


def test_vc_mvr_targeting_passes_the_ihx_approach_to_the_cascade(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        vc_mvr_targeting,
        "VapourCompressionMvrCascade",
        _recording_cycle_class(seen),
    )
    monkeypatch.setattr(
        vc_mvr_targeting,
        "_parse_vc_mvr_state_variables",
        lambda x, args: _STATE.model_copy(),
    )
    monkeypatch.setattr(
        vc_mvr_targeting,
        "_unpack_vc_mvr_state",
        lambda state_vars, args: dict.fromkeys(
            (
                "T_evap_vc",
                "T_cond_vc",
                "dT_lift_mvr",
                "mvr_source_split",
                "mvr_process_split",
                "dT_subcool_vc",
                "dT_subcool_mvr",
            )
        ),
    )

    with pytest.raises(_RecordingSolve):
        vc_mvr_targeting._compute_vc_mvr_system_obj(
            np.zeros(1), _target_inputs({"HPR_DT_IHX": 7.0})
        )

    assert seen.get("dt_ihx") == pytest.approx(7.0)


@pytest.mark.parametrize(
    ("key", "default"),
    [("HPR_DT_IHX", 5.0), ("HPR_DT_CASCADE_HX", 5.0), ("HPR_DT_CONT", 1.0)],
)
def test_heat_pump_exchangers_default_to_finite_approaches(key, default):
    assert CONFIG_FIELD_SPECS[key].default == default
