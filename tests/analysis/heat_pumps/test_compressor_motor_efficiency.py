"""Motor efficiency of the vapour compression compressors."""

from types import SimpleNamespace

import numpy as np
import pytest

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
from OpenPinch.analysis.heat_pumps.performance_maps.target_records import (
    build_coolprop_target_simulation_record,
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
from OpenPinch.contracts.hpr import HPRParsedState, HPRTopologyIdentifier
from OpenPinch.domain.configuration import Configuration

from .helpers import _base_args

pytest.importorskip("CoolProp")


def _solve_single(**kwargs):
    cycle = VapourCompressionCycle()
    work = cycle.solve(
        T_evap=10.0,
        T_cond=60.0,
        dtcont=0.0,
        dT_superheat=5.0,
        dT_ihx_gas_side=0.0,
        eta_comp=0.7,
        refrigerant="AMMONIA",
        Q_heat=1000.0,
        **kwargs,
    )
    return cycle, work


def test_motor_losses_raise_the_electrical_work_and_leave_the_duties_unchanged():
    shaft, shaft_work = _solve_single()
    cycle, work = _solve_single(eta_motor=0.8)

    assert shaft.eta_motor == 1.0
    assert cycle.eta_motor == 0.8
    assert work == pytest.approx(shaft_work / 0.8, rel=1e-12)
    assert cycle.work == pytest.approx(work, rel=1e-12)
    assert cycle.w_net == pytest.approx(shaft.w_net / 0.8, rel=1e-12)
    assert cycle.Q_cond == pytest.approx(shaft.Q_cond, rel=1e-12)
    assert cycle.Q_evap == pytest.approx(shaft.Q_evap, rel=1e-12)
    assert cycle.COP_h == pytest.approx(cycle.Q_cond / work, rel=1e-12)


@pytest.mark.parametrize("eta_motor", [0.0, -0.5, 1.2, float("nan")])
def test_motor_efficiency_must_be_in_the_unit_interval(eta_motor):
    with pytest.raises(ValueError, match="eta_motor"):
        _solve_single(eta_motor=eta_motor)


def test_parallel_cycles_apply_the_motor_efficiency_to_every_cycle():
    def solve(**kwargs):
        cycle = ParallelVapourCompressionCycles()
        work = cycle.solve(
            T_evap=np.array([20.0, 10.0]),
            T_cond=np.array([80.0, 60.0]),
            dtcont=0.0,
            refrigerant="AMMONIA",
            Q_heat=np.array([500.0, 500.0]),
            **kwargs,
        )
        return cycle, work

    shaft, shaft_work = solve()
    cycle, work = solve(eta_motor=0.8)

    assert work == pytest.approx(shaft_work / 0.8, rel=1e-10)
    assert cycle.Q_cond == pytest.approx(shaft.Q_cond, rel=1e-10)


def test_cascade_cycles_apply_the_motor_efficiency_to_every_stage():
    def solve(**kwargs):
        cycle = CascadeVapourCompressionCycle()
        work = cycle.solve(
            T_evap=np.array([10.0]),
            T_cond=np.array([100.0, 70.0]),
            dtcont=0.0,
            refrigerant="AMMONIA",
            dt_cascade_hx=5.0,
            Q_heat=np.array([600.0, 400.0]),
            **kwargs,
        )
        return cycle, work

    shaft, shaft_work = solve()
    cycle, work = solve(eta_motor=0.8)

    assert shaft.solved and cycle.solved
    assert work == pytest.approx(shaft_work / 0.8, rel=1e-10)
    assert cycle.Q_cond == pytest.approx(shaft.Q_cond, rel=1e-10)


def _target_inputs(options):
    return hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -50.0, -100.0]),
        H_cold=np.array([100.0, 50.0, 0.0]),
        config=Configuration(options=options),
    )


def test_target_inputs_carry_separate_vapour_compression_and_mvr_motor_efficiencies():
    defaults = _target_inputs({})
    args = _target_inputs({"HPR_ETA_MOTOR": 0.9, "HPR_MVR_ETA_MOTOR": 0.97})

    assert defaults.eta_motor == pytest.approx(0.95)
    assert defaults.eta_mvr_motor == pytest.approx(0.95)
    assert args.eta_motor == pytest.approx(0.9)
    assert args.eta_mvr_motor == pytest.approx(0.97)


def test_vc_mvr_cascade_applies_the_vapour_compression_motor_efficiency_to_its_vc_stages():
    def solve(**kwargs):
        cascade = VapourCompressionMvrCascade()
        cascade.solve(
            T_evap_vc=np.array([20.0, 30.0]),
            T_cond_vc=np.array([90.0, 80.0]),
            dT_lift_mvr=np.array([15.0, 12.0]),
            Q_heat_vc=np.array([1000.0, 500.0]),
            mvr_source_split=0.25,
            mvr_process_split=np.array([0.4]),
            dT_subcool_vc=np.array([0.0, 0.0]),
            dT_subcool_mvr=np.array([4.0, 3.0]),
            dT_ihx_gas_side_vc=np.array([0.0, 0.0]),
            eta_comp=0.75,
            eta_mvr_comp=0.75,
            eta_motor=0.9,
            refrigerant=["R134A", "R134A"],
            dt_cascade_hx=5.0,
            **kwargs,
        )
        return cascade

    shaft = solve()
    cascade = solve(eta_vc_motor=0.8)

    assert [cycle.eta_motor for cycle in cascade.vc_cycles] == [0.8, 0.8]
    assert [cycle.eta_motor for cycle in cascade.mvr_cycles] == [0.9, 0.9]
    vc_work = sum(cycle.work for cycle in cascade.vc_cycles)
    shaft_vc_work = sum(cycle.work for cycle in shaft.vc_cycles)
    assert vc_work == pytest.approx(shaft_vc_work / 0.8, rel=1e-10)


class _RecordingCycle:
    calls = []
    solved = False
    dT_subcool = np.zeros(1)

    def solve(self, **kwargs):
        type(self).calls.append(kwargs)


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
def test_vapour_compression_targeting_passes_the_motor_efficiency_to_the_cycles(
    monkeypatch, module, builder, cycle_class
):
    _RecordingCycle.calls = []
    monkeypatch.setattr(module, cycle_class, _RecordingCycle)
    state = SimpleNamespace(
        T_evap=np.array([10.0]),
        T_cond=np.array([60.0]),
        dT_subcool=np.zeros(1),
        dT_ihx_gas_side=np.zeros(1),
        Q_heat_base=None,
        x_heat_split=None,
        Q_heat_available=None,
        Q_cool_base=None,
        x_cool_split=None,
        Q_cool_available=None,
    )

    getattr(module, builder)(state, _target_inputs({"HPR_ETA_MOTOR": 0.9}), True)

    (call,) = _RecordingCycle.calls
    assert call["eta_motor"] == pytest.approx(0.9)


class _CascadeSolved(Exception):
    pass


def test_vc_mvr_targeting_passes_both_motor_efficiencies_to_the_cascade(monkeypatch):
    seen = {}

    class RecordingCascade:
        def solve(self, **kwargs):
            seen.update(kwargs)
            raise _CascadeSolved

    state = HPRParsedState(T_evap=np.array([10.0]), T_cond=np.array([60.0]))
    monkeypatch.setattr(
        vc_mvr_targeting, "VapourCompressionMvrCascade", RecordingCascade
    )
    monkeypatch.setattr(
        vc_mvr_targeting, "_parse_vc_mvr_state_variables", lambda x, args: state
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
    args = _target_inputs({"HPR_ETA_MOTOR": 0.9, "HPR_MVR_ETA_MOTOR": 0.97})

    with pytest.raises(_CascadeSolved):
        vc_mvr_targeting._compute_vc_mvr_system_obj(np.zeros(1), args)

    assert seen["eta_vc_motor"] == pytest.approx(0.9)
    assert seen["eta_motor"] == pytest.approx(0.97)


def test_simulation_records_give_each_stage_its_motor_efficiency():
    args = _base_args(
        n_cond=1,
        n_evap=1,
        n_mvr=1,
        refrigerant_ls=["R134a"],
        eta_mvr_comp=0.7,
        eta_motor=0.9,
        eta_mvr_motor=0.97,
        is_heat_pumping=True,
    )
    cycle = SimpleNamespace(
        Q_heat_arr=np.array([5.0, 10.0]),
        Q_cool_arr=np.array([4.0, 0.0]),
        work_arr=np.array([1.0, 2.0]),
        T_evap=np.array([20.0, 80.0]),
        T_cond=np.array([80.0, 100.0]),
    )

    record = build_coolprop_target_simulation_record(
        args=args,
        state=HPRParsedState(T_evap=cycle.T_evap, T_cond=cycle.T_cond),
        cycle=cycle,
        topology_id=HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
    )

    motors = [loop.stages[0].motor_efficiency for loop in record.loops]
    assert motors == [0.9, 0.97]


def test_motor_efficiency_survives_the_ihx_and_subcooling_resolve():
    # Subcooling of 80 K cannot be met without process duty, so the cycle
    # re-solves with the subcooling it can apply.
    def stage(**kwargs):
        cycle = VapourCompressionCycle()
        work = cycle.solve(
            T_evap=0.0,
            T_cond=80.0,
            dtcont=5.0,
            dT_subcool=80.0,
            eta_comp=0.7,
            refrigerant="Ammonia",
            dT_ihx_gas_side=30.0,
            Q_heat=0.0,
            Q_cas_heat=100.0,
            **kwargs,
        )
        return cycle, work

    shaft, shaft_work = stage()
    cycle, work = stage(eta_motor=0.8)

    assert cycle.dT_subcool == pytest.approx(0.0, abs=1e-6)
    assert cycle.eta_motor == 0.8
    assert work == pytest.approx(shaft_work / 0.8, rel=1e-12)
