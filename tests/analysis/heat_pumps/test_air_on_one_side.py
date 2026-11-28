"""Ambient air is on one side of a heat pump or refrigerator only.

Air is a source for a machine's evaporators or a sink for its condensers.
A machine that draws on air and rejects to air pumps heat from air to air:
that heat is a penalty, as wrong-side heat is, so the design is not a
target.
"""

from __future__ import annotations

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.common.shared as hp_shared
from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
    reject_penalised_design,
)
from OpenPinch.contracts.hpr import HPRBackendResult, HPRParsedState
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle

from .helpers import _base_args, _sc, _stream


def _air_to_air_args(is_heat_pumping):
    # A sink that takes 40 kW at 95-45 C and a process that gives no heat.
    return _base_args(
        T_hot=np.array([15.0, 5.0]),
        H_hot=np.array([0.0, 0.0]),
        T_cold=np.array([95.0, 45.0]),
        H_cold=np.array([40.0, 0.0]),
        z_amb_hot=np.zeros(2),
        z_amb_cold=np.zeros(2),
        Q_heat_max=40.0,
        Q_cool_max=0.0,
        is_heat_pumping=is_heat_pumping,
    )


def _air_to_air_streams():
    # 100 kW of condenser heat at 120-100 C, of which the sink takes 40 kW
    # and air (20 C) the other 60 kW; 50 kW of evaporator at 5 C, all of it
    # from air.
    return _sc(
        _stream("cond", 120.0, 100.0, 100.0, is_process_stream=False),
        _stream("evap", 5.0, 6.0, 50.0, is_process_stream=False),
    )


@pytest.mark.parametrize("is_heat_pumping", [True, False])
def test_a_carnot_machine_with_air_on_both_sides_is_penalised(is_heat_pumping):
    from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
        build_hpr_accounting,
    )

    args = _air_to_air_args(is_heat_pumping)

    result = hp_shared.evaluate_carnot_hpr_result(
        args=args,
        state=HPRParsedState(T_cond=np.array([110.0]), T_evap=np.array([5.0])),
        w_net=50.0,
        w_hpr=np.array([50.0]),
        Q_cond_total=np.array([100.0]),
        Q_evap_total=np.array([50.0]),
    )

    assert (result.Q_amb_hot, result.Q_amb_cold) == pytest.approx((50.0, 60.0))
    # The 50 kW that goes from air to air, as wrong-side heat is penalised.
    *_, penalty, _objective = build_hpr_accounting(
        work=50.0, Q_ext_heat=0.0, Q_ext_cold=0.0, args=args, penalty_terms=[50.0]
    )
    assert penalty > 0.0
    assert result.feasibility_penalty == pytest.approx(penalty)
    assert result.Q_air_to_air == pytest.approx(50.0)


@pytest.mark.parametrize(
    "hpr_type",
    [
        HeatPumpAndRefrigerationCycle.CascadeCarnot.value,
        HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
    ],
)
def test_a_final_design_with_air_on_both_sides_is_not_a_target(hpr_type):
    # A Carnot design keeps a penalty, so the penalty alone does not stop
    # it; the heat it takes from air to air does, for any cycle.
    args = _base_args(hpr_type=hpr_type, is_heat_pumping=False, Q_hpr_target=200.0)
    design = HPRBackendResult(
        obj=0.5,
        utility_tot=0.5,
        w_net=0.5,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=50.0,
        Q_amb_cold=60.0,
        Q_cond=np.array([100.0]),
        Q_evap=np.array([50.0]),
    ).with_updates(Q_air_to_air=50.0)

    rejected = reject_penalised_design(design, args)

    assert not rejected.success
    assert "air on both sides" in rejected.failure_reason


def test_a_vapour_refrigerator_with_air_on_both_sides_is_penalised():
    args = _air_to_air_args(is_heat_pumping=False)

    result = hp_shared.evaluate_vapour_hpr_result(
        args=args,
        state=HPRParsedState(),
        work=50.0,
        work_arr=np.array([50.0]),
        Q_heat=np.array([100.0]),
        Q_cool=np.array([50.0]),
        cop_h=2.0,
        hpr_streams=_air_to_air_streams(),
    )

    assert (result.Q_amb_hot, result.Q_amb_cold) == pytest.approx((50.0, 60.0))
    # The 50 kW that goes from air to air.
    assert result.Q_air_to_air == pytest.approx(50.0)
    assert result.feasibility_penalty == pytest.approx(
        hp_shared._cycle_penalty(args=args, cycle_penalty_terms=[50.0])
        * hp_shared.hpr_penalty_cost_scale(args)
    )
