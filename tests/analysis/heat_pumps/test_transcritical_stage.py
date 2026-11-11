"""A stage condensing above its fluid's critical point runs transcritical.

Its high pressure is the critical isochore's pressure at the "condensing"
temperature T_cond, and its gas cooler glides from the discharge down to
T_cond less the subcooling.
"""

from __future__ import annotations

from types import SimpleNamespace

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.targeting.cascade_vapour_compression as hp_cascade
from OpenPinch.analysis.heat_pumps.common.shared import (
    SUBCRITICAL_CONDENSING_MARGIN_K,
    limit_parallel_pressure_ratios,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)

from .helpers import _base_args


def _co2_stage(**kwargs):
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=0.0,
        T_cond=90.0,
        dtcont=0.0,
        dT_subcool=50.0,
        refrigerant="CO2",
        dT_ihx_gas_side=0.0,
        Q_heat=1000.0,
        **kwargs,
    )
    return cycle


def test_a_transcritical_stage_runs_on_the_critical_isochore():
    cycle = _co2_stage()

    assert cycle.solved
    P_high = CP.PropsSI(
        "P", "T", 90.0 + 273.15, "Dmass", CP.PropsSI("rhocrit", "CO2"), "CO2"
    )
    assert cycle.Ps[1] == pytest.approx(P_high, rel=1e-9)


def test_a_transcritical_gas_cooler_glides_down_to_t_cond_less_the_subcooling():
    cycle = _co2_stage()

    assert cycle.Ts[2] - 273.15 == pytest.approx(40.0, abs=1e-6)
    assert cycle.dT_subcool == pytest.approx(50.0, abs=1e-6)
    profile = cycle._build_condenser_profile()
    assert float(np.min(profile[:, 1])) == pytest.approx(40.0, abs=1e-6)
    assert float(np.max(profile[:, 1])) == pytest.approx(cycle.Ts[1] - 273.15, abs=1e-6)


def test_a_transcritical_warm_start_is_seeded_within_the_pressure_ratio_limit():
    args = SimpleNamespace(
        refrigerant_ls=["CO2"], simulation_backend="coolprop", max_pressure_ratio=5.0
    )

    (T_evap,) = limit_parallel_pressure_ratios(
        np.array([90.0]), np.array([-40.0]), args
    )

    P_high = CP.PropsSI(
        "P", "T", 90.0 + 273.15, "Dmass", CP.PropsSI("rhocrit", "CO2"), "CO2"
    )
    P_low = CP.PropsSI("P", "T", T_evap + 273.15, "Q", 1, "CO2")
    assert P_high / P_low == pytest.approx(5.0, rel=1e-6)


@pytest.mark.parametrize(
    ("refrigerant", "T_cond"),
    [("Ammonia", 140.0), ("CO2", 60.0), ("R1233zd(E)", 180.0)],
)
def test_a_transcritical_stage_without_subcooling_reports_none(refrigerant, T_cond):
    # Round-off between the T_cond state and the outlet flash must not leave a
    # negative subcooling, which a simulation record rejects.
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=0.0,
        T_cond=T_cond,
        dtcont=0.0,
        refrigerant=refrigerant,
        dT_ihx_gas_side=0.0,
        Q_heat=1000.0,
    )

    assert cycle.dT_subcool == 0.0


def test_a_warm_start_above_the_critical_point_is_seeded_subcritical():
    # The search may go transcritical, but the Carnot warm start, which knows
    # nothing about critical points, starts from a subcritical design.
    args = _base_args(n_cond=1, n_evap=1, refrigerant_ls=["n-Butane"])
    args.T_cold = np.array([200.0, 150.0, 100.0])
    args.T_hot = np.array([120.0, 80.0, 40.0])
    seed = SimpleNamespace(
        T_cond=np.array([180.0]),
        T_evap=np.array([110.0]),
        Q_cond=np.array([100.0]),
        Q_evap=np.array([80.0]),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    x0, _ = hp_cascade._get_cascade_hp_opt_setup(seed, args)
    state = hp_cascade._parse_cascade_hp_state_variables(x0, args)

    t_crit = CP.PropsSI("Tcrit", "n-Butane") - 273.15
    assert state.T_cond[0] == pytest.approx(t_crit - SUBCRITICAL_CONDENSING_MARGIN_K)
