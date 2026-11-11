"""The effective topology: the stages a selected design actually uses."""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.common.topology import effective_topology
from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
    CascadeVapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
    aggregate_hpr_period_results,
)
from OpenPinch.contracts.hpr import (
    HPRActiveStage,
    HPRBackendResult,
    HPREffectiveTopology,
)
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle
from OpenPinch.presentation.reporting.hpr import hpr_summary

from .helpers import _base_args


def _parallel_with_one_idle_machine():
    cycle = ParallelVapourCompressionCycles()
    cycle.solve(
        T_evap=np.array([10.0, 20.0]),
        T_cond=np.array([60.0, 50.0]),
        dtcont=0.0,
        refrigerant=["Ammonia", "Propane"],
        dT_ihx_gas_side=0.0,
        Q_heat=np.array([100.0, 0.0]),
    )
    assert cycle.solved
    return cycle


def test_an_idle_machine_is_not_an_active_stage():
    args = _base_args(
        hpr_type=HeatPumpAndRefrigerationCycle.ParallelVapourComp.value,
        n_cond=2,
        n_evap=2,
        min_stage_lift=5.0,
    )

    topology = effective_topology(_parallel_with_one_idle_machine(), args)

    assert topology.configured_stages == 2
    (stage,) = topology.active_stages
    assert stage.index == 0
    assert stage.fluid.casefold() == "ammonia"
    assert stage.evaporating_temperature == pytest.approx(10.0, abs=1e-6)
    assert stage.condensing_temperature == pytest.approx(60.0, abs=1e-6)
    assert stage.pressure_ratio > 1.0
    assert stage.work > 0.0
    assert stage.process_heat == pytest.approx(100.0)
    assert "2x2 requested; 1 of 2 stages used" in topology.label
    assert "AMMONIA 10.0 -> 60.0 C" in topology.label.upper()


def test_a_stage_lifting_less_than_the_minimum_is_not_active():
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([5.0]),
        T_cond=np.array([80.0, 78.0]),
        dtcont=0.0,
        refrigerant=["Ammonia", "Ammonia"],
        dT_ihx_gas_side=0.0,
        dt_cascade_hx=1.0,
        Q_heat=np.array([100.0, 50.0]),
    )
    assert cycle.solved
    args = _base_args(
        hpr_type=HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
        n_cond=2,
        n_evap=1,
        min_stage_lift=5.0,
    )

    topology = effective_topology(cycle, args)

    # The top stage evaporates at 78 - 1 = 77 C and lifts 3 K to 80 C, less
    # than the 5 K minimum; the lower stage lifts 5 C to 78 C.
    assert [stage.index for stage in topology.active_stages] == [1]


_SMALL_SEARCH = dict(
    load_fraction=0.5,
    maximum_restarts=1,
    maximum_iterations=2,
    maximum_evaluations=20,
)
_PERMISSIVE = {
    "COSTING_HPR_CAPITAL_RECOVERY_ENABLED": False,
    "HPR_MAX_PRESSURE_RATIO": 1e6,
    "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.0,
}


def test_a_vapour_compression_target_reports_its_effective_topology():
    target = PinchProblem("basic_pinch.json").target.vapour_compression_heat_pump(
        refrigerants=["R1233zd(E)"],
        condensers=1,
        evaporators=1,
        options=_PERMISSIVE,
        **_SMALL_SEARCH,
    )

    topology = target.hpr_details.effective_topology
    assert topology.configured_stages == 1
    assert len(topology.active_stages) == 1
    assert topology.active_stages[0].role == "vapour_compression"
    assert "1 of 1 stages used" in topology.label


def test_an_mvr_target_reports_its_mvr_stages():
    target = PinchProblem("basic_pinch.json").target.mvr_heat_pump(
        mvr_fluids=["R1233zd(E)"],
        mvr_stages=1,
        condensers=1,
        evaporators=1,
        options=_PERMISSIVE,
        **_SMALL_SEARCH,
    )

    topology = target.hpr_details.effective_topology
    assert topology.configured_stages == 2
    assert [stage.role for stage in topology.active_stages][-1:] == ["mvr"]


def _period_result(active_index):
    stage = HPRActiveStage(
        index=active_index,
        role="vapour_compression",
        fluid="Ammonia",
        evaporating_temperature=10.0 * active_index,
        condensing_temperature=60.0,
        pressure_ratio=3.0,
        work=10.0,
    )
    return HPRBackendResult(
        obj=1.0,
        utility_tot=1.0,
        w_net=1.0,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        effective_topology=HPREffectiveTopology.from_stages(
            cycle="parallel",
            condensers=2,
            evaporators=2,
            configured_stages=2,
            active_stages=(stage,),
        ),
    )


def test_a_shared_design_uses_every_stage_that_works_in_any_period():
    aggregated, _ = aggregate_hpr_period_results(
        {"p0": _period_result(0), "p1": _period_result(1)}, [1.0, 1.0]
    )

    topology = aggregated.effective_topology
    assert [stage.index for stage in topology.active_stages] == [0, 1]
    assert "2 of 2 stages used" in topology.label


def test_the_summary_reports_the_effective_topology_and_alternatives():
    target = PinchProblem("basic_pinch.json").target.vapour_compression_heat_pump(
        refrigerants=["R1233zd(E)"],
        condensers=1,
        evaporators=1,
        options=_PERMISSIVE,
        **_SMALL_SEARCH,
    )

    summary = hpr_summary(target)

    assert summary["effective_topology"] == target.hpr_details.effective_topology.label
    assert summary["alternatives"] == [
        {"design": a.topology.label, "objective_gap": a.objective_gap}
        for a in target.hpr_details.alternatives
    ]
