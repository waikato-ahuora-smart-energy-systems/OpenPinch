"""Standalone thermal-derivative method tests."""

from __future__ import annotations

from OpenPinch.classes.heat_exchanger import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerStreamRole,
)
from OpenPinch.classes.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.settings import (
    SynthesisWorkflowSettings,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.thermal_derivative_method import (
    build_seeded_thermal_derivative_method_tasks,
)


def test_thermal_derivative_method_builds_seeded_tasks() -> None:
    tasks = build_seeded_thermal_derivative_method_tasks(
        _settings(),
        (_seed_network(method="pinch_design_method"),),
    )

    assert len(tasks) == 1
    assert tasks[0].method == "thermal_derivative_method"
    assert tasks[0].seed_network_index == 0
    assert tasks[0].topology_restrictions


def _settings() -> SynthesisWorkflowSettings:
    return SynthesisWorkflowSettings(
        run_id="method-test",
        approach_temperatures=(10.0,),
        derivative_thresholds=(0.5,),
        stage_selection=(1,),
        method_sequence=("thermal_derivative_method",),
        output_formats=(),
        solve_tolerance=1e-3,
        best_solutions_to_save=1,
        max_parallel=1,
        pdm_solver="couenne",
        tdm_solver="couenne",
        esm_solver="ipopt-pyomo",
        pdm_solver_options={},
        tdm_solver_options={},
        esm_solver_options={},
    )


def _seed_network(*, method: str) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="seed-recovery",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                duty=100.0,
                approach_temperatures=(10.0,),
            ),
        ),
        run_id="seed-run",
        method=method,
        stage_count=1,
        summary_metrics={"approach_temperature": 10.0},
    )
