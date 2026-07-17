"""Standalone network-evolution method tests."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    _task_model_approach_temperature,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    SynthesisWorkflowSettings,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method import (
    build_seeded_network_evolution_method_tasks,
)
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, HeatExchangerStreamRole
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def test_network_evolution_method_builds_retrofit_seeded_tasks() -> None:
    tasks = build_seeded_network_evolution_method_tasks(
        _settings(),
        (_seed_network(method="thermal_derivative_method"),),
    )

    assert len(tasks) == 1
    assert tasks[0].method == "network_evolution_method"
    assert tasks[0].seed_network_index == 0
    assert tasks[0].derivative_threshold == 0.5
    assert tasks[0].topology_restrictions


def test_seeded_network_evolution_prefers_solver_dtmin_metadata() -> None:
    seed = _seed_network(method="thermal_derivative_method").model_copy(
        update={
            "summary_metrics": {},
            "source_metadata": {"solver_dTmin": 12.0},
            "exchangers": (
                _seed_network(method="thermal_derivative_method")
                .exchangers[0]
                .model_copy(
                    update={
                        "period_states": (
                            HeatExchangerPeriodState(
                                period_id="0",
                                period_idx=0,
                                duty=100.0,
                                approach_temperatures=(99.0,),
                            ),
                        )
                    }
                ),
            ),
        },
    )

    tasks = build_seeded_network_evolution_method_tasks(_settings(), (seed,))

    assert tasks[0].approach_temperature == 12.0


def test_task_model_approach_temperature_matches_equation_layer_behavior() -> None:
    pdm_task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        approach_temperature=14.0,
        stage_count=1,
    )
    tdm_task = pdm_task.model_copy(update={"method": "thermal_derivative_method"})
    evm_task = pdm_task.model_copy(update={"method": "network_evolution_method"})

    assert _task_model_approach_temperature(pdm_task) == 14.0
    assert _task_model_approach_temperature(tdm_task) == 0.1
    assert _task_model_approach_temperature(evm_task) == 0.1


def test_quality_seeded_network_evolution_uses_canonical_topology() -> None:
    seed = _seed_network(method="thermal_derivative_method").model_copy(
        update={
            "stage_count": 3,
            "exchangers": (
                _recovery_exchanger("H1", "C1", 1),
                _recovery_exchanger("H2", "C2", 3),
            ),
        },
    )

    tasks = build_seeded_network_evolution_method_tasks(
        _settings(synthesis_quality_tier=3),
        (seed,),
    )

    assert len(tasks) == 1
    assert [item.stage for item in tasks[0].topology_restrictions] == [1, 2]
    assert tasks[0].stage_count == 2


def _settings(*, synthesis_quality_tier: int = 1) -> SynthesisWorkflowSettings:
    return SynthesisWorkflowSettings(
        run_id="method-test",
        approach_temperatures=(10.0,),
        derivative_thresholds=(0.5,),
        stage_selection=(1,),
        method_sequence=("network_evolution_method",),
        output_formats=(),
        solve_tolerance=1e-3,
        best_solutions_to_save=1,
        max_parallel=1,
        synthesis_quality_tier=synthesis_quality_tier,
        pdm_solver="couenne",
        tdm_solver="couenne",
        evm_solver="ipopt-pyomo",
        pdm_solver_options={},
        tdm_solver_options={},
        evm_solver_options={},
    )


def _seed_network(*, method: str) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(_recovery_exchanger("H1", "C1", 1),),
        run_id="seed-run",
        method=method,
        stage_count=1,
        summary_metrics={
            "approach_temperature": 10.0,
            "derivative_threshold": 0.5,
        },
    )


def _recovery_exchanger(hot: str, cold: str, stage: int) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"seed-recovery-{hot}-{cold}-{stage}",
        kind=HeatExchangerKind.RECOVERY,
        source_stream=hot,
        sink_stream=cold,
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=stage,
        period_states=(
            HeatExchangerPeriodState(
                period_id="0",
                period_idx=0,
                duty=100.0,
                approach_temperatures=(10.0,),
            ),
        ),
    )
