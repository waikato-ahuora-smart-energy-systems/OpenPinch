"""Standalone network-evolution method tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
    _task_model_approach_temperature,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    SynthesisWorkflowSettings,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method import (
    build_direct_network_evolution_method_tasks,
    build_seeded_network_evolution_method_tasks,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import FIXTURES_ROOT


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


def test_seeded_network_evolution_prefers_the_recorded_dt_cont_multiplier() -> None:
    seed = _seed_network(method="thermal_derivative_method").model_copy(
        update={
            "summary_metrics": {},
            "source_metadata": {"dt_cont_multiplier": 1.2},
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

    assert tasks[0].dt_cont_multiplier == 1.2


def test_task_model_approach_temperature_matches_equation_layer_behavior() -> None:
    pdm_task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        dt_cont_multiplier=1.4,
        stage_count=1,
    )
    tdm_task = pdm_task.model_copy(update={"method": "thermal_derivative_method"})
    evm_task = pdm_task.model_copy(update={"method": "network_evolution_method"})

    assert _task_model_approach_temperature(pdm_task, 14.0) == 14.0
    assert _task_model_approach_temperature(tdm_task, 14.0) == 0.1
    assert _task_model_approach_temperature(evm_task, 14.0) == 0.1


def test_min_exchanger_approach_share_raises_the_cost_driven_floor() -> None:
    pdm_task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        dt_cont_multiplier=1.4,
        stage_count=1,
    )
    tdm_task = pdm_task.model_copy(update={"method": "thermal_derivative_method"})
    evm_task = pdm_task.model_copy(update={"method": "network_evolution_method"})

    assert (
        _task_model_approach_temperature(pdm_task, 14.0, min_approach_share=0.5) == 14.0
    )
    assert (
        _task_model_approach_temperature(tdm_task, 14.0, min_approach_share=0.5) == 7.0
    )
    assert (
        _task_model_approach_temperature(evm_task, 14.0, min_approach_share=0.5) == 7.0
    )
    # The 0.1 K floor still applies to a smaller share.
    assert (
        _task_model_approach_temperature(evm_task, 14.0, min_approach_share=0.001)
        == 0.1
    )


def test_build_problem_reads_the_min_exchanger_approach_share() -> None:
    payload = json.loads(
        (
            FIXTURES_ROOT / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
        ).read_text(encoding="utf-8")
    )
    payload["options"]["HENS_MIN_EXCHANGER_APPROACH_SHARE"] = 0.5
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="network_evolution_method",
        dt_cont_multiplier=1.0,
        stage_count=1,
    )

    internal = LocalSynthesisExecutor(print_output=False)._build_problem(
        task,
        problem=PinchProblem(source=payload),
        parent_outcomes={},
    )

    assert internal.dTmin == 5.0


def test_an_evolution_task_holds_a_quarter_of_the_pinch_design_approach() -> None:
    """With no option set, a match of two 5 K streams, held to 10 K in pinch
    design at multiplier 1, is held to 2.5 K in evolution:
    max((dtcont_src + dtcont_snk) * m / 4, 0.1 K)."""
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        pair_minimum_approach,
    )

    payload = json.loads(
        (
            FIXTURES_ROOT / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
        ).read_text(encoding="utf-8")
    )
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="network_evolution_method",
        dt_cont_multiplier=1.0,
        stage_count=1,
    )

    internal = LocalSynthesisExecutor(print_output=False)._build_problem(
        task,
        problem=PinchProblem(source=payload),
        parent_outcomes={},
    )

    arrays = internal.solver_arrays.arrays
    hot = float(arrays["T_h_cont_period"][0][0])
    cold = float(arrays["T_c_cont_period"][0][0])
    assert pair_minimum_approach(hot, cold) == pytest.approx(2.5, abs=1e-9)


def test_a_share_of_zero_holds_every_match_to_the_floor() -> None:
    """At a share of 0 a match of two 10 K streams, 20 K in pinch design, may
    close to 0.1 K like any other match."""
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        pair_minimum_approach,
    )

    payload = json.loads(
        (
            FIXTURES_ROOT / "openhens" / "Four-stream-Yee-and-Grossmann-1990-1.json"
        ).read_text(encoding="utf-8")
    )
    for stream in payload["streams"]:
        stream["dt_cont"] = {"value": 10.0, "unit": "delta_degC"}
    payload["options"]["HENS_MIN_EXCHANGER_APPROACH_SHARE"] = 0.0
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="network_evolution_method",
        dt_cont_multiplier=1.0,
        stage_count=1,
    )

    internal = LocalSynthesisExecutor(print_output=False)._build_problem(
        task,
        problem=PinchProblem(source=payload),
        parent_outcomes={},
    )

    arrays = internal.solver_arrays.arrays
    hot = float(arrays["T_h_cont_period"][0][0])
    cold = float(arrays["T_c_cont_period"][0][0])
    assert pair_minimum_approach(hot, cold) == pytest.approx(0.1, abs=1e-9)


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
        _settings(),
        (seed,),
    )

    assert len(tasks) == 1
    assert [item.stage for item in tasks[0].topology_restrictions] == [1, 2]
    assert tasks[0].stage_count == 2


def _settings() -> SynthesisWorkflowSettings:
    return SynthesisWorkflowSettings(
        run_id="method-test",
        dt_cont_multipliers=(1.0,),
        derivative_thresholds=(0.5,),
        stage_selection=(1,),
        method_sequence=("network_evolution_method",),
        output_formats=(),
        solve_tolerance=1e-3,
        best_solutions_to_save=1,
        max_parallel=1,
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
            "dt_cont_multiplier": 1.0,
            "derivative_threshold": 0.5,
        },
    )


def _recovery_exchanger(hot: str, cold: str, stage: int) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"seed-recovery-{hot}-{cold}-{stage}",
        kind=HeatExchangerKind.RECOVERY,
        source_stream=hot,
        sink_stream=cold,
        source_stream_role=StreamID.Process,
        sink_stream_role=StreamID.Process,
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


def test_direct_evolution_tasks_skip_repeated_multipliers() -> None:
    settings = replace(
        _settings(), dt_cont_multipliers=(1.0, 1.0, 0.5), stage_selection=(2,)
    )

    tasks = build_direct_network_evolution_method_tasks(settings)

    assert [task.dt_cont_multiplier for task in tasks] == [1.0, 0.5]
    assert len({task.task_id for task in tasks}) == 2
