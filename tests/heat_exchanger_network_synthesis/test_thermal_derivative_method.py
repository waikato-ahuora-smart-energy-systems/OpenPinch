"""Standalone thermal-derivative method tests."""

from __future__ import annotations

import pytest

from OpenPinch.classes.heat_exchanger import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerStreamRole,
)
from OpenPinch.classes.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.pathways import (
    TierPathway,
    pathway_metadata,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.settings import (
    SynthesisWorkflowSettings,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services import (
    pinch_design_method,
    thermal_derivative_method,
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


def test_pinch_design_quality_task_builder_adds_deduped_stage_pair_tasks() -> None:
    settings = _settings(
        method_sequence=("pinch_design_method",),
        synthesis_quality_tier=2,
        dt_cont_multipliers=(1.0, 1.5),
        user_dt_cont_multipliers=True,
        pdm_stage_pair_limit=2,
    )

    assert pinch_design_method.build_pinch_design_method_tasks(settings)
    base_task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        approach_temperature=10.0,
    )
    tasks = pinch_design_method._with_quality_pdm_tasks(settings, (base_task,))
    quality_tasks = [
        task for task in tasks if task.metadata.get("quality_candidate") is not None
    ]

    assert any(
        task.metadata["quality_candidate"] == "dt_cont_multiplier"
        and task.approach_temperature == 15.0
        for task in quality_tasks
    )
    assert [
        task.settings["stage_selection"]
        for task in quality_tasks
        if task.metadata["quality_candidate"] == "stage_pair"
    ] == [[1, 2], [2, 1]]

    no_stage_pair_settings = _settings(
        method_sequence=("pinch_design_method",),
        synthesis_quality_tier=2,
        dt_cont_multipliers=(1.0, 1.5),
        user_dt_cont_multipliers=True,
        pdm_stage_pair_limit=0,
    )
    no_stage_pair_tasks = pinch_design_method._with_quality_pdm_tasks(
        no_stage_pair_settings,
        (base_task,),
    )
    assert all(
        task.metadata.get("quality_candidate") != "stage_pair"
        for task in no_stage_pair_tasks
    )

    duplicate_stage_pair_task = base_task.model_copy(
        update={"settings": {"stage_selection": [1, 2]}}
    )
    duplicate_checked = pinch_design_method._with_quality_pdm_tasks(
        settings,
        (base_task, duplicate_stage_pair_task),
    )
    assert [
        task.settings.get("stage_selection")
        for task in duplicate_checked
        if task.metadata.get("quality_candidate") == "stage_pair"
    ] == [[2, 1]]

    with pytest.raises(ValueError, match="requires a multiplier"):
        pinch_design_method._pathway_approach_temperature(
            settings,
            TierPathway(
                pathway_id="bad",
                tier_origin=2,
                pathway_kind="quality",
                pdm_mode="compact",
                multiplier=None,
                uses_tdm=False,
                evm_n_ad_branches=1,
                evm_n_rm_branches=1,
                evm_no_improvement_patience=None,
                protected=False,
            ),
        )


def test_pinch_design_task_builder_skips_non_exact_pathways_without_multiplier(
    monkeypatch,
) -> None:
    path_without_multiplier = TierPathway(
        pathway_id="bad",
        tier_origin=2,
        pathway_kind="quality",
        pdm_mode="compact",
        multiplier=None,
        uses_tdm=False,
        evm_n_ad_branches=1,
        evm_n_rm_branches=1,
        evm_no_improvement_patience=None,
        protected=False,
        exact_open_hens=False,
    )
    monkeypatch.setattr(
        pinch_design_method,
        "tier_pathways",
        lambda _settings: (path_without_multiplier,),
    )

    assert (
        pinch_design_method.build_pinch_design_method_tasks(
            _settings(method_sequence=("pinch_design_method",)),
        )
        == ()
    )


def test_thermal_derivative_skips_non_tdm_pathways_and_empty_default_stage():
    non_tdm_pathway = TierPathway(
        pathway_id="compact-only",
        tier_origin=0,
        pathway_kind="tier0_pdm_evm",
        pdm_mode="compact",
        multiplier=1.0,
        uses_tdm=False,
        evm_n_ad_branches=1,
        evm_n_rm_branches=1,
        evm_no_improvement_patience=None,
        protected=True,
    )
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        approach_temperature=10.0,
        metadata=pathway_metadata((non_tdm_pathway,)),
    )
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
    )

    assert (
        thermal_derivative_method.build_thermal_derivative_method_tasks(
            _settings(),
            [outcome],
        )
        == ()
    )

    tasks, outcomes = thermal_derivative_method.execute_thermal_derivative_method_stage(
        problem=object(),
        settings=_settings(),
        pdm_outcomes=[],
        parent_outcomes={},
    )

    assert tasks == ()
    assert outcomes == ()

    failed_outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task.model_copy(update={"metadata": {}}),
        status="failed",
        error="solver failed",
    )
    assert (
        thermal_derivative_method.build_thermal_derivative_method_tasks(
            _settings(),
            [failed_outcome],
        )
        == ()
    )


def test_seeded_quality_thermal_derivative_tasks_deduplicate_topologies() -> None:
    settings = _settings(
        synthesis_quality_tier=2,
        derivative_thresholds=(0.25, 0.25, 0.5),
    )
    seed = _seed_network(method="pinch_design_method")

    tasks = build_seeded_thermal_derivative_method_tasks(settings, (seed, seed))

    assert len(tasks) == 2
    assert [task.derivative_threshold for task in tasks] == [0.25, 0.5]
    assert {task.seed_network_index for task in tasks} == {0}
    assert all(task.stage_count == 1 for task in tasks)


def test_thermal_derivative_parent_task_builder_accepts_plain_successful_pdm():
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="pinch_design_method",
        approach_temperature=10.0,
    )
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=_seed_network(method="pinch_design_method"),
    )

    tasks = thermal_derivative_method.build_thermal_derivative_method_tasks(
        _settings(),
        [outcome],
    )

    assert len(tasks) == 1
    assert tasks[0].parent_task_id == task.task_id
    assert tasks[0].topology_restrictions


def test_pinch_design_workflow_wrappers_use_executor_contract(monkeypatch):
    settings = _settings(method_sequence=("pinch_design_method",))
    default_executor = _RecordingExecutor()
    monkeypatch.setattr(
        pinch_design_method,
        "LocalSynthesisExecutor",
        lambda: default_executor,
    )

    tasks, outcomes = pinch_design_method.execute_pinch_design_method_stage(
        problem=object(),
        settings=settings,
    )

    assert tasks
    assert len(outcomes) == len(tasks)
    assert default_executor.calls[0]["parent_outcomes"] == {}

    workflow_executor = _RecordingExecutor()
    result = pinch_design_method._execute_pinch_design_method_workflow(
        problem=object(),
        settings=settings,
        executor=workflow_executor,
    )

    assert result.accepted_result.method == "pinch_design_method"
    assert workflow_executor.calls[0]["max_parallel"] == settings.max_parallel


def test_thermal_derivative_workflow_wrappers_use_executor_contract(monkeypatch):
    settings = _settings()
    default_executor = _RecordingExecutor()
    monkeypatch.setattr(
        thermal_derivative_method,
        "LocalSynthesisExecutor",
        lambda: default_executor,
    )

    tasks, outcomes = (
        thermal_derivative_method.execute_seeded_thermal_derivative_method_stage(
            problem=object(),
            settings=settings,
            seed_networks=(_seed_network(method="pinch_design_method"),),
        )
    )

    assert tasks
    assert len(outcomes) == len(tasks)
    assert default_executor.calls[0]["parent_outcomes"] == {}

    workflow_executor = _RecordingExecutor()
    result = thermal_derivative_method._execute_thermal_derivative_method_workflow(
        problem=object(),
        settings=settings,
        seed_networks=(_seed_network(method="pinch_design_method"),),
        executor=workflow_executor,
    )

    assert result.accepted_result.method == "thermal_derivative_method"
    assert workflow_executor.calls[0]["max_parallel"] == settings.max_parallel


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
        self.calls.append(
            {
                "tasks": tasks,
                "problem": problem,
                "parent_outcomes": parent_outcomes,
                "max_parallel": max_parallel,
            }
        )
        return tuple(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=task,
                status="success",
                network=_seed_network(method=_method_value(task.method)),
                objective_value=float(index + 1),
                solver_status="ok",
            )
            for index, task in enumerate(tasks)
        )


def _method_value(method) -> str:
    return method.value if hasattr(method, "value") else str(method)


def _settings(**updates) -> SynthesisWorkflowSettings:
    params = dict(
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
        evm_solver="ipopt-pyomo",
        pdm_solver_options={},
        tdm_solver_options={},
        evm_solver_options={},
    )
    params.update(updates)
    return SynthesisWorkflowSettings(**params)


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
                area=10.0,
                approach_temperatures=(10.0,),
            ),
        ),
        run_id="seed-run",
        method=method,
        stage_count=1,
        summary_metrics={"approach_temperature": 10.0},
    )
