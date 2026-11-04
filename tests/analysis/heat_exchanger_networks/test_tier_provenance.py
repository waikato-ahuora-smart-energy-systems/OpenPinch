"""What a HEN result records about the stages and quality tier that ran."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    _evolution_options,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.pathways import (
    tier_pathways,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method import (
    execute_open_hens_method,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.domain.enums import HeatExchangerNetworkDesignMethod

from .test_full_sequence import (
    _small_problem,
    _small_problem_with_options,
    _use_fake_default_executor,
)


@pytest.mark.parametrize(
    ("run_design", "expected_sequence"),
    [
        (
            lambda design: design.heat_exchanger_network(),
            ("pinch_design_method", "network_evolution_method"),
        ),
        (
            lambda design: design.open_hens(),
            (
                "pinch_design_method",
                "thermal_derivative_method",
                "network_evolution_method",
            ),
        ),
    ],
    ids=["tier-0", "tier-1"],
)
def test_manifest_method_sequence_records_the_stages_that_ran(
    monkeypatch,
    run_design,
    expected_sequence: tuple[str, ...],
) -> None:
    _use_fake_default_executor(monkeypatch)

    design = run_design(_small_problem().design)

    assert design.result.manifest.method_sequence == expected_sequence


class _NoCouenneForMethod(FakeSynthesisExecutor):
    """Fail one method's tasks the way a missing Couenne binary does."""

    def __init__(self, method: str) -> None:
        super().__init__()
        self.method = method

    def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
        if tasks and tasks[0].method == self.method:
            return tuple(
                HeatExchangerNetworkSynthesisTaskOutcome(
                    task=task,
                    status="failed",
                    solver_status="failed",
                    error="The 'couenne' solver executable was not found on PATH.",
                )
                for task in tasks
            )
        return super().execute(
            tasks,
            problem=problem,
            parent_outcomes=parent_outcomes,
            max_parallel=max_parallel,
        )


@pytest.mark.parametrize(
    ("missing_for", "expected_sequence"),
    [
        (
            "thermal_derivative_method",
            ("pinch_design_method", "network_evolution_method"),
        ),
        ("pinch_design_method", ("network_evolution_method",)),
    ],
)
def test_manifest_method_sequence_omits_a_stage_couenne_could_not_run(
    missing_for: str,
    expected_sequence: tuple[str, ...],
) -> None:
    problem = _small_problem()

    with pytest.warns(RuntimeWarning, match="Couenne is unavailable"):
        result = execute_open_hens_method(
            problem,
            workflow_settings_from_problem(problem),
            executor=_NoCouenneForMethod(missing_for),
        )

    assert result.accepted_result.manifest.method_sequence == expected_sequence


_TIER = "HENS_SYNTHESIS_QUALITY_TIER"


@pytest.mark.parametrize(
    ("stored_options", "run_design", "expected_tier"),
    [
        ({}, lambda design: design.heat_exchanger_network(), 0),
        ({}, lambda design: design.enhanced_heat_exchanger_network(), 2),
        (
            {_TIER: 4},
            lambda design: design.enhanced_heat_exchanger_network(quality_tier=0),
            0,
        ),
        ({}, lambda design: design.heat_exchanger_network(options={_TIER: 3}), 3),
        ({_TIER: 3}, lambda design: design.heat_exchanger_network(), 3),
        ({_TIER: 3}, lambda design: design.open_hens(), 1),
        ({}, lambda design: design.open_hens(options={_TIER: 3}), 1),
    ],
    ids=[
        "default-tier-0",
        "enhanced-default-tier-2",
        "argument-beats-stored",
        "call-option",
        "stored-option",
        "open-hens-ignores-stored",
        "open-hens-ignores-call-option",
    ],
)
def test_provenance_settings_record_the_quality_tier_that_ran(
    monkeypatch,
    stored_options: dict,
    run_design,
    expected_tier: int,
) -> None:
    _use_fake_default_executor(monkeypatch)

    design = run_design(_small_problem_with_options(stored_options).design)

    assert design.result.provenance.effective_settings[_TIER] == expected_tier
    assert design.result.manifest.synthesis_quality_tier == expected_tier


def _evolution_breadths(problem, tasks) -> dict[tuple[str, ...], tuple[int, int]]:
    breadths = {}
    for task in tasks:
        if task.method != HeatExchangerNetworkDesignMethod.NetworkEvolution:
            continue
        options = _evolution_options(problem, task, max_parallel=1)
        breadths[tuple(task.metadata["pathway_ids"])] = (
            options["n_ad_branches"],
            options["n_rm_branches"],
        )
    return breadths


@pytest.mark.parametrize("tier", range(6))
def test_breadth_options_apply_on_every_route_and_manifest_agrees(tier: int) -> None:
    problem = _small_problem_with_options(
        {
            "HENS_SYNTHESIS_QUALITY_TIER": tier,
            "HENS_EVM_N_AD_BRANCHES": 2,
            "HENS_EVM_N_RM_BRANCHES": 3,
        }
    )

    result = execute_open_hens_method(
        problem,
        workflow_settings_from_problem(problem),
        executor=FakeSynthesisExecutor(),
    )

    manifest = result.accepted_result.manifest
    breadths = _evolution_breadths(problem, result.tasks)
    assert breadths
    assert set(breadths.values()) == {(2, 3)}
    assert (manifest.evm_n_ad_branches, manifest.evm_n_rm_branches) == (2, 3)


@pytest.mark.parametrize("tier", [0, 1])
@pytest.mark.parametrize(
    ("options", "expected_patience"),
    [({}, None), ({"HENS_EVM_N_AD_BRANCHES": 2}, 2)],
)
def test_tier_zero_and_one_routes_prune_only_when_widened(
    tier: int,
    options: dict[str, int],
    expected_patience: int | None,
) -> None:
    problem = _small_problem_with_options(
        {"HENS_SYNTHESIS_QUALITY_TIER": tier, **options}
    )

    pathways = tier_pathways(workflow_settings_from_problem(problem))

    assert pathways
    assert {pathway.evm_no_improvement_patience for pathway in pathways} == {
        expected_patience
    }


def test_protected_routes_stay_at_one_by_default_while_manifest_reports_widest() -> (
    None
):
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 4})

    result = execute_open_hens_method(
        problem,
        workflow_settings_from_problem(problem),
        executor=FakeSynthesisExecutor(),
    )

    manifest = result.accepted_result.manifest
    breadths = _evolution_breadths(problem, result.tasks)
    assert set(breadths.values()) == {(1, 1), (2, 2)}
    assert (manifest.evm_n_ad_branches, manifest.evm_n_rm_branches) == (2, 2)


@pytest.mark.parametrize(
    "missing_for", ["pinch_design_method", "thermal_derivative_method"]
)
def test_couenne_fallback_follows_the_breadth_options_and_manifest_agrees(
    missing_for: str,
) -> None:
    problem = _small_problem_with_options(
        {
            "HENS_SYNTHESIS_QUALITY_TIER": 1,
            "HENS_EVM_N_AD_BRANCHES": 2,
            "HENS_EVM_N_RM_BRANCHES": 3,
        }
    )

    with pytest.warns(RuntimeWarning, match="Couenne is unavailable"):
        result = execute_open_hens_method(
            problem,
            workflow_settings_from_problem(problem),
            executor=_NoCouenneForMethod(missing_for),
        )

    manifest = result.accepted_result.manifest
    evolution_tasks = [
        task
        for task in result.tasks
        if task.method == HeatExchangerNetworkDesignMethod.NetworkEvolution
    ]
    assert evolution_tasks
    for task in evolution_tasks:
        options = _evolution_options(problem, task, max_parallel=1)
        assert (options["n_ad_branches"], options["n_rm_branches"]) == (2, 3)
    assert (manifest.evm_n_ad_branches, manifest.evm_n_rm_branches) == (2, 3)
