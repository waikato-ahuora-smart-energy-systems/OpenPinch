"""What a HEN result records about the stages and search levels that ran."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    _evolution_options,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
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
            # Depth 1 given in the call runs its heuristic, over the small
            # problem's stored HENS_HEURISTIC_PDM_NETWORKS of 0.
            lambda design: design.heat_exchanger_network(depth=1),
            ("heuristic_pinch_design_method", "network_evolution_method"),
        ),
        (
            lambda design: design.heat_exchanger_network(),
            (
                "pinch_design_method",
                "thermal_derivative_method",
                "network_evolution_method",
            ),
        ),
    ],
    ids=["depth-1", "depth-2"],
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
    monkeypatch,
) -> None:
    _neither_couenne_nor_scip(monkeypatch)
    problem = _small_problem()

    with pytest.warns(RuntimeWarning, match="Couenne is unavailable"):
        result = execute_open_hens_method(
            problem,
            workflow_settings_from_problem(problem),
            executor=_NoCouenneForMethod(missing_for),
        )

    assert result.accepted_result.manifest.method_sequence == expected_sequence


def test_manifest_method_sequence_omits_an_evolution_stage_couenne_could_not_run():
    problem = _small_problem()

    result = execute_open_hens_method(
        problem,
        workflow_settings_from_problem(problem),
        executor=_NoCouenneForMethod("network_evolution_method"),
    )

    assert result.accepted_result.manifest.method_sequence == (
        "pinch_design_method",
        "thermal_derivative_method",
    )


_DEPTH = "HENS_SEARCH_DEPTH"


@pytest.mark.parametrize(
    ("stored_options", "run_design", "expected_depth"),
    [
        ({}, lambda design: design.heat_exchanger_network(), 2),
        ({_DEPTH: 4}, lambda design: design.heat_exchanger_network(depth=0), 0),
        ({}, lambda design: design.heat_exchanger_network(options={_DEPTH: 3}), 3),
        ({_DEPTH: 3}, lambda design: design.heat_exchanger_network(), 3),
    ],
    ids=["fixture-depth", "argument-beats-stored", "call-option", "stored-option"],
)
def test_provenance_settings_record_the_depth_that_ran(
    monkeypatch,
    stored_options: dict,
    run_design,
    expected_depth: int,
) -> None:
    _use_fake_default_executor(monkeypatch)

    design = run_design(_small_problem_with_options(stored_options).design)

    assert design.result.provenance.effective_settings[_DEPTH] == expected_depth
    assert design.result.manifest.search_depth == expected_depth


def _evolution_breadths(problem, tasks) -> set[tuple[int, int, int | None]]:
    breadths = set()
    for task in tasks:
        if task.method != HeatExchangerNetworkDesignMethod.NetworkEvolution:
            continue
        options = _evolution_options(problem, task, max_parallel=1)
        breadths.add(
            (
                options["n_ad_branches"],
                options["n_rm_branches"],
                options["no_improvement_patience"],
            )
        )
    return breadths


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({_DEPTH: 2}, {(1, 1, None)}),
        ({_DEPTH: 3}, {(2, 2, 2)}),
        ({_DEPTH: 4}, {(3, 3, 2)}),
        ({_DEPTH: 4, "HENS_EVM_N_AD_BRANCHES": 1}, {(1, 3, 2)}),
        ({_DEPTH: 2, "HENS_EVM_N_RM_BRANCHES": 2}, {(1, 2, 2)}),
    ],
)
def test_every_evolution_task_and_the_manifest_use_the_depth_breadth(
    options: dict, expected: set
) -> None:
    problem = _small_problem_with_options(options)

    result = execute_open_hens_method(
        problem,
        workflow_settings_from_problem(problem),
        executor=FakeSynthesisExecutor(),
    )

    assert _evolution_breadths(problem, result.tasks) == expected
    manifest = result.accepted_result.manifest
    ((n_ad, n_rm, _patience),) = expected
    assert (manifest.evm_n_ad_branches, manifest.evm_n_rm_branches) == (n_ad, n_rm)


@pytest.mark.parametrize(
    "missing_for", ["pinch_design_method", "thermal_derivative_method"]
)
def test_couenne_fallback_evolution_uses_the_depth_breadth(
    missing_for: str, monkeypatch
) -> None:
    _neither_couenne_nor_scip(monkeypatch)
    problem = _small_problem_with_options({_DEPTH: 3})

    with pytest.warns(RuntimeWarning, match="Couenne is unavailable"):
        result = execute_open_hens_method(
            problem,
            workflow_settings_from_problem(problem),
            executor=_NoCouenneForMethod(missing_for),
        )

    assert _evolution_breadths(problem, result.tasks) == {(2, 2, 2)}


def _neither_couenne_nor_scip(monkeypatch) -> None:
    """Without Couenne and SCIP the TDM stays on Couenne, so its skip applies.

    With SCIP installed, a missing default Couenne falls back to SCIP and the
    thermal derivative method still runs.
    """

    from OpenPinch.analysis.heat_exchanger_networks.solver import backend

    real = backend.require_solver_backend

    def require(name):
        if name in {"couenne", "scip"}:
            raise backend.MissingSynthesisSolverError(f"{name} missing")
        real(name)

    monkeypatch.setattr(backend, "require_solver_backend", require)
