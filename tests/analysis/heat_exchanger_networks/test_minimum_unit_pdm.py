"""Minimum-unit networks from the pinch design method.

Each pinch side is re-solved for the fewest units at its utility target, and
no-good cuts collect further minimum-unit and one-more-unit topologies. Side
pairs become extra networks at each approach temperature, all seeds for
network evolution.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
    build_pinch_design_decomposition,
)
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)

pytestmark = pytest.mark.synthesis


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _problem(dTmin: float, limits: tuple[int, int]):
    pinch = PinchProblem(
        source=json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8")),
        project_name="Four-stream",
    )
    return InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(pinch, dTmin),
        framework="PDM",
        solver="highs",
        dTmin=dTmin,
        pinch_decompositions={
            side: build_pinch_design_decomposition(pinch, dTmin, pinch_location=side)
            for side in ("above", "below")
        },
        pdm_network_limits=limits,
    )


def _units(network) -> int:
    return len(network.exchangers)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_pdm_collects_distinct_minimum_unit_networks_as_variants():
    problem = _problem(14.0, (3, 3))

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    networks = [problem.extract_network(run_id="pdm")] + [
        variant.extract_network(run_id="pdm") for variant in problem.pdm_variants
    ]
    assert len(problem.pdm_variants) >= 2
    assert all(variant.case.mSuccess == 1 for variant in problem.pdm_variants)
    signatures = [network_structure_signature(network) for network in networks]
    assert len(set(signatures)) == len(signatures)
    variant_units = [_units(network) for network in networks[1:]]
    assert max(variant_units) - min(variant_units) <= 1
    assert len(variant_units) <= 6


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_pdm_without_network_limits_has_no_variants():
    problem = _problem(14.0, (0, 0))

    problem.get_solution(print_output=False)

    assert problem.pdm_variants == []


def _pinch_problem(options: dict) -> PinchProblem:
    payload = json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8"))
    payload["options"] = payload["options"] | {
        "HENS_SOLVER_PDM": "highs",
        "HENS_DT_CONT_MULTIPLIERS": [1.4],
        "HENS_SYNTHESIS_QUALITY_TIER": 1,
        **options,
    }
    pinch = PinchProblem(source=payload, project_name="Four-stream")
    pinch.target.all_heat_integration()
    return pinch


def test_pdm_network_limits_come_from_the_configuration():
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
        workflow_settings_from_problem,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
        build_pinch_design_method_tasks,
    )

    pinch = _pinch_problem(
        {"HENS_PDM_MIN_UNIT_NETWORKS": 2, "HENS_PDM_EXTRA_UNIT_NETWORKS": 1}
    )
    task = build_pinch_design_method_tasks(workflow_settings_from_problem(pinch))[0]

    built = LocalSynthesisExecutor(print_output=False)._build_problem(
        task, problem=pinch, parent_outcomes={}
    )

    assert built.pdm_network_limits == (2, 1)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_pdm_stage_returns_each_variant_as_its_own_seed_outcome():
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        LocalSynthesisExecutor,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
        workflow_settings_from_problem,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method import (
        build_network_evolution_method_tasks_from_pinch_design_method,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
        execute_pinch_design_method_stage,
    )

    pinch = _pinch_problem(
        {"HENS_PDM_MIN_UNIT_NETWORKS": 2, "HENS_PDM_EXTRA_UNIT_NETWORKS": 1}
    )
    settings = workflow_settings_from_problem(pinch)
    executor = LocalSynthesisExecutor(print_output=False)

    tasks, outcomes = execute_pinch_design_method_stage(
        pinch, settings, executor=executor
    )

    variants = [o for o in outcomes if "pdm_variant" in o.task.settings]
    assert variants
    assert len(outcomes) == len(tasks)
    assert {o.task.task_id for o in outcomes} == {t.task_id for t in tasks}
    assert all(o.status == "success" for o in variants)
    assert all(o.network.task_id == o.task.task_id for o in variants)
    assert all(o.task.task_id in executor.problems_by_task_id for o in variants)
    evolution_parents = {
        task.parent_task_id
        for task in build_network_evolution_method_tasks_from_pinch_design_method(
            outcomes, settings=settings
        )
    }
    assert {o.task.task_id for o in variants} <= evolution_parents


def test_pdm_variants_seed_evolution_directly_and_skip_the_derivative_method():
    # The thermal derivative method refines each approach temperature's
    # minimum-utility network; the extra networks go straight to evolution at
    # every quality tier, which bounds the added solves.
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        _pdm_variant_task,
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
    from tests.analysis.heat_exchanger_networks.test_full_sequence import (
        _small_problem,
    )

    class VariantExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, **kwargs):
            if tasks and tasks[0].method == "pinch_design_method":
                tasks = tuple(tasks) + tuple(
                    _pdm_variant_task(task, 1) for task in tasks
                )
            return super().execute(tasks, **kwargs)

    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    assert settings.synthesis_quality_tier == 1

    result = execute_open_hens_method(problem, settings, executor=VariantExecutor())

    variant_ids = {
        task.task_id for task in result.tasks if "pdm_variant" in task.settings
    }
    parents = {
        method: {task.parent_task_id for task in result.tasks if task.method == method}
        for method in ("thermal_derivative_method", "network_evolution_method")
    }
    assert variant_ids
    assert parents["thermal_derivative_method"]
    assert not variant_ids & parents["thermal_derivative_method"]
    assert variant_ids <= parents["network_evolution_method"]


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_pdm_stage_reduction_solves_its_linear_model_with_highs():
    # The reduced PDM network fixes its topology, which leaves an LP that
    # HiGHS solves.
    problem = _problem(14.0, (0, 0))

    problem.get_solution(print_output=False)

    assert problem.case.name.startswith("reduced-")
    assert problem.case.solver == "highs"
    assert problem.case.mSuccess == 1


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_main_pdm_network_has_the_fewest_units_its_matches_allow():
    # Minimum utility cost leaves the binaries free once the targets are met,
    # so a MILP solver may keep matches it does not need: on the pulp mill
    # HiGHS left 2,994 below-pinch matches, most carrying under 1 kW. Each
    # side is re-solved for the fewest units among the matches it used, at
    # the same utility. At 26 K the four-stream network drops from 7 units
    # to 6.
    problem = _problem(26.0, (0, 0))

    problem.get_solution(print_output=False)

    assert problem.below.minimisation_goal == "min units"
    assert len(problem.extract_network(run_id="pdm").exchangers) == 6


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_fewest_units_resolve_is_time_limited_and_keeps_the_side_converged(
    monkeypatch,
):
    # Proving the fewest units can take far longer than finding them (the
    # pulp mill's 1,277 used matches). The re-solve stops at a time limit,
    # and its network keeps the minimum utility the first solve proved.
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
        PinchDecompModel,
    )

    options = []
    optimise = PinchDecompModel.optimise

    def time_limited(self, print_output):
        optimise(self, print_output)
        if self.name.endswith("fewest units"):
            options.append(self.solver_options)
            self.solver_run = replace(
                self.solver_run, termination_condition="maxTimeLimit"
            )

    monkeypatch.setattr(PinchDecompModel, "optimise", time_limited)
    problem = _problem(26.0, (0, 0))

    problem.get_solution(print_output=False)

    assert options and all(
        o == {"time_limit": backend.FEWEST_UNITS_TIME_LIMIT} for o in options
    )
    assert problem.below.minimisation_goal == "min units"
    assert problem.below.solver_run.converged
    assert problem.case.solver_run.converged
