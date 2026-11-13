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


def _problem(dTmin: float, limits: tuple[int, int], *, extra_utilities=(), source=None):
    payload = json.loads((source or FOUR_STREAM_JSON).read_text(encoding="utf-8"))
    payload["utilities"] += list(extra_utilities)
    pinch = PinchProblem(source=payload, project_name="Four-stream")
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
def test_pdm_collects_variants_with_a_choice_of_hot_utility():
    # With a utility choice the variants are capped at the minimum utility
    # cost; a cap taken from the fewest-units objective made them infeasible.
    medium_pressure_steam = {
        "name": "MPS",
        "type": "Hot",
        "t_supply": {"value": 600.0, "unit": "K"},
        "t_target": {"value": 600.0, "unit": "K"},
        "price": {"value": 60.0, "unit": "$/MWh"},
        "htc": {"value": 5.0, "unit": "kW/m^2/K"},
        "heat_flow": None,
    }
    problem = _problem(14.0, (2, 1), extra_utilities=(medium_pressure_steam,))

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    assert problem.pdm_variants
    assert all(variant.case.mSuccess == 1 for variant in problem.pdm_variants)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_variants_with_utility_choices_are_distinct_and_at_minimum_utility():
    # An unused heater or cooler binary set to 1 at zero duty must not count
    # as a new topology, and stage reduction must not trade recovery for
    # utility.
    utilities = (
        {
            "name": "MPS",
            "type": "Hot",
            "t_supply": {"value": 600.0, "unit": "K"},
            "t_target": {"value": 600.0, "unit": "K"},
            "price": {"value": 60.0, "unit": "$/MWh"},
            "htc": {"value": 5.0, "unit": "kW/m^2/K"},
            "heat_flow": None,
        },
        {
            "name": "RW",
            "type": "Cold",
            "t_supply": {"value": 290.0, "unit": "K"},
            "t_target": {"value": 300.0, "unit": "K"},
            "price": {"value": 10.0, "unit": "$/MWh"},
            "htc": {"value": 1.0, "unit": "kW/m^2/K"},
            "heat_flow": None,
        },
    )
    problem = _problem(14.0, (3, 3), extra_utilities=utilities)

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    networks = [problem.extract_network(run_id="pdm")] + [
        variant.extract_network(run_id="pdm") for variant in problem.pdm_variants
    ]
    assert len(problem.pdm_variants) >= 2
    signatures = [network_structure_signature(network) for network in networks]
    assert len(set(signatures)) == len(signatures)
    minimum = networks[0].utility_cost
    for network in networks[1:]:
        assert network.utility_cost == pytest.approx(minimum, rel=1e-6)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_stage_reduction_keeps_variants_at_minimum_utility(monkeypatch):
    # Reducing a variant's stages by their share of recovery traded recovery
    # for utility (2% on Ten-stream Ahmad with utility choices). A threshold
    # every stage falls under makes that reduction certain to be tried.
    reference = _problem(14.0, (0, 0))
    reference.get_solution(print_output=False)
    minimum = reference.extract_network(run_id="pdm").utility_cost
    monkeypatch.setattr(
        InternalHeatExchangerNetworkProblem,
        "_utilisation_threshold",
        lambda self, stage_count: 60.0,
    )
    problem = _problem(14.0, (3, 3))

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    assert problem.pdm_variants
    for variant in problem.pdm_variants:
        network = variant.extract_network(run_id="pdm")
        assert network.utility_cost == pytest.approx(minimum, rel=1e-6)


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
        "HENS_SEARCH_DEPTH": 2,
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
    assert settings.search_depth == 2

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


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_an_unconverged_fewest_units_resolve_does_not_replace_the_side(monkeypatch):
    # Couenne stopped at a limit can report success with a point that is not
    # a solution; the fewest-units side then replaced a good one.
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
        PinchDecompModel,
    )

    real_optimise = PinchDecompModel.optimise

    def stopped_at_a_limit(self, *args, **kwargs):
        result = real_optimise(self, *args, **kwargs)
        if "fewest units" in self.name:
            self.solver = "couenne"
            self.solver_run = replace(
                self.solver_run, termination_condition="maxIterations"
            )
        return result

    monkeypatch.setattr(PinchDecompModel, "optimise", stopped_at_a_limit)
    problem = _problem(14.0, (0, 0))

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    assert "fewest units" not in problem.above.name
    assert "fewest units" not in problem.below.name


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_failure_collecting_variants_keeps_the_main_network(monkeypatch):
    def broken(self, model_factories):
        raise ValueError("variant model failed")

    monkeypatch.setattr(
        InternalHeatExchangerNetworkProblem, "_solve_pdm_variants", broken
    )
    problem = _problem(14.0, (2, 1))

    solved = problem.get_solution(print_output=False)

    assert solved is not None and solved.mSuccess == 1
    assert problem.pdm_variants == []


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_stage_reduction_without_its_solver_keeps_the_case(monkeypatch):
    problem = _problem(14.0, (0, 0))
    problem.get_solution(print_output=False)
    case = problem.case

    def missing(solver_name):
        raise MissingSynthesisSolverError(f"{solver_name} is missing")

    monkeypatch.setattr(
        InternalHeatExchangerNetworkProblem,
        "_get_active_stages",
        lambda self, case, threshold=None: [0],
    )
    monkeypatch.setattr(backend, "require_solver_backend", missing)

    assert problem.remove_unused_stages(case) is case


def test_one_bad_variant_does_not_lose_the_others(monkeypatch):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.errors import (
        WorkflowContractError,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution import (
        executor as executor_module,
    )
    from OpenPinch.contracts.synthesis.task import (
        HeatExchangerNetworkSynthesisTask,
        HeatExchangerNetworkSynthesisTaskOutcome,
    )

    def solved_outcome(task, problem, case):
        if case == "bad":
            raise WorkflowContractError("verification failed")
        return HeatExchangerNetworkSynthesisTaskOutcome(task=task, status="success")

    monkeypatch.setattr(executor_module, "_solved_outcome", solved_outcome)
    task = HeatExchangerNetworkSynthesisTask(
        run_id="run", method="pinch_design_method", dt_cont_multiplier=1.0
    )
    internal = SimpleNamespace(
        pdm_variants=[SimpleNamespace(case="bad"), SimpleNamespace(case="good")]
    )

    outcomes = executor_module.LocalSynthesisExecutor()._pdm_variant_outcomes(
        task, internal
    )

    assert [outcome.status for outcome in outcomes] == ["success"]


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_minimum_unit_solves_are_time_limited(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
        PinchDecompModel,
    )

    limits = []
    real_init = PinchDecompModel.__init__

    def recording(self, *args, **kwargs):
        if "min units" in kwargs.get("name", ""):
            limits.append((kwargs.get("solver_options") or {}).get("time_limit"))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(PinchDecompModel, "__init__", recording)
    problem = _problem(14.0, (2, 1))

    problem.get_solution(print_output=False)

    assert limits and all(limit == backend.FEWEST_UNITS_TIME_LIMIT for limit in limits)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_stage_reduction_minimises_utility_cost_with_a_utility_choice(monkeypatch):
    # The reduced network was re-solved for minimum utility duty, so with a
    # choice of utilities the duty held but load could move to a dearer one
    # (0.38% over the side minimum on Thirteen-stream Kim's variants).
    from OpenPinch.analysis.heat_exchanger_networks.models import problem as module

    goals = []
    real = module.StageWiseModel

    def recording(*args, **kwargs):
        goals.append(kwargs["minimisation_goal"])
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "StageWiseModel", recording)
    medium_pressure_steam = {
        "name": "MPS",
        "type": "Hot",
        "t_supply": {"value": 600.0, "unit": "K"},
        "t_target": {"value": 600.0, "unit": "K"},
        "price": {"value": 60.0, "unit": "$/MWh"},
        "htc": {"value": 5.0, "unit": "kW/m^2/K"},
        "heat_flow": None,
    }
    monkeypatch.setattr(
        InternalHeatExchangerNetworkProblem,
        "_utilisation_threshold",
        lambda self, stage_count: 60.0,
    )
    problem = _problem(14.0, (0, 0), extra_utilities=(medium_pressure_steam,))

    problem.get_solution(print_output=False)

    assert goals and set(goals) == {"utility costs"}


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_the_one_more_unit_quota_is_filled_after_the_minimum_one():
    # With enough minimum-unit alternatives on each side, every variant had
    # the fewest units and HENS_PDM_EXTRA_UNIT_NETWORKS did nothing.
    source = REPOSITORY_ROOT / "tests/fixtures/openhens/Ten-stream-Ahmad-1985-1.json"
    problem = _problem(10.0, (3, 3), source=source)

    problem.get_solution(print_output=False)

    # Unit tiers are counted on the pinch sides.
    units = sorted(
        variant.above.n_units + variant.below.n_units
        for variant in problem.pdm_variants
    )
    assert units and units[-1] == units[0] + 1


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_the_primary_pinch_design_solve_is_time_limited(monkeypatch):
    # With a utility choice the primary MILP could run for minutes without
    # closing its gap (Ten-stream Ahmad with two extra utilities).
    from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
        PinchDecompModel,
    )

    limits = []
    real_init = PinchDecompModel.__init__

    def recording(self, *args, **kwargs):
        name = kwargs.get("name", "")
        if "units" not in name and "heuristic" not in name:
            limits.append((kwargs.get("solver_options") or {}).get("time_limit"))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(PinchDecompModel, "__init__", recording)
    problem = _problem(14.0, (0, 0))

    problem.get_solution(print_output=False)

    assert limits and all(limit == backend.PDM_TIME_LIMIT for limit in limits)


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_side_with_fewer_units_than_the_primary_is_kept():
    # The fewest-units re-solve only searches the primary's own matches, so
    # the primary can have more units than the true minimum (13 against 9
    # above the pinch on Ten-stream Ahmad); such a side was discarded.
    source = REPOSITORY_ROOT / "tests/fixtures/openhens/Ten-stream-Ahmad-1985-1.json"
    problem = _problem(10.0, (3, 3), source=source)

    problem.get_solution(print_output=False)

    # Unit tiers are counted on the pinch sides: amalgamation and stage
    # reduction change the network's own count.
    above = [variant.above.n_units for variant in problem.pdm_variants]
    assert above and min(above) < problem.above.n_units
