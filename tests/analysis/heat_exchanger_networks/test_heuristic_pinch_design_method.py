"""The heuristic pinch design method, the default pinch design method.

Heuristic networks are labelled with their own method, fan out like the
extra PDM networks, seed network evolution and never go to the thermal
derivative method. The MILP pinch design method runs from search depth 2,
when ``HENS_PDM_MILP`` is set, or when ``HENS_HEURISTIC_PDM_NETWORKS`` is 0.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method as workflow_module
from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.heuristic_pinch_design_method import (
    build_heuristic_pinch_design_method_tasks,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method import (
    build_network_evolution_method_tasks_from_pinch_design_method,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
    _execute_pinch_design_method_workflow,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerNetworkDesignMethod
from tests.analysis.heat_exchanger_networks.test_full_sequence import (
    _small_payload,
)
from tests.support.paths import REPOSITORY_ROOT

HEURISTIC = HeatExchangerNetworkDesignMethod.HeuristicPinchDesign
MILP = HeatExchangerNetworkDesignMethod.PinchDesign
FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


BASIC_PINCH_JSON = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/basic_pinch.json"


def _small_problem_with_options(options: dict) -> PinchProblem:
    """The small test problem with the default pinch design options."""
    payload = _small_payload()
    del payload["options"]["HENS_HEURISTIC_PDM_NETWORKS"]
    payload["options"] |= options
    return PinchProblem(source=payload, project_name="HENS Demo")


def _pinch_problem(source, options: dict) -> PinchProblem:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["options"] = payload.get("options", {}) | {
        # The heuristic alone.
        "HENS_SEARCH_DEPTH": 1,
        "HENS_SOLVER_PDM": "highs",
        "HENS_MAX_PARALLEL": 1,
        **options,
    }
    pinch = PinchProblem(source=payload, project_name="heuristic")
    pinch.target.all_heat_integration()
    return pinch


def test_tasks_run_one_per_multiplier_with_the_network_cap():
    settings = replace(
        workflow_settings_from_problem(
            _small_problem_with_options({"HENS_DT_CONT_MULTIPLIERS": [0.5, 1.0]})
        ),
        heuristic_pdm_networks=4,
    )

    tasks = build_heuristic_pinch_design_method_tasks(settings)

    assert [task.method for task in tasks] == [HEURISTIC, HEURISTIC]
    assert [task.dt_cont_multiplier for task in tasks] == [0.5, 1.0]
    assert all(task.settings["heuristic_networks"] == 4 for task in tasks)


def test_the_heuristic_gives_ten_networks_and_depth_two_adds_the_milp():
    settings = workflow_settings_from_problem(_small_problem_with_options({}))
    shallow = workflow_settings_from_problem(
        _small_problem_with_options({"HENS_SEARCH_DEPTH": 1})
    )

    assert settings.heuristic_pdm_networks == 10
    assert settings.pdm_milp is True
    assert shallow.pdm_milp is False
    assert shallow.runs_milp_pinch_design is False
    assert replace(shallow, pdm_milp=True).runs_milp_pinch_design
    # With no heuristic networks the MILP runs, or there would be no pinch
    # design networks at all.
    assert replace(shallow, heuristic_pdm_networks=0).runs_milp_pinch_design


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_standalone_run_gives_distinct_verified_networks_labelled_heuristic():
    problem = _pinch_problem(
        FOUR_STREAM_JSON, {"HENS_DT_CONT_MULTIPLIERS": [0.6, 1.0, 1.4]}
    )
    settings = workflow_settings_from_problem(problem)

    result = _execute_pinch_design_method_workflow(
        problem, settings, executor=LocalSynthesisExecutor()
    )

    solved = [o for o in result.outcomes if o.status == "success"]
    assert len(solved) >= 2
    assert all(o.task.method == HEURISTIC for o in result.outcomes)
    assert all(o.network.method == HEURISTIC.value for o in solved)
    keys = {
        (o.task.dt_cont_multiplier, network_structure_signature(o.network))
        for o in solved
    }
    assert len(keys) == len(solved)
    for outcome in solved:
        assert verify_network_feasibility(outcome.network) == ()
    assert result.accepted_result.method == HEURISTIC
    assert result.accepted_result.design_method == MILP


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_extra_heuristic_networks_get_their_own_variant_tasks():
    problem = _pinch_problem(BASIC_PINCH_JSON, {"HENS_DT_CONT_MULTIPLIERS": [1.0]})
    executor = LocalSynthesisExecutor()

    result = _execute_pinch_design_method_workflow(
        problem, workflow_settings_from_problem(problem), executor=executor
    )

    variants = [o for o in result.outcomes if "heuristic_variant" in o.task.settings]
    assert variants and all(o.status == "success" for o in variants)
    assert len({o.task.task_id for o in result.outcomes}) == len(result.outcomes)
    for outcome in variants:
        assert outcome.task.task_id in executor.problems_by_task_id


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_heuristic_outcomes_seed_evolution_and_skip_the_derivative_method(depth):
    problem = _small_problem_with_options(
        {"HENS_SEARCH_DEPTH": depth, "HENS_DT_CONT_MULTIPLIERS": [1.0]}
    )
    settings = replace(
        workflow_settings_from_problem(problem), heuristic_pdm_networks=3
    )
    executor = FakeSynthesisExecutor()

    result = workflow_module.execute_open_hens_method(
        problem, settings, executor=executor
    )

    assert (MILP in executor.stage_order) == (depth >= 2)
    heuristic_ids = {t.task_id for t in result.tasks if t.method == HEURISTIC}
    assert heuristic_ids
    evolution_parents = {
        t.parent_task_id
        for t in result.tasks
        if t.method == HeatExchangerNetworkDesignMethod.NetworkEvolution
    }
    assert heuristic_ids <= evolution_parents
    derivative_parents = {
        t.parent_task_id
        for t in result.tasks
        if t.method == HeatExchangerNetworkDesignMethod.ThermalDerivative
    }
    assert not heuristic_ids & derivative_parents


def test_the_milp_option_runs_the_milp_before_the_heuristic():
    problem = _small_problem_with_options(
        {"HENS_DT_CONT_MULTIPLIERS": [1.0], "HENS_PDM_MILP": True}
    )
    executor = FakeSynthesisExecutor()

    result = workflow_module.execute_open_hens_method(
        problem, workflow_settings_from_problem(problem), executor=executor
    )

    assert executor.stage_order[:2] == [MILP, HEURISTIC]
    milp_ids = {t.task_id for t in result.tasks if t.method == MILP}
    derivative_parents = {
        t.parent_task_id
        for t in result.tasks
        if t.method == HeatExchangerNetworkDesignMethod.ThermalDerivative
    }
    assert milp_ids & derivative_parents


def test_without_a_network_cap_the_workflow_runs_only_the_milp():
    problem = _small_problem_with_options(
        {"HENS_DT_CONT_MULTIPLIERS": [1.0], "HENS_HEURISTIC_PDM_NETWORKS": 0}
    )
    executor = FakeSynthesisExecutor()

    workflow_module.execute_open_hens_method(
        problem, workflow_settings_from_problem(problem), executor=executor
    )

    assert executor.stage_order[0] == MILP
    assert HEURISTIC not in executor.stage_order


def test_evolution_takes_heuristic_parents_at_every_tier():
    problem = _small_problem_with_options(
        {"HENS_SEARCH_DEPTH": 4, "HENS_DT_CONT_MULTIPLIERS": [1.0]}
    )
    settings = replace(
        workflow_settings_from_problem(problem), heuristic_pdm_networks=2
    )
    (outcome,) = FakeSynthesisExecutor().execute(
        build_heuristic_pinch_design_method_tasks(settings),
        problem=problem,
        parent_outcomes={},
        max_parallel=1,
    )

    (task,) = build_network_evolution_method_tasks_from_pinch_design_method(
        (outcome,), settings=settings
    )

    assert task.parent_task_id == outcome.task.task_id
    assert task.method == HeatExchangerNetworkDesignMethod.NetworkEvolution


def test_heuristic_tasks_build_pinch_design_problems_with_the_cap():
    problem = _small_problem_with_options({"HENS_DT_CONT_MULTIPLIERS": [1.0]})
    problem.target.all_heat_integration()
    settings = replace(
        workflow_settings_from_problem(problem), heuristic_pdm_networks=5
    )
    (task,) = build_heuristic_pinch_design_method_tasks(settings)

    internal = LocalSynthesisExecutor()._build_problem(
        task, problem=problem, parent_outcomes={}
    )

    assert internal.framework == "PDM"
    assert internal.heuristic_network_limit == 5
    assert internal.pdm_network_limits == (0, 0)
    assert set(internal.pinch_decompositions) == {"above", "below"}
    assert internal.solver == str(problem.master_zone.config.hens.solver_pdm)


@pytest.mark.parametrize(
    ("options", "stages"),
    [
        ({"HENS_SEARCH_DEPTH": 1}, [HEURISTIC]),
        ({"HENS_SEARCH_DEPTH": 1, "HENS_PDM_MILP": True}, [MILP, HEURISTIC]),
        ({}, [MILP, HEURISTIC]),
    ],
)
def test_the_pinch_design_service_runs_the_heuristic_and_optionally_the_milp(
    options, stages
):
    from OpenPinch.analysis.heat_exchanger_networks.service import (
        heat_exchanger_network_synthesis_service,
    )

    problem = _small_problem_with_options(
        {"HENS_DT_CONT_MULTIPLIERS": [1.0], **options}
    )
    executor = FakeSynthesisExecutor()

    result = heat_exchanger_network_synthesis_service(
        problem, method=MILP.value, executor=executor
    )

    assert executor.stage_order == stages
    assert result.design_method == MILP


def test_the_heuristic_is_not_a_design_method_of_its_own():
    from OpenPinch.analysis.heat_exchanger_networks.service import (
        heat_exchanger_network_synthesis_service,
    )

    problem = _small_problem_with_options({"HENS_DT_CONT_MULTIPLIERS": [1.0]})

    with pytest.raises(ValueError, match="pinch_design_method"):
        heat_exchanger_network_synthesis_service(
            problem, method=HEURISTIC, executor=FakeSynthesisExecutor()
        )


def test_the_network_cap_comes_from_the_configuration():
    problem = _small_problem_with_options(
        {"HENS_DT_CONT_MULTIPLIERS": [1.0], "HENS_HEURISTIC_PDM_NETWORKS": 4}
    )
    executor = FakeSynthesisExecutor()

    settings = workflow_settings_from_problem(problem)
    workflow_module.execute_open_hens_method(problem, settings, executor=executor)

    assert settings.heuristic_pdm_networks == 4
    assert HEURISTIC in executor.stage_order
    assert (
        workflow_settings_from_problem(
            _small_problem_with_options({})
        ).heuristic_pdm_networks
        == 10
    )


def test_the_pinch_design_accessor_runs_the_heuristic_method(monkeypatch):
    import OpenPinch.analysis.heat_exchanger_networks.targeting.heuristic_pinch_design_method as heuristic_module

    monkeypatch.setattr(
        heuristic_module, "LocalSynthesisExecutor", FakeSynthesisExecutor
    )
    problem = _small_problem_with_options(
        {"HENS_DT_CONT_MULTIPLIERS": [1.0], "HENS_SEARCH_DEPTH": 1}
    )

    design = problem.design.pinch_design()

    assert design.selected_network.method == HEURISTIC
    assert not hasattr(problem.design, "heuristic_pinch_design")


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_default_pinch_design_gives_a_network_where_the_heuristic_cannot():
    # Six-stream Yee and Grossmann: no heuristic proposal is realised at the
    # default multipliers, so the MILP runs in its place.
    payload = json.loads(
        (
            REPOSITORY_ROOT
            / "tests/fixtures/openhens/Six-stream-Yee-and-Grossmann-1990-1.json"
        ).read_text(encoding="utf-8")
    )
    for key in ("HENS_DT_CONT_MULTIPLIERS", "HENS_SOLVER_PDM"):
        payload["options"].pop(key, None)
    payload["options"] |= {"HENS_OUTPUT_FORMATS": [], "HENS_MAX_PARALLEL": 1}
    problem = PinchProblem(source=payload, project_name="six-stream")

    design = problem.design.heat_exchanger_network(depth=0)

    assert design.selected_network.exchangers
    assert verify_network_feasibility(design.selected_network) == ()
