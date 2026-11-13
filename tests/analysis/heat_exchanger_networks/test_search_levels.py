"""Synthesis search width (dt_cont multipliers) and depth (stages and effort).

``HENS_SEARCH_WIDTH`` picks the multipliers and ``HENS_SEARCH_DEPTH`` the
stages and their effort; an option set explicitly overrides the level.
"""

from __future__ import annotations

import pytest

import OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method as workflow_module
from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.search import (
    DEPTH_LEVELS,
    WIDTH_MULTIPLIERS,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.task_builders import (
    is_pdm_variant_outcome,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerNetworkDesignMethod
from tests.analysis.heat_exchanger_networks.test_full_sequence import (
    _small_payload,
)

HEURISTIC = HeatExchangerNetworkDesignMethod.HeuristicPinchDesign
MILP = HeatExchangerNetworkDesignMethod.PinchDesign
TDM = HeatExchangerNetworkDesignMethod.ThermalDerivative
EVM = HeatExchangerNetworkDesignMethod.NetworkEvolution

# Options the levels set; the shared small payload pins some of them.
_LEVEL_OPTIONS = (
    "HENS_DT_CONT_MULTIPLIERS",
    "HENS_HEURISTIC_PDM_NETWORKS",
    "HENS_PDM_MILP",
    "HENS_SEARCH_DEPTH",
)


def _problem(options: dict) -> PinchProblem:
    payload = _small_payload()
    for key in _LEVEL_OPTIONS:
        payload["options"].pop(key, None)
    payload["options"] |= options
    return PinchProblem(source=payload, project_name="HENS Demo")


def test_the_levels_are_the_agreed_presets():
    assert WIDTH_MULTIPLIERS == {
        0: (1.0,),
        1: (0.75, 1.0, 1.5),
        2: (0.5, 0.75, 1.0, 1.5, 2.0),
        3: (0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0),
    }
    assert [
        (
            level.evolution,
            level.heuristic_networks,
            level.milp,
            level.evolution_breadth,
        )
        for level in (DEPTH_LEVELS[depth] for depth in range(5))
    ] == [
        (False, 10, False, 1),
        (True, 10, False, 1),
        (True, 10, True, 1),
        (True, 20, True, 2),
        (True, 50, True, 3),
    ]


def test_by_default_synthesis_searches_width_one_at_depth_one():
    settings = workflow_settings_from_problem(_problem({}))

    assert (settings.search_width, settings.search_depth) == (1, 1)
    assert settings.dt_cont_multipliers == (0.75, 1.0, 1.5)
    assert settings.heuristic_pdm_networks == 10
    assert settings.pdm_milp is False
    assert settings.runs_evolution is True
    assert settings.runs_thermal_derivative_method is False
    assert settings.effective_evm_n_ad_branches == 1
    assert settings.effective_evm_n_rm_branches == 1


def test_the_levels_set_the_options_left_unset():
    settings = workflow_settings_from_problem(
        _problem({"HENS_SEARCH_WIDTH": 3, "HENS_SEARCH_DEPTH": 4})
    )

    assert settings.dt_cont_multipliers == WIDTH_MULTIPLIERS[3]
    assert settings.heuristic_pdm_networks == 50
    assert settings.effective_evm_n_ad_branches == 3
    assert settings.effective_evm_n_rm_branches == 3


def test_an_option_set_explicitly_overrides_the_level():
    settings = workflow_settings_from_problem(
        _problem(
            {
                "HENS_SEARCH_WIDTH": 2,
                "HENS_SEARCH_DEPTH": 4,
                "HENS_DT_CONT_MULTIPLIERS": [1.0],
                "HENS_HEURISTIC_PDM_NETWORKS": 4,
                "HENS_PDM_MILP": False,
                "HENS_EVM_N_AD_BRANCHES": 1,
            }
        )
    )

    assert settings.dt_cont_multipliers == (1.0,)
    assert settings.heuristic_pdm_networks == 4
    assert settings.pdm_milp is False
    assert settings.effective_evm_n_ad_branches == 1
    assert settings.effective_evm_n_rm_branches == 3


@pytest.mark.parametrize(
    ("depth", "stages"),
    [
        (0, [HEURISTIC]),
        (1, [HEURISTIC, EVM]),
        (2, [MILP, HEURISTIC, EVM, TDM, EVM]),
        (3, [MILP, HEURISTIC, EVM, TDM, EVM]),
    ],
)
def test_depth_sets_the_stages_that_run(depth, stages):
    problem = _problem({"HENS_SEARCH_DEPTH": depth, "HENS_SEARCH_WIDTH": 0})
    executor = FakeSynthesisExecutor()

    workflow_module.execute_open_hens_method(
        problem, workflow_settings_from_problem(problem), executor=executor
    )

    assert executor.stage_order == stages


@pytest.mark.parametrize("depth", [2, 3])
def test_the_derivative_method_refines_only_the_main_milp_networks(depth):
    problem = _problem({"HENS_SEARCH_DEPTH": depth, "HENS_SEARCH_WIDTH": 0})

    result = workflow_module.execute_open_hens_method(
        problem,
        workflow_settings_from_problem(problem),
        executor=FakeSynthesisExecutor(),
    )

    by_id = {task.task_id: task for task in result.tasks}
    derivative_parents = {
        by_id[task.parent_task_id].method for task in result.tasks if task.method == TDM
    }
    assert derivative_parents == {MILP}
    direct_parents = {
        by_id[task.parent_task_id].method
        for task in result.tasks
        if task.method == EVM and by_id[task.parent_task_id].method != TDM
    }
    # From depth 3 every MILP network also seeds evolution directly.
    assert direct_parents == ({HEURISTIC, MILP} if depth == 3 else {HEURISTIC})


def test_depth_three_seeds_every_milp_network_into_evolution():
    problem = _problem({"HENS_SEARCH_DEPTH": 3, "HENS_SEARCH_WIDTH": 0})

    result = workflow_module.execute_open_hens_method(
        problem,
        workflow_settings_from_problem(problem),
        executor=FakeSynthesisExecutor(),
    )

    milp = [o for o in result.outcomes if o.task.method == MILP]
    seeded = {t.parent_task_id for t in result.tasks if t.method == EVM}
    assert {o.task.task_id for o in milp if not is_pdm_variant_outcome(o)} <= seeded


@pytest.mark.parametrize(
    ("options", "limits"),
    [
        ({"HENS_SEARCH_DEPTH": 4}, (0, 0)),
        (
            {
                "HENS_SEARCH_DEPTH": 2,
                "HENS_PDM_MIN_UNIT_NETWORKS": 2,
                "HENS_PDM_EXTRA_UNIT_NETWORKS": 1,
            },
            (2, 1),
        ),
    ],
)
def test_the_minimum_unit_milp_networks_come_only_from_their_options(options, limits):
    problem = _problem({"HENS_SEARCH_WIDTH": 0, **options})
    problem.target.all_heat_integration()
    settings = workflow_settings_from_problem(problem)
    from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
        build_pinch_design_method_tasks,
    )

    (task,) = build_pinch_design_method_tasks(settings)
    internal = LocalSynthesisExecutor()._build_problem(
        task, problem=problem, parent_outcomes={}
    )

    assert internal.pdm_network_limits == limits


def test_the_design_accessor_takes_width_and_depth(monkeypatch):
    import OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method as module

    monkeypatch.setattr(module, "LocalSynthesisExecutor", FakeSynthesisExecutor)
    problem = _problem({})

    design = problem.design.heat_exchanger_network(width=0, depth=1)

    manifest = design.result.manifest
    assert (manifest.search_width, manifest.search_depth) == (0, 1)
    assert manifest.dt_cont_multipliers == (1.0,)
    assert not hasattr(problem.design, "enhanced_heat_exchanger_network")
    assert not hasattr(problem.design, "open_hens")


def test_the_quality_tier_option_is_gone():
    from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS

    assert "HENS_SYNTHESIS_QUALITY_TIER" not in USER_CONFIG_FIELD_SPECS
    assert {"HENS_SEARCH_WIDTH", "HENS_SEARCH_DEPTH"} <= set(USER_CONFIG_FIELD_SPECS)


@pytest.mark.parametrize(
    ("multipliers", "expected"),
    [((0.75, 1.0, 1.5), 1.0), ((1.5, 1.0), 1.0), ((0.5, 2.0), 0.5)],
)
def test_fallbacks_run_at_1_whenever_it_is_one_of_the_multipliers(
    multipliers, expected
):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
        duty_optimisation_task,
    )
    from OpenPinch.analysis.heat_exchanger_networks.execution.task_builders import (
        dt_cont_multiplier_from_network,
    )
    from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
        build_pinch_design_method_tasks,
    )
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    settings = workflow_settings_from_problem(
        _problem({"HENS_DT_CONT_MULTIPLIERS": list(multipliers)})
    )
    settings = type(settings)(**{**settings.__dict__, "pdm_stage_pair_limit": 1})

    # A user network with no recorded multiplier, stage-pair candidates and
    # duty optimisation all fall back to the reference multiplier.
    assert dt_cont_multiplier_from_network(HeatExchangerNetwork(), settings) == expected
    stage_pair = [
        task
        for task in build_pinch_design_method_tasks(settings)
        if task.settings.get("stage_selection")
    ]
    assert [task.dt_cont_multiplier for task in stage_pair] == [expected]
    duty_settings = SimpleNamespace(
        run_id="run",
        dt_cont_multipliers=multipliers,
        problem_id=None,
        workspace_variant=None,
        period_id=None,
    )
    from tests.analysis.heat_exchanger_networks.test_duty_optimisation import (
        _request,
    )

    assert duty_optimisation_task(_request(), duty_settings).dt_cont_multiplier == (
        expected
    )


@pytest.mark.parametrize("depth", [0, 1])
def test_the_milp_runs_where_the_heuristic_realises_no_network(depth):
    from OpenPinch.analysis.heat_exchanger_networks.targeting.heuristic_pinch_design_method import (
        build_heuristic_pinch_design_method_tasks,
    )

    problem = _problem({"HENS_SEARCH_DEPTH": depth, "HENS_SEARCH_WIDTH": 1})
    settings = workflow_settings_from_problem(problem)
    # The heuristic fails at 0.75 only.
    failing = {
        task.task_id
        for task in build_heuristic_pinch_design_method_tasks(settings)
        if task.dt_cont_multiplier == 0.75
    }
    executor = FakeSynthesisExecutor(failures=failing)

    result = workflow_module.execute_open_hens_method(
        problem, settings, executor=executor
    )

    milp = [task for task in result.tasks if task.method == MILP]
    assert [task.dt_cont_multiplier for task in milp] == [0.75]
    assert executor.stage_order[:2] == [HEURISTIC, MILP]
    if depth == 1:
        evolution_parents = {
            task.parent_task_id for task in result.tasks if task.method == EVM
        }
        assert milp[0].task_id in evolution_parents
    assert TDM not in executor.stage_order
