"""Private heat exchanger network synthesis model-boundary tests."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.classes.heat_exchanger import HeatExchangerKind
from OpenPinch.lib.config import tol
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import backend
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.arrays import (
    PreparedSolverArrays,
    problem_to_solver_arrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.extraction import (
    extract_heat_exchanger_network,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.network_evolution_method import (
    build_network_evolution_method_tasks,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.pinch_design_method import (
    build_pinch_design_decomposition,
    build_pinch_design_method_tasks,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.thermal_derivative_method import (
    build_thermal_derivative_method_tasks,
)
from OpenPinch.services.heat_exchanger_network_synthesis.unit_models.pinch_design import (
    PinchDecompModel,
)
from OpenPinch.services.heat_exchanger_network_synthesis.unit_models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.services.heat_exchanger_network_synthesis.unit_models.stagewise import (
    StageWiseModel,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FOUR_STREAM_JSON = (
    REPO_ROOT / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)
SYNTHESIS_ONLY_MODULES = [
    "gekko",
    "pyomo",
    "pyomo.environ",
    "pyomo.opt",
    "plotly",
    "plotly.graph_objects",
    "kaleido",
    "openpyxl",
    "wakepy",
]


def test_models_package_import_is_optional_and_lazy() -> None:
    script = f"""
import sys

import OpenPinch
from OpenPinch.services.heat_exchanger_network_synthesis.unit_models import (
    InternalHeatExchangerNetworkProblem,
)

_ = InternalHeatExchangerNetworkProblem

forbidden = {SYNTHESIS_ONLY_MODULES!r}
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit(f"model boundary loaded optional modules: {{loaded}}")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_create_gekko_model_reports_missing_optional_dependency(monkeypatch) -> None:
    def missing_dependency(*_args, **_kwargs):
        raise MissingSynthesisDependencyError(
            "gekko is required for GEKKO heat exchanger network equation model construction. "
            'Install "openpinch[synthesis]".'
        )

    monkeypatch.setattr(backend, "require_synthesis_dependency", missing_dependency)

    with pytest.raises(
        MissingSynthesisDependencyError,
        match=r"gekko.*GEKKO heat exchanger network equation model construction.*openpinch\[synthesis\]",
    ):
        backend.create_gekko_model()


def test_pyomo_solver_configuration_reports_missing_binary_before_factory_import(
    monkeypatch,
) -> None:
    def missing_binary(binary_name: str, *, purpose: str | None = None) -> str:
        raise MissingSynthesisSolverError(
            f"The {binary_name!r} solver executable is required for {purpose}, "
            "but it was not found on PATH."
        )

    monkeypatch.setattr(backend, "require_solver_binary", missing_binary)

    with pytest.raises(
        MissingSynthesisSolverError,
        match=r"couenne.*couenne heat exchanger network synthesis solves.*PATH",
    ):
        backend.require_solver_backend("couenne")


def test_gekko_solver_configuration_preserves_source_defaults(monkeypatch) -> None:
    calls = []

    def present_dependency(import_name: str, **_kwargs):
        calls.append(import_name)
        return object()

    monkeypatch.setattr(backend, "require_synthesis_dependency", present_dependency)
    model = _FakeGekkoModel()

    run = backend.configure_gekko_solver(model, "apopt")

    assert run.name == "apopt"
    assert run.extension == 0
    assert model.options.SOLVER == "apopt"
    assert model.options.SOLVER_EXTENSION == 0
    assert model.options.MAX_ITER == 1000
    assert model.options.RTOL == 1e-2
    assert model.options.OTOL == 1e-2
    assert calls == ["gekko"]


def test_user_couenne_options_are_written_to_model_run_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def present_binary(binary_name: str, *, purpose: str | None = None) -> str:
        del purpose
        return f"/usr/local/bin/{binary_name}"

    class _FakeSolverFactory:
        def available(self, exception_flag: bool = False) -> bool:
            del exception_flag
            return True

    def present_dependency(import_name: str, **_kwargs):
        if import_name == "pyomo.environ":
            return SimpleNamespace(SolverFactory=lambda _name: _FakeSolverFactory())
        return object()

    monkeypatch.setattr(backend, "require_solver_binary", present_binary)
    monkeypatch.setattr(backend, "require_synthesis_dependency", present_dependency)
    model = _FakeGekkoModel(tmp_path / "gekko-run")

    run = backend.configure_gekko_solver(
        model,
        "couenne",
        solver_options={
            "node_limit": 50,
            "time_limit": 120,
            "delete_redundant": "no",
        },
    )

    option_file = tmp_path / "gekko-run" / "couenne.opt"
    couenne_options = option_file.read_text(encoding="utf-8")

    assert run.option_file == str(option_file)
    assert run.solver_options == {
        "node_limit": 50,
        "delete_redundant": "no",
        "time_limit": 120,
    }
    assert model.options.SOLVER == "couenne"
    assert model.options.SOLVER_EXTENSION == "pyomo"
    assert "node_limit 50" in couenne_options
    assert "node_limit 2000" not in couenne_options
    assert "feas_tolerance" not in couenne_options
    assert "allowable_gap" not in couenne_options
    assert "allowable_fraction_gap" not in couenne_options
    assert "delete_redundant no" in couenne_options
    assert "time_limit 120" in couenne_options

    original_cwd = Path.cwd()
    solve_run = backend.solve_gekko_model(model, solver_name="couenne")

    assert Path.cwd() == original_cwd
    assert model.cwd_during_solve == option_file.parent
    assert solve_run.option_file == str(option_file)
    assert solve_run.failure_reason is None


def test_couenne_without_user_options_matches_source_no_option_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def present_binary(binary_name: str, *, purpose: str | None = None) -> str:
        del purpose
        return f"/usr/local/bin/{binary_name}"

    class _FakeSolverFactory:
        def available(self, exception_flag: bool = False) -> bool:
            del exception_flag
            return True

    def present_dependency(import_name: str, **_kwargs):
        if import_name == "pyomo.environ":
            return SimpleNamespace(SolverFactory=lambda _name: _FakeSolverFactory())
        return object()

    monkeypatch.setattr(backend, "require_solver_binary", present_binary)
    monkeypatch.setattr(backend, "require_synthesis_dependency", present_dependency)
    model = _FakeGekkoModel(tmp_path / "gekko-run")

    run = backend.configure_gekko_solver(model, "couenne")

    assert run.option_file is None
    assert run.solver_options is None
    assert not (tmp_path / "gekko-run" / "couenne.opt").exists()

    original_cwd = Path.cwd()
    solve_run = backend.solve_gekko_model(model, solver_name="couenne")

    assert Path.cwd() == original_cwd
    assert model.cwd_during_solve == original_cwd
    assert solve_run.option_file is None
    assert solve_run.solver_options is None


def test_internal_problem_runs_stage_reduction_before_evolution(monkeypatch) -> None:
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=_solver_arrays(),
        framework="ESM",
        stages=2,
    )
    case = _EvolutionCase()

    def fake_load_model(*, model_factories=None):
        del model_factories
        problem.case = case

    monkeypatch.setattr(problem, "load_model", fake_load_model)

    solved = problem.get_solution(print_output=False, evolution=True)

    assert solved is case
    assert case.calls == ["optimise", "evolution"]


def test_stage_reduction_active_stage_selection_uses_source_thresholds() -> None:
    problem = InternalHeatExchangerNetworkProblem(solver_arrays=_solver_arrays())
    case = _StageReductionCase(stage_duties=[90.0, 4.0, 6.0, 0.0])

    assert problem._utilisation_threshold(3) == 0.1
    assert problem._utilisation_threshold(5) == 5.0
    assert problem._utilisation_threshold(8) == 8.0
    assert problem._utilisation_threshold(10) == 10.0
    assert problem._utilisation_threshold(11) == 0.0
    assert problem._get_active_stages(case) == [0, 2]


def test_stagewise_evolution_candidate_selection_and_match_ranking() -> None:
    model = StageWiseModel.__new__(StageWiseModel)
    failed = _CandidateModel(mSuccess=0, TAC=999.0)
    minus = _CandidateModel(mSuccess=1, TAC=90.0)
    plus = _CandidateModel(mSuccess=1, TAC=80.0)

    assert model._select_best_candidate(model, minus, plus) is plus
    assert model._select_best_candidate(model, minus, failed) is minus
    assert model._select_best_candidate(model, failed, failed) is None

    ranking_model = StageWiseModel.__new__(StageWiseModel)
    ranking_model.I = 1
    ranking_model.J = 2
    ranking_model.S = 2
    ranking_model.z = [[[[1], [0]], [[1], [0]]]]
    ranking_model.Q_r = [[[[10.0], [0.0]], [[20.0], [0.0]]]]
    ranking_model.alpha = [[[[0.1], [0.0]], [[0.2], [0.0]]]]
    ranking_model.hu_cost = [5.0]
    ranking_model.cu_cost = [5.0]
    ranking_model.unit_cost = [50.0]
    ranking_model.A_coeff = [2.0]
    ranking_model.A_exp = [1.0]
    ranking_model.area_r = [[[5.0, 0.0], [15.0, 0.0]]]
    ranking_model.alpha_dqda = [[[0.0, 3.0], [0.0, 9.0]]]
    ranking_model.z_feasible = [[[1, 1], [1, 1]]]
    ranking_model.tol = 1e-3

    assert ranking_model.get_lowest_benefit_HX() == [[0, 0, 0]]
    assert ranking_model.get_lowest_benefit_HX_candidates(2) == [
        [0, 0, 0],
        [0, 1, 0],
    ]
    assert ranking_model.get_max_benefit_HX() == [[0, 1, 1]]
    assert ranking_model.get_max_benefit_HX_candidates(2) == [
        [0, 1, 1],
        [0, 0, 1],
    ]


def test_stagewise_source_evolution_solves_one_add_and_one_remove_candidate() -> None:
    model = _branching_root_model(
        rm_candidates=((0, 0, 0), (0, 0, 1)),
        add_candidates=((0, 0, 2), (0, 0, 3)),
    )
    solved: list[str] = []

    def solve_minus(**kwargs):
        solved.append(kwargs.get("branch_label"))
        return _BranchEvolutionCase(
            name="minus",
            tac=90.0,
            z=kwargs["z_allowed_removed"],
        )

    def solve_plus(**kwargs):
        solved.append(kwargs.get("branch_label"))
        return _BranchEvolutionCase(
            name="plus",
            tac=95.0,
            z=kwargs["z_allowed_added"],
        )

    model._build_and_solve_n_minus_one_evolution = solve_minus
    model._build_and_solve_n_plus_one_evolution = solve_plus

    model.get_net_benefit_evolution(print_output=False, max_depth=1)

    assert solved == [None, None]
    assert model.updated_with == "minus"


def test_stagewise_evolution_branch_frontier_selects_best_global_tac() -> None:
    model = _branching_root_model(
        rm_candidates=((0, 0, 0), (0, 0, 1)),
        add_candidates=((0, 0, 2), (0, 0, 3)),
    )
    solved: list[str] = []

    def solve_minus(**kwargs):
        solved.append(kwargs["branch_label"])
        return _BranchEvolutionCase(
            name=kwargs["branch_label"],
            tac={
                "0-b0-minus1": 120.0,
                "0-b0-minus2": 80.0,
            }[kwargs["branch_label"]],
            z=kwargs["z_allowed_removed"],
        )

    def solve_plus(**kwargs):
        solved.append(kwargs["branch_label"])
        label = kwargs["branch_label"]
        if label == "0-b0-plus1":
            return _BranchEvolutionCase(
                name=label,
                tac=90.0,
                z=kwargs["z_allowed_added"],
                add_candidates=((0, 0, 3),),
            )
        if label == "0-b0-plus2":
            return _BranchEvolutionCase(
                name=label,
                tac=110.0,
                z=kwargs["z_allowed_added"],
            )
        return _BranchEvolutionCase(
            name="plus-branch-best",
            tac=70.0,
            z=kwargs["z_allowed_added"],
        )

    model._build_and_solve_n_minus_one_evolution = solve_minus
    model._build_and_solve_n_plus_one_evolution = solve_plus

    model.get_net_benefit_evolution(
        print_output=False,
        max_depth=2,
        n_ad_branches=2,
        n_rm_branches=2,
    )

    assert solved == [
        "0-b0-minus1",
        "0-b0-minus2",
        "0-b0-plus1",
        "0-b0-plus2",
        "1-b2-plus1",
    ]
    assert model.updated_with == "plus-branch-best"


def test_stagewise_evolution_failed_child_does_not_stop_siblings() -> None:
    model = _branching_root_model(
        rm_candidates=((0, 0, 0),),
        add_candidates=((0, 0, 2),),
    )
    solved: list[str] = []

    def solve_minus(**kwargs):
        solved.append(kwargs["branch_label"])
        raise RuntimeError("forced branch failure")

    def solve_plus(**kwargs):
        solved.append(kwargs["branch_label"])
        return _BranchEvolutionCase(
            name="usable-plus",
            tac=90.0,
            z=kwargs["z_allowed_added"],
        )

    model._build_and_solve_n_minus_one_evolution = solve_minus
    model._build_and_solve_n_plus_one_evolution = solve_plus

    model.get_net_benefit_evolution(
        print_output=False,
        max_depth=1,
        no_improvement_patience=2,
    )

    assert solved == ["0-b0-minus1", "0-b0-plus1"]
    assert model.updated_with == "usable-plus"


def test_solver_arrays_include_evm_branch_options() -> None:
    arrays = problem_to_solver_arrays(
        _four_stream_problem(
            options={
                "HENS_EVM_N_AD_BRANCHES": 2,
                "HENS_EVM_N_RM_BRANCHES": 3,
            }
        ),
        14.0,
    )

    assert arrays.configuration["HENS_EVM_N_AD_BRANCHES"] == 2
    assert arrays.configuration["HENS_EVM_N_RM_BRANCHES"] == 3


def test_solver_arrays_expand_period_stream_data_and_weights() -> None:
    arrays = problem_to_solver_arrays(_two_state_problem(), 14.0)

    assert arrays.axis_maps["periods"] == {"base": 0, "peak": 1}
    assert arrays.arrays["period_weights"].tolist() == [1.0, 3.0]
    assert arrays.arrays["f_h_period"].shape[0] == 2
    assert "f_h" not in arrays.arrays
    assert arrays.arrays["f_h_period"][0][0] == pytest.approx(10.0)
    assert arrays.arrays["f_h_period"][1][0] == pytest.approx(20.0)
    assert arrays.arrays["T_h_in_period"][1][0] == pytest.approx(660.0)


def test_solver_arrays_represent_single_state_as_one_state_row() -> None:
    arrays = problem_to_solver_arrays(_four_stream_problem(), 14.0)

    assert arrays.axis_maps["periods"] == {"0": 0}
    assert arrays.arrays["period_ids"].tolist() == ["0"]
    assert arrays.arrays["period_weights"].tolist() == [1.0]
    assert arrays.arrays["T_h_in_period"].shape[0] == 1
    assert arrays.arrays["T_c_in_period"].shape[0] == 1
    assert "T_h_in" not in arrays.arrays
    assert "T_c_in" not in arrays.arrays


def test_stagewise_model_rejects_solver_arrays_without_state_metadata() -> None:
    arrays = problem_to_solver_arrays(_four_stream_problem(), 14.0)
    legacy_arrays = {
        key: value
        for key, value in arrays.arrays.items()
        if key not in {"period_ids", "period_weights"}
    }
    bad_arrays = PreparedSolverArrays(
        arrays=legacy_arrays,
        axis_maps=arrays.axis_maps,
        unit_conventions=arrays.unit_conventions,
        stream_identities=arrays.stream_identities,
        utility_identities=arrays.utility_identities,
        configuration=arrays.configuration,
        preparation=arrays.preparation,
    )

    with pytest.raises(ValueError, match="period_ids is required"):
        StageWiseModel(
            name="legacy-arrays",
            framework="ESM",
            solver="apopt",
            solver_arrays=bad_arrays,
            stages=1,
            dTmin=14.0,
            z_restriction=None,
            min_dqda=0.0,
            minimisation_goal="variable total cost",
            non_isothermal_model=False,
            integers=True,
            tol=1e-3,
        )


def test_solver_arrays_reject_zero_total_period_weight() -> None:
    with pytest.raises(ValueError, match="positive finite period-weight sum"):
        problem_to_solver_arrays(
            _two_state_problem(
                options={"PROBLEM_PERIOD_WEIGHTS": [0.0, 0.0]},
            ),
            14.0,
        )


def test_stagewise_multiperiod_cost_objective_uses_shared_topology_and_area() -> None:
    arrays = problem_to_solver_arrays(
        _two_state_problem(
            options={
                "COSTING_HX_AREA_EXP": 1.0,
                "COSTING_HX_AREA_COEFF": 150.0,
                "COSTING_HX_UNIT_COST": 5500.0,
            }
        ),
        14.0,
    )

    model = StageWiseModel(
        name="two-state-cost",
        framework="ESM",
        solver="apopt",
        solver_arrays=arrays,
        stages=1,
        dTmin=14.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="variable total cost",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
    )

    assert model.N_periods == 2
    assert len(model.Q_r_by_period) == 2
    assert model.Q_r_by_period[0][0][0][0] is not model.Q_r_by_period[1][0][0][0]
    assert len(model.z) == model.I
    assert not isinstance(model.z[0][0][0], list)
    assert hasattr(model, "area_r_shared")
    assert hasattr(model, "capital_cost_total")
    assert hasattr(model, "weighted_operating_cost")
    assert model._weighted_state_average([10.0, 20.0]) == pytest.approx(17.5)


def test_internal_problem_loads_pdm_and_stagewise_with_parent_context() -> None:
    pdm_problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(_four_stream_problem(), 14.0),
        name="pdm",
        framework="PDM",
        solver="apopt",
        dTmin=14.0,
        pinch_decompositions=_pdm_decompositions(_four_stream_problem(), dTmin=14.0),
    )
    parent = InternalHeatExchangerNetworkProblem(
        solver_arrays=_solver_arrays(),
        synthesis_task_id="parent-task",
    )
    parent.case = _ParentCase()
    child = InternalHeatExchangerNetworkProblem(
        solver_arrays=_solver_arrays(),
        name="child",
        framework="ESM",
        parent=parent,
    )

    pdm_problem.load_model()
    child.load_model(model_factories={"stagewise": _RecordingStageWise})

    assert pdm_problem.above.pinch_loc == "above"
    assert pdm_problem.below.pinch_loc == "below"
    assert child.case.stages == parent.case.stages
    assert child.case.initialised_from is parent.case


def test_local_executor_preserves_parent_links_and_esm_only_evolution(
    monkeypatch,
) -> None:
    problem = _four_stream_problem(
        options={
            "HENS_APPROACH_TEMPERATURES": [14.0],
            "HENS_DERIVATIVE_THRESHOLDS": [0.5],
            "HENS_STAGE_SELECTION": [3],
            "HENS_RUN_ID": "hens-07-parent-test",
        }
    )
    settings = workflow_settings_from_problem(problem)
    executor = LocalSynthesisExecutor(evolution=True)
    parent_links = []
    evolution_flags = []

    def fake_get_solution(
        self, *, print_output=True, evolution=None, model_factories=None
    ):
        del print_output, model_factories
        evolution_flags.append((self.synthesis_task_id, evolution))
        if self.parent is not None:
            parent_links.append((self.synthesis_task_id, self.parent.synthesis_task_id))
        self.case = _SolvedCase()
        self.case.S = self.stages or 3
        self.case.stages = self.case.S
        self.case.mSuccess = 1
        return self.case

    monkeypatch.setattr(
        InternalHeatExchangerNetworkProblem,
        "get_solution",
        fake_get_solution,
    )
    pdm_tasks = build_pinch_design_method_tasks(settings)
    pdm_outcomes = executor.execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=1,
    )
    tdm_tasks = build_thermal_derivative_method_tasks(settings, pdm_outcomes)
    tdm_outcomes = executor.execute(
        tdm_tasks,
        problem=problem,
        parent_outcomes={
            outcome.task.task_id: outcome
            for outcome in pdm_outcomes
            if outcome.task.task_id is not None
        },
        max_parallel=1,
    )
    upstream_outcomes = {
        outcome.task.task_id: outcome
        for outcome in (*pdm_outcomes, *tdm_outcomes)
        if outcome.task.task_id is not None
    }
    esm_tasks = build_network_evolution_method_tasks(settings, tdm_outcomes)
    esm_outcomes = executor.execute(
        esm_tasks,
        problem=problem,
        parent_outcomes=upstream_outcomes,
        max_parallel=1,
    )

    assert pdm_outcomes[0].status == "success"
    assert tdm_outcomes[0].status == "success"
    assert esm_outcomes[0].status == "success"
    assert parent_links == [
        (tdm_tasks[0].task_id, pdm_tasks[0].task_id),
        (esm_tasks[0].task_id, tdm_tasks[0].task_id),
    ]
    assert evolution_flags == [
        (pdm_tasks[0].task_id, False),
        (tdm_tasks[0].task_id, False),
        (esm_tasks[0].task_id, True),
    ]


def test_local_executor_honours_max_parallel_for_stage_tasks(monkeypatch) -> None:
    problem = _four_stream_problem(
        options={
            "HENS_APPROACH_TEMPERATURES": [14.0, 16.0],
            "HENS_DERIVATIVE_THRESHOLDS": [0.5],
            "HENS_STAGE_SELECTION": [3],
            "HENS_RUN_ID": "hens-parallel-test",
        }
    )
    settings = workflow_settings_from_problem(problem)
    executor = LocalSynthesisExecutor(worker_pool_factory=ThreadPoolExecutor)
    active_count = 0
    max_active_count = 0
    lock = threading.Lock()

    def fake_get_solution(
        self, *, print_output=True, evolution=None, model_factories=None
    ):
        nonlocal active_count, max_active_count
        del print_output, evolution, model_factories
        with lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
        try:
            time.sleep(0.05)
            self.case = _SolvedCase()
            self.case.S = self.stages or 3
            self.case.stages = self.case.S
            self.case.mSuccess = 1
            return self.case
        finally:
            with lock:
                active_count -= 1

    monkeypatch.setattr(
        InternalHeatExchangerNetworkProblem,
        "get_solution",
        fake_get_solution,
    )
    pdm_tasks = build_pinch_design_method_tasks(settings)
    pdm_outcomes = executor.execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=2,
    )

    assert [outcome.status for outcome in pdm_outcomes] == ["success", "success"]
    assert executor.executed_tasks == list(pdm_tasks)
    assert max_active_count == 2


def test_local_executor_passes_user_solver_options_to_internal_problem() -> None:
    problem = _four_stream_problem(
        options={
            "HENS_SOLVER_OPTIONS_PDM": {
                "node_limit": 25,
                "feas_tolerance": 0.02,
            },
        }
    )
    settings = workflow_settings_from_problem(problem)
    task = build_pinch_design_method_tasks(settings)[0]

    internal_problem = LocalSynthesisExecutor()._build_problem(
        task,
        problem=problem,
        parent_outcomes={},
    )

    assert settings.solver_options_for("pinch_design_method") == {
        "node_limit": 25,
        "feas_tolerance": 0.02,
    }
    assert internal_problem.solver_options == {
        "node_limit": 25,
        "feas_tolerance": 0.02,
    }


def test_local_executor_merges_task_solver_options_into_internal_problem() -> None:
    problem = _four_stream_problem(
        options={
            "HENS_SOLVER_OPTIONS_PDM": {
                "node_limit": 25,
            },
        }
    )
    settings = workflow_settings_from_problem(problem)
    task = build_pinch_design_method_tasks(settings)[0].model_copy(
        update={"settings": {"solver_options": {"time_limit": 60}}},
    )

    internal_problem = LocalSynthesisExecutor()._build_problem(
        task,
        problem=problem,
        parent_outcomes={},
    )

    assert internal_problem.solver_options == {
        "node_limit": 25,
        "time_limit": 60,
    }


def test_internal_problem_load_reports_missing_pdm_solver_binary(monkeypatch) -> None:
    def missing_binary(binary_name: str, *, purpose: str | None = None) -> str:
        raise MissingSynthesisSolverError(
            f"The {binary_name!r} solver executable is required for {purpose}."
        )

    monkeypatch.setattr(backend, "require_solver_binary", missing_binary)
    decompositions = _pdm_decompositions(_four_stream_problem(), dTmin=14.0)
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(_four_stream_problem(), 14.0),
        framework="PDM",
        solver="couenne",
        dTmin=14.0,
        pinch_decompositions=decompositions,
    )

    with pytest.raises(MissingSynthesisSolverError, match="couenne.*synthesis solves"):
        problem.load_model()


@pytest.mark.parametrize("model_cls", [StageWiseModel, PinchDecompModel])
def test_lmtd_replacement_uses_endpoint_lmtd_for_active_units(model_cls) -> None:
    model = _source_shaped_lmtd_post_process_model(model_cls)

    model.get_post_process()

    if model_cls is PinchDecompModel:
        recovery_lmtd = [
            _source_openhens_lmtd(200.0, 140.0, 1.0, formula_allowed=True),
            _source_openhens_lmtd(140.0, 40.0, 1.0, formula_allowed=True),
        ]
    else:
        recovery_lmtd = [
            _source_openhens_lmtd(70.0, 40.0, 1.0, formula_allowed=True),
            _source_openhens_lmtd(70.0, 20.0, 1.0, formula_allowed=False),
        ]
    hot_utility_lmtd = _source_openhens_lmtd(
        220.0,
        320.0,
        1.0,
        formula_allowed=True,
    )
    cold_utility_lmtd = _source_openhens_lmtd(
        40.0,
        40.0,
        1.0,
        formula_allowed=False,
        fallback_delta=40.0,
    )
    expected_area_r = [
        100.0 / 0.5 / recovery_lmtd[0],
        50.0 / 0.5 / recovery_lmtd[1],
    ]
    expected_area_hu = 30.0 / 0.25 / hot_utility_lmtd
    expected_area_cu = 20.0 / 0.4 / cold_utility_lmtd
    expected_tac = (
        2.0 * 30.0
        + 3.0 * 20.0
        + 7.0 * 4
        + 11.0 * sum(area**0.6 for area in expected_area_r)
        + 13.0 * expected_area_hu**0.6
        + 17.0 * expected_area_cu**0.6
    )

    np.testing.assert_allclose(model.LMTD_r, [[[recovery_lmtd[0], recovery_lmtd[1]]]])
    np.testing.assert_allclose(model.LMTD_hu, [hot_utility_lmtd])
    np.testing.assert_allclose(model.LMTD_cu, [cold_utility_lmtd])
    np.testing.assert_allclose(
        model.area_r, [[[expected_area_r[0], expected_area_r[1]]]]
    )
    np.testing.assert_allclose(model.area_hu, [expected_area_hu])
    np.testing.assert_allclose(model.area_cu, [expected_area_cu])
    assert model.n_recovery_units == 2
    assert model.n_hu_units == 1
    assert model.n_cu_units == 1
    assert model.n_units == 4
    assert model.Q_r_total == 150.0
    assert model.Q_hu_total == 30.0
    assert model.Q_cu_total == 20.0
    assert model.TAC_model == 123.45
    assert model.TAC == pytest.approx(expected_tac)


def test_extraction_converts_solver_arrays_to_identity_labelled_network() -> None:
    solved = _SolvedCase()
    arrays = _solver_arrays()

    network = extract_heat_exchanger_network(
        solved,
        arrays,
        run_id="hens-06",
        task_id="task-1",
        method="TDM",
    )

    assert len(network.exchangers) == 4
    recovery = network.exchanger_between(
        source_stream="hot-A",
        sink_stream="cold-A",
        stage=1,
        kind=HeatExchangerKind.RECOVERY,
    )
    assert recovery is not None
    assert recovery.duty == 10.0
    assert recovery.area == 1.5
    assert recovery.source_inlet_temperature == 650.0
    assert recovery.source_outlet_temperature == 620.0
    assert recovery.sink_inlet_temperature == 420.0
    assert recovery.sink_outlet_temperature == 450.0
    assert recovery.approach_temperatures == (200.0, 170.0)

    hot_utility = network.exchanger_between(
        source_stream="hot-utility",
        sink_stream="cold-B",
        kind=HeatExchangerKind.HOT_UTILITY,
    )
    cold_utility = network.exchanger_between(
        source_stream="hot-A",
        sink_stream="cold-utility",
        kind=HeatExchangerKind.COLD_UTILITY,
    )
    assert hot_utility is not None
    assert cold_utility is not None
    assert hot_utility.duty == 5.0
    assert cold_utility.duty == 3.0
    assert network.total_duty(kind=HeatExchangerKind.RECOVERY) == 30.0
    assert network.summary_metrics == {
        "total_units": 4,
        "recovery_units": 2,
        "hot_utility_units": 1,
        "cold_utility_units": 1,
        "hot_utility_load": 5.0,
        "cold_utility_load": 3.0,
        "recovery_load": 30.0,
    }
    assert network.source_metadata["hot_stage_boundary_temperatures"] == [
        [650.0, 620.0, 390.0],
        [610.0, 580.0, 360.0],
    ]
    assert network.source_metadata["cold_stage_boundary_temperatures"] == [
        [480.0, 420.0, 300.0],
        [510.0, 470.0, 330.0],
    ]
    assert network.source_metadata["hot_stream_heat_transfer_coefficients"] == [
        1.0,
        2.0,
    ]
    assert network.source_metadata["cold_stream_heat_transfer_coefficients"] == [
        3.0,
        4.0,
    ]
    assert network.source_metadata["hot_utility_heat_transfer_coefficients"] == [5.0]
    assert network.source_metadata["cold_utility_heat_transfer_coefficients"] == [6.0]


def test_internal_problem_result_serializes_without_solver_arrays() -> None:
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=_solver_arrays(),
        name="internal",
        framework="TDM",
        solver="couenne",
        stages=2,
        synthesis_task_id="task-1",
    )
    problem.case = _SolvedCase()

    result = problem.extract_result(run_id="hens-06", problem_id="problem-1")
    payload = result.model_dump(mode="json")
    encoded = json.dumps(payload)

    assert result.network.exchangers
    assert payload["problem_id"] == "problem-1"
    assert payload["task_id"] == "task-1"
    assert payload["method"] == "thermal_derivative_method"
    assert payload["network"]["method"] == "thermal_derivative_method"
    assert "solver_axis_metadata" not in payload["network"]
    assert "source_metadata" not in payload["network"]
    assert "Q_r" not in encoded
    assert "Q_h" not in encoded
    assert "Q_c" not in encoded


def _four_stream_problem(*, options: dict | None = None) -> PinchProblem:
    payload = json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8"))
    if options:
        payload["options"] = {**payload["options"], **options}
    return PinchProblem(source=payload, project_name="HENS-07 Four-stream")


def _two_state_problem(*, options: dict | None = None) -> PinchProblem:
    payload = json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8"))
    payload["options"] = {
        **payload["options"],
        "PROBLEM_PERIOD_IDS": ["base", "peak"],
        "PROBLEM_PERIOD_WEIGHTS": [1.0, 3.0],
        **(options or {}),
    }
    first_hot = payload["streams"][0]
    first_hot["heat_capacity_flowrate"] = {
        "unit": "kW/delta_degC",
        "values": [10.0, 20.0],
    }
    first_hot["t_supply"] = {
        "unit": "K",
        "values": [650.0, 660.0],
    }
    first_hot["t_target"] = {
        "unit": "K",
        "values": [370.0, 360.0],
    }
    first_hot["heat_flow"] = {
        "unit": "kW",
        "values": [2800.0, 6000.0],
    }
    return PinchProblem(source=payload, project_name="HENS two-state")


def _pdm_decompositions(
    problem: PinchProblem,
    *,
    dTmin: float,
    stage_selection="automated",
):
    return {
        "above": build_pinch_design_decomposition(
            problem,
            dTmin,
            pinch_location="above",
            stage_selection=stage_selection,
        ),
        "below": build_pinch_design_decomposition(
            problem,
            dTmin,
            pinch_location="below",
            stage_selection=stage_selection,
        ),
    }


def _moved_pdm_models(stage_selection="automated"):
    problem = _four_stream_problem()
    arrays = problem_to_solver_arrays(problem, 14.0)
    decompositions = _pdm_decompositions(
        problem,
        dTmin=14.0,
        stage_selection=stage_selection,
    )
    above = PinchDecompModel(
        name="moved-above",
        framework="PDM",
        solver="apopt",
        solver_arrays=arrays,
        dTmin=14.0,
        z_restriction=None,
        min_dqda=0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="above",
        pinch_decomposition=decompositions["above"],
        stage_selection=stage_selection,
    )
    below = PinchDecompModel(
        name="moved-below",
        framework="PDM",
        solver="apopt",
        solver_arrays=arrays,
        dTmin=14.0,
        z_restriction=None,
        min_dqda=0,
        minimisation_goal="cold utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="below",
        pinch_decomposition=decompositions["below"],
        stage_selection=stage_selection,
    )
    return above, below


def _source_shaped_lmtd_post_process_model(model_cls):
    model = model_cls.__new__(model_cls)
    model.mSuccess = 1
    model.I = 1
    model.J = 1
    model.S = 2
    model.N_periods = 1
    model.period_weights = [1.0]
    model.period_weight_sum = 1.0
    model.tol = 1e-3
    model.dTmin = 20.0
    model.minimisation_goal = "variable total cost"
    model.Q_r = [[[[100.0], [50.0]]]]
    model.Q_h = [[30.0]]
    model.Q_c = [[20.0]]
    model.z = [[[[1.0], [1.0]]]]
    model.z_hu = [[1.0]]
    model.z_cu = [[1.0]]
    model.theta_1 = [[[[70.0], [70.0]]]]
    model.theta_2 = [[[[40.0], [20.0]]]]
    model.U_r = [[0.5]]
    model.U_hu = [0.25]
    model.U_cu = [0.4]
    model.T_h = [[[500.0], [470.0], [360.0]]]
    model.T_c = [[[300.0], [330.0], [320.0]]]
    model.T_h_out = [340.0]
    model.T_c_out = [430.0]
    model.T_hu_in = [650.0]
    model.T_hu_out = [620.0]
    model.T_cu_in = [300.0]
    model.T_cu_out = [320.0]
    model.Q_r_by_period = [model.Q_r]
    model.Q_h_by_period = [model.Q_h]
    model.Q_c_by_period = [model.Q_c]
    model.T_h_by_period = [model.T_h]
    model.T_c_by_period = [model.T_c]
    model.theta_1_by_period = [model.theta_1]
    model.theta_2_by_period = [model.theta_2]
    model.U_r_period = [model.U_r]
    model.U_hu_period = [model.U_hu]
    model.U_cu_period = [model.U_cu]
    model.T_h_out_period = [model.T_h_out]
    model.T_c_out_period = [model.T_c_out]
    model.T_hu_in_period = [model.T_hu_in]
    model.T_hu_out_period = [model.T_hu_out]
    model.T_cu_in_period = [model.T_cu_in]
    model.T_cu_out_period = [model.T_cu_out]
    model.hu_cost = [2.0]
    model.cu_cost = [3.0]
    model.hu_cost_period = [model.hu_cost]
    model.cu_cost_period = [model.cu_cost]
    model.unit_cost = [7.0]
    model.A_coeff = [11.0]
    model.A_exp = [0.6]
    model.hu_coeff = [13.0]
    model.hu_exp = [0.6]
    model.cu_coeff = [17.0]
    model.cu_exp = [0.6]
    model.alpha = [[[[1.0], [1.0]]]]
    model.get_alpha_values = lambda: [[[[1.0], [1.0]]]]
    model.m = SimpleNamespace(options=SimpleNamespace(objfcnval=123.45))
    return model


def _source_openhens_lmtd(
    delta_1: float,
    delta_2: float,
    active: float,
    *,
    formula_allowed: bool,
    fallback_delta: float | None = None,
) -> float:
    if not formula_allowed:
        return (delta_1 if fallback_delta is None else fallback_delta) * active
    return active * (delta_1 - delta_2) / math.log(delta_1 / delta_2)


def _solver_arrays() -> PreparedSolverArrays:
    return PreparedSolverArrays(
        arrays={},
        axis_maps={
            "hot_process_streams": {"hot-B": 1, "hot-A": 0},
            "cold_process_streams": {"cold-B": 1, "cold-A": 0},
            "hot_utilities": {"hot-utility": 0},
            "cold_utilities": {"cold-utility": 0},
            "stages": {"1": 0, "2": 1},
        },
        unit_conventions={"temperature": "K"},
        stream_identities={
            "hot_process_streams": ["hot-A", "hot-B"],
            "cold_process_streams": ["cold-A", "cold-B"],
        },
        utility_identities={
            "hot_utilities": ["hot-utility"],
            "cold_utilities": ["cold-utility"],
        },
        configuration={"HENS_RUN_ID": "hens-06"},
        preparation={"prepared_zone_class": "Zone"},
    )


class _FakeOptions:
    SOLVER_EXTENSION = None
    SOLVER = None
    MAX_ITER = None
    RTOL = None
    OTOL = None
    SOLVESTATUS = None
    objfcnval = None


class _FakeGekkoModel:
    def __init__(self, path: Path | None = None) -> None:
        self.options = _FakeOptions()
        self.solver_options = []
        self.cwd_during_solve = None
        if path is not None:
            self._path = str(path)

    def solve(self, *, disp: bool = False, debug: int = 0) -> None:
        del disp, debug
        self.cwd_during_solve = Path.cwd()
        self.options.SOLVESTATUS = 1
        self.options.objfcnval = 1.0


class _ParentCase:
    stages = 3
    S = 3


class _EvolutionCase:
    framework = "ESM"
    name = "evolution"
    stages = 2
    tol = 1e-3
    mSuccess = 1

    Q_r = [[[[10.0], [10.0]]]]

    def __init__(self) -> None:
        self.I = 1
        self.J = 1
        self.S = 2
        self.calls = []

    def optimise(self, print_output: bool) -> None:
        assert print_output is False
        self.calls.append("optimise")

    def get_net_benefit_evolution(self, print_output: bool, **kwargs):
        assert print_output is False
        assert kwargs["n_ad_branches"] == 1
        assert kwargs["n_rm_branches"] == 1
        self.calls.append("evolution")
        return self


class _StageReductionCase:
    def __init__(self, *, stage_duties: list[float]) -> None:
        self.I = 1
        self.J = 1
        self.S = 4
        self.Q_r = [[[[duty] for duty in stage_duties]]]


class _CandidateModel:
    def __init__(self, *, mSuccess: int, TAC: float) -> None:
        self.mSuccess = mSuccess
        self.TAC = TAC


class _BranchEvolutionCase:
    def __init__(
        self,
        *,
        name: str,
        tac: float,
        z: list,
        rm_candidates: tuple[tuple[int, int, int], ...] = (),
        add_candidates: tuple[tuple[int, int, int], ...] = (),
        success: int = 1,
        valid: bool = True,
    ) -> None:
        self.name = name
        self.TAC = tac
        self.z = z
        self.mSuccess = success
        self._rm_candidates = rm_candidates
        self._add_candidates = add_candidates
        self._valid = valid

    def get_lowest_benefit_HX_candidates(self, limit: int) -> list[list[int]]:
        return [list(position) for position in self._rm_candidates[:limit]]

    def get_max_benefit_HX_candidates(self, limit: int) -> list[list[int]]:
        return [list(position) for position in self._add_candidates[:limit]]

    def verify(self) -> tuple[bool, list[str]]:
        return self._valid, [] if self._valid else ["forced"]


def _branching_root_model(
    *,
    rm_candidates: tuple[tuple[int, int, int], ...],
    add_candidates: tuple[tuple[int, int, int], ...],
):
    model = StageWiseModel.__new__(StageWiseModel)
    model.name = "root"
    model.I = 1
    model.J = 1
    model.S = 4
    model.tol = 1e-3
    model.mSuccess = 1
    model.TAC = 100.0
    model.z = [[[[1], [1], [0], [0]]]]
    model.m = SimpleNamespace(cleanup=lambda: setattr(model, "cleaned", True))
    model.cleaned = False
    model.updated_with = None
    model.get_lowest_benefit_HX_candidates = lambda limit: [
        list(position) for position in rm_candidates[:limit]
    ]
    model.get_max_benefit_HX_candidates = lambda limit: [
        list(position) for position in add_candidates[:limit]
    ]
    model._update_with_best_model = lambda best: setattr(
        model,
        "updated_with",
        best.name,
    )
    return model


class _RecordingStageWise:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.stages = kwargs["stages"]
        self.initialised_from = None

    def set_initial_values_for_variables(self, parent_case) -> None:
        self.initialised_from = parent_case


class _SolvedCase:
    name = "solved"
    framework = "TDM"
    S = 2
    stages = 2
    mSuccess = 1
    TAC = 1234.5
    TAC_model = 1234.5
    n_units = 4
    n_recovery_units = 2
    n_hu_units = 1
    n_cu_units = 1
    Q_hu_total = 5.0
    Q_cu_total = 3.0
    Q_r_total = 30.0
    hu_cost_total = 50.0
    cu_cost_total = 30.0
    hu_area_cost_total = 11.0
    cu_area_cost_total = 7.0
    recovery_area_cost_total = 40.0
    htc_h = [1.0, 2.0]
    htc_c = [3.0, 4.0]
    htc_hu = [5.0]
    htc_cu = [6.0]

    Q_r = [
        [[[10.0], [0.0]], [[0.0], [0.0]]],
        [[[0.0], [0.0]], [[0.0], [20.0]]],
    ]
    Q_h = [[0.0], [5.0]]
    Q_c = [[3.0], [0.0]]
    z = [
        [[[1.0], [0.0]], [[0.0], [0.0]]],
        [[[0.0], [0.0]], [[0.0], [1.0]]],
    ]
    z_allowed = [
        [[[1.0], [1.0]], [[1.0], [1.0]]],
        [[[1.0], [1.0]], [[1.0], [1.0]]],
    ]
    z_hu = [[0.0], [1.0]]
    z_cu = [[1.0], [0.0]]
    z_hu_allowed = [1.0, 1.0]
    z_cu_allowed = [1.0, 1.0]
    area_r = [
        [[1.5, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 2.5]],
    ]
    area_hu = [0.0, 0.7]
    area_cu = [0.3, 0.0]
    theta_1 = [
        [[[200.0], [0.0]], [[0.0], [0.0]]],
        [[[0.0], [0.0]], [[0.0], [160.0]]],
    ]
    theta_2 = [
        [[[170.0], [0.0]], [[0.0], [0.0]]],
        [[[0.0], [0.0]], [[0.0], [140.0]]],
    ]
    T_h = [[[650.0], [620.0], [390.0]], [[610.0], [580.0], [360.0]]]
    T_c = [[[480.0], [420.0], [300.0]], [[510.0], [470.0], [330.0]]]
    T_h_out = [360.0, 350.0]
    T_c_out = [480.0, 510.0]
    T_h_out_x = [
        [[[620.0], [0.0]], [[0.0], [0.0]]],
        [[[0.0], [0.0]], [[0.0], [560.0]]],
    ]
    T_c_out_y = [
        [[[450.0], [0.0]], [[0.0], [0.0]]],
        [[[0.0], [0.0]], [[0.0], [500.0]]],
    ]
    T_hu_in = [700.0]
    T_hu_out = [680.0]
    T_cu_in = [300.0]
    T_cu_out = [320.0]

    def __init__(self) -> None:
        for name, value in type(self).__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            setattr(self, name, deepcopy(value))


def test_extraction_uses_tolerance_for_binary_and_duty_activity() -> None:
    solved = _SolvedCase()
    solved.Q_h[1][0] = tol / 2.0

    network = extract_heat_exchanger_network(
        solved,
        _solver_arrays(),
        run_id="hens-06",
        include_inactive=False,
    )

    assert (
        network.exchanger_between(
            source_stream="hot-utility",
            sink_stream="cold-B",
            kind=HeatExchangerKind.HOT_UTILITY,
        )
        is None
    )
