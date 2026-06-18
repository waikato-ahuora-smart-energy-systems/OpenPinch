"""Private HEN synthesis model-boundary tests."""

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
from OpenPinch.services.heat_exchanger_network_synthesis._dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.services.heat_exchanger_network_synthesis.array_adapter import (
    PreparedSolverArrays,
    problem_to_solver_arrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models import backend
from OpenPinch.services.heat_exchanger_network_synthesis.models.extraction import (
    extract_heat_exchanger_network,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models.pinch_decomposition import (
    PinchDecompModel,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models.problem import (
    InternalHeatExchangerNetworkProblem,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.services.heat_exchanger_network_synthesis.pinch_decomposition import (
    build_pinch_decomposition_snapshot,
)
from OpenPinch.services.heat_exchanger_network_synthesis.workflow import (
    LocalSynthesisExecutor,
    build_energy_stage_refinement_tasks,
    build_pinch_decomposition_tasks,
    build_topology_design_tasks,
    workflow_settings_from_problem,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENHENS_ROOT = Path("/Users/ca107/Desktop/ahuora/OpenHENS")
OPENHENS_FOUR_STREAM_CSV = (
    OPENHENS_ROOT / "examples/cases/Four-stream-Yee-and-Grossmann-1990-1.csv"
)
FOUR_STREAM_JSON = (
    REPO_ROOT / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)
SYNTHESIS_ONLY_MODULES = [
    "gekko",
    "pyomo",
    "pyomo.environ",
    "pyomo.opt",
    "matplotlib",
    "matplotlib.pyplot",
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
from OpenPinch.services.heat_exchanger_network_synthesis.models import (
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
            "gekko is required for GEKKO HEN equation model construction. "
            'Install "openpinch[synthesis]".'
        )

    monkeypatch.setattr(backend, "require_synthesis_dependency", missing_dependency)

    with pytest.raises(
        MissingSynthesisDependencyError,
        match=r"gekko.*GEKKO HEN equation model construction.*openpinch\[synthesis\]",
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
        match=r"couenne.*couenne HEN synthesis solves.*PATH",
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


def test_couenne_option_file_preserves_openhens_runtime_limits() -> None:
    couenne_options = (REPO_ROOT / "couenne.opt").read_text(encoding="utf-8")

    assert "node_limit 2000" in couenne_options
    assert "feas_tolerance 0.01" in couenne_options
    assert "allowable_gap 0.01" in couenne_options
    assert "allowable_fraction_gap 0.1" in couenne_options
    assert "delete_redundant yes" in couenne_options


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

    assert ranking_model.get_lowest_benefit_HX() == [[0, 0, 0]]
    assert ranking_model.get_max_benefit_HX() == [[0, 1, 1]]


def test_stagewise_construction_matches_source_four_stream_structural_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    SourceStageWiseModel, _SourcePinchDecompModel = _source_openhens_models(
        monkeypatch,
        tmp_path,
    )
    moved = StageWiseModel(
        name="moved-stagewise",
        framework="TDM",
        solver="apopt",
        solver_arrays=problem_to_solver_arrays(_four_stream_problem(), 0.1),
        stages=3,
        dTmin=0.1,
        z_restriction=None,
        min_dqda=0.5,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
    )
    source = SourceStageWiseModel(
        name="source-stagewise",
        framework="TDM",
        solver="apopt",
        import_file=OPENHENS_FOUR_STREAM_CSV,
        stages=3,
        dTmin=0.1,
        z_restriction=[None, None, None],
        min_dqda=0.5,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
    )

    assert (moved.I, moved.J, moved.S, moved.K) == (
        source.I,
        source.J,
        source.S,
        source.K,
    )
    np.testing.assert_allclose(moved.Qtot_sh, source.Qtot_sh)
    np.testing.assert_allclose(moved.Qtot_sc, source.Qtot_sc)
    np.testing.assert_allclose(moved.Q_max, source.Q_max)
    assert moved.z_feasible == source.z_feasible
    assert moved.z_hu_feasible == source.z_hu_feasible
    assert moved.z_cu_feasible == source.z_cu_feasible
    assert moved.Q_r[0][0][0].name == source.Q_r[0][0][0].name
    assert moved.theta_1[0][0][0].name == source.theta_1[0][0][0].name
    assert moved.z[0][0][0].name == source.z[0][0][0].name


def test_esm_stagewise_construction_matches_source_four_stream_structural_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    SourceStageWiseModel, _SourcePinchDecompModel = _source_openhens_models(
        monkeypatch,
        tmp_path,
    )
    moved = StageWiseModel(
        name="moved-esm-stagewise",
        framework="ESM",
        solver="apopt",
        solver_arrays=problem_to_solver_arrays(_four_stream_problem(), 0.1),
        stages=3,
        dTmin=0.1,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="variable total cost",
        non_isothermal_model=True,
        integers=False,
        tol=1e-3,
    )
    source = SourceStageWiseModel(
        name="source-esm-stagewise",
        framework="ESM",
        solver="apopt",
        import_file=OPENHENS_FOUR_STREAM_CSV,
        stages=3,
        dTmin=0.1,
        z_restriction=[None, None, None],
        min_dqda=0.0,
        minimisation_goal="variable total cost",
        non_isothermal_model=True,
        integers=False,
        tol=1e-3,
    )

    assert (moved.I, moved.J, moved.S, moved.K) == (
        source.I,
        source.J,
        source.S,
        source.K,
    )
    assert moved.non_isothermal_model is source.non_isothermal_model is True
    assert moved.integers is source.integers is False
    assert moved.minimisation_goal == source.minimisation_goal == "variable total cost"
    np.testing.assert_allclose(moved.Qtot_sh, source.Qtot_sh)
    np.testing.assert_allclose(moved.Qtot_sc, source.Qtot_sc)
    np.testing.assert_allclose(moved.Q_max, source.Q_max)
    assert moved.z_feasible == source.z_feasible
    assert moved.z_hu_feasible == source.z_hu_feasible
    assert moved.z_cu_feasible == source.z_cu_feasible
    assert len(moved.m._equations) == len(source.m._equations)
    assert len(moved.m._objectives) == len(source.m._objectives)
    assert moved.Q_r[0][0][0].name == source.Q_r[0][0][0].name
    assert moved.theta_1[0][0][0].name == source.theta_1[0][0][0].name
    assert moved.z[0][0][0].name == source.z[0][0][0].name
    assert moved.X[0][0][0].name == source.X[0][0][0].name
    assert moved.Y[0][0][0].name == source.Y[0][0][0].name
    assert moved.T_h_out_x[0][0][0].name == source.T_h_out_x[0][0][0].name
    assert moved.T_c_out_y[0][0][0].name == source.T_c_out_y[0][0][0].name
    assert moved.hu_cost_total.name == source.hu_cost_total.name
    assert moved.cu_cost_total.name == source.cu_cost_total.name
    assert moved.hu_area_cost_total.name == source.hu_area_cost_total.name
    assert moved.cu_area_cost_total.name == source.cu_area_cost_total.name
    assert (
        moved.recovery_area_cost_filtered[0][0].name
        == source.recovery_area_cost_filtered[0][0].name
    )
    assert not hasattr(moved, "utility_unit_cost_total")
    assert not hasattr(source, "utility_unit_cost_total")


def test_pdm_construction_matches_source_four_stream_above_below_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _SourceStageWiseModel, SourcePinchDecompModel = _source_openhens_models(
        monkeypatch,
        tmp_path,
    )
    moved_above, moved_below = _moved_pdm_models()
    source_above = SourcePinchDecompModel(
        name="source-above",
        framework="PDM",
        solver="apopt",
        import_file=OPENHENS_FOUR_STREAM_CSV,
        dTmin=14.0,
        z_restriction=[None, None, None],
        min_dqda=0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="above",
        stage_selection="automated",
    )
    source_below = SourcePinchDecompModel(
        name="source-below",
        framework="PDM",
        solver="apopt",
        import_file=OPENHENS_FOUR_STREAM_CSV,
        dTmin=14.0,
        z_restriction=[None, None, None],
        min_dqda=0,
        minimisation_goal="cold utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="below",
        stage_selection="automated",
    )

    for moved, source in (
        (moved_above, source_above),
        (moved_below, source_below),
    ):
        assert moved.HU_target == pytest.approx(source.HU_target)
        assert moved.CU_target == pytest.approx(source.CU_target)
        assert moved.T_pinch == pytest.approx(source.T_pinch)
        assert (moved.I, moved.J, moved.S, moved.K) == (
            source.I,
            source.J,
            source.S,
            source.K,
        )
        assert moved.z_i_active == source.z_i_active
        assert moved.z_j_active == source.z_j_active
        np.testing.assert_allclose(moved.T_h_in, source.T_h_in)
        np.testing.assert_allclose(moved.T_h_out, source.T_h_out)
        np.testing.assert_allclose(moved.T_c_in, source.T_c_in)
        np.testing.assert_allclose(moved.T_c_out, source.T_c_out)
        np.testing.assert_allclose(moved.Q_max, source.Q_max)
        assert moved.z_feasible == source.z_feasible
        assert moved.z_hu_feasible == source.z_hu_feasible
        assert moved.z_cu_feasible == source.z_cu_feasible
        assert moved.Q_r[0][0][0].name == source.Q_r[0][0][0].name
        assert moved.T_h[0][0].name == source.T_h[0][0].name


def test_pdm_manual_above_below_stage_selection_matches_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _SourceStageWiseModel, SourcePinchDecompModel = _source_openhens_models(
        monkeypatch,
        tmp_path,
    )
    moved_above, moved_below = _moved_pdm_models(stage_selection=[2, 3])
    source_above = SourcePinchDecompModel(
        name="source-above-manual",
        framework="PDM",
        solver="apopt",
        import_file=OPENHENS_FOUR_STREAM_CSV,
        dTmin=14.0,
        z_restriction=[None, None, None],
        min_dqda=0,
        minimisation_goal="hot utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="above",
        stage_selection=[2, 3],
    )
    source_below = SourcePinchDecompModel(
        name="source-below-manual",
        framework="PDM",
        solver="apopt",
        import_file=OPENHENS_FOUR_STREAM_CSV,
        dTmin=14.0,
        z_restriction=[None, None, None],
        min_dqda=0,
        minimisation_goal="cold utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc="below",
        stage_selection=[2, 3],
    )

    assert moved_above.S == source_above.S == 2
    assert moved_below.S == source_below.S == 3
    assert moved_above.K == source_above.K == 3
    assert moved_below.K == source_below.K == 4


def test_internal_problem_loads_pdm_and_stagewise_with_parent_context() -> None:
    pdm_problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(_four_stream_problem(), 14.0),
        name="pdm",
        framework="PDM",
        solver="apopt",
        dTmin=14.0,
        pinch_snapshots=_pdm_snapshots(_four_stream_problem(), dTmin=14.0),
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
    pdm_tasks = build_pinch_decomposition_tasks(settings)
    pdm_outcomes = executor.execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=1,
    )
    tdm_tasks = build_topology_design_tasks(settings, pdm_outcomes)
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
    esm_tasks = build_energy_stage_refinement_tasks(settings, tdm_outcomes)
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
    pdm_tasks = build_pinch_decomposition_tasks(settings)
    pdm_outcomes = executor.execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=2,
    )

    assert [outcome.status for outcome in pdm_outcomes] == ["success", "success"]
    assert executor.executed_tasks == list(pdm_tasks)
    assert max_active_count == 2


def test_internal_problem_load_reports_missing_pdm_solver_binary(monkeypatch) -> None:
    def missing_binary(binary_name: str, *, purpose: str | None = None) -> str:
        raise MissingSynthesisSolverError(
            f"The {binary_name!r} solver executable is required for {purpose}."
        )

    monkeypatch.setattr(backend, "require_solver_binary", missing_binary)
    snapshots = _pdm_snapshots(_four_stream_problem(), dTmin=14.0)
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(_four_stream_problem(), 14.0),
        framework="PDM",
        solver="couenne",
        dTmin=14.0,
        pinch_snapshots=snapshots,
    )

    with pytest.raises(MissingSynthesisSolverError, match="couenne.*synthesis solves"):
        problem.load_model()


@pytest.mark.parametrize("model_cls", [StageWiseModel, PinchDecompModel])
def test_lmtd_replacement_preserves_openhens_post_process_metrics(model_cls) -> None:
    model = _source_shaped_lmtd_post_process_model(model_cls)

    model.get_post_process()

    recovery_lmtd = [
        _source_openhens_lmtd(70.0, 40.0, 1.0, formula_allowed=True),
        _source_openhens_lmtd(20.0, 20.0, 1.0, formula_allowed=False),
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
    np.testing.assert_allclose(model.area_r, [[[expected_area_r[0], expected_area_r[1]]]])
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
    assert payload["method"] == "topology_design"
    assert payload["network"]["method"] == "topology_design"
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


def _pdm_snapshots(
    problem: PinchProblem,
    *,
    dTmin: float,
    stage_selection="automated",
):
    return {
        "above": build_pinch_decomposition_snapshot(
            problem,
            dTmin,
            pinch_location="above",
            stage_selection=stage_selection,
        ),
        "below": build_pinch_decomposition_snapshot(
            problem,
            dTmin,
            pinch_location="below",
            stage_selection=stage_selection,
        ),
    }


def _moved_pdm_models(stage_selection="automated"):
    problem = _four_stream_problem()
    arrays = problem_to_solver_arrays(problem, 14.0)
    snapshots = _pdm_snapshots(
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
        pinch_snapshot=snapshots["above"],
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
        pinch_snapshot=snapshots["below"],
        stage_selection=stage_selection,
    )
    return above, below


def _source_openhens_models(monkeypatch, tmp_path: Path):
    if not OPENHENS_ROOT.exists() or not OPENHENS_FOUR_STREAM_CSV.exists():
        pytest.skip("source OpenHENS checkout is not available")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source_root = str(OPENHENS_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    from openhens.classes.pinch_decomp_model import (  # noqa: PLC0415
        PinchDecompModel as SourcePinchDecompModel,
    )
    from openhens.classes.stage_wise_model import (  # noqa: PLC0415
        StageWiseModel as SourceStageWiseModel,
    )

    return SourceStageWiseModel, SourcePinchDecompModel


def _source_shaped_lmtd_post_process_model(model_cls):
    model = model_cls.__new__(model_cls)
    model.mSuccess = 1
    model.I = 1
    model.J = 1
    model.S = 2
    model.tol = 1e-3
    model.dTmin = 20.0
    model.minimisation_goal = "variable total cost"
    model.Q_r = [[[[100.0], [50.0]]]]
    model.Q_h = [[30.0]]
    model.Q_c = [[20.0]]
    model.z = [[[[1.0], [1.0]]]]
    model.z_hu = [[1.0]]
    model.z_cu = [[1.0]]
    model.theta_1 = [[[[70.0], [20.0]]]]
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
    model.hu_cost = [2.0]
    model.cu_cost = [3.0]
    model.unit_cost = [7.0]
    model.A_coeff = [11.0]
    model.A_exp = [0.6]
    model.hu_coeff = [13.0]
    model.cu_coeff = [17.0]
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
    def __init__(self) -> None:
        self.options = _FakeOptions()
        self.solver_options = []


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

    def get_net_benefit_evolution(self, print_output: bool):
        assert print_output is False
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
