"""Private heat exchanger network synthesis model-boundary tests."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from OpenPinch import PinchProblem
from OpenPinch.classes.heat_exchanger import HeatExchangerKind
from OpenPinch.classes.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.lib.config import tol
from OpenPinch.lib.schemas.synthesis import HeatExchangerNetworkSynthesisTask
from OpenPinch.services.heat_exchanger_network_synthesis.common.errors import (
    WorkflowContractError,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution import (
    executor as executor_module,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.executor import (
    LocalSynthesisExecutor,
    _branch_breadth,
    _build_and_solve_root_task,
    _failed_task_outcome,
    _legacy_pdm_stage_selection,
    _optional_int,
    _pathway_evolution_options,
    _process_pool,
    _solve_built_task,
    _solver_options_for_task,
    _solver_status,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import (
    arrays as arrays_module,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import backend
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import (
    extraction as extraction_module,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import (
    pinch_design_decomposition as pdm_decomposition,
)
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


def test_solver_backend_rejects_unsupported_solver_name() -> None:
    with pytest.raises(MissingSynthesisSolverError, match="Unsupported.*not-real"):
        backend.require_solver_backend("not-real")


def test_default_process_pool_uses_interpreter_start_method(monkeypatch) -> None:
    captured_kwargs = {}
    fake_pool = object()

    def fake_process_pool_executor(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_pool

    monkeypatch.setattr(
        "OpenPinch.services.heat_exchanger_network_synthesis.common.execution."
        "executor.ProcessPoolExecutor",
        fake_process_pool_executor,
    )

    pool = _process_pool(3)

    assert pool is fake_pool
    assert captured_kwargs == {"max_workers": 3}
    assert "mp_context" not in captured_kwargs


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


def test_gekko_solver_configuration_merges_ipopt_and_apopt_options(monkeypatch):
    monkeypatch.setattr(
        backend,
        "require_synthesis_dependency",
        lambda *_args, **_kwargs: object(),
    )

    ipopt_model = _FakeGekkoModel()
    ipopt_run = backend.configure_gekko_solver(
        ipopt_model,
        "ipopt-GEKKO",
        solver_options=["max_iter 20", "# ignored", "warm_start"],
    )

    assert ipopt_run.extension == 0
    assert ipopt_run.solver_options["max_iter"] == "20"
    assert ipopt_run.solver_options["warm_start"] == ""
    assert "tol 1e-3" in ipopt_model.solver_options
    assert "max_iter 20" in ipopt_model.solver_options
    assert "warm_start" in ipopt_model.solver_options

    apopt_model = _FakeGekkoModel()
    apopt_run = backend.configure_gekko_solver(
        apopt_model,
        "apopt",
        solver_options={
            "minlp_maximum_iterations": 4,
            "skip_none": None,
            "use_warm_start": True,
        },
    )

    assert apopt_run.solver_options == {
        "minlp_maximum_iterations": 4,
        "use_warm_start": True,
    }
    assert "use_warm_start yes" in apopt_model.solver_options


def test_gekko_solve_suppresses_known_numpy_array_copy_warning_only() -> None:
    class WarningGekkoModel(_FakeGekkoModel):
        def solve(self, *, disp: bool = False, debug: int = 0) -> None:
            warnings.warn_explicit(
                "__array__ implementation doesn't accept a copy keyword, so "
                "passing copy=False failed. __array__ must implement 'dtype' "
                "and 'copy' keyword arguments.",
                DeprecationWarning,
                filename="/site-packages/gekko/gk_write_files.py",
                lineno=159,
                module="gekko.gk_write_files",
            )
            warnings.warn("unrelated solver warning", RuntimeWarning, stacklevel=2)
            super().solve(disp=disp, debug=debug)

    model = WarningGekkoModel()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = backend.solve_gekko_model(model, solver_name="apopt")

    assert run.failure_reason is None
    assert [(warning.category, str(warning.message)) for warning in caught] == [
        (RuntimeWarning, "unrelated solver warning")
    ]


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


def test_ipopt_pyomo_option_file_uses_factory_availability_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def present_binary(binary_name: str, *, purpose: str | None = None) -> str:
        del purpose
        return f"/usr/local/bin/{binary_name}"

    class _FallbackSolverFactory:
        def available(self, *args, **kwargs) -> bool:
            if kwargs:
                raise TypeError("keyword form unsupported")
            assert args == (False,)
            return True

    def present_dependency(import_name: str, **_kwargs):
        if import_name == "pyomo.environ":
            return SimpleNamespace(SolverFactory=lambda _name: _FallbackSolverFactory())
        return object()

    monkeypatch.setattr(backend, "require_solver_binary", present_binary)
    monkeypatch.setattr(backend, "require_synthesis_dependency", present_dependency)
    model = _FakeGekkoModel(tmp_path / "gekko-run")

    run = backend.configure_gekko_solver(
        model,
        "ipopt-pyomo",
        solver_options=["tol 1e-6"],
    )

    option_file = tmp_path / "gekko-run" / "ipopt.opt"
    assert run.option_file == str(option_file)
    assert run.solver_options == {"tol": "1e-6"}
    assert option_file.read_text(encoding="utf-8") == "tol 1e-6\n"


def test_pyomo_solver_configuration_reports_unavailable_factory(monkeypatch) -> None:
    def present_binary(binary_name: str, *, purpose: str | None = None) -> str:
        del purpose
        return f"/usr/local/bin/{binary_name}"

    class _UnavailableSolverFactory:
        def available(self, exception_flag: bool = False) -> bool:
            del exception_flag
            return False

    def present_dependency(import_name: str, **_kwargs):
        if import_name == "pyomo.environ":
            return SimpleNamespace(
                SolverFactory=lambda _name: _UnavailableSolverFactory()
            )
        return object()

    monkeypatch.setattr(backend, "require_solver_binary", present_binary)
    monkeypatch.setattr(backend, "require_synthesis_dependency", present_dependency)

    with pytest.raises(MissingSynthesisSolverError, match="not available"):
        backend.configure_gekko_solver(_FakeGekkoModel(), "ipopt-pyomo")


def test_solver_backend_solve_reports_exceptions_and_non_success_status():
    class FailingModel(_FakeGekkoModel):
        def solve(self, *, disp: bool = False, debug: int = 0) -> None:
            del disp, debug
            self.options.SOLVESTATUS = None
            self.options.objfcnval = "not-a-number"
            raise RuntimeError("solver exploded")

    failed_run = backend.solve_gekko_model(FailingModel(), solver_name="apopt")

    assert failed_run.failure_reason == "solver exploded"
    assert failed_run.objective_value is None

    class StatusOnlyFailureModel(_FakeGekkoModel):
        def solve(self, *, disp: bool = False, debug: int = 0) -> None:
            del disp, debug
            self.options.SOLVESTATUS = 2
            self.options.objfcnval = 9.5

    status_run = backend.solve_gekko_model(
        StatusOnlyFailureModel(),
        solver_name="apopt",
    )

    assert status_run.failure_reason == "solver status 2"
    assert status_run.objective_value == 9.5


def test_solver_backend_low_level_option_helpers_cover_edge_cases(monkeypatch):
    assert backend._as_float_or_none(None) is None
    assert backend._as_float_or_none("bad") is None
    assert backend._normalise_solver_options(["", "# comment", "plain"]) == {
        "plain": ""
    }

    with pytest.raises(ValueError, match="mapping or a list"):
        backend._normalise_solver_options("tol 1e-6")

    with pytest.raises(RuntimeError, match="does not expose a run path"):
        backend._write_solver_option_file(
            _FakeGekkoModel(),
            "missing.opt",
            {"tol": "1e-6"},
        )

    monkeypatch.setattr(backend, "require_solver_backend", lambda _solver_name: None)
    model = _FakeGekkoModel()
    model.options.SOLVER_EXTENSION = "legacy"

    run = backend.configure_gekko_solver(model, "custom")

    assert run.extension == "legacy"
    assert model.options.SOLVER == "custom"


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


def test_solver_arrays_serialise_shapes_and_reject_invalid_entry_points() -> None:
    arrays = problem_to_solver_arrays(_four_stream_problem(), 14.0)

    assert arrays.array_shapes["T_h_in_period"] == [1, 2]
    payload = arrays.to_json_dict()
    assert payload["array_shapes"]["T_h_in_period"] == [1, 2]
    assert payload["arrays"]["period_ids"] == ["0"]

    with pytest.raises(TypeError, match="prepared PinchProblem"):
        problem_to_solver_arrays({"raw": "payload"}, 14.0)

    unprepared = PinchProblem.__new__(PinchProblem)
    unprepared._master_zone = None
    with pytest.raises(RuntimeError, match="prepare_problem"):
        problem_to_solver_arrays(unprepared, 14.0)

    with pytest.raises(ValueError, match="finite and positive"):
        problem_to_solver_arrays(_four_stream_problem(), 0.0)


def test_solver_arrays_reject_missing_stream_and_utility_collections() -> None:
    missing_hot = _four_stream_problem()
    missing_hot.master_zone.hot_streams._streams = {}
    with pytest.raises(ValueError, match="hot and cold streams"):
        problem_to_solver_arrays(missing_hot, 14.0)

    missing_hot_utility = _four_stream_problem()
    missing_hot_utility.master_zone.hot_utilities._streams = {}
    with pytest.raises(ValueError, match="hot and cold utilities"):
        problem_to_solver_arrays(missing_hot_utility, 14.0)


def test_solver_array_private_helpers_cover_period_and_value_edges() -> None:
    assert arrays_module._period_ids_and_weights(
        SimpleNamespace(period_ids={"base": 0}, weights=None)
    ) == (("base",), (1.0,))
    assert arrays_module._period_ids_and_weights(
        SimpleNamespace(period_ids={"base": 0, "peak": 1}, weights=[1.0])
    ) == (("base", "peak"), (1.0, 1.0))
    with pytest.raises(ValueError, match="operating period"):
        arrays_module._period_ids_and_weights(
            SimpleNamespace(period_ids=_TruthyEmptyMapping(), weights=[])
        )

    assert (
        arrays_module._temperature_contribution(
            SimpleNamespace(dt_cont=_SinglePeriodValue(0.25)),
            20.0,
        )
        == 5.0
    )
    assert (
        arrays_module._stream_heat_capacity_flowrate(
            SimpleNamespace(CP=_SinglePeriodValue(8.0)),
            SimpleNamespace(),
        )
        == 8.0
    )

    synthetic_utilities = arrays_module._ordered_utility_items(
        SimpleNamespace(_validated_data=SimpleNamespace(utilities=())),
        [
            ("zone.HU", SimpleNamespace(name="HU")),
            ("steam", SimpleNamespace(name="steam")),
        ],
    )
    assert [key for key, _stream, _record in synthetic_utilities] == ["steam"]
    assert arrays_module._items_in_input_order(
        [("stream-1", SimpleNamespace(name="A"))],
        (),
    ) == [("stream-1", SimpleNamespace(name="A"), None)]
    assert (
        arrays_module._matching_item_index(
            [("zone-a.hot-1", SimpleNamespace(name="hot-1"))],
            SimpleNamespace(name="hot-1", zone="zone-b"),
        )
        == 0
    )


def test_solver_array_optional_value_helper_accepts_supported_shapes() -> None:
    assert arrays_module._value(None, "K") == 0.0
    assert arrays_module._optional_value(None, "K") is None
    assert arrays_module._optional_value(SimpleNamespace(values=None), "K") is None
    assert arrays_module._optional_value(SimpleNamespace(values=[3.5]), "K") == 3.5
    assert (
        arrays_module._optional_value(
            SimpleNamespace(values=[4.0], unit="kW/delta_degC"),
            "kW/delta_degC",
        )
        == 4.0
    )
    assert arrays_module._optional_value(SimpleNamespace(value=None), "K") is None
    assert arrays_module._optional_value(SimpleNamespace(value=[5.0]), "K") == 5.0
    assert (
        arrays_module._optional_value(
            SimpleNamespace(value=[6.0], unit="K"),
            "K",
        )
        == 6.0
    )
    assert arrays_module._optional_value([7.0], "K") == 7.0
    assert arrays_module._optional_value(8.0, "K") == 8.0


def test_stagewise_model_rejects_solver_arrays_without_state_metadata() -> None:
    arrays = problem_to_solver_arrays(_four_stream_problem(), 14.0)
    incomplete_arrays = {
        key: value
        for key, value in arrays.arrays.items()
        if key not in {"period_ids", "period_weights"}
    }
    bad_arrays = PreparedSolverArrays(
        arrays=incomplete_arrays,
        axis_maps=arrays.axis_maps,
        unit_conventions=arrays.unit_conventions,
        stream_identities=arrays.stream_identities,
        utility_identities=arrays.utility_identities,
        configuration=arrays.configuration,
        preparation=arrays.preparation,
    )

    with pytest.raises(ValueError, match="period_ids is required"):
        StageWiseModel(
            name="solver-arrays-without-state",
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
    with pytest.raises(ValueError, match="period weight must be positive"):
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


def test_local_executor_collects_missing_worker_result_and_parent_errors() -> None:
    problem = _four_stream_problem()
    settings = workflow_settings_from_problem(problem)
    task = build_pinch_design_method_tasks(settings)[0]
    executor = LocalSynthesisExecutor()

    outcomes = executor._collect_outcomes((task,), (), {})

    assert outcomes[0].status == "failed"
    assert "worker did not return" in outcomes[0].error

    child = task.model_copy(update={"parent_task_id": "parent-task"})
    with pytest.raises(WorkflowContractError, match="Missing parent outcome"):
        executor._parent_problem(child, {})

    failed_parent = SimpleNamespace(status="failed")
    with pytest.raises(WorkflowContractError, match="failed parent"):
        executor._parent_problem(child, {"parent-task": failed_parent})

    successful_parent = SimpleNamespace(status="success")
    with pytest.raises(WorkflowContractError, match="no private solver problem"):
        executor._parent_problem(child, {"parent-task": successful_parent})


def test_local_executor_execute_records_build_failures(monkeypatch) -> None:
    problem = _four_stream_problem()
    settings = workflow_settings_from_problem(problem)
    task = build_pinch_design_method_tasks(settings)[0]
    executor = LocalSynthesisExecutor()

    monkeypatch.setattr(
        executor,
        "_build_problem",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cannot build")),
    )

    outcomes = executor.execute(
        (task,),
        problem=problem,
        parent_outcomes={},
        max_parallel=1,
    )

    assert outcomes[0].status == "failed"
    assert outcomes[0].error == "cannot build"


def test_local_executor_uses_worker_pool_for_prebuilt_tasks(monkeypatch) -> None:
    problem = _four_stream_problem()
    settings = workflow_settings_from_problem(problem)
    tasks = build_pinch_design_method_tasks(settings)
    tasks = (tasks[0], tasks[0].model_copy(update={"run_id": "second-worker-task"}))
    pool_calls = []

    class FakePool:
        def __init__(self, worker_count):
            self.worker_count = worker_count

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            args = tuple(iterable)
            pool_calls.append((self.worker_count, len(args)))
            return tuple(func(item) for item in args)

    class SuccessfulInternalProblem:
        case = SimpleNamespace(solver_run=SimpleNamespace(status="ok"))

        def get_solution(self, **kwargs):
            return SimpleNamespace(mSuccess=1)

        def extract_network(self, *, run_id):
            return HeatExchangerNetwork(run_id=run_id, total_annual_cost=1.0)

    executor = LocalSynthesisExecutor(worker_pool_factory=FakePool)
    monkeypatch.setattr(
        executor,
        "_build_problem",
        lambda *args, **kwargs: SuccessfulInternalProblem(),
    )

    outcomes = executor.execute(
        tasks,
        problem=problem,
        parent_outcomes={"force-local-build": SimpleNamespace(status="success")},
        max_parallel=2,
    )

    assert [outcome.status for outcome in outcomes] == ["success", "success"]
    assert pool_calls == [(2, 2)]


def test_solve_built_task_legacy_args_failures_and_verification_paths() -> None:
    task = HeatExchangerNetworkSynthesisTask(
        run_id="executor-helper",
        method="pinch_design_method",
        approach_temperature=14.0,
    )

    class FailedInternalProblem:
        solution_failure_reason = "solver stopped"

        def get_solution(self, **kwargs):
            self.kwargs = kwargs
            return None

    task_out, outcome, internal_problem = _solve_built_task(
        (task, FailedInternalProblem(), False, None)
    )

    assert task_out is task
    assert outcome.status == "failed"
    assert outcome.error == "solver stopped"
    assert internal_problem is None

    network_task = task.model_copy(
        update={
            "method": "network_evolution_method",
            "stage_count": 2,
            "derivative_threshold": 0.5,
        }
    )

    class InvalidSolvedCase:
        mSuccess = 1

        def verify(self):
            return False, ["temperature"]

    class InvalidInternalProblem:
        def get_solution(self, **kwargs):
            self.kwargs = kwargs
            return InvalidSolvedCase()

    invalid_problem = InvalidInternalProblem()
    _task, invalid_outcome, returned_problem = _solve_built_task(
        (
            network_task,
            invalid_problem,
            False,
            None,
            {
                "n_ad_branches": 2,
                "n_rm_branches": 3,
                "max_parallel": 4,
                "no_improvement_patience": 5,
            },
        )
    )

    assert invalid_outcome.status == "failed"
    assert invalid_outcome.error == "verification failed: temperature"
    assert returned_problem is invalid_problem
    assert invalid_problem.solution_failure_reason == invalid_outcome.error
    assert invalid_problem.kwargs["evolution_n_ad_branches"] == 2


def test_solve_built_task_success_exception_and_root_build_failure(monkeypatch) -> None:
    task = HeatExchangerNetworkSynthesisTask(
        run_id="executor-success",
        method="pinch_design_method",
        approach_temperature=14.0,
    )

    class SuccessfulInternalProblem:
        case = SimpleNamespace(solver_run=SimpleNamespace(status="ok"))

        def get_solution(self, **kwargs):
            return SimpleNamespace(mSuccess=1)

        def extract_network(self, *, run_id):
            return HeatExchangerNetwork(run_id=run_id, total_annual_cost=42.0)

    _task, outcome, internal_problem = _solve_built_task(
        (task, SuccessfulInternalProblem(), False, None)
    )

    assert outcome.status == "success"
    assert outcome.objective_value == pytest.approx(42.0)
    assert outcome.solver_status == "ok"
    assert internal_problem is not None

    class BrokenInternalProblem:
        def get_solution(self, **kwargs):
            raise RuntimeError("solver crashed")

    _task, failed_outcome, _problem = _solve_built_task(
        (task, BrokenInternalProblem(), False, None)
    )
    assert failed_outcome.status == "failed"
    assert failed_outcome.error == "solver crashed"

    monkeypatch.setattr(
        executor_module.LocalSynthesisExecutor,
        "_build_problem",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad build")),
    )

    _task, root_outcome, _problem = _build_and_solve_root_task(
        (
            task,
            _four_stream_problem(),
            False,
            None,
            {
                "n_ad_branches": 1,
                "n_rm_branches": 1,
                "max_parallel": 1,
                "no_improvement_patience": None,
            },
        )
    )
    assert root_outcome.status == "failed"
    assert root_outcome.error == "bad build"


def test_executor_option_helpers_cover_validation_and_fallback_paths() -> None:
    problem = _four_stream_problem(
        options={
            "HENS_STAGE_SELECTION": [1, 2, 3],
            "HENS_SYNTHESIS_QUALITY_TIER": 3,
        }
    )
    settings = workflow_settings_from_problem(problem)
    task = build_pinch_design_method_tasks(settings)[0].model_copy(
        update={"settings": {"solver_options": ["bad"]}}
    )

    with pytest.raises(WorkflowContractError, match="solver_options"):
        _solver_options_for_task(problem, task)

    assert _branch_breadth(4, tier=1) == 4
    assert (
        _pathway_evolution_options(
            HeatExchangerNetworkSynthesisTask(
                run_id="no-pathway",
                method="network_evolution_method",
                approach_temperature=14.0,
                derivative_threshold=0.5,
                stage_count=2,
                metadata={},
            )
        )
        == {}
    )
    assert _optional_int(None) is None
    assert _optional_int("7") == 7
    assert _legacy_pdm_stage_selection(problem) == "automated"
    two_stage_problem = _four_stream_problem(options={"HENS_STAGE_SELECTION": [2, 3]})
    assert _legacy_pdm_stage_selection(two_stage_problem) == [2, 3]
    assert _legacy_pdm_stage_selection(
        problem,
        task.model_copy(update={"settings": {"stage_selection": [2, 4]}}),
    ) == [2, 4]
    assert _solver_status(SimpleNamespace(case=SimpleNamespace(solver_run=None))) == (
        "success"
    )
    assert (
        _solver_status(
            SimpleNamespace(
                case=SimpleNamespace(
                    solver_run=SimpleNamespace(failure_reason="infeasible", status="ok")
                )
            )
        )
        == "infeasible"
    )
    assert _failed_task_outcome(task, "").error == (
        "heat exchanger network synthesis task failed"
    )


def test_executor_stage_packing_scope_and_factory_helpers() -> None:
    pdm_problem = _four_stream_problem(options={"HENS_STAGE_PACKING": "pdm"})
    tdm_problem = _four_stream_problem(options={"HENS_STAGE_PACKING": "tdm"})
    auto_problem = _four_stream_problem(options={"HENS_STAGE_PACKING": "auto"})
    settings = workflow_settings_from_problem(pdm_problem)
    pdm_task = build_pinch_design_method_tasks(settings)[0]
    tdm_task = HeatExchangerNetworkSynthesisTask(
        run_id="tdm-stage-packed",
        method="thermal_derivative_method",
        approach_temperature=14.0,
        derivative_threshold=0.5,
        stage_count=2,
    )

    pdm_factories = LocalSynthesisExecutor()._model_factories_for_task(
        pdm_task,
        pdm_problem,
    )
    tdm_factories = LocalSynthesisExecutor()._model_factories_for_task(
        tdm_task,
        tdm_problem,
    )

    assert "pinch_design_method" in pdm_factories
    assert "stagewise" in tdm_factories
    assert executor_module._stage_packing_scope(auto_problem) == "none"
    assert executor_module._stage_packed_pdm_factory().__name__ == (
        "StagePackedPinchDecompModel"
    )
    assert executor_module._stage_packed_stagewise_factory().__name__ == (
        "StagePackedStageWiseModel"
    )


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
    model.T_h_cont_period = [[10.0]]
    model.T_c_cont_period = [[10.0]]
    model.T_hu_cont_period = [[10.0]]
    model.T_cu_cont_period = [[10.0]]
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


def _pdm_target(**updates):
    values = {
        "period_id": "0",
        "period_idx": 0,
        "hot_utility_target": 1.0,
        "cold_utility_target": 0.0,
        "heat_recovery_target": 10.0,
        "hot_pinch": 100.0,
        "cold_pinch": 90.0,
        "shifted_pinch_temperature": 373.15,
    }
    values.update(updates)
    return pdm_decomposition.PinchDesignTarget(**values)


def _pdm_decomposition_kwargs(updates: dict | None = None) -> dict:
    values = {
        "pinch_location": "above",
        "period_targets": (_pdm_target(),),
        "z_i_active": (1,),
        "z_j_active": (1,),
        "z_i_active_by_period": ((1,),),
        "z_j_active_by_period": ((1,),),
        "clipped_hot_supply_temperatures_by_period": ((420.0,),),
        "clipped_hot_target_temperatures_by_period": ((380.0,),),
        "clipped_cold_supply_temperatures_by_period": ((300.0,),),
        "clipped_cold_target_temperatures_by_period": ((360.0,),),
        "S": 1,
        "K": 2,
        "manual_stage_selection": None,
        "hot_stream_identities": ("hot-1",),
        "cold_stream_identities": ("cold-1",),
        "unit_conventions": {"temperature": "K"},
        "dt_cont_convention": "test",
    }
    values.update(updates or {})
    return values


class _MultiPeriodDtCont:
    def __init__(self, period_values: list[float]) -> None:
        self.period_values = np.asarray(period_values, dtype=float)
        self.num_periods = len(period_values)

    def to(self, unit: str) -> "_MultiPeriodDtCont":
        assert unit == "delta_degC"
        return self


class _SinglePeriodValue:
    num_periods = 1

    def __init__(self, value: float) -> None:
        self.value = value

    def to(self, unit: str) -> "_SinglePeriodValue":
        assert unit
        return self


class _TruthyEmptyMapping(dict):
    def __bool__(self) -> bool:
        return True


class _AlgebraModel:
    def __init__(self) -> None:
        self.equations = []
        self.actions = []

    def Var(self, *, value=0.0, **kwargs):
        del kwargs
        return float(value)

    def Param(self, *, value=0.0, **kwargs):
        del kwargs
        return float(value)

    def Equation(self, expression):
        self.equations.append(expression)
        return expression

    def Minimize(self, expression):
        self.actions.append(("minimize", expression))
        return expression

    def Maximize(self, expression):
        self.actions.append(("maximize", expression))
        return expression

    def sum(self, values):
        return sum(values)


class _ValueCell:
    def __init__(self, value: float) -> None:
        self.VALUE = SimpleNamespace(value=[value])

    def __getitem__(self, index: int) -> float:
        return self.VALUE.value[index]


def _pdm_amalgamation_driver(
    *,
    multiperiod: bool = False,
    non_isothermal: bool = False,
) -> PinchDecompModel:
    driver = PinchDecompModel.__new__(PinchDecompModel)
    driver.framework = "PDM"
    driver.solver = "apopt"
    driver.solver_arrays = problem_to_solver_arrays(
        _two_state_problem() if multiperiod else _four_stream_problem(),
        14.0,
    )
    driver.dTmin = 14.0
    driver.z_restriction = None
    driver.min_dqda = 0.0
    driver.non_isothermal_model = non_isothermal
    driver.solver_options = []
    driver.I = 2
    driver.J = 2
    driver.tol = 1e-3
    return driver


def _pdm_multiperiod_side_result(*, required: bool) -> SimpleNamespace:
    side = _pdm_side_result(hot_utility_target=1.0 if required else 0.0)
    stream_count = 2
    stages = side.S
    boundary_count = side.K
    side.N_periods = 2
    side.HU_target_by_period = [0.0, 1.0] if required else [0.0, 0.0]
    side.CU_target_by_period = [0.0, 0.0]
    side.side_required = required
    side.z_i_active_period = [[0, 0], [1, 1]]
    side.z_j_active_period = [[0, 0], [1, 1]]
    side.Q_r_by_period = [
        _value_cell_grid(value, stream_count, stream_count, stages)
        for value in (0.0, 12.0)
    ]
    side.theta_1_by_period = [
        _value_cell_grid(value, stream_count, stream_count, stages)
        for value in (14.0, 30.0)
    ]
    side.theta_2_by_period = [
        _value_cell_grid(value, stream_count, stream_count, stages)
        for value in (14.0, 20.0)
    ]
    side.T_h_by_period = [
        [
            [
                _ValueCell(500.0 + 20.0 * n - 10.0 * i - 20.0 * k)
                for k in range(boundary_count)
            ]
            for i in range(stream_count)
        ]
        for n in range(2)
    ]
    side.T_c_by_period = [
        [
            [
                _ValueCell(300.0 + 20.0 * n + 10.0 * j + 20.0 * k)
                for k in range(boundary_count)
            ]
            for j in range(stream_count)
        ]
        for n in range(2)
    ]
    side.Q_h_by_period = [
        [_ValueCell(value) for _ in range(stream_count)] for value in (0.0, 6.0)
    ]
    side.Q_c_by_period = [
        [_ValueCell(0.0) for _ in range(stream_count)] for _ in range(2)
    ]
    side.non_isothermal_model = True
    side.X_by_period = [
        _value_cell_grid(value, stream_count, stream_count, stages)
        for value in (0.0, 0.25)
    ]
    side.Y_by_period = [
        _value_cell_grid(value, stream_count, stream_count, stages)
        for value in (0.0, 0.5)
    ]
    side.T_h_out_x_by_period = [
        _value_cell_grid(value, stream_count, stream_count, stages)
        for value in (0.0, 470.0)
    ]
    side.T_c_out_y_by_period = [
        _value_cell_grid(value, stream_count, stream_count, stages)
        for value in (0.0, 330.0)
    ]
    return side


def _pdm_side_result(
    *,
    hot_utility_target: float = 0.0,
    cold_utility_target: float = 0.0,
    m_success: int = 1,
    tac: float = 11.0,
    solve_time: float = 1.0,
    stages: int = 1,
    active: bool = True,
) -> SimpleNamespace:
    stream_count = 2
    boundary_count = stages + 1
    active_flags = [1 if active else 0] * stream_count
    t_h = [
        [_ValueCell(500.0 - 10.0 * i - 20.0 * k) for k in range(boundary_count)]
        for i in range(stream_count)
    ]
    t_c = [
        [_ValueCell(300.0 + 10.0 * j + 20.0 * k) for k in range(boundary_count)]
        for j in range(stream_count)
    ]

    return SimpleNamespace(
        side_required=bool(hot_utility_target or cold_utility_target),
        HU_target_by_period=[hot_utility_target],
        CU_target_by_period=[cold_utility_target],
        mSuccess=m_success,
        S=stages,
        K=boundary_count,
        TAC=tac,
        solve_time=solve_time,
        z_i_active=active_flags,
        z_j_active=active_flags,
        z_i_active_period=[active_flags],
        z_j_active_period=[active_flags],
        N_periods=1,
        non_isothermal_model=False,
        z=_value_cell_grid(1.0, stream_count, stream_count, stages),
        Q_r_by_period=[_value_cell_grid(12.0, stream_count, stream_count, stages)],
        theta_1_by_period=[_value_cell_grid(30.0, stream_count, stream_count, stages)],
        theta_2_by_period=[_value_cell_grid(20.0, stream_count, stream_count, stages)],
        T_h_by_period=[t_h],
        T_c_by_period=[t_c],
        Q_h_by_period=[[_ValueCell(5.0 + j) for j in range(stream_count)]],
        z_hu=[_ValueCell(1.0) for _ in range(stream_count)],
        Q_c_by_period=[[_ValueCell(7.0 + i) for i in range(stream_count)]],
        z_cu=[_ValueCell(1.0) for _ in range(stream_count)],
    )


def _value_cell_grid(
    base_value: float,
    rows: int,
    columns: int,
    depth: int,
) -> list[list[list[_ValueCell]]]:
    return [
        [
            [_ValueCell(base_value + row + column + item) for item in range(depth)]
            for column in range(columns)
        ]
        for row in range(rows)
    ]


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


def test_extraction_handles_absent_solver_arrays_as_empty_network() -> None:
    network = extract_heat_exchanger_network(
        SimpleNamespace(),
        _solver_arrays(),
        run_id="empty",
        method="PDM",
    )

    assert network.exchangers == ()
    assert network.method == "PDM"
    assert network.total_annual_cost is None
    assert network.utility_cost is None
    assert network.capital_cost is None
    assert network.source_metadata["hot_stage_boundary_temperatures"] == []
    assert network.source_metadata["cold_stage_boundary_temperatures"] == []


def test_extraction_uses_identity_fallbacks_and_requires_utilities() -> None:
    arrays_without_axis_maps = PreparedSolverArrays(
        arrays={},
        axis_maps={},
        unit_conventions={},
        stream_identities={
            "hot_process_streams": ["hot-1"],
            "cold_process_streams": ["cold-1"],
        },
        utility_identities={
            "hot_utilities": ["steam"],
            "cold_utilities": ["cooling-water"],
        },
        configuration={},
        preparation={},
    )

    assert extraction_module._identities_by_axis(
        arrays_without_axis_maps,
        "hot_process_streams",
    ) == ("hot-1",)
    assert extraction_module._identities_by_axis(
        arrays_without_axis_maps,
        "hot_utilities",
    ) == ("steam",)

    missing_hot_utility = PreparedSolverArrays(
        arrays={},
        axis_maps={
            "hot_process_streams": {"hot-1": 0},
            "cold_process_streams": {"cold-1": 0},
        },
        unit_conventions={},
        stream_identities={
            "hot_process_streams": ["hot-1"],
            "cold_process_streams": ["cold-1"],
        },
        utility_identities={"hot_utilities": [], "cold_utilities": ["cooling-water"]},
        configuration={},
        preparation={},
    )
    with pytest.raises(ValueError, match="hot utility"):
        extract_heat_exchanger_network(
            SimpleNamespace(),
            missing_hot_utility,
            run_id="missing-utility",
        )


def test_extraction_private_helpers_cover_temperature_and_metadata_edges() -> None:
    calculated_hot = extraction_module._hot_recovery_outlet(
        SimpleNamespace(T_h=[[650.0, 620.0]], f_h=[2.0]),
        0,
        0,
        0,
        10.0,
    )
    fallback_hot = extraction_module._hot_recovery_outlet(
        SimpleNamespace(T_h=[[650.0, 620.0]]),
        0,
        0,
        0,
        10.0,
    )
    calculated_cold = extraction_module._cold_recovery_outlet(
        SimpleNamespace(T_c=[[400.0, 450.0]], f_c=[5.0]),
        0,
        0,
        0,
        25.0,
    )
    fallback_cold = extraction_module._cold_recovery_outlet(
        SimpleNamespace(T_c=[[400.0, 450.0]]),
        0,
        0,
        0,
        25.0,
    )

    assert calculated_hot == 645.0
    assert fallback_hot == 620.0
    assert calculated_cold == 455.0
    assert fallback_cold == 400.0
    assert (
        extraction_module._capital_cost(SimpleNamespace(capital_cost_value=22.0))
        == 22.0
    )
    assert extraction_module._allowed(None) is True
    assert extraction_module._third_dimension(1.0) == 0
    assert extraction_module._index([[1.0]], None) is None


def test_extraction_operating_state_metadata_and_optional_float_edges() -> None:
    solved = SimpleNamespace(
        N_periods=2,
        period_ids=["base", "peak"],
        period_weights=[1.0, 3.0],
        Q_hu_total_by_period=[1.0, 2.0],
        Q_cu_total_by_period=[3.0, 4.0],
        Q_r_total_by_period=[5.0, 6.0],
        operating_cost_by_period=[7.0, 8.0],
        weighted_operating_cost_value=9.0,
        capital_cost_value=10.0,
    )

    network = extract_heat_exchanger_network(solved, _solver_arrays(), run_id="periods")

    assert network.source_metadata["operating_periods"] == {
        "period_ids": ["base", "peak"],
        "period_weights": [1.0, 3.0],
        "hot_utility_load_by_period": [1.0, 2.0],
        "cold_utility_load_by_period": [3.0, 4.0],
        "recovery_load_by_period": [5.0, 6.0],
        "operating_cost_by_period": [7.0, 8.0],
        "weighted_operating_cost": 9.0,
        "shared_capital_cost": 10.0,
    }
    assert extraction_module._optional_float(object()) is None
    assert extraction_module._optional_float({}) is None
    assert extraction_module._optional_float(SimpleNamespace(value=["4.5"])) == 4.5
    assert extraction_module._optional_float(SimpleNamespace(VALUE=["6.5"])) == 6.5


def test_pdm_target_and_decomposition_validators_reject_bad_inputs() -> None:
    with pytest.raises(ValidationError, match="pinch target values must be finite"):
        pdm_decomposition.PinchDesignTarget(
            period_id="0",
            period_idx=0,
            hot_utility_target=np.inf,
            cold_utility_target=0.0,
            heat_recovery_target=10.0,
            hot_pinch=100.0,
            cold_pinch=None,
            shifted_pinch_temperature=373.15,
        )
    with pytest.raises(ValidationError, match="pinch temperatures must be finite"):
        pdm_decomposition.PinchDesignTarget(
            period_id="0",
            period_idx=0,
            hot_utility_target=1.0,
            cold_utility_target=0.0,
            heat_recovery_target=10.0,
            hot_pinch=np.inf,
            cold_pinch=None,
            shifted_pinch_temperature=373.15,
        )

    target = _pdm_target(cold_pinch=None)
    assert target.cold_pinch is None

    invalid_cases = [
        ({"z_i_active": (2,)}, "active-stream flags"),
        (
            {"clipped_hot_supply_temperatures_by_period": ((np.nan,),)},
            "clipped stream temperatures",
        ),
        ({"S": -1, "K": 0}, "stage count S"),
        ({"K": 4}, "boundary count K"),
        ({"z_i_active": (1, 1)}, "hot active flags"),
        ({"z_j_active": (1, 1)}, "cold active flags"),
    ]
    for update, message in invalid_cases:
        with pytest.raises(ValidationError, match=message):
            pdm_decomposition.PinchDesignDecomposition(
                **_pdm_decomposition_kwargs(update)
            )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        pdm_decomposition.PinchDesignDecomposition(
            **_pdm_decomposition_kwargs(),
            target=_pdm_target(),
        )

    utility_only = pdm_decomposition.PinchDesignDecomposition(
        **_pdm_decomposition_kwargs(
            {
                "z_i_active": (0,),
                "z_j_active": (0,),
                "z_i_active_by_period": ((0,),),
                "z_j_active_by_period": ((0,),),
                "S": 0,
                "K": 1,
            }
        )
    )
    assert utility_only.S == 0
    assert utility_only.K == 1


def test_pdm_decomposition_keeps_distinct_period_targets_and_union_topology() -> None:
    arrays = problem_to_solver_arrays(_two_state_problem(), 10.0)
    arrays.arrays["T_h_in_period"][:, 0] = [500.0, 600.0]
    arrays.arrays["T_h_out_period"][:, 0] = [450.0, 550.0]
    targets = (
        _pdm_target(
            period_id="base",
            period_idx=0,
            hot_utility_target=10.0,
            shifted_pinch_temperature=500.0,
        ),
        _pdm_target(
            period_id="peak",
            period_idx=1,
            hot_utility_target=20.0,
            shifted_pinch_temperature=520.0,
        ),
    )

    decomposition = pdm_decomposition._build_decomposition(
        arrays=arrays,
        period_targets=targets,
        dTmin=10.0,
        pinch_location="above",
        stage_selection="automated",
    )

    assert [target.period_id for target in decomposition.period_targets] == [
        "base",
        "peak",
    ]
    assert [target.hot_utility_target for target in decomposition.period_targets] == [
        10.0,
        20.0,
    ]
    assert decomposition.z_i_active_by_period[0][0] == 0
    assert decomposition.z_i_active_by_period[1][0] == 1
    assert decomposition.z_i_active[0] == 1
    assert decomposition.clipped_hot_supply_temperatures_by_period[0][0] == 0.0
    assert decomposition.clipped_hot_supply_temperatures_by_period[1][0] == 600.0


def test_pdm_decomposition_private_helpers_cover_guard_and_stage_edges() -> None:
    with pytest.raises(ValueError, match="pinch_location"):
        pdm_decomposition.build_pinch_design_decomposition(
            _four_stream_problem(),
            14.0,
            pinch_location="sideways",
        )
    with pytest.raises(ValueError, match="prepared PinchProblem"):
        pdm_decomposition._zone_with_hen_dt_contribution(
            SimpleNamespace(master_zone=None),
            dTmin=14.0,
        )

    assert pdm_decomposition._stream_dt_cont_with_minimum(
        SimpleNamespace(),
        minimum_dt_cont=7.0,
    ) == {"value": 7.0, "unit": "delta_degC"}
    assert pdm_decomposition._stream_dt_cont_with_minimum(
        SimpleNamespace(_dt_cont=_MultiPeriodDtCont([1.0, 9.0])),
        minimum_dt_cont=7.0,
    ) == {"values": [7.0, 9.0], "unit": "delta_degC"}

    assert (
        pdm_decomposition._shifted_pinch_temperature(
            hot_utility_target=0.0,
            cold_utility_target=1.0,
            hot_pinch=80.0,
            cold_pinch=40.0,
        )
        == 313.15
    )
    assert (
        pdm_decomposition._shifted_pinch_temperature(
            hot_utility_target=1.0,
            cold_utility_target=0.0,
            hot_pinch=80.0,
            cold_pinch=40.0,
        )
        == 353.15
    )
    assert (
        pdm_decomposition._shifted_pinch_temperature(
            hot_utility_target=0.0,
            cold_utility_target=1.0,
            hot_pinch=80.0,
            cold_pinch=None,
        )
        is None
    )

    with pytest.raises(ValueError, match="shifted pinch"):
        pdm_decomposition._build_decomposition(
            arrays=problem_to_solver_arrays(_four_stream_problem(), 10.0),
            period_targets=(_pdm_target(shifted_pinch_temperature=None),),
            dTmin=10.0,
            pinch_location="above",
            stage_selection="automated",
        )

    T_h_in = np.array([410.0, 500.0])
    T_h_out = np.array([390.0, 500.0])
    T_c_in = np.array([390.0, 410.0])
    T_c_out = np.array([420.0, 430.0])
    z_i_active, z_j_active = pdm_decomposition._clip_stream_temperatures(
        T_h_in=T_h_in,
        T_h_out=T_h_out,
        T_c_in=T_c_in,
        T_c_out=T_c_out,
        shifted_pinch_temperature=400.0,
        dTmin=10.0,
        pinch_location="below",
    )

    assert z_i_active == (1, 0)
    assert z_j_active == (1, 0)
    np.testing.assert_allclose(T_h_in, [405.0, 0.0])
    np.testing.assert_allclose(T_h_out, [390.0, 0.0])
    np.testing.assert_allclose(T_c_in, [390.0, 0.0])
    np.testing.assert_allclose(T_c_out, [395.0, 0.0])
    assert (
        pdm_decomposition._stage_count(
            pinch_location="below",
            stage_selection=(2, 3),
            z_i_active=(1,),
            z_j_active=(1,),
        )
        == 3
    )

    with pytest.raises(ValueError, match="exactly two"):
        pdm_decomposition._manual_stage_selection((1,))
    with pytest.raises(ValueError, match="positive integers"):
        pdm_decomposition._manual_stage_selection((1, 0))


def test_pinch_decomp_model_calculate_pinch_rejects_mismatched_or_missing_pinch() -> (
    None
):
    model = PinchDecompModel.__new__(PinchDecompModel)
    model.pinch_loc = "above"
    model.N_periods = 1
    model.period_ids = np.array(["0"])
    model.tol = 1e-3
    model.pinch_decomposition = SimpleNamespace(
        pinch_location="below",
        period_targets=(_pdm_target(),),
    )
    with pytest.raises(ValueError, match="location"):
        model.calculate_pinch()

    model.pinch_decomposition = SimpleNamespace(
        pinch_location="above",
        period_targets=(_pdm_target(shifted_pinch_temperature=None),),
    )
    with pytest.raises(ValueError, match="shifted pinch"):
        model.calculate_pinch()


def test_pinch_decomp_calculate_pinch_keeps_period_targets_and_later_side_need():
    model = PinchDecompModel.__new__(PinchDecompModel)
    model.pinch_loc = "above"
    model.N_periods = 2
    model.period_ids = np.array(["base", "peak"])
    model.tol = 1e-3
    model.pinch_decomposition = SimpleNamespace(
        pinch_location="above",
        period_targets=(
            _pdm_target(
                period_id="base",
                period_idx=0,
                hot_utility_target=0.0,
                cold_utility_target=10.0,
                shifted_pinch_temperature=400.0,
            ),
            _pdm_target(
                period_id="peak",
                period_idx=1,
                hot_utility_target=25.0,
                cold_utility_target=0.0,
                shifted_pinch_temperature=450.0,
            ),
        ),
    )

    model.calculate_pinch()

    assert model.HU_target_by_period == [0.0, 25.0]
    assert model.CU_target_by_period == [10.0, 0.0]
    assert model.T_pinch_by_period == [400.0, 450.0]
    assert model.side_required is True


def test_pinch_decomp_below_preprocessing_handles_inactive_streams_and_manual_stages() -> (
    None
):
    model = PinchDecompModel.__new__(PinchDecompModel)
    model.pinch_loc = "below"
    model.pinch_decomposition = SimpleNamespace(
        manual_stage_selection=(2, 3),
        S=3,
        clipped_hot_supply_temperatures_by_period=((405.0, 0.0),),
        clipped_hot_target_temperatures_by_period=((390.0, 0.0),),
        clipped_cold_supply_temperatures_by_period=((390.0, 0.0),),
        clipped_cold_target_temperatures_by_period=((395.0, 0.0),),
        z_i_active_by_period=((1, 0),),
        z_j_active_by_period=((1, 0),),
        z_i_active=(1, 0),
        z_j_active=(1, 0),
    )
    model.dTmin = 10.0
    model.N_periods = 1
    model.f_h_period = np.array([[2.0, 3.0]])
    model.f_c_period = np.array([[4.0, 5.0]])
    model.T_h_in_period = np.array([[410.0, 500.0]])
    model.T_h_out_period = np.array([[390.0, 500.0]])
    model.T_c_in_period = np.array([[390.0, 410.0]])
    model.T_c_out_period = np.array([[420.0, 430.0]])
    model.htc_h_period = np.array([[1.0, 2.0]])
    model.htc_c_period = np.array([[3.0, 4.0]])
    model.htc_hu_period = np.array([[5.0]])
    model.htc_cu_period = np.array([[6.0]])
    model.tol = 1e-6
    model._recovery_approach_temperature = lambda i, j, n: 10.0

    model._set_multiperiod_preprocessing()

    assert model.S == 3
    assert model.K == 4
    assert model.z_i_active == [1, 0]
    assert model.z_j_active == [1, 0]
    np.testing.assert_allclose(model.T_h_in_period[0], [405.0, 0.0])
    np.testing.assert_allclose(model.T_h_out_period[0], [390.0, 0.0])
    np.testing.assert_allclose(model.T_c_in_period[0], [390.0, 0.0])
    np.testing.assert_allclose(model.T_c_out_period[0], [395.0, 0.0])
    assert model.z_hu_feasible == [0, 0]
    assert model.z_cu_feasible == [1, 0]


def test_pinch_decomp_non_integer_superstructure_builds_param_binaries() -> None:
    model = PinchDecompModel.__new__(PinchDecompModel)
    model.m = _AlgebraModel()
    model.N_periods = 1
    model.I = 1
    model.J = 1
    model.S = 2
    model.K = 3
    model.integers = False
    model.Q_max_period = np.array([[[100.0]]])
    model.Qtot_sh_period = np.array([[20.0]])
    model.Qtot_sc_period = np.array([[15.0]])
    model.z_allowed = [[[1, 0]]]
    model.z_cu_allowed = [0]
    model.z_hu_allowed = [1]
    model.z_i_active = [1]
    model.z_j_active = [1]
    model.z_i_active_period = [[1]]
    model.z_j_active_period = [[1]]
    model.T_h_in_period = np.array([[500.0]])
    model.T_h_out_period = np.array([[450.0]])
    model.T_c_in_period = np.array([[300.0]])
    model.T_c_out_period = np.array([[350.0]])
    model.f_h_period = np.array([[2.0]])
    model.f_c_period = np.array([[3.0]])
    model.T_h_cont_period = np.array([[5.0]])
    model.T_c_cont_period = np.array([[5.0]])
    model.T_hu_cont_period = np.array([[5.0]])
    model.T_cu_cont_period = np.array([[5.0]])
    model.T_hu_in_period = np.array([[600.0]])
    model.T_hu_out_period = np.array([[580.0]])
    model.T_cu_in_period = np.array([[280.0]])
    model.T_cu_out_period = np.array([[300.0]])
    model.dTmin = 10.0
    model._recovery_approach_temperature = lambda i, j, n: 10.0

    model._set_multiperiod_stage_wise_superstructure()

    assert model.z == [[[1.0, 0.0]]]
    assert model.z_hu == [1.0]
    assert model.z_cu == [0.0]
    assert model.dqda == []
    assert model.alpha == []
    assert model.m.equations


@pytest.mark.parametrize(
    "goal, expected_action",
    [
        ("total utility", "minimize"),
        ("heat recovery", "maximize"),
        ("min units", "minimize"),
    ],
)
def test_pinch_decomp_set_obj_covers_remaining_objective_modes(
    goal: str,
    expected_action: str,
) -> None:
    model = PinchDecompModel.__new__(PinchDecompModel)
    model.m = _AlgebraModel()
    model.minimisation_goal = goal
    model.N_periods = 1
    model.I = 1
    model.J = 1
    model.S = 1
    model.Q_h_by_period = [[2.0]]
    model.Q_c_by_period = [[3.0]]
    model.Q_r_by_period = [[[[4.0]]]]
    model.z = [[[1.0]]]
    model._weighted_state_average = lambda values: sum(values) / len(values)

    model.set_obj()

    assert model.m.actions[-1][0] == expected_action


def test_pinch_decomp_post_process_skips_failed_model_and_copy_helpers_cover_shapes() -> (
    None
):
    failed = PinchDecompModel.__new__(PinchDecompModel)
    failed.mSuccess = 0

    assert failed.get_post_process() is None

    source = SimpleNamespace(
        N_periods=1,
        z=[[[_ValueCell(1.0)]]],
        Q_r_by_period=[[[[_ValueCell(12.0)]]]],
        theta_1_by_period=[[[[_ValueCell(30.0)]]]],
        theta_2_by_period=[[[[_ValueCell(20.0)]]]],
        non_isothermal_model=False,
    )
    target = SimpleNamespace(
        N_periods=1,
        z=[[[_ValueCell(0.0)]]],
        Q_r_by_period=[[[[_ValueCell(0.0)]]]],
        theta_1_by_period=[[[[_ValueCell(0.0)]]]],
        theta_2_by_period=[[[[_ValueCell(0.0)]]]],
    )
    copier = PinchDecompModel.__new__(PinchDecompModel)

    copier._copy_recovery_match(target, source, 0, 0, 0, 0)

    assert target.z[0][0][0][0] == 1.0
    assert target.Q_r_by_period[0][0][0][0][0] == 12.0
    assert target.theta_1_by_period[0][0][0][0][0] == 30.0
    assert target.theta_2_by_period[0][0][0][0][0] == 20.0

    non_iso_source = SimpleNamespace(
        N_periods=1,
        non_isothermal_model=True,
        z=[[[_ValueCell(1.0)]]],
        Q_r_by_period=[[[[_ValueCell(10.0)]]]],
        theta_1_by_period=[[[[_ValueCell(9.0)]]]],
        theta_2_by_period=[[[[_ValueCell(8.0)]]]],
        X_by_period=[[[[_ValueCell(1.0)]]]],
        Y_by_period=[[[[_ValueCell(0.0)]]]],
        T_h_out_x_by_period=[[[[_ValueCell(470.0)]]]],
        T_c_out_y_by_period=[[[[_ValueCell(330.0)]]]],
    )
    non_iso_target = SimpleNamespace(
        N_periods=1,
        z=[[[_ValueCell(0.0)]]],
        Q_r_by_period=[[[[_ValueCell(0.0)]]]],
        theta_1_by_period=[[[[_ValueCell(0.0)]]]],
        theta_2_by_period=[[[[_ValueCell(0.0)]]]],
        X_by_period=[[[[_ValueCell(0.0)]]]],
        Y_by_period=[[[[_ValueCell(0.0)]]]],
        T_h_out_x_by_period=[[[[_ValueCell(0.0)]]]],
        T_c_out_y_by_period=[[[[_ValueCell(0.0)]]]],
    )

    copier._copy_recovery_match(non_iso_target, non_iso_source, 0, 0, 0, 0)

    assert non_iso_target.Q_r_by_period[0][0][0][0][0] == 10.0
    assert non_iso_target.X_by_period[0][0][0][0][0] == 1.0
    assert non_iso_target.Y_by_period[0][0][0][0][0] == 0.0
    assert non_iso_target.T_h_out_x_by_period[0][0][0][0][0] == 470.0
    assert non_iso_target.T_c_out_y_by_period[0][0][0][0][0] == 330.0


def test_pinch_decomp_amalgamate_handles_failure_and_single_sided_networks() -> None:
    driver = _pdm_amalgamation_driver()

    with pytest.raises(ValueError, match="Pinch Decomposition failed"):
        driver.amalgamate_networks(
            above_case=_pdm_side_result(hot_utility_target=1.0, m_success=0),
            below_case=_pdm_side_result(),
        )

    above_only = driver.amalgamate_networks(
        above_case=_pdm_side_result(hot_utility_target=1.0, active=False),
        below_case=_pdm_side_result(),
    )
    assert above_only.S == 1
    assert above_only.K == 2
    assert above_only.mSuccess == 1
    assert above_only.minimisation_goal == "hot utility"
    assert above_only.z_allowed[0][0] == [1]

    below_only = driver.amalgamate_networks(
        above_case=_pdm_side_result(),
        below_case=_pdm_side_result(cold_utility_target=1.0, active=False),
    )
    assert below_only.S == 1
    assert below_only.K == 2
    assert below_only.mSuccess == 1
    assert below_only.minimisation_goal == "cold utility"
    assert below_only.z_allowed[0][0] == [1]


def test_pdm_amalgamation_retains_later_period_only_nonisothermal_matches() -> None:
    driver = _pdm_amalgamation_driver(multiperiod=True, non_isothermal=True)

    network = driver.amalgamate_networks(
        above_case=_pdm_multiperiod_side_result(required=True),
        below_case=_pdm_multiperiod_side_result(required=False),
    )

    assert network.N_periods == 2
    assert network.Q_r_by_period[0][0][0][0][0] == pytest.approx(0.0)
    assert network.Q_r_by_period[1][0][0][0][0] == pytest.approx(12.0)
    assert network.Q_h_by_period[0][0][0] == pytest.approx(0.0)
    assert network.Q_h_by_period[1][0][0] == pytest.approx(6.0)
    assert network.z_allowed[0][0][0] == 1
    assert network.X_by_period[1][0][0][0][0] == pytest.approx(0.25)
    assert network.Y_by_period[1][0][0][0][0] == pytest.approx(0.5)
    assert network.T_h_out_x_by_period[1][0][0][0][0] == pytest.approx(470.0)
    assert network.T_c_out_y_by_period[1][0][0][0][0] == pytest.approx(330.0)


def test_pinch_decomp_amalgamate_combines_successful_above_and_below_networks() -> None:
    driver = _pdm_amalgamation_driver()

    amalgamated = driver.amalgamate_networks(
        above_case=_pdm_side_result(hot_utility_target=1.0, tac=11.0, solve_time=2.0),
        below_case=_pdm_side_result(cold_utility_target=1.0, tac=13.0, solve_time=3.0),
    )

    assert amalgamated.S == 2
    assert amalgamated.K == 3
    assert amalgamated.mSuccess == 1
    assert amalgamated.TAC == 24.0
    assert amalgamated.solve_time == 5.0
    assert amalgamated.z_allowed[0][0] == [1, 1]
