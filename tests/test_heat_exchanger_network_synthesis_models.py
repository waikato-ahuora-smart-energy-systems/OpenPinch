"""HENS-06 private model-boundary tests."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from OpenPinch.classes.heat_exchanger import HeatExchangerKind
from OpenPinch.lib.config import tol
from OpenPinch.services.heat_exchanger_network_synthesis._dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.services.heat_exchanger_network_synthesis.array_adapter import (
    PreparedSolverArrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models import backend
from OpenPinch.services.heat_exchanger_network_synthesis.models.extraction import (
    extract_heat_exchanger_network,
)
from OpenPinch.services.heat_exchanger_network_synthesis.models.problem import (
    InternalHeatExchangerNetworkProblem,
    ModelSliceUnavailableError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
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


@pytest.mark.parametrize("framework", ["PDM", "TDM", "ESM"])
def test_internal_problem_rejects_concrete_model_loads_in_hens06(
    framework: str,
) -> None:
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=_solver_arrays(),
        framework=framework,
        stages=2,
    )

    with pytest.raises(
        ModelSliceUnavailableError,
        match=r"HENS-06.*Concrete PDM/TDM/ESM.*HENS-07.*HENS-08",
    ):
        problem.load_model()


@pytest.mark.parametrize(
    ("framework", "factory_name"),
    [
        ("PDM", "pinch_decomposition"),
        ("TDM", "stagewise"),
        ("ESM", "stagewise"),
    ],
)
def test_internal_problem_rejects_registered_factories_before_they_can_run(
    framework: str,
    factory_name: str,
) -> None:
    problem = InternalHeatExchangerNetworkProblem(
        solver_arrays=_solver_arrays(),
        framework=framework,
        stages=2,
    )

    def forbidden_factory(*_args, **_kwargs):
        raise AssertionError("HENS-06 must not call concrete model factories")

    with pytest.raises(
        ModelSliceUnavailableError,
        match=rf"Factory registrations were ignored: {factory_name}",
    ):
        problem.get_solution(model_factories={factory_name: forbidden_factory})


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

    assert network.exchanger_between(
        source_stream="hot-utility",
        sink_stream="cold-B",
        kind=HeatExchangerKind.HOT_UTILITY,
    ) is None
