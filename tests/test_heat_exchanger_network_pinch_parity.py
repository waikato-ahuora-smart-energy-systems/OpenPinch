"""Pinch target fixture matrix tests for the OpenHENS migration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import (
    pinch_design_snapshot as snapshot_module,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.arrays import (
    PreparedSolverArrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver.pinch_design_snapshot import (
    _source_style_target_snapshot,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.pinch_design_method import (
    build_pinch_design_method_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
CASE_IDS = (
    "Four-stream-Yee-and-Grossmann-1990-1",
    "Nine-stream-Linnhoff-and-Ahmad-1999-1",
)
COMPARISON_CASE_IDS = (
    "Four-stream-Escobar-and-Trierweiler-2013-1",
    "Four-stream-Yee-and-Grossmann-1990-1",
    "Five-stream-Bogataj-and-Kravanja-2012-1",
    "Five-stream-Kim-et-al-2017-1",
    "Six-stream-Spray-Dryer-2025-1",
    "Six-stream-Yee-and-Grossmann-1990-1",
)
REQUIRED_DTMIN_GRID = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
AUTO_STAGE_SELECTION = "automated"
ABS_TOL = 1e-6


@pytest.mark.parametrize("case_id", COMPARISON_CASE_IDS)
def test_requested_openhens_comparison_fixture_loads(case_id: str) -> None:
    problem = _load_problem(case_id)

    assert len(problem.hot_streams) > 0
    assert len(problem.cold_streams) > 0
    assert len(problem.hot_utilities) > 0
    assert len(problem.cold_utilities) > 0


def test_required_case_matrix_has_no_hu_or_cu_threshold_rows_to_cover() -> None:
    hu_thresholds = []
    cu_thresholds = []
    for case_id in CASE_IDS:
        problem = _load_problem(case_id)
        for dTmin in REQUIRED_DTMIN_GRID:
            snapshot = build_pinch_design_method_snapshot(
                problem,
                dTmin,
                pinch_location="above",
                stage_selection=AUTO_STAGE_SELECTION,
            )
            if abs(snapshot.target.hot_utility_target) <= ABS_TOL:
                hu_thresholds.append((case_id, dTmin))
            if abs(snapshot.target.cold_utility_target) <= ABS_TOL:
                cu_thresholds.append((case_id, dTmin))

    assert hu_thresholds == []
    assert cu_thresholds == []


def test_source_style_pdm_targets_preserve_openhens_float_literals() -> None:
    kim_targets = _source_style_target_snapshot(
        _solver_arrays(
            hot_capacity_flows=[228.5, 20.4, 53.8],
            cold_capacity_flows=[93.3, 196.1],
        )
    )
    bogataj_targets = _source_style_target_snapshot(
        _solver_arrays(
            hot_target_temperatures=[350.15, 353.15, 363.15],
            hot_capacity_flows=[2.285, 0.204, 0.538],
            cold_capacity_flows=[0.933, 1.961],
        )
    )

    assert repr(kim_targets["hot_utility_target"]) == "np.float64(10645.2)"
    assert repr(kim_targets["cold_utility_target"]) == "np.float64(8395.2)"
    assert repr(bogataj_targets["hot_utility_target"]) == (
        "np.float64(106.45200000000001)"
    )
    assert repr(bogataj_targets["cold_utility_target"]) == (
        "np.float64(85.58400000000002)"
    )


def test_source_style_targets_use_problem_environment_temperature(monkeypatch) -> None:
    problem = _load_problem(CASE_IDS[0])
    problem.master_zone.config.T_ENV = 23.5
    captured: dict[str, float] = {}

    def fake_source_style_target_snapshot(
        arrays: PreparedSolverArrays,
        *,
        reference_temperature: float | None = None,
    ) -> dict[str, float]:
        del arrays
        captured["reference_temperature"] = float(reference_temperature)
        return {
            "hot_utility_target": 1.0,
            "cold_utility_target": 1.0,
            "heat_recovery_target": 1.0,
            "hot_pinch": None,
            "cold_pinch": None,
        }

    monkeypatch.setattr(
        snapshot_module,
        "_source_style_target_snapshot",
        fake_source_style_target_snapshot,
    )

    snapshot_module._calculate_openpinch_targets(
        problem,
        _solver_arrays(
            hot_capacity_flows=[228.5, 20.4, 53.8],
            cold_capacity_flows=[93.3, 196.1],
        ),
    )

    assert captured["reference_temperature"] == 23.5


def _load_problem(case_id: str, *, reordered: bool = False) -> PinchProblem:
    suffix = ".reordered" if reordered else ""
    return PinchProblem(source=FIXTURE_ROOT / f"{case_id}{suffix}.json")


def _solver_arrays(
    *,
    hot_target_temperatures: list[float] | None = None,
    hot_capacity_flows: list[float],
    cold_capacity_flows: list[float],
) -> PreparedSolverArrays:
    return PreparedSolverArrays(
        arrays={
            "T_h_in": np.array([432.15, 540.15, 616.15]),
            "T_h_out": np.array(hot_target_temperatures or [350.15, 361.15, 363.15]),
            "f_h": np.array(hot_capacity_flows),
            "T_h_cont": np.array([5.0, 5.0, 5.0]),
            "T_c_in": np.array([299.15, 391.15]),
            "T_c_out": np.array([400.15, 538.15]),
            "f_c": np.array(cold_capacity_flows),
            "T_c_cont": np.array([5.0, 5.0]),
        },
        axis_maps={},
        unit_conventions={},
        stream_identities={},
        utility_identities={},
        configuration={},
        preparation={},
    )
