"""Pinch target fixture matrix tests for the OpenHENS migration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import (
    pinch_design_decomposition as decomposition_module,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
    build_pinch_design_decomposition,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.configuration import tol
from tests.support.paths import FIXTURES_ROOT, REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
FIXTURE_ROOT = FIXTURES_ROOT / "openhens"
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
            decomposition = build_pinch_design_decomposition(
                problem,
                dTmin,
                pinch_location="above",
                stage_selection=AUTO_STAGE_SELECTION,
            )
            target = decomposition.period_targets[0]
            if abs(target.hot_utility_target) <= ABS_TOL:
                hu_thresholds.append((case_id, dTmin))
            if abs(target.cold_utility_target) <= ABS_TOL:
                cu_thresholds.append((case_id, dTmin))

    assert hu_thresholds == []
    assert cu_thresholds == []


def test_decomposition_targets_are_finite_for_required_fixture() -> None:
    problem = _load_problem(CASE_IDS[0])

    (target,) = decomposition_module._calculate_openpinch_targets(problem, dTmin=10.0)

    assert target.hot_utility_target > 0.0
    assert target.cold_utility_target > 0.0
    assert target.heat_recovery_target > 0.0
    assert target.shifted_pinch_temperature is not None


@pytest.mark.parametrize("multiplier", [0.0, 0.5, 2.0])
def test_decomposition_targets_preserve_original_effective_contributions(
    monkeypatch,
    multiplier,
) -> None:
    problem = _load_problem(CASE_IDS[0])
    original_streams = list(problem._master_zone.all_streams)
    bases = [1.0, 9.0, 0.0, tol / 4.0, 3.0, 0.0]
    assert len(original_streams) == len(bases)
    for stream, base in zip(original_streams, bases):
        stream.delta_t_contribution = base
    problem._master_zone.dt_cont_multiplier = multiplier
    for stream in original_streams:
        stream.delta_t_contribution_multiplier_locked = True
    expected = [base * multiplier if base * multiplier > tol else 6.0 for base in bases]
    captured: dict[str, float | bool | list[float]] = {}

    def fake_compute_direct_integration_targets(zone, args=None):
        assert args == {"period_id": "0"}
        captured["same_zone"] = zone is problem._master_zone
        captured["dt_cont_multiplier"] = zone.dt_cont_multiplier
        captured["process_dt_cont"] = [
            float(stream.delta_t_contribution.to("delta_degC").value)
            for stream in zone.all_streams
        ]
        captured["process_dt_cont_act"] = [
            float(stream.effective_delta_t_contribution.to("delta_degC").value)
            for stream in zone.all_streams
        ]
        return SimpleNamespace(
            hot_utility_target=1.0,
            cold_utility_target=1.0,
            heat_recovery_target=1.0,
            hot_pinch=100.0,
            cold_pinch=90.0,
        )

    monkeypatch.setattr(
        decomposition_module,
        "compute_direct_integration_targets",
        fake_compute_direct_integration_targets,
    )

    (target,) = decomposition_module._calculate_openpinch_targets(problem, dTmin=12.0)

    assert captured["same_zone"] is False
    assert captured["dt_cont_multiplier"] == pytest.approx(1.0)
    assert captured["process_dt_cont"] == pytest.approx(expected)
    assert captured["process_dt_cont_act"] == pytest.approx(expected)
    assert target.shifted_pinch_temperature == pytest.approx(373.15)

    for stream, base in zip(original_streams, bases):
        assert stream.delta_t_contribution.value == pytest.approx(base)
        assert stream.effective_delta_t_contribution.value == pytest.approx(
            base * multiplier
        )
        assert stream.delta_t_contribution_multiplier == multiplier
        assert stream.delta_t_contribution_multiplier_locked is True
    assert problem._master_zone.dt_cont_multiplier == multiplier


def _load_problem(case_id: str, *, reordered: bool = False) -> PinchProblem:
    suffix = ".reordered" if reordered else ""
    return PinchProblem(source=FIXTURE_ROOT / f"{case_id}{suffix}.json")
