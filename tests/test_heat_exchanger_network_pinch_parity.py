"""Pinch target fixture matrix tests for the OpenHENS migration."""

from __future__ import annotations

from pathlib import Path

from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.pinch_design_method import (
    build_pinch_design_method_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
CASE_IDS = (
    "Four-stream-Yee-and-Grossmann-1990-1",
    "Nine-stream-Linnhoff-and-Ahmad-1999-1",
)
REQUIRED_DTMIN_GRID = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
AUTO_STAGE_SELECTION = "automated"
ABS_TOL = 1e-6


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


def _load_problem(case_id: str, *, reordered: bool = False) -> PinchProblem:
    suffix = ".reordered" if reordered else ""
    return PinchProblem(source=FIXTURE_ROOT / f"{case_id}{suffix}.json")
