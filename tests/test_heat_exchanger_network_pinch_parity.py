"""Pinch target parity tests for the OpenHENS migration."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.services.heat_exchanger_network_synthesis.array_adapter import (
    problem_to_solver_arrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.pinch_decomposition import (
    PinchDecompositionSnapshot,
    build_pinch_decomposition_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
OPENHENS_ROOT = REPO_ROOT.parent / "OpenHENS"
OPENHENS_SOURCE_COMMIT = "2afc14b7779482fc829edb1c3fa187b918d7fb19"
CASE_IDS = (
    "Four-stream-Yee-and-Grossmann-1990-1",
    "Nine-stream-Linnhoff-and-Ahmad-1999-1",
)
REQUIRED_DTMIN_GRID = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
PINCH_LOCATIONS = ("above", "below")
AUTO_STAGE_SELECTION = "automated"
MANUAL_STAGE_SELECTION = (2, 3)
ABS_TOL = 1e-6


@dataclass(frozen=True)
class SourcePinchSnapshot:
    """Fields from source OpenHENS ``PinchDecompModel`` methods."""

    hot_utility_target: float
    cold_utility_target: float
    heat_recovery_target: float
    hot_pinch: float
    cold_pinch: float
    shifted_pinch_temperature: float
    z_i_active: tuple[int, ...]
    z_j_active: tuple[int, ...]
    clipped_hot_supply_temperatures: tuple[float, ...]
    clipped_hot_target_temperatures: tuple[float, ...]
    clipped_cold_supply_temperatures: tuple[float, ...]
    clipped_cold_target_temperatures: tuple[float, ...]
    S: int
    K: int


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("dTmin", REQUIRED_DTMIN_GRID)
@pytest.mark.parametrize("pinch_location", PINCH_LOCATIONS)
def test_openpinch_problem_table_matches_source_openhens_pinch_decomposition_grid(
    case_id: str,
    dTmin: float,
    pinch_location: Literal["above", "below"],
) -> None:
    problem = _load_problem(case_id)

    source = _source_openhens_pinch_snapshot(
        problem,
        dTmin,
        pinch_location=pinch_location,
        stage_selection=AUTO_STAGE_SELECTION,
    )
    native = build_pinch_decomposition_snapshot(
        problem,
        dTmin,
        pinch_location=pinch_location,
        stage_selection=AUTO_STAGE_SELECTION,
    )

    _assert_snapshots_match(native, source)
    assert native.target.target_access_contract == (
        "DirectIntegrationTarget.hot_utility_target",
        "DirectIntegrationTarget.cold_utility_target",
        "DirectIntegrationTarget.heat_recovery_target",
        "DirectIntegrationTarget.hot_pinch",
        "DirectIntegrationTarget.cold_pinch",
    )
    assert native.dt_cont_convention.startswith("OpenHENS PDM fallback")
    assert native.unit_conventions["pinch_temperatures"].endswith(
        "shifted_pinch_temperature in K"
    )


@pytest.mark.parametrize("pinch_location", PINCH_LOCATIONS)
def test_manual_stage_selection_matches_source_openhens_pdm_convention(
    pinch_location: Literal["above", "below"],
) -> None:
    problem = _load_problem("Four-stream-Yee-and-Grossmann-1990-1")

    source = _source_openhens_pinch_snapshot(
        problem,
        14.0,
        pinch_location=pinch_location,
        stage_selection=MANUAL_STAGE_SELECTION,
    )
    native = build_pinch_decomposition_snapshot(
        problem,
        14.0,
        pinch_location=pinch_location,
        stage_selection=MANUAL_STAGE_SELECTION,
    )

    _assert_snapshots_match(native, source)
    assert native.manual_stage_selection == MANUAL_STAGE_SELECTION


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("dTmin", REQUIRED_DTMIN_GRID)
@pytest.mark.parametrize("pinch_location", PINCH_LOCATIONS)
def test_pinch_decomposition_parity_is_independent_of_fixture_stream_row_order(
    case_id: str,
    dTmin: float,
    pinch_location: Literal["above", "below"],
) -> None:
    base_problem = _load_problem(case_id)
    reordered_problem = _load_problem(case_id, reordered=True)

    base_native = build_pinch_decomposition_snapshot(
        base_problem,
        dTmin,
        pinch_location=pinch_location,
        stage_selection=AUTO_STAGE_SELECTION,
    )
    reordered_native = build_pinch_decomposition_snapshot(
        reordered_problem,
        dTmin,
        pinch_location=pinch_location,
        stage_selection=AUTO_STAGE_SELECTION,
    )
    reordered_source = _source_openhens_pinch_snapshot(
        reordered_problem,
        dTmin,
        pinch_location=pinch_location,
        stage_selection=AUTO_STAGE_SELECTION,
    )

    assert reordered_native == base_native
    _assert_snapshots_match(reordered_native, reordered_source)


def test_required_case_matrix_has_no_hu_or_cu_threshold_rows_to_cover() -> None:
    hu_thresholds = []
    cu_thresholds = []
    for case_id in CASE_IDS:
        problem = _load_problem(case_id)
        for dTmin in REQUIRED_DTMIN_GRID:
            snapshot = build_pinch_decomposition_snapshot(
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


def _source_openhens_pinch_snapshot(
    problem: PinchProblem,
    dTmin: float,
    *,
    pinch_location: Literal["above", "below"],
    stage_selection: Literal["automated"] | tuple[int, int],
) -> SourcePinchSnapshot:
    PinchDecompModel = _source_pinch_decomp_model()
    arrays = problem_to_solver_arrays(problem, dTmin).arrays
    model = PinchDecompModel.__new__(PinchDecompModel)
    model.dTmin = float(dTmin)
    model.pinch_loc = pinch_location
    model.stage_selection = stage_selection
    model.tol = 1e-3

    for attr in (
        "T_h_in",
        "T_h_out",
        "T_h_cont",
        "htc_h",
        "f_h",
        "hot_names",
        "T_c_in",
        "T_c_out",
        "T_c_cont",
        "htc_c",
        "f_c",
        "cold_names",
        "T_hu_in",
        "T_hu_out",
        "htc_hu",
        "T_cu_in",
        "T_cu_out",
        "htc_cu",
    ):
        setattr(model, attr, arrays[attr].copy())

    model.T_h_in_OG = model.T_h_in.copy()
    model.T_h_out_OG = model.T_h_out.copy()
    model.T_c_in_OG = model.T_c_in.copy()
    model.T_c_out_OG = model.T_c_out.copy()

    PinchDecompModel.calculate_pinch(model)
    PinchDecompModel.set_preprocessing(model)

    heat_recovery_target = float(
        np.sum((arrays["T_h_in"] - arrays["T_h_out"]) * arrays["f_h"])
        - model.CU_target
    )
    return SourcePinchSnapshot(
        hot_utility_target=float(model.HU_target),
        cold_utility_target=float(model.CU_target),
        heat_recovery_target=heat_recovery_target,
        hot_pinch=float(model.process.hot_pinch),
        cold_pinch=float(model.process.cold_pinch),
        shifted_pinch_temperature=float(model.T_pinch),
        z_i_active=tuple(model.z_i_active),
        z_j_active=tuple(model.z_j_active),
        clipped_hot_supply_temperatures=tuple(model.T_h_in.tolist()),
        clipped_hot_target_temperatures=tuple(model.T_h_out.tolist()),
        clipped_cold_supply_temperatures=tuple(model.T_c_in.tolist()),
        clipped_cold_target_temperatures=tuple(model.T_c_out.tolist()),
        S=int(model.S),
        K=int(model.K),
    )


def _source_pinch_decomp_model():
    if not OPENHENS_ROOT.exists():
        raise AssertionError(
            "HENS-04 parity requires the source OpenHENS checkout at "
            f"{OPENHENS_ROOT}; clone or place it beside OpenPinch before "
            "running this required parity gate."
        )
    actual_commit = _source_openhens_commit()
    if actual_commit != OPENHENS_SOURCE_COMMIT:
        raise AssertionError(
            "HENS-04 parity requires source OpenHENS commit "
            f"{OPENHENS_SOURCE_COMMIT}, found {actual_commit} at {OPENHENS_ROOT}."
        )
    source_path = str(OPENHENS_ROOT)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        from openhens.classes.pinch_decomp_model import PinchDecompModel
    except ImportError as exc:
        raise AssertionError(
            "HENS-04 parity requires importable source OpenHENS "
            f"PinchDecompModel from {OPENHENS_ROOT}: {exc}"
        ) from exc
    return PinchDecompModel


def _source_openhens_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(OPENHENS_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "HENS-04 parity requires a git checkout for source OpenHENS at "
            f"{OPENHENS_ROOT}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _assert_snapshots_match(
    native: PinchDecompositionSnapshot,
    source: SourcePinchSnapshot,
) -> None:
    assert native.target.hot_utility_target == pytest.approx(
        source.hot_utility_target,
        abs=ABS_TOL,
    )
    assert native.target.cold_utility_target == pytest.approx(
        source.cold_utility_target,
        abs=ABS_TOL,
    )
    assert native.target.heat_recovery_target == pytest.approx(
        source.heat_recovery_target,
        abs=ABS_TOL,
    )
    assert native.target.hot_pinch == pytest.approx(source.hot_pinch, abs=ABS_TOL)
    assert native.target.cold_pinch == pytest.approx(source.cold_pinch, abs=ABS_TOL)
    assert native.target.shifted_pinch_temperature == pytest.approx(
        source.shifted_pinch_temperature,
        abs=ABS_TOL,
    )
    assert native.z_i_active == source.z_i_active
    assert native.z_j_active == source.z_j_active
    assert native.clipped_hot_supply_temperatures == pytest.approx(
        source.clipped_hot_supply_temperatures,
        abs=ABS_TOL,
    )
    assert native.clipped_hot_target_temperatures == pytest.approx(
        source.clipped_hot_target_temperatures,
        abs=ABS_TOL,
    )
    assert native.clipped_cold_supply_temperatures == pytest.approx(
        source.clipped_cold_supply_temperatures,
        abs=ABS_TOL,
    )
    assert native.clipped_cold_target_temperatures == pytest.approx(
        source.clipped_cold_target_temperatures,
        abs=ABS_TOL,
    )
    assert native.S == source.S
    assert native.K == source.K
