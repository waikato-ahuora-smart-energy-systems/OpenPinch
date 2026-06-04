"""Regression checks for packaged sample cases."""

from __future__ import annotations

from pathlib import Path

from OpenPinch import PinchProblem
from OpenPinch.resources import copy_sample_case, list_sample_cases


def test_packaged_multistate_cases_are_listed():
    sample_cases = list_sample_cases()

    assert "crude_preheat_train_multistate.json" in sample_cases
    assert "zonal_site_multistate.json" in sample_cases


def test_crude_multistate_case_validates_and_changes_across_states(tmp_path: Path):
    case_path = copy_sample_case(
        "crude_preheat_train_multistate.json",
        tmp_path / "crude_preheat_train_multistate.json",
    )
    problem = PinchProblem(case_path, project_name="crude_multistate")

    assert problem.state_ids == {"turndown": 0, "base": 1, "peak": 2}

    results = problem.target_all_states(parallel=False)
    assert list(results) == ["turndown", "base", "peak"]

    direct_targets = [
        next(target for target in output.targets if target.name.endswith("/Direct Integration"))
        for output in results.values()
    ]

    assert len(direct_targets) == 3
    assert len({round(target.Qh, 6) for target in direct_targets}) > 1
    assert len({round(target.temp_pinch.hot_temp, 6) for target in direct_targets}) > 1


def test_zonal_site_multistate_case_validates_and_changes_across_states(
    tmp_path: Path,
):
    case_path = copy_sample_case(
        "zonal_site_multistate.json",
        tmp_path / "zonal_site_multistate.json",
    )
    problem = PinchProblem(case_path, project_name="seasonal_site")

    assert problem.state_ids == {"summer": 0, "shoulder": 1, "winter": 2}

    results = problem.target_all_states(parallel=False)
    assert list(results) == ["summer", "shoulder", "winter"]

    total_site_targets = [
        next(target for target in output.targets if target.name.endswith("/Total Site Target"))
        for output in results.values()
    ]

    assert len(total_site_targets) == 3
    assert len({round(target.Qc, 6) for target in total_site_targets}) > 1
    assert len({round(target.Qr, 6) for target in total_site_targets}) > 1

    problem.target(state_id="winter")
    focused = problem.target.indirect_heat_integration(state_id="winter")

    assert focused.state_id == "winter"
    assert round(float(focused.cold_utility_target), 6) == round(
        total_site_targets[-1].Qc,
        6,
    )
