"""Worker counts follow what the Python runtime supports (threads in browsers)."""

from __future__ import annotations

import sys

from OpenPinch.analysis.parallelism import supported_workers


def test_workers_fall_back_to_one_where_threads_cannot_start(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")

    assert supported_workers(4) == 1


def test_all_period_targets_run_serially_without_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import OpenPinch.application._problem.accessors.target as target_accessor
    from OpenPinch import PinchProblem

    class NoThreads(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(target_accessor, "ThreadPoolExecutor", NoThreads)
    payload = PinchProblem(
        "Four-stream-Yee-and-Grossmann-1990-1.json"
    ).to_problem_json()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["base", "turndown"]
    payload["options"]["PROBLEM_PERIOD_WEIGHTS"] = [0.7, 0.3]
    problem = PinchProblem(payload, project_name="Two periods")
    problem.target.all_periods.all_heat_integration()

    exergy = problem.target.all_periods.exergy(workers=2)

    assert set(exergy) == {"base", "turndown"}


def test_all_period_dt_min_runs_serially_without_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import OpenPinch.application.heat_recovery_dt_min as dt_min
    from OpenPinch import PinchProblem

    class NoThreads(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(dt_min, "ThreadPoolExecutor", NoThreads)
    payload = PinchProblem(
        "Four-stream-Yee-and-Grossmann-1990-1.json"
    ).to_problem_json()
    payload["options"]["PROBLEM_PERIOD_IDS"] = ["base", "turndown"]
    payload["options"]["PROBLEM_PERIOD_WEIGHTS"] = [0.7, 0.3]
    problem = PinchProblem(payload, project_name="Two periods")
    problem.target.all_periods.all_heat_integration()

    solved = problem.target.all_periods.heat_recovery_dt_min(
        heat_recovery=0.0, workers=2
    )

    assert set(solved) == {"base", "turndown"}


def test_evolution_candidates_run_serially_without_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import OpenPinch.analysis.heat_exchanger_networks.models._stagewise.evolution as evolution

    class NoThreads(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(sys, "platform", "emscripten")
    monkeypatch.setattr(evolution, "ThreadPoolExecutor", NoThreads)
    monkeypatch.setattr(
        evolution, "_solve_evolution_candidate", lambda owner, spec, print_output: spec
    )

    candidates = evolution._solve_evolution_candidates(
        object(), ["a", "b", "c"], print_output=False, max_parallel=4
    )

    assert [spec for spec, _ in candidates] == ["a", "b", "c"]
