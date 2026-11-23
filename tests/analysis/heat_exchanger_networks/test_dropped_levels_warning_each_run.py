"""Every run that drops utility levels shows its warning.

Python's default filter shows a warning once for each message and place.
Two runs of one design give the same message, so the second run showed
nothing: on the review's heat pump level case, run twice in one process, one
line appeared in all. A user who reruns a design after a change would read
the silence as the levels now holding.
"""

from __future__ import annotations

import warnings

from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.results.assembly import (
    build_synthesis_result,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
    build_pinch_design_method_tasks,
)
from tests.analysis.heat_exchanger_networks.test_full_sequence import _small_problem


def test_each_run_shows_the_warning_once_for_each_task() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    tasks = build_pinch_design_method_tasks(settings)
    outcomes = FakeSynthesisExecutor().execute(
        tasks, problem=problem, parent_outcomes={}, max_parallel=1
    )
    # Three networks from each task, as a heuristic task returns many.
    dropped = 3 * [
        outcome.model_copy(update={"diagnostics": {"levels_dropped": "above"}})
        for outcome in outcomes
    ]
    shown: list[str] = []
    runs: list[list[str]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.showwarning = lambda message, *_a, **_k: shown.append(str(message))
        # Both runs from one line, as a rerun is: the filter keys on it.
        for _run in range(2):
            shown.clear()
            build_synthesis_result(settings, tasks, dropped)
            runs.append([text for text in shown if "utility levels" in text])

    first_run, second_run = runs
    assert len(first_run) == len(set(first_run)) == len(outcomes)
    assert second_run == first_run
