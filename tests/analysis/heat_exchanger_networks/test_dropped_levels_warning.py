"""A pinch design side that drops its utility levels says so.

A side that cannot hold its utility levels at their targets is designed
without them, choosing levels by price. That was a log line only, so a
design could hold the levels on one side and not the other (an evaporator
kept and its condenser replaced by steam) with no warning to the caller.
"""

from __future__ import annotations

import pytest

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


def test_a_side_that_dropped_its_levels_warns_once_per_task() -> None:
    """The warning is raised where the result is assembled, so it reaches
    the caller from a worker process too."""
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    tasks = build_pinch_design_method_tasks(settings)
    outcomes = FakeSynthesisExecutor().execute(
        tasks, problem=problem, parent_outcomes={}, max_parallel=1
    )
    dropped = [
        outcome.model_copy(update={"diagnostics": {"levels_dropped": "above"}})
        for outcome in outcomes
    ]

    with pytest.warns(UserWarning, match="utility levels") as caught:
        build_synthesis_result(settings, tasks, dropped)

    messages = [
        str(item.message) for item in caught if "utility levels" in str(item.message)
    ]
    assert len(messages) == len(outcomes)
    # The message names where: "above", "below" or "above and below" the
    # pinch (reworded from "the above pinch side", which read badly for two).
    assert all("no design above the pinch" in message for message in messages)


def test_a_solved_task_records_the_sides_that_dropped_their_levels() -> None:
    """The model notes each side it designed without levels; the outcome
    carries them to where the result is assembled."""
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        _solved_outcome,
    )

    problem = _small_problem()
    task = build_pinch_design_method_tasks(workflow_settings_from_problem(problem))[0]
    (solved,) = FakeSynthesisExecutor().execute(
        (task,), problem=problem, parent_outcomes={}, max_parallel=1
    )

    def outcome(sides):
        internal = SimpleNamespace(
            extract_network=lambda run_id: solved.network,
            case=SimpleNamespace(solver_run=None),
            _sides_without_levels=sides,
        )
        return _solved_outcome(task, internal, SimpleNamespace())

    assert outcome({"below", "above"}).diagnostics == {
        "levels_dropped": "above and below"
    }
    assert outcome({"below"}).diagnostics == {"levels_dropped": "below"}
    assert outcome(set()).diagnostics == {}
