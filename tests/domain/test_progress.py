"""Progress reports from long-running analyses."""

import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest

from OpenPinch.domain import progress


def test_emit_outside_reporting_calls_nothing():
    progress.emit(analysis="hen_synthesis", stage="pinch_design_method")


def test_reporting_passes_each_event_to_the_callback_with_elapsed_time():
    events = []
    times = iter([10.0, 12.5])

    with progress.reporting(events.append, clock=lambda: next(times)):
        progress.emit(
            analysis="utility_placement",
            stage="search",
            done=40,
            total=4000,
            unit="evaluations",
            best_objective=1.5e6,
            objective="total annual cost",
        )

    assert events == [
        progress.Progress(
            analysis="utility_placement",
            stage="search",
            message="Search: 40 of 4000 evaluations, best total annual cost 1.5e+06",
            done=40,
            total=4000,
            unit="evaluations",
            best_objective=1.5e6,
            objective="total annual cost",
            elapsed=2.5,
            scope=None,
        )
    ]


def test_reporting_drops_events_within_the_interval_but_keeps_new_and_finished_stages():
    events = []
    # Installed at 0; each emit reads the clock once.
    times = iter([0.0, 0.0, 0.1, 0.2, 0.3, 0.35, 0.4])

    with progress.reporting(
        events.append, min_interval=0.25, clock=lambda: next(times)
    ):
        progress.emit(analysis="a", stage="search", done=1, total=10)  # first
        progress.emit(analysis="a", stage="search", done=2, total=10)  # dropped
        progress.emit(analysis="a", stage="polish", done=0, total=2)  # new stage
        progress.emit(analysis="a", stage="search", done=3, total=10)  # 0.3 s on
        progress.emit(analysis="a", stage="search", done=4, total=10)  # dropped
        progress.emit(analysis="a", stage="search", done=10, total=10)  # finished

    assert [(event.stage, event.done) for event in events] == [
        ("search", 1),
        ("polish", 0),
        ("search", 3),
        ("search", 10),
    ]


def test_progress_json_is_strict_json():
    events = []
    with progress.reporting(events.append):
        progress.emit(analysis="a", stage="search", best_objective=float("nan"))

    assert events[0].to_json() == {
        "analysis": "a",
        "stage": "search",
        "message": "Search",
        "done": None,
        "total": None,
        "unit": None,
        "best_objective": None,
        "objective": None,
        "elapsed": events[0].elapsed,
        "scope": None,
    }


def test_a_callback_exception_propagates_and_uninstalls_the_reporter():
    class Stop(Exception):
        pass

    def cancel(event):
        raise Stop

    with pytest.raises(Stop), progress.reporting(cancel):
        progress.emit(analysis="a", stage="search")

    progress.emit(analysis="a", stage="search")


def test_events_carry_the_scope_they_run_in_also_in_copied_contexts():
    events = []

    def solve(period):
        with progress.scope(period):
            progress.emit(analysis="a", stage="search", done=1, total=1)

    with progress.reporting(events.append):
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [
                pool.submit(contextvars.copy_context().run, solve, period)
                for period in ("P1", "P2")
            ]:
                future.result()

    assert sorted(event.scope for event in events) == ["P1", "P2"]


def test_reports_from_a_forked_worker_process_are_not_passed_on(monkeypatch):
    events = []

    with progress.reporting(events.append):
        monkeypatch.setattr(progress.os, "getpid", lambda: -1)
        progress.emit(analysis="a", stage="search")

    assert events == []
