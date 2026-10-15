"""Process-pool fallback and objective-error handling in the optimiser."""

from __future__ import annotations

import logging
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest

import OpenPinch.optimisation.candidates as candidates
import OpenPinch.optimisation.execution as execution
from OpenPinch.optimisation.models import OptimisationOptions, OptimisationProblem
from OpenPinch.optimisation.service import run_multistart_minimisation

_CALLS: list[int] = []


class _InProcessPool:
    """Pool double that runs tasks in this process, like a healthy pool."""

    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable):
        return (fn(item) for item in list(iterable))


class _BreakingPool(_InProcessPool):
    """Pool double whose workers die once tasks are running."""

    def map(self, fn, iterable):
        def results():
            raise BrokenProcessPool("worker died")
            yield  # pragma: no cover

        return results()


class _StartupFailingPool(_InProcessPool):
    def __enter__(self):
        raise PermissionError("no semaphores")


def _record_run(run):
    _CALLS.append(run)
    return [np.asarray([float(run)])], [float(run)]


def _failing_run(run):
    _CALLS.append(run)
    raise RuntimeError("objective failed")


def _quadratic(x, *args):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - 0.3) ** 2))


@pytest.fixture(autouse=True)
def _reset_calls():
    _CALLS.clear()


def test_objective_runtime_error_propagates_without_serial_rerun():
    with pytest.raises(RuntimeError, match="objective failed"):
        execution._collect_candidates_in_parallel(
            _failing_run, n_runs=3, pool_executor_cls=_InProcessPool
        )

    # Only the pooled attempt ran; nothing was re-executed serially.
    assert _CALLS == [0]


@pytest.mark.parametrize("pool_cls", [_BreakingPool, _StartupFailingPool])
def test_pool_infrastructure_failure_falls_back_to_serial(pool_cls, caplog):
    with caplog.at_level(logging.WARNING, logger=execution.__name__):
        xs, fs = execution._collect_candidates_in_parallel(
            _record_run, n_runs=2, pool_executor_cls=pool_cls
        )

    assert fs.tolist() == [0.0, 1.0]
    assert _CALLS == [0, 1]
    assert "running tasks serially" in caplog.text


def test_unpicklable_task_runs_serially_without_starting_a_pool():
    class _ForbiddenPool(_InProcessPool):
        def __init__(self, max_workers):
            raise AssertionError("no pool should start for an unpicklable task")

    xs, fs = execution._collect_candidates_in_parallel(
        lambda run: ([np.asarray([float(run)])], [float(run)]),
        n_runs=2,
        pool_executor_cls=_ForbiddenPool,
    )

    assert fs.tolist() == [0.0, 1.0]


@pytest.mark.parametrize("n_runs", [1, 2])
def test_lambda_objective_runs_with_real_pools(n_runs):
    # Several kept basins send polishing to a pool even with one run.
    result = run_multistart_minimisation(
        OptimisationProblem(
            objective=lambda x: float(np.sum((np.asarray(x) - 0.3) ** 2)),
            bounds=((0.0, 1.0), (0.0, 1.0)),
        ),
        method="dual_annealing",
        options=OptimisationOptions(
            n_runs=n_runs, maxiter=5, maxfun=60, max_minima=3, cluster_tol=1e-6
        ),
    )

    assert result.best.objective == pytest.approx(0.0, abs=1e-6)


def _polish(monkeypatch, exc):
    monkeypatch.setattr(
        candidates,
        "minimize",
        lambda **kwargs: (_ for _ in ()).throw(exc),
    )
    return candidates._polish_single_candidate(
        idx=0,
        all_x=np.asarray([[0.2]], dtype=float),
        all_f=np.asarray([0.01], dtype=float),
        func=_quadratic,
        args=(),
        local_method="SLSQP",
        bounds_arg=np.asarray([[0.0, 1.0]], dtype=float),
        constraints=(),
    )


@pytest.mark.parametrize(
    "exc",
    [ValueError("bad step"), FloatingPointError("overflow"), np.linalg.LinAlgError()],
)
def test_polish_tolerates_numerical_solver_failures_with_warning(
    monkeypatch, caplog, exc
):
    with caplog.at_level(logging.WARNING, logger=candidates.__name__):
        x, f = _polish(monkeypatch, exc)

    np.testing.assert_allclose(x, [0.2])
    assert f == pytest.approx(0.01)
    assert "keeping it unpolished" in caplog.text


@pytest.mark.parametrize(
    "exc", [TypeError("bad call"), KeyError("x"), RuntimeError("bug")]
)
def test_polish_propagates_programming_errors(monkeypatch, exc):
    with pytest.raises(type(exc)):
        _polish(monkeypatch, exc)


class _ScopedObjective:
    """Task-scoped objective double that records how it was scoped."""

    def __init__(self, label="parent"):
        self.label = label
        self.calls = 0
        self.merged = None

    def __call__(self, x, *args):
        self.calls += 1
        return _quadratic(x)

    def for_task(self, index, count, *, phase="search"):
        return _ScopedObjective(f"{phase}:{index}/{count}")

    def task_report(self):
        return (self.label, self.calls)

    def merge_task_reports(self, reports):
        self.merged = list(reports)


def _run_with(run, *, func):
    func(np.asarray([0.1 * run]))
    return [np.asarray([0.1 * run])], [func(np.asarray([0.1 * run]))]


@pytest.mark.parametrize("pool_cls", [_InProcessPool, _BreakingPool])
def test_task_scoped_objective_gets_one_copy_per_run(pool_cls):
    objective = _ScopedObjective()

    xs, fs = execution._collect_candidates_in_parallel(
        _run_with, n_runs=3, pool_executor_cls=pool_cls, func=objective
    )

    assert xs.shape == (3, 1)
    assert objective.calls == 0
    assert objective.merged == [
        ("search:0/3", 2),
        ("search:1/3", 2),
        ("search:2/3", 2),
    ]
