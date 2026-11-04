"""Process-pool fallback and objective-error handling in the optimiser."""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import threading
import time
import types
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial

import numpy as np
import pytest
from scipy.optimize import dual_annealing

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


@pytest.fixture
def native_platform(monkeypatch):
    """Exercise the pool path even where the suite runs under Pyodide.

    Pyodide reports emscripten and a single CPU, and its test runner's main
    script may have no main guard; each would send the tasks down the serial
    path. The main-guard tests set their own start method afterwards.
    """
    monkeypatch.setattr(execution.sys, "platform", "linux")
    monkeypatch.setattr(execution.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(execution.multiprocessing, "get_start_method", lambda: "fork")


def test_objective_runtime_error_propagates_without_serial_rerun():
    with pytest.raises(RuntimeError, match="objective failed"):
        execution._collect_candidates_in_parallel(
            _failing_run, n_runs=3, pool_executor_cls=_InProcessPool
        )

    # Only the pooled attempt ran; nothing was re-executed serially.
    assert _CALLS == [0]


@pytest.mark.parametrize("pool_cls", [_BreakingPool, _StartupFailingPool])
def test_pool_infrastructure_failure_falls_back_to_serial(
    native_platform, pool_cls, caplog
):
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


def test_polishing_runs_when_the_cpu_count_is_unknown(monkeypatch):
    # os.cpu_count() returns None where the count cannot be determined.
    monkeypatch.setattr(candidates.os, "cpu_count", lambda: None)

    result = run_multistart_minimisation(
        OptimisationProblem(
            objective=lambda x: float(np.sum((np.asarray(x) - 0.3) ** 2)),
            bounds=((0.0, 1.0), (0.0, 1.0)),
        ),
        method="dual_annealing",
        options=OptimisationOptions(
            n_runs=1, maxiter=5, maxfun=60, max_minima=3, cluster_tol=1e-6
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


class _ForbiddenPool(_InProcessPool):
    def __init__(self, max_workers):
        raise AssertionError("no pool should start here")


def test_emscripten_runs_tasks_serially_without_a_pool(monkeypatch, caplog):
    monkeypatch.setattr(execution.sys, "platform", "emscripten")

    with caplog.at_level(logging.WARNING, logger=execution.__name__):
        xs, fs = execution._collect_candidates_in_parallel(
            _record_run, n_runs=2, pool_executor_cls=_ForbiddenPool
        )

    assert fs.tolist() == [0.0, 1.0]
    assert caplog.text == ""


def _main_script(monkeypatch, tmp_path, source):
    script = tmp_path / "user_script.py"
    script.write_text(source)
    main = types.ModuleType("__main__")
    main.__file__ = str(script)
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.setattr(execution.multiprocessing, "get_start_method", lambda: "spawn")


def test_unguarded_main_module_runs_serially_with_a_warning(
    native_platform, monkeypatch, tmp_path, caplog
):
    _main_script(monkeypatch, tmp_path, "import OpenPinch\nrun_placement()\n")

    with caplog.at_level(logging.WARNING, logger=execution.__name__):
        xs, fs = execution._collect_candidates_in_parallel(
            _record_run, n_runs=2, pool_executor_cls=_ForbiddenPool
        )

    assert fs.tolist() == [0.0, 1.0]
    assert 'if __name__ == "__main__":' in caplog.text


def test_guarded_main_module_uses_worker_processes(
    native_platform, monkeypatch, tmp_path
):
    _main_script(
        monkeypatch,
        tmp_path,
        'import OpenPinch\n\nif __name__ == "__main__":\n    run_placement()\n',
    )
    assert execution._pool_worker_count(2, 2) == 2


def test_search_pool_is_capped_at_the_cpu_count(native_platform, monkeypatch):
    monkeypatch.setattr(execution.os, "cpu_count", lambda: 2)
    started = []

    class _RecordingPool(_InProcessPool):
        def __init__(self, max_workers):
            started.append(max_workers)

    xs, fs = execution._collect_candidates_in_parallel(
        _record_run, n_runs=5, pool_executor_cls=_RecordingPool
    )

    assert started == [2]
    assert fs.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


needs_threads = pytest.mark.skipif(
    sys.platform == "emscripten", reason="Pyodide cannot start threads"
)


@pytest.fixture
def importing_workers(native_platform, monkeypatch):
    """Use a start method whose workers import OpenPinch before any task."""
    monkeypatch.setattr(execution.multiprocessing, "get_start_method", lambda: "spawn")
    monkeypatch.setattr(execution, "_main_module_reruns_in_workers", lambda: False)


@needs_threads
def test_quick_batch_never_starts_a_pool_that_must_import_openpinch(importing_workers):
    xs, fs = execution._collect_candidates_in_parallel(
        _record_run, n_runs=3, pool_executor_cls=_ForbiddenPool
    )

    assert fs.tolist() == [0.0, 1.0, 2.0]
    assert _CALLS == [0, 1, 2]


_POOL_RAN_A_TASK = threading.Event()


def _wait_for_the_pool(run):
    """Parent's task: finish only once another task has run in the pool."""
    if run == 0:
        assert _POOL_RAN_A_TASK.wait(timeout=30)
    else:
        _POOL_RAN_A_TASK.set()
    return [np.asarray([float(run)])], [float(run)]


class _ThreadWorkers(ThreadPoolExecutor):
    """Pool double with real concurrency and instantly ready workers."""


@needs_threads
def test_slow_batch_shares_its_remaining_tasks_with_a_pool(
    importing_workers, monkeypatch
):
    _POOL_RAN_A_TASK.clear()
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)

    xs, fs = execution._collect_candidates_in_parallel(
        _wait_for_the_pool, n_runs=3, pool_executor_cls=_ThreadWorkers
    )

    assert fs.tolist() == [0.0, 1.0, 2.0]


class _StillImportingWorkers(ThreadPoolExecutor):
    """Pool double whose workers never finish importing OpenPinch."""

    instances: list = []

    def __init__(self, max_workers):
        super().__init__(max_workers=max_workers)
        self.submitted = []
        self.instances.append(self)

    def submit(self, fn, *args, **kwargs):
        self.submitted.append(fn)
        if fn is execution._worker_ready:
            return Future()
        return super().submit(fn, *args, **kwargs)


def _run_while_the_pool_starts(run):
    # The parent gets past its first task only once the pool has been started,
    # so the remaining tasks are still unclaimed when it starts.
    if run == 0:
        deadline = time.monotonic() + 30
        while not _StillImportingWorkers.instances:
            assert time.monotonic() < deadline
            time.sleep(0.001)
    return _record_run(run)


@needs_threads
def test_parent_keeps_tasks_from_workers_that_are_still_importing(
    importing_workers, monkeypatch
):
    _StillImportingWorkers.instances = []
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)

    xs, fs = execution._collect_candidates_in_parallel(
        _run_while_the_pool_starts,
        n_runs=3,
        pool_executor_cls=_StillImportingWorkers,
    )

    assert fs.tolist() == [0.0, 1.0, 2.0]
    assert _CALLS == [0, 1, 2]
    assert (
        _StillImportingWorkers.instances[0].submitted == [execution._worker_ready] * 2
    )


class _BrokenWorkers(ThreadPoolExecutor):
    """Pool double whose workers die while running their first tasks."""

    def submit(self, fn, *args, **kwargs):
        future = Future()
        if fn is execution._worker_ready:
            future.set_result(None)
        else:
            future.set_exception(BrokenProcessPool("worker died"))
        return future


def _pause_for_the_pool(run):
    if run == 0:
        time.sleep(0.05)
    return _record_run(run)


@needs_threads
def test_tasks_of_a_broken_late_pool_rerun_in_the_parent(
    importing_workers, monkeypatch, caplog
):
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)

    with caplog.at_level(logging.WARNING, logger=execution.__name__):
        xs, fs = execution._collect_candidates_in_parallel(
            _pause_for_the_pool, n_runs=4, pool_executor_cls=_BrokenWorkers
        )

    assert fs.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert sorted(_CALLS) == [0, 1, 2, 3]
    assert "running tasks serially" in caplog.text


class _NoSemaphores:
    def __init__(self, max_workers):
        raise PermissionError("no semaphores")


@needs_threads
def test_late_pool_that_cannot_start_leaves_tasks_to_the_parent(
    importing_workers, monkeypatch, caplog
):
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)

    with caplog.at_level(logging.WARNING, logger=execution.__name__):
        xs, fs = execution._collect_candidates_in_parallel(
            _pause_for_the_pool, n_runs=3, pool_executor_cls=_NoSemaphores
        )

    assert fs.tolist() == [0.0, 1.0, 2.0]
    assert _CALLS == [0, 1, 2]
    assert "running tasks serially" in caplog.text


class _CountingObjective:
    """Objective with state its tasks change, which is not task scoped."""

    def __init__(self):
        self.count = 0

    def __call__(self, x):
        return float(self.count)


def _change_state_then_report_it(run, *, func):
    if run == 0:
        func.count = 99
        assert _POOL_RAN_A_TASK.wait(timeout=30)
        return [np.asarray([0.0])], [0.0]
    seen = float(func.count)
    _POOL_RAN_A_TASK.set()
    return [np.asarray([seen])], [seen]


@needs_threads
def test_workers_see_the_objective_as_it_was_when_the_batch_began(
    importing_workers, monkeypatch
):
    _POOL_RAN_A_TASK.clear()
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)

    xs, fs = execution._collect_candidates_in_parallel(
        _change_state_then_report_it,
        n_runs=2,
        pool_executor_cls=_ThreadWorkers,
        func=_CountingObjective(),
    )

    assert fs.tolist() == [0.0, 0.0]


def _record_item(item):
    _CALLS.append(item[0])
    return item[0]


@needs_threads
def test_later_unpicklable_item_runs_the_batch_serially(importing_workers):
    items = [(0, "picklable"), (1, lambda: None)]

    results = execution._map_in_process_pool(
        _record_item, items, max_workers=2, pool_executor_cls=_ForbiddenPool
    )

    assert results == [0, 1]


def _search_once_a_worker_has_run(run, *, parent_pid, flag):
    """One seeded search; the parent's task waits so a worker must take a task."""
    if os.getpid() == parent_pid:
        deadline = time.monotonic() + 120
        while not flag.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
    else:
        flag.touch()
    found = dual_annealing(
        _quadratic, bounds=[(0.0, 1.0)] * 2, seed=run, maxiter=5, no_local_search=True
    )
    return found.x.tolist(), float(found.fun), os.getpid()


@pytest.mark.skipif(sys.platform == "emscripten", reason="Pyodide has no processes")
@pytest.mark.parametrize(
    "method",
    [
        method
        for method in ("spawn", "forkserver")
        if method in multiprocessing.get_all_start_methods()
    ],
)
def test_real_late_pool_results_match_serial_results(method, monkeypatch, tmp_path):
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)
    flag = tmp_path / "worker-ran"
    task = partial(_search_once_a_worker_has_run, parent_pid=os.getpid(), flag=flag)

    pooled = execution._map_with_late_pool(
        task,
        list(range(4)),
        workers=3,
        pool_executor_cls=partial(
            ProcessPoolExecutor, mp_context=multiprocessing.get_context(method)
        ),
        broken_pool_exc=BrokenProcessPool,
    )
    serial = [task(run) for run in range(4)]

    assert {pid for *_, pid in pooled} - {os.getpid()}
    for (pooled_x, pooled_f, _), (serial_x, serial_f, _) in zip(pooled, serial):
        np.testing.assert_array_equal(pooled_x, serial_x)
        np.testing.assert_array_equal(pooled_f, serial_f)


def _raise_in_worker(run):
    if run == 0:
        assert _POOL_RAN_A_TASK.wait(timeout=30)
        return run
    _POOL_RAN_A_TASK.set()
    raise OSError("objective failed in a worker")


@needs_threads
def test_objective_error_in_a_worker_propagates_unchanged(
    importing_workers, monkeypatch
):
    _POOL_RAN_A_TASK.clear()
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)

    with pytest.raises(OSError, match="objective failed in a worker"):
        execution._map_in_process_pool(
            _raise_in_worker,
            [0, 1],
            max_workers=2,
            pool_executor_cls=_ThreadWorkers,
        )


@needs_threads
def test_objective_error_in_the_parent_propagates_unchanged(importing_workers):
    with pytest.raises(RuntimeError, match="objective failed"):
        execution._collect_candidates_in_parallel(
            _failing_run, n_runs=2, pool_executor_cls=_ForbiddenPool
        )


class _SlowToStartWorkers(ThreadPoolExecutor):
    """Pool double whose first worker start blocks, as a new forkserver's does."""

    instances: list = []

    def __init__(self, max_workers):
        super().__init__(max_workers=max_workers)
        self.batch_returned = threading.Event()
        self.returned_during_start = None
        self.shut_down = threading.Event()
        self.instances.append(self)

    def submit(self, fn, *args, **kwargs):
        self.returned_during_start = self.batch_returned.wait(timeout=5)
        return Future()

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shut_down.set()
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


def _run_while_the_pool_blocks(run):
    if run == 0:
        deadline = time.monotonic() + 30
        while not _SlowToStartWorkers.instances:
            assert time.monotonic() < deadline
            time.sleep(0.001)
    return _record_run(run)


@needs_threads
def test_batch_returns_while_the_first_worker_is_still_starting(
    importing_workers, monkeypatch
):
    _SlowToStartWorkers.instances = []
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)

    xs, fs = execution._collect_candidates_in_parallel(
        _run_while_the_pool_blocks,
        n_runs=2,
        pool_executor_cls=_SlowToStartWorkers,
    )
    pool = _SlowToStartWorkers.instances[0]
    pool.batch_returned.set()

    assert fs.tolist() == [0.0, 1.0]
    assert pool.shut_down.wait(timeout=30)
    assert pool.returned_during_start is True


def _die_in_a_worker(run, *, parent_pid, flag):
    if os.getpid() != parent_pid:
        flag.touch()
        os._exit(1)
    deadline = time.monotonic() + 120
    while not flag.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return run * 2


@pytest.mark.skipif(sys.platform == "emscripten", reason="Pyodide has no processes")
def test_tasks_of_a_real_late_pool_whose_worker_dies_rerun_here(monkeypatch, tmp_path):
    monkeypatch.setattr(execution, "_POOL_PATIENCE_SECONDS", 0.0)
    task = partial(
        _die_in_a_worker, parent_pid=os.getpid(), flag=tmp_path / "worker-died"
    )

    results = execution._map_with_late_pool(
        task,
        list(range(4)),
        workers=3,
        pool_executor_cls=partial(
            ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")
        ),
        broken_pool_exc=BrokenProcessPool,
    )

    assert results == [0, 2, 4, 6]
