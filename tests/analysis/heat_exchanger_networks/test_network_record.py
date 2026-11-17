"""The live record of verified networks a HEN synthesis run admits."""

from __future__ import annotations

import sys

import pytest

from OpenPinch.analysis.heat_exchanger_networks.results.record import (
    HeatExchangerNetworkRecord,
    admit,
    recording,
    running,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _network(matches, tac: float | None) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=tuple(
            HeatExchanger(
                exchanger_id=f"E{n}",
                kind=HeatExchangerKind.RECOVERY,
                source_stream=hot,
                sink_stream=cold,
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Process,
                stage=stage,
                period_states=(
                    HeatExchangerPeriodState(period_id="0", period_idx=0, duty=100.0),
                ),
                area=1.0,
            )
            for n, (hot, cold, stage) in enumerate(matches, start=1)
        ),
        run_id="run",
        total_annual_cost=tac,
    )


def _outcome(
    task_id: str,
    matches,
    tac: float | None,
    *,
    method: str = "network_evolution_method",
    multiplier: float = 1.0,
    **fields,
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    task = HeatExchangerNetworkSynthesisTask(
        task_id=task_id,
        run_id="run",
        method=method,
        dt_cont_multiplier=multiplier,
    )
    network = _network(matches, tac)
    return HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        network=network,
        objective_value=tac,
        **({"status": "success", "solver_status": "optimal"} | fields),
    )


A = (("H1", "C1", 1),)
B = (("H1", "C1", 1), ("H2", "C1", 2))
C = (("H2", "C2", 1),)


def test_a_new_record_is_pending_and_empty():
    record = HeatExchangerNetworkRecord()

    assert record.version == 0
    assert record.status == "pending"
    assert record.snapshot() == {
        "version": 0,
        "status": "pending",
        "selected": None,
        "entries": [],
    }


def test_admitted_networks_are_ordered_by_total_annual_cost():
    record = HeatExchangerNetworkRecord()

    with recording(record):
        admit(_outcome("t2", B, 300.0))
        admit(_outcome("t1", A, 100.0))
        admit(_outcome("t3", C, 200.0))

    assert [entry.network.total_annual_cost for entry in record.outcomes()] == [
        100.0,
        200.0,
        300.0,
    ]
    assert record.version == 3


def test_one_entry_per_topology_keeps_the_cheaper():
    record = HeatExchangerNetworkRecord()

    with recording(record):
        admit(_outcome("t1", A, 150.0))
        admit(_outcome("t2", A, 100.0))
        admit(_outcome("t3", A, 120.0))

    (entry,) = record.outcomes()
    assert entry.task.task_id == "t2"
    # The costlier repeat changes nothing, so the version does not move.
    assert record.version == 2


def test_equal_costs_are_ordered_by_task_id_whatever_the_arrival_order():
    first = HeatExchangerNetworkRecord()
    second = HeatExchangerNetworkRecord()

    with recording(first):
        admit(_outcome("b", A, 100.0))
        admit(_outcome("a", A, 100.0))
        admit(_outcome("c", C, 100.0))
    with recording(second):
        admit(_outcome("c", C, 100.0))
        admit(_outcome("a", A, 100.0))
        admit(_outcome("b", A, 100.0))

    def content(record):
        # An entry's version says when it arrived, so it alone may differ.
        return [
            {key: value for key, value in entry.items() if key != "version"}
            for entry in record.snapshot()["entries"]
        ]

    assert [o.task.task_id for o in first.outcomes()] == ["a", "c"]
    assert content(first) == content(second)


def test_one_tasks_equal_cost_topologies_are_ordered_whatever_the_arrival_order():
    first = HeatExchangerNetworkRecord()
    second = HeatExchangerNetworkRecord()

    with recording(first):
        for matches in (A, B, C):
            admit(_outcome("evm", matches, 100.0))
    with recording(second):
        for matches in (C, B, A):
            admit(_outcome("evm", matches, 100.0))

    assert [entry["key"] for entry in first.snapshot()["entries"]] == [
        entry["key"] for entry in second.snapshot()["entries"]
    ]


def _doubled(outcome):
    """A stand-in fit: the same network at twice the cost."""
    network = outcome.network.model_copy(
        update={"total_annual_cost": 2 * outcome.network.total_annual_cost}
    )
    return outcome.model_copy(
        update={"network": network, "objective_value": network.total_annual_cost}
    )


def test_a_finalising_record_holds_each_network_in_its_final_form():
    basis = HeatExchangerNetworkRecord()
    fitted = HeatExchangerNetworkRecord()
    seen = []

    def fit(outcome):
        seen.append(outcome.task.task_id)
        return _doubled(outcome)

    with recording(fitted, finalise=fit), recording(basis):
        admit(_outcome("t1", A, 100.0))
        admit(_outcome("rejected", A, None))

    assert [o.network.total_annual_cost for o in basis.outcomes()] == [100.0]
    assert [o.network.total_annual_cost for o in fitted.outcomes()] == [200.0]
    # Only networks that pass the admission rule are fitted.
    assert seen == ["t1"]


def test_a_network_whose_fit_fails_or_is_infeasible_stays_out_of_that_record():
    basis = HeatExchangerNetworkRecord()
    fitted = HeatExchangerNetworkRecord()

    def fit(outcome):
        if outcome.task.task_id == "no-fit":
            return None
        if outcome.task.task_id == "infeasible-fit":
            return outcome.model_copy(
                update={"status": "failed", "error": "fit did not converge"}
            )
        return outcome

    with recording(fitted, finalise=fit), recording(basis):
        assert admit(_outcome("no-fit", A, 100.0)) is True
        admit(_outcome("infeasible-fit", B, 100.0))
        admit(_outcome("fits", C, 100.0))

    assert len(basis.outcomes()) == 3
    assert [o.task.task_id for o in fitted.outcomes()] == ["fits"]


def test_a_finaliser_runs_outside_the_records_it_feeds():
    from OpenPinch.analysis.heat_exchanger_networks.results.record import run_record

    record = HeatExchangerNetworkRecord()
    inside = {}

    def fit(outcome):
        # A fit is itself a solve; its own network must not enter the run.
        inside["admitted"] = admit(_outcome("fit", C, 1.0))
        inside["run_record"] = run_record(())
        return outcome

    with running(record, finalise=fit):
        admit(_outcome("t1", A, 100.0))

    assert inside["admitted"] is False
    assert inside["run_record"] is not record
    assert [o.task.task_id for o in record.outcomes()] == ["t1"]


@pytest.mark.parametrize(
    "outcome",
    [
        _outcome("failed", A, 100.0, status="failed", error="boom"),
        _outcome("unconverged", A, 100.0, diagnostics={"unconverged_termination": "x"}),
        _outcome("uncosted", A, None),
        HeatExchangerNetworkSynthesisTaskOutcome(
            task=_outcome("no-network", A, 1.0).task, status="success"
        ),
    ],
    ids=["failed", "unconverged", "uncosted", "no-network"],
)
def test_failed_unconverged_uncosted_or_missing_networks_are_rejected(outcome):
    record = HeatExchangerNetworkRecord()

    with recording(record):
        assert admit(outcome) is False

    assert record.outcomes() == ()
    assert record.version == 0


def test_an_infeasible_network_is_rejected():
    outcome = _outcome("infeasible", A, 100.0)
    network = outcome.network.model_copy(
        update={
            "exchangers": tuple(
                exchanger.model_copy(update={"area": 0.0})
                for exchanger in outcome.network.exchangers
            )
        }
    )
    record = HeatExchangerNetworkRecord()

    with recording(record):
        assert admit(outcome.model_copy(update={"network": network})) is False

    assert record.outcomes() == ()


def test_a_snapshot_lists_each_entry_as_json_data_in_rank_order():
    record = HeatExchangerNetworkRecord()
    with recording(record):
        admit(_outcome("tdm", B, 300.0, method="thermal_derivative_method"))
        admit(_outcome("pdm", A, 100.0, method="pinch_design_method", multiplier=0.75))

    snapshot = record.snapshot()

    assert snapshot["version"] == 2
    assert [entry["rank"] for entry in snapshot["entries"]] == [1, 2]
    first, second = snapshot["entries"]
    assert {key: first[key] for key in first if key not in ("key", "network")} == {
        "rank": 1,
        "method": "pinch_design_method",
        "task_id": "pdm",
        "dt_cont_multiplier": 0.75,
        "total_annual_cost": 100.0,
        "unit_count": 1,
        "version": 2,
    }
    assert second["unit_count"] == 2
    assert first["network"]["total_annual_cost"] == 100.0
    assert first["key"] != second["key"]
    import json

    assert json.loads(json.dumps(snapshot)) == snapshot


def test_a_snapshot_is_a_copy_the_reader_cannot_change():
    record = HeatExchangerNetworkRecord()
    with recording(record):
        admit(_outcome("t1", A, 100.0))

    record.snapshot()["entries"][0]["network"]["total_annual_cost"] = 0.0

    assert record.snapshot()["entries"][0]["network"]["total_annual_cost"] == 100.0


def test_changed_since_returns_only_entries_changed_after_that_version():
    record = HeatExchangerNetworkRecord()
    with recording(record):
        admit(_outcome("t1", A, 150.0))
        admit(_outcome("t2", C, 200.0))
        seen = record.version
        assert record.changed_since(seen) is None
        admit(_outcome("t3", A, 100.0))
        admit(_outcome("t4", B, 120.0))

    delta = record.changed_since(seen)

    assert delta["version"] == 4
    assert delta["status"] == "pending"
    assert [entry["task_id"] for entry in delta["entries"]] == ["t3", "t4"]
    keys = {entry["task_id"]: entry["key"] for entry in record.snapshot()["entries"]}
    assert delta["order"] == [keys["t3"], keys["t4"], keys["t2"]]
    assert record.changed_since(0)["entries"] == record.snapshot()["entries"]


def test_a_run_marks_the_record_running_then_complete_with_rank_one_selected():
    record = HeatExchangerNetworkRecord()

    with running(record):
        assert record.status == "running"
        admit(_outcome("t2", C, 200.0))
        admit(_outcome("t1", A, 100.0))

    snapshot = record.snapshot()
    assert snapshot["status"] == "complete"
    assert snapshot["selected"] == snapshot["entries"][0]["key"]
    assert snapshot["version"] == 4
    assert record.changed_since(3)["entries"] == []


def test_a_run_that_raises_marks_the_record_failed_and_keeps_its_entries():
    record = HeatExchangerNetworkRecord()

    with pytest.raises(RuntimeError, match="solver"), running(record):
        admit(_outcome("t1", A, 100.0))
        raise RuntimeError("solver crashed")

    assert record.status == "failed"
    assert record.snapshot()["selected"] is None
    assert len(record.outcomes()) == 1


def test_a_record_collects_one_run():
    record = HeatExchangerNetworkRecord()
    with running(record):
        pass

    with pytest.raises(ValueError, match="one run"), running(record):
        pass


def test_the_design_result_ranks_exactly_the_records_networks(monkeypatch):
    from tests.analysis.heat_exchanger_networks.test_design_workflow import (
        _public_example_problem,
        _use_fake_default_executor,
    )

    _use_fake_default_executor(monkeypatch)
    record = HeatExchangerNetworkRecord()

    view = _public_example_problem().design.heat_exchanger_network(record=record)

    assert record.outcomes()
    assert view.result.ranked_networks == record.outcomes()
    snapshot = record.snapshot()
    assert snapshot["status"] == "complete"
    assert snapshot["selected"] == snapshot["entries"][0]["key"]
    assert view.selected_network == record.outcomes()[0].network


needs_threads = pytest.mark.skipif(
    sys.platform == "emscripten", reason="Pyodide cannot start threads"
)


@needs_threads
def test_a_reader_thread_polls_networks_while_the_run_works(monkeypatch):
    import threading

    from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
        FakeSynthesisExecutor,
    )
    from tests.analysis.heat_exchanger_networks.test_design_workflow import (
        _public_example_problem,
        _use_fake_default_executor,
    )

    stages = []
    later_stage = threading.Event()
    looked = threading.Event()

    class PausingExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, **arguments):
            stages.append(tasks[0].method)
            if len(stages) == 2:
                # The first stage's networks are in; let the reader look.
                later_stage.set()
                assert looked.wait(10)
            return super().execute(tasks, **arguments)

    _use_fake_default_executor(monkeypatch)
    from OpenPinch.analysis.heat_exchanger_networks.targeting import (
        network_evolution_method,
        open_hens_method,
    )

    for module in (open_hens_method, network_evolution_method):
        monkeypatch.setattr(module, "LocalSynthesisExecutor", PausingExecutor)
    record = HeatExchangerNetworkRecord()
    mid_run = {}
    deltas = []

    def reader():
        assert later_stage.wait(10)
        mid_run.update(record.snapshot())
        looked.set()
        version = 0
        while True:
            delta = record.changed_since(version)
            if delta is not None:
                deltas.append(delta)
                version = delta["version"]
                if delta["status"] != "running":
                    return

    thread = threading.Thread(target=reader)
    thread.start()
    view = _public_example_problem().design.heat_exchanger_network(
        depth=2, record=record
    )
    thread.join(10)

    assert mid_run["status"] == "running"
    assert mid_run["entries"]
    versions = [delta["version"] for delta in deltas]
    assert versions == sorted(set(versions))
    assert deltas[-1]["status"] == "complete"
    merged = {}
    for delta in deltas:
        merged.update({entry["key"]: entry for entry in delta["entries"]})
    assert deltas[-1]["order"] == [
        entry["key"] for entry in record.snapshot()["entries"]
    ]
    assert set(merged) == set(deltas[-1]["order"])
    assert len(view.result.ranked_networks) == len(merged)


_EVOLUTION_TOPOLOGIES = {90.0: A, 80.0: B, 70.0: C}


class _EvolvingInternalProblem:
    """A solve whose evolution search improves twice before it ends."""

    def __init__(self, improvements=(90.0, 80.0)):
        self.improvements = improvements
        self.case = None

    def get_solution(self, **kwargs):
        from types import SimpleNamespace

        from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
            evolution,
        )

        for tac in self.improvements:
            evolution._report_improvement(SimpleNamespace(TAC=tac, mSuccess=1))
        self.case = SimpleNamespace(mSuccess=1, TAC=self.improvements[-1])
        return self.case

    def extract_candidate_network(self, candidate, *, run_id):
        return _network(_EVOLUTION_TOPOLOGIES[candidate.TAC], candidate.TAC)

    def extract_network(self, *, run_id):
        return self.extract_candidate_network(self.case, run_id=run_id)


def _evolution_task(task_id: str = "evm"):
    return _outcome(task_id, A, 1.0).task.model_copy(update={"stage_count": 2})


def test_an_evolution_search_admits_each_improvement_while_it_runs():
    from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
        _solve_built_task,
    )

    record = HeatExchangerNetworkRecord()
    task = _evolution_task()

    with recording(record):
        _task, outcome, _problem = _solve_built_task(
            (task, _EvolvingInternalProblem(), False, None)
        )

    assert outcome.status == "success"
    assert [o.network.total_annual_cost for o in record.outcomes()] == [80.0, 90.0]
    assert {o.task.task_id for o in record.outcomes()} == {"evm"}


class _WorkerPool:
    """Runs each task as a worker process would: in a context of its own.

    Tasks run in reverse order, so their networks reach the main process in a
    different order from a serial run.
    """

    def __init__(self, worker_count):
        self.worker_count = worker_count

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def map(self, function, items):
        import contextvars

        items = list(items)
        results = [contextvars.Context().run(function, item) for item in items[::-1]]
        return iter(results[::-1])


def _evolution_options(*args, max_parallel):
    return {
        "n_ad_branches": 2,
        "n_rm_branches": 2,
        "max_parallel": max_parallel,
        "no_improvement_patience": None,
        "beam_width": 4,
    }


def _evolution_tasks_through_the_executor(
    monkeypatch,
    *,
    max_parallel,
    pool=None,
    problems=None,
    record=None,
    handoff_queue_factory=None,
):
    import queue

    from OpenPinch.analysis.heat_exchanger_networks.execution import executor

    tasks = (
        _evolution_task("evm-a"),
        _evolution_task("evm-b"),
    )
    if problems is None:
        problems = {
            "evm-a": _EvolvingInternalProblem((90.0, 80.0)),
            "evm-b": _EvolvingInternalProblem((70.0,)),
        }
    local = executor.LocalSynthesisExecutor(
        worker_pool_factory=pool or _WorkerPool,
        handoff_queue_factory=handoff_queue_factory or queue.Queue,
    )
    monkeypatch.setattr(
        local,
        "_build_problem",
        lambda task, **kwargs: problems[task.task_id],
    )
    monkeypatch.setattr(local, "_model_factories_for_task", lambda *args: None)
    monkeypatch.setattr(executor, "_evolution_options", _evolution_options)
    monkeypatch.setattr(executor, "supported_workers", lambda requested: requested)
    if record is None:
        record = HeatExchangerNetworkRecord()
    with recording(record):
        local.execute(
            tasks,
            problem=None,
            parent_outcomes={"force-local-build": None},
            max_parallel=max_parallel,
        )
    return record


@needs_threads
def test_workers_hand_their_improvements_to_the_main_process(monkeypatch):
    pooled = _evolution_tasks_through_the_executor(monkeypatch, max_parallel=2)
    serial = _evolution_tasks_through_the_executor(monkeypatch, max_parallel=1)

    assert [o.network.total_annual_cost for o in pooled.outcomes()] == [
        70.0,
        80.0,
        90.0,
    ]
    assert _content(pooled) == _content(serial)


class _LookingInternalProblem(_EvolvingInternalProblem):
    """A solve that notes which tasks' networks the record holds as it starts."""

    def __init__(self, tac, record, seen):
        super().__init__((tac,))
        self.record = record
        self.seen = seen

    def get_solution(self, **kwargs):
        from types import SimpleNamespace

        self.seen.append([o.task.task_id for o in self.record.outcomes()])
        self.case = SimpleNamespace(mSuccess=1, TAC=self.improvements[-1])
        return self.case


def test_each_task_network_enters_the_record_before_the_next_task_solves(
    monkeypatch,
):
    record = HeatExchangerNetworkRecord()
    seen = []
    problems = {
        "evm-a": _LookingInternalProblem(90.0, record, seen),
        "evm-b": _LookingInternalProblem(70.0, record, seen),
    }

    _evolution_tasks_through_the_executor(
        monkeypatch, max_parallel=1, problems=problems, record=record
    )

    assert seen == [[], ["evm-a"]]
    assert [o.task.task_id for o in record.outcomes()] == ["evm-b", "evm-a"]


def test_a_stage_offers_each_network_to_a_finalising_record_once(monkeypatch):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.targeting._stage import (
        run_stage,
    )

    finalised = []

    def finalise(outcome):
        finalised.append(outcome.task.task_id)
        return outcome

    record = HeatExchangerNetworkRecord()
    problems = {
        "evm-a": _LookingInternalProblem(90.0, record, []),
        "evm-b": _LookingInternalProblem(70.0, record, []),
    }
    from OpenPinch.analysis.heat_exchanger_networks.execution import executor

    local = executor.LocalSynthesisExecutor()
    monkeypatch.setattr(
        local, "_build_problem", lambda task, **kwargs: problems[task.task_id]
    )
    monkeypatch.setattr(local, "_model_factories_for_task", lambda *args: None)
    monkeypatch.setattr(executor, "_evolution_options", _evolution_options)
    with recording(record, finalise=finalise):
        run_stage(
            (_evolution_task("evm-a"), _evolution_task("evm-b")),
            problem=None,
            settings=SimpleNamespace(max_parallel=1),
            executor=local,
            default_executor=None,
            parent_outcomes={"force-local-build": None},
        )

    assert sorted(finalised) == ["evm-a", "evm-b"]
    assert len(record.outcomes()) == 2


def test_a_pool_starts_no_hand_off_when_nothing_records(monkeypatch):
    def no_queue():
        raise AssertionError("a hand-off queue started with no record")

    from OpenPinch.analysis.heat_exchanger_networks.execution import executor

    tasks = (_evolution_task("evm-a"), _evolution_task("evm-b"))
    problems = {
        "evm-a": _EvolvingInternalProblem((90.0,)),
        "evm-b": _EvolvingInternalProblem((70.0,)),
    }
    local = executor.LocalSynthesisExecutor(
        worker_pool_factory=_WorkerPool, handoff_queue_factory=no_queue
    )
    monkeypatch.setattr(
        local, "_build_problem", lambda task, **kwargs: problems[task.task_id]
    )
    monkeypatch.setattr(local, "_model_factories_for_task", lambda *args: None)
    monkeypatch.setattr(executor, "_evolution_options", _evolution_options)
    monkeypatch.setattr(executor, "supported_workers", lambda requested: requested)

    outcomes = local.execute(
        tasks,
        problem=None,
        parent_outcomes={"force-local-build": None},
        max_parallel=2,
    )

    assert [outcome.status for outcome in outcomes] == ["success", "success"]


def _content(record):
    """The snapshot entries without their versions, which record arrival."""
    return [
        {key: value for key, value in entry.items() if key != "version"}
        for entry in record.snapshot()["entries"]
    ]


def test_admitting_outside_a_recording_does_nothing():
    assert admit(_outcome("t1", A, 100.0)) is False
