"""Evolution searches of one run share the topologies they have solved."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.execution import (
    executor as executor_module,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
    _solve_built_task,
)
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
    evolution as stagewise_evolution,
)
from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)


class _Arrays:
    def __init__(self, values):
        self.values = values

    def to_json_dict(self):
        return {"arrays": {"T": self.values}}


class _Variable:
    fixed = False
    value = None

    def set_value(self, value, *, skip_validation):
        self.value = value


def _owner(*, dTmin: float = 0.1, arrays=(1.0, 2.0), stages: int = 1):
    model = StageWiseModel.__new__(StageWiseModel)
    model.name = "root"
    model.TAC = 100.0
    model.mSuccess = 1
    model.I = 1
    model.J = 1
    model.S = stages
    model.m = SimpleNamespace(_path=tempfile.mkdtemp(prefix="openpinch-test-"))
    model.framework = "ESM"
    model.stages = stages
    model.dTmin = dTmin
    model.min_dqda = 0.0
    model.minimisation_goal = "total cost"
    model.non_isothermal_model = True
    model.solver = "ipopt-pyomo"
    model.solver_options = None
    model.tol = 1e-3
    model.solver_arrays = _Arrays(list(arrays))
    return model


def _solved_dicts_seen(owner, store, **search):
    seen = []

    def candidate(**kwargs):
        seen.append(kwargs["solved"])
        return None

    owner.get_n_minus_one_evolution = candidate
    owner.get_n_plus_one_evolution = candidate
    stagewise_evolution.get_net_benefit_evolution(
        owner, False, 1, solved_topologies=store, **search
    )
    return seen


def test_searches_with_the_same_model_inputs_share_their_solved_topologies():
    store: dict = {}

    first = _solved_dicts_seen(_owner(), store)
    second = _solved_dicts_seen(_owner(), store)

    assert first[0] is second[0]


def test_searches_with_different_model_inputs_keep_apart():
    store: dict = {}

    base = _solved_dicts_seen(_owner(), store)[0]

    for other in (
        _owner(dTmin=0.2),
        _owner(arrays=(1.0, 3.0)),
        _owner(stages=2),
    ):
        assert _solved_dicts_seen(other, store)[0] is not base


def test_branched_search_reuses_a_topology_solved_by_another_search(monkeypatch):
    store: dict = {}
    solves = []

    class _Child:
        TAC = 90.0
        mSuccess = 1
        solver_run = SimpleNamespace(converged=True)

        def __init__(self, **kwargs):
            solves.append(kwargs["name"])
            self.name = kwargs["name"]
            self.z = [[[_Variable()]]]

        def set_initial_values_for_variables(self, parent):
            pass

        def optimise(self, *, print_output):
            pass

    monkeypatch.setattr(stagewise_evolution, "_stagewise_model_class", lambda: _Child)
    monkeypatch.setattr(
        stagewise_evolution, "_is_usable_evolution_candidate", lambda child: True
    )
    monkeypatch.setattr(
        stagewise_evolution, "_update_with_best_model", lambda owner, best: None
    )

    def specs(owner, frontier, *, unit, seen_signatures, **kwargs):
        if unit:
            return []
        return [
            stagewise_evolution._EvolutionCandidateSpec(
                kind="plus",
                unit=0,
                branch_index=0,
                rank=1,
                prev_case=owner,
                position=(0, 0, 0),
                z_allowed=[[[1]]],
                signature=((0, 0, 0),),
            )
        ]

    monkeypatch.setattr(stagewise_evolution, "_evolution_candidate_specs", specs)

    for _ in range(2):
        stagewise_evolution.get_net_benefit_evolution(
            _owner(), False, 2, n_ad_branches=2, solved_topologies=store
        )

    assert len(solves) == 1


def test_stagewise_model_passes_the_store_to_the_search(monkeypatch):
    received = []
    monkeypatch.setattr(
        stagewise_evolution,
        "get_net_benefit_evolution",
        lambda owner, *args, **kwargs: received.append(kwargs["solved_topologies"]),
    )
    store: dict = {}

    _owner().get_net_benefit_evolution(print_output=False, solved_topologies=store)

    assert received == [store]


def _evolution_task(index: int) -> HeatExchangerNetworkSynthesisTask:
    return HeatExchangerNetworkSynthesisTask(
        run_id=f"evolution-{index}",
        method="network_evolution_method",
        dt_cont_multiplier=1.0 + index,
    )


def _stub_stage(monkeypatch):
    received = []
    monkeypatch.setattr(
        LocalSynthesisExecutor, "_build_problem", lambda self, task, **kwargs: task
    )
    monkeypatch.setattr(
        LocalSynthesisExecutor,
        "_model_factories_for_task",
        lambda self, task, problem: None,
    )
    monkeypatch.setattr(
        executor_module, "_evolution_options", lambda problem, task, **kwargs: {}
    )

    def solve(worker_args):
        task = worker_args[0]
        received.append(worker_args[4].get("solved_topologies"))
        return (
            task,
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=task, status="success", objective_value=1.0
            ),
            None,
        )

    monkeypatch.setattr(executor_module, "_solve_built_task", solve)
    return received


def test_a_serial_stage_gives_every_task_one_store(monkeypatch):
    received = _stub_stage(monkeypatch)

    LocalSynthesisExecutor().execute(
        [_evolution_task(index) for index in range(3)],
        problem=None,
        parent_outcomes={},
        max_parallel=1,
    )

    assert len(received) == 3
    assert isinstance(received[0], dict)
    assert received[1] is received[0] and received[2] is received[0]


def test_each_serial_stage_starts_a_new_store(monkeypatch):
    received = _stub_stage(monkeypatch)
    executor = LocalSynthesisExecutor()

    for _ in range(2):
        executor.execute(
            [_evolution_task(0)], problem=None, parent_outcomes={}, max_parallel=1
        )

    assert received[0] is not received[1]


class _InProcessPool:
    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def map(self, func, inputs):
        return map(func, inputs)


def test_tasks_in_worker_processes_get_no_store(monkeypatch):
    # Which tasks share a worker is up to the pool, so sharing there would
    # make results depend on scheduling.
    received = _stub_stage(monkeypatch)

    LocalSynthesisExecutor(worker_pool_factory=_InProcessPool).execute(
        [_evolution_task(index) for index in range(2)],
        problem=None,
        parent_outcomes={},
        max_parallel=2,
    )

    assert received == [None, None]


def test_a_built_evolution_task_hands_the_store_to_its_solve():
    store: dict = {}

    class _Problem:
        def get_solution(self, **kwargs):
            self.kwargs = kwargs
            return None

    problem = _Problem()
    _solve_built_task(
        (
            _evolution_task(0),
            problem,
            False,
            None,
            {
                "n_ad_branches": 1,
                "n_rm_branches": 1,
                "max_parallel": 1,
                "no_improvement_patience": None,
                "beam_width": 4,
                "solved_topologies": store,
            },
        )
    )

    assert problem.kwargs["evolution_solved_topologies"] is store


def test_the_internal_problem_hands_the_store_to_evolution(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
        InternalHeatExchangerNetworkProblem,
    )

    received = []

    class _Case:
        mSuccess = 1

        def optimise(self, *, print_output):
            pass

        def get_net_benefit_evolution(self, **kwargs):
            received.append(kwargs["solved_topologies"])
            return self

    problem = InternalHeatExchangerNetworkProblem.__new__(
        InternalHeatExchangerNetworkProblem
    )
    problem.framework = "ESM"
    problem.heuristic_network_limit = 0
    case = _Case()

    def load_model(*, model_factories=None):
        problem.case = case

    monkeypatch.setattr(problem, "load_model", load_model, raising=False)
    store: dict = {}

    problem.get_solution(
        print_output=False, evolution=True, evolution_solved_topologies=store
    )

    assert received == [store]
