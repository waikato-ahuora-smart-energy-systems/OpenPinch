"""A task holds the levels the network it starts from holds, no others.

Two gaps in holding a single process's levels through evolution:

- a task seeded from a network (the standalone thermal derivative and
  network evolution methods) has no parent task, so it had no targets to
  hold and ``HENS_EVOLUTION_LEVEL_HOLDS: "fixed"`` did nothing;
- a task built on a pinch design that dropped the levels on one side of the
  pinch was held to targets its parent did not meet.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from tests.analysis.heat_exchanger_networks.test_retrofit_loops import _network
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    _heating_problem,
)

# A parent that meets its cold utility target and misses its hot one.
DUTIES = {"H1->CU": 90.0, "H2->CU": 10.0, "HU->C1": 200.0, "HU->C2": 30.0}
TARGETS = {"CU": [100.0], "HU": [250.0]}


def _parent_network():
    return _network(DUTIES).model_copy(
        update={"source_metadata": {"utility_level_targets": TARGETS}}
    )


def _problem(holds: str):
    problem = _heating_problem()
    problem.update_options({"HENS_EVOLUTION_LEVEL_HOLDS": holds})
    return problem


def test_a_child_task_holds_only_the_levels_its_parent_network_holds():
    parent = HeatExchangerNetworkSynthesisTask(
        run_id="method-test", method="pinch_design_method", dt_cont_multiplier=1.0
    )
    child = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="network_evolution_method",
        dt_cont_multiplier=1.0,
        stage_count=1,
        parent_task_id=parent.task_id,
    )
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=parent, status="success", network=_parent_network()
    )

    def built(holds: str):
        executor = LocalSynthesisExecutor(print_output=False)
        executor.problems_by_task_id[parent.task_id] = SimpleNamespace()
        return executor._build_problem(
            child, problem=_problem(holds), parent_outcomes={parent.task_id: outcome}
        )

    assert built("fixed").held_level_targets == {"CU": [100.0]}
    assert built("none").held_level_targets is None
