"""A synthesis task takes its level holds from the configuration."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    _heating_problem,
)


def test_a_task_takes_its_level_holds_from_the_configuration():
    """``HENS_EVOLUTION_LEVEL_HOLDS`` reaches the problem built for a task;
    unset, the task holds nothing, as before."""
    task = HeatExchangerNetworkSynthesisTask(
        run_id="method-test",
        method="network_evolution_method",
        dt_cont_multiplier=1.0,
        stage_count=1,
    )

    def built(**options):
        problem = _heating_problem()
        problem.update_options(options)
        return LocalSynthesisExecutor(print_output=False)._build_problem(
            task, problem=problem, parent_outcomes={}
        )

    assert built().level_holds == "none"
    assert built(HENS_EVOLUTION_LEVEL_HOLDS="fixed").level_holds == "fixed"
