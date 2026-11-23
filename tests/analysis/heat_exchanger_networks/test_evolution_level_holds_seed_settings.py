"""The problem built for a seeded task takes the seed's levels from the
task's settings, as it takes them from a parent task otherwise."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
)
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from tests.analysis.heat_exchanger_networks.test_evolution_level_holds_parents import (
    TARGETS,
    _problem,
)


def test_a_task_built_from_seed_settings_holds_the_seeds_levels():
    def built(holds: str):
        task = HeatExchangerNetworkSynthesisTask(
            run_id="method-test",
            method="network_evolution_method",
            dt_cont_multiplier=1.0,
            stage_count=1,
            settings={
                "utility_level_targets": TARGETS,
                "held_utility_levels": {"CU": [100.0]},
            },
        )
        return LocalSynthesisExecutor(print_output=False)._build_problem(
            task, problem=_problem(holds), parent_outcomes={}
        )

    fixed = built("fixed")
    assert fixed.utility_level_targets == TARGETS
    assert fixed.held_level_targets == {"CU": [100.0]}
    # Without the option the targets are still recorded, for ranking.
    unheld = built("none")
    assert unheld.utility_level_targets == TARGETS
    assert unheld.held_level_targets is None
