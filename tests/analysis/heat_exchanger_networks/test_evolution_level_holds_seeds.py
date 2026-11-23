"""A task seeded from a network carries that network's utility levels.

The standalone thermal derivative and network evolution methods start from a
network with no parent task. The seed's ``utility_level_targets`` were not
passed on, so its tasks had nothing to hold and their networks recorded no
targets.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method import (
    build_seeded_network_evolution_method_tasks,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.analysis.heat_exchanger_networks.test_evolution_level_holds_parents import (
    DUTIES,
    TARGETS,
    _problem,
)


def _seed() -> HeatExchangerNetwork:
    """One recovery match, with the heaters and coolers of ``DUTIES``: it
    meets its cold utility target and misses its hot one."""
    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)],
        heaters=["C1", "C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )
    duties = DUTIES | {"H1->C1": 50.0}
    return structure.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(
                    update={
                        "period_states": (
                            HeatExchangerPeriodState(
                                period_id="0",
                                period_idx=0,
                                duty=duties[f"{e.source_stream}->{e.sink_stream}"],
                            ),
                        )
                    }
                )
                for e in structure.exchangers
            ),
            "source_metadata": {"utility_level_targets": TARGETS},
        }
    )


def test_a_seeded_task_takes_its_seeds_targets_and_the_levels_it_holds():
    settings = workflow_settings_from_problem(_problem("fixed"))

    (task,) = build_seeded_network_evolution_method_tasks(settings, [_seed()])

    # Every target, so the task's networks rank as the seed does, and the
    # part the seed holds (its cooling; it misses its heating target).
    assert task.settings["utility_level_targets"] == TARGETS
    assert task.settings["held_utility_levels"] == {"CU": [100.0]}
