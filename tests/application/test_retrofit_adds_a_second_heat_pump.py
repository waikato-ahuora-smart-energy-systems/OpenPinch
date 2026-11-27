"""A retrofit adds a heat pump beside the one a network already holds.

The retrofit's heat pump step named its streams as every heat pump target
does (``HP_H1``, ``HP_C1``), the names the existing machine's streams
already hold in the case built from the network. The step's case was
refused ("already has a stream named 'HP_H1' ... rename the process
stream") and the step left out at every size, so such a network could
never gain a second machine.

On the basic pinch case with a half-load machine in the network and
electricity at 20 $/MWh a retrofit now adds a 223 kW machine ``HP2``
beside the 375 kW ``HP``. Whether a second machine pays there turns on a
search of a few hundred evaluations, which differs between platforms, so
the naming is tested on its own.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application import heat_pump_retrofit
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _level(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=duty
    )


def test_the_step_names_its_streams_after_its_machine_where_theirs_are_taken(
    monkeypatch,
):
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _level("HP_H1", 96.0, 95.99, 500.0)},
        hpr_cold_streams={"b": _level("HP_C1", 20.0, 20.01, 300.0)},
        amb_streams=StreamCollection(),
        w_net=200.0,
        cop_h=2.5,
    )
    monkeypatch.setattr(
        heat_pump_retrofit, "as_piped_heat_pump", lambda *args, **kwargs: solved
    )
    problem = SimpleNamespace(
        master_zone=SimpleNamespace(
            config=SimpleNamespace(costing=Configuration().costing)
        )
    )
    settings = SimpleNamespace(
        heat_pump_cycle="carnot", heat_pump_cost=0.0, dt_cont_multiplier=1.0
    )

    step, _solved = heat_pump_retrofit.heat_pump_step(
        problem,
        SimpleNamespace(network=object()),
        settings,
        machine="HP2",
        # The names an existing machine's streams hold.
        taken=frozenset({"HP_H1", "HP_C1"}),
    )

    assert step.record.machine == "HP2"
    assert [stream.name for stream in step.condensers] == ["HP2_H1"]
    assert [stream.name for stream in step.evaporators] == ["HP2_C1"]


def test_the_search_gives_the_step_the_names_the_existing_machines_hold(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
    )

    asked = {}

    def step(_problem, _base, _settings, **arguments):
        asked.update(arguments)
        return None, None

    monkeypatch.setattr(heat_pump_retrofit, "heat_pump_step", step)
    monkeypatch.setattr(
        heat_pump_retrofit,
        "run_retrofit_search",
        lambda *args, **kwargs: SimpleNamespace(frontier=[object()]),
    )
    existing = SimpleNamespace(
        heat_pumps=[SimpleNamespace(streams=({"name": "HP_H1"}, {"name": "HP_C1"}))]
    )

    heat_pump_retrofit.retrofit_with_heat_pump(
        object(),
        existing,
        RetrofitSettings(heat_pump_cycle="carnot", max_modifications=1),
        machine="HP2",
    )

    assert asked["machine"] == "HP2"
    assert asked["taken"] == {"HP_H1", "HP_C1"}


def test_a_retrofits_machine_takes_a_name_the_network_does_not_hold():
    from OpenPinch.application._problem.accessors.design import _free_machine_name

    def network(*machines):
        return SimpleNamespace(
            heat_pumps=[SimpleNamespace(machine=name) for name in machines]
        )

    assert _free_machine_name(network()) == "HP"
    assert _free_machine_name(network("HP")) == "HP2"
    assert _free_machine_name(network("HP", "HP2")) == "HP3"


def test_a_retrofits_machine_avoids_the_names_its_streams_would_take():
    # A machine labelled HP2 whose streams kept a target's own names
    # (HP_H1, HP_C1): a new machine HP would name its streams the same.
    from OpenPinch.application._problem.accessors.design import _free_machine_name

    held = SimpleNamespace(
        machine="HP2",
        streams=({"name": "HP_H1"}, {"name": "HP_C1"}, {"name": "AIR_H1"}),
    )

    assert _free_machine_name(SimpleNamespace(heat_pumps=[held])) == "HP3"
