"""The retrofit's heat pump case keeps the heat pump's approach under the
multiplier of a designed existing network.

A retrofit of a network designed at a dt_cont multiplier builds its model
with every stream's contribution at that multiplier. The as-piped target
stood the heat pump ``HPR_DT_CONT`` off the coolers and heaters as given,
so the case's heat pump streams are written with ``HPR_DT_CONT`` over the
multiplier: at 1.5 and ``HPR_DT_CONT`` of 1 K, 0.667 K, which the model
scales back to 1 K.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    levels_problem,
)
from tests.application.heat_pump_stream_cases import step_of_no_size


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name,
        supply_temperature=supply,
        target_temperature=target,
        heat_flow=duty,
        heat_transfer_coefficient=1.0,
    )


def test_the_heat_pump_streams_carry_hpr_dt_cont_over_the_networks_multiplier(
    monkeypatch,
):
    problem = levels_problem()
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 110.0, 109.99, 120.0)},
        hpr_cold_streams={"b": _stream("HP_C1", 33.99, 34.0, 80.0)},
        amb_streams=StreamCollection(),
    )
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    cases = []

    def search(case, existing, settings, **arguments):
        if "heat_pump" not in arguments:
            return base
        cases.append(case)
        return "search from the heat pump"

    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", search)
    monkeypatch.setattr(
        heat_pump_retrofit,
        "heat_pump_step",
        lambda *a, **k: (step_of_no_size(), solved),
    )
    monkeypatch.setattr(heat_pump_retrofit, "joined", lambda first, *rest: rest)

    heat_pump_retrofit.retrofit_with_heat_pump(
        problem,
        object(),
        RetrofitSettings(
            heat_pump_cycle="carnot", approach=None, dt_cont_multiplier=1.5
        ),
    )

    written = {
        record["name"]: record["dt_cont"]["value"]
        for record in cases[0].to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    approach = float(problem._require_prepared_root_zone().config.hpr.dt_cont)
    assert written == {
        "HP_H1": pytest.approx(approach / 1.5),
        "HP_C1": pytest.approx(approach / 1.5),
    }
