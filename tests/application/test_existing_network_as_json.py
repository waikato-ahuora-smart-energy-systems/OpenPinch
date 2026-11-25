"""A network's JSON is taken as the existing network through run_command.

A client of the command layer (the OpenPinch-WASM app) holds a solved
network only as the JSON a design command returned, and ``design.retrofit``,
``design.retrofit_heat_pumps`` and ``design.optimise_duties`` read every
mapping as a hand-written structure and refused it (R11-1 in the app's
follow-ups, 2026-11-25). Each solve is stubbed: only the network it is given
is checked.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.application.commands import run_command
from tests.analysis.heat_exchanger_networks.test_retrofit import (
    CASE_JSON,
    _existing_network,
)


class _Captured(RuntimeError):
    pass


def _run(monkeypatch, command: str, argument: str, target, name: str):
    captured = {}

    def capture(*args, **kwargs):
        captured["network"] = next(
            value
            for value in (*args, *kwargs.values())
            if type(value).__name__ == "HeatExchangerNetwork"
        )
        raise _Captured

    monkeypatch.setattr(target, name, capture)
    network = _existing_network()
    with pytest.raises(_Captured):
        run_command(
            {
                "command": command,
                "problem": {"data": json.loads(CASE_JSON.read_text(encoding="utf-8"))},
                "arguments": {argument: network.model_dump(mode="json")},
            }
        )
    return network, captured["network"]


def test_retrofit_takes_a_networks_json_as_the_existing_network(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    given, received = _run(
        monkeypatch, "design.retrofit", "existing", search, "run_retrofit_search"
    )

    assert received == given


def test_retrofit_heat_pumps_takes_a_networks_json_as_the_existing_network(
    monkeypatch,
):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    given, received = _run(
        monkeypatch,
        "design.retrofit_heat_pumps",
        "existing",
        heat_pump_targets,
        "retrofit_heat_pumps",
    )

    assert received == given


def test_optimise_duties_takes_a_networks_json_as_the_network(monkeypatch):
    from OpenPinch.application._problem.accessors.design import _DesignAccessor

    given, received = _run(
        monkeypatch,
        "design.optimise_duties",
        "network",
        _DesignAccessor,
        "_optimise_duties",
    )

    assert received == given
