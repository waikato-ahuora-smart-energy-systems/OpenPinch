"""A site design says when a process's design failed one way.

Each process is designed twice before the site evolution: with levels
chosen by price and with its levels held at their targets. When one of the
two raised, the other stood in for it with no warning and no log line; on
the steam main case the site then cost 25,015 against 20,581 $/y (review
finding 33). The solves are stubbed: no synthesis solver runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application._problem.accessors import design
from tests.analysis.heat_exchanger_networks.test_site_design import (
    _network,
    _recovery,
)
from tests.application import test_site_heat_exchanger_network as site


def test_a_process_whose_level_held_design_fails_is_named(monkeypatch):
    def designed(self, *, zone=None, options=None, **arguments):
        if zone == "Cooling" and options["HENS_UTILITY_PLACEMENT"] == "temperature":
            raise ValueError("no network holds the levels")
        network = _network([_recovery(f"{zone}.H1", f"{zone}.C1", 1)])
        return SimpleNamespace(top=lambda count: [SimpleNamespace(network=network)])

    monkeypatch.setattr(design._DesignAccessor, "heat_exchanger_network", designed)
    monkeypatch.setattr(
        design._DesignAccessor, "network_evolution", lambda self, seeds, **_: seeds
    )

    with pytest.warns(UserWarning, match="Cooling") as caught:
        site._site().design._site_design(width=0, depth=1, options=None)

    (message,) = [str(w.message) for w in caught if "Cooling" in str(w.message)]
    assert "held at their targets" in message
    assert "no network holds the levels" in message
    assert "chosen by price" in message
