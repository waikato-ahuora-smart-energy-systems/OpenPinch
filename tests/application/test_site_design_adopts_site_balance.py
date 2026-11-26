"""A site design whose evolution adopts a candidate keeps its site balance.

The Heating and Cooling processes on an LP steam main, beside a unit
operation with a hot and a cold stream of its own. Evolution adopts a
candidate that changes what the levels use and raise; the model kept its
own imports, dumps and letdowns, so every level missed its balance and the
design raised "verification failed: site utility level ..." with no network.
"""

from __future__ import annotations

import pytest

from OpenPinch.application.problem import PinchProblem
from tests.application import test_site_heat_exchanger_network as site

pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.solver,
    pytest.mark.skipif(not site._solvers_available(), reason="needs highspy and ipopt"),
]


def _site_with_a_unit_operation() -> PinchProblem:
    streams = [
        site._stream("Heating", "H1", 220.0, 150.0, 70.0),
        site._stream("Heating", "H2", 100.0, 70.0, 30.0),
        site._stream("Heating", "C1", 60.0, 200.0, 140.0),
        site._stream("Cooling", "H3", 200.0, 60.0, 140.0),
        site._stream("Cooling", "C2", 50.0, 120.0, 70.0),
        site._stream("Cooling", "C3", 160.0, 190.0, 30.0),
        site._stream("Extra", "H7", 200.0, 100.0, 100.0),
        site._stream("Extra", "C7", 60.0, 160.0, 100.0),
    ]
    utilities = [
        site._utility("HPS", "Hot", 250.0, 249.9, 40.0),
        site._utility("LPS", "Both", 140.0, 139.9, 30.0),
        site._utility("CW", "Cold", 20.0, 25.0, 5.0),
    ]
    zones = [
        {"name": "Heating", "type": "Process Zone"},
        {"name": "Cooling", "type": "Process Zone"},
        {"name": "Extra", "type": "Unit Operation"},
    ]
    return PinchProblem(
        source={
            "streams": streams,
            "utilities": utilities,
            "zone_tree": {"name": "Site", "type": "Site", "children": zones},
        }
    )


def test_a_site_design_returns_a_network_after_evolution_adopts_a_candidate():
    view = _site_with_a_unit_operation().design.total_site_heat_exchanger_network(
        width=0,
        depth=1,
        options={"HENS_EVM_N_AD_BRANCHES": 1, "HENS_MAX_PARALLEL": 1},
    )

    network = view.selected_network
    balance = network.source_metadata["site_utility_balance"]
    for row in balance.balances:
        inflow = row.generation + row.letdown_in + row.imported
        outflow = row.use + row.letdown_out + row.dump
        assert inflow == pytest.approx(outflow, abs=1e-3)
