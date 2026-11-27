"""Verification says when a network carries no stream data to check.

A network's JSON keeps its exchangers but not the stream data in its source
metadata (loads, film coefficients, required approaches), which most checks
compare against. A network read back from JSON skipped them and was reported
feasible, even with a halved duty or a crossed approach (review finding 26).
It is still not failed, but a warning says it could not be checked and how
to check it: rate it on its problem.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def test_a_network_without_stream_data_is_verified_with_a_warning():
    network = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)],
        heaters=["C1"],
        coolers=["H1"],
        hot_utility="HU",
        cold_utility="CU",
    )

    with pytest.warns(UserWarning, match="no stream data"):
        verify_network_feasibility(network)


def test_heat_capacity_flowrates_are_stream_data_too():
    """The checks also read streams' heat capacity flowrates with their
    supply and target temperatures; a network carrying those is checked and
    does not warn (Codex review of #556)."""
    import warnings

    network = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)],
        heaters=["C1"],
        coolers=["H1"],
        hot_utility="HU",
        cold_utility="CU",
    )
    network = network.model_copy(
        update={
            "source_metadata": {
                "hot_stream_heat_capacity_flowrates": [1.0],
                "cold_stream_heat_capacity_flowrates": [1.0],
            }
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        verify_network_feasibility(network)

    assert not [w for w in caught if "no stream data" in str(w.message)]


def test_a_designs_own_checks_do_not_warn_of_missing_stream_data():
    """A design verifies networks it built, some without stream data (a
    site's joined process networks); the warning is for a network read back
    and checked by a caller, not for the design's own checks."""
    import warnings

    from tests.application.test_site_linked_only_by_utilities import (
        _utility_linked_site,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _utility_linked_site().design.total_site_heat_exchanger_network(
            width=0, depth=0
        )

    assert not [w for w in caught if "no stream data" in str(w.message)]
