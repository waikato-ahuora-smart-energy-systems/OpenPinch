"""Seed networks for standalone thermal derivative and evolution runs."""

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.results import seeds


def _problem(method, selected, ranked):
    design = SimpleNamespace(
        method=method,
        network=selected,
        ranked=tuple(SimpleNamespace(network=network) for network in ranked),
    )
    return SimpleNamespace(results=SimpleNamespace(design=design), master_zone=None)


@pytest.fixture
def named_networks(monkeypatch):
    # Networks are stand-in strings; the ranking and topology signature are
    # covered by the reporting tests.
    monkeypatch.setattr(seeds, "ranked_networks", lambda design: design.ranked)
    monkeypatch.setattr(
        seeds, "network_structure_signature", lambda network: network.rstrip("'")
    )


@pytest.mark.parametrize(
    "method",
    [
        "pinch_design_method",
        "heuristic_pinch_design_method",
        "thermal_derivative_method",
        "network_evolution_method",
        "open_hens_method",
    ],
)
def test_any_cached_design_seeds_a_later_method(named_networks, method):
    problem = _problem(method, "A", ("A", "B"))

    assert seeds.resolve_seed_networks(
        problem, None, method_name="network_evolution_method", limit=5
    ) == ("A", "B")


def test_cached_seeds_lead_with_the_selected_network(named_networks):
    # The user selected rank 2, so B leads; A' repeats A's topology.
    problem = _problem("pinch_design_method", "B", ("A", "B", "A'", "C", "D"))

    assert seeds.resolve_seed_networks(
        problem, None, method_name="thermal_derivative_method", limit=3
    ) == ("B", "A", "C")


def test_missing_cached_design_still_needs_initial_networks():
    problem = SimpleNamespace(results=None)

    with pytest.raises(ValueError, match="requires initial_networks"):
        seeds.resolve_seed_networks(
            problem, None, method_name="thermal_derivative_method", limit=5
        )
