"""HEN design-view data: ranking, totals, utility duties and the grid layout."""

import pytest

from OpenPinch.presentation.network_grid.geometry import build_grid_model
from tests.analysis.heat_exchanger_networks.test_design_workflow import (
    _public_example_problem,
    _use_fake_default_executor,
)


@pytest.fixture
def design(monkeypatch):
    _use_fake_default_executor(monkeypatch)
    return _public_example_problem().design.heat_exchanger_network()


def test_grid_layout_is_the_grid_model_as_data(design):
    model = build_grid_model(design.network(rank=1))

    layout = design.grid_layout()

    assert layout.period_id == model.period_id
    assert layout.hot_streams == model.hot_streams
    assert layout.cold_streams == model.cold_streams
    assert layout.stages == model.stages
    expected = [
        (kind, match)
        for kind, matches in (
            ("recovery", model.recovery_matches),
            ("hot_utility", model.hot_utility_matches),
            ("cold_utility", model.cold_utility_matches),
        )
        for match in matches
    ]
    assert [m.kind for m in layout.matches] == [kind for kind, _ in expected]
    for row, (_, match) in zip(layout.matches, expected, strict=True):
        assert row.exchanger_id == match.exchanger.exchanger_id
        assert row.source_stream == match.source_stream
        assert row.sink_stream == match.sink_stream
        assert row.stage == match.stage
        assert row.duty == pytest.approx(match.duty, rel=1e-12)
        assert row.source_inlet_temperature == match.state.source_inlet_temperature
        assert row.source_outlet_temperature == match.state.source_outlet_temperature
        assert row.sink_inlet_temperature == match.state.sink_inlet_temperature
        assert row.sink_outlet_temperature == match.state.sink_outlet_temperature
    assert {
        (branch.stream, branch.stage): branch.count for branch in layout.branch_counts
    } == model.branch_counts


def test_view_totals_and_utility_duties_match_the_design_view(design):
    view = design.view()

    assert view.rank == 1
    assert view.total_heat_recovery == pytest.approx(
        design.total_heat_recovery, rel=1e-12
    )
    assert view.total_hot_utility == pytest.approx(design.total_hot_utility, rel=1e-12)
    assert view.total_cold_utility == pytest.approx(
        design.total_cold_utility, rel=1e-12
    )
    assert view.unit_count == len(design.selected_network.exchangers)
    assert [(duty.name, duty.side) for duty in view.utility_duties] == [
        ("Hot Utility.HPS", "hot"),
        ("Cold Utility.CW", "cold"),
    ]
    for duty in view.utility_duties:
        assert duty.duty == pytest.approx(design.utility(duty.name), rel=1e-12)
    assert view.grid == design.grid_layout()


def test_views_of_a_single_period_network_is_its_one_view(design):
    assert design.views() == (design.view(),)


def test_view_ranking_follows_the_duplicate_free_ranking(design):
    ranked = design.top(10)

    ranking = design.view().ranking

    assert [row.rank for row in ranking] == list(range(1, len(ranked) + 1))
    for row, outcome in zip(ranking, ranked, strict=True):
        assert row.method == outcome.method.value
        assert row.status == outcome.status
        assert row.objective_value == pytest.approx(outcome.objective_value, rel=1e-12)
        assert row.total_annual_cost == pytest.approx(
            outcome.network.total_annual_cost, rel=1e-12
        )
        assert row.unit_count == len(outcome.network.exchangers)


@pytest.mark.parametrize("rank", [0, True, 1.0])
def test_view_rejects_a_rank_that_is_not_one_based(design, rank):
    with pytest.raises(ValueError, match="rank must be a positive one-based integer"):
        design.view(rank=rank)
