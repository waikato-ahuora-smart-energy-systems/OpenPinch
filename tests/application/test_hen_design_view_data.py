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
