"""The overdesign margin search, with a stand-in rating solve."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.overdesign import apply_overdesign
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _rating(failing):
    """A rating whose cost rises with the margin; ``failing(percentage)``
    makes it raise, as an infeasible rating solve does."""

    def rate(installed):
        percentage = round(100.0 * (installed.summary_metrics["scale"] - 1.0), 9)
        if failing(percentage):
            raise RuntimeError("Solver termination condition: infeasible")
        return SimpleNamespace(
            selected_network=HeatExchangerNetwork(total_annual_cost=100.0 + percentage)
        )

    return rate


def _network():
    # No recovery areas to scale: the stand-in reads the margin from a
    # metric each scaled copy carries instead.
    return HeatExchangerNetwork(summary_metrics={"scale": 1.0})


def _scaled(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks import overdesign

    monkeypatch.setattr(
        overdesign,
        "with_scaled_recovery_areas",
        lambda network, factor: network.model_copy(
            update={"summary_metrics": {"scale": factor}}
        ),
    )


def test_an_optimised_margin_with_no_positive_margin_rated_warns(monkeypatch):
    _scaled(monkeypatch)

    with pytest.warns(UserWarning, match="no positive overdesign margin"):
        _result, metrics = apply_overdesign(
            _network(), "optimise", maximum=50.0, rate=_rating(lambda p: p > 0.0)
        )

    assert metrics["percentage"] == 0.0
    assert metrics["search"] == [[0.0, 100.0]]
    assert metrics["failed_margins"]
    assert all(point > 0.0 for point in metrics["failed_margins"])


def test_an_optimised_margin_skips_the_margins_that_fail(monkeypatch):
    _scaled(monkeypatch)
    rate = _rating(lambda p: p > 40.0)

    _result, metrics = apply_overdesign(_network(), "optimise", maximum=50.0, rate=rate)

    assert metrics["percentage"] == 0.0
    assert 50.0 in metrics["failed_margins"]
    assert all(point <= 40.0 for point, _cost in metrics["search"])
