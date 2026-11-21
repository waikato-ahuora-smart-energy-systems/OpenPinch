"""Site design options: the letdown price and options a given-levels design
ignores.

``HENS_SITE_LETDOWN_PRICE`` ($/MWh, default 0) prices heat the site utility
balance lets down from one level to a colder one. A ``levels="given"``
design uses the case's utilities in one design, so it ignores
``level_counts``, ``placement`` and ``HENS_SITE_TIME_LIMIT``, and warns when
any is set.
"""

from __future__ import annotations

import pytest

from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS
from tests.application.test_site_heat_exchanger_network import (
    _site,
    _solvers_available,
)


def _level_held(view):
    """The ranked network that holds Heating on LPS, its site cost and the
    heat it lets down (kW)."""
    for outcome in view.result.ranked_networks:
        network = outcome.network
        balance = network.source_metadata["site_utility_balance"]
        if balance.level("LPS").use > 1.0:
            letdown = sum(row.letdown_out for row in balance.balances)
            return network.summary_metrics["site_total_annual_cost"], letdown
    raise AssertionError("no level-held network")


def _design(options):
    return _site().design.total_site_heat_exchanger_network(
        width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1} | options
    )


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_letdown_price_raises_the_site_cost_of_a_network_that_lets_down():
    """The level-held network holds Heating's 40 kW on LPS and Cooling's
    40 kW on LPG: the site buys HPS (40 $/MWh) and lets it down to LPS (60),
    and lets LPG down to cooling water rather than dump it (20). At
    10 $/MWh both letdowns still pay, so the 80 kW let down costs
    80 * 10 * 8300 / 1000 $/y more. A price of 0 is the default."""
    default = _design({})
    free = _design({"HENS_SITE_LETDOWN_PRICE": 0.0})
    priced = _design({"HENS_SITE_LETDOWN_PRICE": 10.0})

    cost, letdown = _level_held(default)
    assert letdown == pytest.approx(80.0, abs=1e-4)
    assert _level_held(free)[0] == pytest.approx(cost, rel=1e-9)
    assert free.selected_network.summary_metrics[
        "site_total_annual_cost"
    ] == pytest.approx(
        default.selected_network.summary_metrics["site_total_annual_cost"], rel=1e-9
    )
    assert _level_held(priced)[0] == pytest.approx(
        cost + 80.0 * 10.0 * 8300.0 / 1000.0, rel=1e-6
    )


def test_the_letdown_price_is_a_configuration_key():
    spec = USER_CONFIG_FIELD_SPECS["HENS_SITE_LETDOWN_PRICE"]

    assert spec.default == 0.0
    assert spec.config_path == ("hens", "site_letdown_price")
    with pytest.raises(ValueError, match="HENS_SITE_LETDOWN_PRICE"):
        _site().design.total_site_heat_exchanger_network(
            options={"HENS_SITE_LETDOWN_PRICE": -1.0}
        )


@pytest.fixture
def given_design(monkeypatch):
    """Stub the given-levels design; record that it ran."""
    from OpenPinch.application._problem.accessors import design

    calls = []
    monkeypatch.setattr(
        design._DesignAccessor,
        "_site_design",
        lambda self, **kwargs: calls.append(kwargs) or "designed",
    )
    return calls


@pytest.mark.parametrize(
    ("arguments", "ignored"),
    [
        ({"level_counts": [2, 3]}, "level_counts"),
        ({"placement": {"evaluation_limit": 10}}, "placement"),
        ({"options": {"HENS_SITE_TIME_LIMIT": 60.0}}, "HENS_SITE_TIME_LIMIT"),
    ],
)
def test_a_given_levels_design_warns_of_options_it_ignores(
    given_design, arguments, ignored
):
    with pytest.warns(UserWarning, match=ignored):
        result = _site().design.total_site_heat_exchanger_network(
            levels="given", **arguments
        )

    assert result == "designed"
    assert len(given_design) == 1


def test_a_given_levels_design_without_site_options_does_not_warn(
    given_design, recwarn
):
    _site().design.total_site_heat_exchanger_network(levels="given")

    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
