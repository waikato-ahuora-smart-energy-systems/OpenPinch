"""Carnot ORC targeting on the grand composite curve below the pinch."""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.analysis.orc.carnot import (
    carnot_orc_design,
    carnot_orc_work,
    orc_temperature_window,
)
from OpenPinch.analysis.orc.costing import orc_costs, orc_machine_capital_costs
from OpenPinch.analysis.orc.inputs import OrcTargetInputs
from OpenPinch.analysis.orc.profile import OrcHeatSource, orc_heat_source_from_gcc
from OpenPinch.analysis.orc.search import (
    OrcTargetingError,
    evaluate_carnot_orc,
    optimise_carnot_orc,
)

# A GCC with 1000 kW of hot utility above a 150 degC pinch and 4000 kW of
# surplus below it, falling linearly to 20 degC.
T_GCC = np.array([250.0, 200.0, 150.0, 100.0, 20.0])
H_GCC = np.array([1000.0, 500.0, 0.0, 2000.0, 4000.0])


def _source() -> OrcHeatSource:
    source = orc_heat_source_from_gcc(T_GCC, H_GCC)
    assert source is not None
    return source


def _inputs(**overrides) -> OrcTargetInputs:
    values = {"max_multistart": 2, "maximum_iterations": 50, **overrides}
    return OrcTargetInputs(**values)


def test_heat_source_starts_at_the_pinch():
    source = _source()

    assert source.T_pinch == 150.0
    assert source.surplus == 4000.0
    assert source.heat_at(125.0) == pytest.approx(1000.0)


def test_capacity_respects_pockets_below_each_temperature():
    # A pocket: the cascade falls back to 500 kW at 60 degC.
    source = OrcHeatSource(
        T=np.array([150.0, 100.0, 60.0, 20.0]),
        H=np.array([0.0, 2000.0, 500.0, 3000.0]),
    )

    assert source.capacity(120.0) == pytest.approx(500.0)
    assert source.capacity(50.0) == pytest.approx(500.0 + (2500.0 / 40.0) * 10.0)
    assert source.capacity(150.0) == 0.0


def test_no_surplus_gives_no_heat_source():
    assert orc_heat_source_from_gcc(np.array([200.0, 100.0]), np.zeros(2)) is None


def test_carnot_work_is_a_fraction_of_the_carnot_limit():
    W = carnot_orc_work(np.array([127.0]), 27.0, np.array([1000.0]), 0.5)

    assert W[0] == pytest.approx(0.5 * (1.0 - 300.15 / 400.15) * 1000.0)


def test_window_runs_from_the_pinch_to_condenser_plus_lift():
    inputs = _inputs(T_cond=30.0, min_lift=10.0, dt_cont=5.0)

    T_hi, T_lo = orc_temperature_window(_source(), inputs)

    assert T_hi == 150.0
    assert T_lo == pytest.approx(45.0 + inputs.dt_phase_change)


def test_surplus_colder_than_the_condenser_has_no_window():
    inputs = _inputs(T_cond=140.0)

    assert orc_temperature_window(_source(), inputs) is None
    with pytest.raises(OrcTargetingError, match="too cold"):
        optimise_carnot_orc(_source(), inputs)


@pytest.mark.parametrize("n_stages", [1, 2, 3])
def test_designs_never_take_more_than_the_cascade_allows(n_stages):
    source = _source()
    inputs = _inputs(n_stages=n_stages)
    rng = np.random.default_rng(20260715)

    for _ in range(200):
        design = carnot_orc_design(rng.random(2 * n_stages), source, inputs)
        cumulative = np.cumsum(design.Q_in)
        for T_s, taken in zip(design.T_evap_shifted, cumulative, strict=True):
            assert taken <= source.capacity(T_s - inputs.dt_phase_change) + 1e-9
        assert list(design.T_evap_shifted) == sorted(
            design.T_evap_shifted, reverse=True
        )
        assert all(w >= 0.0 for w in design.W_net)
        assert design.Q_out_total == pytest.approx(
            design.Q_in_total - design.W_net_total
        )


def test_load_fraction_caps_the_heat_taken():
    source = _source()
    inputs = _inputs(load_fraction=0.25, n_stages=2)

    design = carnot_orc_design(np.array([0.9, 0.9, 1.0, 1.0]), source, inputs)

    assert design.Q_in_total <= 0.25 * source.surplus + 1e-9


def test_capital_follows_the_power_law_and_skips_empty_units():
    inputs = _inputs()

    costs = orc_machine_capital_costs(np.array([1000.0, 0.0]), inputs)

    assert costs[0] == pytest.approx(1.3 * 2.3e6)
    assert costs[1] == 0.0


def test_cost_change_credits_power_and_charges_net_cooling():
    inputs = _inputs()

    costs = orc_costs(W_net=np.array([100.0]), Q_in=1000.0, Q_out=900.0, inputs=inputs)

    assert costs.power_value == pytest.approx(100.0 * 100.0 * 8300.0 / 1000.0)
    assert costs.cooling_cost_change == pytest.approx(-100.0 * 2.5 * 8300.0 / 1000.0)
    assert costs.total_annualized_cost_change == pytest.approx(
        costs.annualized_capital_cost - costs.power_value + costs.cooling_cost_change
    )


def test_search_finds_a_paying_design_no_worse_than_its_starts():
    source = _source()
    inputs = _inputs(n_stages=2)

    result = optimise_carnot_orc(source, inputs)

    assert result.costs.total_annualized_cost_change < 0.0
    assert result.design.W_net_total > 0.0
    for start in ((0.5, 0.5, 1.0, 1.0), (0.25, 0.5, 1.0, 1.0)):
        _, start_costs = evaluate_carnot_orc(np.array(start), source, inputs)
        assert (
            result.costs.total_annualized_cost_change
            <= start_costs.total_annualized_cost_change + 1e-6
        )


def test_more_units_never_cost_more():
    # A second unit can always take no heat, so it never has to cost more.
    source = _source()

    one = optimise_carnot_orc(source, _inputs(n_stages=1))
    two = optimise_carnot_orc(source, _inputs(n_stages=2))

    slack = 0.01 * abs(one.costs.total_annualized_cost_change)
    assert (
        two.costs.total_annualized_cost_change
        <= one.costs.total_annualized_cost_change + slack
    )


def test_unaffordable_capital_means_no_beneficial_orc():
    inputs = _inputs(equipment_cost=1e12)

    with pytest.raises(OrcTargetingError, match="No beneficial ORC"):
        optimise_carnot_orc(_source(), inputs)


def test_residual_gcc_loses_each_units_heat_below_its_evaporator():
    from OpenPinch.analysis.orc.carnot import OrcDesign
    from OpenPinch.analysis.orc.service import orc_residual_gcc

    design = OrcDesign(
        T_evap_shifted=(120.0, 80.0),
        T_evap=(115.0, 75.0),
        T_cond=30.0,
        Q_in=(500.0, 300.0),
        W_net=(50.0, 20.0),
    )

    T, H = orc_residual_gcc(T_GCC, H_GCC, design, dt_phase_change=0.01)

    def at(t):
        return float(np.interp(t, T[::-1], H[::-1]))

    assert at(250.0) == pytest.approx(1000.0)
    # 130 degC is above the hotter unit, so the cascade there is unchanged.
    assert at(130.0) == pytest.approx(800.0)
    assert at(110.0) == pytest.approx(1600.0 - 500.0)
    assert at(100.0) == pytest.approx(2000.0 - 500.0)
    assert at(20.0) == pytest.approx(4000.0 - 800.0)
    assert np.all(np.diff(T) <= 0.0)
