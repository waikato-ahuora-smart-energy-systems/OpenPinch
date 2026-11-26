"""The site utility balance a site network's model carries.

Processes draw from a level (heaters, ``use``) and supply it (coolers,
``generation``). Inside the network's own model the site closes each level
per period by buying into it (``imported``), letting heat down to a colder
level, or dumping a surplus, and the model's utility cost is what that
balance costs: the site pays only for its net utility use. Here the block
is built on a bare model with fixed loads and solved by HiGHS.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch.analysis.heat_exchanger_networks.site.utility_system import (
    add_site_balance,
    solved_balance,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pyomo_model import PyomoModel
from OpenPinch.contracts.synthesis.site import SiteUtilityLevel, SiteUtilityLevels


def _highs() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(not _highs(), reason="needs highspy"),
]


def _level(name, kind, t_high, t_low, *, imp=None, dump=None, cap=None, periods=1):
    return SiteUtilityLevel(
        name=name,
        kind=kind,
        t_high=(t_high,) * periods,
        t_low=(t_low,) * periods,
        import_price=None if imp is None else (imp,) * periods,
        dump_price=None if dump is None else (dump,) * periods,
        import_capacity=cap,
    )


def _levels(**overrides) -> SiteUtilityLevels:
    """A boiler (VHPS), one steam main (LPS) and cooling water (CW)."""
    levels = {
        "VHPS": _level("VHPS", "hot", 250.1, 250.0, imp=40.0),
        "LPS": _level("LPS", "both", 150.1, 150.0, imp=30.0, dump=30.0),
        "CW": _level("CW", "cold", 25.0, 20.0, dump=5.0),
    }
    levels.update(overrides)
    return SiteUtilityLevels(levels=tuple(levels.values()))


def _solve(levels: SiteUtilityLevels, use: dict, generation: dict):
    """The cheapest balance of fixed loads (kW per period, by level)."""
    m = PyomoModel()
    periods = len(levels.period_weights)
    zero = (0.0,) * periods
    blocks = [
        add_site_balance(
            m,
            levels,
            n,
            [m.Param(use.get(level.name, zero)[n]) for level in levels.levels],
            [m.Param(generation.get(level.name, zero)[n]) for level in levels.levels],
        )
        for n in range(periods)
    ]
    m.Minimize(
        sum(
            w * (block.hot_cost + block.cold_cost)
            for w, block in zip(levels.period_weights, blocks, strict=True)
        )
    )
    run = backend.solve_pyomo_model(m, solver_name="highs")
    return run, blocks


def _balance(levels, use, generation):
    run, blocks = _solve(levels, use, generation)
    assert run.failure_reason is None, run.failure_reason
    return solved_balance(levels, blocks)


def _closes(balance) -> None:
    for row in balance.balances:
        assert row.use + row.letdown_out + row.dump == pytest.approx(
            row.generation + row.letdown_in + row.imported, abs=1e-7
        ), row.level


def test_a_short_main_buys_only_what_its_generation_does_not_cover():
    """A uses 30 kW of LPS and B raises 10: LPS buys the other 20 at 30 $
    (cheaper than letting VHPS down at 40); the raised steam is not charged."""
    balance = _balance(_levels(), {"LPS": (30.0,)}, {"LPS": (10.0,)})

    lps = balance.level("LPS")
    assert lps.use == pytest.approx(30.0, abs=1e-7)
    assert lps.generation == pytest.approx(10.0, abs=1e-7)
    assert lps.imported == pytest.approx(20.0, abs=1e-7)
    assert lps.dump == pytest.approx(0.0, abs=1e-7)
    assert lps.cost == pytest.approx(600.0, abs=1e-5)
    assert balance.operating_cost == pytest.approx(600.0, abs=1e-5)
    assert balance.hot_utility == pytest.approx((20.0,), abs=1e-7)
    assert balance.cold_utility == pytest.approx((0.0,), abs=1e-7)
    _closes(balance)


def test_the_balance_of_fixed_loads_needs_no_model_of_ones_own():
    """What a joined set of process networks uses and raises on each level
    is known without a site model: the cheapest balance of those loads is
    solved on its own, by level name."""
    from OpenPinch.analysis.heat_exchanger_networks.site.utility_system import (
        balance_of_loads,
    )

    balance = balance_of_loads(_levels(), {"LPS": (10.0,)}, {"LPS": (25.0,)})

    assert balance.level("LPS").letdown_out == pytest.approx(15.0, abs=1e-7)
    assert balance.level("CW").dump == pytest.approx(15.0, abs=1e-7)
    assert balance.operating_cost == pytest.approx(75.0, abs=1e-5)
    _closes(balance)


def test_a_network_is_priced_by_the_site_balance_of_its_own_duties():
    """A heater draws 30 kW of LPS and a cooler raises 10: the site buys
    the other 20 at 30 $, and that is the network's utility cost."""
    from OpenPinch.analysis.heat_exchanger_networks.site.levels import (
        SiteUtilitySystem,
        priced_by_site_balance,
    )
    from OpenPinch.domain._heat_exchanger.period_state import (
        HeatExchangerPeriodState,
    )
    from OpenPinch.domain.enums import HeatExchangerKind, StreamID
    from OpenPinch.domain.heat_exchanger import HeatExchanger
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    def unit(kind, source, sink, duty):
        heater = kind is HeatExchangerKind.HOT_UTILITY
        return HeatExchanger(
            exchanger_id=f"{source}->{sink}",
            kind=kind,
            source_stream=source,
            sink_stream=sink,
            source_stream_role=StreamID.Utility if heater else StreamID.Process,
            sink_stream_role=StreamID.Process if heater else StreamID.Utility,
            period_states=(
                HeatExchangerPeriodState(period_id="0", period_idx=0, duty=duty),
            ),
        )

    network = HeatExchangerNetwork(
        exchangers=(
            unit(HeatExchangerKind.HOT_UTILITY, "Hot Utility.LPS", "A.C1", 30.0),
            unit(HeatExchangerKind.COLD_UTILITY, "B.H1", "Cold Utility.LPS", 10.0),
        ),
        run_id="site-seed",
        method="site_pinch_design",
        capital_cost=1000.0,
    )
    system = SiteUtilitySystem(
        levels=_levels(),
        hot_utilities=(("Hot Utility.VHPS", "VHPS"), ("Hot Utility.LPS", "LPS")),
        cold_utilities=(("Cold Utility.LPS", "LPS"), ("Cold Utility.CW", "CW")),
    )

    priced = priced_by_site_balance(network, system)

    assert priced.utility_cost == pytest.approx(600.0, abs=1e-5)
    assert priced.total_annual_cost == pytest.approx(1600.0, abs=1e-5)
    assert priced.summary_metrics["site_total_annual_cost"] == pytest.approx(1600.0)
    balance = priced.source_metadata["site_utility_balance"]
    assert balance.level("LPS").imported == pytest.approx(20.0, abs=1e-7)


def test_a_surplus_on_a_main_is_let_down_to_the_cheapest_dump():
    """B raises 25 kW of LPS and A uses 10: the 15 kW surplus goes down to
    cooling water at 5 $ rather than being dumped on LPS at 30 $."""
    balance = _balance(_levels(), {"LPS": (10.0,)}, {"LPS": (25.0,)})

    lps, cw = balance.level("LPS"), balance.level("CW")
    assert lps.letdown_out == pytest.approx(15.0, abs=1e-7)
    assert cw.letdown_in == pytest.approx(15.0, abs=1e-7)
    assert cw.dump == pytest.approx(15.0, abs=1e-7)
    assert balance.operating_cost == pytest.approx(75.0, abs=1e-5)
    _closes(balance)


def test_each_period_balances_on_its_own_and_costs_are_weighted():
    """Period 0 is short of LPS, period 1 has a surplus; the operating cost
    is the weighted mean of the two."""
    levels = SiteUtilityLevels(
        levels=(
            _level("LPS", "both", 150.1, 150.0, imp=30.0, dump=30.0, periods=2),
            _level("CW", "cold", 25.0, 20.0, dump=5.0, periods=2),
        ),
        period_weights=(3.0, 1.0),
    )

    balance = _balance(levels, {"LPS": (30.0, 10.0)}, {"LPS": (10.0, 25.0)})

    assert balance.period_costs == pytest.approx((600.0, 75.0), abs=1e-5)
    assert balance.operating_cost == pytest.approx((3 * 600.0 + 75.0) / 4, abs=1e-5)
    _closes(balance)


def test_letdown_follows_each_periods_temperatures():
    """In period 1 the main sits below cooling water, so its surplus cannot
    go down to it and is dumped on the main itself."""
    levels = SiteUtilityLevels(
        levels=(
            SiteUtilityLevel(
                name="LPS",
                kind="both",
                t_high=(150.1, 15.1),
                t_low=(150.0, 15.0),
                import_price=(30.0, 30.0),
                dump_price=(30.0, 30.0),
            ),
            _level("CW", "cold", 25.0, 20.0, dump=5.0, periods=2),
        ),
        period_weights=(1.0, 1.0),
    )

    balance = _balance(levels, {}, {"LPS": (10.0, 10.0)})

    assert balance.level("LPS", 0).letdown_out == pytest.approx(10.0, abs=1e-7)
    assert balance.level("LPS", 1).letdown_out == pytest.approx(0.0, abs=1e-7)
    assert balance.level("LPS", 1).dump == pytest.approx(10.0, abs=1e-7)


def test_an_import_capacity_sends_the_rest_down_from_a_hotter_level():
    capped = _level("LPS", "both", 150.1, 150.0, imp=30.0, dump=30.0, cap=(5.0,))

    balance = _balance(_levels(LPS=capped), {"LPS": (20.0,)}, {})

    assert balance.level("LPS").imported == pytest.approx(5.0, abs=1e-7)
    assert balance.level("VHPS").letdown_out == pytest.approx(15.0, abs=1e-7)
    assert balance.operating_cost == pytest.approx(5 * 30.0 + 15 * 40.0, abs=1e-5)
    _closes(balance)


def test_a_balance_the_levels_cannot_close_fails_the_solve():
    capped = _level("VHPS", "hot", 250.1, 250.0, imp=40.0, cap=(1.0,))

    run, _blocks = _solve(SiteUtilityLevels(levels=(capped,)), {"VHPS": (5.0,)}, {})

    assert run.failure_reason is not None


@st.composite
def _cascades(draw):
    """Levels 5 K apart (0.1 K wide), import prices rising with
    temperature, one dump price for every level, random loads."""
    count = draw(st.integers(min_value=2, max_value=5))
    kinds = draw(
        st.lists(
            st.sampled_from(("hot", "cold", "both")), min_size=count, max_size=count
        )
    )
    dump = draw(st.floats(min_value=0.0, max_value=20.0))
    steps = draw(
        st.lists(
            st.floats(min_value=0.5, max_value=10.0), min_size=count, max_size=count
        )
    )
    # No load or difference of loads within the cascade's resolution (it
    # drops duties of 1e-6 kW or less): whole hundredths of a kW.
    duty = st.integers(min_value=0, max_value=10000).map(lambda q: q / 100.0)
    levels, use, generation = [], [], []
    price = 10.0
    for i, kind in enumerate(reversed(kinds)):
        price += steps[i]
        t = 50.0 + 5.0 * i
        levels.append(
            _level(
                f"U{i}",
                kind,
                t + 0.1,
                t,
                imp=price if kind != "cold" else None,
                dump=dump if kind != "hot" else None,
            )
        )
        use.append(draw(duty) if kind != "cold" else 0.0)
        generation.append(draw(duty) if kind != "hot" else 0.0)
    return levels, use, generation, dump


def _cascade_cost(levels, use, generation, dump) -> tuple[float, float, float]:
    """Cost, site hot and site cold utility from the Total Site cascade."""
    from OpenPinch.analysis.targeting.indirect import _target_site_utilities
    from OpenPinch.domain.stream import Stream
    from OpenPinch.domain.stream_collection import StreamCollection

    def utility(level, heat_flow, hot):
        high, low = level.t_high[0], level.t_low[0]
        supply, target = (high, low) if hot else (low, high)
        return Stream(
            name=level.name,
            supply_temperature=supply,
            target_temperature=target,
            heat_flow=heat_flow,
            is_process_stream=False,
        )

    hot = StreamCollection()
    cold = StreamCollection()
    for level, q_use, q_gen in zip(levels, use, generation, strict=True):
        if level.kind != "cold":
            hot.add(utility(level, q_use, True))
        if level.kind != "hot":
            cold.add(utility(level, q_gen, False))
    unmet, surplus = _target_site_utilities(hot, cold)
    prices = {level.name: level.import_price for level in levels}
    imported = [(u.name, float(u.heat_flow)) for u in unmet]
    dumped = sum(float(u.heat_flow) for u in surplus)
    cost = sum(q * prices[name][0] for name, q in imported) + dump * dumped
    return cost, sum(q for _, q in imported), dumped


@settings(deadline=None)
@given(_cascades())
def test_the_balance_of_fixed_loads_costs_what_the_models_own_block_does(case):
    """``balance_of_loads`` writes the same rows for SciPy that
    ``add_site_balance`` writes into a model; the two must not drift."""
    from OpenPinch.analysis.heat_exchanger_networks.site.utility_system import (
        balance_of_loads,
    )

    levels, use, generation, _dump = case
    names = [level.name for level in levels]
    site = SiteUtilityLevels(levels=tuple(levels))
    used = {name: (q,) for name, q in zip(names, use, strict=True)}
    raised = {name: (q,) for name, q in zip(names, generation, strict=True)}

    alone = balance_of_loads(site, used, raised)

    in_a_model = _balance(site, used, raised)
    assert alone.operating_cost == pytest.approx(
        in_a_model.operating_cost, rel=1e-7, abs=1e-5
    )
    _closes(alone)


@settings(deadline=None)
@given(_cascades())
def test_with_prices_rising_with_temperature_the_balance_is_the_site_cascade(case):
    """The Total Site cascade sends generation down to the hottest unmet
    use; with import prices rising with temperature and one dump price,
    that is the cheapest balance, so the block gives its cost and its site
    hot and cold utility."""
    levels, use, generation, dump = case
    names = [level.name for level in levels]

    balance = _balance(
        SiteUtilityLevels(levels=tuple(levels)),
        {name: (q,) for name, q in zip(names, use, strict=True)},
        {name: (q,) for name, q in zip(names, generation, strict=True)},
    )
    cost, hot, cold = _cascade_cost(levels, use, generation, dump)

    assert balance.operating_cost == pytest.approx(cost, rel=1e-7, abs=1e-5)
    assert balance.hot_utility[0] == pytest.approx(hot, rel=1e-7, abs=1e-6)
    assert balance.cold_utility[0] == pytest.approx(cold, rel=1e-7, abs=1e-6)
    _closes(balance)
