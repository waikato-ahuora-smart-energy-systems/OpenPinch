"""Site utility accounting: the balance of each utility level over a site.

Processes draw from a level (heaters, ``use``) and supply it (coolers,
``generation``). The site closes each level by buying from it (``imported``),
letting heat down to a colder level, or dumping a surplus. The accounting is
the cheapest such balance, and its dual on a level is what one more kW of use
there costs: generation on a main that is short is a credit.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.heat_exchanger_networks.site.utility_system import (
    account_utility_system,
)
from OpenPinch.contracts.synthesis.site import (
    LevelLoad,
    SiteUtilityLevel,
    SiteUtilityLevels,
)


def _levels(**overrides) -> SiteUtilityLevels:
    """A boiler (VHPS), one steam main (LPS) and cooling water (CW)."""
    levels = {
        "VHPS": SiteUtilityLevel(
            name="VHPS", kind="hot", t_high=250.1, t_low=250.0, import_price=(40.0,)
        ),
        "LPS": SiteUtilityLevel(
            name="LPS",
            kind="both",
            t_high=150.1,
            t_low=150.0,
            import_price=(30.0,),
            dump_price=(30.0,),
        ),
        "CW": SiteUtilityLevel(
            name="CW", kind="cold", t_high=25.0, t_low=20.0, dump_price=(5.0,)
        ),
    }
    levels.update(overrides)
    return SiteUtilityLevels(levels=tuple(levels.values()))


# Prices come from a second HiGHS solve over the optimal duals, accurate
# to its tolerances (about 1e-9 relative), so they are compared at 1e-7.


def _closes(balance) -> None:
    for row in balance.balances:
        assert row.use + row.letdown_out + row.dump == pytest.approx(
            row.generation + row.letdown_in + row.imported, abs=1e-9
        ), row.level


def test_a_short_main_imports_and_credits_generation_at_its_price():
    """A uses 30 kW of LPS and B raises 10: LPS buys the other 20 at 30 $
    (cheaper than letting VHPS down at 40), so one more kW of use costs 30
    and one more kW raised saves 30."""
    balance = account_utility_system(
        _levels(),
        {
            "A": {"LPS": LevelLoad(use=(30.0,), generation=(0.0,))},
            "B": {"LPS": LevelLoad(use=(0.0,), generation=(10.0,))},
        },
    )

    lps = balance.level("LPS")
    assert lps.use == pytest.approx(30.0, abs=1e-9)
    assert lps.generation == pytest.approx(10.0, abs=1e-9)
    assert lps.imported == pytest.approx(20.0, abs=1e-9)
    assert lps.dump == pytest.approx(0.0, abs=1e-9)
    assert lps.cost == pytest.approx(600.0, abs=1e-6)
    assert lps.marginal_price == pytest.approx(30.0, rel=1e-7)
    assert balance.operating_cost == pytest.approx(600.0, abs=1e-6)
    assert balance.hot_utility == pytest.approx((20.0,), abs=1e-9)
    assert balance.cold_utility == pytest.approx((0.0,), abs=1e-9)
    _closes(balance)


def test_a_surplus_on_a_main_is_let_down_to_the_cheapest_dump():
    """B raises 25 kW of LPS and A uses 10: the 15 kW surplus goes down to
    cooling water at 5 $ rather than being dumped on LPS at 30 $, so more
    steam raised there costs 5 per kW."""
    balance = account_utility_system(
        _levels(),
        {
            "A": {"LPS": LevelLoad(use=(10.0,), generation=(0.0,))},
            "B": {"LPS": LevelLoad(use=(0.0,), generation=(25.0,))},
        },
    )

    lps, cw = balance.level("LPS"), balance.level("CW")
    assert lps.letdown_out == pytest.approx(15.0, abs=1e-9)
    assert cw.letdown_in == pytest.approx(15.0, abs=1e-9)
    assert cw.dump == pytest.approx(15.0, abs=1e-9)
    assert balance.operating_cost == pytest.approx(75.0, abs=1e-6)
    assert lps.marginal_price == pytest.approx(-5.0, rel=1e-7)
    assert cw.marginal_price == pytest.approx(-5.0, rel=1e-7)
    _closes(balance)


def test_each_period_balances_on_its_own_and_costs_are_weighted():
    """Period 0 is short of LPS, period 1 has a surplus; the operating cost
    is the weighted mean of the two."""
    levels = SiteUtilityLevels(
        levels=(
            SiteUtilityLevel(
                name="LPS",
                kind="both",
                t_high=150.1,
                t_low=150.0,
                import_price=(30.0, 30.0),
                dump_price=(30.0, 30.0),
            ),
            SiteUtilityLevel(
                name="CW", kind="cold", t_high=25.0, t_low=20.0, dump_price=(5.0, 5.0)
            ),
        ),
        period_weights=(3.0, 1.0),
    )

    balance = account_utility_system(
        levels, {"A": {"LPS": LevelLoad(use=(30.0, 10.0), generation=(10.0, 25.0))}}
    )

    assert balance.period_costs == pytest.approx((600.0, 75.0), abs=1e-6)
    assert balance.operating_cost == pytest.approx((3 * 600.0 + 75.0) / 4, abs=1e-6)
    assert balance.level("LPS", 0).marginal_price == pytest.approx(30.0, rel=1e-7)
    assert balance.level("LPS", 1).marginal_price == pytest.approx(-5.0, rel=1e-7)
    _closes(balance)


def test_an_import_capacity_sends_the_rest_down_from_a_hotter_level():
    capped = SiteUtilityLevel(
        name="LPS",
        kind="both",
        t_high=150.1,
        t_low=150.0,
        import_price=(30.0,),
        dump_price=(30.0,),
        import_capacity=(5.0,),
    )

    balance = account_utility_system(
        _levels(LPS=capped), {"A": {"LPS": LevelLoad(use=(20.0,), generation=(0.0,))}}
    )

    assert balance.level("LPS").imported == pytest.approx(5.0, abs=1e-9)
    assert balance.level("VHPS").letdown_out == pytest.approx(15.0, abs=1e-9)
    assert balance.operating_cost == pytest.approx(5 * 30.0 + 15 * 40.0, abs=1e-6)
    assert balance.level("LPS").marginal_price == pytest.approx(40.0, rel=1e-7)
    _closes(balance)


@pytest.mark.parametrize(
    ("loads", "message"),
    [
        ({"A": {"HPS": LevelLoad(use=(1.0,), generation=(0.0,))}}, "unknown level"),
        ({"A": {"CW": LevelLoad(use=(1.0,), generation=(0.0,))}}, "cannot supply"),
        ({"A": {"VHPS": LevelLoad(use=(0.0,), generation=(1.0,))}}, "cannot take"),
        ({"A": {"LPS": LevelLoad(use=(1.0, 1.0), generation=(0.0, 0.0))}}, "period"),
    ],
)
def test_loads_a_level_cannot_take_are_refused(loads, message):
    with pytest.raises(ValueError, match=message):
        account_utility_system(_levels(), loads)


def test_a_balance_the_levels_cannot_close_is_refused():
    capped = SiteUtilityLevel(
        name="VHPS",
        kind="hot",
        t_high=250.1,
        t_low=250.0,
        import_price=(40.0,),
        import_capacity=(1.0,),
    )
    levels = SiteUtilityLevels(levels=(capped,))

    with pytest.raises(ValueError, match="cannot balance period 0"):
        account_utility_system(
            levels, {"A": {"VHPS": LevelLoad(use=(5.0,), generation=(0.0,))}}
        )


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
            SiteUtilityLevel(
                name=f"U{i}",
                kind=kind,
                t_high=t + 0.1,
                t_low=t,
                import_price=(price,) if kind != "cold" else None,
                dump_price=(dump,) if kind != "hot" else None,
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
        supply, target = (
            (level.t_high, level.t_low) if hot else (level.t_low, level.t_high)
        )
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


@given(_cascades())
def test_with_prices_rising_with_temperature_the_balance_is_the_site_cascade(case):
    """The Total Site cascade sends generation down to the hottest unmet
    use; with import prices rising with temperature and one dump price,
    that is the cheapest balance, so the accounting gives its cost and its
    site hot and cold utility."""
    levels, use, generation, dump = case
    loads = {
        "P": {
            level.name: LevelLoad(use=(q_use,), generation=(q_gen,))
            for level, q_use, q_gen in zip(levels, use, generation, strict=True)
        }
    }

    balance = account_utility_system(SiteUtilityLevels(levels=tuple(levels)), loads)
    cost, hot, cold = _cascade_cost(levels, use, generation, dump)

    assert balance.operating_cost == pytest.approx(cost, rel=1e-7, abs=1e-6)
    assert balance.hot_utility[0] == pytest.approx(hot, rel=1e-7, abs=1e-6)
    assert balance.cold_utility[0] == pytest.approx(cold, rel=1e-7, abs=1e-6)
    _closes(balance)


def test_a_balanced_main_prices_use_and_generation_on_either_side():
    """LPS uses exactly what it raises. One more kW of use must be bought
    (30 $); one more kW raised goes down to cooling water (5 $), so it is
    charged 5. The dual reported is their midpoint. Cooling water, with no
    load, prices heat raised into it at its dump price; a hot level has no
    generation price and a cold one no use price."""
    balance = account_utility_system(
        _levels(), {"A": {"LPS": LevelLoad(use=(10.0,), generation=(10.0,))}}
    )

    lps, cw, vhps = (balance.level(name) for name in ("LPS", "CW", "VHPS"))
    assert lps.use_price == pytest.approx(30.0, rel=1e-7)
    assert lps.generation_price == pytest.approx(-5.0, rel=1e-7)
    assert lps.marginal_price == pytest.approx(12.5, rel=1e-7)
    assert cw.generation_price == pytest.approx(-5.0, rel=1e-7)
    assert cw.use_price is None
    assert vhps.use_price == pytest.approx(40.0, rel=1e-7)
    assert vhps.generation_price is None


def test_without_degeneracy_both_prices_are_the_dual():
    balance = account_utility_system(
        _levels(),
        {
            "A": {"LPS": LevelLoad(use=(30.0,), generation=(0.0,))},
            "B": {"LPS": LevelLoad(use=(0.0,), generation=(10.0,))},
        },
    )

    lps = balance.level("LPS")
    assert lps.use_price == pytest.approx(30.0, rel=1e-7)
    assert lps.generation_price == pytest.approx(30.0, rel=1e-7)
