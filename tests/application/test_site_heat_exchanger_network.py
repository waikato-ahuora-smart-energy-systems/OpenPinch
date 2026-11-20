"""A site network: each process designed at its levels, evolved as one.

``design.total_site_heat_exchanger_network`` designs each process zone with
the pinch design method, holding its utility levels at their targets, joins
the networks into one site network per dt_cont multiplier, and evolves those
seeds as one problem: every process keeps its levels and recovers heat only
within itself. There is no thermal derivative step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind

D_TMIN = 10.0


def _solvers_available() -> bool:
    try:
        backend.require_solver_backend("highs")
        backend.require_solver_backend("ipopt-pyomo")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _stream(zone, name, supply, target, heat_flow):
    if isinstance(heat_flow, list):
        flow = {"values": heat_flow, "unit": "kW"}
    else:
        flow = _quantity(heat_flow, "kW")
    return {
        "zone": f"Site/{zone}",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": flow,
        "dt_cont": _quantity(D_TMIN / 2, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, supply, target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(price, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
    }


def _site(processes=("Heating", "Cooling"), *, periods=False) -> PinchProblem:
    """Heating needs 40 kW from LPS between two matches; Cooling raises 40 kW
    of LP steam between two. By price HPS and cooling water would take
    both. With ``periods``, a second period carries more on C1 and H3."""
    c1 = [140.0, 168.0] if periods else 140.0
    h3 = [140.0, 168.0] if periods else 140.0
    streams = {
        "Heating": [
            _stream("Heating", "H1", 220.0, 150.0, 70.0),
            _stream("Heating", "H2", 100.0, 70.0, 30.0),
            _stream("Heating", "C1", 60.0, 200.0, c1),
        ],
        "Cooling": [
            _stream("Cooling", "H3", 200.0, 60.0, h3),
            _stream("Cooling", "C2", 50.0, 120.0, 70.0),
            _stream("Cooling", "C3", 160.0, 190.0, 30.0),
        ],
    }
    return PinchProblem(
        source={
            "streams": [stream for zone in processes for stream in streams[zone]],
            "utilities": [
                _utility("LPS", "Hot", 140.0, 139.9, 60.0),
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("LPG", "Cold", 120.0, 120.1, 20.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [
                    {"name": zone, "type": "Process Zone"} for zone in processes
                ],
            },
            "options": (
                {
                    "PROBLEM_PERIOD_IDS": ["base", "peak"],
                    "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
                }
                if periods
                else {}
            ),
        }
    )


def _process(stream: str) -> str:
    return stream.rsplit(".", 1)[0]


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_site_network_holds_each_processs_levels_and_matches_within_it():
    """Heating's targets put 40 kW on LPS and Cooling's 40 kW on LPG; by
    price HPS (40 $/MWh, against LPS at 60) takes Heating's heat. Both
    families evolve; the level-held one is among the results, every network
    matches only within a process, and the cheapest is selected."""
    problem = _site()

    view = problem.design.total_site_heat_exchanger_network(
        width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1}
    )
    networks = [outcome.network for outcome in view.result.ranked_networks]

    assert view.result.method == "network_evolution_method"
    for network in networks:
        for exchanger in network.exchangers:
            if exchanger.kind is HeatExchangerKind.RECOVERY:
                assert _process(exchanger.source_stream) == _process(
                    exchanger.sink_stream
                )
    held = [
        network
        for network in networks
        if _holds(
            network,
            {
                ("Heating", "Hot Utility.LPS"): [40.0],
                ("Cooling", "Cold Utility.LPG"): [40.0],
            },
        )
    ]
    assert held
    selected = view.selected_network
    assert (
        selected.summary_metrics["site_total_annual_cost"]
        <= min(network.summary_metrics["site_total_annual_cost"] for network in held)
        + 1e-6
    )
    assert verify_network_feasibility(selected) == ()


def _holds(network, expected, *, abs_tol=1e-3) -> bool:
    """Whether each (process, utility) carries the expected duty per period."""
    duties = {
        (process.rsplit("/", 1)[-1], utility): levels
        for (process, utility), levels in _level_duties(network).items()
    }
    return all(
        duties.get(key) is not None
        and duties[key] == pytest.approx(levels, abs=abs_tol)
        for key, levels in expected.items()
    )


def _level_duties(network) -> dict[tuple[str, str], list[float]]:
    """Each process's heater and cooler duty per period, by utility."""
    duties: dict[tuple[str, str], list[float]] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            key = (_process(exchanger.sink_stream), exchanger.source_stream)
        elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
            key = (_process(exchanger.source_stream), exchanger.sink_stream)
        else:
            continue
        by_period = duties.setdefault(key, [0.0] * len(exchanger.period_states))
        for state in exchanger.period_states:
            if state.active:
                by_period[state.period_idx] += float(state.duty)
    return duties


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_multiperiod_site_network_holds_each_processs_levels_in_every_period():
    """In the peak period C1 and H3 carry 20% more: Heating then needs 57 kW
    of LPS (C1 below LPS's reach, less H2) and 11 kW of HPS, and Cooling
    raises 54.5 kW of LP steam with 13.5 kW to cooling water. Each process
    is designed over both periods, so the level-held family holds its
    levels in each."""
    problem = _site(periods=True)

    view = problem.design.total_site_heat_exchanger_network(
        width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1}
    )
    networks = [outcome.network for outcome in view.result.ranked_networks]

    expected = {
        ("Heating", "Hot Utility.LPS"): [40.0, 57.0],
        ("Heating", "Hot Utility.HPS"): [0.0, 11.0],
        ("Cooling", "Cold Utility.LPG"): [40.0, 54.5],
        ("Cooling", "Cold Utility.CW"): [0.0, 13.5],
    }
    held = [network for network in networks if _holds(network, expected)]
    assert held
    for network in networks:
        for exchanger in network.exchangers:
            if exchanger.kind is HeatExchangerKind.RECOVERY:
                assert _process(exchanger.source_stream) == _process(
                    exchanger.sink_stream
                )
    assert verify_network_feasibility(held[0]) == ()


def test_a_site_network_needs_more_than_one_process():
    with pytest.raises(ValueError, match="more than one process zone"):
        _site(("Heating",)).design.total_site_heat_exchanger_network()


def test_a_pooled_design_of_a_site_warns_that_processes_may_match(monkeypatch):
    """``heat_exchanger_network()`` on a site pools every process's streams;
    it says so and names the site method. A single process does not warn."""
    from OpenPinch.application._problem.accessors import design

    monkeypatch.setattr(
        design._DesignAccessor, "_synthesise", lambda self, **kwargs: "designed"
    )

    with pytest.warns(UserWarning, match="total_site_heat_exchanger_network"):
        assert _site().design.heat_exchanger_network() == "designed"


def test_a_single_process_design_does_not_warn(monkeypatch, recwarn):
    from OpenPinch.application._problem.accessors import design

    monkeypatch.setattr(
        design._DesignAccessor, "_synthesise", lambda self, **kwargs: "designed"
    )

    _site(("Heating",)).design.heat_exchanger_network()
    site = _site()
    site.design.heat_exchanger_network(zone=next(iter(site.master_zone.subzones)))

    assert not [w for w in recwarn if "total_site" in str(w.message)]


def _steam_main_site() -> PinchProblem:
    """``_site`` with LP steam as a main: Cooling raises it and Heating
    uses it, so the site buys only what Cooling does not raise."""
    streams = [
        _stream("Heating", "H1", 220.0, 150.0, 70.0),
        _stream("Heating", "H2", 100.0, 70.0, 30.0),
        _stream("Heating", "C1", 60.0, 200.0, 140.0),
        _stream("Cooling", "H3", 200.0, 60.0, 140.0),
        _stream("Cooling", "C2", 50.0, 120.0, 70.0),
        _stream("Cooling", "C3", 160.0, 190.0, 30.0),
    ]
    return PinchProblem(
        source={
            "streams": streams,
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("LPS", "Both", 140.0, 139.9, 30.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [
                    {"name": zone, "type": "Process Zone"}
                    for zone in ("Heating", "Cooling")
                ],
            },
        }
    )


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_site_network_is_costed_with_site_utility_accounting():
    """Steam Cooling raises on the LPS main meets Heating's use, so the site
    buys less LPS than Heating uses; the network's own model carries that
    balance, so its utility cost is the site's net cost."""
    problem = _steam_main_site()

    view = problem.design.total_site_heat_exchanger_network(
        width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1}
    )
    network = view.selected_network

    metrics = network.summary_metrics
    balance = network.source_metadata["site_utility_balance"]
    lps = balance.level("LPS")
    duties = _level_duties(network)
    heating = next(p for p, _ in duties if "Heating" in p)
    cooling = next(p for p, _ in duties if "Cooling" in p)
    assert lps.use == pytest.approx(duties[(heating, "Hot Utility.LPS")][0], rel=1e-6)
    assert lps.generation == pytest.approx(
        duties[(cooling, "Cold Utility.LPS")][0], rel=1e-6
    )
    assert lps.generation > 0.0
    assert lps.imported == pytest.approx(max(0.0, lps.use - lps.generation), abs=1e-5)
    for row in balance.balances:
        assert row.use + row.letdown_out + row.dump == pytest.approx(
            row.generation + row.letdown_in + row.imported, abs=1e-5
        )
    assert network.utility_cost == pytest.approx(balance.operating_cost, rel=1e-6)
    assert metrics["site_total_annual_cost"] == pytest.approx(
        network.capital_cost + balance.operating_cost, rel=1e-9
    )
    for outcome in view.result.ranked_networks[1:]:
        other = outcome.network.summary_metrics.get("site_total_annual_cost")
        assert other is None or other >= metrics["site_total_annual_cost"] - 1e-6


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_site_networks_own_cost_credits_steam_raised_on_a_main():
    """Evolution prices each utility at its level's site price: LPS that
    Cooling raises saves what LPS costs to buy, so the network's own total
    annual cost is its site total annual cost, not one that charges the
    raised steam as a cooler."""
    network = (
        _steam_main_site()
        .design.total_site_heat_exchanger_network(
            width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1}
        )
        .selected_network
    )

    assert network.total_annual_cost == pytest.approx(
        network.summary_metrics["site_total_annual_cost"], rel=1e-6
    )


def test_each_process_is_designed_by_price_and_by_its_level_targets(monkeypatch):
    """The site design runs each process's pinch design twice, with levels
    chosen by price and with levels held at their targets, and seeds
    evolution from both."""
    from types import SimpleNamespace

    from OpenPinch.application._problem.accessors import design

    placements: list[tuple[str, str]] = []
    captured = {}

    def fake_design(self, *, zone, options):
        placements.append((zone, options["HENS_UTILITY_PLACEMENT"]))
        return SimpleNamespace(top=lambda n: ())

    def fake_seeds(pools, *, level_held=None):
        captured["pools"], captured["held"] = pools, level_held
        return ("seed",)

    monkeypatch.setattr(design._DesignAccessor, "heat_exchanger_network", fake_design)
    monkeypatch.setattr(
        design._DesignAccessor,
        "network_evolution",
        lambda self, seeds, **kwargs: seeds,
    )
    monkeypatch.setattr(
        "OpenPinch.analysis.heat_exchanger_networks.site.site_seeds", fake_seeds
    )

    assert _site().design.total_site_heat_exchanger_network() == ("seed",)
    assert sorted(placements) == [
        ("Cooling", "price"),
        ("Cooling", "temperature"),
        ("Heating", "price"),
        ("Heating", "temperature"),
    ]
    assert set(captured["pools"]) == set(captured["held"]) == {"Heating", "Cooling"}


def _fake_process_designs(monkeypatch, failing):
    """Each process design returns its (process, placement) as its networks;
    the pairs in ``failing`` raise. Returns the pools site seeding receives."""
    from OpenPinch.analysis.heat_exchanger_networks import site
    from OpenPinch.application._problem.accessors import design

    def heat_exchanger_network(self, *, zone=None, options=None, **kwargs):
        placement = options["HENS_UTILITY_PLACEMENT"]
        if (zone, placement) in failing:
            raise RuntimeError(f"{zone} by {placement} failed")
        return SimpleNamespace(
            top=lambda n: [SimpleNamespace(network=(zone, placement))]
        )

    seeded = {}

    def site_seeds(pools, *, level_held=None):
        seeded.update(pools=pools, level_held=level_held)
        return ("seed",)

    monkeypatch.setattr(
        design._DesignAccessor, "heat_exchanger_network", heat_exchanger_network
    )
    monkeypatch.setattr(site, "site_seeds", site_seeds)
    monkeypatch.setattr(
        design._DesignAccessor,
        "network_evolution",
        lambda self, seeds, **kwargs: seeds,
    )
    return seeded


def test_a_process_whose_design_by_price_fails_seeds_from_its_level_held_networks(
    monkeypatch,
):
    seeded = _fake_process_designs(monkeypatch, failing={("Heating", "price")})

    assert _site().design.total_site_heat_exchanger_network() == ("seed",)
    assert seeded["pools"] == {
        "Heating": (("Heating", "temperature"),),
        "Cooling": (("Cooling", "price"),),
    }
    assert seeded["level_held"]["Heating"] == (("Heating", "temperature"),)


def test_a_process_whose_level_held_design_fails_seeds_from_its_price_networks(
    monkeypatch,
):
    seeded = _fake_process_designs(monkeypatch, failing={("Cooling", "temperature")})

    _site().design.total_site_heat_exchanger_network()

    assert seeded["level_held"]["Cooling"] == (("Cooling", "price"),)


def test_a_process_whose_designs_both_fail_fails_the_site_design(monkeypatch):
    _fake_process_designs(
        monkeypatch, failing={("Heating", "price"), ("Heating", "temperature")}
    )

    with pytest.raises(RuntimeError, match="Heating by temperature failed"):
        _site().design.total_site_heat_exchanger_network()


def test_duty_optimisation_refuses_a_site_network():
    """The fixed-structure model neither holds each process's levels nor
    keeps the site utility balance, so a site network is refused rather
    than re-priced unit by unit."""
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    site = HeatExchangerNetwork(
        exchangers=(),
        run_id="site",
        source_metadata={"site_processes": {"Site/Heating": {}}},
    )

    with pytest.raises(ValueError, match="site network"):
        _site().design.optimise_duties(site)


@pytest.mark.parametrize(
    "marker",
    [
        {"source_metadata": {"site_utility_balance": {}}},
        {"summary_metrics": {"site_total_annual_cost": 1.0}},
        {"source_metadata": {"site_processes": {"Site/Heating": {}}}},
    ],
    ids=["site balance", "site cost", "site seed"],
)
@pytest.mark.parametrize("tool", ["optimise_duties", "retrofit", "retrofit_heat_pumps"])
def test_fixed_structure_tools_refuse_a_site_network(marker, tool):
    """The network ``total_site_heat_exchanger_network`` returns carries the
    site utility balance and site cost, not the seed's ``site_processes``;
    any of them marks a site network, which the fixed-structure tools
    refuse rather than re-price unit by unit."""
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    site = HeatExchangerNetwork(exchangers=(), run_id="site", **marker)

    with pytest.raises(ValueError, match="site network"):
        getattr(_site().design, tool)(site)


def test_a_stored_overdesign_margin_does_not_reach_the_process_designs(monkeypatch):
    """Rating a process network at a margin drops its level targets, so the
    site's process designs run without the stored margin."""
    from OpenPinch.application._problem.accessors import design

    seen = []

    def process_design(self, *, zone=None, options=None, **kwargs):
        seen.append(dict(options or {}))
        raise ValueError("stop after the first process design")

    monkeypatch.setattr(
        design._DesignAccessor, "heat_exchanger_network", process_design
    )
    source = _site().to_problem_json()
    source["options"] = dict(source.get("options") or {}) | {
        "HENS_OVERDESIGN_PERCENTAGE": 10.0
    }
    problem = PinchProblem(source=source)

    with pytest.raises(ValueError, match="stop after"):
        problem.design.total_site_heat_exchanger_network()

    assert seen
    assert all(options["HENS_OVERDESIGN_PERCENTAGE"] is None for options in seen)
