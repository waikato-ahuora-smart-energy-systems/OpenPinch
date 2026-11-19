"""A site network: each process designed at its levels, evolved as one.

``design.total_site_heat_exchanger_network`` designs each process zone with
the pinch design method, holding its utility levels at their targets, joins
the networks into one site network per dt_cont multiplier, and evolves those
seeds as one problem: every process keeps its levels and recovers heat only
within itself. There is no thermal derivative step.
"""

from __future__ import annotations

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
    problem = _site()

    view = problem.design.total_site_heat_exchanger_network(
        width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1}
    )
    network = view.selected_network

    assert view.result.method == "network_evolution_method"
    recovery = [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
    ]
    assert recovery
    for exchanger in recovery:
        assert _process(exchanger.source_stream) == _process(exchanger.sink_stream)
    duties: dict[tuple[str, str], float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            key = (_process(exchanger.sink_stream), exchanger.source_stream)
        elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
            key = (_process(exchanger.source_stream), exchanger.sink_stream)
        else:
            continue
        duties[key] = duties.get(key, 0.0) + exchanger.state().duty
    heating = next(p for p, _ in duties if "Heating" in p)
    cooling = next(p for p, _ in duties if "Cooling" in p)
    assert duties.get((heating, "Hot Utility.LPS"), 0.0) == pytest.approx(
        40.0, rel=1e-3
    )
    assert duties.get((cooling, "Cold Utility.LPG"), 0.0) == pytest.approx(
        40.0, rel=1e-3
    )
    assert verify_network_feasibility(network) == ()


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
    is designed over both periods, so its levels hold in each."""
    problem = _site(periods=True)

    view = problem.design.total_site_heat_exchanger_network(
        width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1}
    )
    network = view.selected_network

    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            assert _process(exchanger.source_stream) == _process(exchanger.sink_stream)
    duties = _level_duties(network)
    heating = next(p for p, _ in duties if "Heating" in p)
    cooling = next(p for p, _ in duties if "Cooling" in p)
    expected = {
        (heating, "Hot Utility.LPS"): [40.0, 57.0],
        (heating, "Hot Utility.HPS"): [0.0, 11.0],
        (cooling, "Hot Utility.HPS"): [0.0, 0.0],
        (cooling, "Cold Utility.LPG"): [40.0, 54.5],
        (cooling, "Cold Utility.CW"): [0.0, 13.5],
    }
    for key, levels in expected.items():
        assert duties.get(key, [0.0, 0.0]) == pytest.approx(levels, abs=1e-3), key
    assert verify_network_feasibility(network) == ()


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
