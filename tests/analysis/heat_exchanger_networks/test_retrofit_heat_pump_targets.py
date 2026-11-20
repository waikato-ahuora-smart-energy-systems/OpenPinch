"""Heat pump targets for a retrofit: the ideal and the as-piped target."""

from __future__ import annotations

import json

import numpy as np
import pytest

from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.resources import read_sample_case

CASE = "Five-stream-Ciric-and-Floudas-1989.json"
KELVIN = 273.15
# The case's temperature contribution of every process stream.
DT_CONT = 5.0


def _solved_network() -> HeatExchangerNetwork:
    """Ciric and Floudas's existing network with the baseline's solved
    states (temperatures in K): one heater on C2, coolers on H1 and H2."""

    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H3", "C2", 1), ("H1", "C1", 1), ("H2", "C2", 2), ("H3", "C1", 2)],
        heaters=["C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )
    # (duty, source in, source out, sink in, sink out)
    states = {
        "cold-utility:H1->CU": (13603.6, 409.68, 350.15, 293.15, 313.15),
        "cold-utility:H2->CU": (1665.1, 442.77, 361.15, 293.15, 313.15),
        "hot-utility:HU->C2": (17518.7, 773.15, 772.15, 448.81, 538.15),
    }

    def solved(exchanger):
        if exchanger.exchanger_id not in states:
            return exchanger
        duty, source_in, source_out, sink_in, sink_out = states[exchanger.exchanger_id]
        state = HeatExchangerPeriodState(
            period_id="0",
            period_idx=0,
            duty=duty,
            source_inlet_temperature=source_in,
            source_outlet_temperature=source_out,
            sink_inlet_temperature=sink_in,
            sink_outlet_temperature=sink_out,
        )
        return exchanger.model_copy(update={"period_states": (state,)})

    return structure.model_copy(
        update={"exchangers": tuple(solved(e) for e in structure.exchangers)}
    )


def _problem() -> PinchProblem:
    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )

    problem = PinchProblem(json.loads(read_sample_case(CASE)))
    ensure_target_results(problem, {}, None)
    return problem


def test_the_as_piped_profiles_are_the_existing_coolers_and_heaters():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
        as_piped_profiles,
    )

    T_vals, H_hot, H_cold = as_piped_profiles(_problem(), _solved_network())

    # Shifted temperatures (degC), hottest first: the heater's outlet raised
    # and the coldest cooler outlet lowered by the stream's contribution.
    assert T_vals[0] == pytest.approx(538.15 - KELVIN + DT_CONT)
    assert T_vals[-1] == pytest.approx(350.15 - KELVIN - DT_CONT)
    assert np.all(np.diff(T_vals) < 0.0)
    # The sink is the heater's duty, the source the coolers', unnetted.
    assert abs(H_cold).max() == pytest.approx(17518.7)
    assert abs(H_hot).max() == pytest.approx(13603.6 + 1665.1)
    # The sink lies wholly above the heater's inlet, the source below the
    # hottest cooler inlet.
    assert np.all(H_cold[T_vals <= 448.81 - KELVIN + DT_CONT] == 0.0)
    assert np.all(H_hot[T_vals >= 442.77 - KELVIN - DT_CONT] == 0.0)


def test_the_as_piped_profiles_keep_a_streams_contribution_multiplier():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
        as_piped_profiles,
    )

    problem = _problem()
    problem.set_dt_cont_multiplier(2.0)

    T_vals, _H_hot, _H_cold = as_piped_profiles(problem, _solved_network())

    # The heater's outlet raised and H1's cooler outlet, the coldest,
    # lowered by twice the contribution.
    assert T_vals[0] == pytest.approx(538.15 - KELVIN + 2.0 * DT_CONT)
    assert T_vals[-1] == pytest.approx(350.15 - KELVIN - 2.0 * DT_CONT)


def test_a_cooler_on_a_stream_the_problem_lacks_is_an_error():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
        as_piped_profiles,
    )

    network = _solved_network()
    network = network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"source_stream": "H9"})
                if e.source_stream == "H1"
                else e
                for e in network.exchangers
            )
        }
    )

    with pytest.raises(ValueError, match="H9"):
        as_piped_profiles(_problem(), network)


def test_the_ideal_and_as_piped_targets_relieve_their_own_utilities(monkeypatch):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )
    from OpenPinch.domain.enums import ProblemTableLabel, TargetType

    # The HPR service, faked: each target delivers 1000 kW at two levels
    # for 300 kW of work, 100 kW of its heat from ambient air, and leaves
    # 5000 kW of source heat to air.
    calls = []

    def fake_target(*, T_vals, H_hot, H_cold, config, args=None, **_period):
        calls.append((np.asarray(H_hot), np.asarray(H_cold), args))
        return SimpleNamespace(
            Q_cond=np.array([600.0, 400.0]),
            T_cond=np.array([270.0, 200.0]),
            Q_evap=np.array([700.0]),
            T_evap=np.array([90.0]),
            Q_amb_hot=100.0,
            Q_amb_cold=5000.0,
            w_net=300.0,
            cop_h=1000.0 / 300.0,
            hpr_total_annualized_cost=None,
        )

    monkeypatch.setattr(
        heat_pump_targets, "compute_profile_heat_pump_target", fake_target
    )
    problem = _problem()

    targets = heat_pump_targets.retrofit_heat_pumps(
        problem, _solved_network(), args={"maximum_iterations": 7}
    )

    direct = problem.master_zone.targets[TargetType.DI.value]
    ideal, piped = calls
    np.testing.assert_array_equal(ideal[1], direct.pt[ProblemTableLabel.H_NET_COLD])
    assert piped[1].max() == pytest.approx(17518.7)
    assert ideal[2] == piped[2] == {"maximum_iterations": 7}
    # The ideal starts from the process's utility targets, the as-piped
    # from the existing heaters and coolers.
    assert targets.ideal.hot_utility_before == pytest.approx(direct.hot_utility_target)
    assert targets.ideal.cold_utility_before == pytest.approx(
        direct.cold_utility_target
    )
    piped = targets.as_piped
    assert piped.hot_utility_before == pytest.approx(17518.7)
    assert piped.hot_utility_after == pytest.approx(17518.7 - 1000.0)
    assert piped.cold_utility_before == pytest.approx(13603.6 + 1665.1)
    # Only process heat relieves a cooler; the air's does not.
    assert piped.cold_utility_after == pytest.approx(13603.6 + 1665.1 - 600.0)
    assert (piped.work, piped.cop) == (300.0, pytest.approx(1000.0 / 300.0))
    assert piped.ambient_heat == 100.0
    assert piped.condensing_temperatures == (270.0, 200.0)
    assert piped.condenser_duties == (600.0, 400.0)
    assert (piped.evaporating_temperatures, piped.evaporator_duties) == (
        (90.0,),
        (700.0,),
    )
    # The existing network's own shortfall from the targets, priced, and the
    # saving the as-piped heat pump falls short of the ideal's by (none
    # here: the faked service gives both the same heat pump).
    ideal = targets.ideal
    assert targets.network_gap == pytest.approx(
        (17518.7 - direct.hot_utility_target) * ideal.hot_utility_price
        + (13603.6 + 1665.1 - direct.cold_utility_target) * ideal.cold_utility_price
    )
    assert targets.heat_pump_gap == pytest.approx(0.0, abs=1e-6)


def test_each_target_is_priced_at_the_utilities_and_hpr_electricity(monkeypatch):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        annual_cost_scales,
    )

    monkeypatch.setattr(
        heat_pump_targets,
        "compute_profile_heat_pump_target",
        lambda **_kwargs: SimpleNamespace(
            Q_cond=np.array([1000.0]),
            T_cond=np.array([270.0]),
            Q_evap=np.array([700.0]),
            T_evap=np.array([90.0]),
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            w_net=300.0,
            cop_h=1000.0 / 300.0,
            hpr_total_annualized_cost=None,
        ),
    )
    problem = _problem()

    piped = heat_pump_targets.retrofit_heat_pumps(problem, _solved_network()).as_piped

    # The case's utilities ($/MWh) and the HPR electricity price, on the
    # retrofit's $/y basis (price times kW times operating hours / 1000).
    costing = problem.master_zone.config.costing
    scale, _capital = annual_cost_scales(costing)
    hot, cold = 40.68 * scale, 8.568 * scale
    electricity = float(costing.hpr_ele_price) * scale
    assert piped.operating_cost_before == pytest.approx(
        17518.7 * hot + (13603.6 + 1665.1) * cold
    )
    assert piped.operating_cost_after == pytest.approx(
        (17518.7 - 1000.0) * hot
        + (13603.6 + 1665.1 - 700.0) * cold
        + 300.0 * electricity
    )


def test_a_target_without_load_is_a_heat_pump_that_delivers_nothing(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    # The HPR service finds no load (as for a network with no heater).
    monkeypatch.setattr(
        heat_pump_targets, "compute_profile_heat_pump_target", lambda **_k: None
    )

    targets = heat_pump_targets.retrofit_heat_pumps(_problem(), _solved_network())

    piped = targets.as_piped
    assert (piped.heat_delivered, piped.work, piped.condenser_duties) == (0, 0, ())
    assert piped.operating_cost_after == pytest.approx(piped.operating_cost_before)
    assert targets.heat_pump_gap == 0.0


def test_both_targets_take_the_first_periods_load(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    # A period-valued HPR load is read at the network's first period.
    periods = []

    def fake_target(*, period_id=None, period_idx=None, **_kwargs):
        periods.append((period_id, period_idx))

    monkeypatch.setattr(
        heat_pump_targets, "compute_profile_heat_pump_target", fake_target
    )

    heat_pump_targets.retrofit_heat_pumps(_problem(), _solved_network())

    assert periods == [("0", 0)] * 2


def test_design_retrofit_heat_pumps_run_the_chosen_cycle(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )
    from OpenPinch.contracts.synthesis.retrofit import RetrofitHeatPumpTargets
    from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle

    seen = []

    def fake_target(*, T_vals, H_hot, H_cold, config, args=None, **_period):
        seen.append((config.hpr.type, config.hpr.load_fraction, args))

    monkeypatch.setattr(
        heat_pump_targets, "compute_profile_heat_pump_target", fake_target
    )
    problem = PinchProblem(json.loads(read_sample_case(CASE)))

    targets = problem.design.retrofit_heat_pumps(
        _solved_network(),
        cycle="carnot",
        topology="parallel",
        maximum_iterations=7,
        options={"HPR_LOAD_FRACTION": 0.5},
    )

    assert isinstance(targets, RetrofitHeatPumpTargets)
    parallel = HeatPumpAndRefrigerationCycle.ParallelCarnot
    assert [(str(t), f) for t, f, _ in seen] == [(parallel.value, 0.5)] * 2
    assert [args["maximum_iterations"] for _, _, args in seen] == [7, 7]


def test_design_retrofit_heat_pumps_target_the_first_period(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks import context
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    # The cached direct target must be the first period's, whose states the
    # as-piped profiles and prices read, not whichever ran last.
    requested = []

    def fake_targets(problem, runtime_options, period_id):
        requested.append((runtime_options.get("period_id"), period_id))

    monkeypatch.setattr(context, "ensure_target_results", fake_targets)
    monkeypatch.setattr(
        heat_pump_targets, "retrofit_heat_pumps", lambda *_a, **_k: "targets"
    )
    problem = PinchProblem(json.loads(read_sample_case(CASE)))

    problem.design.retrofit_heat_pumps(_solved_network())

    assert requested == [("0", "0")]


def test_an_unsolved_network_is_solved_as_the_retrofit_baseline(monkeypatch):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    # The existing network as design.retrofit takes it: structure and areas,
    # no temperatures. The retrofit search's depth zero solves it.
    unsolved = HeatExchangerNetwork.from_structure(
        recovery=[("H3", "C2", 1), ("H1", "C1", 1), ("H2", "C2", 2), ("H3", "C1", 2)],
        heaters=["C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )
    searched = []

    def fake_search(problem, existing, settings):
        searched.append((existing, settings.max_modifications))
        return SimpleNamespace(frontier=(SimpleNamespace(network=_solved_network()),))

    monkeypatch.setattr(heat_pump_targets, "run_retrofit_search", fake_search)
    monkeypatch.setattr(
        heat_pump_targets, "compute_profile_heat_pump_target", lambda **_k: None
    )

    targets = heat_pump_targets.retrofit_heat_pumps(_problem(), unsolved)

    assert searched == [(unsolved, 0)]
    assert targets.as_piped.hot_utility_before == pytest.approx(17518.7)


def test_a_bare_stream_name_two_zones_share_is_refused_by_the_targets():
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
        _process_stream,
    )

    streams = SimpleNamespace(
        items=lambda: [("A.H1", "first"), ("B.H1", "second"), ("A.C1", "cold")]
    )

    assert _process_stream(streams, "A.H1") == "first"
    assert _process_stream(streams, "C1") == "cold"
    with pytest.raises(ValueError, match="'H1'.*A.H1.*B.H1"):
        _process_stream(streams, "H1")


def test_a_target_is_priced_at_the_utilities_the_network_uses(monkeypatch):
    """A dearer hot utility listed first is not the one a heat pump saves:
    each side is priced at the duty-weighted utilities its heaters (or
    coolers) use, not at the first listed."""
    import copy

    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        annual_cost_scales,
    )

    monkeypatch.setattr(
        heat_pump_targets, "compute_profile_heat_pump_target", lambda **_k: None
    )
    source = json.loads(read_sample_case(CASE))
    hot = next(u for u in source["utilities"] if u["name"] == "HU")
    dear = copy.deepcopy(hot) | {"name": "HU dear"}
    dear["price"] = {"value": 500.0, "unit": hot["price"]["unit"]}
    source["utilities"].insert(0, dear)
    problem = PinchProblem(source)
    ensure_target_results(problem, {}, None)

    piped = heat_pump_targets.retrofit_heat_pumps(problem, _solved_network()).as_piped

    scale, _capital = annual_cost_scales(problem.master_zone.config.costing)
    assert piped.hot_utility_price == pytest.approx(40.68 * scale)
