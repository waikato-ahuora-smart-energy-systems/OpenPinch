"""An as-piped heat pump's saving is priced at the utilities it relieves.

C1 is heated from 60 to 110 degC on LPS (20 $/MWh) and C2 from 150 to
200 degC on HPS (40 $/MWh). A heat pump condensing near 116 degC can
reach C1's heater alone, yet its saving was priced at the duty-weighted
blend of both heaters (30 $/MWh): 175,197 $/y reported where the heater
it serves gives 95,197. The heat it delivers is now priced at the heaters
it can reach, the dearest relieved first, and likewise its cooling.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState

_KELVIN = 273.15


def _q(value, unit):
    return {"value": value, "unit": unit}


def _stream(name, supply, target, duty):
    return {
        "zone": "Site/A",
        "name": name,
        "t_supply": _q(supply, "degC"),
        "t_target": _q(target, "degC"),
        "heat_flow": _q(duty, "kW"),
        "dt_cont": _q(5.0, "delta_degC"),
        "htc": _q(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, supply, target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _q(supply, "degC"),
        "t_target": _q(target, "degC"),
        "dt_cont": _q(5.0, "delta_degC"),
        "price": _q(price, "$/MWh"),
        "htc": _q(1.0, "kW/m^2/delta_degC"),
    }


def _problem() -> PinchProblem:
    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )

    problem = PinchProblem(
        source={
            "streams": [
                _stream("C1", 60.0, 110.0, 1000.0),
                _stream("C2", 150.0, 200.0, 1000.0),
                _stream("H1", 90.0, 40.0, 1500.0),
                _stream("H2", 35.0, 25.0, 200.0),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.0, 40.0),
                _utility("LPS", "Hot", 150.0, 149.0, 20.0),
                _utility("CW", "Cold", 10.0, 15.0, 5.0),
                _utility("ChW", "Cold", 2.0, 6.0, 30.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "A", "type": "Process Zone"}],
            },
        }
    )
    ensure_target_results(problem, {}, None)
    return problem


def _network(problem):
    """Heaters on C1 (LPS) and C2 (HPS), coolers on H1 (CW) and H2 (ChW),
    each over its stream's whole range, solved."""
    network = problem.design._network_from_structure(
        {
            "recovery": [],
            "heaters": [
                {"stream": "C1", "utility": "LPS", "area": 30.0, "id": "HT1"},
                {"stream": "C2", "utility": "HPS", "area": 60.0, "id": "HT2"},
            ],
            "coolers": [
                {"stream": "H1", "utility": "CW", "area": 200.0, "id": "CL1"},
                {"stream": "H2", "utility": "ChW", "area": 60.0, "id": "CL2"},
            ],
        }
    )
    # (duty, process inlet, process outlet), degC
    process = {
        "HT1": (1000.0, 60.0, 110.0),
        "HT2": (1000.0, 150.0, 200.0),
        "CL1": (1500.0, 90.0, 40.0),
        "CL2": (200.0, 35.0, 25.0),
    }

    def solved(exchanger):
        duty, inlet, outlet = process[exchanger.exchanger_id]
        heater = exchanger.kind.name == "HOT_UTILITY"
        utility = (250.0, 249.0) if heater else (10.0, 15.0)
        source = utility if heater else (inlet, outlet)
        sink = (inlet, outlet) if heater else utility
        state = HeatExchangerPeriodState(
            period_id="0",
            period_idx=0,
            duty=duty,
            source_inlet_temperature=source[0] + _KELVIN,
            source_outlet_temperature=source[1] + _KELVIN,
            sink_inlet_temperature=sink[0] + _KELVIN,
            sink_outlet_temperature=sink[1] + _KELVIN,
        )
        return exchanger.model_copy(update={"period_states": (state,)})

    return network.model_copy(
        update={"exchangers": tuple(solved(e) for e in network.exchangers)}
    )


def test_the_as_piped_saving_is_priced_at_the_utilities_it_relieves(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    # A machine condensing at 116 degC, delivering 1,000 kW, its
    # evaporator at 30 degC drawing 800 kW from H1.
    result = SimpleNamespace(
        Q_cond=np.array([1000.0]),
        Q_evap=np.array([800.0]),
        Q_heat=None,
        Q_cool=None,
        Q_amb_hot=0.0,
        w_net=200.0,
        cop_h=5.0,
        T_cond=np.array([116.0]),
        T_evap=np.array([30.0]),
        hpr_total_annualized_cost=None,
    )
    monkeypatch.setattr(
        heat_pump_targets,
        "compute_profile_heat_pump_target",
        lambda **_arguments: result,
    )
    problem = _problem()

    piped = heat_pump_targets.retrofit_heat_pumps(problem, _network(problem)).as_piped

    # The prices before are the heaters' and coolers' duty-weighted blends:
    # (1000 x 20 + 1000 x 40) / 2000 and (1500 x 5 + 200 x 30) / 1700 $/MWh.
    # The heat relieves LPS alone (C2 starts above the condenser) and the
    # cooling CW alone (H2 ends below the evaporator).
    lps = piped.hot_utility_price * 20.0 / 30.0
    cw = piped.cold_utility_price * 5.0 / ((1500.0 * 5.0 + 200.0 * 30.0) / 1700.0)
    saving = piped.operating_cost_before - piped.operating_cost_after
    assert saving == pytest.approx(
        1000.0 * lps + 800.0 * cw - 200.0 * piped.electricity_price
    )


def test_the_as_piped_machine_is_sized_at_the_utilities_it_relieves(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    asked = []

    def target(**arguments):
        asked.append(arguments["config"].costing.hpr_price_ratio_heat_to_ele)
        return SimpleNamespace(
            Q_cond=np.array([1000.0]),
            Q_evap=np.array([800.0]),
            Q_heat=None,
            Q_cool=None,
            Q_amb_hot=0.0,
            w_net=200.0,
            cop_h=5.0,
            T_cond=np.array([116.0]),
            T_evap=np.array([30.0]),
            hpr_total_annualized_cost=None,
        )

    monkeypatch.setattr(heat_pump_targets, "compute_profile_heat_pump_target", target)
    problem = _problem()

    heat_pump_targets.retrofit_heat_pumps(problem, _network(problem))

    # Heat at the heaters' blend first (30 of 100 $/MWh), then the as-piped
    # machine again at LPS, the heater it relieves (20 of 100).
    assert asked[-1] == pytest.approx(0.2)


def test_the_retrofits_heat_pump_step_is_sized_at_the_utilities_it_relieves(
    monkeypatch,
):
    # The retrofit's heat pump step builds its machine from as_piped_heat_pump.
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    asked = []

    def target(**arguments):
        asked.append(arguments["config"].costing.hpr_price_ratio_heat_to_ele)
        return SimpleNamespace(
            Q_cond=np.array([1000.0]),
            Q_evap=np.array([800.0]),
            Q_heat=None,
            Q_cool=None,
            Q_amb_hot=0.0,
            w_net=200.0,
            cop_h=5.0,
            T_cond=np.array([116.0]),
            T_evap=np.array([30.0]),
            hpr_total_annualized_cost=None,
        )

    monkeypatch.setattr(heat_pump_targets, "compute_profile_heat_pump_target", target)
    problem = _problem()

    heat_pump_targets.as_piped_heat_pump(problem, _network(problem))

    assert asked == pytest.approx([0.3, 0.2])


def test_at_a_given_approach_the_heaters_reach_is_taken_unshifted(monkeypatch):
    # A retrofit at min_approach_temperature 10 K targets the coolers and
    # heaters unshifted, with the heat pump 10 K off them. A condenser at
    # 158 degC then reaches C1 alone (C2 starts at 150, above 148); on the
    # contributions it would reach C2's bottom 2 K too, and price 40 of its
    # 1,000 kW at HPS.
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )

    asked = []

    def target(**arguments):
        asked.append(arguments["config"].costing.hpr_price_ratio_heat_to_ele)
        return SimpleNamespace(
            Q_cond=np.array([1000.0]),
            Q_evap=np.array([800.0]),
            Q_heat=None,
            Q_cool=None,
            Q_amb_hot=0.0,
            w_net=200.0,
            cop_h=5.0,
            T_cond=np.array([158.0]),
            T_evap=np.array([30.0]),
            hpr_total_annualized_cost=None,
        )

    monkeypatch.setattr(heat_pump_targets, "compute_profile_heat_pump_target", target)
    problem = _problem()

    heat_pump_targets.as_piped_heat_pump(problem, _network(problem), approach=10.0)

    assert asked[-1] == pytest.approx(0.2)
