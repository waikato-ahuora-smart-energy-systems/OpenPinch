"""Only net heating and cooling are charged (user rule, 2026-11-27).

Every rejected heat ends at ambient. The coldest cold utility that is not
refrigeration (its supply above ``ENV_TEMPERATURE``) rejects to ambient and
its price is the cooling price. Any other cold utility above ambient, steam
raising included, costs that price: raising steam is not a cost in itself,
and steam no one uses is vented or condensed on cooling water. Refrigeration
costs its own price (the work) plus the ambient price on the heat it takes
and that work, the work per unit of heat being its price over
``COSTING_HPR_ELE_PRICE``. Hot utilities keep their own price.
"""

from __future__ import annotations

import pytest

from OpenPinch.application.problem import PinchProblem


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _stream(name, t_supply, t_target, heat_flow):
    return {
        "zone": "Plant",
        "name": name,
        "t_supply": _quantity(t_supply, "degC"),
        "t_target": _quantity(t_target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(5.0, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, t_supply, t_target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(t_supply, "degC"),
        "t_target": _quantity(t_target, "degC"),
        "price": _quantity(price, "$/MWh"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _problem() -> PinchProblem:
    return PinchProblem(
        {
            "streams": [
                _stream("H1", 200.0, 0.0, 200.0),
                _stream("C1", 30.0, 160.0, 100.0),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("LPS", "Both", 140.0, 139.9, 30.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
                _utility("ChW", "Cold", 5.0, 10.0, 25.0),
            ],
            "options": {"ENV_TEMPERATURE": 15.0, "COSTING_HPR_ELE_PRICE": 100.0},
        }
    )


def _prices(utilities) -> dict[str, float]:
    return {u.name: float(u.price.to("$/MW/h").value) for u in utilities}


def test_cold_utilities_above_ambient_cost_the_ambient_price():
    root = _problem()._require_prepared_root_zone()

    cold = _prices(root.cold_utilities)

    assert cold["CW"] == pytest.approx(5.0)
    assert cold["LPS"] == pytest.approx(5.0)
    assert _prices(root.hot_utilities)["LPS"] == pytest.approx(30.0)


def test_refrigeration_costs_its_work_and_the_rejection_of_heat_and_work():
    cold = _prices(_problem()._require_prepared_root_zone().cold_utilities)

    # 25 for the work, and 5 on the heat plus 0.25 of work per unit of heat.
    assert cold["ChW"] == pytest.approx(25.0 + 5.0 * 1.25)


def test_with_none_above_ambient_the_warmest_cold_utility_rejects_to_it():
    """Water at 10 and 5 C reaches the process's cold end, so no default CU
    is added and no cold utility is above ambient (15 C). The warmest is
    taken as the one rejecting to ambient (cooling water as given, as in
    ``basic_pinch``, whose only cold utility is at 10 C): it keeps its
    price, and the colder one is refrigeration rejecting through it."""
    problem = PinchProblem(
        {
            "streams": [_stream("H1", 200.0, 40.0, 160.0)],
            "utilities": [
                _utility("CW", "Cold", 10.0, 15.0, 4.0),
                _utility("ChW", "Cold", 5.0, 10.0, 25.0),
            ],
            "options": {"ENV_TEMPERATURE": 15.0, "COSTING_HPR_ELE_PRICE": 100.0},
        }
    )

    cold = _prices(problem._require_prepared_root_zone().cold_utilities)

    assert set(cold) == {"CW", "ChW"}
    assert cold["CW"] == pytest.approx(4.0)
    assert cold["ChW"] == pytest.approx(25.0 + 4.0 * 1.25)


def test_a_capped_rejector_leaves_the_marginal_price_to_the_default_utility():
    """Cooling water is capped at 50 kW, so a default CU is added for the
    rest. Heat beyond the cap ends at ambient through the default CU, so its
    price, not cooling water's, is what other cold utilities cost; cooling
    water keeps its own (Codex review of #552)."""
    source = _problem().to_problem_json()
    for utility in source["utilities"]:
        if utility["name"] == "CW":
            utility["maximum_heat_flow"] = _quantity(50.0, "kW")
    problem = PinchProblem(source)

    cold = _prices(problem._require_prepared_root_zone().cold_utilities)

    assert cold["CW"] == pytest.approx(5.0)
    assert cold["CU"] == pytest.approx(100.0)
    assert cold["LPS"] == pytest.approx(100.0)


def test_a_residual_case_keeps_the_input_prices():
    """A residual case is written from a target's prepared utilities. Their
    prices include the rejection charge, and preparing the residual case
    charged it again: chilled water at 25 $/MWh came back at 37.5 and grew
    with each round trip (Codex review of #552)."""
    source = PinchProblem("basic_pinch.json").to_problem_json()
    source["utilities"].append(_utility("ChW", "Cold", 5.0, 10.0, 25.0))
    problem = PinchProblem(source)
    heat_pump = problem.target.carnot_heat_pump(
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=0,
        maximum_evaluations=1,
    )

    residual = problem.residual_utility(base_target=heat_pump)

    (chw,) = [u for u in residual.to_problem_json()["utilities"] if u["name"] == "ChW"]
    assert chw["price"]["value"] == pytest.approx(25.0)


def test_a_targets_provenance_names_the_utility_that_sets_the_ambient_price():
    """Nothing told a user which utility the costs took as rejecting heat to
    ambient; every result's provenance now names it, period by period."""
    target = _problem().target.direct_heat_integration()

    sink = target.provenance.effective_settings["AMBIENT_SINK"]

    assert sink == {"0": {"utility": "CW", "price": pytest.approx(5.0)}}


def test_a_case_with_no_cold_utility_above_ambient_is_warned():
    """With none at or above ENV_TEMPERATURE, the warmest cold utility is
    assumed to reject heat to ambient; the user is told which, and that
    ENV_TEMPERATURE decides it."""
    with pytest.warns(UserWarning, match=r"CW.*ENV_TEMPERATURE"):
        PinchProblem(
            {
                "streams": [_stream("H1", 200.0, 40.0, 160.0)],
                "utilities": [
                    _utility("CW", "Cold", 10.0, 15.0, 4.0),
                    _utility("ChW", "Cold", 5.0, 10.0, 25.0),
                ],
                "options": {"ENV_TEMPERATURE": 15.0},
            }
        )._require_prepared_root_zone()


def test_a_lone_cold_utility_below_ambient_is_not_warned():
    """``basic_pinch`` cools on one utility at 10 C, below the 15 C default
    ambient. Alone it keeps its own price, so the assumption prices nothing
    else and is not worth a warning on every use of the case."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PinchProblem("basic_pinch.json")._require_prepared_root_zone()

    assert not [w for w in caught if "ENV_TEMPERATURE" in str(w.message)]


def test_a_capped_ambient_sink_is_warned():
    """Cooling water capped at 50 kW leaves the default CU to set the price
    of heat rejected beyond the cap; the user is told so."""
    source = _problem().to_problem_json()
    for utility in source["utilities"]:
        if utility["name"] == "CW":
            utility["maximum_heat_flow"] = _quantity(50.0, "kW")

    with pytest.warns(UserWarning, match=r"CW.*capped.*CU"):
        PinchProblem(source)._require_prepared_root_zone()


def test_a_credit_on_steam_raising_that_is_replaced_is_warned():
    """A negative price on steam raising was a credit; it is now replaced by
    the ambient price (steam raised is worth something only where it is
    used), and the user is told."""
    source = _problem().to_problem_json()
    source["utilities"].append(_utility("LPG", "Cold", 120.0, 120.1, -10.0))

    with pytest.warns(UserWarning, match=r"LPG.*-10.*credit"):
        PinchProblem(source)._require_prepared_root_zone()


def test_the_ambient_rejector_keeps_its_own_segment_prices():
    from tests.analysis.heat_exchanger_networks.test_segmented_streams import (
        _segmented_problem,
    )

    root = _segmented_problem(segmented_cold_utility=True)
    (cu,) = root._require_prepared_root_zone().cold_utilities

    assert [float(s.price.to("$/MW/h").value) for s in cu.segments] == [30.0, 90.0]
