"""Refrigerant screening through the problem target accessor."""

from __future__ import annotations

import inspect

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.refrigerant_screen import NATURAL_REFRIGERANTS
from OpenPinch.contracts.hpr_refrigerant_screen import (
    HprRefrigerantScreenResult,
    RefrigerantScreenQuantity,
)


def test_explicit_levels_screen_the_selected_fluids():
    problem = PinchProblem("basic_pinch.json", project_name="Site")

    # Paper Carnot levels, brewery utility placement at 25% load.
    result = problem.target.hpr_refrigerant_screen(
        T_evap=-0.048759,
        T_cond=84.934468,
        refrigerants="natural",
        options={"HPR_DT_CONT": 5.0},
    )

    assert isinstance(result, HprRefrigerantScreenResult)
    assert result.source == "explicit_levels"
    assert [row.fluid for row in result.practical] == [
        "Ammonia",
        "IsoButane",
        "n-Butane",
        "Neopentane",
    ]
    propane = next(row for row in result.rows if row.fluid == "n-Propane")
    assert propane.T_cond == RefrigerantScreenQuantity(value=84.934468, unit="degC")
    assert propane.p_evap.unit == "bar"
    assert propane.p_evap.value == pytest.approx(4.737515, rel=1e-4)
    assert propane.suction_vhc.unit == "kJ/m^3"
    assert result.dt_cont == RefrigerantScreenQuantity(value=5.0, unit="delta_degC")
    assert result.min_suction_pressure == RefrigerantScreenQuantity(
        value=0.5, unit="bar"
    )


def test_screen_has_provenance_and_leaves_the_problem_unchanged():
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    before = problem.to_problem_json()

    result = problem.target.hpr_refrigerant_screen(
        T_evap=[0.0, 10.0], T_cond=60.0, refrigerants=["Ammonia"]
    )

    assert result.provenance.method_id == "target.hpr_refrigerant_screen"
    assert result.provenance.zone_address == "Site"
    assert result.provenance.effective_settings["T_evap"] == [0.0, 10.0]
    assert result.provenance.effective_settings["refrigerants"] == ["Ammonia"]
    assert problem.to_problem_json() == before
    assert problem.results is None


@pytest.fixture(scope="module")
def carnot_heat_pump():
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    target = problem.target.carnot_heat_pump(
        load_fraction=0.25, condensers=1, evaporators=1, maximum_restarts=1
    )
    return problem, target


def test_a_carnot_target_supplies_the_levels(carnot_heat_pump):
    problem, target = carnot_heat_pump

    result = problem.target.hpr_refrigerant_screen(
        target=target, refrigerants="natural"
    )

    details = target.hpr_details
    assert result.source == "carnot_target"
    assert [q.value for q in result.T_evap_levels] == pytest.approx(
        list(details.T_evap)
    )
    assert [q.value for q in result.T_cond_levels] == pytest.approx(
        list(details.T_cond)
    )
    assert sorted(row.fluid for row in result.rows) == sorted(NATURAL_REFRIGERANTS)
    assert result.provenance.prerequisite_ids == (target.provenance.identity,)
    assert result.provenance.zone_address == target.provenance.zone_address
    assert result.provenance.period_ids == target.provenance.period_ids


def test_only_distinct_levels_that_carry_duty_are_screened():
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    target = problem.target.carnot_heat_pump(
        load_fraction=0.25,
        is_cascade_cycle=False,
        condensers=2,
        evaporators=2,
        maximum_restarts=1,
    )
    details = target.hpr_details
    assert 0.0 in details.Q_cond

    result = problem.target.hpr_refrigerant_screen(
        target=target, refrigerants=["Water"]
    )

    def placed(levels, duties):
        return list(dict.fromkeys(t for t, q in zip(levels, duties) if q > 0.0))

    assert [q.value for q in result.T_cond_levels] == pytest.approx(
        placed(details.T_cond, details.Q_cond)
    )
    assert [q.value for q in result.T_evap_levels] == pytest.approx(
        placed(details.T_evap, details.Q_evap)
    )


def test_levels_come_from_exactly_one_successful_carnot_source(carnot_heat_pump):
    problem, target = carnot_heat_pump
    screen = problem.target.hpr_refrigerant_screen
    message = "either a Carnot target or both T_evap and T_cond"

    with pytest.raises(ValueError, match=message):
        screen()
    with pytest.raises(ValueError, match=message):
        screen(T_evap=0.0)
    with pytest.raises(ValueError, match=message):
        screen(target=target, T_cond=60.0)
    with pytest.raises(TypeError, match="Carnot heat pump or refrigeration target"):
        screen(target=problem.target.direct_heat_integration())
    failed = target.model_copy(update={"hpr_success": False})
    with pytest.raises(ValueError, match="successful"):
        screen(target=failed)


def test_explicit_levels_accept_units():
    problem = PinchProblem("basic_pinch.json", project_name="Site")

    result = problem.target.hpr_refrigerant_screen(
        T_evap={"value": 273.15, "unit": "K"},
        T_cond=[{"value": 140.0, "unit": "degF"}],
        refrigerants=["Ammonia"],
    )

    assert result.T_evap_levels[0].value == pytest.approx(0.0, abs=1e-9)
    assert result.T_cond_levels[0].value == pytest.approx(60.0)


def test_signature_is_keyword_only_and_only_the_active_case_mirrors_it():
    from OpenPinch import PinchWorkspace

    method = PinchProblem().target.hpr_refrigerant_screen
    parameters = inspect.signature(method).parameters
    assert tuple(parameters) == (
        "target",
        "T_evap",
        "T_cond",
        "refrigerants",
        "options",
    )
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())
    assert all(p.default is None for p in parameters.values())

    workspace = PinchWorkspace("basic_pinch.json", project_name="Site")
    result = workspace.target.hpr_refrigerant_screen(
        T_evap=0.0, T_cond=60.0, refrigerants=["Ammonia"]
    )
    assert result.rows[0].fluid == "Ammonia"
    # A screen takes one target or explicit levels and has no period or case
    # dependence to mirror.
    assert not hasattr(workspace.target.all_periods, "hpr_refrigerant_screen")
    assert not hasattr(workspace.cases().target, "hpr_refrigerant_screen")
