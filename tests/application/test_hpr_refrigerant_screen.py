"""Refrigerant screening through the problem target accessor."""

from __future__ import annotations

import inspect

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.refrigerant_screen import NATURAL_REFRIGERANTS
from OpenPinch.contracts.hpr import HPRTopologyIdentifier
from OpenPinch.contracts.hpr_refrigerant_screen import (
    HprRefrigerantScreenResult,
    RefrigerantScreenQuantity,
)


def test_explicit_levels_screen_every_layout_of_the_topology():
    problem = PinchProblem("basic_pinch.json", project_name="Site")

    result = problem.target.hpr_refrigerant_screen(
        T_evap=50.0,
        T_cond=[90.0, 180.0, 120.0],
        topology="cascade",
        refrigerants="natural",
        options={"HPR_DT_CONT": 5.0, "HPR_DT_CASCADE_HX": 5.0},
    )

    assert isinstance(result, HprRefrigerantScreenResult)
    assert result.source == "explicit_levels"
    assert result.topology == "cascade"
    assert result.is_heat_pumping
    assert [q.value for q in result.T_cond_levels] == [180.0, 120.0, 90.0]
    assert [layout.n_fluid_sets for layout in result.layouts] == [
        1,
        6,
        7,
        17,
        13,
        34,
        81,
        6,
        7,
        7,
        27,
        7,
    ]
    top = result.layouts[3].stages[0]
    assert top.T_evap == RefrigerantScreenQuantity(value=115.0, unit="degC")
    assert set(top.feasible) == {"Water", "n-Pentane", "Isopentane"}
    water = next(row for row in top.rows if row.fluid == "Water")
    assert water.p_evap.unit == "bar"
    assert water.p_evap.value == pytest.approx(1.69, rel=1e-2)
    assert water.suction_vhc.unit == "kJ/m^3"
    assert result.dt_cont == RefrigerantScreenQuantity(value=5.0, unit="delta_degC")
    assert result.dt_cascade_hx == RefrigerantScreenQuantity(
        value=5.0, unit="delta_degC"
    )
    assert result.critical_margin == RefrigerantScreenQuantity(
        value=5.0, unit="delta_degC"
    )
    assert result.min_suction_pressure == RefrigerantScreenQuantity(
        value=0.5, unit="bar"
    )


def test_explicit_refrigeration_levels_have_no_mvr_layouts():
    problem = PinchProblem("basic_pinch.json", project_name="Site")

    result = problem.target.hpr_refrigerant_screen(
        T_evap=50.0,
        T_cond=[180.0, 120.0],
        topology="cascade",
        is_refrigeration=True,
        refrigerants="natural",
    )

    assert not result.is_heat_pumping
    assert all(
        layout.method is not HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR
        for layout in result.layouts
    )


def test_screen_has_provenance_and_leaves_the_problem_unchanged():
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    before = problem.to_problem_json()

    result = problem.target.hpr_refrigerant_screen(
        T_evap=[0.0, 10.0], T_cond=60.0, topology="parallel", refrigerants=["Ammonia"]
    )

    assert result.provenance.effective_settings["topology"] == "parallel"
    assert result.provenance.effective_settings["is_heat_pumping"] is True
    assert result.provenance.method_id == "target.hpr_refrigerant_screen"
    assert result.provenance.zone_address == "Site"
    assert result.provenance.effective_settings["T_evap"] == [10.0, 0.0]
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
    assert result.topology == "cascade"
    assert result.is_heat_pumping
    assert [q.value for q in result.T_evap_levels] == pytest.approx(
        list(details.T_evap)
    )
    assert [q.value for q in result.T_cond_levels] == pytest.approx(
        list(details.T_cond)
    )
    (single,) = result.layouts
    assert single.method is HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION
    rows = single.stages[0].rows
    assert sorted(row.fluid for row in rows) == sorted(NATURAL_REFRIGERANTS)
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
        kept = dict.fromkeys(t for t, q in zip(levels, duties) if q > 0.0)
        return sorted(kept, reverse=True)

    assert result.topology == "parallel"

    assert [q.value for q in result.T_cond_levels] == pytest.approx(
        placed(details.T_cond, details.Q_cond)
    )
    assert [q.value for q in result.T_evap_levels] == pytest.approx(
        placed(details.T_evap, details.Q_evap)
    )


def test_a_carnot_refrigeration_target_screens_refrigeration_layouts():
    problem = PinchProblem("basic_pinch.json", project_name="Site")
    target = problem.target.carnot_refrigeration(
        load_fraction=0.25, condensers=1, evaporators=1, maximum_restarts=1
    )

    result = problem.target.hpr_refrigerant_screen(
        target=target, refrigerants=["Ammonia"]
    )

    assert result.topology == "cascade"
    assert not result.is_heat_pumping


def test_the_topology_comes_from_the_target_or_is_named(carnot_heat_pump):
    problem, target = carnot_heat_pump
    screen = problem.target.hpr_refrigerant_screen

    with pytest.raises(ValueError, match="topology 'parallel' or 'cascade'"):
        screen(T_evap=0.0, T_cond=60.0)
    with pytest.raises(ValueError, match="'parallel' or 'cascade'"):
        screen(T_evap=0.0, T_cond=60.0, topology="series")
    with pytest.raises(ValueError, match="come from the Carnot target"):
        screen(target=target, topology="cascade")
    with pytest.raises(ValueError, match="come from the Carnot target"):
        screen(target=target, is_refrigeration=False)


def test_levels_come_from_exactly_one_successful_carnot_source(carnot_heat_pump):
    problem, target = carnot_heat_pump
    screen = problem.target.hpr_refrigerant_screen
    message = "either a Carnot target or both T_evap and T_cond"

    with pytest.raises(ValueError, match=message):
        screen()
    with pytest.raises(ValueError, match=message):
        screen(T_evap=0.0, topology="cascade")
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
        topology="parallel",
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
        "topology",
        "is_refrigeration",
        "refrigerants",
        "options",
    )
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())
    assert all(p.default is None for p in parameters.values())

    workspace = PinchWorkspace("basic_pinch.json", project_name="Site")
    result = workspace.target.hpr_refrigerant_screen(
        T_evap=0.0, T_cond=60.0, topology="parallel", refrigerants=["Ammonia"]
    )
    assert result.layouts[0].stages[0].rows[0].fluid == "Ammonia"
    # A screen takes one target or explicit levels and has no period or case
    # dependence to mirror.
    assert not hasattr(workspace.target.all_periods, "hpr_refrigerant_screen")
    assert not hasattr(workspace.cases().target, "hpr_refrigerant_screen")


def test_foreign_stale_or_superseded_targets_are_rejected(carnot_heat_pump):
    _, target = carnot_heat_pump
    message = "current Carnot target of this problem"
    other = PinchProblem("basic_pinch.json", project_name="Site")
    with pytest.raises(ValueError, match=message):
        other.target.hpr_refrigerant_screen(target=target)

    problem = PinchProblem("basic_pinch.json", project_name="Site")
    kwargs = dict(condensers=1, evaporators=1, maximum_restarts=1)
    first = problem.target.carnot_heat_pump(load_fraction=0.25, **kwargs)
    problem.target.carnot_heat_pump(load_fraction=0.5, **kwargs)
    with pytest.raises(ValueError, match=message):
        problem.target.hpr_refrigerant_screen(target=first)

    current = problem.target.carnot_heat_pump(load_fraction=0.25, **kwargs)
    problem.update_options({"ENV_TEMPERATURE": 30.0})
    with pytest.raises(ValueError, match=message):
        problem.target.hpr_refrigerant_screen(target=current)
