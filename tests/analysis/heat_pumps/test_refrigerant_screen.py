"""Single-stage refrigerant screening between Carnot-cycle levels."""

from __future__ import annotations

import CoolProp.CoolProp as CP
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.refrigerant_screen import (
    NATURAL_REFRIGERANTS,
    RefrigerantScreenThresholds,
    resolve_refrigerants,
    screen_refrigerant,
    screen_refrigerants,
)
from OpenPinch.contracts.hpr_refrigerant_screen import RefrigerantScreenVerdict as V
from OpenPinch.domain.configuration import Configuration

# The paper runs used a 5 K working-fluid approach (HPR_DT_CONT).
PAPER = RefrigerantScreenThresholds(dt_cont=5.0)


def _verdicts(entries):
    return {entry.fluid: entry.verdict for entry in entries}


def test_brewery_refrigeration_to_hot_water_keeps_propane_under_the_ratio_limit():
    # Paper Carnot levels, brewery utility placement at 25% load: chilled
    # water at 5 C and heat rejected to 80 C hot water.
    entries = screen_refrigerants(
        (-0.048759,), (84.934468,), NATURAL_REFRIGERANTS, PAPER
    )
    propane = next(e for e in entries if e.fluid == "n-Propane")
    assert propane.pressure_ratio == pytest.approx(7.244225, rel=1e-4)
    assert _verdicts(entries) == {
        "Water": V.BELOW_TRIPLE_POINT,
        "Ammonia": V.PRACTICAL,
        "CarbonDioxide": V.ABOVE_CRITICAL,
        "n-Propane": V.NEAR_CRITICAL,
        "n-Butane": V.PRACTICAL,
        "IsoButane": V.PRACTICAL,
        "n-Pentane": V.OVER_RATIO_LIMIT,
        "Isopentane": V.DEEP_VACUUM_SUCTION,
        "Neopentane": V.PRACTICAL,
    }


def test_ethylbenzene_direct_tenth_load_ranks_practical_fluids_by_suction_vhc():
    # Paper Carnot levels, ethylbenzene direct placement at 10% load.
    entries = screen_refrigerants(
        (110.969873,), (179.377425,), NATURAL_REFRIGERANTS, PAPER
    )
    assert _verdicts(entries)["Isopentane"] is V.NEAR_CRITICAL
    assert [(e.fluid, e.rank) for e in entries[:3]] == [
        ("n-Pentane", 1),
        ("Water", 2),
        ("Ammonia", None),
    ]
    assert entries[1].pressure_ratio == pytest.approx(6.674179, rel=1e-4)
    assert entries[1].suction_vhc == pytest.approx(1898.403888, rel=1e-4)


def test_pairs_without_lift_are_skipped_and_no_pair_raises():
    entries = screen_refrigerants((0.0, 50.0), (60.0, 40.0), ("Ammonia",), PAPER)
    assert [(e.T_evap, e.T_cond) for e in entries] == [
        (0.0, 60.0),
        (0.0, 40.0),
        (50.0, 60.0),
    ]
    with pytest.raises(ValueError, match="no evaporating level below"):
        screen_refrigerants((60.0,), (40.0,), ("Ammonia",), PAPER)


def test_resolve_refrigerants_takes_groups_or_names():
    everything = tuple(CP.get_global_param_string("FluidsList").split(","))
    assert resolve_refrigerants(None) == everything
    assert resolve_refrigerants("all") == everything
    assert resolve_refrigerants("natural") == NATURAL_REFRIGERANTS
    assert resolve_refrigerants(["R290", "water"]) == ("R290", "water")
    with pytest.raises(ValueError, match="Invalid CoolProp fluid_name"):
        resolve_refrigerants(["not-a-fluid"])
    with pytest.raises(ValueError, match="'natural'"):
        resolve_refrigerants("hfo")
    with pytest.raises(ValueError, match="at least one"):
        resolve_refrigerants([])


def test_resolve_refrigerants_rejects_mixtures():
    with pytest.raises(ValueError, match="pure fluids"):
        resolve_refrigerants(["Water", "HEOS::R32[0.5]&R125[0.5]"])


@seed(20260715)
@settings(max_examples=200)
@given(
    fluid=st.sampled_from(resolve_refrigerants(None)),
    T_evap=st.floats(min_value=-150.0, max_value=300.0),
    lift=st.floats(min_value=0.5, max_value=200.0),
)
def test_every_coolprop_fluid_screens_and_its_verdict_matches_its_properties(
    fluid, T_evap, lift
):
    entry = screen_refrigerant(fluid, T_evap, T_evap + lift, PAPER)
    if entry.verdict in (V.ABOVE_CRITICAL, V.BELOW_TRIPLE_POINT):
        assert entry.pressure_ratio is entry.suction_vhc is None
        return
    assert entry.pressure_ratio == pytest.approx(entry.p_cond / entry.p_evap)
    assert entry.pressure_ratio > 1.0
    assert entry.suction_vhc > 0.0
    over_ratio = entry.pressure_ratio > PAPER.max_pressure_ratio
    assert (entry.verdict is V.OVER_RATIO_LIMIT) == over_ratio
    if entry.verdict is V.PRACTICAL:
        assert entry.p_evap >= PAPER.min_suction_pressure
        assert entry.T_crit - entry.T_cond >= PAPER.critical_margin


def test_thresholds_come_from_the_configuration():
    assert RefrigerantScreenThresholds.from_configuration(
        Configuration()
    ) == RefrigerantScreenThresholds()
    config = Configuration(
        options={
            "HPR_DT_CONT": 5.0,
            "HPR_MAX_PRESSURE_RATIO": 12.0,
            "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.3,
            "HPR_SCREEN_CRITICAL_MARGIN": 10.0,
        }
    )
    assert RefrigerantScreenThresholds.from_configuration(
        config
    ) == RefrigerantScreenThresholds(
        dt_cont=5.0,
        max_pressure_ratio=12.0,
        min_suction_pressure=0.3e5,
        critical_margin=10.0,
    )
