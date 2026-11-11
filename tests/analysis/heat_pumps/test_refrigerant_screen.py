"""Refrigerant screening of every layout of the Carnot-cycle levels."""

from __future__ import annotations

import CoolProp.CoolProp as CP
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.refrigerant_screen import (
    NATURAL_REFRIGERANTS,
    RefrigerantScreenThresholds,
    is_feasible,
    resolve_refrigerants,
    screen_layouts,
    screen_refrigerant,
)
from OpenPinch.contracts.hpr import HPRTopologyIdentifier
from OpenPinch.contracts.hpr_refrigerant_screen import RefrigerantScreenVerdict as V
from OpenPinch.domain.configuration import Configuration

# The paper runs used a 5 K working-fluid approach (HPR_DT_CONT).
PAPER = RefrigerantScreenThresholds(dt_cont=5.0)


def _verdicts(entries):
    return {entry.fluid: entry.verdict for entry in entries}


def test_brewery_refrigeration_to_hot_water_keeps_propane_under_the_ratio_limit():
    # Paper Carnot levels, brewery utility placement at 25% load: chilled
    # water at 5 C and heat rejected to 80 C hot water.
    entries = [
        screen_refrigerant(fluid, -0.048759, 84.934468, PAPER)
        for fluid in NATURAL_REFRIGERANTS
    ]
    propane = next(e for e in entries if e.fluid == "n-Propane")
    assert propane.pressure_ratio == pytest.approx(7.244225, rel=1e-4)
    assert _verdicts(entries) == {
        "Water": V.BELOW_TRIPLE_POINT,
        "Ammonia": V.PRACTICAL,
        "CarbonDioxide": V.ABOVE_CRITICAL,
        "n-Propane": V.PRACTICAL,
        "n-Butane": V.PRACTICAL,
        "IsoButane": V.PRACTICAL,
        "n-Pentane": V.OVER_RATIO_LIMIT,
        "Isopentane": V.DEEP_VACUUM_SUCTION,
        "Neopentane": V.PRACTICAL,
    }


def test_near_critical_needs_a_margin_wider_than_the_approach():
    wide = RefrigerantScreenThresholds(dt_cont=5.0, critical_margin=15.0)
    propane = screen_refrigerant("n-Propane", -0.048759, 84.934468, wide)
    assert propane.verdict is V.NEAR_CRITICAL
    assert is_feasible(V.NEAR_CRITICAL) and is_feasible(V.PRACTICAL)
    assert not any(
        is_feasible(v)
        for v in (
            V.ABOVE_CRITICAL,
            V.BELOW_TRIPLE_POINT,
            V.OVER_RATIO_LIMIT,
            V.DEEP_VACUUM_SUCTION,
        )
    )


# Carnot levels of the worked example: condensing at 180, 120 and 90 C,
# evaporating at 50 C, with a 5 K cascade exchanger approach.
EXAMPLE = RefrigerantScreenThresholds(dt_cont=5.0, dt_cascade_hx=5.0)
SC = HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION
PAR = HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION
CAS = HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION
MVR = HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR
TO_180 = {"Isopentane"}
TO_120 = {"Ammonia", "n-Butane", "IsoButane", "n-Pentane", "Isopentane", "Neopentane"}
TO_90 = TO_120 | {"n-Propane"}
CASCADE_TOP = {"Water", "n-Pentane", "Isopentane"}
CASCADE_MIDDLE = TO_120 | {"Water"}


def _example(topology, **kwargs):
    return screen_layouts(
        (50.0,),
        (90.0, 180.0, 120.0),
        NATURAL_REFRIGERANTS,
        EXAMPLE,
        topology=topology,
        **kwargs,
    )


def _shape(layout):
    return (
        layout.method,
        layout.T_cond_levels,
        tuple(
            (stage.kind, stage.T_evap, stage.T_cond, set(stage.feasible))
            for stage in layout.stages
        ),
    )


def _single_stages():
    return [
        (SC, (180.0,), (("vapour_compression", 50.0, 180.0, TO_180),)),
        (SC, (120.0,), (("vapour_compression", 50.0, 120.0, TO_120),)),
        (SC, (90.0,), (("vapour_compression", 50.0, 90.0, TO_90),)),
    ]


def _carnot_level_layouts(layouts):
    return [layout for layout in layouts if not layout.added_levels]


def test_parallel_screen_covers_every_subset_of_the_levels():
    layouts = _carnot_level_layouts(_example("parallel"))

    vc = "vapour_compression"
    assert [_shape(layout) for layout in layouts] == _single_stages() + [
        (PAR, (180.0, 120.0), ((vc, 50.0, 180.0, TO_180), (vc, 50.0, 120.0, TO_120))),
        (PAR, (180.0, 90.0), ((vc, 50.0, 180.0, TO_180), (vc, 50.0, 90.0, TO_90))),
        (PAR, (120.0, 90.0), ((vc, 50.0, 120.0, TO_120), (vc, 50.0, 90.0, TO_90))),
        (
            PAR,
            (180.0, 120.0, 90.0),
            (
                (vc, 50.0, 180.0, TO_180),
                (vc, 50.0, 120.0, TO_120),
                (vc, 50.0, 90.0, TO_90),
            ),
        ),
    ]
    assert [layout.n_fluid_sets for layout in layouts] == [1, 6, 7, 5, 6, 27, 20]
    assert all(layout.T_evap_levels == (50.0,) for layout in layouts)


def test_cascade_stages_evaporate_below_the_next_level_down():
    layouts = [
        layout
        for layout in _carnot_level_layouts(_example("cascade"))
        if layout.method is not MVR
    ]

    vc = "vapour_compression"
    assert [_shape(layout) for layout in layouts] == _single_stages() + [
        (
            CAS,
            (180.0, 120.0),
            ((vc, 115.0, 180.0, CASCADE_TOP), (vc, 50.0, 120.0, TO_120)),
        ),
        (
            CAS,
            (180.0, 90.0),
            ((vc, 85.0, 180.0, {"n-Pentane", "Isopentane"}), (vc, 50.0, 90.0, TO_90)),
        ),
        (
            CAS,
            (120.0, 90.0),
            ((vc, 85.0, 120.0, CASCADE_MIDDLE), (vc, 50.0, 90.0, TO_90)),
        ),
        (
            CAS,
            (180.0, 120.0, 90.0),
            (
                (vc, 115.0, 180.0, CASCADE_TOP),
                (vc, 85.0, 120.0, CASCADE_MIDDLE),
                (vc, 50.0, 90.0, TO_90),
            ),
        ),
    ]
    assert [layout.n_fluid_sets for layout in layouts] == [1, 6, 7, 17, 13, 34, 81]


def test_mvr_serves_the_top_levels_on_water_in_stages_of_at_most_20_k():
    layouts = [layout for layout in _example("cascade") if layout.method is MVR]

    def mvr(*bounds):
        # Hottest stage first, like the vapour compression stages.
        pairs = zip(bounds, bounds[1:])
        return tuple(("mvr", lo, hi, {"Water"}) for lo, hi in reversed(list(pairs)))

    vc = "vapour_compression"
    assert [_shape(layout) for layout in layouts] == [
        (
            MVR,
            (180.0, 120.0),
            mvr(115.0, 131.25, 147.5, 163.75, 180.0) + ((vc, 50.0, 120.0, TO_120),),
        ),
        (
            MVR,
            (180.0, 90.0),
            mvr(85.0, 104.0, 123.0, 142.0, 161.0, 180.0) + ((vc, 50.0, 90.0, TO_90),),
        ),
        (MVR, (120.0, 90.0), mvr(85.0, 102.5, 120.0) + ((vc, 50.0, 90.0, TO_90),)),
        (
            MVR,
            (180.0, 120.0, 90.0),
            mvr(115.0, 131.25, 147.5, 163.75, 180.0)
            + ((vc, 50.0, 120.0, TO_120), (vc, 50.0, 90.0, TO_90)),
        ),
        (
            MVR,
            (180.0, 120.0, 90.0),
            mvr(85.0, 102.5, 120.0, 140.0, 160.0, 180.0) + ((vc, 50.0, 90.0, TO_90),),
        ),
    ]
    assert [layout.n_fluid_sets for layout in layouts] == [6, 7, 7, 27, 7]


def test_an_mvr_stage_in_deep_vacuum_leaves_no_fluid_set():
    # Water evaporates at 0.39 bar at 75 C, below the 0.5 bar suction floor.
    (layout,) = [
        layout
        for layout in screen_layouts(
            (40.0,), (80.0, 95.0), ("Ammonia",), EXAMPLE, topology="cascade"
        )
        if layout.method is MVR
    ]
    assert layout.stages[0].feasible == ()
    assert layout.n_fluid_sets == 0
    assert layout.fluid_sets() == ()


def test_refrigeration_has_no_mvr_and_stacks_evaporating_levels():
    layouts = screen_layouts(
        (0.0, 20.0),
        (100.0,),
        ("Ammonia",),
        EXAMPLE,
        topology="cascade",
        is_heat_pumping=False,
    )

    layouts = _carnot_level_layouts(layouts)
    assert [(layout.method, layout.T_evap_levels) for layout in layouts] == [
        (SC, (20.0,)),
        (SC, (0.0,)),
        (CAS, (20.0, 0.0)),
    ]
    assert [(s.T_evap, s.T_cond) for s in layouts[-1].stages] == [
        (20.0, 100.0),
        (0.0, 25.0),
    ]
    parallel = screen_layouts(
        (0.0, 20.0),
        (100.0,),
        ("Ammonia",),
        EXAMPLE,
        topology="parallel",
        is_heat_pumping=False,
    )
    assert [
        (s.T_evap, s.T_cond) for s in _carnot_level_layouts(parallel)[-1].stages
    ] == [
        (20.0, 100.0),
        (0.0, 100.0),
    ]


def test_stage_fluids_are_ranked_by_suction_capacity_and_sets_expand_in_stage_order():
    layout = _example("cascade")[3]
    for stage in layout.stages:
        feasible = [e for e in stage.entries if e.rank is not None]
        assert [e.fluid for e in feasible] == list(stage.feasible)
        assert [e.rank for e in feasible] == list(range(1, len(feasible) + 1))
        vhc = [e.suction_vhc for e in feasible]
        assert vhc == sorted(vhc, reverse=True)
        assert all(not is_feasible(e.verdict) for e in stage.entries[len(feasible) :])

    sets = layout.fluid_sets()
    assert len(sets) == layout.n_fluid_sets == 17
    assert sets[0] == (layout.stages[0].feasible[0], layout.stages[1].feasible[0])


def test_fluid_sets_never_raise_the_critical_temperature_down_the_stages():
    # Targeting sorts a refrigerant list hottest stage first by critical
    # temperature, so a hotter stage never runs a lower-Tcrit fluid.
    for layout in _example("cascade") + _example("parallel"):
        sets = layout.fluid_sets()
        assert len(sets) == layout.n_fluid_sets
        for fluids in sets:
            T_crit = [CP.PropsSI("Tcrit", fluid) for fluid in fluids]
            assert T_crit == sorted(T_crit, reverse=True)
    middle_down = _example("cascade")[5]
    assert ("Ammonia", "Water") not in middle_down.fluid_sets()
    assert ("Water", "Ammonia") in middle_down.fluid_sets()
    assert ("Ammonia", "Ammonia") in middle_down.fluid_sets()


def test_stages_lifting_more_than_the_limit_are_also_screened_as_cascades():
    # 50 -> 180 C lifts 130 K: two stages of 67.5 K with the 5 K cascade gap.
    (parallel_split,) = [lay for lay in _example("parallel") if lay.added_levels]
    assert parallel_split.method is CAS
    assert parallel_split.added_levels == (117.5,)
    assert parallel_split.T_cond_levels == (180.0, 117.5)
    assert [(s.T_evap, s.T_cond) for s in parallel_split.stages] == [
        (112.5, 180.0),
        (50.0, 117.5),
    ]
    # 50 -> 120 C lifts exactly 70 K and is not split; the 85 -> 180 C top
    # stage of the 180 and 90 C cascade lifts 95 K and is.
    splits = [lay for lay in _example("cascade") if lay.added_levels]
    assert [(lay.T_cond_levels, lay.added_levels) for lay in splits] == [
        ((180.0, 117.5), (117.5,)),
        ((180.0, 135.0, 90.0), (135.0,)),
    ]
    assert [(s.T_evap, s.T_cond) for s in splits[1].stages] == [
        (130.0, 180.0),
        (85.0, 135.0),
        (50.0, 90.0),
    ]
    assert all(s.T_cond - s.T_evap <= 70.0 for lay in splits for s in lay.stages)


def test_refrigeration_splits_on_the_evaporating_side():
    layouts = screen_layouts(
        (-30.0,),
        (60.0,),
        ("Ammonia",),
        EXAMPLE,
        topology="cascade",
        is_heat_pumping=False,
    )

    (split,) = [layout for layout in layouts if layout.added_levels]
    assert split.T_evap_levels == (12.5, -30.0)
    assert split.added_levels == (12.5,)
    assert [(s.T_evap, s.T_cond) for s in split.stages] == [
        (12.5, 60.0),
        (-30.0, 17.5),
    ]


def test_a_wider_lift_limit_splits_nothing():
    wide = RefrigerantScreenThresholds(
        dt_cont=5.0, dt_cascade_hx=5.0, max_stage_lift=200.0
    )
    layouts = screen_layouts(
        (50.0,), (180.0,), ("Isopentane",), wide, topology="cascade"
    )
    assert [layout.added_levels for layout in layouts] == [()]


def test_layouts_without_lift_are_dropped_and_none_left_raises():
    layouts = screen_layouts(
        (50.0, 70.0), (60.0,), ("Ammonia",), EXAMPLE, topology="parallel"
    )
    assert [(layout.T_evap_levels, layout.T_cond_levels) for layout in layouts] == [
        ((50.0,), (60.0,))
    ]
    with pytest.raises(ValueError, match="no evaporating level below"):
        screen_layouts((60.0,), (40.0,), ("Ammonia",), EXAMPLE, topology="cascade")
    with pytest.raises(ValueError, match="lift limit above the cascade"):
        screen_layouts(
            (0.0,),
            (40.0,),
            ("Ammonia",),
            RefrigerantScreenThresholds(dt_cascade_hx=5.0, max_stage_lift=5.0),
            topology="cascade",
        )
    with pytest.raises(ValueError, match="'parallel' or 'cascade'"):
        screen_layouts((0.0,), (40.0,), ("Ammonia",), EXAMPLE, topology="series")


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
    assert RefrigerantScreenThresholds().critical_margin == 5.0
    assert (
        RefrigerantScreenThresholds.from_configuration(Configuration())
        == RefrigerantScreenThresholds()
    )
    config = Configuration(
        options={
            "HPR_DT_CONT": 5.0,
            "HPR_MAX_PRESSURE_RATIO": 12.0,
            "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.3,
            "HPR_SCREEN_CRITICAL_MARGIN": 10.0,
            "HPR_DT_CASCADE_HX": 5.0,
            "HPR_SCREEN_MAX_STAGE_LIFT": 60.0,
        }
    )
    assert RefrigerantScreenThresholds.from_configuration(
        config
    ) == RefrigerantScreenThresholds(
        dt_cont=5.0,
        max_pressure_ratio=12.0,
        min_suction_pressure=0.3e5,
        critical_margin=10.0,
        dt_cascade_hx=5.0,
        max_stage_lift=60.0,
    )
