"""The heat pump and refrigeration cycle registry.

Each cycle is described once: its topologies and the model family each one
selects, the duties it serves, its own options and how they map onto
``HPR_*`` configuration keys, its dependencies and its screening methods.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_pumps.cycles.registry import (
    cycle_spec,
    registered_cycles,
    spec_for_cycle,
)
from OpenPinch.contracts.hpr import HPRTopologyIdentifier
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle as Cycle


def test_todays_cycles_are_registered():
    assert [spec.id for spec in registered_cycles()] == [
        "carnot",
        "vapour_compression",
        "mvr",
        "brayton",
    ]


def test_every_model_family_belongs_to_exactly_one_cycle_and_topology():
    owners = {
        family: [
            (spec.id, topology)
            for spec in registered_cycles()
            for topology, member in spec.topologies.items()
            if member is family
        ]
        for family in Cycle
    }

    assert owners == {
        Cycle.CascadeCarnot: [("carnot", "cascade")],
        Cycle.ParallelCarnot: [("carnot", "parallel")],
        Cycle.CascadeVapourComp: [("vapour_compression", "cascade")],
        Cycle.ParallelVapourComp: [("vapour_compression", "parallel")],
        Cycle.VapourCompMVR: [("mvr", "cascade")],
        Cycle.Brayton: [("brayton", "single")],
    }
    for family, ((cycle, topology),) in owners.items():
        assert spec_for_cycle(family) == (cycle_spec(cycle), topology)


def test_specs_say_what_each_cycle_supports():
    assert cycle_spec("mvr").duties == frozenset({"heat_pump"})
    assert cycle_spec("carnot").duties == frozenset({"heat_pump", "refrigeration"})
    assert not cycle_spec("brayton").available
    assert cycle_spec("carnot").default_topology == "cascade"
    assert cycle_spec("vapour_compression").screening_methods["parallel"] == (
        HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION,
        HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION,
    )
    assert cycle_spec("mvr").screening_methods["cascade"] == (
        HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
    )


def test_an_unknown_cycle_is_rejected():
    with pytest.raises(ValueError, match="unknown cycle 'stirling'"):
        cycle_spec("stirling")


def test_cycle_options_map_onto_their_configuration_keys():
    vapour = cycle_spec("vapour_compression")
    mvr = cycle_spec("mvr")

    assert vapour.to_configuration(
        vapour.options_model(refrigerants=["Ammonia"], sort_refrigerants=False)
    ) == {"HPR_REFRIGERANTS": ["Ammonia"], "HPR_REFRIGERANT_SORT_ENABLED": False}
    assert mvr.to_configuration(
        {"mvr_fluids": ["Water"], "mvr_stages": 2, "motor_efficiency": 0.95}
    ) == {"HPR_MVR_FLUIDS": ["Water"], "HPR_MVR_COUNT": 2, "HPR_MVR_ETA_MOTOR": 0.95}
    assert cycle_spec("carnot").to_configuration(None) == {}


def test_an_option_another_cycle_owns_is_rejected():
    with pytest.raises(ValueError, match="mvr_fluids"):
        cycle_spec("vapour_compression").to_configuration({"mvr_fluids": ["Water"]})


def test_a_cycle_registers_once_and_owns_its_model_families():
    from dataclasses import replace

    from OpenPinch.analysis.heat_pumps.cycles.registry import register_cycle

    carnot = cycle_spec("carnot")

    with pytest.raises(ValueError, match="already registered"):
        register_cycle(carnot)
    with pytest.raises(ValueError, match="reuses another cycle's model family"):
        register_cycle(replace(carnot, id="carnot_copy"))


def test_every_model_family_names_its_analysis_handlers():
    from OpenPinch.analysis.heat_pumps._multiperiod import setup
    from OpenPinch.analysis.heat_pumps.cycles.registry import (
        family_handlers,
        resolve_handler,
    )
    from OpenPinch.analysis.heat_pumps.targeting import (
        brayton,
        cascade_carnot,
        cascade_vapour_compression,
        parallel_carnot,
        parallel_vapour_compression,
        vapour_compression_mvr,
    )

    expected = {
        Cycle.CascadeCarnot: (
            cascade_carnot.optimise_cascade_carnot_heat_pump_placement,
            setup._cascade_carnot_setup,
            None,
        ),
        Cycle.ParallelCarnot: (
            parallel_carnot.optimise_parallel_carnot_heat_pump_placement,
            setup._parallel_carnot_setup,
            None,
        ),
        Cycle.CascadeVapourComp: (
            cascade_vapour_compression.optimise_cascade_heat_pump_placement,
            setup._cascade_vapour_compression_setup,
            HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION,
        ),
        Cycle.ParallelVapourComp: (
            parallel_vapour_compression.optimise_parallel_heat_pump_placement,
            setup._parallel_vapour_compression_setup,
            HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION,
        ),
        Cycle.VapourCompMVR: (
            vapour_compression_mvr.optimise_vapour_compression_mvr_heat_pump_placement,
            setup._vapour_compression_mvr_setup,
            HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR,
        ),
        Cycle.Brayton: (brayton.optimise_brayton_heat_pump_placement, None, None),
    }
    for family, (placement, multiperiod, topology) in expected.items():
        handlers = family_handlers(family.value)
        assert resolve_handler(handlers.placement) is placement
        assert resolve_handler(handlers.multiperiod_setup) is multiperiod
        assert handlers.topology_identifier is topology


def test_an_unregistered_model_family_has_no_handlers():
    from OpenPinch.analysis.heat_pumps.cycles.registry import family_handlers

    with pytest.raises(ValueError, match="no registered cycle"):
        family_handlers("Stirling cycle")
