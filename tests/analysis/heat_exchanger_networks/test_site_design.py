"""Site seeds: each process's pinch design networks combined into one.

A site network holds each process's utility levels and matches streams only
within a process. Its seeds take one network per process at each dt_cont
multiplier, with each process's level duties.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.site import (
    process_level_duties,
    site_seeds,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import HeatExchangerKind, StreamID
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _state(duty: float) -> HeatExchangerPeriodState:
    return HeatExchangerPeriodState(period_id="0", period_idx=0, duty=duty)


def _recovery(hot: str, cold: str, stage: int, duty: float = 5.0) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"recovery:{hot}->{cold}:S{stage}",
        kind=HeatExchangerKind.RECOVERY,
        source_stream=hot,
        sink_stream=cold,
        source_stream_role=StreamID.Process,
        sink_stream_role=StreamID.Process,
        stage=stage,
        period_states=(_state(duty),),
    )


def _heater(utility: str, cold: str, duty: float) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"hot-utility:{utility}->{cold}",
        kind=HeatExchangerKind.HOT_UTILITY,
        source_stream=utility,
        sink_stream=cold,
        source_stream_role=StreamID.Utility,
        sink_stream_role=StreamID.Process,
        period_states=(_state(duty),),
    )


def _cooler(hot: str, utility: str, duty: float) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"cold-utility:{hot}->{utility}",
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream=hot,
        sink_stream=utility,
        source_stream_role=StreamID.Process,
        sink_stream_role=StreamID.Utility,
        period_states=(_state(duty),),
    )


def _network(exchangers, *, multiplier=1.0, stages=None, targets=None):
    return HeatExchangerNetwork(
        exchangers=tuple(exchangers),
        run_id="process",
        method="heuristic_pinch_design_method",
        stage_count=stages,
        summary_metrics={"dt_cont_multiplier": multiplier},
        source_metadata=({} if targets is None else {"utility_level_targets": targets}),
    )


LPS, HPS, CW = "Hot Utility.LPS", "Hot Utility.HPS", "Cold Utility.CW"


def test_a_level_held_network_gives_its_level_targets():
    targets = {LPS: [10.0], HPS: [0.0], CW: [4.0]}
    network = _network(
        [
            _recovery("A.H1", "A.C1", 1),
            _heater(LPS, "A.C1", 10.0005),
            _cooler("A.H1", CW, 4.0),
        ],
        stages=1,
        targets=targets,
    )

    assert process_level_duties(network) == targets


def test_a_network_without_targets_gives_its_own_utility_duties():
    network = _network(
        [
            _heater(LPS, "R.C1", 3.0),
            _heater(LPS, "R.C2", 2.0),
            _heater(HPS, "R.C2", 1.0),
        ]
    )

    assert process_level_duties(network) == {LPS: [5.0], HPS: [1.0]}


def test_a_level_held_network_whose_duties_drift_gives_its_duties():
    """A network that fell back to price-chosen levels does not hold the
    targets it records; its own duties are what a site seed can hold."""
    network = _network(
        [_recovery("A.H1", "A.C1", 1), _heater(HPS, "A.C1", 10.0)],
        stages=1,
        targets={LPS: [10.0], HPS: [0.0]},
    )

    assert process_level_duties(network) == {HPS: [10.0]}


def test_site_seeds_take_one_network_per_process_at_each_multiplier():
    a_fine = _network(
        [_recovery("A.H1", "A.C1", 2), _heater(LPS, "A.C1", 10.0)],
        multiplier=0.75,
        stages=3,
        targets={LPS: [10.0]},
    )
    a_nominal = _network(
        [_recovery("A.H1", "A.C1", 1), _heater(LPS, "A.C1", 12.0)],
        multiplier=1.0,
        stages=1,
        targets={LPS: [12.0]},
    )
    a_nominal_worse = _network(
        [_recovery("A.H2", "A.C1", 1), _heater(HPS, "A.C1", 30.0)],
        multiplier=1.0,
        stages=1,
    )
    b_nominal = _network(
        [_recovery("B.H1", "B.C1", 1), _cooler("B.H1", CW, 4.0)],
        multiplier=1.0,
        stages=2,
        targets={CW: [4.0]},
    )
    residential = _network([_heater(LPS, "R.C1", 3.0)], multiplier=1.0)

    seeds = site_seeds(
        {
            "A": (a_nominal, a_fine, a_nominal_worse),
            "B": (b_nominal,),
            "R": (residential,),
        }
    )

    assert [seed.summary_metrics["dt_cont_multiplier"] for seed in seeds] == [
        0.75,
        1.0,
    ]
    fine, nominal = seeds
    assert {e.exchanger_id for e in fine.exchangers} == {
        "recovery:A.H1->A.C1:S2",
        "recovery:B.H1->B.C1:S1",
    }
    assert fine.stage_count == 3
    assert fine.source_metadata["site_processes"] == {
        "A": {LPS: [10.0]},
        "B": {CW: [4.0]},
        "R": {LPS: [3.0]},
    }
    assert {e.exchanger_id for e in nominal.exchangers} == {
        "recovery:A.H1->A.C1:S1",
        "recovery:B.H1->B.C1:S1",
    }
    assert nominal.stage_count == 2
    assert nominal.source_metadata["site_processes"]["A"] == {LPS: [12.0]}


def test_site_seeds_need_some_heat_recovery():
    with pytest.raises(ValueError, match="no process network recovers heat"):
        site_seeds({"R": (_network([_heater(LPS, "R.C1", 3.0)]),)})


def _costed(network, *, units: int, tac: float):
    return network.model_copy(
        update={
            "total_annual_cost": tac,
            "summary_metrics": network.summary_metrics | {"total_units": units},
        }
    )


def test_site_seeds_come_from_the_cheapest_and_the_level_held_fewest_unit_networks():
    """Each process offers two pinch design runs: one with levels chosen by
    price, one holding its level targets. From each run a seed takes the
    network with the fewest units (then the lowest TAC), so at one
    multiplier there is a lowest-cost seed and a level-held seed."""
    cheap_few = _costed(
        _network([_recovery("A.H1", "A.C1", 1), _heater(HPS, "A.C1", 12.0)], stages=1),
        units=2,
        tac=900.0,
    )
    cheap_many = _costed(
        _network(
            [
                _recovery("A.H1", "A.C1", 1),
                _recovery("A.H2", "A.C1", 2),
                _heater(HPS, "A.C1", 8.0),
            ],
            stages=2,
        ),
        units=3,
        tac=800.0,
    )
    held_few = _costed(
        _network(
            [_recovery("A.H2", "A.C1", 1), _heater(LPS, "A.C1", 12.0)],
            stages=1,
            targets={LPS: [12.0], HPS: [0.0]},
        ),
        units=2,
        tac=1000.0,
    )
    held_missed = _costed(
        _network([_heater(HPS, "A.C1", 12.0)], targets={LPS: [12.0], HPS: [0.0]}),
        units=1,
        tac=700.0,
    )
    b = _costed(
        _network([_recovery("B.H1", "B.C1", 1), _cooler("B.H1", CW, 4.0)], stages=1),
        units=2,
        tac=100.0,
    )

    seeds = site_seeds(
        {"A": (cheap_many, cheap_few), "B": (b,)},
        level_held={"A": (held_missed, held_few)},
    )

    by_kind = {seed.summary_metrics["site_seed"]: seed for seed in seeds}
    assert set(by_kind) == {"lowest_cost", "level_held"}
    assert {e.exchanger_id for e in by_kind["lowest_cost"].exchangers} == {
        "recovery:A.H1->A.C1:S1",
        "recovery:B.H1->B.C1:S1",
    }
    assert {e.exchanger_id for e in by_kind["level_held"].exchangers} == {
        "recovery:A.H2->A.C1:S1",
        "recovery:B.H1->B.C1:S1",
    }
    assert by_kind["level_held"].source_metadata["site_processes"]["A"] == {
        LPS: [12.0],
        HPS: [0.0],
    }
    assert by_kind["lowest_cost"].source_metadata["site_processes"]["A"] == {
        HPS: [12.0]
    }
