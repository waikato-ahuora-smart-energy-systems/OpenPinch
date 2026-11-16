"""Retrofit modifications applied to a network structure (no solver)."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
    NewMatch,
    NewUtility,
    apply_new_utility,
    new_utility_candidates,
    apply_new_match,
    apply_relocation,
    match_potential,
    new_splits,
    screen_new_matches,
)
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork


def _network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1), ("H2", "C1", 2)],
        heaters=[("C1", "HU", 2)],
        coolers=["H1"],
        hot_utility="HU",
        cold_utility="CU",
    )


def _stages(network: HeatExchangerNetwork) -> dict[str, int | None]:
    return {e.exchanger_id: e.stage for e in network.exchangers}


def test_a_match_in_a_new_stage_shifts_the_later_stages():
    network = apply_new_match(
        _network(), NewMatch("H2", "C2", stage=2, new_stage=True), "N1"
    )

    assert _stages(network) == {
        "recovery:H1->C1:S1": 1,
        "recovery:H2->C1:S2": 3,
        # The heater after stage 2 follows its stage.
        "hot-utility:HU->C1:S2": 3,
        "cold-utility:H1->CU": None,
        "N1": 2,
    }
    assert network.stage_count == 3


def test_a_match_joining_a_stage_splits_streams_already_matched_there():
    network = _network()

    assert new_splits(network, NewMatch("H1", "C2", 1, new_stage=False)) == ("H1",)
    assert new_splits(network, NewMatch("H2", "C1", 1, new_stage=False)) == ("C1",)
    assert new_splits(network, NewMatch("H2", "C2", 1, new_stage=False)) == ()
    assert new_splits(network, NewMatch("H1", "C1", 2, new_stage=True)) == ()


def test_a_relocated_exchanger_keeps_its_id_and_installed_area():
    network = _network()
    network = network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"area": 50.0})
                if e.exchanger_id == "recovery:H1->C1:S1"
                else e
                for e in network.exchangers
            )
        }
    )

    moved = apply_relocation(
        network, "recovery:H1->C1:S1", NewMatch("H2", "C2", 2, new_stage=False)
    )

    exchanger = next(
        e for e in moved.exchangers if e.exchanger_id == "recovery:H1->C1:S1"
    )
    assert (exchanger.source_stream, exchanger.sink_stream, exchanger.stage) == (
        "H2",
        "C2",
        2,
    )
    assert exchanger.area == 50.0
    assert len(moved.exchangers) == len(network.exchangers)
    # The network keeps its two stages (from_structure leaves stage_count unset).
    assert moved.stage_count == 2



def test_a_new_utility_goes_on_a_stream_end_without_one():
    network = _network()

    # C1 has a heater after stage 2 but none at its end; H1 has an end cooler.
    candidates = new_utility_candidates(
        network,
        hot_streams=("H1", "H2"),
        cold_streams=("C1", "C2"),
        hot_utilities=("HU",),
        cold_utilities=("CU",),
    )
    assert candidates == (
        NewUtility("C1", "HU", heater=True),
        NewUtility("C2", "HU", heater=True),
        NewUtility("H2", "CU", heater=False),
    )

    added = apply_new_utility(network, NewUtility("H2", "CU", heater=False), "N1")
    (cooler,) = (e for e in added.exchangers if e.exchanger_id == "N1")
    assert (cooler.source_stream, cooler.sink_stream, cooler.stage) == (
        "H2",
        "CU",
        None,
    )
    assert cooler.area is None
    assert len(added.exchangers) == len(network.exchangers) + 1

# Stream supply, target (C) and heat capacity flowrate (kW/K).
ENDS = {
    "H1": (200.0, 100.0, 10.0),
    "H2": (150.0, 90.0, 5.0),
    "C1": (40.0, 180.0, 8.0),
    "C2": (50.0, 120.0, 4.0),
}


def _solved_network() -> HeatExchangerNetwork:
    """H1-C1 in stage 1 and H2-C1 in stage 2, end heaters and an H1 cooler."""

    from OpenPinch.domain._heat_exchanger.period_state import (
        HeatExchangerPeriodState,
    )

    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1), ("H2", "C1", 2)],
        heaters=["C1", "C2"],
        coolers=["H1"],
        hot_utility="HU",
        cold_utility="CU",
    )
    # (duty, source in, source out, sink in, sink out)
    states = {
        "recovery:H1->C1:S1": (600.0, 200.0, 140.0, 77.5, 152.5),
        "recovery:H2->C1:S2": (300.0, 150.0, 90.0, 40.0, 77.5),
        "hot-utility:HU->C1": (220.0, 250.0, 250.0, 152.5, 180.0),
        "hot-utility:HU->C2": (280.0, 250.0, 250.0, 50.0, 120.0),
        "cold-utility:H1->CU": (400.0, 140.0, 100.0, 20.0, 30.0),
    }
    exchangers = []
    for exchanger in structure.exchangers:
        duty, source_in, source_out, sink_in, sink_out = states[exchanger.exchanger_id]
        state = HeatExchangerPeriodState(
            period_id="0",
            period_idx=0,
            duty=duty,
            active=True,
            source_inlet_temperature=source_in,
            source_outlet_temperature=source_out,
            sink_inlet_temperature=sink_in,
            sink_outlet_temperature=sink_out,
        )
        exchangers.append(exchanger.model_copy(update={"period_states": (state,)}))
    return structure.model_copy(update={"exchangers": tuple(exchangers)})


def test_match_potential_uses_the_stream_temperatures_at_the_position():
    network = _solved_network()

    def potential(hot, cold, stage, new_stage):
        return match_potential(
            network, ENDS, NewMatch(hot, cold, stage, new_stage), approach=10.0
        )

    # H2 enters stage 2 at 150 C and C2 is still at 50 C: 4 * (150 - 50 - 10).
    assert potential("H2", "C2", 2, False) == 360.0
    # A new last stage sees H1 leaving stage 1 (its cooler inlet, 140 C).
    assert potential("H1", "C2", 3, True) == 320.0
    # H2 is at its 90 C target after stage 2; C1 enters stage 2 at 40 C.
    assert potential("H2", "C1", 3, True) == 200.0
    # No overlap: a new first stage sees H2 at 150 C and C1 leaving stage 1
    # at 152.5 C.
    assert potential("H2", "C1", 1, True) == 0.0


def test_screen_ranks_new_matches_by_potential_then_fewest_splits():
    network = _solved_network()

    ranked = screen_new_matches(network, ENDS, approach=10.0, limit=2)

    # Hand derivation (approach 10 K). H1 is at 200 C before stage 1 and at
    # 140 C (its cooler inlet) after it; C2 is at 50 C everywhere (it has only
    # its end heater); C1 enters stage 2 at 40 C and stage 1 at 77.5 C.
    #   H1-C2 at stage 1 (new or joining): min(10, 4) * (200 - 50 - 10) = 560
    #   H1-C1 below stage 1 (new stage 3, or joining stage 2):
    #       min(10, 8) * (140 - 40 - 10) = 720
    #   H1-C1 in a new stage 2 (C1 at 77.5 C): 8 * (140 - 77.5 - 10) = 420
    # So the two 720 kW moves lead, the new stage first because joining
    # stage 2 splits C1.
    assert ranked == (
        NewMatch("H1", "C1", 3, new_stage=True),
        NewMatch("H1", "C1", 2, new_stage=False),
    )
    everything = screen_new_matches(network, ENDS, approach=10.0, limit=100)
    # An existing match is never proposed again in its own stage.
    assert NewMatch("H1", "C1", 1, new_stage=False) not in everything
    assert all(
        match_potential(network, ENDS, move, approach=10.0) > 0.0 for move in everything
    )
