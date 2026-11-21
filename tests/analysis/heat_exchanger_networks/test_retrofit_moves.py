"""Retrofit modifications applied to a network structure (no solver)."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.moves import (
    NewMatch,
    NewUtility,
    apply_new_match,
    apply_new_utility,
    apply_relocation,
    match_potential,
    new_splits,
    new_utility_candidates,
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


def test_screen_ranks_new_matches_by_utility_relief_then_potential():
    network = _solved_network()

    ranked = screen_new_matches(network, ENDS, approach=10.0, limit=2)

    # Hand derivation (approach 10 K). H1 is at 200 C before stage 1 and at
    # 140 C (its cooler inlet) after it; C2 is at 50 C everywhere (it has only
    # its end heater); C1 enters stage 2 at 40 C and stage 1 at 77.5 C.
    #   H1-C2 at stage 1 (new or joining): min(10, 4) * (200 - 50 - 10) = 560
    #   H1-C1 below stage 1 (new stage 3, or joining stage 2):
    #       min(10, 8) * (140 - 40 - 10) = 720
    # Ranked by potential alone the two 720 kW H1-C1 moves led. A new match
    # first cuts utility only as far as its hot stream has cooling and its
    # cold stream heating: H1 is cooled 400 kW, C1 heated 220 kW and C2
    # 280 kW, so H1-C1 relieves min(720, 400, 220) = 220 kW and H1-C2 at
    # stage 1 min(560, 400, 280) = 280 kW. The H1-C2 moves lead (a new stage
    # first, since joining stage 1 splits H1); every H2 move relieves nothing,
    # H2 having no cooler. This replaces ranking by potential alone, which
    # tried matches that could not cut utility first.
    assert ranked == (
        NewMatch("H1", "C2", 1, new_stage=True),
        NewMatch("H1", "C2", 1, new_stage=False),
    )
    everything = screen_new_matches(network, ENDS, approach=10.0, limit=100)
    # An existing match is never proposed again in its own stage.
    assert NewMatch("H1", "C1", 1, new_stage=False) not in everything
    assert all(
        match_potential(network, ENDS, move, approach=10.0) > 0.0 for move in everything
    )


def test_the_screen_keeps_each_match_within_one_process_when_given_processes():
    network = _solved_network()
    processes = {"H1": "A", "C1": "A", "H2": "B", "C2": "B"}

    everything = screen_new_matches(network, ENDS, approach=10.0, limit=100)
    within = screen_new_matches(
        network, ENDS, approach=10.0, limit=100, processes=processes
    )

    # H1-C2 leads the unrestricted screen but joins two processes.
    assert any(m.hot == "H1" and m.cold == "C2" for m in everything)
    assert within == tuple(
        m for m in everything if processes[m.hot] == processes[m.cold]
    )
    assert within


def test_the_process_screen_drops_streams_outside_every_process():
    network = _solved_network()
    # H2 and C2 sit outside every process zone: their pair is not proven to
    # be within one process.
    processes = {"H1": "A", "C1": "A"}

    within = screen_new_matches(
        network, ENDS, approach=10.0, limit=100, processes=processes
    )

    assert within
    assert all(m.hot in processes and m.cold in processes for m in within)


def test_process_zones_are_told_apart_by_their_address():
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit.processes import (
        stream_process_zones,
    )

    def streams(**items):
        return SimpleNamespace(items=lambda: list(items.items()))

    h1, c1, h2, c2 = (object() for _ in range(4))

    def zone(name, address, kind, hot, cold, subzones=None):
        return SimpleNamespace(
            name=name,
            address=address,
            type=kind,
            hot_streams=streams(**hot),
            cold_streams=streams(**cold),
            subzones=subzones or {},
        )

    # Two sites each holding a process zone named "Dairy".
    north = zone("Dairy", "Site/North/Dairy", "Process Zone", {"x": h1}, {"y": c1})
    south = zone("Dairy", "Site/South/Dairy", "Process Zone", {"x": h2}, {"y": c2})
    root = zone(
        "Site",
        "Site",
        "Site",
        {"North.Dairy.H1": h1, "South.Dairy.H1": h2},
        {"North.Dairy.C1": c1, "South.Dairy.C1": c2},
        {
            "North": zone("North", "Site/North", "Site", {}, {}, {"Dairy": north}),
            "South": zone("South", "Site/South", "Site", {}, {}, {"Dairy": south}),
        },
    )

    assert stream_process_zones(root) == {
        "North.Dairy.H1": "Site/North/Dairy",
        "North.Dairy.C1": "Site/North/Dairy",
        "South.Dairy.H1": "Site/South/Dairy",
        "South.Dairy.C1": "Site/South/Dairy",
    }
