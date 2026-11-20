"""Canonical topology normalization tests."""

from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from OpenPinch.analysis.heat_exchanger_networks.targeting.topology import (
    canonical_topology_restrictions,
    topology_restriction_signature,
)
from OpenPinch.contracts.synthesis.topology import (
    HeatExchangerNetworkTopologyRestriction,
)


def test_canonical_topology_removes_empty_stage_gaps() -> None:
    restrictions = canonical_topology_restrictions((_restriction("H1", "C1", 3),))

    assert [item.stage for item in restrictions] == [1]


def test_canonical_topology_splits_independent_matches_in_one_stage() -> None:
    restrictions = canonical_topology_restrictions(
        (
            _restriction("H1", "C1", 1),
            _restriction("H2", "C2", 1),
        ),
        hot_stream_order=("H1", "H2"),
        cold_stream_order=("C1", "C2"),
    )

    assert [
        (item.source_stream, item.sink_stream, item.stage) for item in restrictions
    ] == [
        ("H1", "C1", 1),
        ("H2", "C2", 2),
    ]


def test_canonical_topology_keeps_shared_stream_matches_together() -> None:
    restrictions = canonical_topology_restrictions(
        (
            _restriction("H1", "C1", 1),
            _restriction("H1", "C2", 1),
            _restriction("H2", "C3", 1),
        )
    )

    assert [item.stage for item in restrictions] == [1, 1, 2]


def test_canonical_topology_signature_matches_equivalent_grid_diagrams() -> None:
    left = canonical_topology_restrictions(
        (
            _restriction("H1", "C1", 1),
            _restriction("H2", "C2", 1),
        )
    )
    right = canonical_topology_restrictions(
        (
            _restriction("H1", "C1", 1),
            _restriction("H2", "C2", 3),
        )
    )

    assert topology_restriction_signature(left) == topology_restriction_signature(right)


def _signature(matches) -> tuple:
    return topology_restriction_signature(
        canonical_topology_restrictions(
            _restriction(hot, cold, stage) for hot, cold, stage in matches
        )
    )


def test_matches_in_stages_that_share_no_stream_sign_by_stream_order() -> None:
    """The four-stream networks from the round 3 review (M7): each stream
    meets its matches in the same order, so both are one network."""
    left = (("RM", "MC", 1), ("HF", "MC", 2), ("RM", "CIP", 3))
    right = (("RM", "MC", 1), ("RM", "CIP", 2), ("HF", "MC", 3))

    assert _signature(left) == _signature(right)


def test_another_match_order_on_a_stream_signs_differently() -> None:
    first = (("RM", "MC", 1), ("RM", "CIP", 2))
    second = (("RM", "CIP", 1), ("RM", "MC", 2))

    assert _signature(first) != _signature(second)


_STREAM = st.tuples(
    st.sampled_from(("H1", "H2", "H3")), st.sampled_from(("C1", "C2", "C3"))
)


def _stage_sequence(pairs) -> tuple:
    # One match per stage, in list order.
    return tuple((hot, cold, stage) for stage, (hot, cold) in enumerate(pairs, 1))


def _share_a_stream(left, right) -> bool:
    return left[0] == right[0] or left[1] == right[1]


@given(
    pairs=st.lists(_STREAM, min_size=2, max_size=7),
    swaps=st.lists(st.integers(min_value=0, max_value=5), max_size=20),
)
def test_relabelling_stages_of_disjoint_matches_keeps_the_signature(
    pairs, swaps
) -> None:
    relabelled = list(pairs)
    for index in swaps:
        if index + 1 < len(relabelled) and not _share_a_stream(
            relabelled[index], relabelled[index + 1]
        ):
            relabelled[index], relabelled[index + 1] = (
                relabelled[index + 1],
                relabelled[index],
            )

    assert _signature(_stage_sequence(pairs)) == _signature(_stage_sequence(relabelled))


@given(
    pairs=st.lists(_STREAM, min_size=2, max_size=7),
    first=st.integers(min_value=0, max_value=6),
    second=st.integers(min_value=0, max_value=6),
)
def test_changing_the_match_order_on_a_stream_changes_the_signature(
    pairs, first, second
) -> None:
    first, second = sorted((first % len(pairs), second % len(pairs)))
    left, right = pairs[first], pairs[second]
    assume(left != right and _share_a_stream(left, right))
    reordered = list(pairs)
    reordered[first], reordered[second] = right, left

    assert _signature(_stage_sequence(pairs)) != _signature(_stage_sequence(reordered))


def _restriction(
    hot: str,
    cold: str,
    stage: int,
) -> HeatExchangerNetworkTopologyRestriction:
    return HeatExchangerNetworkTopologyRestriction(
        source_stream=hot,
        sink_stream=cold,
        stage=stage,
        duty=100.0,
    )
