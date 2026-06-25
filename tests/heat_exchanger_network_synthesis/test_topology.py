"""Canonical topology normalization tests."""

from __future__ import annotations

from OpenPinch.lib.schemas.synthesis import HeatExchangerNetworkTopologyRestriction
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.topology import (
    canonical_topology_restrictions,
    topology_restriction_signature,
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
