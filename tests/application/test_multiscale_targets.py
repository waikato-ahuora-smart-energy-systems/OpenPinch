"""Structural traversal tests for explicit multi-scale targeting."""

from __future__ import annotations

import pytest

from OpenPinch.application._problem.targeting.dispatch import (
    run_targeting_for_zone_and_subzones,
)
from OpenPinch.domain.enums import ZT
from OpenPinch.domain.zone import Zone


def _zone_tree() -> Zone:
    site = Zone(name="Site", type=ZT.S.value)
    process = Zone(name="Process", type=ZT.P.value)
    operation = Zone(name="Operation", type=ZT.O.value)
    process.add_zone(operation)
    site.add_zone(process)
    return site


def test_targeting_traversal_is_child_first_and_structural():
    calls: list[str] = []
    tree = _zone_tree()

    result = run_targeting_for_zone_and_subzones(
        tree,
        direct_service_func=lambda zone, _args: calls.append(f"direct:{zone.name}"),
        indirect_service_func=lambda zone, _args: calls.append(f"indirect:{zone.name}"),
    )

    assert result is tree
    assert calls == [
        "direct:Operation",
        "direct:Process",
        "indirect:Process",
        "direct:Site",
        "indirect:Site",
    ]


def test_targeting_traversal_passes_the_same_arguments_to_each_service():
    received: list[tuple[str, dict[str, object] | None]] = []
    arguments = {"period_id": "peak"}

    run_targeting_for_zone_and_subzones(
        _zone_tree(),
        direct_service_func=lambda zone, args: received.append((zone.name, args)),
        args=arguments,
    )

    assert received == [
        ("Operation", arguments),
        ("Process", arguments),
        ("Site", arguments),
    ]
    assert all(args is arguments for _, args in received)


def test_targeting_traversal_skips_utility_zones():
    utility = Zone(name="Utilities", type=ZT.U.value)
    calls: list[str] = []

    result = run_targeting_for_zone_and_subzones(
        utility,
        direct_service_func=lambda zone, _args: calls.append(zone.name),
        indirect_service_func=lambda zone, _args: calls.append(zone.name),
    )

    assert result is utility
    assert calls == []


def test_targeting_traversal_rejects_unknown_zone_types():
    with pytest.raises(ValueError, match="No valid zone"):
        run_targeting_for_zone_and_subzones(Zone(name="Unknown", type="unknown"))
