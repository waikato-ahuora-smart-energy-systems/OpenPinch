"""Regression tests for the HEN audit fixes (area 5)."""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.execution.task_builders import (
    has_recovery_topology,
)
from OpenPinch.domain.enums import HeatExchangerKind


def _exchanger(kind, *, active=True, stage=1):
    return SimpleNamespace(
        kind=kind,
        match_allowed=True,
        stage=stage,
        period_states=(SimpleNamespace(active=active),),
    )


def test_utility_only_outcome_cannot_seed_later_methods():
    utility_only = SimpleNamespace(
        network=SimpleNamespace(
            exchangers=(_exchanger(HeatExchangerKind.HOT_UTILITY, stage=None),)
        )
    )
    with_recovery = SimpleNamespace(
        network=SimpleNamespace(exchangers=(_exchanger(HeatExchangerKind.RECOVERY),))
    )

    assert has_recovery_topology(utility_only) is False
    assert has_recovery_topology(with_recovery) is True
    assert has_recovery_topology(SimpleNamespace(network=None)) is False
