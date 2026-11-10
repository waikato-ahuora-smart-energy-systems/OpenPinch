"""Frozen public values for single-stage refrigerant screening."""

from __future__ import annotations

from enum import StrEnum


class RefrigerantScreenVerdict(StrEnum):
    """Screening verdicts, in the order they are judged."""

    ABOVE_CRITICAL = "above critical"
    BELOW_TRIPLE_POINT = "below triple point"
    OVER_RATIO_LIMIT = "over ratio limit"
    DEEP_VACUUM_SUCTION = "deep vacuum suction"
    NEAR_CRITICAL = "near critical"
    PRACTICAL = "practical"


__all__ = ["RefrigerantScreenVerdict"]
