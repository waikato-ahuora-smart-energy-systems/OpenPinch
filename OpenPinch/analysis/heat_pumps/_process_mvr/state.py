"""Parent-owned Process MVR membership and stream records."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....domain.stream import Stream
from ....domain.zone import Zone
from ..direct_mvr.models import DirectGasMVRStageResult


@dataclass
class _StreamMembership:
    """One occurrence of a stream inside a zone hot-stream collection."""

    zone: Zone
    key: str


@dataclass
class _ProcessMVRStreamRecord:
    """Original and replacement streams for one MVR source stream."""

    original_stream: Stream
    original_memberships: list[_StreamMembership]
    replacement_streams: list[Stream]
    replacement_memberships: list[_StreamMembership]
    stage_results_by_period: dict[str, list[DirectGasMVRStageResult]]
    period_label_by_index: dict[int, str] = field(default_factory=dict)
