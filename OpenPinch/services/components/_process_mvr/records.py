"""Parent-owned Process MVR membership and stream records."""

from __future__ import annotations

from dataclasses import dataclass, field

from ....classes.stream import Stream
from ....classes.zone import Zone
from ..direct_mvr import DirectGasMVRStageResult


@dataclass
class StreamMembership:
    """One occurrence of a stream inside a zone hot-stream collection."""

    zone: Zone
    key: str


@dataclass
class ProcessMVRStreamRecord:
    """Original and replacement streams for one MVR source stream."""

    original_stream: Stream
    original_memberships: list[StreamMembership]
    replacement_streams: list[Stream]
    replacement_memberships: list[StreamMembership]
    stage_results_by_period: dict[str, list[DirectGasMVRStageResult]]
    period_label_by_index: dict[int, str] = field(default_factory=dict)
