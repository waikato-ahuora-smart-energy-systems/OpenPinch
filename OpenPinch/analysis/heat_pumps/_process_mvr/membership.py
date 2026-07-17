"""Zone membership operations for parent-owned Process MVR records."""

from __future__ import annotations

from ....domain.stream import Stream
from ....domain.zone import Zone
from .state import _ProcessMVRStreamRecord, _StreamMembership


def add_replacement_streams_to_memberships(
    record: _ProcessMVRStreamRecord,
    component_id: str,
) -> None:
    """Attach replacement streams wherever the original stream is present."""
    for membership in record.original_memberships:
        for stream in record.replacement_streams:
            key = f"{membership.key}.{component_id}.{stream.name}"
            membership.zone.hot_streams.add(stream, key=key, prevent_overwrite=False)
            record.replacement_memberships.append(
                _StreamMembership(zone=membership.zone, key=key)
            )


def find_hot_stream_memberships(
    root: Zone,
    stream: Stream,
) -> list[_StreamMembership]:
    """Return every hot-stream collection containing ``stream`` by identity."""
    memberships: list[_StreamMembership] = []
    for zone in walk_zones(root):
        for key, candidate in zone.hot_streams.items():
            if candidate is stream:
                memberships.append(_StreamMembership(zone=zone, key=key))
    return memberships


def record_affects_zone(record: _ProcessMVRStreamRecord, zone: Zone) -> bool:
    """Return whether the record belongs to the zone or one of its descendants."""
    zone_address = zone.address
    for membership in record.original_memberships:
        member_address = membership.zone.address
        if member_address == zone_address or member_address.startswith(
            f"{zone_address}/"
        ):
            return True
    return False


def walk_zones(zone: Zone):
    """Yield a zone tree in deterministic pre-order."""
    yield zone
    for subzone in zone.subzones.values():
        yield from walk_zones(subzone)


__all__: list[str] = []
