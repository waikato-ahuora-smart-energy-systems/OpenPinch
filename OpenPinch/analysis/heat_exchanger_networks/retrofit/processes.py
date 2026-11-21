"""Which process zone each stream of a problem belongs to."""

from __future__ import annotations

from ....domain.enums import ZoneType


def stream_process_zones(zone) -> dict[str, str]:
    """Each stream key of ``zone`` mapped to the name of its process zone.

    A stream belongs to the outermost ``Process Zone`` at or below ``zone``
    that holds it; a stream outside every process zone is left out. Keys are
    ``zone``'s own stream keys (``Heating.H1``), as the solver arrays use.
    """

    keys = {
        id(stream): str(key)
        for streams in (zone.hot_streams, zone.cold_streams)
        for key, stream in streams.items()
    }
    found: dict[str, str] = {}

    def visit(node) -> None:
        if str(getattr(node, "type", "")) == ZoneType.P.value:
            for streams in (node.hot_streams, node.cold_streams):
                for _key, stream in streams.items():
                    if id(stream) in keys:
                        found.setdefault(keys[id(stream)], str(node.name))
            return
        for child in (node.subzones or {}).values():
            visit(child)

    visit(zone)
    return found


__all__ = ["stream_process_zones"]
