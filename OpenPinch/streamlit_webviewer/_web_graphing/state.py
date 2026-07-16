"""Private dashboard graph grouping state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, MutableMapping


@dataclass(slots=True)
class StreamlitGraphSet:
    """Convenience wrapper storing graphs grouped by target name."""

    name: str
    graphs: List[MutableMapping]

    @classmethod
    def from_graph_data(
        cls, graph_set_data: Mapping[str, object]
    ) -> "StreamlitGraphSet":
        """Build a graph-set wrapper from JSON-style graph data."""
        return cls(
            name=str(graph_set_data.get("name", "Graph Set")),
            graphs=list(graph_set_data.get("graphs", [])),
        )
