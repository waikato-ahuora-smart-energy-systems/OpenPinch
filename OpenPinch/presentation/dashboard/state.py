"""Private dashboard graph grouping state."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass


@dataclass(slots=True)
class _DashboardGraphSet:
    """Graphs grouped for one target during a dashboard render."""

    name: str
    graphs: list[MutableMapping]

    @classmethod
    def from_graph_data(
        cls, graph_set_data: Mapping[str, object]
    ) -> "_DashboardGraphSet":
        """Build a graph-set wrapper from JSON-style graph data."""
        return cls(
            name=str(graph_set_data.get("name", "Graph Set")),
            graphs=list(graph_set_data.get("graphs", [])),
        )
