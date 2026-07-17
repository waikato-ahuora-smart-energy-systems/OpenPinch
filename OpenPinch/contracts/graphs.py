"""Graph data schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DataPoint(BaseModel):
    """Coordinate used to construct composite curves and other plots."""

    x: float
    y: float


class Segment(BaseModel):
    """Continuous plot segment optionally annotated with colour/arrows."""

    title: Optional[str] = None
    colour: Optional[int] = Field(
        default=None,
        description="Optional integer colour (e.g., RGB packed int or palette index).",
    )
    arrow: Optional[str] = Field(
        default=None,
        description="Optional arrow style; consider making this an Enum.",
    )
    data_points: List[DataPoint] = Field(default_factory=list)


class Graph(BaseModel):
    """Collection of segments representing a single graph (e.g., GCC)."""

    type: str
    segments: List[Segment] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)


class GraphSet(BaseModel):
    """Named group of graphs emitted for a particular zone or context."""

    name: str = "GraphSet"
    target_type: Optional[str] = None
    period_id: Optional[str] = None
    zone_name: Optional[str] = None
    zone_address: Optional[str] = None
    graphs: List[Graph] = Field(default_factory=list)


__all__ = ["DataPoint", "Graph", "GraphSet", "Segment"]
