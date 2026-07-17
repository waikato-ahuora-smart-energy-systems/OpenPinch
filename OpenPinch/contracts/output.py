"""Schema returned by the external OpenPinch analysis contract."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from .graphs import GraphSet
from .reporting import TargetResults
from .synthesis.result import HeatExchangerNetworkSynthesisResult


class TargetOutput(BaseModel):
    """Top-level cached targeting output for :class:`PinchProblem`."""

    name: str = "Site"
    period_id: Optional[str] = None
    targets: List[TargetResults]
    graphs: Optional[Dict[str, GraphSet]] = None
    design: Optional[HeatExchangerNetworkSynthesisResult] = None


__all__ = ["TargetOutput"]


TargetOutput.model_rebuild()
