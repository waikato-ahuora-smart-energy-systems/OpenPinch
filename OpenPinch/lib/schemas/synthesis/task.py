"""Task-level HEN synthesis schemas."""

from __future__ import annotations

from .method import (
    HeatExchangerNetworkSynthesisMethodInput,
    HeatExchangerNetworkSynthesisMethodOutput,
)


class HeatExchangerNetworkSynthesisTask(HeatExchangerNetworkSynthesisMethodInput):
    """One deterministic OpenPinch heat exchanger network synthesis task record."""


class HeatExchangerNetworkSynthesisTaskOutcome(
    HeatExchangerNetworkSynthesisMethodOutput,
):
    """Outcome for one OpenPinch heat exchanger network synthesis task."""

    task: HeatExchangerNetworkSynthesisTask


HeatExchangerNetworkSynthesisTask.model_rebuild()
HeatExchangerNetworkSynthesisTaskOutcome.model_rebuild()
