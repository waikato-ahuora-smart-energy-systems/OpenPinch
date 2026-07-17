"""Typed aliases for heat exchanger network labelled access."""

from __future__ import annotations

from typing import TypeAlias

from ..enums import HeatExchangerNetworkLabel

HeatExchangerNetworkLabelKey: TypeAlias = str | HeatExchangerNetworkLabel

__all__ = ["HeatExchangerNetworkLabelKey"]
