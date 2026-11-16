"""Retrofit frontier over the number of modifications."""

from __future__ import annotations

from collections.abc import Sequence

from ....contracts.synthesis.retrofit import HeatExchangerNetworkRetrofitDesign


def mark_dominated(
    designs: Sequence[HeatExchangerNetworkRetrofitDesign],
) -> tuple[HeatExchangerNetworkRetrofitDesign, ...]:
    """``designs`` with dominated ones marked.

    A design is dominated when one with fewer modifications has at least its
    net annual benefit.
    """

    marked = []
    for design in designs:
        dominated = any(
            other.modification_count < design.modification_count
            and other.net_annual_benefit >= design.net_annual_benefit
            for other in designs
        )
        marked.append(design.model_copy(update={"dominated": dominated}))
    return tuple(marked)
