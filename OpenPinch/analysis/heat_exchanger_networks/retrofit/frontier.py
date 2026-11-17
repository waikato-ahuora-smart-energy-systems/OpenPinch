"""Retrofit frontier over the number of search steps."""

from __future__ import annotations

from collections.abc import Sequence

from ....contracts.synthesis.retrofit import HeatExchangerNetworkRetrofitDesign


def mark_dominated(
    designs: Sequence[HeatExchangerNetworkRetrofitDesign],
) -> tuple[HeatExchangerNetworkRetrofitDesign, ...]:
    """``designs`` (one per search step, in order) with dominated ones
    marked.

    A design is dominated when one from an earlier step has at least its net
    annual benefit. Steps, not modification records, count: a structural
    step can keep several area additions.
    """

    return tuple(
        design.model_copy(
            update={
                "dominated": any(
                    earlier.net_annual_benefit >= design.net_annual_benefit
                    for earlier in designs[:step]
                )
            }
        )
        for step, design in enumerate(designs)
    )
