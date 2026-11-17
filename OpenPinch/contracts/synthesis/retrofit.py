"""Heat exchanger network retrofit result schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...domain.heat_exchanger_network import HeatExchangerNetwork


class HeatExchangerNetworkRetrofitModification(BaseModel):
    """One modification of an existing network.

    ``kind`` is ``"new_match"`` (a new recovery exchanger, ``exchanger_id``,
    between ``hot_stream`` and ``cold_stream`` at ``stage``),
    ``"area_addition"`` (area added to existing exchanger ``exchanger_id``)
    or ``"relocation"`` (existing recovery exchanger ``exchanger_id``, with
    its installed area, moved to ``hot_stream`` and ``cold_stream`` at
    ``stage``). ``added_area`` is in m2; ``fixed_cost`` and ``area_cost`` are
    installed costs in $; ``new_splits`` names the streams a new or moved
    match splits. A new or moved match may come with the new end-of-stream
    heater or cooler that lets it work, ``utility_exchanger_id`` with
    ``utility_area`` (m2); its unit and area costs are in ``fixed_cost`` and
    ``area_cost``. ``repiped_streams`` names the streams whose piping the
    modification changes; their repiping is in ``fixed_cost``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["new_match", "area_addition", "relocation", "enhancement"]
    exchanger_id: str
    added_area: float = 0.0
    fixed_cost: float = 0.0
    area_cost: float = 0.0
    hot_stream: str | None = None
    cold_stream: str | None = None
    stage: int | None = None
    new_stage: bool = False
    new_splits: tuple[str, ...] = ()
    utility_exchanger_id: str | None = None
    utility_area: float = 0.0
    repiped_streams: tuple[str, ...] = ()


class HeatExchangerNetworkRetrofitDesign(BaseModel):
    """One retrofit design: the existing network with its modifications.

    Utilities are period-weighted kW; ``utility_cost``, ``annual_savings``
    (against the existing network), ``annualised_capital_cost`` and
    ``net_annual_benefit`` (savings less annualised capital) are $/y;
    ``capital_cost`` is the installed cost of the modifications in $ and
    ``payback_years`` is capital over savings (``None`` without savings).
    ``dominated`` marks a design no better in net annual benefit than one
    from an earlier search step.
    """

    model_config = ConfigDict(extra="forbid")

    modifications: tuple[HeatExchangerNetworkRetrofitModification, ...] = ()
    network: HeatExchangerNetwork
    hot_utility: float
    cold_utility: float
    utility_cost: float
    annual_savings: float
    capital_cost: float
    annualised_capital_cost: float
    net_annual_benefit: float
    payback_years: float | None = None
    dominated: bool = False

    @property
    def modification_count(self) -> int:
        return len(self.modifications)


class HeatExchangerNetworkRetrofitResult(BaseModel):
    """Retrofit search result: the best design for each number of steps.

    ``frontier[n]`` is the design after ``n`` steps that the search found
    best by net annual benefit; ``frontier[0]`` is the existing network. A
    step is one structural change with the area additions it keeps (the
    structure strategy) or one modification (the stepwise strategy), so
    ``modification_count`` can exceed ``n``.
    """

    model_config = ConfigDict(extra="forbid")

    frontier: tuple[HeatExchangerNetworkRetrofitDesign, ...] = Field(min_length=1)
    evaluated_designs: int = 0

    @property
    def selected(self) -> HeatExchangerNetworkRetrofitDesign:
        """The design with the best net annual benefit, fewest steps on a
        tie."""

        best = max(
            range(len(self.frontier)),
            key=lambda n: (self.frontier[n].net_annual_benefit, -n),
        )
        return self.frontier[best]
