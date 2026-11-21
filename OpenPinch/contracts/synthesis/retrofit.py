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
    ``"heat_pump"`` is a heat pump from ``hot_stream`` to ``cold_stream``:
    its evaporator ``exchanger_id`` (``added_area``) and condenser
    ``utility_exchanger_id`` (``utility_area``), with ``heat_pump_cop`` and
    ``heat_pump_capacity`` (its largest condenser duty, kW) and, for a
    real-fluid cycle, ``heat_pump_refrigerant``; ``fixed_cost`` holds its
    compressor and package, or its whole machine for a real-fluid cycle
    (then ``area_cost`` is 0). ``"heat_recovery_loop"`` is a new heat
    recovery loop linking ``hot_stream`` of one process to ``cold_stream``
    of another: its loop cooler ``exchanger_id`` (``added_area``) on the hot
    stream and loop heater ``utility_exchanger_id`` (``utility_area``) on
    the cold one, at ``loop_temperature`` (degC), carrying ``loop_duty``
    (its largest duty, kW); ``fixed_cost`` holds its two new units, the
    loop's pump and piping and the repiping.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "new_match",
        "area_addition",
        "relocation",
        "enhancement",
        "heat_pump",
        "heat_recovery_loop",
    ]
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
    heat_pump_cop: float | None = None
    heat_pump_capacity: float | None = None
    heat_pump_refrigerant: str | None = None
    loop_temperature: float | None = None
    loop_duty: float | None = None


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


class RetrofitHeatPumpTarget(BaseModel):
    """One HPR service heat pump target, against the utility it relieves.

    Duties and work in kW, temperatures in degC. ``*_before`` is the hot and
    cold utility the target starts from, ``*_after`` what is left once the
    heat pump delivers ``heat_delivered`` and takes the process part of its
    evaporator heat (all of it but ``ambient_heat``, drawn from air).
    """

    model_config = ConfigDict(extra="forbid")

    hot_utility_before: float
    cold_utility_before: float
    heat_delivered: float
    process_heat_absorbed: float
    ambient_heat: float
    work: float
    cop: float
    condensing_temperatures: tuple[float, ...]
    condenser_duties: tuple[float, ...]
    evaporating_temperatures: tuple[float, ...]
    evaporator_duties: tuple[float, ...]
    hot_utility_price: float
    cold_utility_price: float
    electricity_price: float
    total_annualized_cost: float | None = None

    @property
    def hot_utility_after(self) -> float:
        return self.hot_utility_before - self.heat_delivered

    @property
    def cold_utility_after(self) -> float:
        return self.cold_utility_before - self.process_heat_absorbed

    @property
    def operating_cost_before(self) -> float:
        """The utility cost without the heat pump, $/y."""

        return (
            self.hot_utility_before * self.hot_utility_price
            + self.cold_utility_before * self.cold_utility_price
        )

    @property
    def operating_cost_after(self) -> float:
        """The utility and electricity cost with the heat pump, $/y."""

        return (
            self.hot_utility_after * self.hot_utility_price
            + self.cold_utility_after * self.cold_utility_price
            + self.work * self.electricity_price
        )


class RetrofitHeatPumpTargets(BaseModel):
    """The ideal and the as-piped heat pump target of an existing network.

    ``ideal`` serves the process's heating target from its cooling target
    (the direct-integration cascade), as a new design could; ``as_piped``
    serves only the existing network's heaters from only its coolers, so
    it needs no process-side repiping. A target with no load is a heat pump
    that delivers nothing.

    The gap between their operating costs after the heat pump splits in
    two: ``network_gap``, what the existing network's utility costs beyond
    the targets before any heat pump (the heat recovery a network retrofit
    could win), and ``heat_pump_gap``, the operating saving the as-piped
    heat pump falls short of the ideal's by (what reworking the network
    for the heat pump could win). Neither counts heat pump capital.
    """

    model_config = ConfigDict(extra="forbid")

    ideal: RetrofitHeatPumpTarget
    as_piped: RetrofitHeatPumpTarget

    @property
    def network_gap(self) -> float:
        """$/y."""

        return self.as_piped.operating_cost_before - self.ideal.operating_cost_before

    @property
    def heat_pump_gap(self) -> float:
        """$/y."""

        def saving(target: RetrofitHeatPumpTarget) -> float:
            return target.operating_cost_before - target.operating_cost_after

        return saving(self.ideal) - saving(self.as_piped)
