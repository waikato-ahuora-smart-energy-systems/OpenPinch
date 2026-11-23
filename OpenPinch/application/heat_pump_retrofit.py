"""A retrofit search that also tries a heat pump added as streams."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from ..analysis.heat_exchanger_networks.retrofit.frontier import mark_dominated
from ..analysis.heat_exchanger_networks.retrofit.heat_pump_streams import HeatPumpStep
from ..analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_heat_pump,
)
from ..analysis.heat_exchanger_networks.retrofit.search import run_retrofit_search
from ..analysis.heat_exchanger_networks.solver.arrays import annual_cost_scales
from ..contracts.synthesis.retrofit import HeatExchangerNetworkRetrofitResult
from ..domain.enums import StreamType
from ..domain.heat_exchanger_network import NetworkHeatPump
from ._problem.arguments import temporary_zone_configuration
from ._problem.targeting.state import snapshot_problem
from .case_derivation import _case_with_heat_pump_streams


def heat_pump_machine(
    solved, *, name: str, cycle: str, cost_per_kw: float, costing
) -> NetworkHeatPump:
    """The as-piped heat pump target ``solved`` as the machine a retrofit
    search starts from.

    Its duties are those of its condenser and evaporator streams (kW).
    The electricity is its work at ``COSTING_HPR_ELE_PRICE`` over the
    year's operating hours; the compressor costs ``cost_per_kw`` ($) per
    kW of condenser capacity, annualised as exchanger capital is.
    """

    def duty(streams) -> float:
        return sum(float(stream.heat_flow.value) for _key, stream in streams.items())

    price_scale, capital_scale = annual_cost_scales(costing)
    capacity = duty(solved.hpr_hot_streams)
    work = float(solved.w_net)
    capital = float(cost_per_kw) * capacity
    return NetworkHeatPump(
        machine=name,
        cycle=cycle,
        condenser_duty=capacity,
        evaporator_duty=duty(solved.hpr_cold_streams),
        work=work,
        cop=float(solved.cop_h),
        electricity_cost=work * float(costing.hpr_ele_price) * price_scale,
        capital_cost=capital,
        annual_capital_cost=capital * capital_scale,
    )


def heat_pump_case(problem, solved, *, machine: str):
    """A copy of ``problem`` holding the streams of the as-piped heat pump
    target ``solved``.

    Each condenser is a hot stream and each evaporator a cold stream of
    the problem's process zone, marked with ``machine`` and its role, at
    ``HPR_DT_CONT``: the as-piped target stands the heat pump that far
    off the coolers and heaters.
    """

    root = problem._require_prepared_root_zone()
    return _case_with_heat_pump_streams(
        problem,
        SimpleNamespace(
            provenance=SimpleNamespace(zone_address=root.address),
            hpr_details=SimpleNamespace(amb_streams=solved.amb_streams),
            hpr_hot_streams=solved.hpr_hot_streams,
            hpr_cold_streams=solved.hpr_cold_streams,
        ),
        dt_cont=float(root.config.hpr.dt_cont),
        machine=machine,
    )


def heat_pump_step(problem, base, settings, *, limits=None, machine: str = "HP"):
    """The heat pump step a second retrofit search starts from, and the
    as-piped heat pump target it was built from.

    ``base`` is the existing network's design from the search without the
    heat pump; the target is solved on its coolers and heaters (``limits``
    bounds that search) and the machine priced at
    ``settings.heat_pump_cost`` per kW of condenser capacity.
    """

    solved = as_piped_heat_pump(
        problem, base.network, args=dict(limits) if limits else None
    )
    if solved is None:
        # The heaters carry no load: no heat pump to add.
        return None, None
    step = HeatPumpStep(
        record=heat_pump_machine(
            solved,
            name=machine,
            cycle=str(settings.heat_pump_cycle),
            cost_per_kw=settings.heat_pump_cost,
            costing=problem.master_zone.config.costing,
        ),
        condensers=tuple(stream for _key, stream in solved.hpr_hot_streams.items()),
        evaporators=tuple(stream for _key, stream in solved.hpr_cold_streams.items()),
        base=base,
        # The air the evaporators draw on; air that takes heat the heat
        # pump leaves is not a stream of the case.
        ambient_sources=tuple(
            stream
            for _key, stream in solved.amb_streams.items()
            if stream.stream_type == StreamType.Hot.value
        ),
    )
    return step, solved


def joined(base, pumped) -> HeatExchangerNetworkRetrofitResult:
    """One result from the search without the heat pump (``base``) and the
    one started from it (``pumped``).

    The heat pump step is the second search's first step, so its design
    after ``n`` further steps stands at step ``n + 1``. Each number of
    steps keeps the design with the better net annual benefit; the
    candidates of both are counted and their failed solves added step by
    step.
    """

    steps = max(len(base.frontier), len(pumped.frontier) + 1)
    frontier = [base.frontier[0]]
    for step in range(1, steps):
        designs = [
            series[index]
            for series, index in ((base.frontier, step), (pumped.frontier, step - 1))
            if index < len(series)
        ]
        frontier.append(max(designs, key=lambda design: design.net_annual_benefit))
    failed = [0] * (steps - 1)
    for series, offset in ((base.failed_solves, 0), (pumped.failed_solves, 1)):
        for index, count in enumerate(series):
            failed[index + offset] += int(count)
    return HeatExchangerNetworkRetrofitResult(
        frontier=mark_dominated(frontier),
        evaluated_designs=base.evaluated_designs + pumped.evaluated_designs,
        failed_solves=tuple(failed),
    )


def retrofit_with_heat_pump(
    problem,
    existing,
    settings,
    *,
    configuration=None,
    limits=None,
    machine: str = "HP",
) -> HeatExchangerNetworkRetrofitResult:
    """The retrofit search of ``existing``, and a second search started
    from its as-piped heat pump added as streams, as one result.

    The second search runs on ``heat_pump_case`` under ``configuration``,
    as the first runs on ``problem``, from ``heat_pump_step``.
    """

    base = run_retrofit_search(problem, existing, settings)
    step, solved = heat_pump_step(
        problem, base.frontier[0], settings, limits=limits, machine=machine
    )
    if step is None:
        return base
    case = snapshot_problem(heat_pump_case(problem, solved, machine=machine))
    with temporary_zone_configuration(
        case._build_execution_master_zone(), configuration
    ):
        pumped = run_retrofit_search(
            case, existing, replace(settings, heat_pump_cycle=None), heat_pump=step
        )
    return joined(base, pumped)
