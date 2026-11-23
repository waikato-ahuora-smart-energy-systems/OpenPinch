"""A retrofit search that also tries a heat pump added as streams."""

from __future__ import annotations

import warnings
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
            # Only the levels that carry duty, as in the step.
            hpr_hot_streams={s.name: s for s in _carrying(solved.hpr_hot_streams)},
            hpr_cold_streams={s.name: s for s in _carrying(solved.hpr_cold_streams)},
        ),
        # The zone scales every stream's contribution by its multiplier;
        # the target stood the heat pump HPR_DT_CONT off as given.
        dt_cont=float(root.config.hpr.dt_cont) / float(root.dt_cont_multiplier),
        machine=machine,
    )


def _carrying(streams) -> tuple:
    """The streams of ``streams`` that carry duty."""

    return tuple(
        stream
        for _key, stream in streams.items()
        if float(stream.heat_flow.value) > 0.0
    )


def heat_pump_step(
    problem, base, settings, *, limits=None, machine: str = "HP", load_fraction=None
):
    """The heat pump step a second retrofit search starts from, and the
    as-piped heat pump target it was built from.

    ``base`` is the existing network's design from the search without the
    heat pump; the target is solved on its coolers and heaters (``limits``
    bounds that search) and the machine priced at
    ``settings.heat_pump_cost`` per kW of condenser capacity.
    """

    solved = as_piped_heat_pump(
        problem,
        base.network,
        args=dict(limits) if limits else None,
        load_fraction=load_fraction,
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
        ).model_copy(update={"load_fraction": load_fraction}),
        # A level the target left without duty is not a stream: it would
        # need a match it may not be able to have.
        condensers=_carrying(solved.hpr_hot_streams),
        evaporators=_carrying(solved.hpr_cold_streams),
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


def joined(base, *sized) -> HeatExchangerNetworkRetrofitResult:
    """One result from the search without the heat pump (``base``) and the
    searches started from a heat pump step (``sized``, one per size).

    A heat pump step is its search's first step, so that search's design
    after ``n`` further steps stands at step ``n + 1``. Each number of
    steps keeps the design with the best net annual benefit; the
    candidates of all are counted and their failed solves added step by
    step.
    """

    results = ((base, 0), *((result, 1) for result in sized))
    steps = max(len(result.frontier) + offset for result, offset in results)
    frontier = [base.frontier[0]]
    for step in range(1, steps):
        designs = [
            result.frontier[step - offset]
            for result, offset in results
            if step - offset < len(result.frontier)
        ]
        frontier.append(max(designs, key=lambda design: design.net_annual_benefit))
    # A search that ran out of candidates also counts the failures of the
    # last depth it tried, which gave no design: its failed solves can run
    # one step past its frontier.
    failed = [0] * max(
        [steps - 1, *(len(r.failed_solves) + offset for r, offset in results)]
    )
    for result, offset in results:
        for index, count in enumerate(result.failed_solves):
            failed[index + offset] += int(count)
    return HeatExchangerNetworkRetrofitResult(
        frontier=mark_dominated(frontier),
        evaluated_designs=sum(result.evaluated_designs for result, _ in results),
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
    if int(settings.max_modifications) < 1:
        # The heat pump step is a modification.
        return base
    periods = len(getattr(problem, "period_ids", None) or ())
    if periods > 1:
        # The as-piped target reads the first period alone.
        warnings.warn(
            "the heat pump step was left out of the retrofit: it is not built "
            f"for a problem with {periods} periods yet.",
            UserWarning,
            stacklevel=2,
        )
        return base
    step, solved = heat_pump_step(
        problem, base.frontier[0], settings, limits=limits, machine=machine
    )
    if step is None:
        return base
    # A case that refuses the streams (process components, several process
    # zones) or a network that cannot serve them is a heat pump this
    # retrofit cannot use, not a failed retrofit.
    try:
        case = snapshot_problem(heat_pump_case(problem, solved, machine=machine))
        with temporary_zone_configuration(
            case._build_execution_master_zone(), configuration
        ):
            pumped = run_retrofit_search(
                case, existing, replace(settings, heat_pump_cycle=None), heat_pump=step
            )
    except ValueError as error:
        warnings.warn(
            f"the heat pump step was left out of the retrofit: {error}",
            UserWarning,
            stacklevel=2,
        )
        return base
    return joined(base, pumped)
