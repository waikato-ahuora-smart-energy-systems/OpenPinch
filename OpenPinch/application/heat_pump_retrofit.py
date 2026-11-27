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
from .case_derivation import (
    _case_with_heat_pump_streams,
    _heat_pump_zone_address,
    _period_heat_pump_streams,
    _zone_multiplier,
    heat_pump_stream_records,
)


def heat_pump_machine(
    solved,
    *,
    name: str,
    cycle: str,
    cost_per_kw: float,
    costing,
    period_weights=None,
) -> NetworkHeatPump:
    """The as-piped heat pump target ``solved`` as the machine a retrofit
    search starts from.

    Its duties are those of its condenser and evaporator streams (kW).
    The electricity is its work at ``COSTING_HPR_ELE_PRICE`` over the
    year's operating hours; the compressor costs ``cost_per_kw`` ($) per
    kW of condenser capacity, annualised as exchanger capital is. A
    "vapour_compression" machine costs what the HPR service installs it
    at without its process and air exchangers instead
    (``hpr_machine_internal_capital_costs``: compressors, motors, cascade
    exchangers and IHXs): its condensers and evaporators are the
    network's exchangers, priced there.

    With ``period_weights`` (by period id) and a shared design's
    ``period_outputs``, the duties are the largest of any period, the work
    each period's at its share of the weights (none where the machine is
    off) and the COP the heat so delivered over that work; a
    "vapour_compression" machine installs each of its machines at the
    largest internal capital any period needs.
    """

    def duty(streams) -> float:
        return sum(float(stream.heat_flow.value) for _key, stream in streams.items())

    price_scale, capital_scale = annual_cost_scales(costing)
    capacity = duty(solved.hpr_hot_streams)
    evaporator = duty(solved.hpr_cold_streams)
    work = float(solved.w_net)
    cop = float(solved.cop_h)
    internal = tuple(getattr(solved, "hpr_machine_internal_capital_costs", None) or ())
    outputs = getattr(solved, "period_outputs", None)
    if period_weights and outputs:
        # A shared design: sized for the period it delivers most in, each
        # period's work drawn for its share of the year. A period the
        # machine is off in has no output and draws nothing.
        total = float(sum(period_weights.values()))
        running = {p: outputs[p] for p in period_weights if p in outputs}
        delivered = {p: duty(o["hpr_hot_streams"]) for p, o in running.items()}
        work = sum(
            float(period_weights[p]) / total * float(o["w_net"])
            for p, o in running.items()
        )
        heat = sum(float(period_weights[p]) / total * q for p, q in delivered.items())
        cop = heat / work if work else cop
        capacity = max(delivered.values(), default=0.0)
        evaporator = max(
            (duty(o["hpr_cold_streams"]) for o in running.values()), default=0.0
        )
        # Each machine installed for the period that needs it largest: the
        # solve's own costs are the selected period's alone.
        each = [internal] + [
            tuple(o.get("hpr_machine_internal_capital_costs") or ())
            for o in running.values()
        ]
        internal = tuple(
            max(float(costs[n]) for costs in each if n < len(costs))
            for n in range(max(map(len, each)))
        )
    capital = (
        sum(float(c) for c in internal)
        if cycle == "vapour_compression"
        else float(cost_per_kw) * capacity
    )
    return NetworkHeatPump(
        machine=name,
        cycle=cycle,
        condenser_duty=capacity,
        evaporator_duty=evaporator,
        work=work,
        cop=cop,
        electricity_cost=work * float(costing.hpr_ele_price) * price_scale,
        capital_cost=capital,
        annual_capital_cost=capital * capital_scale,
    )


def heat_pump_case(problem, solved, *, machine: str, dt_cont_multiplier: float = 1.0):
    """A copy of ``problem`` holding the streams of the as-piped heat pump
    target ``solved``.

    Each condenser is a hot stream and each evaporator a cold stream of
    the problem's process zone, marked with ``machine`` and its role, at
    ``HPR_DT_CONT``: the as-piped target stands the heat pump that far
    off the coolers and heaters. ``dt_cont_multiplier`` is the multiplier
    the retrofit's model scales every contribution by (a designed existing
    network's, ``RetrofitSettings.dt_cont_multiplier``); the streams are
    written over it, so the model holds them at ``HPR_DT_CONT`` and the air
    at its own contribution still. A shared design over several periods
    (``period_outputs``) writes each stream in every period, with no duty
    where it is off; a level without duty in any period is no stream.
    """

    root = problem._require_prepared_root_zone()
    outputs = getattr(solved, "period_outputs", None)
    target = SimpleNamespace(
        provenance=SimpleNamespace(zone_address=root.address),
        # The air a period draws on is built from it when the solve's own
        # period draws none.
        config=root.config,
        # A shared design over several periods: the case writes each
        # stream in every period from its period outputs, a level without
        # duty in a period off there.
        hpr_shared_design=bool(outputs),
        hpr_details=SimpleNamespace(
            amb_streams=solved.amb_streams,
            period_outputs={
                period: output
                | {
                    group: {s.name: s for s in _carrying(output.get(group) or {})}
                    for group in ("hpr_hot_streams", "hpr_cold_streams")
                }
                for period, output in (outputs or {}).items()
            },
        ),
        # Only the levels that carry duty, as in the step.
        hpr_hot_streams={s.name: s for s in _carrying(solved.hpr_hot_streams)},
        hpr_cold_streams={s.name: s for s in _carrying(solved.hpr_cold_streams)},
    )
    # The zone the streams join scales each one's contribution by its own
    # multiplier; the target stood the heat pump HPR_DT_CONT off, and the
    # air its own contribution, as given. The retrofit's model scales them
    # once more, by the existing network's multiplier.
    multiplier = _zone_multiplier(
        problem, _heat_pump_zone_address(problem, target, None)
    ) * float(dt_cont_multiplier)
    return _case_with_heat_pump_streams(
        problem,
        target,
        dt_cont=float(root.config.hpr.dt_cont) / multiplier,
        ambient_scale=1.0 / multiplier,
        machine=machine,
    )


def _carrying(streams) -> tuple:
    """The streams of ``streams`` that carry duty."""

    return tuple(
        stream
        for _key, stream in streams.items()
        if float(stream.heat_flow.value) > 0.0
    )


def _name_streams_for(solved, machine: str, taken) -> None:
    """Rename the streams of the target ``solved`` after ``machine`` where
    one of their names is among ``taken``, the stream names the problem
    already holds.

    Every heat pump target names its streams alike (``HP_H1``, ``HP_C1``,
    ``AIR_H1``), so the step of a network that already holds a machine
    would write streams the existing machine's already name. They become
    ``<machine>_H1``, ``<machine>_C1`` and ``<machine>_AIR_H1``, in every
    period's outputs too; with no clash the names stay.
    """
    streams = [
        stream
        for group in (solved.hpr_hot_streams, solved.hpr_cold_streams)
        for _key, stream in group.items()
    ]
    streams += [stream for _key, stream in solved.amb_streams.items()]
    for output in (getattr(solved, "period_outputs", None) or {}).values():
        for group in ("hpr_hot_streams", "hpr_cold_streams"):
            streams += [s for _key, s in ((output or {}).get(group) or {}).items()]
    if not any(stream.name in taken for stream in streams):
        return
    for stream in streams:
        if not stream.name.startswith(f"{machine}_"):
            stream.name = f"{machine}_{stream.name.removeprefix('HP_')}"


def heat_pump_step(
    problem,
    base,
    settings,
    *,
    limits=None,
    machine: str = "HP",
    load_fraction=None,
    taken=frozenset(),
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
        cycle=settings.heat_pump_cycle,
        # The coolers and heaters shifted as the retrofit's model holds them.
        dt_cont_multiplier=settings.dt_cont_multiplier,
    )
    if solved is None:
        # The heaters carry no load: no heat pump to add.
        return None, None
    _name_streams_for(solved, machine, taken)
    # A level the target left without duty is not a stream: it would need
    # a match it may not be able to have.
    condensers = _carrying(solved.hpr_hot_streams)
    evaporators = _carrying(solved.hpr_cold_streams)
    # The air the evaporators draw on; air that takes heat the heat pump
    # leaves is not a stream of the case.
    ambient_sources = tuple(
        stream
        for _key, stream in solved.amb_streams.items()
        if stream.stream_type == StreamType.Hot.value
    )
    in_periods = weights = None
    if getattr(solved, "period_outputs", None):
        # A shared design: a level is a stream if it carries duty in any
        # period, taken from the first it runs in.
        periods = sorted(problem.period_ids, key=problem.period_ids.get)
        weights = {
            period: float(problem.master_zone.weights[problem.period_ids[period]])
            for period in periods
        }
        in_periods, levels = {}, []
        for group in ("hpr_hot_streams", "hpr_cold_streams"):
            each = [
                {
                    stream.name: stream
                    for stream in _carrying(
                        (solved.period_outputs.get(period) or {}).get(group) or {}
                    )
                }
                for period in periods
            ]
            names = list(dict.fromkeys(name for streams in each for name in streams))
            for name in names:
                in_periods[name] = tuple(streams.get(name) for streams in each)
            levels.append(
                tuple(
                    next(s for s in in_periods[name] if s is not None) for name in names
                )
            )
        condensers, evaporators = levels
        # The air of any period, each period's at that period's duty: the
        # solve's own air is the selected period's, which may draw none.
        air = _period_heat_pump_streams(
            SimpleNamespace(
                config=problem.master_zone.config,
                hpr_details=SimpleNamespace(
                    amb_streams=solved.amb_streams,
                    period_outputs=solved.period_outputs,
                ),
            ),
            periods,
        ).get("ambient_source", {})
        in_periods |= air
        ambient_sources = tuple(
            next(s for s in each if s is not None) for each in air.values()
        )
    step = HeatPumpStep(
        record=heat_pump_machine(
            solved,
            name=machine,
            cycle=str(settings.heat_pump_cycle),
            cost_per_kw=settings.heat_pump_cost,
            costing=problem.master_zone.config.costing,
            period_weights=weights,
        ).model_copy(update={"load_fraction": load_fraction}),
        condensers=condensers,
        evaporators=evaporators,
        in_periods=in_periods,
        base=base,
        ambient_sources=ambient_sources,
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
    sizes=(None,),
) -> HeatExchangerNetworkRetrofitResult:
    """The retrofit search of ``existing``, and a second search started
    from its as-piped heat pump added as streams, as one result.

    The second search runs on ``heat_pump_case`` under ``configuration``,
    as the first runs on ``problem``, from ``heat_pump_step``.
    """

    if settings.heat_pump_cycle not in ("carnot", "vapour_compression"):
        # Checked here: the plain search below runs with no cycle, so its
        # own check of the cycle would not see this one.
        raise ValueError(
            "the heat pump stream step takes heat_pump_cycle='carnot' or "
            f"'vapour_compression' only; got {settings.heat_pump_cycle!r}."
        )
    # The stream step replaces the linked heat pump move: the plain search
    # offers none of its own.
    base = run_retrofit_search(
        problem, existing, replace(settings, heat_pump_cycle=None)
    )
    if int(settings.max_modifications) < 1:
        # The heat pump step is a modification.
        return base
    sized = []
    # The names the existing network's machines hold for their streams:
    # the step's streams take others.
    taken = frozenset(
        record["name"]
        for held in getattr(existing, "heat_pumps", ())
        for record in held.streams
    )
    for size in sizes:
        # A target with no machine (none the cycle can model pays), a case
        # that refuses the streams (process components, several process
        # zones) or a network that cannot serve them is a heat pump this
        # retrofit cannot use, not a failed retrofit.
        try:
            step, solved = heat_pump_step(
                problem,
                base.frontier[0],
                settings,
                limits=limits,
                machine=machine,
                load_fraction=size,
                taken=taken,
            )
            if step is None:
                continue
            derived = heat_pump_case(
                problem,
                solved,
                machine=machine,
                dt_cont_multiplier=settings.dt_cont_multiplier,
            )
            # The streams the case added go with the machine, as a designed
            # heat pump network's do: its networks can then be read again
            # from the problem and themselves alone. The step's own only:
            # the problem may hold a machine of the same name already,
            # whose streams its own record carries.
            own = {
                stream.name
                for stream in (
                    *step.condensers,
                    *step.evaporators,
                    *step.ambient_sources,
                )
            }
            step = replace(
                step,
                record=step.record.model_copy(
                    update={
                        "streams": tuple(
                            record
                            for record in heat_pump_stream_records(derived, machine)
                            if record["name"] in own
                        )
                    }
                ),
            )
            case = snapshot_problem(derived)
            with temporary_zone_configuration(
                case._build_execution_master_zone(), configuration
            ):
                sized.append(
                    run_retrofit_search(
                        case,
                        existing,
                        replace(settings, heat_pump_cycle=None),
                        heat_pump=step,
                    )
                )
        except ValueError as error:
            label = "" if size is None else f" at size {size:g}"
            warnings.warn(
                f"the heat pump step{label} was left out of the retrofit: {error}",
                UserWarning,
                stacklevel=2,
            )
    return joined(base, *sized) if sized else base
