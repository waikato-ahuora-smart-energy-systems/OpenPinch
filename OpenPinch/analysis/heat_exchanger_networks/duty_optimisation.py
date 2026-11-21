"""Duty allocation for a user-defined heat exchanger network structure.

The user fixes the network: recovery exchangers (hot stream, cold stream,
stage) plus heaters and coolers. A stream may have several utility exchangers,
one per utility and position, and a utility exchanger with a stage sits just
after its stream leaves that stage (without one it sits at the stream end).
Only exchanger duties and the split fractions of streams shared by several
recovery matches in a stage are optimised.

Objectives:

``"utility"``
    Minimise total utility use subject to a minimum approach temperature on
    each exchanger (a global value, per-exchanger values, or the stream
    temperature contributions when neither is given).
``"area"``
    Minimise the total of the exchangers' common areas subject to a maximum
    hot and/or cold utility duty in every period. Each exchanger has one area
    for all periods, at least what any period needs; a period needing less
    runs with a bypass.
``"cost"``
    Minimise total annual cost: exchanger capital on the common areas
    (``COSTING_HX_*`` settings) plus utility cost. Only feasibility is
    imposed: a small positive approach (1 K by default) at both ends of every
    exchanger.

A listed exchanger that ends at zero duty in every period is removed, together
with its approach constraint, and the problem is solved again until every
remaining exchanger carries duty. Removed exchangers are named in the result's
``summary_metrics["removed_exchangers"]``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ...contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ...contracts.synthesis.topology import HeatExchangerNetworkTopologyRestriction
from ...domain._heat_exchanger.period_state import HeatExchangerPeriodState
from ...domain.enums import (
    HeatExchangerKind,
    HeatExchangerNetworkDesignMethod,
    StreamID,
)
from ...domain.heat_exchanger import HeatExchanger
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .context import finalise_design_result, prepare_service_context
from .execution.executor import SynthesisExecutor, _failed_task_outcome
from .execution.search import reference_dt_cont_multiplier
from .execution.settings import (
    SynthesisWorkflowSettings,
    configured_stage_solver,
    stage_solver_options,
)
from .models.fixed_structure import (
    COOLER,
    HEATER,
    FixedStructureSpec,
    RecoveryMatch,
    UtilityMatch,
)
from .prepared_problem import SynthesisProblem
from .reporting.verification import _TEMPERATURE_ABS_TOL, verify_network_feasibility
from .solver.utility_slices import (
    slice_label,
    slice_labels,
    slice_parent,
    utility_slices,
    with_named_slices,
)
from .targeting._stage import run_single_method_workflow, run_stage

DutyObjective = Literal["utility", "area", "cost", "rating"]

DUTY_OBJECTIVE_GOALS: dict[str, str] = {
    "utility": "total utility",
    "area": "total area",
    "cost": "total cost",
    "rating": "rating",
}
DEFAULT_FEASIBILITY_APPROACH = 1.0
"""Minimum approach (K) imposed by the ``"cost"`` objective when none is given."""

_METHOD = HeatExchangerNetworkDesignMethod.NetworkEvolution


@dataclass(frozen=True)
class DutyOptimisationRequest:
    """Validated user request for one fixed-structure duty optimisation."""

    network: HeatExchangerNetwork
    objective: str
    min_approach_temperature: float | None = None
    exchanger_approach_temperatures: Mapping[str, float] = field(default_factory=dict)
    max_hot_utility: float | None = None
    max_cold_utility: float | None = None
    # Utility name to per-period duty: each utility's exchangers together
    # carry its target in every period.
    level_targets: Mapping[str, tuple[float, ...]] | None = None
    # The dt_cont multiplier the contributions are scaled by (the network's
    # own); None is the reference multiplier.
    dt_cont_multiplier: float | None = None

    @property
    def minimisation_goal(self) -> str:
        return DUTY_OBJECTIVE_GOALS[self.objective]

    def task_settings(self) -> dict[str, Any]:
        """Return JSON-safe task settings recorded with the synthesis task."""
        return {
            "fixed_structure": True,
            "duty_objective": self.objective,
            "min_approach_temperature": self.min_approach_temperature,
            "exchanger_approach_temperatures": dict(
                sorted(self.exchanger_approach_temperatures.items())
            ),
            "max_hot_utility": self.max_hot_utility,
            "max_cold_utility": self.max_cold_utility,
        }


def build_duty_optimisation_request(
    network: HeatExchangerNetwork,
    *,
    objective: str,
    min_approach_temperature: float | None = None,
    exchanger_approach_temperatures: Mapping[str, float] | None = None,
    max_hot_utility: float | None = None,
    max_cold_utility: float | None = None,
    problem_approach: bool = False,
    level_targets: Mapping[str, Sequence[float]] | None = None,
    dt_cont_multiplier: float | None = None,
) -> DutyOptimisationRequest:
    """Validate public arguments without touching the problem or a solver.

    With ``problem_approach`` and no ``min_approach_temperature``, the
    ``"cost"`` and ``"rating"`` objectives keep each exchanger at its
    streams' temperature contributions (the case's minimum approach) instead
    of 1 K.
    """

    if not isinstance(network, HeatExchangerNetwork):
        raise TypeError("network must be a HeatExchangerNetwork.")
    if objective not in DUTY_OBJECTIVE_GOALS:
        allowed = ", ".join(repr(name) for name in DUTY_OBJECTIVE_GOALS)
        raise ValueError(f"objective must be one of {allowed}; got {objective!r}.")
    if not _structural_exchangers(network):
        raise ValueError("network must contain at least one allowed exchanger.")

    approach = _optional_positive(min_approach_temperature, "min_approach_temperature")
    per_exchanger = _exchanger_approaches(network, exchanger_approach_temperatures)
    hot_cap = _optional_non_negative(max_hot_utility, "max_hot_utility")
    cold_cap = _optional_non_negative(max_cold_utility, "max_cold_utility")

    if objective == "area" and hot_cap is None and cold_cap is None:
        raise ValueError(
            "the 'area' objective needs max_hot_utility and/or max_cold_utility; "
            "without a utility cap the minimum-area allocation does no heat "
            "recovery."
        )
    if objective == "rating":
        if hot_cap is not None or cold_cap is not None:
            raise ValueError(
                "the 'rating' objective minimises utility cost at the installed "
                "areas; utility caps apply to the 'area' objective only."
            )
        missing = [
            _label(exchanger)
            for exchanger in _structural_exchangers(network)
            if not (exchanger.area or 0.0) > 0.0
        ]
        if missing:
            raise ValueError(
                "the 'rating' objective needs an installed area on every "
                "exchanger; none on " + ", ".join(missing) + "."
            )
        if approach is None and not problem_approach:
            approach = DEFAULT_FEASIBILITY_APPROACH
    if objective == "cost":
        if hot_cap is not None or cold_cap is not None:
            raise ValueError(
                "the 'cost' objective trades utility against area itself; "
                "utility caps apply to the 'area' objective only."
            )
        if approach is None and not problem_approach:
            approach = DEFAULT_FEASIBILITY_APPROACH
    if objective == "utility" and (hot_cap is not None or cold_cap is not None):
        raise ValueError(
            "the 'utility' objective minimises utility; utility caps apply to "
            "the 'area' objective only."
        )
    return DutyOptimisationRequest(
        network=network,
        objective=objective,
        min_approach_temperature=approach,
        exchanger_approach_temperatures=per_exchanger,
        max_hot_utility=hot_cap,
        max_cold_utility=cold_cap,
        level_targets=(
            None
            if level_targets is None
            else {
                name: tuple(float(duty) for duty in duties)
                for name, duties in level_targets.items()
            }
        ),
        dt_cont_multiplier=dt_cont_multiplier,
    )


@dataclass(frozen=True)
class FixedNetworkStructure:
    """Solver-index view of the user's exchangers for the fixed-structure model.

    ``positions`` are indices into the network's allowed exchangers (those kept
    in this solve); ``slots`` say where each one sits in ``spec``:
    ``("recovery", r)`` or ``("utility", e)``.
    """

    spec: FixedStructureSpec
    positions: tuple[int, ...]
    slots: tuple[tuple[str, int], ...]

    @property
    def stage_count(self) -> int:
        return self.spec.stage_count

    def results(self, model) -> list:
        """Return the solved exchanger results in ``positions`` order."""
        return [
            model.recovery_results[index]
            if kind == "recovery"
            else model.utility_results[index]
            for kind, index in self.slots
        ]


def fixed_network_structure(
    request: DutyOptimisationRequest,
    axis_maps: Mapping[str, Mapping[str, int]],
    *,
    excluded: frozenset[int] = frozenset(),
    warm_duties: Mapping[int, float] | None = None,
) -> FixedNetworkStructure:
    """Map the user's exchangers onto solver indices and validate the layout.

    ``excluded`` lists allowed-exchanger positions removed after an earlier
    solve left them at zero duty; ``warm_duties`` maps positions to starting
    duties (defaults to the duties carried by the network itself).
    """

    hot_axis = axis_maps["hot_process_streams"]
    cold_axis = axis_maps["cold_process_streams"]
    hot_utility_axis = axis_maps["hot_utilities"]
    cold_utility_axis = axis_maps["cold_utilities"]
    exchangers = _structural_exchangers(request.network)
    stage_count = _stage_count(request.network, exchangers)

    recovery: list[RecoveryMatch] = []
    utilities: list[UtilityMatch] = []
    positions: list[int] = []
    slots: list[tuple[str, int]] = []
    recovery_approach: dict[int, float] = {}
    utility_approach: dict[int, float] = {}
    recovery_duties: dict[int, float] = {}
    utility_duties: dict[int, float] = {}
    recovery_areas: dict[int, float] = {}
    utility_areas: dict[int, float] = {}
    recovery_points: dict[int, dict[int, tuple]] = {}
    utility_points: dict[int, dict[int, tuple]] = {}
    # Solver period index by period id; without one, the network's own.
    periods = axis_maps.get("periods")

    for position, exchanger in enumerate(exchangers):
        if position in excluded:
            continue
        label = _label(exchanger)
        approach = request.exchanger_approach_temperatures.get(
            exchanger.exchanger_id or ""
        )
        duty = (
            warm_duties.get(position)
            if warm_duties is not None
            else max(state.duty for state in exchanger.period_states)
        )
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            match = RecoveryMatch(
                hot=_resolve_stream(
                    exchanger.source_stream, hot_axis, "hot process", label
                ),
                cold=_resolve_stream(
                    exchanger.sink_stream, cold_axis, "cold process", label
                ),
                stage=int(exchanger.stage) - 1,
            )
            if match in recovery:
                raise ValueError(
                    f"{label} duplicates another recovery exchanger on the same "
                    "hot stream, cold stream and stage."
                )
            index = len(recovery)
            recovery.append(match)
            slots.append(("recovery", index))
            if approach is not None:
                recovery_approach[index] = approach
            if duty:
                recovery_duties[index] = float(duty)
            if exchanger.area is not None:
                recovery_areas[index] = float(exchanger.area)
                recovery_points[index] = _design_points(exchanger, periods)
        else:
            heater = exchanger.kind is HeatExchangerKind.HOT_UTILITY
            # A sliced utility runs on the slice its id names, when the axis
            # has it (``with_named_slices``).
            sliced = slice_label(exchanger)
            if heater:
                utility = _resolve_stream(
                    sliced if sliced in hot_utility_axis else exchanger.source_stream,
                    hot_utility_axis,
                    "hot utility",
                    label,
                )
                stream = _resolve_stream(
                    exchanger.sink_stream, cold_axis, "cold process", label
                )
                default_stage = 1
            else:
                stream = _resolve_stream(
                    exchanger.source_stream, hot_axis, "hot process", label
                )
                utility = _resolve_stream(
                    sliced if sliced in cold_utility_axis else exchanger.sink_stream,
                    cold_utility_axis,
                    "cold utility",
                    label,
                )
                default_stage = stage_count
            stage = exchanger.stage if exchanger.stage is not None else default_stage
            match = UtilityMatch(
                side=HEATER if heater else COOLER,
                stream=stream,
                utility=utility,
                after_stage=int(stage) - 1,
            )
            if match in utilities:
                raise ValueError(
                    f"{label} duplicates another exchanger with the same utility "
                    "at the same position on the same stream."
                )
            index = len(utilities)
            utilities.append(match)
            slots.append(("utility", index))
            if approach is not None:
                utility_approach[index] = approach
            if duty:
                utility_duties[index] = float(duty)
            if exchanger.area is not None:
                utility_areas[index] = float(exchanger.area)
                utility_points[index] = _design_points(exchanger, periods)
        positions.append(position)

    _require_every_stream_served(hot_axis, cold_axis, recovery, utilities)
    level_holds = _level_holds(request, exchangers, slots, positions)
    return FixedNetworkStructure(
        spec=FixedStructureSpec(
            stage_count=stage_count,
            recovery=tuple(recovery),
            utilities=tuple(utilities),
            recovery_approach=recovery_approach,
            utility_approach=utility_approach,
            initial_recovery_duties=recovery_duties,
            initial_utility_duties=utility_duties,
            recovery_areas=recovery_areas,
            utility_areas=utility_areas,
            recovery_design_points=recovery_points,
            utility_design_points=utility_points,
            level_holds=level_holds,
        ),
        positions=tuple(positions),
        slots=tuple(slots),
    )


def _level_holds(request, exchangers, slots, positions) -> tuple:
    """Each held level's utility exchangers (by utility index) and targets.

    A level is matched to the heaters (by source) or coolers (by sink) that
    name its utility; a level with duty to carry but no exchanger raises.
    """
    if not request.level_targets:
        return ()
    units: dict[str, list[int]] = {name: [] for name in request.level_targets}
    for position, (kind, index) in zip(positions, slots, strict=True):
        if kind != "utility":
            continue
        exchanger = exchangers[position]
        name = (
            exchanger.source_stream
            if exchanger.kind is HeatExchangerKind.HOT_UTILITY
            else exchanger.sink_stream
        )
        # A unit the fit's padding named by its slice counts for its parent.
        name = slice_parent(name, units)
        if name in units:
            units[name].append(index)
    holds = []
    for name, targets in request.level_targets.items():
        if not units[name]:
            if any(abs(duty) > 1e-9 for duty in targets):
                raise ValueError(
                    f"utility level {name!r} has a target but no exchanger."
                )
            continue
        holds.append((tuple(units[name]), tuple(targets)))
    return tuple(holds)


def build_solution_network(
    request: DutyOptimisationRequest,
    structure: FixedNetworkStructure,
    model,
    solver_arrays,
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    removed: Sequence[str] = (),
) -> HeatExchangerNetwork:
    """Convert the solved fixed-structure model into a design network."""

    axis_maps = solver_arrays.axis_maps
    names = {axis: _axis_names(axis_maps[axis]) for axis in axis_maps}
    period_ids = tuple(str(value) for value in solver_arrays.arrays["period_ids"])
    user_exchangers = _structural_exchangers(request.network)
    spec = structure.spec
    # Each slice reported under its parent utility, as synthesis does.
    parents = {
        label: parent
        for side in utility_slices(solver_arrays).values()
        for label, (parent, _share) in side.items()
    }

    exchangers: list[HeatExchanger] = []
    for position, (kind, index), result in zip(
        structure.positions, structure.slots, structure.results(model), strict=True
    ):
        user = user_exchangers[position]
        if kind == "recovery":
            match = spec.recovery[index]
            source = names["hot_process_streams"][match.hot]
            sink = names["cold_process_streams"][match.cold]
            exchanger_kind = HeatExchangerKind.RECOVERY
            roles = (StreamID.Process, StreamID.Process)
            stage: int | None = match.stage + 1
            default_id = f"recovery:{source}->{sink}:S{stage}"
        else:
            match = spec.utilities[index]
            stage = user.stage
            if match.side == HEATER:
                source = parents.get(
                    names["hot_utilities"][match.utility],
                    names["hot_utilities"][match.utility],
                )
                sink = names["cold_process_streams"][match.stream]
                exchanger_kind = HeatExchangerKind.HOT_UTILITY
                roles = (StreamID.Utility, StreamID.Process)
                default_id = f"hot-utility:{source}->{sink}"
            else:
                source = names["hot_process_streams"][match.stream]
                sink = parents.get(
                    names["cold_utilities"][match.utility],
                    names["cold_utilities"][match.utility],
                )
                exchanger_kind = HeatExchangerKind.COLD_UTILITY
                roles = (StreamID.Process, StreamID.Utility)
                default_id = f"cold-utility:{source}->{sink}"
            if stage is not None:
                default_id += f":S{stage}"
        states = tuple(
            HeatExchangerPeriodState(
                period_id=period_ids[n],
                period_idx=n,
                duty=period.duty,
                active=period.active,
                approach_temperatures=tuple(max(0.0, t) for t in period.approach),
                source_split_fraction=_fraction(period.source_split),
                sink_split_fraction=_fraction(period.sink_split),
                source_inlet_temperature=period.source_inlet,
                source_outlet_temperature=period.source_outlet,
                sink_inlet_temperature=period.sink_inlet,
                sink_outlet_temperature=period.sink_outlet,
            )
            for n, period in enumerate(result.periods)
        )
        slices = tuple(s for period in result.periods for s in period.slices)
        exchangers.append(
            HeatExchanger(
                exchanger_id=user.exchanger_id or default_id,
                kind=exchanger_kind,
                source_stream=source,
                sink_stream=sink,
                source_stream_role=roles[0],
                sink_stream_role=roles[1],
                stage=stage,
                period_states=states,
                # An installed (rated) area may exceed the slices' area.
                area=result.area
                if request.objective == "rating" or not slices
                else None,
                capital_cost=result.capital_cost,
                segment_area_contributions=slices,
            )
        )

    summary: dict[str, float | int | str | bool | None] = {
        "hot_utility_load": _weighted_load(model, exchangers, "hot"),
        "cold_utility_load": _weighted_load(model, exchangers, "cold"),
        "recovery_load": _weighted_load(model, exchangers, "recovery"),
        **_unit_counts(exchangers),
        "fixed_structure": True,
        "duty_objective": request.objective,
        "total_area": float(model.total_area),
        "removed_exchanger_count": len(removed),
        "removed_exchangers": ", ".join(removed),
    }
    if request.min_approach_temperature is not None:
        summary["approach_temperature"] = float(request.min_approach_temperature)
    network = HeatExchangerNetwork(
        exchangers=tuple(exchangers),
        run_id=run_id,
        task_id=task_id,
        method=_METHOD,
        stage_count=spec.stage_count,
        total_annual_cost=float(model.TAC),
        utility_cost=float(model.utility_cost_value),
        capital_cost=float(model.capital_cost_value),
        summary_metrics=summary,
        solver_axis_metadata={"axis_maps": axis_maps},
        source_metadata=_source_metadata(model, solver_arrays)
        | (
            {
                "utility_level_targets": {
                    k: list(v) for k, v in request.level_targets.items()
                }
            }
            if request.level_targets
            else {}
        ),
    )
    return network.model_copy(
        update={"objective_value": _objective_value(request.objective, network)}
    )


class FixedStructureDutyExecutor:
    """Executor that solves one fixed-structure duty-allocation task.

    After each solve, a listed exchanger left at zero duty in every period is
    removed (with its approach constraint) and the problem is solved again,
    until every remaining exchanger carries duty.
    """

    def __init__(
        self,
        request: DutyOptimisationRequest,
        *,
        print_output: bool = False,
        model_factory: Any | None = None,
    ) -> None:
        self.request = request
        self.print_output = print_output
        self.model_factory = model_factory
        self.models_by_task_id: dict[str, Any] = {}

    def execute(
        self,
        tasks: Sequence[HeatExchangerNetworkSynthesisTask],
        *,
        problem,
        parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
        max_parallel: int,
    ) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
        del parent_outcomes, max_parallel
        return tuple(self._solve(task, problem) for task in tasks)

    def _solve(
        self,
        task: HeatExchangerNetworkSynthesisTask,
        problem,
    ) -> HeatExchangerNetworkSynthesisTaskOutcome:
        from .models.fixed_structure import FixedStructureModel
        from .solver.arrays import (
            nominal_approach_temperature,
            problem_to_solver_arrays,
        )

        request = self.request
        hens = problem.master_zone.config.hens
        factory = self.model_factory or FixedStructureModel
        solver = configured_stage_solver(hens, "HENS_SOLVER_EVM")
        solver_options = stage_solver_options(hens, "evm", solver)
        user_exchangers = _structural_exchangers(request.network)
        excluded: set[int] = set()
        removed: list[str] = []
        warm: dict[int, float] | None = None
        try:
            arrays = with_named_slices(
                problem_to_solver_arrays(
                    problem,
                    nominal_approach_temperature(
                        problem.master_zone.config, task.dt_cont_multiplier
                    ),
                ),
                slice_labels(user_exchangers),
            )
            while True:
                structure = fixed_network_structure(
                    request,
                    arrays.axis_maps,
                    excluded=frozenset(excluded),
                    warm_duties=warm,
                )
                model = factory(
                    name=f"fixed-structure-{request.objective}",
                    solver=solver,
                    solver_arrays=arrays,
                    spec=structure.spec,
                    minimisation_goal=request.minimisation_goal,
                    default_approach=request.min_approach_temperature,
                    max_hot_utility=request.max_hot_utility,
                    max_cold_utility=request.max_cold_utility,
                    tol=float(hens.solve_tolerance),
                    solver_options=dict(solver_options),
                )
                model.optimise(print_output=self.print_output)
                if getattr(model, "mSuccess", 0) != 1:
                    reason = _solver_failure(model)
                    if removed:
                        reason += (
                            " (after removing zero-duty exchangers: "
                            + ", ".join(removed)
                            + ")"
                        )
                    return _failed_task_outcome(task, reason)
                results = structure.results(model)
                idle = [
                    (result.max_duty, position)
                    for position, result in zip(structure.positions, results)
                    if not result.active
                ]
                if not idle:
                    break
                _duty, position = min(idle)
                excluded.add(position)
                removed.append(_display_id(user_exchangers[position]))
                warm = {
                    position: result.max_duty
                    for position, result in zip(structure.positions, results)
                }
            network = build_solution_network(
                request,
                structure,
                model,
                arrays,
                run_id=task.run_id,
                task_id=task.task_id,
                removed=removed,
            )
            allowances = getattr(model, "rating_allowances", None)
            if allowances:
                # A rating's cap is calibrated with Chen's LMTD at the design
                # point; off it, exact sizing can need a little more area. It
                # is judged as the model rated it, as retrofit does, and the
                # largest allowance is recorded.
                network.summary_metrics["rating_area_allowance"] = max(
                    allowances.values()
                )
                network.source_metadata["rating_area_allowances"] = {
                    exchanger.exchanger_id: allowances.get(slot, 1.0)
                    for exchanger, slot in zip(
                        network.exchangers, structure.slots, strict=True
                    )
                    if exchanger.exchanger_id is not None
                }
                failures = verify_solution(
                    structure, model, _with_allowances(network, structure, allowances)
                )
            else:
                failures = verify_solution(structure, model, network)
        except ValueError as exc:
            return _failed_task_outcome(task, str(exc))
        if failures:
            # An unconverged point that fails verification keeps its
            # termination, and its error says to raise the solver limit: a
            # failure summary reads only the error.
            diagnostics = _unconverged_diagnostics(model)
            error = "verification failed: " + "; ".join(failures)
            termination = diagnostics.get("unconverged_termination")
            if termination is not None:
                error += (
                    f" (the solver stopped at a limit ({termination}) before "
                    "converging: raise max_iter or max_wall_time in "
                    "HENS_SOLVER_OPTIONS_EVM)"
                )
            return _failed_task_outcome(task, error).model_copy(
                update={"diagnostics": diagnostics}
            )
        if task.task_id is not None:
            self.models_by_task_id[task.task_id] = model
        return HeatExchangerNetworkSynthesisTaskOutcome(
            task=task,
            status="success",
            network=network,
            objective_value=network.objective_value,
            solver_status=_solver_status(model),
            diagnostics=_unconverged_diagnostics(model),
        )


def _with_allowances(
    network: HeatExchangerNetwork,
    structure: FixedNetworkStructure,
    allowances: Mapping[tuple[str, int], float],
) -> HeatExchangerNetwork:
    """``network`` with each exchanger's area widened by its rating
    allowance (``FixedStructureModel.rating_allowances``), for verification
    only: the reported areas stay as installed."""

    return network.model_copy(
        update={
            "exchangers": tuple(
                exchanger
                if exchanger.area is None
                else exchanger.model_copy(
                    update={"area": exchanger.area * allowances.get(slot, 1.0)}
                )
                for exchanger, slot in zip(
                    network.exchangers, structure.slots, strict=True
                )
            )
        }
    )


def verify_solution(
    structure: FixedNetworkStructure,
    model,
    network: HeatExchangerNetwork,
) -> tuple[str, ...]:
    """Return why a solved duty allocation is not an acceptable design.

    The solver accepts loosely converged points, so the solution is checked
    against each exchanger's own required approach (the network metadata only
    carries the smallest one) and then the reporting-layer feasibility checks
    (exchanger and stream heat balances to target temperatures).
    """

    failures: list[str] = []
    for exchanger, (kind, index), result in zip(
        network.exchangers, structure.slots, structure.results(model), strict=True
    ):
        limits = model.recovery_dt if kind == "recovery" else model.utility_dt
        for n, period in enumerate(result.periods):
            if not period.active or not period.approach:
                continue
            approach = min(period.approach)
            required = float(limits[n][index])
            if approach + _TEMPERATURE_ABS_TOL < required:
                failures.append(
                    f"{exchanger.exchanger_id} approach "
                    f"{approach:.6g} is below its required approach "
                    f"{required:.6g} in period {n}"
                )
    failures.extend(verify_network_feasibility(network))
    return tuple(failures)


def duty_optimisation_task(
    request: DutyOptimisationRequest,
    settings: SynthesisWorkflowSettings,
) -> HeatExchangerNetworkSynthesisTask:
    """Build the single synthesis task that records this request."""

    exchangers = _structural_exchangers(request.network)
    restrictions = tuple(
        HeatExchangerNetworkTopologyRestriction(
            source_stream=exchanger.source_stream,
            sink_stream=exchanger.sink_stream,
            stage=int(exchanger.stage),
            duty=max(state.duty for state in exchanger.period_states),
        )
        for exchanger in exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
    )
    return HeatExchangerNetworkSynthesisTask(
        run_id=settings.run_id,
        method=_METHOD,
        # Contributions at the request's multiplier, else the reference one;
        # a requested minimum approach overrides them in the model.
        dt_cont_multiplier=(
            request.dt_cont_multiplier
            if request.dt_cont_multiplier is not None
            else reference_dt_cont_multiplier(settings.dt_cont_multipliers)
        ),
        stage_count=_stage_count(request.network, exchangers),
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        period_id=settings.period_id,
        settings=request.task_settings(),
        seed_network=request.network,
        seed_network_index=0,
        topology_restrictions=restrictions,
    )


def heat_exchanger_network_duty_optimisation_service(
    problem: SynthesisProblem,
    network: HeatExchangerNetwork,
    *,
    objective: str,
    min_approach_temperature: float | None = None,
    exchanger_approach_temperatures: Mapping[str, float] | None = None,
    max_hot_utility: float | None = None,
    max_cold_utility: float | None = None,
    options: dict[str, Any] | None = None,
    workspace_variant: str | None = None,
    executor: SynthesisExecutor | None = None,
    problem_approach: bool = False,
    level_targets: Mapping[str, Sequence[float]] | None = None,
    dt_cont_multiplier: float | None = None,
) -> HeatExchangerNetworkSynthesisResult:
    """Optimise exchanger duties on a fixed, user-defined network structure."""

    request = build_duty_optimisation_request(
        network,
        objective=objective,
        min_approach_temperature=min_approach_temperature,
        exchanger_approach_temperatures=exchanger_approach_temperatures,
        max_hot_utility=max_hot_utility,
        max_cold_utility=max_cold_utility,
        problem_approach=problem_approach,
        level_targets=level_targets,
        dt_cont_multiplier=dt_cont_multiplier,
    )
    target_output, settings = prepare_service_context(
        problem,
        options=options,
        workspace_variant=workspace_variant,
    )

    def run_method_stage(method_settings: SynthesisWorkflowSettings):
        return run_stage(
            (duty_optimisation_task(request, method_settings),),
            problem=problem,
            settings=method_settings,
            executor=executor,
            default_executor=lambda: FixedStructureDutyExecutor(request),
        )

    workflow_result = run_single_method_workflow(settings, _METHOD, run_method_stage)
    _warn_if_unconverged(workflow_result.outcomes)
    return finalise_design_result(problem, target_output, workflow_result)


def _warn_if_unconverged(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> None:
    """Warn when the returned allocation stopped at a solver limit.

    It passed verification, so it is kept (unlike a synthesis task's), but its
    objective is not a converged optimum.
    """
    for outcome in outcomes:
        termination = outcome.diagnostics.get("unconverged_termination")
        if outcome.status == "success" and termination is not None:
            warnings.warn(
                "the fixed-structure duty optimisation stopped at a solver "
                f"limit ({termination}) before converging; its duties are "
                "feasible but may not be optimal. Raise the limit (max_iter or "
                "max_wall_time for Ipopt) in HENS_SOLVER_OPTIONS_EVM.",
                UserWarning,
                stacklevel=2,
            )


def _design_points(exchanger: HeatExchanger, periods) -> dict[int, tuple]:
    """Each active period's design point, by solver period index.

    ``(duty, theta_1, theta_2, required area or None)``: the end approaches
    are the recorded ones, else follow from the inlet and outlet temperatures
    (counter-current); the required area is the period's segment slice area
    when the exchanger has slices. Periods the solver arrays lack are left
    out, as are periods with no approaches.
    """
    slice_areas = exchanger.segment_area_by_period
    points: dict[int, tuple] = {}
    for state in exchanger.period_states:
        index = (
            state.period_idx if periods is None else periods.get(str(state.period_id))
        )
        if index is None or not state.active:
            continue
        thetas = tuple(state.approach_temperatures[:2])
        temperatures = (
            state.source_inlet_temperature,
            state.source_outlet_temperature,
            state.sink_inlet_temperature,
            state.sink_outlet_temperature,
        )
        if len(thetas) < 2 and None not in temperatures:
            source_in, source_out, sink_in, sink_out = temperatures
            thetas = (source_in - sink_out, source_out - sink_in)
        if len(thetas) < 2:
            continue
        points[int(index)] = (
            float(state.duty),
            float(thetas[0]),
            float(thetas[1]),
            slice_areas.get(str(state.period_id)),
        )
    return points


def _structural_exchangers(network: HeatExchangerNetwork) -> tuple[HeatExchanger, ...]:
    return tuple(
        exchanger for exchanger in network.exchangers if exchanger.match_allowed
    )


def _stage_count(
    network: HeatExchangerNetwork,
    exchangers: Sequence[HeatExchanger],
) -> int:
    stages = [int(e.stage) for e in exchangers if e.stage is not None]
    highest = max(stages, default=1)
    if network.stage_count is None:
        return highest
    if highest > network.stage_count:
        raise ValueError(
            f"network stage_count is {network.stage_count} but an exchanger is "
            f"placed in stage {highest}."
        )
    return int(network.stage_count)


def _resolve_stream(
    name: str,
    axis: Mapping[str, int],
    role: str,
    label: str,
) -> int:
    if name in axis:
        return axis[name]
    matches = [index for key, index in axis.items() if key.endswith("." + name)]
    if len(matches) == 1:
        return matches[0]
    known = ", ".join(sorted(axis))
    if len(matches) > 1:
        raise ValueError(
            f"{label}: {role} stream {name!r} is ambiguous; use one of: {known}."
        )
    raise ValueError(
        f"{label}: {name!r} is not a {role} stream; expected one of: {known}."
    )


def _require_every_stream_served(
    hot_axis: Mapping[str, int],
    cold_axis: Mapping[str, int],
    recovery: Sequence[RecoveryMatch],
    utilities: Sequence[UtilityMatch],
) -> None:
    served_hot = {m.hot for m in recovery} | {
        m.stream for m in utilities if m.side == COOLER
    }
    served_cold = {m.cold for m in recovery} | {
        m.stream for m in utilities if m.side == HEATER
    }
    missing = [name for name, i in hot_axis.items() if i not in served_hot] + [
        name for name, j in cold_axis.items() if j not in served_cold
    ]
    if missing:
        raise ValueError(
            "every process stream needs at least one exchanger in a fixed "
            "structure; no exchanger serves: " + ", ".join(missing) + "."
        )


def _exchanger_approaches(
    network: HeatExchangerNetwork,
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(
            "exchanger_approach_temperatures must map exchanger_id to kelvin."
        )
    known = {
        exchanger.exchanger_id
        for exchanger in _structural_exchangers(network)
        if exchanger.exchanger_id is not None
    }
    unknown = sorted(str(key) for key in values if key not in known)
    if unknown:
        raise ValueError(
            "exchanger_approach_temperatures names unknown exchangers: "
            + ", ".join(unknown)
            + "."
        )
    return {
        str(key): _positive(value, f"exchanger_approach_temperatures[{key!r}]")
        for key, value in values.items()
    }


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


def _optional_positive(value: Any, name: str) -> float | None:
    return None if value is None else _positive(value, name)


def _optional_non_negative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return number


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _label(exchanger: HeatExchanger) -> str:
    if exchanger.exchanger_id is not None:
        return f"exchanger {exchanger.exchanger_id!r}"
    stage = f" stage {exchanger.stage}" if exchanger.stage is not None else ""
    return (
        f"{exchanger.kind.value} exchanger {exchanger.source_stream!r} -> "
        f"{exchanger.sink_stream!r}{stage}"
    )


def _display_id(exchanger: HeatExchanger) -> str:
    if exchanger.exchanger_id is not None:
        return exchanger.exchanger_id
    stage = f":S{exchanger.stage}" if exchanger.stage is not None else ""
    return f"{exchanger.source_stream}->{exchanger.sink_stream}{stage}"


def _axis_names(axis: Mapping[str, int]) -> list[str]:
    names = [""] * len(axis)
    for name, index in axis.items():
        names[index] = name
    return names


def _fraction(value: float | None) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, float(value)))


def _weighted_load(model, exchangers: Sequence[HeatExchanger], side: str) -> float:
    kind = {
        "hot": HeatExchangerKind.HOT_UTILITY,
        "cold": HeatExchangerKind.COLD_UTILITY,
        "recovery": HeatExchangerKind.RECOVERY,
    }[side]
    weights = [float(w) for w in model.period_weights]
    total = sum(weights) or 1.0
    return (
        sum(
            weights[state.period_idx] * state.duty
            for exchanger in exchangers
            if exchanger.kind is kind
            for state in exchanger.period_states
        )
        / total
    )


def _profile_totals(model, attribute: str) -> list[float] | None:
    by_period = _profile_totals_by_period(model, attribute)
    return None if by_period is None else by_period[0]


def _profile_totals_by_period(model, attribute: str) -> list[list[float]] | None:
    profiles = getattr(model, attribute, None)
    if not profiles:
        return None
    return [[float(profile.total) for profile in row] for row in profiles]


def _source_metadata(model, solver_arrays) -> dict[str, Any]:
    from .solver.arrays import SEGMENT_PROFILE_VERSION

    arrays = solver_arrays.arrays

    def first(name: str) -> list[float]:
        return [float(v) for v in arrays[name][0]]

    def by_period(name: str) -> list[list[float]]:
        return [[float(v) for v in row] for row in arrays[name]]

    recovery_limits = [
        value for row in getattr(model, "recovery_dt", []) for value in row
    ]
    return {
        "solver_model_class": type(model).__name__,
        "solver_model_name": getattr(model, "name", None),
        "solver_framework": "ESM",
        "solver_non_isothermal_model": True,
        "minimum_recovery_approach": (
            min(recovery_limits) if recovery_limits else None
        ),
        "segment_profile_version": SEGMENT_PROFILE_VERSION,
        "hot_stream_heat_capacity_flowrates": first("f_h_period"),
        "hot_stream_heat_capacity_flowrates_by_period": by_period("f_h_period"),
        "cold_stream_heat_capacity_flowrates": first("f_c_period"),
        "cold_stream_heat_capacity_flowrates_by_period": by_period("f_c_period"),
        "hot_stream_heat_transfer_coefficients": first("htc_h_period"),
        "hot_stream_heat_transfer_coefficients_by_period": by_period("htc_h_period"),
        "cold_stream_heat_transfer_coefficients": first("htc_c_period"),
        "cold_stream_heat_transfer_coefficients_by_period": by_period("htc_c_period"),
        "hot_utility_heat_transfer_coefficients": first("htc_hu_period"),
        "hot_utility_heat_transfer_coefficients_by_period": by_period("htc_hu_period"),
        "cold_utility_heat_transfer_coefficients": first("htc_cu_period"),
        "cold_utility_heat_transfer_coefficients_by_period": by_period("htc_cu_period"),
        "hot_stream_supply_temperatures": first("T_h_in_period"),
        "hot_stream_supply_temperatures_by_period": by_period("T_h_in_period"),
        "hot_stream_target_temperatures": first("T_h_out_period"),
        "hot_stream_target_temperatures_by_period": by_period("T_h_out_period"),
        "cold_stream_supply_temperatures": first("T_c_in_period"),
        "cold_stream_supply_temperatures_by_period": by_period("T_c_in_period"),
        "cold_stream_target_temperatures": first("T_c_out_period"),
        "cold_stream_target_temperatures_by_period": by_period("T_c_out_period"),
        "hot_stream_total_duties": _profile_totals(model, "hot_profiles"),
        "hot_stream_total_duties_by_period": _profile_totals_by_period(
            model, "hot_profiles"
        ),
        "cold_stream_total_duties": _profile_totals(model, "cold_profiles"),
        "cold_stream_total_duties_by_period": _profile_totals_by_period(
            model, "cold_profiles"
        ),
    }


def _unit_counts(exchangers: Sequence[HeatExchanger]) -> dict[str, int]:
    def active(kind: HeatExchangerKind | None) -> int:
        return sum(
            1
            for exchanger in exchangers
            if (kind is None or exchanger.kind is kind)
            and any(state.active for state in exchanger.period_states)
        )

    return {
        "recovery_units": active(HeatExchangerKind.RECOVERY),
        "hot_utility_units": active(HeatExchangerKind.HOT_UTILITY),
        "cold_utility_units": active(HeatExchangerKind.COLD_UTILITY),
        "total_units": active(None),
    }


def _objective_value(objective: str, network: HeatExchangerNetwork) -> float | None:
    if objective == "area":
        return float(network.summary_metrics["total_area"])
    if objective == "utility":
        return float(network.summary_metrics["hot_utility_load"]) + float(
            network.summary_metrics["cold_utility_load"]
        )
    return network.total_annual_cost


def _solver_failure(model: Any) -> str:
    solver_run = getattr(model, "solver_run", None)
    reason = getattr(solver_run, "failure_reason", None)
    return str(reason or "solver did not return a successful duty allocation")


def _unconverged_diagnostics(model: Any) -> dict[str, str]:
    solver_run = getattr(model, "solver_run", None)
    if getattr(solver_run, "converged", True):
        return {}
    return {"unconverged_termination": str(solver_run.termination_condition)}


def _solver_status(model: Any) -> str:
    solver_run = getattr(model, "solver_run", None)
    status = getattr(solver_run, "status", None)
    return "success" if status is None else str(status)


__all__ = [
    "DEFAULT_FEASIBILITY_APPROACH",
    "DUTY_OBJECTIVE_GOALS",
    "DutyOptimisationRequest",
    "FixedNetworkStructure",
    "FixedStructureDutyExecutor",
    "build_duty_optimisation_request",
    "duty_optimisation_task",
    "build_solution_network",
    "fixed_network_structure",
    "heat_exchanger_network_duty_optimisation_service",
]
