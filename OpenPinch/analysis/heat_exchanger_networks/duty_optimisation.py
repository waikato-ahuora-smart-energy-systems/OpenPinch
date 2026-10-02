"""Duty allocation for a user-defined heat exchanger network structure.

The user builds a :class:`HeatExchangerNetwork` by hand: recovery exchangers
with a source hot stream, a sink cold stream and a stage, plus any heaters and
coolers. The structure is fixed and only the duty on each exchanger (and the
split fractions where several exchangers share a stream in one stage) are
optimised with the StageWise ESM NLP.

Objectives:

``"utility"``
    Minimise total utility use subject to a minimum approach temperature on
    each exchanger (a global value, per-exchanger values, or the stream
    temperature contributions when neither is given).
``"area"``
    Minimise total heat-transfer area subject to a maximum hot and/or cold
    utility duty.
``"cost"``
    Minimise total annual cost (annualised exchanger capital from the
    ``COSTING_HX_*`` settings plus utility cost). Only feasibility is imposed:
    a small positive approach (1 K by default) at both ends of every
    exchanger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ...contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ...contracts.synthesis.topology import HeatExchangerNetworkTopologyRestriction
from ...domain.enums import HeatExchangerKind, HeatExchangerNetworkDesignMethod
from ...domain.heat_exchanger import HeatExchanger
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .context import finalise_design_result, prepare_service_context
from .execution.executor import SynthesisExecutor, _failed_task_outcome
from .execution.settings import SynthesisWorkflowSettings
from .prepared_problem import SynthesisProblem
from .targeting._stage import run_single_method_workflow, run_stage

DutyObjective = Literal["utility", "area", "cost"]

DUTY_OBJECTIVE_GOALS: dict[str, str] = {
    "utility": "total utility",
    "area": "total area",
    "cost": "total cost",
}
DEFAULT_FEASIBILITY_APPROACH = 1.0
"""Minimum approach (K) imposed by the ``"cost"`` objective when none is given."""

_METHOD = HeatExchangerNetworkDesignMethod.NetworkEvolution

RecoveryKey = tuple[int, int, int]
ExchangerKey = tuple[HeatExchangerKind, int, int, int | None]


@dataclass(frozen=True)
class DutyOptimisationRequest:
    """Validated user request for one fixed-structure duty optimisation."""

    network: HeatExchangerNetwork
    objective: str
    min_approach_temperature: float | None = None
    exchanger_approach_temperatures: Mapping[str, float] = field(default_factory=dict)
    max_hot_utility: float | None = None
    max_cold_utility: float | None = None

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
) -> DutyOptimisationRequest:
    """Validate public arguments without touching the problem or a solver."""

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
    if objective == "cost":
        if hot_cap is not None or cold_cap is not None:
            raise ValueError(
                "the 'cost' objective trades utility against area itself; "
                "utility caps apply to the 'area' objective only."
            )
        if approach is None:
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
    )


@dataclass(frozen=True)
class FixedNetworkStructure:
    """Solver-index view of a user network for the StageWise model."""

    stage_count: int
    hot_count: int
    cold_count: int
    recovery: tuple[RecoveryKey, ...]
    heaters: tuple[int, ...]
    coolers: tuple[int, ...]
    exchanger_keys: tuple[ExchangerKey, ...]
    recovery_approach: dict[RecoveryKey, float]
    hot_utility_approach: dict[int, float]
    cold_utility_approach: dict[int, float]
    initial_recovery_duties: dict[RecoveryKey, float]

    def z_restriction(self) -> list:
        """Return ``[recovery, heaters, coolers]`` in the model restriction shape."""
        recovery = [
            [[0 for _k in range(self.stage_count)] for _j in range(self.cold_count)]
            for _i in range(self.hot_count)
        ]
        for i, j, k in self.recovery:
            recovery[i][j][k] = 1
        heaters = [1 if j in self.heaters else 0 for j in range(self.cold_count)]
        coolers = [1 if i in self.coolers else 0 for i in range(self.hot_count)]
        return [recovery, heaters, coolers]


def fixed_network_structure(
    request: DutyOptimisationRequest,
    axis_maps: Mapping[str, Mapping[str, int]],
) -> FixedNetworkStructure:
    """Map the user's exchangers onto solver indices and validate the layout."""

    hot_axis = axis_maps["hot_process_streams"]
    cold_axis = axis_maps["cold_process_streams"]
    hot_utility_axis = axis_maps["hot_utilities"]
    cold_utility_axis = axis_maps["cold_utilities"]
    exchangers = _structural_exchangers(request.network)
    stage_count = _stage_count(request.network, exchangers)

    recovery: list[RecoveryKey] = []
    heaters: list[int] = []
    coolers: list[int] = []
    keys: list[ExchangerKey] = []
    recovery_approach: dict[RecoveryKey, float] = {}
    hot_utility_approach: dict[int, float] = {}
    cold_utility_approach: dict[int, float] = {}
    initial_duties: dict[RecoveryKey, float] = {}

    for exchanger in exchangers:
        label = _label(exchanger)
        approach = request.exchanger_approach_temperatures.get(
            exchanger.exchanger_id or ""
        )
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            i = _resolve_stream(exchanger.source_stream, hot_axis, "hot process", label)
            j = _resolve_stream(exchanger.sink_stream, cold_axis, "cold process", label)
            k = int(exchanger.stage) - 1
            key: RecoveryKey = (i, j, k)
            if key in recovery:
                raise ValueError(
                    f"{label} duplicates another recovery exchanger on the same "
                    "hot stream, cold stream and stage."
                )
            recovery.append(key)
            keys.append((exchanger.kind, i, j, k))
            if approach is not None:
                recovery_approach[key] = approach
            duty = max(state.duty for state in exchanger.period_states)
            if duty > 0.0:
                initial_duties[key] = duty
        elif exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            _resolve_stream(
                exchanger.source_stream, hot_utility_axis, "hot utility", label
            )
            j = _resolve_stream(exchanger.sink_stream, cold_axis, "cold process", label)
            if j in heaters:
                raise ValueError(
                    f"{label} is a second heater on one cold stream; the "
                    "stage-wise model allows one heater per cold stream."
                )
            heaters.append(j)
            keys.append((exchanger.kind, 0, j, None))
            if approach is not None:
                hot_utility_approach[j] = approach
        else:
            i = _resolve_stream(exchanger.source_stream, hot_axis, "hot process", label)
            _resolve_stream(
                exchanger.sink_stream, cold_utility_axis, "cold utility", label
            )
            if i in coolers:
                raise ValueError(
                    f"{label} is a second cooler on one hot stream; the "
                    "stage-wise model allows one cooler per hot stream."
                )
            coolers.append(i)
            keys.append((exchanger.kind, i, 0, None))
            if approach is not None:
                cold_utility_approach[i] = approach

    _require_every_stream_served(hot_axis, cold_axis, recovery, heaters, coolers)
    return FixedNetworkStructure(
        stage_count=stage_count,
        hot_count=len(hot_axis),
        cold_count=len(cold_axis),
        recovery=tuple(recovery),
        heaters=tuple(heaters),
        coolers=tuple(coolers),
        exchanger_keys=tuple(keys),
        recovery_approach=recovery_approach,
        hot_utility_approach=hot_utility_approach,
        cold_utility_approach=cold_utility_approach,
        initial_recovery_duties=initial_duties,
    )


def map_solution_to_network(
    request: DutyOptimisationRequest,
    structure: FixedNetworkStructure,
    extracted: HeatExchangerNetwork,
) -> HeatExchangerNetwork:
    """Return the solved network with only the user's exchangers and identities.

    ``extracted`` must include inactive exchangers so that zero-duty members of
    the fixed structure are still reported.
    """

    axis_maps = extracted.solver_axis_metadata.get("axis_maps", {})
    solved_by_key: dict[ExchangerKey, HeatExchanger] = {}
    for exchanger in extracted.exchangers:
        key = _solved_key(exchanger, axis_maps)
        if key is not None:
            solved_by_key[key] = exchanger

    mapped: list[HeatExchanger] = []
    for user_exchanger, key in zip(
        _structural_exchangers(request.network), structure.exchanger_keys, strict=True
    ):
        solved = solved_by_key.get(key)
        if solved is None:
            raise ValueError(
                f"solver result is missing {_label(user_exchanger)}; the fixed "
                "structure was not preserved."
            )
        update: dict[str, Any] = {}
        if user_exchanger.exchanger_id is not None:
            update["exchanger_id"] = user_exchanger.exchanger_id
        mapped.append(solved.model_copy(update=update))

    total_area = sum(_exchanger_area(exchanger) for exchanger in mapped)
    summary = dict(extracted.summary_metrics)
    summary.update(_unit_counts(mapped))
    summary.update(
        {
            "fixed_structure": True,
            "duty_objective": request.objective,
            "total_area": total_area,
        }
    )
    if request.min_approach_temperature is not None:
        summary["approach_temperature"] = float(request.min_approach_temperature)
    objective_value = _objective_value(request.objective, mapped, extracted)
    return extracted.model_copy(
        update={
            "exchangers": tuple(mapped),
            "summary_metrics": summary,
            "objective_value": objective_value,
        }
    )


class FixedStructureDutyExecutor:
    """Executor that solves one fixed-structure duty-allocation task."""

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
        self.problems_by_task_id: dict[str, Any] = {}

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
        from .extraction.service import extract_heat_exchanger_network
        from .models.fixed_structure import FixedStructureStageWiseModel
        from .models.problem import InternalHeatExchangerNetworkProblem
        from .solver.arrays import problem_to_solver_arrays

        request = self.request
        hens = problem.master_zone.config.hens
        try:
            arrays = problem_to_solver_arrays(problem, task.approach_temperature)
            structure = fixed_network_structure(request, arrays.axis_maps)
            factory = partial(
                self.model_factory or FixedStructureStageWiseModel,
                default_approach=request.min_approach_temperature,
                recovery_approach=structure.recovery_approach,
                hot_utility_approach=structure.hot_utility_approach,
                cold_utility_approach=structure.cold_utility_approach,
                max_hot_utility=request.max_hot_utility,
                max_cold_utility=request.max_cold_utility,
                initial_recovery_duties=structure.initial_recovery_duties,
            )
            internal = InternalHeatExchangerNetworkProblem(
                solver_arrays=arrays,
                name=f"fixed-structure-{request.objective}-S{structure.stage_count}",
                framework="ESM",
                solver=str(hens.solver_evm),
                dTmin=float(task.approach_temperature),
                z_restriction=structure.z_restriction(),
                minimisation_goal=request.minimisation_goal,
                non_isothermal_model=True,
                integers=False,
                tol=float(hens.solve_tolerance),
                solver_options=dict(hens.solver_options_evm),
                stages=structure.stage_count,
                synthesis_task_id=task.task_id,
            )
            solved = internal.get_solution(
                print_output=self.print_output,
                evolution=False,
                model_factories={"stagewise": factory},
            )
            if solved is None or getattr(solved, "mSuccess", 0) != 1:
                reason = getattr(
                    internal,
                    "solution_failure_reason",
                    _solver_failure(solved),
                )
                return _failed_task_outcome(task, reason)
            is_valid, reasons = solved.verify()
            if not is_valid:
                return _failed_task_outcome(
                    task, "verification failed: " + ", ".join(map(str, reasons))
                )
            extracted = extract_heat_exchanger_network(
                solved,
                arrays,
                run_id=task.run_id,
                task_id=task.task_id,
                method=_METHOD,
                stage_count=structure.stage_count,
                include_inactive=True,
            )
            network = map_solution_to_network(request, structure, extracted)
        except ValueError as exc:
            return _failed_task_outcome(task, str(exc))
        if task.task_id is not None:
            self.problems_by_task_id[task.task_id] = internal
        return HeatExchangerNetworkSynthesisTaskOutcome(
            task=task,
            status="success",
            network=network,
            objective_value=network.objective_value,
            solver_status=_solver_status(internal),
        )


def duty_optimisation_task(
    request: DutyOptimisationRequest,
    settings: SynthesisWorkflowSettings,
) -> HeatExchangerNetworkSynthesisTask:
    """Build the single synthesis task that records this request."""

    exchangers = _structural_exchangers(request.network)
    approach = request.min_approach_temperature or float(
        settings.approach_temperatures[0]
    )
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
        approach_temperature=approach,
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
) -> HeatExchangerNetworkSynthesisResult:
    """Optimise exchanger duties on a fixed, user-defined network structure."""

    request = build_duty_optimisation_request(
        network,
        objective=objective,
        min_approach_temperature=min_approach_temperature,
        exchanger_approach_temperatures=exchanger_approach_temperatures,
        max_hot_utility=max_hot_utility,
        max_cold_utility=max_cold_utility,
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
    return finalise_design_result(problem, target_output, workflow_result)


def _structural_exchangers(network: HeatExchangerNetwork) -> tuple[HeatExchanger, ...]:
    return tuple(
        exchanger for exchanger in network.exchangers if exchanger.match_allowed
    )


def _stage_count(
    network: HeatExchangerNetwork,
    exchangers: Sequence[HeatExchanger],
) -> int:
    stages = [
        int(exchanger.stage)
        for exchanger in exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY and exchanger.stage is not None
    ]
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
    recovery: Sequence[RecoveryKey],
    heaters: Sequence[int],
    coolers: Sequence[int],
) -> None:
    served_hot = {i for i, _j, _k in recovery} | set(coolers)
    served_cold = {j for _i, j, _k in recovery} | set(heaters)
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


def _solved_key(
    exchanger: HeatExchanger,
    axis_maps: Mapping[str, Mapping[str, int]],
) -> ExchangerKey | None:
    hot_axis = axis_maps.get("hot_process_streams", {})
    cold_axis = axis_maps.get("cold_process_streams", {})
    if exchanger.kind is HeatExchangerKind.RECOVERY:
        if exchanger.stage is None:
            return None
        return (
            exchanger.kind,
            hot_axis[exchanger.source_stream],
            cold_axis[exchanger.sink_stream],
            int(exchanger.stage) - 1,
        )
    if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
        return (exchanger.kind, 0, cold_axis[exchanger.sink_stream], None)
    return (exchanger.kind, hot_axis[exchanger.source_stream], 0, None)


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


def _exchanger_area(exchanger: HeatExchanger) -> float:
    if exchanger.area is not None:
        return float(exchanger.area)
    if exchanger.segment_design_area is not None:
        return float(exchanger.segment_design_area)
    return 0.0


def _objective_value(
    objective: str,
    exchangers: Sequence[HeatExchanger],
    extracted: HeatExchangerNetwork,
) -> float | None:
    if objective == "area":
        return sum(_exchanger_area(exchanger) for exchanger in exchangers)
    if objective == "utility":
        return sum(
            max(state.duty for state in exchanger.period_states)
            for exchanger in exchangers
            if exchanger.kind is not HeatExchangerKind.RECOVERY
        )
    return extracted.total_annual_cost


def _solver_failure(solved: Any) -> str:
    solver_run = getattr(solved, "solver_run", None)
    reason = getattr(solver_run, "failure_reason", None)
    return str(reason or "solver did not return a successful duty allocation")


def _solver_status(internal: Any) -> str:
    solver_run = getattr(getattr(internal, "case", None), "solver_run", None)
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
    "fixed_network_structure",
    "heat_exchanger_network_duty_optimisation_service",
    "map_solution_to_network",
]
