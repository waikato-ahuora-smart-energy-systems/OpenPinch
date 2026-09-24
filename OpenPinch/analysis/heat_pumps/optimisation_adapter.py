"""Translate HPR objectives to the reusable scalar optimisation boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from inspect import Parameter, signature
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ...analysis.numerics import g_ineq_penalty
from ...contracts.hpr import (
    HeatPumpTargetInputs,
    HeatPumpTargetOutputs,
    HPRBackendResult,
    HPREvaluationMode,
    HPRFailureCategory,
    HPRFailureDiagnostic,
    HPRFailureSummary,
    HPRSearchBudget,
    HPRSimulationLoopRecord,
    HPRSimulationStageRecord,
    HPRTargetingError,
    HprTargetSimulationRecord,
    HPRTopologyIdentifier,
    MultiPeriodHPRTargetInputs,
)
from ...domain.enums import HeatPumpAndRefrigerationCycle, PenaltyForm
from ...domain.stream_collection import StreamCollection
from ...domain.value import Value
from ...optimisation.errors import NoOptimisationCandidatesError
from ...optimisation.models import (
    OptimisationCandidate,
    OptimisationMethod,
    OptimisationOptions,
    OptimisationProblem,
    OptimisationResult,
)
from ...optimisation.service import run_multistart_minimisation
from .common._shared.streams import get_ambient_air_stream

HPRObjective = Callable[..., HPRBackendResult]
HPRTargetInputs = HeatPumpTargetInputs | MultiPeriodHPRTargetInputs
HPRCandidateSearch = Callable[..., tuple[OptimisationCandidate, ...]]
OptimisationRunner = Callable[..., OptimisationResult]

_FAILED_CANDIDATE_OBJECTIVE = 1e30


class _HPRSearchBudgetExhausted(Exception):
    """Internal control flow; never translated into a physical failure."""


class _HPRSearchCandidates(tuple):
    """Tuple-compatible candidates carrying detached search evidence."""

    def __new__(cls, candidates, diagnostics):
        value = super().__new__(cls, candidates)
        value.diagnostics = diagnostics
        return value

    def __getnewargs__(self):
        return tuple(self), self.diagnostics


@dataclass
class _CachedHPRScalarObjective:
    """Pickle-safe exact-coordinate cache for one HPR search invocation."""

    objective: HPRObjective
    args: HPRTargetInputs
    cache: dict[tuple[float, ...], float] = dataclass_field(default_factory=dict)
    viable: dict[tuple[float, ...], OptimisationCandidate] = dataclass_field(
        default_factory=dict
    )
    exhausted: bool = False
    failure_count: int = 0
    failures: list[HPRFailureDiagnostic] = dataclass_field(default_factory=list)

    def __call__(self, point: Sequence[float] | np.ndarray) -> float:
        key = tuple(
            float(value) for value in np.asarray(point, dtype=float).reshape(-1)
        )
        if not np.isfinite(key).all():
            raise ValueError("HPR candidate coordinates must be finite.")
        if key not in self.cache:
            budget = getattr(self.args, "search_budget", None) or HPRSearchBudget()
            if len(self.cache) >= budget.maximum_evaluations:
                self.exhausted = True
                raise _HPRSearchBudgetExhausted
            result = evaluate_hpr_candidate(
                objective=self.objective,
                point=key,
                args=self.args,
                artifact_mode=HPREvaluationMode.SEARCH,
                debug=False,
            )
            value = float(result.obj)
            if result.success and np.isfinite(value):
                self.viable[key] = OptimisationCandidate(objective=value, point=key)
                # Keep bounded recovery candidates even if a backend discards minima.
                self.viable = {
                    item.point: item for item in sorted(self.viable.values())[:16]
                }
            else:
                value = _FAILED_CANDIDATE_OBJECTIVE
                self.failure_count += 1
                if len(self.failures) < 16:
                    self.failures.append(
                        HPRFailureDiagnostic(
                            category=HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY,
                            reason_code="candidate.no_viable_result",
                            summary=_bounded_failure_summary(result.failure_reason),
                            candidate_index=len(self.cache),
                            topology=_hpr_topology_identifier(self.args),
                        )
                    )
            self.cache[key] = value
        return self.cache[key]


def normalise_hpr_penalty_terms(value: object) -> tuple[float, ...]:
    """Return one stable finite tuple for a documented HPR penalty value."""
    if value is None:
        return ()
    if _contains_boolean(value):
        raise TypeError("HPR penalty terms must be numeric and may not be boolean.")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "HPR penalty terms must form a rectangular numeric array."
        ) from exc
    if raw.dtype.kind not in "iuf":
        raise TypeError("HPR penalty terms must form a rectangular numeric array.")
    try:
        flattened = np.asarray(value, dtype=float).reshape(-1, order="C")
    except (TypeError, ValueError) as exc:
        raise TypeError("HPR penalty terms must be numeric.") from exc
    if not np.isfinite(flattened).all():
        raise ValueError("HPR penalty terms must be finite.")
    return tuple(float(term) for term in flattened)


def initialise_hpr_seed(initialiser, args):
    """An optional screening failure must not rule out a physical cycle search."""
    if not args.initialise_simulated_cycle:
        return None
    try:
        return initialiser(args)
    except HPRTargetingError:
        return None


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.bool_)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_boolean(item) for item in value)
    return False


def solve_hpr_placement(
    f_obj: HPRObjective,
    x0_ls: Sequence[float] | np.ndarray | float | None,
    bnds: Sequence[Sequence[float]],
    args: HeatPumpTargetInputs,
    *,
    candidate_search: HPRCandidateSearch | None = None,
) -> HPRBackendResult:
    """Optimise one HPR case and translate the first successful candidate."""
    search = run_hpr_candidate_search if candidate_search is None else candidate_search
    candidates = search(
        objective=f_obj,
        initial_points=x0_ls,
        bounds=bnds,
        args=args,
    )
    if not candidates:
        raise_hpr_targeting_error(
            args=args,
            message=(
                "Heat pump and refrigeration targeting "
                f"({args.hpr_type}) failed to return any local minima."
            ),
            failures=(),
            evaluated_count=0,
            warm_start_evaluated=bool(normalise_initial_points(x0_ls)),
            search_diagnostics=getattr(candidates, "diagnostics", None),
        )

    failures: list[tuple[int, HPRBackendResult]] = []
    for candidate in candidates:
        result = evaluate_hpr_candidate(
            objective=f_obj,
            point=candidate.point,
            args=args,
        )
        if result.success and np.isfinite(float(result.obj)):
            return translate_hpr_result(result, ambient_args=args)
        failures.append((len(failures), result))

    raise_hpr_targeting_error(
        args=args,
        message=(
            "Heat pump and refrigeration targeting "
            f"({args.hpr_type}) failed to return an optimal result."
        ),
        failures=failures,
        evaluated_count=len(candidates),
        warm_start_evaluated=bool(normalise_initial_points(x0_ls)),
        search_diagnostics=getattr(candidates, "diagnostics", None),
    )


def raise_hpr_targeting_error(
    *,
    args: HPRTargetInputs,
    message: str,
    failures: Sequence[tuple[int, HPRBackendResult]],
    evaluated_count: int,
    warm_start_evaluated: bool,
    search_diagnostics: HPRFailureSummary | None = None,
) -> None:
    """Raise one bounded public failure from detached candidate facts."""
    topology = _hpr_topology_identifier(args)
    representatives = tuple(
        HPRFailureDiagnostic(
            category=HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY,
            reason_code="candidate.no_viable_result",
            summary=_bounded_failure_summary(result.failure_reason),
            candidate_index=index,
            topology=topology,
        )
        for index, result in tuple(failures)[:16]
    )
    category = (
        HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY
        if failures
        else HPRFailureCategory.NO_VIABLE_CANDIDATE
    )
    diagnostics = HPRFailureSummary(
        simulation_backend=getattr(args, "simulation_backend", "coolprop"),
        cycle=str(args.hpr_type),
        evaluated_count=evaluated_count,
        category_counts={category: len(failures) if failures else 1},
        representative_failures=representatives,
        budget=getattr(args, "search_budget", None) or HPRSearchBudget(),
        warm_start_evaluated=warm_start_evaluated,
        warm_start_viable=False,
    )
    if search_diagnostics is not None:
        counts = dict(search_diagnostics.category_counts)
        if failures:
            counts[category] = counts.get(category, 0) + len(failures)
        diagnostics = search_diagnostics.model_copy(
            update={
                "evaluated_count": search_diagnostics.evaluated_count + evaluated_count,
                "category_counts": counts,
                "representative_failures": (
                    search_diagnostics.representative_failures + representatives
                )[:16],
            }
        )
    raise HPRTargetingError(message, diagnostics=diagnostics)


def _bounded_failure_summary(reason: str | None) -> str:
    summary = "candidate returned an unsuccessful or non-finite result"
    if reason:
        summary = " ".join(str(reason).split()) or summary
    return summary[:512]


def _hpr_topology_identifier(
    args: HPRTargetInputs,
) -> HPRTopologyIdentifier | None:
    hpr_type = args.hpr_type
    if hpr_type == HeatPumpAndRefrigerationCycle.CascadeVapourComp.value:
        stage_args: HeatPumpTargetInputs | None
        if isinstance(args, MultiPeriodHPRTargetInputs):
            selected_case = next(
                (
                    case
                    for case in args.period_cases
                    if str(case.period_id) == str(args.selected_period_id)
                ),
                None,
            )
            if selected_case is None:
                selected_case = next(
                    (
                        case
                        for case in args.period_cases
                        if int(case.period_idx) == int(args.selected_period_idx)
                    ),
                    None,
                )
            stage_args = selected_case.args if selected_case is not None else None
        else:
            stage_args = args
        if (
            stage_args is not None
            and int(stage_args.n_cond) == 1
            and int(stage_args.n_evap) == 1
        ):
            return HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION
    mapping = {
        HeatPumpAndRefrigerationCycle.CascadeVapourComp.value: (
            HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION
        ),
        HeatPumpAndRefrigerationCycle.ParallelVapourComp.value: (
            HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION
        ),
        HeatPumpAndRefrigerationCycle.VapourCompMVR.value: (
            HPRTopologyIdentifier.VAPOUR_COMPRESSION_MVR
        ),
    }
    return mapping.get(hpr_type)


def run_hpr_candidate_search(
    *,
    objective: HPRObjective,
    initial_points: Sequence[float] | np.ndarray | float | None,
    bounds: Sequence[Sequence[float]],
    args: HPRTargetInputs,
    optimiser: OptimisationRunner = run_multistart_minimisation,
) -> tuple[OptimisationCandidate, ...]:
    """Return ranked backend and warm-start candidates for an HPR objective."""
    starts = normalise_initial_points(initial_points)
    bound_array = np.asarray(bounds, dtype=float)
    if (
        bound_array.ndim != 2
        or bound_array.shape[1:] != (2,)
        or not len(bound_array)
        or not np.isfinite(bound_array).all()
        or np.any(bound_array[:, 0] > bound_array[:, 1])
    ):
        raise ValueError("HPR search requires finite ordered bounds.")
    for point in starts:
        if (
            len(point) != len(bound_array)
            or not np.isfinite(point).all()
            or np.any(point < bound_array[:, 0])
            or np.any(point > bound_array[:, 1])
        ):
            raise ValueError("HPR initial points must be finite and within bounds.")
    cached_scalar = _CachedHPRScalarObjective(objective=objective, args=args)

    warm_start_candidates = []
    problem = OptimisationProblem(
        objective=cached_scalar,
        bounds=tuple((float(lower), float(upper)) for lower, upper in bounds),
        initial_points=starts,
        args=(),
    )
    budget = getattr(args, "search_budget", None) or HPRSearchBudget()
    options = OptimisationOptions(
        # A request-local counter/cache cannot be shared with process workers.
        # Execute restarts serially so the bound includes every restart/polish.
        n_runs=1,
        maxiter=budget.maximum_iterations,
        maxfun=budget.maximum_evaluations,
        local_method=None,
    )
    backend_candidates = []
    warm_start_viable = False
    method = _resolve_hpr_optimisation_method(args.bb_minimiser)
    try:
        for point in starts:
            warm_start_candidates.append(
                OptimisationCandidate(
                    objective=cached_scalar(point),
                    point=point,
                )
            )
            warm_start_viable = warm_start_viable or point in cached_scalar.viable
        for run in range(max(1, int(args.max_multi_start))):
            try:
                offset = run % len(starts) if starts else 0
                run_problem = replace(
                    problem, initial_points=starts[offset:] + starts[:offset]
                )
                result = optimiser(
                    run_problem, method=method, options=replace(options, seed=run)
                )
                backend_candidates.extend(result.candidates)
                # The generic polishing pool owns private objective copies and
                # deliberately suppresses exceptions. HPR must share its budget
                # and propagate internal failures, so polish in this invocation.
                for candidate in result.candidates:
                    minimize(
                        cached_scalar,
                        candidate.point,
                        method="SLSQP",
                        bounds=problem.bounds,
                        options={"maxiter": budget.maximum_iterations},
                    )
            except NoOptimisationCandidatesError:
                continue
    except _HPRSearchBudgetExhausted:
        pass

    counts = {}
    if cached_scalar.failure_count:
        counts[HPRFailureCategory.CANDIDATE_PHYSICAL_INFEASIBILITY] = (
            cached_scalar.failure_count
        )
    if cached_scalar.exhausted:
        counts[HPRFailureCategory.BUDGET_EXHAUSTION] = 1
    if not cached_scalar.viable:
        counts[HPRFailureCategory.NO_VIABLE_CANDIDATE] = 1
    diagnostics = HPRFailureSummary(
        simulation_backend=getattr(args, "simulation_backend", "coolprop"),
        cycle=str(args.hpr_type),
        evaluated_count=len(cached_scalar.cache),
        category_counts=counts,
        representative_failures=tuple(cached_scalar.failures),
        budget=budget,
        warm_start_evaluated=bool(warm_start_candidates),
        warm_start_viable=warm_start_viable,
    )
    return _HPRSearchCandidates(
        _merge_ranked_candidates(
            backend_candidates, (*warm_start_candidates, *cached_scalar.viable.values())
        ),
        diagnostics,
    )


def normalise_initial_points(
    values: Sequence[float] | np.ndarray | float | None,
) -> tuple[tuple[float, ...], ...]:
    """Normalise accepted HPR warm-start forms into immutable row vectors."""
    if values is None:
        return ()
    block = np.asarray(values, dtype=float)
    if block.size == 0:
        return ()
    if block.ndim == 0:
        block = block.reshape(1, 1)
    elif block.ndim == 1:
        block = block.reshape(1, -1)
    else:
        block = block.reshape(block.shape[0], -1)
    return tuple(tuple(float(value) for value in row) for row in block)


def _resolve_hpr_optimisation_method(method: Any) -> OptimisationMethod:
    """Resolve one exact configured optimiser identifier or enum value."""
    if isinstance(method, OptimisationMethod):
        return method
    if method is None:
        return OptimisationMethod.DUAL_ANNEALING
    raw_value = getattr(method, "value", method)
    if not isinstance(raw_value, str):
        raise TypeError(
            "Optimizer handle must be a string or enum value; "
            f"got {type(method).__name__}."
        )
    try:
        return OptimisationMethod(raw_value)
    except ValueError:
        supported = ", ".join(item.value for item in OptimisationMethod)
        raise ValueError(
            f"Unsupported optimiser identifier {method!r}. "
            f"Supported identifiers: {supported}."
        ) from None


def evaluate_hpr_candidate(
    *,
    objective: HPRObjective,
    point: Sequence[float] | np.ndarray,
    args: HPRTargetInputs,
    debug: bool | None = None,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    """Evaluate and type-check one HPR candidate without hiding failures."""
    objective_kwargs = {"debug": args.debug if debug is None else debug}
    if _accepts_artifact_mode(objective):
        objective_kwargs["artifact_mode"] = artifact_mode
    result = objective(np.asarray(point, dtype=float), args, **objective_kwargs)
    if not isinstance(result, HPRBackendResult):
        raise TypeError(
            "Heat pump and refrigeration objective functions must return "
            "HPRBackendResult."
        )
    if artifact_mode is HPREvaluationMode.SEARCH:
        return result.with_updates(
            artifacts=None,
            target_simulation_record=None,
            period_outputs=_strip_search_period_artifacts(result.period_outputs),
        )
    return result


def _accepts_artifact_mode(objective: HPRObjective) -> bool:
    try:
        parameters = signature(objective).parameters.values()
    except TypeError, ValueError:
        return False
    return any(
        parameter.name == "artifact_mode" or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _strip_search_period_artifacts(
    period_outputs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if period_outputs is None:
        return None
    return {
        period_id: (
            output.with_updates(artifacts=None, target_simulation_record=None)
            if isinstance(output, HPRBackendResult)
            else output
        )
        for period_id, output in period_outputs.items()
    }


def translate_hpr_result(
    result: HPRBackendResult,
    *,
    ambient_args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Attach parent-level ambient streams to a successful backend result."""
    return result.with_updates(
        success=True,
        amb_streams=get_ambient_air_stream(
            result.Q_amb_hot,
            result.Q_amb_cold,
            ambient_args,
        ),
    )


def translate_hpr_output(result: HPRBackendResult) -> HeatPumpTargetOutputs:
    """Validate one internal backend result as the caller-facing HPR contract."""
    output = HeatPumpTargetOutputs.model_validate(result.to_output_fields())
    _validate_hpr_public_value(output, active=set())
    try:
        output = deepcopy(output)
    except Exception as exc:
        raise TypeError(
            "Public HPR output contains a non-detached runtime artifact."
        ) from exc
    return output


def _validate_hpr_public_value(value: Any, *, active: set[int]) -> None:
    """Fail closed for arbitrary copyable objects in permissive nested fields."""
    if value is None or isinstance(
        value, (str, bool, int, float, np.integer, np.floating, np.bool_)
    ):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise TypeError("Public HPR arrays must contain real numeric values.")
        return
    if isinstance(value, (Value, StreamCollection)):
        # These established domain containers have explicit reporting schemas;
        # their caches and unit registries are not public output fields.
        _validate_hpr_public_value(value.to_dict(), active=active)
        return
    identity = id(value)
    if identity in active:
        raise TypeError("Public HPR output must not contain reference cycles.")
    active.add(identity)
    try:
        if isinstance(
            value,
            (
                HeatPumpTargetOutputs,
                HprTargetSimulationRecord,
                HPRSimulationLoopRecord,
                HPRSimulationStageRecord,
            ),
        ):
            children = (getattr(value, name) for name in type(value).model_fields)
        elif isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Public HPR mappings require string keys.")
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        else:
            raise TypeError(
                f"Public HPR output contains unsupported {type(value).__name__}."
            )
        for child in children:
            _validate_hpr_public_value(child, active=active)
    finally:
        active.remove(identity)


def aggregate_hpr_period_results(
    period_outputs: dict[str, HPRBackendResult],
    weights: Sequence[float] | np.ndarray,
) -> tuple[HPRBackendResult, float]:
    """Apply HPR-specific weighted-operation and peak-capital policies."""
    ordered = list(period_outputs.values())
    if not ordered:
        raise ValueError("At least one HPR period result is required.")
    weights_array = np.asarray(weights, dtype=float)
    if (
        weights_array.shape != (len(ordered),)
        or not np.isfinite(weights_array).all()
        or float(weights_array.sum()) <= 0.0
    ):
        raise ValueError("Period weights must be finite and have a positive sum.")

    updates: dict[str, Any] = {}
    for field in (
        "obj",
        "utility_tot",
        "w_net",
        "Q_ext_heat",
        "Q_ext_cold",
        "feasibility_penalty",
        "Q_amb_hot",
        "Q_amb_cold",
        "w_hpr",
        "w_he",
        "heat_recovery",
        "cop_h",
        "eta_he",
        "Q_cond",
        "Q_evap",
        "Q_cond_he",
        "Q_evap_he",
        "Q_heat",
        "Q_cool",
        "hpr_operating_cost",
    ):
        weighted = _aggregate_result_field(
            ordered,
            field,
            weights=weights_array,
            reducer="weighted",
        )
        if weighted is not None:
            updates[field] = weighted

    for field in (
        "hpr_capital_cost",
        "hpr_annualized_capital_cost",
        "hpr_compressor_capital_cost",
        "hpr_heat_exchanger_capital_cost",
    ):
        maximum = _aggregate_result_field(
            ordered,
            field,
            weights=None,
            reducer="max",
        )
        if maximum is not None:
            updates[field] = maximum

    operating = updates.get("hpr_operating_cost")
    annualized_capital = updates.get("hpr_annualized_capital_cost")
    if operating is not None and annualized_capital is not None:
        try:
            updates["hpr_total_annualized_cost"] = operating + annualized_capital
        except TypeError, ValueError:
            pass
    if "hpr_total_annualized_cost" not in updates:
        weighted_total = _aggregate_result_field(
            ordered,
            "hpr_total_annualized_cost",
            weights=weights_array,
            reducer="weighted",
        )
        if weighted_total is not None:
            updates["hpr_total_annualized_cost"] = weighted_total

    shared_objective = _shared_candidate_objective(ordered, weights_array)
    updates["obj"] = shared_objective
    return ordered[0].with_updates(**updates), shared_objective


def build_hpr_accounting(
    *,
    work: float,
    Q_ext_heat: float,
    Q_ext_cold: float,
    args: HeatPumpTargetInputs,
    penalty_terms: object = None,
    penalise_external_cold_when_refrigerating: bool = False,
) -> tuple[float, float, float, float]:
    """Standardise HPR utility, feasibility-penalty, and objective semantics."""
    positive_penalty_terms = np.maximum(
        np.asarray(normalise_hpr_penalty_terms(penalty_terms), dtype=float),
        0.0,
    )
    penalty = (
        float(
            g_ineq_penalty(
                positive_penalty_terms,
                eta=args.eta_penalty,
                rho=args.rho_penalty,
                form=PenaltyForm.SQUARE,
            )
        )
        if positive_penalty_terms.size
        else 0.0
    )
    if penalise_external_cold_when_refrigerating and not getattr(
        args,
        "is_heat_pumping",
        True,
    ):
        penalty += float(
            g_ineq_penalty(
                g=Q_ext_cold,
                rho=args.rho_penalty,
                form=PenaltyForm.SQUARE,
            )
        )
    objective = calc_hpr_obj(
        work=work,
        Q_ext_heat=Q_ext_heat,
        Q_ext_cold=Q_ext_cold,
        Q_hpr_target=args.Q_hpr_target,
        heat_to_power_ratio=args.heat_to_power_ratio,
        cold_to_power_ratio=args.cold_to_power_ratio,
        penalty=penalty,
    )
    return float(Q_ext_heat), float(Q_ext_cold), penalty, float(objective)


def calc_hpr_obj(
    work: float,
    Q_ext_heat: float,
    Q_ext_cold: float,
    Q_hpr_target: float,
    heat_to_power_ratio: float = 1.0,
    cold_to_power_ratio: float = 0.0,
    penalty: float = 0.0,
) -> float:
    """Return the scalar screening objective used by HPR placement solvers."""
    return (
        work
        + (Q_ext_heat * heat_to_power_ratio)
        + (Q_ext_cold * cold_to_power_ratio)
        + penalty
    ) / Q_hpr_target


def _scalar_hpr_objective(
    point: Sequence[float] | np.ndarray,
    objective: HPRObjective,
    args: HPRTargetInputs,
) -> float:
    result = evaluate_hpr_candidate(
        objective=objective,
        point=point,
        args=args,
        debug=False,
        artifact_mode=HPREvaluationMode.SEARCH,
    )
    if not result.success:
        return _FAILED_CANDIDATE_OBJECTIVE
    value = float(result.obj)
    return value if np.isfinite(value) else _FAILED_CANDIDATE_OBJECTIVE


def _merge_ranked_candidates(
    backend_candidates: Sequence[OptimisationCandidate],
    warm_start_candidates: Sequence[OptimisationCandidate],
) -> tuple[OptimisationCandidate, ...]:
    unique: dict[tuple[float, ...], OptimisationCandidate] = {}
    for candidate in (*backend_candidates, *warm_start_candidates):
        existing = unique.get(candidate.point)
        if existing is None or candidate.objective < existing.objective:
            unique[candidate.point] = candidate
    return tuple(sorted(unique.values()))


def _aggregate_result_field(
    results: list[HPRBackendResult],
    field: str,
    *,
    weights: np.ndarray | None,
    reducer: str,
) -> Any:
    values = [getattr(result, field, None) for result in results]
    if any(value is None for value in values):
        return None
    return _aggregate_values(values, weights=weights, reducer=reducer)


def _aggregate_values(
    values: list[Any],
    *,
    weights: np.ndarray | None,
    reducer: str,
) -> Any:
    first = values[0]
    if isinstance(first, Value):
        unit = first.unit
        magnitudes = []
        for value in values:
            if not isinstance(value, Value):
                return None
            magnitudes.append(float(value.to(unit).value))
        aggregate = (
            float(np.average(magnitudes, weights=weights))
            if reducer == "weighted"
            else float(np.max(magnitudes))
        )
        return Value(aggregate, unit)

    try:
        arrays = [np.asarray(value, dtype=float) for value in values]
    except TypeError, ValueError:
        return None
    if len({array.shape for array in arrays}) != 1:
        return None
    stacked = np.stack(arrays, axis=0)
    aggregate = (
        np.average(stacked, axis=0, weights=weights)
        if reducer == "weighted"
        else np.max(stacked, axis=0)
    )
    if aggregate.ndim == 0:
        return float(aggregate)
    return aggregate


def _shared_candidate_objective(
    results: list[HPRBackendResult],
    weights: np.ndarray,
) -> float:
    has_cost_breakdown = any(
        result.hpr_operating_cost is not None
        or result.hpr_annualized_capital_cost is not None
        for result in results
    )
    if not has_cost_breakdown:
        fallback = _aggregate_result_field(
            results,
            "obj",
            weights=weights,
            reducer="weighted",
        )
        if fallback is None:
            raise ValueError("Shared HPR candidates require finite objectives.")
        return float(fallback)

    if any(
        result.hpr_operating_cost is None or result.hpr_annualized_capital_cost is None
        for result in results
    ):
        raise ValueError(
            "Shared HPR candidates require a complete operating and annualized "
            "capital cost breakdown for every period."
        )

    operating = _aggregate_result_field(
        results,
        "hpr_operating_cost",
        weights=weights,
        reducer="weighted",
    )
    penalty = _aggregate_result_field(
        results,
        "feasibility_penalty",
        weights=weights,
        reducer="weighted",
    )
    annualized_capital = _aggregate_result_field(
        results,
        "hpr_annualized_capital_cost",
        weights=None,
        reducer="max",
    )
    return (
        _annual_cost_magnitude(operating)
        + float(penalty)
        + _annual_cost_magnitude(annualized_capital)
    )


def _annual_cost_magnitude(value: Any) -> float:
    if isinstance(value, Value):
        return float(value.to("$/y").value)
    return float(value)


__all__ = [
    "aggregate_hpr_period_results",
    "build_hpr_accounting",
    "calc_hpr_obj",
    "evaluate_hpr_candidate",
    "normalise_initial_points",
    "normalise_hpr_penalty_terms",
    "raise_hpr_targeting_error",
    "run_hpr_candidate_search",
    "solve_hpr_placement",
    "translate_hpr_result",
    "translate_hpr_output",
]
