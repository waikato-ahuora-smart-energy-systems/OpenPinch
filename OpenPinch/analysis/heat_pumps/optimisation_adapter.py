"""Translate HPR objectives to the reusable scalar optimisation boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ...analysis.numerics import g_ineq_penalty
from ...contracts.hpr import (
    HeatPumpTargetInputs,
    HeatPumpTargetOutputs,
    HPRBackendResult,
    MultiPeriodHPRTargetInputs,
)
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
        raise ValueError(
            "Heat pump and refrigeration targeting "
            f"({args.hpr_type}) failed to return any local minima."
        )

    for candidate in candidates:
        result = evaluate_hpr_candidate(
            objective=f_obj,
            point=candidate.point,
            args=args,
        )
        if result.success and np.isfinite(float(result.obj)):
            return translate_hpr_result(result, ambient_args=args)

    raise ValueError(
        "Heat pump and refrigeration targeting "
        f"({args.hpr_type}) failed to return an optimal result."
    )


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
    problem = OptimisationProblem(
        objective=_scalar_hpr_objective,
        bounds=tuple((float(lower), float(upper)) for lower, upper in bounds),
        initial_points=starts,
        args=(objective, args),
    )
    options = OptimisationOptions(n_runs=max(1, int(args.max_multi_start)))
    try:
        result = optimiser(
            problem,
            method=_resolve_hpr_optimisation_method(args.bb_minimiser),
            options=options,
        )
        backend_candidates = result.candidates
    except NoOptimisationCandidatesError:
        backend_candidates = ()

    warm_start_candidates = tuple(
        OptimisationCandidate(
            objective=_scalar_hpr_objective(point, objective, args),
            point=point,
        )
        for point in starts
    )
    return _merge_ranked_candidates(backend_candidates, warm_start_candidates)


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
) -> HPRBackendResult:
    """Evaluate and type-check one HPR candidate without hiding failures."""
    result = objective(
        np.asarray(point, dtype=float),
        args,
        debug=args.debug if debug is None else debug,
    )
    if not isinstance(result, HPRBackendResult):
        raise TypeError(
            "Heat pump and refrigeration objective functions must return "
            "HPRBackendResult."
        )
    return result


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
    return HeatPumpTargetOutputs.model_validate(result.to_output_fields())


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
    penalty_terms: np.ndarray | None = None,
    penalise_external_cold_when_refrigerating: bool = False,
) -> tuple[float, float, float, float]:
    """Standardise HPR utility, feasibility-penalty, and objective semantics."""
    positive_penalty_terms = (
        np.maximum(penalty_terms, 0.0) if penalty_terms is not None else np.array([])
    )
    penalty = (
        float(
            g_ineq_penalty(
                positive_penalty_terms,
                eta=args.eta_penalty,
                rho=args.rho_penalty,
                form="square",
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
            g_ineq_penalty(g=Q_ext_cold, rho=args.rho_penalty, form="square")
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
    "run_hpr_candidate_search",
    "solve_hpr_placement",
    "translate_hpr_result",
    "translate_hpr_output",
]
