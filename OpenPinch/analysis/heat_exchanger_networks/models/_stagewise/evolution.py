"""Private StageWise topology-evolution state and execution."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Literal

from ...indexing import build_index_grid
from .verification import _value

logger = logging.getLogger(__name__)


@dataclass
class _EvolutionCandidateSpec:
    kind: Literal["minus", "plus"]
    unit: int
    branch_index: int
    rank: int
    prev_case: Any
    position: tuple[int, int, int]
    z_allowed: list
    signature: tuple[tuple[int, int, int], ...]


@dataclass
class _EvolutionBranchState:
    model: Any
    best_tac: float
    stale_depths: int = 0


def _stagewise_model_class():
    from ..stagewise import StageWiseModel

    return StageWiseModel


def get_net_benefit_evolution(
    owner,
    print_output: bool,
    max_depth: int = 5,
    n_ad_branches: int = 1,
    n_rm_branches: int = 1,
    max_parallel: int = 1,
    no_improvement_patience: int | None = None,
):
    """Evolve topology using branched add/remove net-benefit heuristics."""

    if owner.mSuccess != 1:
        logger.warning("Initial model was not successful; skipping evolution.")
        return owner

    n_ad_branches = max(1, int(n_ad_branches))
    n_rm_branches = max(1, int(n_rm_branches))
    max_parallel = max(1, int(max_parallel))
    if no_improvement_patience is not None:
        no_improvement_patience = max(1, int(no_improvement_patience))
    if n_ad_branches == 1 and n_rm_branches == 1 and no_improvement_patience is None:
        return owner._get_source_net_benefit_evolution(
            print_output=print_output,
            max_depth=max_depth,
        )
    frontier = [_EvolutionBranchState(model=owner, best_tac=float(owner.TAC))]
    best_model = owner

    for unit in range(max_depth):
        logger.debug(
            "Evolution step %s/%s: active branches %s, best TAC %s",
            unit + 1,
            max_depth,
            len(frontier),
            getattr(best_model, "TAC", None),
        )
        specs = owner._evolution_candidate_specs(
            frontier,
            unit=unit,
            n_ad_branches=n_ad_branches,
            n_rm_branches=n_rm_branches,
        )
        if not specs:
            logger.debug("No evolution candidate topologies found.")
            break

        solved_candidates = owner._solve_evolution_candidates(
            specs,
            print_output=print_output,
            max_parallel=max_parallel,
        )
        next_frontier: list[_EvolutionBranchState] = []
        for spec, candidate in solved_candidates:
            if not _is_usable_evolution_candidate(candidate):
                continue
            parent_period = frontier[spec.branch_index]
            candidate_tac = float(candidate.TAC)
            if candidate_tac < parent_period.best_tac:
                branch_period = _EvolutionBranchState(
                    model=candidate,
                    best_tac=candidate_tac,
                    stale_depths=0,
                )
            else:
                branch_period = _EvolutionBranchState(
                    model=candidate,
                    best_tac=parent_period.best_tac,
                    stale_depths=parent_period.stale_depths + 1,
                )
            if (
                no_improvement_patience is not None
                and branch_period.stale_depths >= no_improvement_patience
            ):
                logger.debug(
                    "Pruning EVM branch %s after %s non-improving steps.",
                    spec.branch_index,
                    branch_period.stale_depths,
                )
                continue
            next_frontier.append(branch_period)
            if candidate.TAC < best_model.TAC:
                best_model = candidate
        frontier = next_frontier
        if not frontier:
            logger.debug("No viable evolution model found.")
            break

        if best_model is not owner:
            logger.debug(
                "New best evolution model %s with TAC %.6f",
                best_model.name,
                best_model.TAC,
            )

    if best_model.mSuccess and best_model.TAC < owner.TAC:
        owner._update_with_best_model(best_model)
    else:
        logger.debug("No evolution improvement found over original model.")
    owner.m.cleanup()
    return owner


def _get_source_net_benefit_evolution(
    owner,
    *,
    print_output: bool,
    max_depth: int,
):
    """Run the original OpenHENS single-path add/remove evolution search."""

    model = owner
    best_model = owner
    for unit in range(max_depth):
        logger.debug(
            "Evolution step %s/%s - Current TAC: %s",
            unit + 1,
            max_depth,
            model.TAC,
        )
        model_minus_one = owner.get_n_minus_one_evolution(
            print_output=print_output,
            unit=unit,
            prev_case=model,
        )
        model_plus_one = owner.get_n_plus_one_evolution(
            print_output=print_output,
            unit=unit,
            prev_case=model,
        )
        model = owner._select_source_best_candidate(
            model,
            model_minus_one,
            model_plus_one,
        )
        if model is None:
            logger.debug("No viable model found; ending evolution.")
            break
        if model.TAC < best_model.TAC:
            best_model = model
            logger.debug(
                "New best model: %s found with TAC: %.6f",
                model.name,
                best_model.TAC,
            )

    if best_model.mSuccess and best_model.TAC < owner.TAC:
        owner._update_with_best_model(best_model)
    else:
        logger.debug("No improvement found over original model.")
    owner.m.cleanup()
    return owner


def _select_source_best_candidate(
    owner,
    current_model,
    model_minus_one,
    model_plus_one,
):
    """Select the next OpenHENS tier-1 evolution candidate by success and TAC."""

    del current_model
    minus_success = bool(getattr(model_minus_one, "mSuccess", 0))
    plus_success = bool(getattr(model_plus_one, "mSuccess", 0))
    if not minus_success and not plus_success:
        return None
    if minus_success and not plus_success:
        return model_minus_one
    if not minus_success and plus_success:
        return model_plus_one
    logger.debug(
        "TAC comparison -1: %.6f, +1: %.6f",
        model_minus_one.TAC,
        model_plus_one.TAC,
    )
    return min(
        [model_minus_one, model_plus_one],
        key=lambda candidate: candidate.TAC,
    )


def _evolution_candidate_specs(
    owner,
    frontier: Sequence[_EvolutionBranchState],
    *,
    unit: int,
    n_ad_branches: int,
    n_rm_branches: int,
) -> list[_EvolutionCandidateSpec]:
    specs: list[_EvolutionCandidateSpec] = []
    seen_signatures: set[tuple[tuple[int, int, int], ...]] = set()
    for branch_index, branch_period in enumerate(frontier):
        prev_case = branch_period.model
        for rank, position in enumerate(
            prev_case.get_lowest_benefit_HX_candidates(n_rm_branches),
            start=1,
        ):
            spec = owner._evolution_candidate_spec(
                kind="minus",
                unit=unit,
                branch_index=branch_index,
                rank=rank,
                prev_case=prev_case,
                position=position,
                z_value=0,
                seen_signatures=seen_signatures,
            )
            if spec is not None:
                specs.append(spec)
        for rank, position in enumerate(
            prev_case.get_max_benefit_HX_candidates(n_ad_branches),
            start=1,
        ):
            spec = owner._evolution_candidate_spec(
                kind="plus",
                unit=unit,
                branch_index=branch_index,
                rank=rank,
                prev_case=prev_case,
                position=position,
                z_value=1,
                seen_signatures=seen_signatures,
            )
            if spec is not None:
                specs.append(spec)
    return specs


def _evolution_candidate_spec(
    owner,
    *,
    kind: Literal["minus", "plus"],
    unit: int,
    branch_index: int,
    rank: int,
    prev_case,
    position: Sequence[int],
    z_value: int,
    seen_signatures: set[tuple[tuple[int, int, int], ...]],
) -> _EvolutionCandidateSpec | None:
    if len(position) != 3:
        return None
    candidate_position = tuple(int(index) for index in position)
    z_allowed = owner._z_allowed_with_candidate(
        prev_case,
        position=candidate_position,
        value=z_value,
    )
    signature = owner._topology_signature_from_z(z_allowed)
    if signature in seen_signatures:
        logger.debug(
            "Skipping duplicate EVM topology at depth %s from branch %s",
            unit,
            branch_index,
        )
        return None
    seen_signatures.add(signature)
    return _EvolutionCandidateSpec(
        kind=kind,
        unit=unit,
        branch_index=branch_index,
        rank=rank,
        prev_case=prev_case,
        position=candidate_position,
        z_allowed=z_allowed,
        signature=signature,
    )


def _solve_evolution_candidates(
    owner,
    specs: Sequence[_EvolutionCandidateSpec],
    *,
    print_output: bool,
    max_parallel: int,
) -> list[tuple[_EvolutionCandidateSpec, Any]]:
    if max_parallel <= 1 or len(specs) <= 1:
        return [
            (spec, candidate)
            for spec in specs
            if (candidate := owner._solve_evolution_candidate(spec, print_output))
            is not None
        ]

    candidates: list[tuple[_EvolutionCandidateSpec, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(specs))) as pool:
        futures = {
            pool.submit(owner._solve_evolution_candidate, spec, print_output): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                candidate = future.result()
            except Exception as exc:
                logger.debug(
                    "Discarding failed EVM %s branch %s rank %s: %s",
                    spec.kind,
                    spec.branch_index,
                    spec.rank,
                    exc,
                )
                continue
            if candidate is not None:
                candidates.append((spec, candidate))
    return candidates


def _solve_evolution_candidate(
    owner,
    spec: _EvolutionCandidateSpec,
    print_output: bool,
):
    try:
        if spec.kind == "minus":
            return owner._build_and_solve_n_minus_one_evolution(
                print_output=print_output,
                unit=spec.unit,
                prev_case=spec.prev_case,
                position=spec.position,
                z_allowed_removed=spec.z_allowed,
                branch_label=owner._evolution_branch_label(spec),
            )
        return owner._build_and_solve_n_plus_one_evolution(
            print_output=print_output,
            unit=spec.unit,
            prev_case=spec.prev_case,
            position=spec.position,
            z_allowed_added=spec.z_allowed,
            branch_label=owner._evolution_branch_label(spec),
        )
    except Exception as exc:
        logger.debug(
            "EVM %s branch %s rank %s failed: %s",
            spec.kind,
            spec.branch_index,
            spec.rank,
            exc,
        )
        return None


def _evolution_branch_label(owner, spec: _EvolutionCandidateSpec) -> str:
    return f"{spec.unit}-b{spec.branch_index}-{spec.kind}{spec.rank}"


def _select_best_candidate(
    owner,
    current_model,
    model_minus_one,
    model_plus_one,
):
    """Select the source plus/minus evolution candidate for the next step."""

    del current_model
    minus_usable = _is_usable_evolution_candidate(model_minus_one)
    plus_usable = _is_usable_evolution_candidate(model_plus_one)
    if not minus_usable and not plus_usable:
        return None
    if minus_usable and not plus_usable:
        return model_minus_one
    if not minus_usable and plus_usable:
        return model_plus_one
    logger.debug(
        "TAC comparison -1: %.6f, +1: %.6f",
        model_minus_one.TAC,
        model_plus_one.TAC,
    )
    return min(
        [model_minus_one, model_plus_one],
        key=lambda candidate: candidate.TAC,
    )


def _update_with_best_model(owner, best_model) -> None:
    """Adopt the selected evolved topology while retaining this model object."""

    best_model.verify()
    owner.alpha = build_index_grid(
        lambda i, j, k: best_model.alpha[i][j][k],
        (owner.I, owner.J, owner.S),
    )
    owner.z_allowed = build_index_grid(
        lambda i, j, k: best_model.z_allowed[i][j][k],
        (owner.I, owner.J, owner.S),
    )
    owner.set_initial_values_for_variables(best_model, brackets=True)

    owner.hu_cost_total = copy.deepcopy(best_model.hu_cost_total)
    owner.cu_cost_total = copy.deepcopy(best_model.cu_cost_total)
    if hasattr(best_model, "recovery_area_cost_filtered"):
        owner.recovery_area_cost_filtered = copy.deepcopy(
            best_model.recovery_area_cost_filtered
        )
    if hasattr(best_model, "recovery_area_cost_total"):
        owner.recovery_area_cost_total = copy.deepcopy(
            best_model.recovery_area_cost_total
        )
    if hasattr(best_model, "capital_cost_total"):
        owner.capital_cost_total = copy.deepcopy(best_model.capital_cost_total)
    if hasattr(best_model, "weighted_operating_cost"):
        owner.weighted_operating_cost = copy.deepcopy(
            best_model.weighted_operating_cost
        )
    if hasattr(best_model, "area_r_shared"):
        owner.area_r_shared = copy.deepcopy(best_model.area_r_shared)
    if hasattr(best_model, "area_hu_shared"):
        owner.area_hu_shared = copy.deepcopy(best_model.area_hu_shared)
    if hasattr(best_model, "area_cu_shared"):
        owner.area_cu_shared = copy.deepcopy(best_model.area_cu_shared)
    owner.hu_area_cost_total = copy.deepcopy(best_model.hu_area_cost_total)
    owner.cu_area_cost_total = copy.deepcopy(best_model.cu_area_cost_total)
    owner.get_post_process()


def get_n_minus_one_evolution(owner, print_output: bool, unit: int, prev_case):
    """Build and solve the source minus-one topology evolution candidate."""

    candidates = prev_case.get_lowest_benefit_HX_candidates(1)
    if not candidates:
        return None
    position = tuple(candidates[0])
    z_allowed_removed = owner._z_allowed_with_candidate(
        prev_case,
        position=position,
        value=0,
    )
    return owner._build_and_solve_n_minus_one_evolution(
        print_output=print_output,
        unit=unit,
        prev_case=prev_case,
        position=position,
        z_allowed_removed=z_allowed_removed,
    )


def _build_and_solve_n_minus_one_evolution(
    owner,
    *,
    print_output: bool,
    unit: int,
    prev_case,
    position: Sequence[int],
    z_allowed_removed: list,
    branch_label: str | None = None,
):
    """Build and solve one minus-one topology evolution candidate."""

    i, j, k = (int(index) for index in position)
    logger.debug("worst selected position i,j,k %s", [i, j, k])
    logger.debug(
        "number in z_allowed_removed %s",
        _count_allowed_matches(z_allowed_removed),
    )
    model_minus_one = _stagewise_model_class()(
        name=(
            f"{owner.name}-n_minus 1 evolution model "
            f"{branch_label if branch_label is not None else unit}"
        ),
        framework=prev_case.framework,
        solver="ipopt-pyomo",
        solver_arrays=prev_case.solver_arrays,
        stages=prev_case.stages,
        dTmin=prev_case.dTmin,
        z_restriction=[z_allowed_removed, None, None],
        min_dqda=prev_case.min_dqda,
        minimisation_goal=prev_case.minimisation_goal,
        non_isothermal_model=prev_case.non_isothermal_model,
        integers=False,
        tol=1e-3,
        solver_options=owner.solver_options,
    )

    model_minus_one.Q_r[i][j][k].VALUE.value = 0.0
    model_minus_one.z[i][j][k].VALUE.value = 0
    approach = owner._recovery_approach_temperature(i, j)
    model_minus_one.theta_1[i][j][k].VALUE.value = approach
    model_minus_one.theta_2[i][j][k].VALUE.value = approach

    model_minus_one.optimise(print_output=print_output)
    return model_minus_one


def get_n_plus_one_evolution(owner, print_output: bool, unit: int, prev_case):
    """Build and solve the source plus-one topology evolution candidate."""

    candidates = prev_case.get_max_benefit_HX_candidates(1)
    if not candidates:
        return None
    position = tuple(candidates[0])
    z_allowed_added = owner._z_allowed_with_candidate(
        prev_case,
        position=position,
        value=1,
    )
    return owner._build_and_solve_n_plus_one_evolution(
        print_output=print_output,
        unit=unit,
        prev_case=prev_case,
        position=position,
        z_allowed_added=z_allowed_added,
    )


def _build_and_solve_n_plus_one_evolution(
    owner,
    *,
    print_output: bool,
    unit: int,
    prev_case,
    position: Sequence[int],
    z_allowed_added: list,
    branch_label: str | None = None,
):
    """Build and solve one plus-one topology evolution candidate."""

    i, j, k = (int(index) for index in position)
    logger.debug("best non-selected position i,j,k %s", [i, j, k])
    logger.debug(
        "number in z_allowed_added %s",
        _count_allowed_matches(z_allowed_added),
    )
    model_plus_one = _stagewise_model_class()(
        name=(
            f"{owner.name}-n_plus 1 evolution model "
            f"{branch_label if branch_label is not None else unit}"
        ),
        framework=prev_case.framework,
        solver="ipopt-pyomo",
        solver_arrays=prev_case.solver_arrays,
        stages=prev_case.stages,
        dTmin=prev_case.dTmin,
        z_restriction=[z_allowed_added, None, None],
        min_dqda=prev_case.min_dqda,
        minimisation_goal=prev_case.minimisation_goal,
        non_isothermal_model=prev_case.non_isothermal_model,
        integers=False,
        tol=1e-3,
        solver_options=owner.solver_options,
    )

    model_plus_one.z[i][j][k].VALUE.value = 1

    model_plus_one.optimise(print_output=print_output)
    return model_plus_one


def _z_allowed_with_candidate(
    owner,
    prev_case,
    *,
    position: Sequence[int],
    value: int,
) -> list:
    z_allowed = copy.deepcopy(prev_case.z)
    i, j, k = (int(index) for index in position)
    owner._set_recovery_binary_value(z_allowed, (i, j, k), value)
    return z_allowed


def _set_recovery_binary_value(
    owner,
    z_values: list,
    position: tuple[int, int, int],
    value: int,
) -> None:
    i, j, k = position
    element = z_values[i][j][k]
    if isinstance(element, (int, float)):
        z_values[i][j][k] = int(value)
        return
    try:
        element[0] = int(value)
        return
    except TypeError:
        pass
    except IndexError:
        pass
    if hasattr(element, "VALUE"):
        element.VALUE.value = int(value)
        return
    z_values[i][j][k] = int(value)


def _topology_signature_from_z(
    owner,
    z_values: Sequence[Sequence[Sequence[Any]]],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (i, j, k)
        for k in range(owner.S)
        for j in range(owner.J)
        for i in range(owner.I)
        if owner._active_binary_value(z_values[i][j][k]) > owner.tol
    )


def _active_binary_value(owner, value) -> float:
    return _value(value)


def _count_allowed_matches(values) -> int:
    count = 0
    for layer in values:
        for row in layer:
            for element in row:
                if isinstance(element, int):
                    count += 1 if element == 1 else 0
                else:
                    count += 1 if element[0] == 1 else 0
    return count


def _is_usable_evolution_candidate(candidate) -> bool:
    if candidate is None:
        return False
    if not candidate.mSuccess:
        return False
    verify = getattr(candidate, "verify", None)
    if not callable(verify):
        return True
    is_valid, reasons = verify()
    if not is_valid:
        logger.debug(
            "Discarding invalid evolution candidate %s: %s",
            getattr(candidate, "name", None),
            ", ".join(reasons),
        )
    return is_valid
