"""Private StageWise topology-evolution state and execution."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from ....parallelism import supported_workers
from ...indexing import build_index_grid
from .._base import approach as _approach
from .._base.execution import _assign_value
from . import staged_utilities as _staged
from .postprocess import _solve_ranking_alpha
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
    signature: tuple[tuple[int, int, int], ...] = ()


_IMPROVEMENT_CALLBACK: ContextVar[Callable[[Any], None] | None] = ContextVar(
    "openpinch_hen_evolution_improvement", default=None
)


@contextmanager
def reporting_improvements(callback: Callable[[Any], None]) -> Iterator[None]:
    """Pass each candidate that improves on its search's best to ``callback``.

    Searches run in the enclosed block call it as they go, with the solved,
    verified candidate model, in the order the search accepts them.
    """
    token = _IMPROVEMENT_CALLBACK.set(callback)
    try:
        yield
    finally:
        _IMPROVEMENT_CALLBACK.reset(token)


def _report_improvement(candidate) -> None:
    callback = _IMPROVEMENT_CALLBACK.get()
    if callback is not None:
        callback(candidate)


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
    *,
    beam_width: int = 4,
    solved_topologies: dict | None = None,
):
    """Evolve topology using branched add/remove net-benefit heuristics.

    Each depth keeps at most ``beam_width`` branches, the best by TAC (ties
    broken on the topology signature), and a topology is solved at most once
    over the whole run, so the solve count is bounded by the beam width rather
    than growing towards (add + remove branches) ** depth.

    ``solved_topologies`` is a store shared by the searches of one run: a
    topology another search already solved, on the same model inputs, is
    reused rather than solved again.
    """

    if owner.mSuccess != 1:
        logger.warning("Initial model was not successful; skipping evolution.")
        return owner

    n_ad_branches = max(1, int(n_ad_branches))
    n_rm_branches = max(1, int(n_rm_branches))
    max_parallel = max(1, int(max_parallel))
    if no_improvement_patience is not None:
        no_improvement_patience = max(1, int(no_improvement_patience))
    solved = _shared_solved_topologies(owner, solved_topologies)
    if n_ad_branches == 1 and n_rm_branches == 1 and no_improvement_patience is None:
        return _get_source_net_benefit_evolution(
            owner,
            print_output=print_output,
            max_depth=max_depth,
            solved=solved,
        )
    beam_width = max(1, int(beam_width))
    seen_signatures: set[tuple[tuple[int, int, int], ...]] = set()
    initial_z = getattr(owner, "z_allowed", None)
    if initial_z is not None:
        seen_signatures.add(_topology_signature_from_z(owner, initial_z))
    best_model = _benchmark_model(owner)
    frontier = [
        _EvolutionBranchState(
            model=owner,
            best_tac=float(owner.TAC) if best_model is owner else float("inf"),
        )
    ]

    for unit in range(max_depth):
        logger.debug(
            "Evolution step %s/%s: active branches %s, best TAC %s",
            unit + 1,
            max_depth,
            len(frontier),
            getattr(best_model, "TAC", None),
        )
        specs = _evolution_candidate_specs(
            owner,
            frontier,
            unit=unit,
            n_ad_branches=n_ad_branches,
            n_rm_branches=n_rm_branches,
            seen_signatures=seen_signatures,
        )
        if not specs:
            logger.debug("No evolution candidate topologies found.")
            break

        solved_candidates = _solve_evolution_candidates(
            owner,
            specs,
            print_output=print_output,
            max_parallel=max_parallel,
            solved=solved,
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
                    signature=spec.signature,
                )
            else:
                branch_period = _EvolutionBranchState(
                    model=candidate,
                    best_tac=parent_period.best_tac,
                    stale_depths=parent_period.stale_depths + 1,
                    signature=spec.signature,
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
            if best_model is None or candidate.TAC < best_model.TAC:
                best_model = candidate
                _report_improvement(candidate)
        # Keep the best branches by TAC, ties broken on topology signature, so
        # the frontier never grows past the beam width.
        next_frontier.sort(
            key=lambda branch: (float(branch.model.TAC), branch.signature)
        )
        frontier = next_frontier[:beam_width]
        if not frontier:
            logger.debug("No viable evolution model found.")
            break

        if best_model is not None and best_model is not owner:
            logger.debug(
                "New best evolution model %s with TAC %.6f",
                best_model.name,
                best_model.TAC,
            )

    if best_model is not None and best_model is not owner:
        _update_with_best_model(owner, best_model)
    else:
        logger.debug("No evolution improvement found over original model.")
    owner.cleanup()
    return owner


def _get_source_net_benefit_evolution(
    owner,
    *,
    print_output: bool,
    max_depth: int,
    solved: dict | None = None,
):
    """Run the original OpenHENS single-path add/remove evolution search."""

    model = owner
    best_model = _benchmark_model(owner)
    # Topologies already solved, by signature: the search cycles through the
    # same moves, and a re-solve finds the same network.
    if solved is None:
        solved = {}
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
            solved=solved,
        )
        model_plus_one = owner.get_n_plus_one_evolution(
            print_output=print_output,
            unit=unit,
            prev_case=model,
            solved=solved,
        )
        model = _select_source_best_candidate(
            owner,
            model,
            model_minus_one,
            model_plus_one,
        )
        if model is None:
            logger.debug("No viable model found; ending evolution.")
            break
        if best_model is None or model.TAC < best_model.TAC:
            best_model = model
            _report_improvement(model)
            logger.debug(
                "New best model: %s found with TAC: %.6f",
                model.name,
                best_model.TAC,
            )

    if best_model is not None and best_model is not owner:
        _update_with_best_model(owner, best_model)
    else:
        logger.debug("No improvement found over original model.")
    owner.cleanup()
    return owner


def _shared_solved_topologies(owner, store: dict | None) -> dict | None:
    """This search's part of a run's store: the topologies solved on its inputs.

    A child model is built from these inputs and its topology alone, so a
    topology solved on equal inputs is the same network problem.
    """

    if store is None:
        return None
    arrays = json.dumps(
        owner.solver_arrays.to_json_dict(), sort_keys=True, default=str
    ).encode()
    key = (
        owner.framework,
        owner.stages,
        owner.dTmin,
        owner.min_dqda,
        owner.minimisation_goal,
        owner.non_isothermal_model,
        owner.solver,
        repr(owner.solver_options),
        hashlib.sha256(arrays).hexdigest(),
        # Children inherit the search root's staged heaters and coolers (a
        # warm start fixes their binaries), so searches share children only
        # when those agree.
        _staged.staged_units_on(owner),
        # Children inherit the level holds and the stream groups too.
        getattr(owner, "level_targets", ()),
        getattr(owner, "stream_groups", None),
    )
    return store.setdefault(key, {})


def _benchmark_model(owner):
    """Return the starting model when its TAC is a fair benchmark, else None.

    A solve that stopped before converging (for example at IPOPT's iteration
    limit) is still reported as successful, but its objective can be far below
    the cost of any real network. Evolution then keeps its best verified
    candidate, and the workflow's final ranking compares it with the seed
    network on verified costs.
    """

    solver_run = getattr(owner, "solver_run", None)
    if getattr(solver_run, "converged", True):
        return owner
    logger.warning(
        "Starting model %s did not converge; evolution candidates are not "
        "compared with its TAC.",
        getattr(owner, "name", None),
    )
    return None


def _select_source_best_candidate(
    owner,
    current_model,
    model_minus_one,
    model_plus_one,
):
    """Select the next OpenHENS tier-1 evolution candidate by success and TAC."""

    del current_model
    # The same check as the branched search: solved and verified.
    minus_success = _is_usable_evolution_candidate(model_minus_one)
    plus_success = _is_usable_evolution_candidate(model_plus_one)
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
    seen_signatures: set[tuple[tuple[int, int, int], ...]] | None = None,
) -> list[_EvolutionCandidateSpec]:
    """Return candidate topologies not yet seen in this run."""
    specs: list[_EvolutionCandidateSpec] = []
    if seen_signatures is None:
        seen_signatures = set()
    for branch_index, branch_period in enumerate(frontier):
        prev_case = branch_period.model
        for rank, position in enumerate(
            prev_case.get_lowest_benefit_HX_candidates(n_rm_branches),
            start=1,
        ):
            spec = _evolution_candidate_spec(
                owner,
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
            spec = _evolution_candidate_spec(
                owner,
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
    z_allowed = _z_allowed_with_candidate(
        owner,
        prev_case,
        position=candidate_position,
        value=z_value,
    )
    signature = _topology_signature_from_z(owner, z_allowed)
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
    solved: dict | None = None,
) -> list[tuple[_EvolutionCandidateSpec, Any]]:
    if supported_workers(max_parallel) <= 1 or len(specs) <= 1:
        return [
            (spec, candidate)
            for spec in specs
            if (
                candidate := _solve_evolution_candidate(
                    owner, spec, print_output, solved
                )
            )
            is not None
        ]

    candidates: list[tuple[_EvolutionCandidateSpec, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_parallel, len(specs))) as pool:
        futures = {
            pool.submit(
                _solve_evolution_candidate, owner, spec, print_output, solved
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            # _solve_evolution_candidate already converts failures to None.
            candidate = future.result()
            if candidate is not None:
                candidates.append((futures[future], candidate))
    # Completion order is arbitrary; process in spec order so parallel and
    # serial runs make the same choices.
    candidates.sort(key=lambda item: (item[0].branch_index, item[0].kind, item[0].rank))
    return candidates


def _solve_evolution_candidate(
    owner,
    spec: _EvolutionCandidateSpec,
    print_output: bool,
    solved: dict | None = None,
):
    try:
        if spec.kind == "minus":
            return _build_and_solve_evolution(
                owner,
                "minus",
                print_output=print_output,
                unit=spec.unit,
                prev_case=spec.prev_case,
                position=spec.position,
                z_allowed=spec.z_allowed,
                branch_label=_evolution_branch_label(owner, spec),
                solved=solved,
            )
        return _build_and_solve_evolution(
            owner,
            "plus",
            print_output=print_output,
            unit=spec.unit,
            prev_case=spec.prev_case,
            position=spec.position,
            z_allowed=spec.z_allowed,
            branch_label=_evolution_branch_label(owner, spec),
            solved=solved,
        )
    except Exception as exc:
        logger.debug(
            "EVM %s branch %s rank %s failed: %s",
            spec.kind,
            spec.branch_index,
            spec.rank,
            exc,
            exc_info=True,
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


def _open_added_match_slots(owner, z_allowed) -> None:
    """Make the slots of matches ``z_allowed`` adds to ``owner`` writable.

    A disallowed match's slots are plain numbers, which take no value; each
    becomes a parameter holding the same number, so the adopted network's
    values can be copied in.
    """

    def open_slot(row: list, k: int) -> None:
        if row[k].__class__ is float:
            row[k] = owner.m.Param(value=row[k])

    added = [
        (i, j, k)
        for i in range(owner.I)
        for j in range(owner.J)
        for k in range(owner.S)
        if z_allowed[i][j][k] > 0 and not owner.z_allowed[i][j][k] > 0
    ]
    if not added:
        return
    by_period = [
        getattr(owner, name)
        for name in ("Q_r_by_period", "theta_1_by_period", "theta_2_by_period")
    ]
    if owner.non_isothermal_model:
        by_period.append(owner.X_by_period)
    for i, j, k in added:
        open_slot(owner.z[i][j], k)
        for grid in by_period:
            for period in grid:
                open_slot(period[i][j], k)
        if owner.non_isothermal_model:
            for period in owner.Y_by_period:
                open_slot(period[j][i], k)


def _update_with_best_model(owner, best_model) -> None:
    """Adopt the selected evolved topology while retaining this model object."""

    verify = getattr(best_model, "verify", None)
    if callable(verify):
        is_valid, reasons = verify()
        if not is_valid:
            logger.debug(
                "Not adopting evolved topology %s: %s",
                getattr(best_model, "name", None),
                ", ".join(reasons),
            )
            return
    # Alpha is solved only for ranking, so the candidate's is solved here from
    # its own values; an empty alpha (failed solve) is solved again for the
    # adopted topology when it is next ranked.
    _solve_ranking_alpha(best_model)
    owner.alpha = (
        build_index_grid(
            lambda i, j, k: best_model.alpha[i][j][k],
            (owner.I, owner.J, owner.S),
        )
        if len(best_model.alpha)
        else []
    )
    _open_added_match_slots(owner, best_model.z_allowed)
    owner.z_allowed = build_index_grid(
        lambda i, j, k: best_model.z_allowed[i][j][k],
        (owner.I, owner.J, owner.S),
    )
    owner.set_initial_values_for_variables(best_model)

    owner.hu_cost_total = copy.deepcopy(best_model.hu_cost_total)
    owner.cu_cost_total = copy.deepcopy(best_model.cu_cost_total)
    # These are Pyomo expressions on the candidate's model; deep-copying one
    # copies that whole model. verify() only reads their solved values.
    if hasattr(best_model, "hu_cost_solver_by_period"):
        owner.hu_cost_solver_by_period = _solved_values(
            best_model.hu_cost_solver_by_period
        )
        owner.cu_cost_solver_by_period = _solved_values(
            best_model.cu_cost_solver_by_period
        )
    if hasattr(best_model, "recovery_area_cost_filtered"):
        owner.recovery_area_cost_filtered = _solved_values(
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
    if hasattr(best_model, "solver_run"):
        owner.solver_run = best_model.solver_run
    owner.get_post_process()


def _solved_values(values):
    """Return ``values`` with each entry, at any list depth, read as a float."""

    if isinstance(values, list):
        return [_solved_values(value) for value in values]
    return _value(values)


def get_n_minus_one_evolution(
    owner,
    print_output: bool,
    unit: int,
    prev_case,
    *,
    solved: dict | None = None,
):
    """Build and solve the source minus-one topology evolution candidate."""

    candidates = prev_case.get_lowest_benefit_HX_candidates(1)
    if not candidates:
        return None
    position = tuple(candidates[0])
    z_allowed_removed = _z_allowed_with_candidate(
        owner,
        prev_case,
        position=position,
        value=0,
    )
    return _build_and_solve_evolution(
        owner,
        "minus",
        print_output=print_output,
        unit=unit,
        prev_case=prev_case,
        position=position,
        z_allowed=z_allowed_removed,
        solved=solved,
    )


def _build_and_solve_evolution(
    owner,
    kind: Literal["minus", "plus"],
    *,
    print_output: bool,
    unit: int,
    prev_case,
    position: Sequence[int],
    z_allowed: list,
    branch_label: str | None = None,
    solved: dict | None = None,
):
    """Build and solve one minus-one or plus-one topology evolution candidate.

    The child model is solved with the root EVM model's configured solver.
    With ``solved``, a topology already in it is returned without a solve,
    and a new one is added to it.
    """

    if solved is not None:
        signature = _topology_signature_from_z(owner, z_allowed)
        if signature in solved:
            return solved[signature]

    i, j, k = (int(index) for index in position)
    logger.debug(
        "%s position i,j,k %s",
        "worst selected" if kind == "minus" else "best non-selected",
        [i, j, k],
    )
    logger.debug(
        "number in z_allowed_%s %s",
        "removed" if kind == "minus" else "added",
        _count_allowed_matches(z_allowed),
    )
    child = _stagewise_model_class()(
        name=(
            f"{owner.name}-n_{kind} 1 evolution model "
            f"{branch_label if branch_label is not None else unit}"
        ),
        framework=prev_case.framework,
        solver=owner.solver,
        solver_arrays=prev_case.solver_arrays,
        stages=prev_case.stages,
        dTmin=prev_case.dTmin,
        z_restriction=[z_allowed, None, None],
        min_dqda=prev_case.min_dqda,
        minimisation_goal=prev_case.minimisation_goal,
        non_isothermal_model=prev_case.non_isothermal_model,
        integers=False,
        tol=1e-3,
        solver_options=owner.solver_options,
        staged_utilities=getattr(prev_case, "staged_utilities", False),
        level_targets=getattr(prev_case, "level_targets", ()),
        stream_groups=getattr(prev_case, "stream_groups", None),
    )

    # Start from the parent's solution, then apply the move over it.
    child.set_initial_values_for_variables(prev_case)
    if kind == "minus":
        _assign_value(child.Q_r[i][j][k], 0.0)
        _assign_value(child.z[i][j][k], 0)
        approach = _approach._recovery_approach_temperature(owner, i, j)
        _assign_value(child.theta_1[i][j][k], approach)
        _assign_value(child.theta_2[i][j][k], approach)
    else:
        _assign_value(child.z[i][j][k], 1)

    child.optimise(print_output=print_output)
    if solved is not None:
        solved[signature] = child
    return child


def get_n_plus_one_evolution(
    owner,
    print_output: bool,
    unit: int,
    prev_case,
    *,
    solved: dict | None = None,
):
    """Build and solve the source plus-one topology evolution candidate."""

    candidates = prev_case.get_max_benefit_HX_candidates(1)
    if not candidates:
        return None
    position = tuple(candidates[0])
    z_allowed_added = _z_allowed_with_candidate(
        owner,
        prev_case,
        position=position,
        value=1,
    )
    return _build_and_solve_evolution(
        owner,
        "plus",
        print_output=print_output,
        unit=unit,
        prev_case=prev_case,
        position=position,
        z_allowed=z_allowed_added,
        solved=solved,
    )


def _z_allowed_with_candidate(
    owner,
    prev_case,
    *,
    position: Sequence[int],
    value: int,
) -> list:
    z_allowed = copy.deepcopy(prev_case.z)
    i, j, k = (int(index) for index in position)
    _set_recovery_binary_value(owner, z_allowed, (i, j, k), value)
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
    if hasattr(element, "VALUE") or hasattr(element, "is_variable_type"):
        _assign_value(element, int(value))
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
                    count += 1 if _value(element) == 1 else 0
    return count


def _is_usable_evolution_candidate(candidate) -> bool:
    if candidate is None:
        return False
    if not candidate.mSuccess:
        return False
    # A solve stopped at a limit keeps mSuccess, but its TAC is not reliable.
    if not getattr(getattr(candidate, "solver_run", None), "converged", True):
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
