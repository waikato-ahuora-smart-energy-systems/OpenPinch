"""Test-owned configuration and pure helpers for the HPR/MVR benchmark."""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

from OpenPinch.contracts.hpr import (
    HPREvaluationMode,
    HPRTargetingError,
)

HPRServiceName = Literal[
    "vapour_compression_heat_pump",
    "vapour_compression_refrigeration",
    "mvr_heat_pump",
]


@dataclass(frozen=True, slots=True)
class HPRProfile:
    """One immutable public-service configuration for a benchmark assignment."""

    profile_id: str
    service_name: HPRServiceName
    is_utility: bool
    is_cascade_cycle: bool | None
    objective_name: str
    condensers: int = 2
    evaporators: int = 1
    load_fraction: float = 0.25
    maximum_restarts: int = 1
    maximum_iterations: int = 3
    maximum_evaluations: int = 12
    maximum_search_observations: int = 24
    sentinel_maximum_evaluations: int | None = None
    sentinel_maximum_search_observations: int | None = None
    refrigerants: tuple[str, ...] = ("Water",)
    mvr_fluids: tuple[str, ...] = ("Water",)
    mvr_stages: int = 1

    def invocation_kwargs(
        self,
        *,
        maximum_evaluations: int | None = None,
    ) -> dict[str, Any]:
        """Build fresh explicit public kwargs without exposing mutable profile data."""
        kwargs: dict[str, Any] = {
            "load_fraction": self.load_fraction,
            "condensers": self.condensers,
            "evaporators": self.evaporators,
            "maximum_restarts": self.maximum_restarts,
            "maximum_iterations": self.maximum_iterations,
            "maximum_evaluations": (
                self.maximum_evaluations
                if maximum_evaluations is None
                else maximum_evaluations
            ),
        }
        utility_key = (
            "is_utility_refrigeration"
            if self.service_name == "vapour_compression_refrigeration"
            else "is_utility_heat_pump"
        )
        kwargs[utility_key] = self.is_utility
        if self.service_name == "mvr_heat_pump":
            kwargs.update(
                mvr_fluids=list(self.mvr_fluids),
                mvr_stages=self.mvr_stages,
                options={"HPR_REFRIGERANTS": list(self.refrigerants)},
            )
        else:
            kwargs.update(
                refrigerants=list(self.refrigerants),
                is_cascade_cycle=self.is_cascade_cycle,
            )
        return kwargs

    def search_limits(self, *, sentinel: bool) -> tuple[int, int]:
        """Return bounded search limits for an ordinary or sentinel assignment."""
        if not sentinel:
            return self.maximum_evaluations, self.maximum_search_observations
        return (
            self.sentinel_maximum_evaluations or self.maximum_evaluations,
            self.sentinel_maximum_search_observations
            or self.maximum_search_observations,
        )


HPR_PROFILES = (
    HPRProfile(
        profile_id="direct-cascade-vc-heat-pump",
        service_name="vapour_compression_heat_pump",
        is_utility=False,
        is_cascade_cycle=True,
        objective_name="_compute_cascade_hp_system_obj",
    ),
    HPRProfile(
        profile_id="utility-parallel-vc-heat-pump",
        service_name="vapour_compression_heat_pump",
        is_utility=True,
        is_cascade_cycle=False,
        objective_name="_compute_parallel_hp_system_obj",
    ),
    HPRProfile(
        profile_id="direct-cascade-vc-refrigeration",
        service_name="vapour_compression_refrigeration",
        is_utility=False,
        is_cascade_cycle=True,
        objective_name="_compute_cascade_hp_system_obj",
    ),
    HPRProfile(
        profile_id="utility-parallel-vc-refrigeration",
        service_name="vapour_compression_refrigeration",
        is_utility=True,
        is_cascade_cycle=False,
        objective_name="_compute_parallel_hp_system_obj",
    ),
    HPRProfile(
        profile_id="direct-optimized-vc-mvr-heat-pump",
        service_name="mvr_heat_pump",
        is_utility=False,
        is_cascade_cycle=None,
        objective_name="_compute_vc_mvr_system_obj",
        maximum_evaluations=20,
        maximum_search_observations=40,
        sentinel_maximum_evaluations=48,
        sentinel_maximum_search_observations=96,
    ),
    HPRProfile(
        profile_id="utility-optimized-vc-mvr-heat-pump",
        service_name="mvr_heat_pump",
        is_utility=True,
        is_cascade_cycle=None,
        objective_name="_compute_vc_mvr_system_obj",
        maximum_evaluations=16,
        maximum_search_observations=32,
        sentinel_maximum_evaluations=48,
        sentinel_maximum_search_observations=96,
    ),
)

HPR_SENTINEL_FILENAMES = {
    "direct-cascade-vc-heat-pump": "p_Adjiman et al.json",
    "utility-parallel-vc-heat-pump": "p_Ahmad (example 1).json",
    "direct-cascade-vc-refrigeration": "p_Ahmad (example 2).json",
    "utility-parallel-vc-refrigeration": "p_Ahmad (example 3).json",
    "direct-optimized-vc-mvr-heat-pump": "p_Pavao et al (example 1).json",
    "utility-optimized-vc-mvr-heat-pump": "p_Feng et al (case study 1).json",
}


@dataclass(frozen=True, slots=True)
class HPRBenchmarkAssignment:
    """A standard problem paired with exactly one HPR profile."""

    path: Path
    ordinal: int
    profile: HPRProfile

    @property
    def parameter_id(self) -> str:
        suffix = "::sentinel" if is_sentinel_assignment(self) else ""
        return f"{self.path.name}::{self.profile.profile_id}{suffix}"


def is_sentinel_assignment(assignment: HPRBenchmarkAssignment) -> bool:
    """Return whether an assignment is the declared sentinel for its profile."""
    return (
        HPR_SENTINEL_FILENAMES.get(assignment.profile.profile_id)
        == assignment.path.name
    )


def build_assignments(
    paths: Sequence[Path],
    profiles: Sequence[HPRProfile] = HPR_PROFILES,
) -> tuple[HPRBenchmarkAssignment, ...]:
    """Assign sorted unique paths to profiles by stable round-robin ordinal."""
    if not profiles:
        raise ValueError("At least one HPR benchmark profile is required.")
    ordered = tuple(sorted(paths, key=lambda path: path.name))
    if len(ordered) != len(set(ordered)):
        raise ValueError("HPR benchmark problem paths must be unique.")
    return tuple(
        HPRBenchmarkAssignment(
            path=path,
            ordinal=ordinal,
            profile=profiles[ordinal % len(profiles)],
        )
        for ordinal, path in enumerate(ordered)
    )


class HPROutcomeKind(str, Enum):
    """The complete public outcome vocabulary accepted by the benchmark."""

    SOLVED = "solved"
    NO_OP = "no_op"
    TYPED_FAILURE = "typed_failure"


@dataclass(frozen=True, slots=True)
class HPRBenchmarkOutcome:
    """One classified public HPR invocation result."""

    kind: HPROutcomeKind
    target: object | None = None
    error: HPRTargetingError | None = None


@dataclass(frozen=True, slots=True)
class ProblemStateSnapshot:
    """Stable public state before or after one benchmark invocation."""

    results_json: str | None
    target_count: int


def classify_hpr_outcome(
    *,
    target: object | None,
    error: BaseException | None = None,
) -> HPRBenchmarkOutcome:
    """Classify only the three approved public outcomes."""
    if target is not None and error is not None:
        raise ValueError("An HPR invocation cannot return both a target and an error.")
    if error is not None:
        if not isinstance(error, HPRTargetingError):
            raise TypeError("Only HPRTargetingError is an accepted failure outcome.")
        return HPRBenchmarkOutcome(
            kind=HPROutcomeKind.TYPED_FAILURE,
            error=error,
        )
    if target is None:
        return HPRBenchmarkOutcome(kind=HPROutcomeKind.NO_OP)
    return HPRBenchmarkOutcome(kind=HPROutcomeKind.SOLVED, target=target)


def snapshot_problem_state(problem: object) -> ProblemStateSnapshot:
    """Capture only the public result serialization and committed target count."""
    results = getattr(problem, "results")
    if results is None:
        return ProblemStateSnapshot(results_json=None, target_count=0)
    return ProblemStateSnapshot(
        results_json=results.model_dump_json(),
        target_count=len(results.targets),
    )


def assert_atomic_outcome(
    before: ProblemStateSnapshot,
    after: ProblemStateSnapshot,
    outcome: HPRBenchmarkOutcome,
) -> None:
    """Require unchanged failure/no-op state or an exact one-target commit."""
    if outcome.kind in {HPROutcomeKind.NO_OP, HPROutcomeKind.TYPED_FAILURE}:
        assert after == before, (
            "no-op and typed failure must leave public state unchanged"
        )
        return
    assert after.target_count == before.target_count + 1, (
        "a successful HPR invocation must commit exactly one target"
    )
    assert after.results_json is not None, "successful HPR results must serialize"


def assert_typed_failure_contract(
    error: HPRTargetingError,
    *,
    maximum_evaluations: int,
) -> None:
    """Validate bounded detached public failure evidence."""
    diagnostics = error.diagnostics
    assert diagnostics.budget.maximum_evaluations == maximum_evaluations, (
        "failure diagnostics must retain the configured evaluation allowance"
    )
    assert diagnostics.evaluated_count <= maximum_evaluations + 16, (
        "failure evaluation evidence exceeded the search allowance plus the "
        "bounded final-candidate reevaluation pool"
    )
    assert len(diagnostics.representative_failures) <= 16
    assert all(count >= 0 for count in diagnostics.category_counts.values())
    assert deepcopy(error).diagnostics == diagnostics
    assert pickle.loads(pickle.dumps(error)).diagnostics == diagnostics
    diagnostics.model_dump_json()


def assert_strict_success_contract(target: object, problem: object) -> None:
    """Validate the detached finite public contract for a solved HPR target."""
    assert getattr(target, "hpr_success") is True
    details = getattr(target, "hpr_details")
    assert details.success is True
    for field_name in (
        "obj",
        "utility_tot",
        "Q_ext",
        "Q_amb_hot",
        "Q_amb_cold",
        "w_net",
    ):
        values = np.asarray(getattr(details, field_name), dtype=float)
        assert values.size and np.isfinite(values).all(), (
            f"successful HPR accounting field {field_name!r} must be finite"
        )
    assert details.model is None, "public HPR details must not retain a live model"
    record = details.target_simulation_record
    assert record is not None, "successful CoolProp targets require simulation evidence"
    assert record.simulation_backend == "coolprop"
    assert math.isfinite(record.nominal_useful_duty)
    assert record.nominal_useful_duty > 0.0
    assert record.loops
    loop_duties = np.asarray([loop.nominal_duty for loop in record.loops], dtype=float)
    loop_work = np.asarray([loop.nominal_work for loop in record.loops], dtype=float)
    assert np.isfinite(loop_duties).all() and (loop_duties >= 0.0).all()
    assert np.isfinite(loop_work).all() and (loop_work >= 0.0).all()
    assert math.isclose(
        record.nominal_useful_duty,
        float(loop_duties.sum()),
        rel_tol=1e-8,
        abs_tol=1e-8,
    )
    copied = deepcopy(target)
    assert copied.hpr_details.model is None
    assert copied.hpr_details.target_simulation_record == record
    results = getattr(problem, "results")
    assert results is not None
    json.loads(results.model_dump_json())
    json.loads(record.model_dump_json())


def prepare_hpr_baseline(problem: object, profile: HPRProfile) -> None:
    """Materialize the public prerequisite target before the HPR transaction."""
    if profile.is_utility:
        problem.target.indirect_heat_integration()
    else:
        problem.target.direct_heat_integration()


@dataclass(frozen=True, slots=True)
class SearchObservation:
    """Detached facts from one real search-mode candidate evaluation."""

    point: tuple[float, ...]
    success: bool
    objective: float | None
    objective_name: str | None = None


def profile_search_observations(
    observations: Sequence[SearchObservation],
    profile: HPRProfile,
) -> tuple[SearchObservation, ...]:
    """Select the configured optimized-cycle objective, excluding seed searches."""
    return tuple(
        observation
        for observation in observations
        if observation.objective_name == profile.objective_name
    )


@dataclass(frozen=True, slots=True)
class ConvergenceWitness:
    """Pure evidence of bounded progress toward the best observed candidate."""

    distinct_evaluations: int
    maximum_evaluations: int
    viable_objectives: tuple[float, ...]
    incumbent_objectives: tuple[float, ...]
    first_viable_objective: float
    best_observed_objective: float
    selected_objective: float
    tolerance: float
    improvement: float
    selected_gap: float


def build_convergence_witness(
    observations: Sequence[SearchObservation],
    *,
    selected_objective: float,
    maximum_evaluations: int,
) -> ConvergenceWitness:
    """Build a deterministic witness from first-seen exact search points."""
    if isinstance(maximum_evaluations, bool) or maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be a positive integer.")
    selected = float(selected_objective)
    if not math.isfinite(selected):
        raise ValueError("selected_objective must be finite.")

    seen: set[tuple[float, ...]] = set()
    distinct: list[SearchObservation] = []
    for observation in observations:
        point = tuple(float(value) for value in observation.point)
        if not point or not all(math.isfinite(value) for value in point):
            raise ValueError("Search observation points must be non-empty and finite.")
        if point not in seen:
            seen.add(point)
            distinct.append(observation)

    viable = tuple(
        float(observation.objective)
        for observation in distinct
        if observation.success
        and observation.objective is not None
        and math.isfinite(float(observation.objective))
    )
    if not viable:
        raise AssertionError("Convergence requires at least one viable search point.")

    incumbent: list[float] = []
    best = float("inf")
    for objective in viable:
        best = min(best, objective)
        incumbent.append(best)
    first = viable[0]
    tolerance = 1e-8 * max(1.0, abs(first))
    return ConvergenceWitness(
        distinct_evaluations=len(distinct),
        maximum_evaluations=maximum_evaluations,
        viable_objectives=viable,
        incumbent_objectives=tuple(incumbent),
        first_viable_objective=first,
        best_observed_objective=best,
        selected_objective=selected,
        tolerance=tolerance,
        improvement=first - best,
        selected_gap=abs(selected - best),
    )


def assert_bounded_convergence(witness: ConvergenceWitness) -> None:
    """Require material real-search progress and best-observed selection."""
    assert witness.distinct_evaluations <= witness.maximum_evaluations, (
        "distinct search evaluations exceeded the configured allowance: "
        f"{witness.distinct_evaluations} > {witness.maximum_evaluations}"
    )
    assert len(witness.viable_objectives) >= 2, (
        "bounded convergence requires at least two distinct viable search points"
    )
    assert witness.improvement > witness.tolerance, (
        "search did not materially improve the first viable objective: "
        f"improvement={witness.improvement}, tolerance={witness.tolerance}"
    )
    assert all(
        later <= earlier
        for earlier, later in zip(
            witness.incumbent_objectives,
            witness.incumbent_objectives[1:],
            strict=False,
        )
    ), "incumbent best objective must be non-increasing"
    assert witness.selected_gap <= witness.tolerance, (
        "selected final objective does not equal the best observed viable objective: "
        f"gap={witness.selected_gap}, tolerance={witness.tolerance}"
    )


@contextmanager
def observe_hpr_search() -> Iterator[list[SearchObservation]]:
    """Observe real search evaluations without changing arguments or results."""
    import OpenPinch.analysis.heat_pumps.optimisation_adapter as adapter

    original = adapter.evaluate_hpr_candidate
    observations: list[SearchObservation] = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("artifact_mode") is HPREvaluationMode.SEARCH:
            point = tuple(
                float(value)
                for value in np.asarray(kwargs["point"], dtype=float).reshape(-1)
            )
            objective = float(result.obj)
            observations.append(
                SearchObservation(
                    point=point,
                    success=bool(result.success and math.isfinite(objective)),
                    objective=(
                        objective
                        if result.success and math.isfinite(objective)
                        else None
                    ),
                    objective_name=getattr(kwargs.get("objective"), "__name__", None),
                )
            )
        return result

    adapter.evaluate_hpr_candidate = observed
    try:
        yield observations
    finally:
        adapter.evaluate_hpr_candidate = original


__all__ = [
    "ConvergenceWitness",
    "HPRBenchmarkAssignment",
    "HPRBenchmarkOutcome",
    "HPROutcomeKind",
    "HPRProfile",
    "HPR_PROFILES",
    "HPR_SENTINEL_FILENAMES",
    "ProblemStateSnapshot",
    "SearchObservation",
    "assert_atomic_outcome",
    "assert_bounded_convergence",
    "assert_strict_success_contract",
    "assert_typed_failure_contract",
    "build_assignments",
    "build_convergence_witness",
    "classify_hpr_outcome",
    "is_sentinel_assignment",
    "observe_hpr_search",
    "prepare_hpr_baseline",
    "profile_search_observations",
    "snapshot_problem_state",
]
