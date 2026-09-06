"""Owned execution snapshots and atomic targeting commits."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import Any

from ....domain.zone import Zone
from ..arguments import split_runtime_and_configuration_options
from .catalog import require_available
from .execution import walk_zone_tree
from .provenance import stamp_targets


@dataclass(frozen=True)
class PreparedPeriodState:
    """One successful period's runtime graph; never persisted as input JSON."""

    root: Zone
    components: dict[str, Any]


@dataclass(frozen=True)
class AnalysisExecutionContext:
    """One application-owned invocation and its isolated prepared input graph."""

    problem: Any
    zone_address: str
    period_id: str | None
    period_idx: int
    effective_configuration: Any
    prepared_inputs: Zone
    components: Any
    prerequisites: Any

    @classmethod
    def prepare(cls, problem, arguments):
        isolated = snapshot_problem(problem)
        root = isolated._build_execution_master_zone()
        zone = isolated._resolve_target_zone(arguments.get("zone"), master_zone=root)
        runtime, configuration = split_runtime_and_configuration_options(
            arguments.get("options")
        )
        configuration.update(arguments.get("configuration") or {})
        if arguments.get("period_id") is not None:
            runtime["period_id"] = arguments["period_id"]
        runtime, sid = isolated._resolve_runtime_period_options(runtime, zone=zone)
        return cls(
            isolated,
            zone.address,
            sid,
            runtime["period_idx"],
            MappingProxyType({**zone.config._values, **configuration}),
            root,
            MappingProxyType(isolated._process_components),
            MappingProxyType(dict(zone.targets)),
        )


def snapshot_problem(problem, state: PreparedPeriodState | None = None):
    """Copy the connected runtime graph once, redirecting all owner references."""
    isolated = copy(problem)
    root = state.root if state is not None else problem._master_zone
    components = state.components if state is not None else problem._process_components
    memo = {id(problem): isolated}
    for component in components.values():
        memo[id(component.problem)] = isolated
    isolated._master_zone, isolated._process_components = deepcopy(
        (root, components), memo
    )
    isolated._results = deepcopy(problem._results)
    isolated._period_results = {}
    isolated._period_states = {}
    isolated._utility_placement_result = None
    return isolated


def prepared_period_state(problem) -> PreparedPeriodState:
    return PreparedPeriodState(problem._master_zone, problem._process_components)


def commit_problem(problem, isolated) -> None:
    """Publish successful analysis while preserving explicit component handles."""
    components = {}
    for name, replacement in isolated._process_components.items():
        existing = problem._process_components.get(name)
        component = replacement if existing is None else existing
        if existing is not None:
            component.__dict__.update(replacement.__dict__)
        component.problem = problem
        components[name] = component
    problem._master_zone = isolated._master_zone
    problem._process_components = components
    problem._results = isolated._results
    problem._last_target_run_spec = isolated._last_target_run_spec


def invalidate_analysis(problem) -> None:
    """Invalidate every cached product of the prepared input model."""
    if problem._master_zone is not None:
        stack = [problem._master_zone]
        while stack:
            zone = stack.pop()
            zone.targets.clear()
            zone.graphs.clear()
            stack.extend(zone.subzones.values())
    problem._results = None
    problem._period_results = {}
    problem._period_states = {}
    problem._last_target_run_spec = None
    problem._utility_placement_result = None


def target_transaction(method):
    """Run one accessor implementation on scratch state and commit on success."""

    @wraps(method)
    def execute(accessor, *args, **kwargs):
        problem = accessor._problem
        surface = kwargs.get("surface") or (
            args[0] if method.__name__ == "_cogeneration" and args else method.__name__
        )
        require_available("target." + surface)
        context = AnalysisExecutionContext.prepare(problem, kwargs)
        isolated = context.problem
        previous = {
            id(target): target
            for zone in (
                []
                if isolated._master_zone is None
                else walk_zone_tree(isolated._master_zone)
            )
            for target in zone.targets.values()
        }
        result = method(type(accessor)(isolated), *args, **kwargs)
        if isolated._master_zone is not None:
            stamp_targets(isolated, surface, previous)
            if isolated._results is not None:
                targets = [
                    t
                    for z in walk_zone_tree(isolated._master_zone)
                    for t in z.targets.values()
                    if t.reportable
                ]
                for row in isolated._results.targets:
                    match = next(
                        (
                            t
                            for t in targets
                            if t.scope == row.scope
                            and t.integration_type == row.integration_type
                            and t.target_method == row.target_method
                            and t.period_idx == row.period_idx
                        ),
                        None,
                    )
                    if match is not None:
                        row.provenance = match.provenance
        if isolated._results is not None:
            include_subzones = kwargs.get(
                "include_subzones", method.__name__ == "all_heat_integration"
            )
            prefix = context.zone_address + "/"

            def in_scope(address):
                return address == context.zone_address or (
                    include_subzones and (address or "").startswith(prefix)
                )

            # Retain prepared prerequisites, but publish only this invocation's scope.
            isolated._results.targets = [
                row for row in isolated._results.targets if in_scope(row.scope)
            ]
            isolated._results.graphs = {
                name: graph
                for name, graph in (isolated._results.graphs or {}).items()
                if in_scope(graph.zone_address)
            }
        # Detach before committing: even failure to copy an output is atomic.
        detached = deepcopy(result)
        retained = prepared_period_state(snapshot_problem(isolated))
        commit_problem(problem, isolated)
        if context.period_id is not None:
            problem._period_states[context.period_id] = retained
        # A scalar mutation supersedes any previously published complete batch.
        problem._period_results = {}
        return detached

    return execute
