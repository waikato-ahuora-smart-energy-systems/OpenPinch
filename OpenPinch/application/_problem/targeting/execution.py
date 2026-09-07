"""Private targeting replay state and execution support."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Optional

from ....analysis.numerics import get_period_index
from ....contracts.output import TargetOutput
from ....domain.enums import TargetType
from ....domain.targets import BaseTargetModel
from ....domain.zone import Zone
from ..output.result_extraction import extract_results
from .dispatch import run_targeting_for_zone_and_subzones

if TYPE_CHECKING:
    from ...problem import PinchProblem

ZoneService = Callable[[Zone, Optional[dict[str, Any]]], Zone]


def resolve_target_zone(
    problem: "PinchProblem",
    application_zone: str | Zone | None = None,
    *,
    master_zone: Zone | None = None,
) -> Zone:
    """Resolve a target application zone against one execution root."""
    selected_master_zone = master_zone or problem._master_zone
    if selected_master_zone is None:
        raise RuntimeError("Load problem source data first before targeting.")
    if isinstance(application_zone, Zone):
        application_zone = application_zone.address
    if application_zone is None:
        return selected_master_zone
    if not isinstance(application_zone, str) or not application_zone.strip():
        raise ValueError("Target zone must be a non-empty local address.")
    selector = application_zone.strip()
    matches = [
        zone
        for zone in walk_zone_tree(selected_master_zone)
        if selector in (zone.address, zone.name)
        or zone.address == f"{selected_master_zone.name}/{selector}"
    ]
    if len(matches) != 1:
        raise ValueError(f"Target zone {selector!r} was not found or is ambiguous.")
    return matches[0]


@contextmanager
def scratch_child_targets(zone: Zone):
    """Keep child prerequisite products private to one selected-zone operation."""
    children = list(walk_zone_tree(zone))[1:]
    previous = [(child, child._targets, child._graphs) for child in children]
    root = zone
    while isinstance(root.parent_zone, Zone):
        root = root.parent_zone
    memo = {id(current): current for current in walk_zone_tree(root)}
    for child in children:
        child._targets, child._graphs = deepcopy((child._targets, child._graphs), memo)
    try:
        yield
    finally:
        for child, targets, graphs in previous:
            child._targets = targets
            child._graphs = graphs


def walk_zone_tree(zone: Zone):
    """Yield one zone hierarchy in stable pre-order."""
    yield zone
    for subzone in zone.subzones.values():
        yield from walk_zone_tree(subzone)


def process_component_work_for_zone(
    problem: "PinchProblem",
    zone: Zone,
    *,
    period_id: str | None,
    period_idx: int | None,
) -> float:
    """Sum all attached component work applicable to one zone."""
    total = 0.0
    for component in problem._process_components.values():
        work_for_zone = getattr(component, "work_for_zone", None)
        if work_for_zone is not None:
            total += float(
                work_for_zone(zone, period_id=period_id, period_idx=period_idx)
            )
    return total


def attach_process_component_work_targets(
    problem: "PinchProblem",
    zone: Zone,
    runtime_options: dict[str, Any] | None,
) -> None:
    """Attach process-component work to every generated target transactionally."""
    if not problem._process_components:
        return
    period_id = (runtime_options or {}).get("period_id")
    period_idx = (runtime_options or {}).get("period_idx")
    for current_zone in walk_zone_tree(zone):
        component_work = process_component_work_for_zone(
            problem,
            current_zone,
            period_id=period_id,
            period_idx=period_idx,
        )
        for target in current_zone.targets.values():
            if (
                period_idx is not None
                and getattr(target, "period_idx", None) != period_idx
            ):
                continue
            if hasattr(target, "process_component_work_target"):
                target.process_component_work_target = component_work
            if (
                component_work > 0.0
                and hasattr(target, "work_target")
                and getattr(target, "work_target", None) is None
            ):
                target.work_target = component_work


def build_execution_master_zone(problem: "PinchProblem") -> Zone:
    """Return the prepared root zone, loading existing inputs when necessary."""
    if problem._problem_data is None and problem._master_zone is None:
        raise RuntimeError("No input loaded. Call load(...) first.")
    if problem._master_zone is None:
        problem.load(problem._problem_data)
    return problem._master_zone


def resolve_runtime_period_options(
    options: dict[str, Any] | None,
    *,
    zone: Zone,
) -> tuple[dict[str, Any], str | None]:
    """Normalize runtime period selectors to canonical id/index values."""
    runtime_options = dict(options or {})
    idx, sid = get_period_index(period_ids=zone.period_ids, args=runtime_options)
    if sid is None:
        sid = next(
            (name for name, index in zone.period_ids.items() if index == idx), None
        )
    runtime_options["period_idx"] = idx
    if sid is not None:
        runtime_options["period_id"] = sid
    return runtime_options, sid


def run_problem_targeting(
    problem: "PinchProblem",
    *,
    zone: Optional[Zone] = None,
    direct_service_func: Optional[ZoneService] = None,
    indirect_service_func: Optional[ZoneService] = None,
    options: Optional[dict[str, Any]] = None,
    sid: str | None = None,
    dispatch_func=run_targeting_for_zone_and_subzones,
    extract_func=extract_results,
) -> TargetOutput:
    """Run targeting against a prepared parent problem and cache the output."""
    if not isinstance(zone, Zone):
        zone = problem._build_execution_master_zone()
    runtime_options, sid = problem._resolve_runtime_period_options(options, zone=zone)
    dispatch_func(
        zone=zone,
        direct_service_func=direct_service_func,
        indirect_service_func=indirect_service_func,
        args=runtime_options,
    )
    problem._attach_process_component_work_targets(zone, runtime_options)
    problem._results = TargetOutput.model_validate(extract_func(zone, period_id=sid))
    return problem._results


def execute_targeting(
    problem: "PinchProblem",
    *,
    target_id: str,
    application_zone: Optional[str | Zone],
    options: Optional[dict[str, Any]],
    include_subzones: bool,
    direct_service_func: Optional[ZoneService] = None,
    indirect_service_func: Optional[ZoneService] = None,
    sid: str | None = None,
    extract_func=extract_results,
) -> BaseTargetModel:
    """Execute one selected target family for a parent problem."""
    master = problem._build_execution_master_zone()
    runtime_options, sid = problem._resolve_runtime_period_options(options, zone=master)
    zone = problem._resolve_target_zone(application_zone, master_zone=master)
    if target_id != "Energy Transfer Analysis":
        parent = zone.parent_zone
        while isinstance(parent, Zone):
            parent.targets.clear()
            parent.graphs.clear()
            parent = parent.parent_zone
    if include_subzones:
        problem._run_targeting_for_zone_and_subzones(
            zone=zone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            options=runtime_options,
            sid=sid,
        )
    else:
        with scratch_child_targets(zone):
            if direct_service_func is not None:
                direct_service_func(zone, runtime_options)
            if indirect_service_func is not None:
                indirect_service_func(zone, runtime_options)
        problem._attach_process_component_work_targets(master, runtime_options)
        problem._results = TargetOutput.model_validate(
            extract_func(master, period_id=sid)
        )
    try:
        return zone.targets[target_id]
    except KeyError as exc:
        if target_id in {
            TargetType.DHP.value,
            TargetType.IHP.value,
            TargetType.DR.value,
            TargetType.IR.value,
        }:
            # A zero selected/available service produces no HPR target.
            return None
        raise RuntimeError(
            f"Targeting did not produce target {target_id!r} for zone {zone.name!r}."
        ) from exc


def execute_cogeneration_targeting(
    problem: "PinchProblem",
    *,
    application_zone: Optional[str | Zone],
    options: Optional[dict[str, Any]],
    include_subzones: bool,
    service_func: Optional[ZoneService] = None,
    sid: str | None = None,
    extract_func=extract_results,
) -> BaseTargetModel:
    """Run cogeneration and return the runtime-selected target family."""
    master = problem._build_execution_master_zone()
    runtime_options, sid = problem._resolve_runtime_period_options(options, zone=master)
    zone = problem._resolve_target_zone(application_zone, master_zone=master)
    if include_subzones:
        problem._run_targeting_for_zone_and_subzones(
            zone=zone,
            direct_service_func=service_func,
            options=runtime_options,
            sid=sid,
        )
    else:
        with scratch_child_targets(zone):
            if service_func is not None:
                service_func(zone, runtime_options)
        problem._attach_process_component_work_targets(master, runtime_options)
        problem._results = TargetOutput.model_validate(
            extract_func(master, period_id=sid)
        )
    selected = getattr(zone, "_selected_cogeneration_target_type", None)
    if not isinstance(selected, str):
        raise RuntimeError(
            f"Cogeneration did not select a compatible target for zone {zone.name!r}."
        )
    try:
        return zone.targets[selected]
    except KeyError as exc:
        raise RuntimeError(
            f"Cogeneration selected target {selected!r} for zone {zone.name!r}, "
            "but that target was not available on the zone."
        ) from exc


def run_exergy_targeting_for_zone_and_subzones(
    *,
    zone: Zone,
    service_func: Optional[ZoneService],
    options: Optional[dict[str, Any]],
) -> None:
    """Run exergy targeting in post-order across one zone tree."""
    child_options = dict(options or {})
    child_options.pop("base_target_type", None)
    for subzone in zone.subzones.values():
        run_exergy_targeting_for_zone_and_subzones(
            zone=subzone,
            service_func=service_func,
            options=child_options,
        )
    if service_func is not None:
        service_func(zone, options)


def execute_exergy_targeting(
    problem: "PinchProblem",
    *,
    application_zone: Optional[str | Zone],
    options: Optional[dict[str, Any]],
    include_subzones: bool,
    service_func: Optional[ZoneService] = None,
    sid: str | None = None,
    extract_func=extract_results,
) -> BaseTargetModel:
    """Apply exergy targeting and return the runtime-selected target family."""
    master = problem._build_execution_master_zone()
    runtime_options, sid = problem._resolve_runtime_period_options(options, zone=master)
    zone = problem._resolve_target_zone(application_zone, master_zone=master)
    if include_subzones:
        problem._run_exergy_targeting_for_zone_and_subzones(
            zone=zone,
            service_func=service_func,
            options=runtime_options,
        )
    elif service_func is not None:
        service_func(zone, runtime_options)
    problem._attach_process_component_work_targets(master, runtime_options)
    problem._results = TargetOutput.model_validate(extract_func(master, period_id=sid))
    selected = getattr(zone, "_selected_exergy_target_type", None)
    if not isinstance(selected, str):
        raise RuntimeError(
            "Exergy targeting did not select a compatible target "
            f"for zone {zone.name!r}."
        )
    try:
        return zone.targets[selected]
    except KeyError as exc:
        raise RuntimeError(
            f"Exergy targeting selected target {selected!r} for zone {zone.name!r}, "
            "but that target was not available on the zone."
        ) from exc
