"""Private targeting replay state and execution support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ....analysis.numerics import get_period_index
from ....contracts.output import TargetOutput
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
        return application_zone
    if application_zone is None:
        return selected_master_zone
    return selected_master_zone.get_subzone(application_zone)


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
    if include_subzones:
        problem._run_targeting_for_zone_and_subzones(
            zone=zone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            options=runtime_options,
            sid=sid,
        )
    else:
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
