"""Private targeting replay state and execution support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from ....lib.schemas.io import TargetOutput
from ....lib.schemas.targets import BaseTargetModel
from ...zone import Zone
from ..output.result_extraction import extract_results
from .dispatch import run_targeting_for_zone_and_subzones

if TYPE_CHECKING:
    from ...pinch_problem import PinchProblem

ZoneService = Callable[[Zone, Optional[dict[str, Any]]], Zone]


@dataclass(frozen=True)
class _TargetRunSpec:
    """A public targeting surface that can be replayed for another period."""

    surface: str
    options: dict[str, Any]
    zone_name: Optional[str] = None
    include_subzones: bool = False


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
