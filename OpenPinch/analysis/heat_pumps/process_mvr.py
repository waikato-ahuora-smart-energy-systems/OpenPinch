"""MVR-specific process component records and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...analysis.numerics import get_period_index
from ...domain.stream import Stream
from ...domain.zone import Zone
from ._process_mvr.membership import (
    add_replacement_streams_to_memberships,
    find_hot_stream_memberships,
)
from ._process_mvr.replacement_streams import build_process_mvr_stream_record
from ._process_mvr.selection import (
    match_source_streams,
    normalise_source_selectors,
)
from ._process_mvr.state import _ProcessMVRStreamRecord
from ._process_mvr.work import work_for_zone as calculate_work_for_zone
from .components import ProcessComponent
from .direct_mvr.execution import coerce_positive_mvr_stage_count
from .direct_mvr.models import (
    DirectGasMVRSettings,
    DirectGasMVRStageResult,
)

if TYPE_CHECKING:
    from ...application.problem import PinchProblem


@dataclass
class ProcessMVRComponent(ProcessComponent):
    """Memory-only direct process MVR component."""

    settings: DirectGasMVRSettings = field(default_factory=DirectGasMVRSettings)
    source_selectors: list[Any] = field(default_factory=list)
    stream_records: list[_ProcessMVRStreamRecord] = field(default_factory=list)
    component_type: str = "process_mvr"

    def activate(self):
        """Use the MVR replacement streams in subsequent targeting."""
        for record in self.stream_records:
            record.original_stream.active = False
            for stream in record.replacement_streams:
                stream.active = True
        return super().activate()

    def deactivate(self):
        """Restore the original source streams for subsequent targeting."""
        for record in self.stream_records:
            for stream in record.replacement_streams:
                stream.active = False
            record.original_stream.active = True
        return super().deactivate()

    @property
    def original_streams(self) -> list[Stream]:
        return [record.original_stream for record in self.stream_records]

    @property
    def replacement_streams(self) -> list[Stream]:
        return [
            stream
            for record in self.stream_records
            for stream in record.replacement_streams
        ]

    @property
    def stage_results_by_period(self) -> dict[str, list[DirectGasMVRStageResult]]:
        combined: dict[str, list[DirectGasMVRStageResult]] = {}
        for record in self.stream_records:
            for period_id, stages in record.stage_results_by_period.items():
                combined.setdefault(period_id, []).extend(stages)
        return combined

    @property
    def affected_zone_paths(self) -> list[str]:
        paths = {
            membership.zone.address
            for record in self.stream_records
            for membership in record.original_memberships
        }
        return sorted(paths)

    def work_for_zone(
        self,
        zone: Zone,
        *,
        period_id: str | None = None,
        period_idx: int | None = None,
    ) -> float:
        """Return active compressor work assigned to streams inside ``zone``."""
        return calculate_work_for_zone(
            self.stream_records,
            active=self.active,
            zone=zone,
            period_id=period_id,
            period_idx=period_idx,
        )


def create_process_mvr_component(
    problem: "PinchProblem",
    *,
    source_streams,
    mvr_id: str | None = None,
    n_stages: int = 1,
    liquid_injection: bool = True,
    mvr_stage_t_lift: float | None = None,
    mvr_stage_pressure_ratio: float | None = None,
    eta_mvr_comp: float | None = None,
    eta_motor: float | None = None,
    options: dict | None = None,
    period_id: str | None = None,
) -> ProcessMVRComponent:
    """Create, activate, and register a direct process MVR component."""
    root = problem._require_prepared_root_zone()
    runtime_options = dict(options or {})
    if period_id is not None:
        option_period_id = runtime_options.get("period_id")
        if option_period_id is not None and str(option_period_id) != str(period_id):
            raise ValueError(
                "Specify only one process MVR period context; period_id and "
                "options['period_id'] conflict."
            )
        runtime_options["period_id"] = period_id
    get_period_index(root.period_ids, runtime_options)
    if mvr_stage_t_lift is not None and mvr_stage_pressure_ratio is not None:
        raise ValueError(
            "Specify either mvr_stage_t_lift or mvr_stage_pressure_ratio, not both."
        )
    resolved_stage_t_lift = (
        None if mvr_stage_t_lift is None else float(mvr_stage_t_lift)
    )
    resolved_n_stages = coerce_positive_mvr_stage_count(
        n_stages,
        context="Process MVR",
    )
    settings = DirectGasMVRSettings(
        n_stages=resolved_n_stages,
        mvr_stage_t_lift=resolved_stage_t_lift,
        mvr_stage_pressure_ratio=mvr_stage_pressure_ratio,
        liquid_injection=bool(liquid_injection),
        eta_mvr_comp=float(
            root.config.process_mvr.eta_comp if eta_mvr_comp is None else eta_mvr_comp
        ),
        eta_motor=float(
            root.config.process_mvr.eta_motor if eta_motor is None else eta_motor
        ),
        dt_diff_max=float(getattr(root.config, "DT_DIFF_MAX", 0.1)),
    )
    component_id = _resolve_component_id(problem, mvr_id)
    selectors = normalise_source_selectors(source_streams)
    matched_streams = match_source_streams(root, selectors)
    records = [
        build_process_mvr_stream_record(
            source_stream=stream,
            memberships=find_hot_stream_memberships(root, stream),
            settings=settings,
            period_ids=root.period_ids,
            num_periods=root.num_periods,
        )
        for stream in matched_streams
    ]
    component = ProcessMVRComponent(
        id=component_id,
        problem=problem,
        settings=settings,
        source_selectors=selectors,
        stream_records=records,
    )
    for record in records:
        add_replacement_streams_to_memberships(record, component_id)
    problem.process_components[component_id] = component
    component.activate()
    return component


def _resolve_component_id(problem: "PinchProblem", requested_id: str | None) -> str:
    if requested_id:
        component_id = str(requested_id)
        if component_id in problem.process_components:
            raise ValueError(f"Process component id {component_id!r} already exists.")
        return component_id
    counter = 1
    while f"mvr_{counter}" in problem.process_components:
        counter += 1
    return f"mvr_{counter}"


__all__ = ["ProcessMVRComponent", "create_process_mvr_component"]
