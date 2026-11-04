"""MVR-specific process component records and factory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from ...analysis.numerics import get_period_index
from ...contracts.input import ProcessComponentSchema, ProcessComponentSourceSchema
from ...domain.stream import Stream
from ...domain.zone import Zone
from ._process_mvr.membership import (
    add_replacement_streams_to_memberships,
    find_hot_stream_memberships,
    walk_zones,
)
from ._process_mvr.replacement_streams import build_process_mvr_stream_record
from ._process_mvr.selection import (
    match_source_streams,
    normalise_source_selectors,
    validate_process_mvr_source,
)
from ._process_mvr.state import _ProcessMVRStreamRecord
from ._process_mvr.work import work_for_zone as calculate_work_for_zone
from .components import ProcessComponent, ProcessComponentHost
from .direct_mvr.execution import coerce_positive_mvr_stage_count
from .direct_mvr.models import (
    DirectGasMVRSettings,
    DirectGasMVRStageResult,
)


@dataclass
class ProcessMVRComponent(ProcessComponent):
    """Direct process MVR component, saved with its case as the call that made it."""

    settings: DirectGasMVRSettings = field(default_factory=DirectGasMVRSettings)
    source_selectors: list[Any] = field(default_factory=list)
    stream_records: list[_ProcessMVRStreamRecord] = field(default_factory=list)
    component_type: str = "process_mvr"
    _requested_eta_mvr_comp: float | None = None
    _requested_eta_motor: float | None = None

    def activate(self):
        """Use the MVR replacement streams in subsequent targeting."""
        for record in self.stream_records:
            record.original_stream.is_active = False
            for stream in record.replacement_streams:
                stream.is_active = True
        return super().activate()

    def deactivate(self):
        """Restore the original source streams for subsequent targeting."""
        for record in self.stream_records:
            for stream in record.replacement_streams:
                stream.is_active = False
            record.original_stream.is_active = True
        return super().deactivate()

    @property
    def original_streams(self) -> list[Stream]:
        return deepcopy([record.original_stream for record in self.stream_records])

    @property
    def replacement_streams(self) -> list[Stream]:
        return deepcopy(
            [
                stream
                for record in self.stream_records
                for stream in record.replacement_streams
            ]
        )

    @property
    def stage_results_by_period(self) -> dict[str, list[DirectGasMVRStageResult]]:
        combined: dict[str, list[DirectGasMVRStageResult]] = {}
        for record in self.stream_records:
            for period_id, stages in record.stage_results_by_period.items():
                combined.setdefault(period_id, []).extend(stages)
        return deepcopy(combined)

    @property
    def affected_zone_paths(self) -> list[str]:
        paths = {
            membership.zone.address
            for record in self.stream_records
            for membership in record.original_memberships
        }
        return sorted(paths)

    def to_record(self) -> ProcessComponentSchema:
        """Return the call that rebuilds this component from the case's input."""
        return ProcessComponentSchema(
            type=self.component_type,
            id=self.id,
            source_streams=[
                ProcessComponentSourceSchema(
                    # The last membership is the zone that owns the stream, where
                    # its name is unique; parents can repeat it across children.
                    zone=_path_below_root(record.original_memberships[-1].zone),
                    name=record.original_stream.name,
                )
                for record in self.stream_records
            ],
            n_stages=self.settings.n_stages,
            liquid_injection=self.settings.liquid_injection,
            mvr_stage_t_lift=self.settings.mvr_stage_t_lift,
            mvr_stage_pressure_ratio=self.settings.mvr_stage_pressure_ratio,
            compressor_efficiency=self._requested_eta_mvr_comp,
            motor_efficiency=self._requested_eta_motor,
            active=self.active,
        )

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
    problem: ProcessComponentHost,
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
        **_settings_from_configuration(root, eta_mvr_comp, eta_motor),
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
        _requested_eta_mvr_comp=eta_mvr_comp,
        _requested_eta_motor=eta_motor,
    )
    for record in records:
        add_replacement_streams_to_memberships(record, component_id)
    problem._process_components[component_id] = component
    component.activate()
    return component


def replay_process_components(
    components: Mapping[str, ProcessMVRComponent],
    root: Zone,
) -> None:
    """Re-solve components on a rebuilt tree, changing none of them on failure."""
    zones = {_path_below_root(zone): zone for zone in walk_zones(root)}
    replayed = [
        _replay_component(component, root, zones) for component in components.values()
    ]
    for component, (settings, records) in zip(components.values(), replayed):
        component.settings = settings
        component.stream_records = records


def restore_process_components(
    problem: ProcessComponentHost,
    records: Sequence[ProcessComponentSchema],
) -> None:
    """Rebuild saved components, in call order, on a freshly prepared problem."""
    zones = {
        _path_below_root(zone): zone
        for zone in walk_zones(problem._require_prepared_root_zone())
    }
    restored: list[ProcessMVRComponent] = []
    for record in records:
        sources = [
            _find_saved_source(record.id, source, zones)
            for source in record.source_streams
        ]
        # An earlier component on this source was inactive when this one was added;
        # every component's saved state is applied below.
        for stream in sources:
            stream.is_active = True
        restored.append(
            create_process_mvr_component(
                problem,
                source_streams=sources,
                mvr_id=record.id,
                n_stages=record.n_stages,
                liquid_injection=record.liquid_injection,
                mvr_stage_t_lift=record.mvr_stage_t_lift,
                mvr_stage_pressure_ratio=record.mvr_stage_pressure_ratio,
                eta_mvr_comp=record.compressor_efficiency,
                eta_motor=record.motor_efficiency,
            )
        )
    pairs = list(zip(restored, records, strict=True))
    for component, record in pairs:
        if not record.active:
            component.deactivate()
    # In call order, so a component fed by an earlier one turns that one's output off.
    for component, record in pairs:
        if record.active:
            component.activate()


def _find_saved_source(
    component_id: str,
    source: ProcessComponentSourceSchema,
    zones: dict[str, Zone],
) -> Stream:
    zone = zones.get(source.zone)
    streams = () if zone is None else zone.hot_streams.items()
    for _, stream in streams:
        if stream.name == source.name:
            return stream
    raise ValueError(
        f"Process component {component_id!r} cannot be rebuilt: source stream "
        f"{source.name!r} was not found in zone {source.zone!r}."
    )


def _replay_component(
    component: ProcessMVRComponent,
    root: Zone,
    zones: dict[str, Zone],
) -> tuple[DirectGasMVRSettings, list[_ProcessMVRStreamRecord]]:
    settings = replace(
        component.settings,
        **_settings_from_configuration(
            root,
            component._requested_eta_mvr_comp,
            component._requested_eta_motor,
        ),
    )
    records = []
    for previous in component.stream_records:
        source = _find_replay_source(component.id, previous, zones)
        # Earlier replays may have switched the source off; its flag is restored below.
        source.is_active = True
        validate_process_mvr_source(source, source.name)
        record = build_process_mvr_stream_record(
            source_stream=source,
            memberships=find_hot_stream_memberships(root, source),
            settings=settings,
            period_ids=root.period_ids,
            num_periods=root.num_periods,
        )
        add_replacement_streams_to_memberships(record, component.id)
        source.is_active = previous.original_stream.is_active
        for stream, previous_stream in zip(
            record.replacement_streams, previous.replacement_streams, strict=True
        ):
            stream.is_active = previous_stream.is_active
        records.append(record)
    return settings, records


def _find_replay_source(
    component_id: str,
    previous: _ProcessMVRStreamRecord,
    zones: dict[str, Zone],
) -> Stream:
    membership = previous.original_memberships[0]
    zone = zones.get(_path_below_root(membership.zone))
    streams = {} if zone is None else dict(zone.hot_streams.items())
    if membership.key not in streams:
        raise ValueError(
            f"Process component {component_id!r} cannot be rebuilt: source stream "
            f"{previous.original_stream.name!r} was not found."
        )
    return streams[membership.key]


def _path_below_root(zone: Zone) -> str:
    return zone.address.partition("/")[2]


def _settings_from_configuration(
    root: Zone,
    eta_mvr_comp: float | None,
    eta_motor: float | None,
) -> dict[str, float]:
    """Resolve settings that take the configuration's value unless given."""
    return {
        "eta_mvr_comp": float(
            root.config.process_mvr.eta_comp if eta_mvr_comp is None else eta_mvr_comp
        ),
        "eta_motor": float(
            root.config.process_mvr.eta_motor if eta_motor is None else eta_motor
        ),
        "dt_diff_max": float(getattr(root.config, "DT_DIFF_MAX", 0.1)),
    }


def _resolve_component_id(
    problem: ProcessComponentHost, requested_id: str | None
) -> str:
    if requested_id:
        component_id = str(requested_id)
        if component_id in problem.process_components:
            raise ValueError(f"Process component id {component_id!r} already exists.")
        return component_id
    counter = 1
    while f"mvr_{counter}" in problem.process_components:
        counter += 1
    return f"mvr_{counter}"


__all__ = [
    "ProcessMVRComponent",
    "create_process_mvr_component",
    "replay_process_components",
    "restore_process_components",
]
