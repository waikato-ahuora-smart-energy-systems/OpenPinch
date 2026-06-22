"""MVR-specific process component records and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from CoolProp.CoolProp import PropsSI

from ...classes.stream import Stream
from ...classes.value import Value
from ...classes.zone import Zone
from ...lib.config import tol
from ...lib.enums import ST
from ..common.miscellaneous import get_state_index
from .direct_mvr import (
    DirectGasMVRSettings,
    DirectGasMVRStageResult,
    coerce_positive_mvr_stage_count,
    solve_direct_gas_mvr_stream,
)
from .process_components import ProcessComponent

if TYPE_CHECKING:
    from ...classes.pinch_problem import PinchProblem


@dataclass
class StreamMembership:
    """One occurrence of a stream inside a zone hot-stream collection."""

    zone: Zone
    key: str


@dataclass
class ProcessMVRStreamRecord:
    """Original and replacement streams for one MVR source stream."""

    original_stream: Stream
    original_memberships: list[StreamMembership]
    replacement_streams: list[Stream]
    replacement_memberships: list[StreamMembership]
    stage_results_by_state: dict[str, list[DirectGasMVRStageResult]]
    state_label_by_index: dict[int, str] = field(default_factory=dict)


@dataclass
class ProcessMVRComponent(ProcessComponent):
    """Memory-only direct process MVR component."""

    settings: DirectGasMVRSettings = field(default_factory=DirectGasMVRSettings)
    source_selectors: list[Any] = field(default_factory=list)
    stream_records: list[ProcessMVRStreamRecord] = field(default_factory=list)
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
    def stage_results_by_state(self) -> dict[str, list[DirectGasMVRStageResult]]:
        combined: dict[str, list[DirectGasMVRStageResult]] = {}
        for record in self.stream_records:
            for state_id, stages in record.stage_results_by_state.items():
                combined.setdefault(state_id, []).extend(stages)
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
        state_id: str | None = None,
        state_idx: int | None = None,
    ) -> float:
        """Return active compressor work assigned to streams inside ``zone``."""
        if not self.active:
            return 0.0
        total = 0.0
        for record in self.stream_records:
            if not _record_affects_zone(record, zone):
                continue
            stages = _record_stage_results_for_state(
                record,
                state_id=state_id,
                state_idx=state_idx,
            )
            total += sum(stage.work for stage in stages)
        return float(total)


def create_process_mvr_component(
    problem: "PinchProblem",
    *,
    source_streams,
    mvr_id: str | None = None,
    n_stages: int = 1,
    liquid_injection: bool = True,
    mvr_stage_t_lift: float | None = None,
    mvr_stage_pressure_ratio: float | None = None,
    max_stage_t_sat_lift: float | None = None,
    eta_mvr_comp: float | None = None,
    eta_motor: float | None = None,
    options: dict | None = None,
    state_id: str | None = None,
) -> ProcessMVRComponent:
    """Create, activate, and register a direct process MVR component."""
    root = problem._require_prepared_root_zone()
    runtime_options = dict(options or {})
    if state_id is not None:
        option_state_id = runtime_options.get("state_id")
        if option_state_id is not None and str(option_state_id) != str(state_id):
            raise ValueError(
                "Specify only one process MVR state context; state_id and "
                "options['state_id'] conflict."
            )
        runtime_options["state_id"] = state_id
    get_state_index(root.state_ids, runtime_options)
    resolved_stage_t_lift = _resolve_stage_t_lift_alias(
        mvr_stage_t_lift=mvr_stage_t_lift,
        max_stage_t_sat_lift=max_stage_t_sat_lift,
        mvr_stage_pressure_ratio=mvr_stage_pressure_ratio,
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
    selectors = _normalise_source_selectors(source_streams)
    matched_streams = _match_source_streams(root, selectors)
    records = [
        _build_process_mvr_stream_record(
            source_stream=stream,
            memberships=_find_hot_stream_memberships(root, stream),
            settings=settings,
            state_ids=root.state_ids,
            num_states=root.num_states,
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
        _add_replacement_streams_to_memberships(record, component_id)
    problem.process_components[component_id] = component
    component.activate()
    return component


def _resolve_stage_t_lift_alias(
    *,
    mvr_stage_t_lift: float | None,
    max_stage_t_sat_lift: float | None,
    mvr_stage_pressure_ratio: float | None,
) -> float | None:
    if mvr_stage_t_lift is not None and max_stage_t_sat_lift is not None:
        raise ValueError(
            "Specify either mvr_stage_t_lift or max_stage_t_sat_lift, not both."
        )
    resolved_stage_t_lift = (
        mvr_stage_t_lift if mvr_stage_t_lift is not None else max_stage_t_sat_lift
    )
    if resolved_stage_t_lift is not None and mvr_stage_pressure_ratio is not None:
        raise ValueError(
            "Specify either mvr_stage_t_lift or mvr_stage_pressure_ratio, not both."
        )
    return None if resolved_stage_t_lift is None else float(resolved_stage_t_lift)


def _build_process_mvr_stream_record(
    *,
    source_stream: Stream,
    memberships: list[StreamMembership],
    settings: DirectGasMVRSettings,
    state_ids: dict[str, int] | None,
    num_states: int | None,
) -> ProcessMVRStreamRecord:
    if not memberships:
        raise ValueError(f"MVR source stream {source_stream.name!r} is not in a zone.")
    state_lookup = state_ids or {"0": 0}
    n_states = int(num_states or max(state_lookup.values(), default=0) + 1 or 1)
    state_labels = _state_labels_by_index(state_lookup, n_states)
    solves = [
        solve_direct_gas_mvr_stream(source_stream, settings=settings, idx=idx)
        for idx in range(n_states)
    ]
    replacement_streams = _build_multistate_stage_streams(
        source_stream=source_stream,
        solves=solves,
        n_states=n_states,
        n_stages=settings.n_stages,
    )
    stage_results_by_state = {
        state_labels[idx]: solves[idx].stage_results for idx in range(n_states)
    }
    return ProcessMVRStreamRecord(
        original_stream=source_stream,
        original_memberships=memberships,
        replacement_streams=replacement_streams,
        replacement_memberships=[],
        stage_results_by_state=stage_results_by_state,
        state_label_by_index=state_labels,
    )


def _add_replacement_streams_to_memberships(
    record: ProcessMVRStreamRecord,
    component_id: str,
) -> None:
    for membership in record.original_memberships:
        for stream in record.replacement_streams:
            key = f"{membership.key}.{component_id}.{stream.name}"
            membership.zone.hot_streams.add(stream, key=key, prevent_overwrite=False)
            record.replacement_memberships.append(
                StreamMembership(zone=membership.zone, key=key)
            )


def _build_multistate_stage_streams(
    *,
    source_stream: Stream,
    solves,
    n_states: int,
    n_stages: int,
) -> list[Stream]:
    streams: list[Stream] = []
    for stage_index in range(1, n_stages + 1):
        stage_profiles = [
            solve.stage_results[stage_index - 1].linearised_profile for solve in solves
        ]
        segment_count = max(len(profile) - 1 for profile in stage_profiles)
        resampled_profiles = [
            _resample_profile_to_segment_count(profile, segment_count)
            for profile in stage_profiles
        ]
        stages = [solve.stage_results[stage_index - 1] for solve in solves]
        for segment_index in range(1, segment_count + 1):
            segment_points = [
                (profile[segment_index - 1], profile[segment_index])
                for profile in resampled_profiles
            ]
            temperature_unit = stages[0].temperature_unit
            pressure_unit = stages[0].pressure_unit
            heat_flow_unit = stages[0].heat_flow_unit
            stream = Stream(
                name=(
                    f"{source_stream.name}_direct_MVR_H{stage_index}_S{segment_index}"
                ),
                t_supply=Value(
                    [float(start[1]) for start, _end in segment_points],
                    temperature_unit,
                ),
                t_target=Value(
                    [
                        _segment_target_temperature(start, end)
                        for start, end in segment_points
                    ],
                    temperature_unit,
                ),
                p_supply=Value([stage.p_out for stage in stages], pressure_unit),
                p_target=Value([stage.p_out for stage in stages], pressure_unit),
                heat_flow=Value(
                    [
                        _stage_segment_heat_flow(stage, start[0], end[0]).value
                        for stage, (start, end) in zip(stages, segment_points)
                    ],
                    heat_flow_unit,
                ),
                dt_cont=[
                    _value_at_idx(source_stream.dt_cont, idx) or 0.0
                    for idx in range(n_states)
                ],
                htc=[
                    _value_at_idx(source_stream.htc, idx) or 1.0
                    for idx in range(n_states)
                ],
                is_process_stream=True,
                fluid_name=source_stream.fluid_name,
                fluid_phase=source_stream.fluid_phase,
            )
            stream.active = True
            streams.append(stream)
    return streams


def _resample_profile_to_segment_count(
    profile,
    segment_count: int,
) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    if len(profile) == segment_count + 1:
        return profile
    h_start = float(profile[0, 0])
    h_end = float(profile[-1, 0])
    h_values = np.linspace(h_start, h_end, segment_count + 1)
    xp = profile[::-1, 0]
    fp = profile[::-1, 1]
    t_values = np.interp(h_values[::-1], xp, fp)[::-1]
    return np.column_stack((h_values, t_values))


def _segment_target_temperature(start, end) -> float:
    t_start = float(start[1])
    t_end = float(end[1])
    if abs(t_start - t_end) < 0.01:
        return t_start - 0.01
    return t_end


def _stage_mass_flow(stage: DirectGasMVRStageResult) -> float:
    delta_h = _enthalpy_delta_to_j_per_kg(
        stage.h_hot_supply,
        stage.h_target,
        stage.enthalpy_unit,
    )
    if delta_h <= tol:
        return 0.0
    heat_flow_w = Value(stage.heat_flow, stage.heat_flow_unit).to("W").value
    return heat_flow_w / delta_h


def _stage_segment_heat_flow(
    stage: DirectGasMVRStageResult,
    h_start: float,
    h_end: float,
) -> Value:
    delta_h = _enthalpy_delta_to_j_per_kg(h_start, h_end, stage.enthalpy_unit)
    return Value(_stage_mass_flow(stage) * delta_h, "W").to(stage.heat_flow_unit)


def _normalise_source_selectors(source_streams) -> list:
    if source_streams is None:
        raise ValueError("source_streams is required for process MVR.")
    if isinstance(source_streams, (str, Stream)):
        return [source_streams]
    selectors = list(source_streams)
    if not selectors:
        raise ValueError("source_streams must include at least one stream selector.")
    return selectors


def _match_source_streams(root: Zone, selectors: list) -> list[Stream]:
    matched: list[Stream] = []
    for selector in selectors:
        selector_matches = _match_one_selector(root, selector)
        for stream in selector_matches:
            if not any(stream is existing for existing in matched):
                matched.append(stream)
    if not matched:
        raise ValueError("No active hot gas process streams matched source_streams.")
    return matched


def _match_one_selector(root: Zone, selector) -> list[Stream]:
    matches: list[Stream] = []
    for zone in _walk_zones(root):
        for key, stream in zone.hot_streams.items():
            if not _selector_matches(selector, key, stream):
                continue
            if not any(stream is existing for existing in matches):
                matches.append(stream)
    if not matches:
        cold_matches = _match_cold_streams(root, selector)
        if cold_matches:
            raise ValueError(f"MVR source stream {selector!r} must be a hot stream.")
        raise ValueError(f"MVR source stream {selector!r} was not found.")
    for stream in matches:
        _validate_process_mvr_source(stream, selector)
    return matches


def _match_cold_streams(root: Zone, selector) -> list[Stream]:
    matches: list[Stream] = []
    for zone in _walk_zones(root):
        for key, stream in zone.cold_streams.items():
            if _selector_matches(selector, key, stream):
                matches.append(stream)
    return matches


def _selector_matches(selector, key: str, stream: Stream) -> bool:
    if isinstance(selector, Stream):
        return stream is selector
    selector_text = str(selector)
    return key == selector_text or stream.name == selector_text


def _validate_process_mvr_source(stream: Stream, selector) -> None:
    if not stream.active:
        raise ValueError(f"MVR source stream {selector!r} is not active.")
    if not stream.is_process_stream:
        raise ValueError(f"MVR source stream {selector!r} must be a process stream.")
    if stream.type != ST.Hot.value:
        raise ValueError(f"MVR source stream {selector!r} must be a hot stream.")
    phase = str(stream.fluid_phase or "").strip().lower()
    if phase not in {"gas", "vapour", "vapor"}:
        raise ValueError(
            f"MVR source stream {selector!r} must have fluid_phase='gas' or "
            "fluid_phase='vapour'."
        )
    if stream.fluid_name is None:
        raise ValueError(f"MVR source stream {selector!r} requires fluid_name.")
    _validate_process_mvr_source_phase(stream, selector)


def _validate_process_mvr_source_phase(stream: Stream, selector) -> None:
    n_states = int(stream.num_states or 1)
    p_supply_values: list[float] = []
    p_target_values: list[float] = []
    for idx in range(n_states):
        t_supply = _required_state_value(stream.t_supply, idx, "t_supply", selector)
        t_target = _required_state_value(stream.t_target, idx, "t_target", selector)
        p_supply = _value_at_idx(stream.p_supply, idx, unit="kPa")
        p_target = _value_at_idx(stream.p_target, idx, unit="kPa")
        t_crit = _to_deg_c(_coolprop_value("TCRIT", str(stream.fluid_name)))
        p_crit = _to_kpa(_coolprop_value("PCRIT", str(stream.fluid_name)))
        delta_t = t_supply - t_target

        if delta_t < -tol:
            raise ValueError(
                f"MVR source stream {selector!r} must cool from supply to target."
            )
        if delta_t < 1.0:
            p_supply = _validate_vapour_state(
                selector=selector,
                fluid=str(stream.fluid_name),
                t_supply=t_supply,
                p_supply=p_supply,
                t_crit=t_crit,
            )
            p_target = p_supply if p_target is None else p_target
        else:
            if p_supply is None:
                raise ValueError(
                    f"MVR gas source stream {selector!r} requires p_supply."
                )
            p_target = p_supply if p_target is None else p_target
            _validate_gas_or_supercritical_state(
                selector=selector,
                fluid=str(stream.fluid_name),
                t_c=t_supply,
                p_kpa=p_supply,
                t_crit=t_crit,
                p_crit=p_crit,
                state_label="supply",
            )
            _validate_gas_or_supercritical_state(
                selector=selector,
                fluid=str(stream.fluid_name),
                t_c=t_target,
                p_kpa=p_target,
                t_crit=t_crit,
                p_crit=p_crit,
                state_label="target",
            )
        p_supply_values.append(float(p_supply))
        p_target_values.append(float(p_target))

    if stream.p_supply is None:
        stream.p_supply = _stateful_or_scalar(p_supply_values)
    if stream.p_target is None:
        stream.p_target = _stateful_or_scalar(p_target_values)


def _validate_vapour_state(
    *,
    selector,
    fluid: str,
    t_supply: float,
    p_supply: float | None,
    t_crit: float,
) -> float:
    if t_supply >= t_crit:
        raise ValueError(
            f"MVR vapour source stream {selector!r} requires t_supply below "
            f"the critical temperature ({t_crit:.3g} degC)."
        )
    p_sat = _to_kpa(PropsSI("P", "T", _to_kelvin(t_supply), "Q", 1.0, fluid))
    if p_supply is None:
        return float(p_sat)
    if not np.isclose(p_supply, p_sat, rtol=1e-2, atol=0.5):
        raise ValueError(
            f"MVR vapour source stream {selector!r} requires p_supply close to "
            f"saturation pressure at t_supply ({p_sat:.3g} kPa)."
        )
    return float(p_supply)


def _validate_gas_or_supercritical_state(
    *,
    selector,
    fluid: str,
    t_c: float,
    p_kpa: float,
    t_crit: float,
    p_crit: float,
    state_label: str,
) -> None:
    if p_kpa > p_crit:
        if t_c <= t_crit:
            raise ValueError(
                f"MVR gas source stream {selector!r} has {state_label} pressure "
                "above critical pressure but temperature below critical "
                "temperature."
            )
        return
    if t_c > t_crit:
        return
    p_sat = _to_kpa(PropsSI("P", "T", _to_kelvin(t_c), "Q", 1.0, fluid))
    if p_kpa >= p_sat:
        raise ValueError(
            f"MVR gas source stream {selector!r} has {state_label} pressure "
            f"above saturation pressure ({p_sat:.3g} kPa) at {t_c:.3g} degC."
        )


def _coolprop_value(output: str, fluid: str) -> float:
    try:
        return float(PropsSI(output, fluid))
    except Exception as exc:
        raise ValueError(
            f"MVR source fluid {fluid!r} is not available in CoolProp."
        ) from exc


def _required_state_value(value, idx: int, name: str, selector) -> float:
    unit = "degC" if name.startswith("t_") else None
    selected = _value_at_idx(value, idx, unit=unit)
    if selected is None:
        raise ValueError(f"MVR source stream {selector!r} requires {name}.")
    return selected


def _stateful_or_scalar(values: list[float]):
    return values[0] if len(values) == 1 else values


def _record_affects_zone(record: ProcessMVRStreamRecord, zone: Zone) -> bool:
    zone_address = zone.address
    for membership in record.original_memberships:
        member_address = membership.zone.address
        if member_address == zone_address or member_address.startswith(
            f"{zone_address}/"
        ):
            return True
    return False


def _record_stage_results_for_state(
    record: ProcessMVRStreamRecord,
    *,
    state_id: str | None,
    state_idx: int | None,
) -> list[DirectGasMVRStageResult]:
    if state_id is not None and state_id in record.stage_results_by_state:
        return record.stage_results_by_state[state_id]
    if state_idx is not None:
        idx_label = record.state_label_by_index.get(int(state_idx), str(state_idx))
        if idx_label in record.stage_results_by_state:
            return record.stage_results_by_state[idx_label]
    return next(iter(record.stage_results_by_state.values()), [])


def _find_hot_stream_memberships(root: Zone, stream: Stream) -> list[StreamMembership]:
    memberships: list[StreamMembership] = []
    for zone in _walk_zones(root):
        for key, candidate in zone.hot_streams.items():
            if candidate is stream:
                memberships.append(StreamMembership(zone=zone, key=key))
    return memberships


def _walk_zones(zone: Zone):
    yield zone
    for subzone in zone.subzones.values():
        yield from _walk_zones(subzone)


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


def _state_labels_by_index(
    state_ids: dict[str, int] | None,
    n_states: int,
) -> dict[int, str]:
    labels = {idx: str(idx) for idx in range(n_states)}
    for state_id, idx in (state_ids or {}).items():
        labels[int(idx)] = str(state_id)
    return labels


def _value_at_idx(value, idx: int, *, unit: str | None = None) -> float | None:
    if value is None:
        return None
    try:
        selected = value[idx]
    except Exception:
        selected = value
    if isinstance(selected, Value):
        if unit is not None:
            selected = selected.to(unit)
        return float(selected.value)
    return float(selected)


def _to_kelvin(t_c: float) -> float:
    return float(Value(t_c, "degC").to("K").value)


def _to_deg_c(t_k: float) -> float:
    return float(Value(t_k, "K").to("degC").value)


def _to_kpa(p_pa: float) -> float:
    return float(Value(p_pa, "Pa").to("kPa").value)


def _enthalpy_delta_to_j_per_kg(
    h_start: float,
    h_end: float,
    enthalpy_unit: str,
) -> float:
    delta_h = abs(float(h_start) - float(h_end))
    return float(Value(delta_h, enthalpy_unit).to("J/kg").value)
