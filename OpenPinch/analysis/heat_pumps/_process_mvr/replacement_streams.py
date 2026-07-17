"""Replacement-stream construction for Process MVR components."""

from __future__ import annotations

from ....domain._stream.linearisation import (
    align_temperature_heat_profiles,
    normalise_temperature_heat_profile,
)
from ....domain._stream.segment import StreamSegment
from ....domain.configuration import tol
from ....domain.stream import Stream
from ....domain.value import Value
from ..direct_mvr.execution import solve_direct_gas_mvr_stream
from ..direct_mvr.models import (
    DirectGasMVRSettings,
    DirectGasMVRStageResult,
)
from .state import _ProcessMVRStreamRecord, _StreamMembership
from .values import enthalpy_delta_to_j_per_kg, value_at_index


def build_process_mvr_stream_record(
    *,
    source_stream: Stream,
    memberships: list[_StreamMembership],
    settings: DirectGasMVRSettings,
    period_ids: dict[str, int] | None,
    num_periods: int | None,
) -> _ProcessMVRStreamRecord:
    """Solve all periods and build one parent-owned source/replacement record."""
    if not memberships:
        raise ValueError(f"MVR source stream {source_stream.name!r} is not in a zone.")
    period_lookup = period_ids or {"0": 0}
    period_count = int(num_periods or max(period_lookup.values(), default=0) + 1 or 1)
    period_labels = period_labels_by_index(period_lookup, period_count)
    solves = [
        solve_direct_gas_mvr_stream(source_stream, settings=settings, idx=index)
        for index in range(period_count)
    ]
    replacement_streams = build_multiperiod_stage_streams(
        source_stream=source_stream,
        solves=solves,
        period_count=period_count,
        stage_count=settings.n_stages,
    )
    stage_results_by_period = {
        period_labels[index]: solves[index].stage_results
        for index in range(period_count)
    }
    return _ProcessMVRStreamRecord(
        original_stream=source_stream,
        original_memberships=memberships,
        replacement_streams=replacement_streams,
        replacement_memberships=[],
        stage_results_by_period=stage_results_by_period,
        period_label_by_index=period_labels,
    )


def build_multiperiod_stage_streams(
    *,
    source_stream: Stream,
    solves,
    period_count: int,
    stage_count: int,
) -> list[Stream]:
    """Build aligned segmented replacement streams across operating periods."""
    streams: list[Stream] = []
    for stage_index in range(1, stage_count + 1):
        stage_profiles = [
            solve.stage_results[stage_index - 1].linearised_profile for solve in solves
        ]
        aligned_profiles = [
            normalise_temperature_heat_profile(profile, is_hot_stream=True)
            for profile in align_temperature_heat_profiles(stage_profiles)
        ]
        segment_count = len(aligned_profiles[0]) - 1
        stages = [solve.stage_results[stage_index - 1] for solve in solves]
        segments: list[StreamSegment] = []
        for segment_index in range(1, segment_count + 1):
            segment_points = [
                (profile[segment_index - 1], profile[segment_index])
                for profile in aligned_profiles
            ]
            temperature_unit = stages[0].temperature_unit
            pressure_unit = stages[0].pressure_unit
            heat_flow_unit = stages[0].heat_flow_unit
            segments.append(
                StreamSegment(
                    name=(
                        f"{source_stream.name}_direct_MVR_H{stage_index}_S"
                        f"{segment_index}"
                    ),
                    t_supply=Value(
                        [float(start[1]) for start, _end in segment_points],
                        temperature_unit,
                    ),
                    t_target=Value(
                        [float(end[1]) for _start, end in segment_points],
                        temperature_unit,
                    ),
                    p_supply=Value([stage.p_out for stage in stages], pressure_unit),
                    p_target=Value([stage.p_out for stage in stages], pressure_unit),
                    heat_flow=Value(
                        [
                            stage_segment_heat_flow(stage, start[0], end[0]).value
                            for stage, (start, end) in zip(stages, segment_points)
                        ],
                        heat_flow_unit,
                    ),
                    dt_cont=[
                        value_at_index(source_stream.dt_cont, index) or 0.0
                        for index in range(period_count)
                    ],
                    htc=[
                        value_at_index(source_stream.htc, index) or 1.0
                        for index in range(period_count)
                    ],
                    is_process_stream=True,
                    fluid_name=source_stream.fluid_name,
                    fluid_phase=source_stream.fluid_phase,
                )
            )
        stream = Stream(
            name=f"{source_stream.name}_direct_MVR_H{stage_index}",
            segments=segments,
            is_process_stream=True,
            fluid_name=source_stream.fluid_name,
            fluid_phase=source_stream.fluid_phase,
        )
        stream.active = True
        streams.append(stream)
    return streams


def stage_mass_flow(stage: DirectGasMVRStageResult) -> float:
    """Infer stage mass flow from process heat and enthalpy change."""
    enthalpy_delta = enthalpy_delta_to_j_per_kg(
        stage.h_hot_supply,
        stage.h_target,
        stage.enthalpy_unit,
    )
    if enthalpy_delta <= tol:
        return 0.0
    heat_flow_watts = Value(stage.heat_flow, stage.heat_flow_unit).to("W").value
    return heat_flow_watts / enthalpy_delta


def stage_segment_heat_flow(
    stage: DirectGasMVRStageResult,
    enthalpy_start: float,
    enthalpy_end: float,
) -> Value:
    """Return one segment duty on the stage's configured heat-flow unit."""
    enthalpy_delta = enthalpy_delta_to_j_per_kg(
        enthalpy_start,
        enthalpy_end,
        stage.enthalpy_unit,
    )
    return Value(stage_mass_flow(stage) * enthalpy_delta, "W").to(stage.heat_flow_unit)


def period_labels_by_index(
    period_ids: dict[str, int] | None,
    period_count: int,
) -> dict[int, str]:
    labels = {index: str(index) for index in range(period_count)}
    for period_id, index in (period_ids or {}).items():
        labels[int(index)] = str(period_id)
    return labels


__all__: list[str] = []
