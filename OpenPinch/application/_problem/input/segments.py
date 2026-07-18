"""Build segmented process streams from validated nested input data."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....contracts.input import StreamSchema, UtilitySchema
from ....contracts.units import standardise_input_value
from ....domain._stream.segment import StreamSegment
from ....domain.configuration import tol
from ....domain.enums import StreamType
from ....domain.stream import Stream
from ....domain.value import Value
from ....domain.zone import Zone


def _create_segmented_process_stream(stream: StreamSchema, zone: Zone) -> Stream:
    """Create one physical parent stream from ordered nested thermal data."""
    _validate_nested_record_semantics(stream, config=zone.config, section="streams")
    common = {
        "delta_t_contribution": standardise_input_value(
            stream.dt_cont,
            field_name="dt_cont",
            config=zone.config,
        ),
        "delta_t_contribution_multiplier": zone.dt_cont_multiplier,
        "heat_transfer_coefficient": standardise_input_value(
            stream.htc,
            field_name="htc",
            config=zone.config,
        ),
        "is_process_stream": True,
        "fluid_name": stream.fluid_name,
        "fluid_phase": stream.fluid_phase,
    }
    if stream.segments is not None:
        segments = [
            StreamSegment(
                name=segment.name or f"{stream.name}.S{index + 1}",
                supply_temperature=standardise_input_value(
                    segment.t_supply,
                    field_name="t_supply",
                    config=zone.config,
                ),
                target_temperature=standardise_input_value(
                    segment.t_target,
                    field_name="t_target",
                    config=zone.config,
                ),
                supply_pressure=standardise_input_value(
                    segment.p_supply,
                    field_name="p_supply",
                    config=zone.config,
                ),
                target_pressure=standardise_input_value(
                    segment.p_target,
                    field_name="p_target",
                    config=zone.config,
                ),
                supply_enthalpy=standardise_input_value(
                    segment.h_supply,
                    field_name="h_supply",
                    config=zone.config,
                ),
                target_enthalpy=standardise_input_value(
                    segment.h_target,
                    field_name="h_target",
                    config=zone.config,
                ),
                heat_flow=standardise_input_value(
                    segment.heat_flow,
                    field_name="heat_flow",
                    config=zone.config,
                ),
                delta_t_contribution=standardise_input_value(
                    segment.dt_cont if segment.dt_cont is not None else stream.dt_cont,
                    field_name="dt_cont",
                    config=zone.config,
                ),
                heat_transfer_coefficient=standardise_input_value(
                    segment.htc if segment.htc is not None else stream.htc,
                    field_name="htc",
                    config=zone.config,
                ),
                price=standardise_input_value(
                    segment.price,
                    field_name="price",
                    config=zone.config,
                ),
                delta_t_contribution_multiplier=zone.dt_cont_multiplier,
                is_process_stream=True,
                fluid_name=stream.fluid_name,
                fluid_phase=stream.fluid_phase,
                segment_index=index,
            )
            for index, segment in enumerate(stream.segments)
        ]
    else:
        segments = _segments_from_profile(stream, zone.config, common)

    parent = Stream(name=stream.name, segments=segments, **common)
    parent.is_active = stream.active
    _validate_supplied_parent_aggregates(stream, parent, zone)
    return parent


def _segments_from_profile(
    stream: StreamSchema | UtilitySchema,
    config,
    common: dict[str, Any],
) -> list[StreamSegment]:
    from ....domain._stream.linearisation import (
        align_temperature_heat_profiles,
        get_piecewise_data_points,
        normalise_temperature_heat_profile,
    )

    profile = stream.profile
    temperatures = [
        Value(
            standardise_input_value(
                point.temperature,
                field_name="t_supply",
                config=config,
            )
        )
        for point in profile.points
    ]
    cumulative_heat = [
        Value(
            standardise_input_value(
                point.cumulative_heat,
                field_name="heat_flow",
                config=config,
            )
        )
        for point in profile.points
    ]
    num_periods = max(value.num_periods for value in [*temperatures, *cumulative_heat])

    def magnitudes(value: Value) -> np.ndarray:
        if value.num_periods == 1:
            return np.full(num_periods, float(value.value), dtype=float)
        if value.num_periods != num_periods:
            raise ValueError(
                f"Profile for stream {stream.name!r} has inconsistent period counts."
            )
        return value.period_values

    segment_common = {
        key: common[key]
        for key in (
            "delta_t_contribution",
            "delta_t_contribution_multiplier",
            "heat_transfer_coefficient",
            "price",
            "is_process_stream",
            "fluid_name",
            "fluid_phase",
        )
        if key in common
    }
    period_profiles = []
    thermal_direction: float | None = None
    for period_index in range(num_periods):
        raw_profile = np.column_stack(
            (
                [magnitudes(value)[period_index] for value in cumulative_heat],
                [magnitudes(value)[period_index] for value in temperatures],
            )
        )
        heat_steps = np.diff(raw_profile[:, 0])
        if not np.isfinite(raw_profile).all():
            raise ValueError(
                f"Profile for stream {stream.name!r} must contain finite values "
                f"in period {period_index}."
            )
        if np.any(heat_steps <= tol):
            raise ValueError(
                f"Profile for stream {stream.name!r} cumulative heat must increase "
                f"strictly in period {period_index}."
            )
        temperature_steps = np.diff(raw_profile[:, 1])
        nonzero_signs = np.sign(temperature_steps[np.abs(temperature_steps) > tol])
        if nonzero_signs.size == 0 or not np.all(nonzero_signs == nonzero_signs[0]):
            raise ValueError(
                f"Profile for stream {stream.name!r} temperatures must preserve "
                f"one direction in period {period_index}."
            )
        period_direction = float(nonzero_signs[0])
        if thermal_direction is None:
            thermal_direction = period_direction
        elif period_direction != thermal_direction:
            raise ValueError(
                f"Profile for stream {stream.name!r} changes thermal direction "
                "between operating periods."
            )
        period_profiles.append(
            normalise_temperature_heat_profile(
                get_piecewise_data_points(
                    curve=raw_profile,
                    is_hot_stream=period_direction < 0.0,
                    dt_diff_max=profile.linearisation_tolerance,
                ),
                is_hot_stream=period_direction < 0.0,
            )
        )
    aligned_profiles = align_temperature_heat_profiles(period_profiles)

    segments = []
    for index in range(len(aligned_profiles[0]) - 1):
        start_t = np.asarray(
            [period_profile[index, 1] for period_profile in aligned_profiles]
        )
        end_t = np.asarray(
            [period_profile[index + 1, 1] for period_profile in aligned_profiles]
        )
        start_h = np.asarray(
            [period_profile[index, 0] for period_profile in aligned_profiles]
        )
        end_h = np.asarray(
            [period_profile[index + 1, 0] for period_profile in aligned_profiles]
        )
        segments.append(
            StreamSegment(
                name=f"{stream.name}.S{index + 1}",
                supply_temperature=Value(start_t, temperatures[0].unit),
                target_temperature=Value(end_t, temperatures[0].unit),
                heat_flow=Value(np.abs(end_h - start_h), cumulative_heat[0].unit),
                segment_index=index,
                **segment_common,
            )
        )
    return segments


def _create_segmented_utility_stream(
    utility: UtilitySchema,
    *,
    utility_type: str,
    config,
    dt_cont_multiplier: float,
) -> Stream:
    """Create one ordered segmented utility, preserving local segment prices."""
    _validate_nested_record_semantics(utility, config=config, section="utilities")
    common = {
        "delta_t_contribution": standardise_input_value(
            utility.dt_cont,
            field_name="dt_cont",
            config=config,
        ),
        "delta_t_contribution_multiplier": dt_cont_multiplier,
        "heat_transfer_coefficient": standardise_input_value(
            utility.htc,
            field_name="htc",
            config=config,
        ),
        "is_process_stream": False,
        "fluid_name": utility.fluid_name,
        "fluid_phase": utility.fluid_phase,
    }
    parent_price = standardise_input_value(
        utility.price,
        field_name="price",
        config=config,
    )
    if utility.segments is not None:
        segments = [
            StreamSegment(
                name=segment.name or f"{utility.name}.S{index + 1}",
                supply_temperature=standardise_input_value(
                    segment.t_supply,
                    field_name="t_supply",
                    config=config,
                ),
                target_temperature=standardise_input_value(
                    segment.t_target,
                    field_name="t_target",
                    config=config,
                ),
                supply_pressure=standardise_input_value(
                    segment.p_supply,
                    field_name="p_supply",
                    config=config,
                ),
                target_pressure=standardise_input_value(
                    segment.p_target,
                    field_name="p_target",
                    config=config,
                ),
                supply_enthalpy=standardise_input_value(
                    segment.h_supply,
                    field_name="h_supply",
                    config=config,
                ),
                target_enthalpy=standardise_input_value(
                    segment.h_target,
                    field_name="h_target",
                    config=config,
                ),
                heat_flow=standardise_input_value(
                    segment.heat_flow,
                    field_name="heat_flow",
                    config=config,
                ),
                delta_t_contribution=standardise_input_value(
                    segment.dt_cont if segment.dt_cont is not None else utility.dt_cont,
                    field_name="dt_cont",
                    config=config,
                ),
                heat_transfer_coefficient=standardise_input_value(
                    segment.htc if segment.htc is not None else utility.htc,
                    field_name="htc",
                    config=config,
                ),
                price=standardise_input_value(
                    segment.price if segment.price is not None else utility.price,
                    field_name="price",
                    config=config,
                ),
                delta_t_contribution_multiplier=dt_cont_multiplier,
                is_process_stream=False,
                fluid_name=utility.fluid_name,
                fluid_phase=utility.fluid_phase,
                segment_index=index,
            )
            for index, segment in enumerate(utility.segments)
        ]
    else:
        segments = _segments_from_profile(
            utility,
            config,
            {**common, "price": parent_price},
        )

    parent = Stream(name=utility.name, segments=segments, price=None, **common)
    parent.is_active = utility.active
    _validate_supplied_parent_aggregates(utility, parent, config)
    if parent.stream_type != utility_type:
        candidates = []
        for segment in reversed(parent.segments):
            candidate = parent._detached_segment(segment)
            candidate._t_supply, candidate._t_target = (
                candidate._t_target,
                candidate._t_supply,
            )
            candidate._p_supply, candidate._p_target = (
                candidate._p_target,
                candidate._p_supply,
            )
            candidate._h_supply, candidate._h_target = (
                candidate._h_target,
                candidate._h_supply,
            )
            candidate.update_derived_properties()
            candidates.append(candidate)
        parent.replace_segments(candidates)
    if parent.stream_type != utility_type or utility_type not in {
        StreamType.Hot.value,
        StreamType.Cold.value,
    }:
        raise ValueError(
            f"Segmented utility {utility.name!r} cannot be oriented as "
            f"{utility_type!r}."
        )
    return parent


def _validate_supplied_parent_aggregates(
    schema: StreamSchema | UtilitySchema,
    parent: Stream,
    zone_or_config,
) -> None:
    config = getattr(zone_or_config, "config", zone_or_config)
    fields = {
        "t_supply": parent.supply_temperature,
        "t_target": parent.target_temperature,
        "heat_flow": parent.heat_flow,
    }
    for field_name, actual in fields.items():
        supplied = getattr(schema, field_name)
        if supplied is None:
            continue
        expected = Value(
            standardise_input_value(
                supplied,
                field_name=field_name,
                config=config,
            )
        ).to(actual.unit)
        expected_values = expected.period_values
        actual_values = actual.period_values
        if expected_values.size == 1 and actual_values.size > 1:
            expected_values = np.full(actual_values.size, expected_values[0])
        if actual_values.size == 1 and expected_values.size > 1:
            actual_values = np.full(expected_values.size, actual_values[0])
        if not np.allclose(expected_values, actual_values, atol=tol, rtol=0.0):
            raise ValueError(
                f"Segmented stream {schema.name!r} supplied {field_name} does not "
                "match the authoritative profile."
            )


def _validate_nested_record_semantics(
    schema: StreamSchema | UtilitySchema,
    *,
    config,
    section: str,
) -> None:
    """Apply the same nested semantic checks used by validation reports."""
    from .semantics import segmented_record_issues

    issues = segmented_record_issues(
        schema,
        section=section,
        record_index=0,
        record_label=schema.name,
        config=config,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        details = "; ".join(f"{issue.path}: {issue.message}" for issue in errors)
        raise ValueError(details)


__all__: list[str] = []
