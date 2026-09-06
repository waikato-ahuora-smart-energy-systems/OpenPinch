"""Lazy canonical operating-point expansion for HPR performance maps."""

from __future__ import annotations

from collections.abc import Iterator

from .models import HprMapGenerationContext, HprOperatingPoint


def iter_hpr_operating_points(
    context: HprMapGenerationContext,
) -> Iterator[HprOperatingPoint]:
    """Yield the canonical source/sink/load Cartesian product exactly once."""
    ordinal = 0
    basis = context.basis
    for source_index, source_temperature in enumerate(
        context.request.source_temperatures
    ):
        evaporating_temperature = source_temperature - basis.source_approach_temperature
        for sink_index, sink_temperature in enumerate(
            context.request.sink_temperatures
        ):
            condensing_temperature = sink_temperature + basis.sink_approach_temperature
            curve_id = f"{context.request.map_id}-s{source_index}-k{sink_index}"
            for load_index, load_fraction in enumerate(context.request.load_fractions):
                validation_error: str | None = None
                if (
                    evaporating_temperature <= -273.15
                    or condensing_temperature <= -273.15
                ):
                    validation_error = (
                        "translated temperatures must exceed absolute zero"
                    )
                elif condensing_temperature <= evaporating_temperature:
                    validation_error = "translated temperatures require positive lift"
                requested_useful_duty = load_fraction * context.reference_capacity
                yield HprOperatingPoint(
                    ordinal=ordinal,
                    source_index=source_index,
                    sink_index=sink_index,
                    load_index=load_index,
                    curve_id=curve_id,
                    name=f"{curve_id}-l{load_index}",
                    source_temperature=source_temperature,
                    sink_temperature=sink_temperature,
                    evaporating_temperature=evaporating_temperature,
                    condensing_temperature=condensing_temperature,
                    load_fraction=load_fraction,
                    requested_useful_duty=requested_useful_duty,
                    validation_error=validation_error,
                )
                ordinal += 1


__all__ = ["iter_hpr_operating_points"]
