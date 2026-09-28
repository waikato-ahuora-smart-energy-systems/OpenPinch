"""Start-point heuristics for the utility-placement decision vector.

Each heuristic returns one candidate point (or ``None`` when it does not
apply); :func:`generate_start_points` combines them in a fixed order. The
model builder then keeps only the points the candidate verifier accepts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionCoordinate,
    DecisionField,
    EffectiveUtilityTemplate,
    PlacementFeasibilityEnvelope,
    UtilityLevelKind,
    UtilitySide,
    UtilityTemplateSet,
)

from .errors import PlacementModelValidationError

_SENSIBLE_START_FRACTIONS = (0.5, 0.2, 0.4, 0.6, 0.8)
_SUPPLY_START_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

Values = dict[CoordinateKey, float]
Point = tuple[float, ...]


def _key(template: EffectiveUtilityTemplate, field: DecisionField) -> CoordinateKey:
    return CoordinateKey(template_key=template.key, field=field)


def _supply_key(template: EffectiveUtilityTemplate) -> CoordinateKey:
    return _key(template, DecisionField.SUPPLY_TEMPERATURE)


def _span_key(template: EffectiveUtilityTemplate) -> CoordinateKey:
    return _key(template, DecisionField.TEMPERATURE_SPAN)


def _point(values: Mapping[CoordinateKey, float], coordinates) -> Point:
    return tuple(values[item.coordinate] for item in coordinates)


def _clamp(value: float, coordinate: DecisionCoordinate) -> float:
    return min(coordinate.bounds.upper, max(coordinate.bounds.lower, value))


def _isothermal(templates: UtilityTemplateSet) -> tuple[EffectiveUtilityTemplate, ...]:
    return tuple(
        item for item in templates.hot if item.kind is UtilityLevelKind.ISOTHERMAL
    )


def _sensible(templates: UtilityTemplateSet) -> tuple[EffectiveUtilityTemplate, ...]:
    return tuple(
        item for item in templates.hot if item.kind is UtilityLevelKind.SENSIBLE
    )


def _last_isothermal_cold_supply(
    templates: UtilityTemplateSet, values: Mapping[CoordinateKey, float]
) -> float:
    """Return the cold-side supply of the last generated isothermal pair."""
    previous_template = _isothermal(templates)[-1]
    previous_supply = values[_supply_key(previous_template)]
    if previous_template.fixed_span is None:
        raise PlacementModelValidationError(
            code="missing_fixed_span",
            message="Generated isothermal utility requires a fixed span.",
            template_key=previous_template.key,
        )
    return previous_supply - previous_template.fixed_span.value


def sensible_profile_start(
    templates: UtilityTemplateSet,
    by_key: Mapping[CoordinateKey, DecisionCoordinate],
    initial_values: Values,
) -> Values:
    """Stretch sensible levels so the last one reaches the coldest cold supply."""
    sensible_templates = _sensible(templates)
    values = dict(initial_values)
    final_cold_lower = templates.cold[-1].supply_bounds.lower
    cold_range = templates.cold[0].supply_bounds.upper - final_cold_lower
    denominator = max(len(sensible_templates) - 1, 1)
    for rank, template in enumerate(sensible_templates):
        supply_key = _supply_key(template)
        supply_coordinate = by_key[supply_key]
        span_coordinate = by_key[_span_key(template)]
        if rank == len(sensible_templates) - 1:
            supply = supply_coordinate.bounds.lower
            desired_cold_supply = final_cold_lower
        else:
            supply = values[supply_key]
            target_fraction = 0.28 * (len(sensible_templates) - rank - 1) / denominator
            desired_cold_supply = final_cold_lower + target_fraction * cold_range
        values[supply_key] = supply
        values[span_coordinate.coordinate] = _clamp(
            supply - desired_cold_supply, span_coordinate
        )
    return values


def coverage_start(
    templates: UtilityTemplateSet,
    by_key: Mapping[CoordinateKey, DecisionCoordinate],
    initial_values: Values,
) -> Values:
    """Spread sensible cold ends evenly below the last isothermal pair."""
    sensible_templates = _sensible(templates)
    values = dict(initial_values)
    previous_cold_supply = _last_isothermal_cold_supply(templates, values)
    final_cold_supply = templates.cold[-1].supply_bounds.lower
    for rank, template in enumerate(sensible_templates, start=1):
        supply = values[_supply_key(template)]
        desired_cold_supply = previous_cold_supply + (
            rank / len(sensible_templates) * (final_cold_supply - previous_cold_supply)
        )
        span_coordinate = by_key[_span_key(template)]
        values[span_coordinate.coordinate] = _clamp(
            supply - desired_cold_supply, span_coordinate
        )
    return values


def interleaved_start(
    templates: UtilityTemplateSet,
    by_key: Mapping[CoordinateKey, DecisionCoordinate],
    initial_values: Values,
    minimum_separation: float,
) -> Values | None:
    """Alternate isothermal and sensible levels from the top of their bounds.

    Returns ``None`` when the separation pushes a level below its bounds.
    """
    isothermal_templates = _isothermal(templates)
    sensible_templates = _sensible(templates)
    values = dict(initial_values)
    order = tuple(
        template
        for rank in range(max(len(isothermal_templates), len(sensible_templates)))
        for template in (
            isothermal_templates[rank : rank + 1] + sensible_templates[rank : rank + 1]
        )
    )
    previous_supply: float | None = None
    for template in order:
        key = _supply_key(template)
        bounds = by_key[key].bounds
        supply = bounds.upper
        if previous_supply is not None:
            supply = min(supply, previous_supply - minimum_separation)
        if supply < bounds.lower:
            return None
        values[key] = supply
        previous_supply = supply
    return values


def spread_start(
    templates: UtilityTemplateSet,
    by_key: Mapping[CoordinateKey, DecisionCoordinate],
    initial_values: Values,
) -> Values:
    """Spread hot supplies evenly across their bounds with minimum spans."""
    values = dict(initial_values)
    denominator = max(len(templates.hot) - 1, 1)
    for rank, template in enumerate(templates.hot):
        supply_coordinate = by_key[_supply_key(template)]
        progress = rank / denominator
        spread_supply = supply_coordinate.bounds.upper + progress * (
            supply_coordinate.bounds.lower - supply_coordinate.bounds.upper
        )
        values[supply_coordinate.coordinate] = _clamp(spread_supply, supply_coordinate)
        if template.kind is UtilityLevelKind.SENSIBLE:
            span_coordinate = by_key[_span_key(template)]
            values[span_coordinate.coordinate] = span_coordinate.bounds.lower
    return values


def gap_start(
    templates: UtilityTemplateSet,
    by_key: Mapping[CoordinateKey, DecisionCoordinate],
    spread_values: Values,
) -> Values:
    """From the spread start, close the gap below the last isothermal pair.

    Sensible cold ends follow an eased (quadratic) progression.
    """
    sensible_templates = _sensible(templates)
    values = dict(spread_values)
    previous_cold_supply = _last_isothermal_cold_supply(templates, values)
    final_cold_supply = templates.cold[-1].supply_bounds.lower
    for rank, template in enumerate(sensible_templates, start=1):
        progress = rank / len(sensible_templates)
        eased_progress = 1.0 - (1.0 - progress) ** 2
        desired_cold_supply = previous_cold_supply + eased_progress * (
            final_cold_supply - previous_cold_supply
        )
        supply = values[_supply_key(template)]
        span_coordinate = by_key[_span_key(template)]
        values[span_coordinate.coordinate] = _clamp(
            supply - desired_cold_supply, span_coordinate
        )
    return values


def fractional_starts(
    templates: UtilityTemplateSet,
    coordinates: Sequence[DecisionCoordinate],
    initial_values: Values,
    existing: list[Point],
    start_limit: int,
) -> None:
    """Append grid starts (supply fraction x span fraction) up to ``start_limit``."""
    supply_progress = {
        template.key: rank / max(len(side_templates) - 1, 1)
        for side_templates in (templates.hot, templates.cold)
        for rank, template in enumerate(side_templates)
    }
    for supply_fraction in _SUPPLY_START_FRACTIONS:
        if len(existing) >= start_limit:
            break
        supply_values = dict(initial_values)
        for coordinate in coordinates:
            key = coordinate.coordinate
            if key.field is not DecisionField.SUPPLY_TEMPERATURE:
                continue
            edge = initial_values[key]
            opposite = (
                coordinate.bounds.lower
                if key.template_key.side is UtilitySide.HOT
                else coordinate.bounds.upper
            )
            supply_values[key] = edge + (
                supply_fraction * supply_progress[key.template_key] * (opposite - edge)
            )
        for span_fraction in _SENSIBLE_START_FRACTIONS:
            if len(existing) >= start_limit:
                break
            values = dict(supply_values)
            for coordinate in coordinates:
                if coordinate.coordinate.field is DecisionField.TEMPERATURE_SPAN:
                    values[coordinate.coordinate] = coordinate.bounds.lower + (
                        span_fraction
                        * (coordinate.bounds.upper - coordinate.bounds.lower)
                    )
            point = _point(values, coordinates)
            if point not in existing:
                existing.append(point)


def generate_start_points(
    templates: UtilityTemplateSet,
    coordinates: Sequence[DecisionCoordinate],
    initial_values: Values,
    envelope: PlacementFeasibilityEnvelope,
    *,
    paired: bool,
) -> list[Point]:
    """Return de-duplicated candidate starts, at most ``start_limit`` of them.

    Order: for generated pairs with sensible levels, the sensible profile,
    coverage, interleaved and gap starts; then (for generated pairs) the
    spread start; then the fractional grid.
    """
    start_limit = max(4, min(20, 20_000 // max(len(coordinates), 1)))
    points: list[Point] = []
    if paired:
        by_key = {item.coordinate: item for item in coordinates}
        has_sensible = bool(_sensible(templates))
        if has_sensible:
            points.append(
                _point(
                    sensible_profile_start(templates, by_key, initial_values),
                    coordinates,
                )
            )
            points.append(
                _point(coverage_start(templates, by_key, initial_values), coordinates)
            )
            interleaved = interleaved_start(
                templates,
                by_key,
                initial_values,
                envelope.minimum_separation.value,
            )
            if interleaved is not None:
                points.append(_point(interleaved, coordinates))
        spread = spread_start(templates, by_key, initial_values)
        if has_sensible:
            points.append(_point(gap_start(templates, by_key, spread), coordinates))
        points.append(_point(spread, coordinates))
    fractional_starts(templates, coordinates, initial_values, points, start_limit)
    return list(dict.fromkeys(points))[:start_limit]


__all__ = [
    "coverage_start",
    "fractional_starts",
    "gap_start",
    "generate_start_points",
    "interleaved_start",
    "sensible_profile_start",
    "spread_start",
]
