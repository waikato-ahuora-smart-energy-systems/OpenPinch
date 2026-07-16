"""Pure pinch-design preprocessing and area helpers."""

from __future__ import annotations

from collections.abc import Iterable


def _overall_heat_transfer_coefficient(left_htc: float, right_htc: float) -> float:
    return 1 / (1 / left_htc + 1 / right_htc)


def _active_period_flag(values: Iterable[float], tolerance: float) -> list[int]:
    return [1 if max(values) > tolerance else 0]


def _lmtd_formula_allowed(
    delta_1: float,
    delta_2: float,
    approach_temperature: float,
    tolerance: float,
    second_approach_temperature: float | None = None,
) -> bool:
    second_approach = (
        approach_temperature
        if second_approach_temperature is None
        else second_approach_temperature
    )
    return (
        abs(delta_1 - delta_2) > tolerance
        and delta_1 - approach_temperature >= tolerance
        and delta_2 - second_approach >= tolerance
    )


def _area_from_heat_load(
    heat_load: float,
    overall_heat_transfer_coefficient: float,
    lmtd: float,
    tolerance: float,
) -> float:
    if lmtd <= tolerance or heat_load <= tolerance:
        return 0.0
    return heat_load / overall_heat_transfer_coefficient / lmtd
