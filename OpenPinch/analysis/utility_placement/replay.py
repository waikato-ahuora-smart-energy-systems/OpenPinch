"""Numerics for replaying utility placements against prepared target profiles.

The application adapter prepares zone profiles from a live ``PinchProblem``;
these functions turn those profiles and a decoded placement into allocation
results without touching the problem.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ...contracts.utility_placement import (
    CandidateDiagnostic,
    DecodedPlacement,
    UtilitySide,
)
from ...domain._value.resolution import get_scalar_value
from ...domain.enums import ProblemTableLabel
from ...domain.problem_table import ProblemTable
from ...domain.stream import Stream
from ...domain.stream_collection import StreamCollection
from ..targeting.cascade import get_process_heat_cascade
from ..targeting.direct import (
    _create_net_hot_and_cold_stream_collections_for_site_analysis,
    _PreparedUtilityLoadProfile,
    _target_prepared_utility_load_profile_duties,
)
from ..targeting.indirect import (
    _build_site_utility_profile,
    _match_utility_gen_and_use_at_same_level,
    _shift_site_process_profiles,
)
from .allocation import AllocationAdapterResult, _build_stream, _fallback_stream
from .context import PlacementPeriodInput, PlacementTargetSnapshot
from .profiles import _calibrate_profile, _finite_tuple, _load_profiles


@dataclass(frozen=True)
class PreparedAggregateComposite:
    """Invariant Total Site process composite for one selected period."""

    temperatures: tuple[float, ...]
    hot_composite: tuple[float, ...]
    cold_composite: tuple[float, ...]


def build_aggregate_composite(
    profiles: Iterable[tuple[str, _PreparedUtilityLoadProfile]],
) -> PreparedAggregateComposite:
    """Build the shifted Total Site process composite from child-zone profiles."""
    net_hot_streams = StreamCollection()
    net_cold_streams = StreamCollection()
    for address, profile in profiles:
        temperatures = profile.pt[ProblemTableLabel.T]
        maximum_temperature = float(np.max(temperatures))
        minimum_temperature = float(np.min(temperatures))
        hot_utility = StreamCollection(
            [
                Stream(
                    name="Cached hot coverage",
                    supply_temperature=maximum_temperature + 1.0,
                    target_temperature=maximum_temperature,
                    heat_flow=profile.hot_utility_target,
                    delta_t_contribution=0.0,
                    is_process_stream=False,
                )
            ]
        )
        cold_utility = StreamCollection(
            [
                Stream(
                    name="Cached cold coverage",
                    supply_temperature=minimum_temperature - 1.0,
                    target_temperature=minimum_temperature,
                    heat_flow=profile.cold_utility_target,
                    delta_t_contribution=0.0,
                    is_process_stream=False,
                )
            ]
        )
        child_hot, child_cold = (
            _create_net_hot_and_cold_stream_collections_for_site_analysis(
                T_vals=temperatures,
                H_vals=profile.pt[ProblemTableLabel.H_NET_A],
                hot_utilities=hot_utility,
                cold_utilities=cold_utility,
                idx=None,
            )
        )
        for key, stream in child_hot.items():
            net_hot_streams.add(
                stream,
                key=f"{address}.{key}",
            )
        for key, stream in child_cold.items():
            net_cold_streams.add(
                stream,
                key=f"{address}.{key}",
            )

    pt = get_process_heat_cascade(
        hot_streams=net_hot_streams,
        cold_streams=net_cold_streams,
        is_shifted=True,
    )
    pt.update(
        **_shift_site_process_profiles(
            T_col=pt[ProblemTableLabel.T],
            H_hot=pt[ProblemTableLabel.H_HOT],
            H_cold=pt[ProblemTableLabel.H_COLD],
        )
    )
    return PreparedAggregateComposite(
        temperatures=_finite_tuple(pt[ProblemTableLabel.T]),
        hot_composite=_finite_tuple(pt[ProblemTableLabel.H_HOT]),
        cold_composite=_finite_tuple(pt[ProblemTableLabel.H_COLD]),
    )


def placement_utility_collections(
    period: PlacementPeriodInput,
    placement: DecodedPlacement,
) -> tuple[StreamCollection, StreamCollection]:
    """Build utility streams for a placement plus the period fallback streams."""
    limits = dict(period.maximum_duties)
    hot = StreamCollection(
        [
            *(
                _build_stream(
                    level,
                    maximum_duty=limits.get(level.template_key.name),
                )
                for level in placement.hot
            ),
            _fallback_stream(period, UtilitySide.HOT),
        ]
    )
    cold = StreamCollection(
        [
            *(
                _build_stream(
                    level,
                    maximum_duty=limits.get(level.template_key.name),
                )
                for level in placement.cold
            ),
            _fallback_stream(period, UtilitySide.COLD),
        ]
    )
    return hot, cold


def allocation_result(
    *,
    hot_utilities,
    cold_utilities,
    period_idx: int | None,
    placement: DecodedPlacement,
    required_hot: float,
    required_cold: float,
    snapshot: PlacementTargetSnapshot,
) -> AllocationAdapterResult:
    """Summarise targeted utility duties and fallbacks for one placement."""

    def duty(utility) -> float:
        return float(get_scalar_value(utility.heat_flow, period_idx=period_idx))

    def temperature(utility, attribute: str) -> float:
        return float(
            get_scalar_value(
                getattr(utility, attribute),
                period_idx=period_idx,
            )
        )

    hot_by_name = {utility.name: utility for utility in hot_utilities}
    cold_by_name = {utility.name: utility for utility in cold_utilities}
    hot_names = {level.template_key.name for level in placement.hot}
    cold_names = {level.template_key.name for level in placement.cold}
    hot_fallbacks = tuple(
        utility
        for utility in hot_utilities
        if utility.name not in hot_names and duty(utility) > 0.0
    )
    cold_fallbacks = tuple(
        utility
        for utility in cold_utilities
        if utility.name not in cold_names and duty(utility) > 0.0
    )

    def fallback_values(utilities, default_name: str):
        if not utilities:
            return default_name, 0.0, None, None
        first = utilities[0]
        return (
            first.name,
            math.fsum(duty(utility) for utility in utilities),
            temperature(first, "supply_temperature"),
            temperature(first, "target_temperature"),
        )

    hot_fallback = fallback_values(hot_fallbacks, "HU")
    cold_fallback = fallback_values(cold_fallbacks, "CU")
    return AllocationAdapterResult(
        hot_duties=tuple(
            duty(hot_by_name[level.template_key.name])
            if level.template_key.name in hot_by_name
            else 0.0
            for level in placement.hot
        ),
        cold_duties=tuple(
            duty(cold_by_name[level.template_key.name])
            if level.template_key.name in cold_by_name
            else 0.0
            for level in placement.cold
        ),
        hot_fallback_name=hot_fallback[0],
        hot_fallback_duty=hot_fallback[1],
        hot_fallback_supply_temperature=hot_fallback[2],
        hot_fallback_target_temperature=hot_fallback[3],
        cold_fallback_name=cold_fallback[0],
        cold_fallback_duty=cold_fallback[1],
        cold_fallback_supply_temperature=cold_fallback[2],
        cold_fallback_target_temperature=cold_fallback[3],
        required_hot_duty=required_hot,
        required_cold_duty=required_cold,
        target_snapshot=snapshot,
    )


def allocate_from_aggregate_profiles(
    period: PlacementPeriodInput,
    placement: DecodedPlacement,
    profiles: Iterable[_PreparedUtilityLoadProfile],
    composite: PreparedAggregateComposite,
) -> AllocationAdapterResult:
    """Allocate a placement against cached child-zone utility load profiles."""
    hot_base, cold_base = placement_utility_collections(period, placement)
    period_idx = None
    aggregate_hot = hot_base.copy(deep=True).set_common_stream_attribute(
        "heat_flow",
        0.0,
        idx=period_idx,
    )
    aggregate_cold = cold_base.copy(deep=True).set_common_stream_attribute(
        "heat_flow",
        0.0,
        idx=period_idx,
    )
    hot_totals = [0.0] * len(aggregate_hot)
    cold_totals = [0.0] * len(aggregate_cold)

    def duty(utility) -> float:
        return float(get_scalar_value(utility.heat_flow, period_idx=period_idx))

    try:
        for profile in profiles:
            targeted_hot, targeted_cold = _target_prepared_utility_load_profile_duties(
                profile,
                hot_utilities=hot_base,
                cold_utilities=cold_base,
                period_idx=period_idx,
            )
            for totals, targeted in (
                (hot_totals, targeted_hot),
                (cold_totals, targeted_cold),
            ):
                for index, utility_duty in enumerate(targeted):
                    totals[index] += utility_duty

        for aggregate, totals in (
            (aggregate_hot, hot_totals),
            (aggregate_cold, cold_totals),
        ):
            for utility, total in zip(aggregate, totals, strict=True):
                if total > 0.0:
                    utility.set_value_attr_at_idx(
                        "heat_flow",
                        total,
                        idx=period_idx,
                    )

        profile = _build_site_utility_profile(
            hot_utilities=aggregate_hot,
            cold_utilities=aggregate_cold,
            is_shifted=False,
            idx=period_idx,
        )
        net_utility = np.asarray(
            profile["updates"][ProblemTableLabel.H_NET_UT],
            dtype=float,
        )
        sugcc = ProblemTable(
            {
                ProblemTableLabel.T: np.asarray(profile["T_col"], dtype=float),
                ProblemTableLabel.H_NET_UT: net_utility,
            }
        )
        temperatures = np.asarray(sugcc[ProblemTableLabel.T], dtype=float)
        net_utility = np.asarray(
            sugcc[ProblemTableLabel.H_NET_UT],
            dtype=float,
        )
        required_hot = float(net_utility[0])
        required_cold = float(net_utility[-1])
        hot_profile, cold_profile = _load_profiles(
            sugcc,
            net_label=ProblemTableLabel.H_NET_UT,
        )
        hot_profile = _calibrate_profile(
            hot_profile,
            residual_duty=required_hot,
        )
        cold_profile = _calibrate_profile(
            cold_profile,
            residual_duty=required_cold,
        )
        hot_pinch, cold_pinch, *_ = sugcc.pinch_idx(ProblemTableLabel.H_NET_UT)
    except (ArithmeticError, ValueError) as exc:
        return AllocationAdapterResult(
            hot_duties=(0.0,) * len(placement.hot),
            cold_duties=(0.0,) * len(placement.cold),
            diagnostics=(
                CandidateDiagnostic(
                    code="targeting_infeasible",
                    constraint="utility_allocation",
                    message=str(exc) or "Utility allocation is infeasible.",
                    period_id=period.period_id,
                ),
            ),
        )

    matched_hot, matched_cold = _match_utility_gen_and_use_at_same_level(
        hot_utilities=aggregate_hot,
        cold_utilities=aggregate_cold,
        period_idx=period_idx,
    )
    snapshot = PlacementTargetSnapshot(
        shifted_temperatures=tuple(float(value) for value in temperatures),
        real_temperatures=composite.temperatures,
        hot_load_profile=hot_profile,
        cold_load_profile=cold_profile,
        real_hot_composite=composite.hot_composite,
        real_cold_composite=composite.cold_composite,
        hot_pinch_index=int(hot_pinch),
        cold_pinch_index=int(cold_pinch),
        entropy_slices=period.snapshot.entropy_slices,
    )
    return allocation_result(
        hot_utilities=matched_hot,
        cold_utilities=matched_cold,
        period_idx=period_idx,
        placement=placement,
        required_hot=required_hot,
        required_cold=required_cold,
        snapshot=snapshot,
    )


__all__ = [
    "PreparedAggregateComposite",
    "allocate_from_aggregate_profiles",
    "allocation_result",
    "build_aggregate_composite",
    "placement_utility_collections",
]
