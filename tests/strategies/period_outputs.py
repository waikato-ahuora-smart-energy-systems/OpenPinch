"""Domain strategies for aligned multi-period targeting outputs."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from OpenPinch.contracts.output import TargetOutput
from OpenPinch.contracts.reporting import HeatUtility, PinchTemp, TargetResults
from OpenPinch.domain.value import Value


@dataclass(frozen=True)
class AlignedPeriodOutputs:
    """Two valid period outputs plus the values used to construct them."""

    outputs: tuple[TargetOutput, TargetOutput]
    weights: tuple[float, float]
    primary_hot_utility: tuple[float, float]
    cold_pinch_present: tuple[bool, bool]


def _target(
    *,
    scope: str,
    period_id: str,
    hot_utility: float,
    cold_pinch_present: bool,
) -> TargetResults:
    cold_pinch = Value(90.0 + hot_utility / 100.0, "degC")
    return TargetResults(
        scope=scope,
        zone_type="Site" if scope == "Site" else "Process Zone",
        integration_type="Process",
        target_method="Heat Exchange",
        period_id=period_id,
        Qh=Value(hot_utility, "kW"),
        Qc=Value(hot_utility / 2.0, "kW"),
        Qr=Value(hot_utility * 2.0, "kW"),
        pinch_temp=PinchTemp(
            cold_temp=cold_pinch if cold_pinch_present else None,
            hot_temp=Value(120.0 + hot_utility / 100.0, "degC"),
        ),
        hot_utilities=[HeatUtility(name="Steam", heat_flow=Value(hot_utility, "kW"))],
        cold_utilities=[
            HeatUtility(
                name="Cooling Water",
                heat_flow=Value(hot_utility / 2.0, "kW"),
            )
        ],
        area=Value(hot_utility * 1.5, "m^2"),
        num_units=hot_utility / 100.0,
    )


@st.composite
def aligned_period_outputs(draw) -> AlignedPeriodOutputs:
    """Generate aligned base/peak outputs with optional pinch diagnostics."""
    finite_positive = st.integers(min_value=1, max_value=1_000_000).map(
        lambda value: value / 10.0
    )
    base_hot = draw(finite_positive)
    peak_hot = draw(finite_positive)
    base_weight = draw(finite_positive)
    peak_weight = draw(finite_positive)
    cold_pinch_present = (draw(st.booleans()), draw(st.booleans()))

    outputs = []
    for period_id, hot_utility, has_cold_pinch in zip(
        ("base", "peak"),
        (base_hot, peak_hot),
        cold_pinch_present,
        strict=True,
    ):
        outputs.append(
            TargetOutput(
                name="Site",
                period_id=period_id,
                targets=[
                    _target(
                        scope="Site",
                        period_id=period_id,
                        hot_utility=hot_utility,
                        cold_pinch_present=has_cold_pinch,
                    ),
                    _target(
                        scope="Site/Process",
                        period_id=period_id,
                        hot_utility=hot_utility / 2.0,
                        cold_pinch_present=has_cold_pinch,
                    ),
                ],
            )
        )

    return AlignedPeriodOutputs(
        outputs=(outputs[0], outputs[1]),
        weights=(base_weight, peak_weight),
        primary_hot_utility=(base_hot, peak_hot),
        cold_pinch_present=cold_pinch_present,
    )


__all__ = ["AlignedPeriodOutputs", "aligned_period_outputs"]
