"""OpenPinch-native heat exchanger network result model."""

from __future__ import annotations

import math
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import HeatExchangerKind, HeatExchangerNetworkLabel
from .heat_exchanger import HeatExchanger

_DUTY_LABEL_KINDS = {
    HeatExchangerNetworkLabel.RECOVERY_DUTY: HeatExchangerKind.RECOVERY,
    HeatExchangerNetworkLabel.HOT_UTILITY_DUTY: HeatExchangerKind.HOT_UTILITY,
    HeatExchangerNetworkLabel.COLD_UTILITY_DUTY: HeatExchangerKind.COLD_UTILITY,
}

_AREA_LABEL_KINDS = {
    HeatExchangerNetworkLabel.RECOVERY_AREA: HeatExchangerKind.RECOVERY,
    HeatExchangerNetworkLabel.HOT_UTILITY_AREA: HeatExchangerKind.HOT_UTILITY,
    HeatExchangerNetworkLabel.COLD_UTILITY_AREA: HeatExchangerKind.COLD_UTILITY,
}


class HeatExchangerNetwork(BaseModel):
    """Ordered heat exchanger network result collection."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    exchangers: tuple[HeatExchanger, ...] = Field(default_factory=tuple)
    run_id: str | None = None
    task_id: str | None = None
    period_id: str | None = None
    method: str | None = None
    stage_count: int | None = None
    objective_value: float | None = None
    total_annual_cost: float | None = None
    utility_cost: float | None = None
    capital_cost: float | None = None
    summary_metrics: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict,
    )
    solver_axis_metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )

    @field_validator("run_id", "task_id", "period_id", "method")
    @classmethod
    def _validate_optional_identity(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("network metadata identities must be non-empty strings")
        return value.strip()

    @field_validator("stage_count")
    @classmethod
    def _validate_stage_count(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stage_count must be a positive integer when supplied")
        return value

    @field_validator(
        "objective_value",
        "total_annual_cost",
        "utility_cost",
        "capital_cost",
    )
    @classmethod
    def _validate_optional_non_negative_finite(
        cls,
        value: float | None,
    ) -> float | None:
        if value is None:
            return value
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("network numeric values must be finite and non-negative")
        return float(value)

    @field_validator("summary_metrics")
    @classmethod
    def _validate_summary_metrics(
        cls,
        value: dict[str, float | int | str | bool | None],
    ) -> dict[str, float | int | str | bool | None]:
        for metric_name, metric_value in value.items():
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ValueError("summary metric names must be non-empty strings")
            if isinstance(metric_value, float) and not math.isfinite(metric_value):
                raise ValueError("summary metric values must be finite")
        return value

    @model_validator(mode="after")
    def _validate_period_state_alignment(self) -> Self:
        if not self.exchangers:
            return self
        ordered = self.exchangers[0].period_ids
        if any(exchanger.period_ids != ordered for exchanger in self.exchangers[1:]):
            raise ValueError(
                "all exchangers in a network must use the same ordered period_ids"
            )
        return self

    @property
    def period_ids(self) -> tuple[str, ...]:
        """Return ordered period identities represented by exchanger states."""

        if not self.exchangers:
            return (self.period_id,) if self.period_id is not None else ()
        return self.exchangers[0].period_ids

    def resolve_period_id(self, period_id: str | None = None) -> str | None:
        """Resolve an optional period identity without ambiguous multiperiod access."""

        period_ids = self.period_ids
        if period_id is not None:
            if period_ids and period_id not in period_ids:
                raise ValueError(
                    f"unknown period_id {period_id!r}; expected one of {period_ids!r}"
                )
            return period_id
        if len(period_ids) > 1:
            raise ValueError(
                "period_id is required when a network has multiple period states"
            )
        return period_ids[0] if period_ids else None

    def exchangers_involving_stream(
        self,
        stream_id: str,
        *,
        active_only: bool = False,
        period_id: str | None = None,
    ) -> tuple[HeatExchanger, ...]:
        """Return all exchangers that use ``stream_id`` as source or sink."""
        resolved_period_id = self.resolve_period_id(period_id) if active_only else None
        return tuple(
            exchanger
            for exchanger in self.exchangers
            if exchanger.involves_stream(stream_id)
            and (exchanger.state(resolved_period_id).active if active_only else True)
        )

    def exchanger_between(
        self,
        *,
        source_stream: str,
        sink_stream: str,
        stage: int | None = None,
        kind: HeatExchangerKind | str | None = None,
    ) -> HeatExchanger | None:
        """Return the unique exchanger for a labelled source/sink/stage link."""
        expected_kind = _coerce_kind(kind)
        matches = [
            exchanger
            for exchanger in self.exchangers
            if exchanger.matches(
                source_stream=source_stream,
                sink_stream=sink_stream,
                stage=stage,
            )
            and (expected_kind is None or exchanger.kind is expected_kind)
        ]
        if len(matches) > 1:
            raise ValueError(
                "multiple exchangers match the supplied source, sink, and stage"
            )
        return matches[0] if matches else None

    def total_duty(
        self,
        *,
        kind: HeatExchangerKind | str | None = None,
        stream: str | None = None,
        stage: int | None = None,
        active_only: bool = True,
        period_id: str | None = None,
    ) -> float:
        """Return duty total filtered by kind, stream identity, and stage."""
        return self._sum_numeric(
            "duty",
            kind=kind,
            stream=stream,
            stage=stage,
            active_only=active_only,
            period_id=period_id,
        )

    def total_area(
        self,
        *,
        kind: HeatExchangerKind | str | None = None,
        stream: str | None = None,
        stage: int | None = None,
        active_only: bool = True,
        period_id: str | None = None,
    ) -> float:
        """Return area total filtered by kind, stream identity, and stage."""
        return self._sum_numeric(
            "area",
            kind=kind,
            stream=stream,
            stage=stage,
            active_only=active_only,
            period_id=period_id,
        )

    def total(
        self,
        label: HeatExchangerNetworkLabel | str,
        *,
        kind: HeatExchangerKind | str | None = None,
        stream: str | None = None,
        stage: int | None = None,
        active_only: bool = True,
        period_id: str | None = None,
    ) -> float:
        """Return a numeric total for a supported heat exchanger network label."""
        normalised_label = HeatExchangerNetworkLabel(label)
        if normalised_label in _DUTY_LABEL_KINDS:
            label_kind = _DUTY_LABEL_KINDS[normalised_label]
            return self.total_duty(
                kind=_resolve_label_kind(kind, label_kind),
                stream=stream,
                stage=stage,
                active_only=active_only,
                period_id=period_id,
            )
        if normalised_label in _AREA_LABEL_KINDS:
            label_kind = _AREA_LABEL_KINDS[normalised_label]
            return self.total_area(
                kind=_resolve_label_kind(kind, label_kind),
                stream=stream,
                stage=stage,
                active_only=active_only,
                period_id=period_id,
            )
        raise ValueError(f"{normalised_label.value!r} is not a numeric total label")

    def labelled_value(
        self,
        label: HeatExchangerNetworkLabel | str,
        *,
        source_stream: str,
        sink_stream: str,
        stage: int | None = None,
        kind: HeatExchangerKind | str | None = None,
        period_id: str | None = None,
    ) -> float | bool | None:
        """Return a labelled value from one source/sink/stage exchanger link."""
        normalised_label = HeatExchangerNetworkLabel(label)
        expected_kind = _coerce_kind(kind)

        if normalised_label in _DUTY_LABEL_KINDS:
            expected_kind = _resolve_label_kind(
                expected_kind,
                _DUTY_LABEL_KINDS[normalised_label],
            )
        elif normalised_label in _AREA_LABEL_KINDS:
            expected_kind = _resolve_label_kind(
                expected_kind,
                _AREA_LABEL_KINDS[normalised_label],
            )

        exchanger = self.exchanger_between(
            source_stream=source_stream,
            sink_stream=sink_stream,
            stage=stage,
            kind=expected_kind,
        )
        if exchanger is None:
            return None

        if normalised_label in _DUTY_LABEL_KINDS:
            return exchanger.state(self.resolve_period_id(period_id)).duty
        if normalised_label in _AREA_LABEL_KINDS:
            return exchanger.area
        if (
            normalised_label
            is HeatExchangerNetworkLabel.HOT_RECOVERY_OUTLET_TEMPERATURE
        ):
            _require_recovery_label(exchanger, normalised_label)
            return exchanger.state(
                self.resolve_period_id(period_id)
            ).source_outlet_temperature
        if (
            normalised_label
            is HeatExchangerNetworkLabel.COLD_RECOVERY_OUTLET_TEMPERATURE
        ):
            _require_recovery_label(exchanger, normalised_label)
            return exchanger.state(
                self.resolve_period_id(period_id)
            ).sink_outlet_temperature
        if normalised_label is HeatExchangerNetworkLabel.MATCH_ACTIVE:
            return exchanger.state(self.resolve_period_id(period_id)).active
        if normalised_label is HeatExchangerNetworkLabel.MATCH_ALLOWED:
            return exchanger.match_allowed
        raise ValueError(  # pragma: no cover - all current enum labels are handled.
            f"unsupported heat exchanger network label: {label!r}"
        )

    def _sum_numeric(
        self,
        attribute: str,
        *,
        kind: HeatExchangerKind | str | None,
        stream: str | None,
        stage: int | None,
        active_only: bool,
        period_id: str | None,
    ) -> float:
        expected_kind = _coerce_kind(kind)
        resolved_period_id = self.resolve_period_id(period_id)
        total = 0.0
        for exchanger in self.exchangers:
            state = exchanger.state(resolved_period_id)
            if active_only and not state.active:
                continue
            if expected_kind is not None and exchanger.kind is not expected_kind:
                continue
            if stream is not None and not exchanger.involves_stream(stream):
                continue
            if stage is not None and exchanger.stage != stage:
                continue
            value = state.duty if attribute == "duty" else getattr(exchanger, attribute)
            if value is not None:
                total += float(value)
        return total


def _coerce_kind(kind: HeatExchangerKind | str | None) -> HeatExchangerKind | None:
    if kind is None:
        return None
    return kind if isinstance(kind, HeatExchangerKind) else HeatExchangerKind(kind)


def _resolve_label_kind(
    supplied_kind: HeatExchangerKind | str | None,
    label_kind: HeatExchangerKind,
) -> HeatExchangerKind:
    normalised_kind = _coerce_kind(supplied_kind)
    if normalised_kind is not None and normalised_kind is not label_kind:
        raise ValueError(
            f"{label_kind.value} label cannot be used with {normalised_kind.value}"
        )
    return label_kind


def _require_recovery_label(
    exchanger: HeatExchanger,
    label: HeatExchangerNetworkLabel,
) -> None:
    if exchanger.kind is not HeatExchangerKind.RECOVERY:
        raise ValueError(f"{label.value} is only valid for recovery exchangers")


__all__ = ["HeatExchangerNetwork"]
