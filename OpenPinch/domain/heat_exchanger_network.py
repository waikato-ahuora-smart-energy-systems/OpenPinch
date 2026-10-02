"""OpenPinch-native heat exchanger network result model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._heat_exchanger import rules as _rules
from ._heat_exchanger.period_state import (
    HeatExchangerPeriodState as _HeatExchangerPeriodState,
)
from .enums import HeatExchangerKind, HeatExchangerNetworkLabel, StreamID
from .heat_exchanger import HeatExchanger

UtilityPlacement = str | tuple[str, str] | tuple[str, str, int]


def _utility_placement(
    entry: UtilityPlacement,
    default_utility: str | None,
    utility_argument: str,
    list_name: str,
) -> tuple[str, str, int | None]:
    """Normalise a heater/cooler entry to ``(stream, utility, stage)``."""

    if isinstance(entry, str):
        if default_utility is None:
            raise ValueError(
                f"{utility_argument} is required when {list_name} name only a stream"
            )
        return entry, str(default_utility), None
    values = tuple(entry)
    if len(values) == 2:
        return str(values[0]), str(values[1]), None
    if len(values) == 3:
        return str(values[0]), str(values[1]), int(values[2])
    raise ValueError(
        f"{list_name} entries must be a stream name, (stream, utility) or "
        "(stream, utility, stage)"
    )


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
        return _rules.optional_identity(value, _rules.NETWORK_IDENTITY_MESSAGE)

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
        return _rules.optional_non_negative_finite(
            value, _rules.NETWORK_NUMERIC_MESSAGE
        )

    @field_validator("summary_metrics")
    @classmethod
    def _validate_summary_metrics(
        cls,
        value: dict[str, float | int | str | bool | None],
    ) -> dict[str, float | int | str | bool | None]:
        _rules.check_summary_metrics(value)
        return value

    @model_validator(mode="after")
    def _validate_period_state_alignment(self) -> Self:
        _rules.check_period_alignment(
            exchanger.period_ids for exchanger in self.exchangers
        )
        return self

    @classmethod
    def from_structure(
        cls,
        *,
        recovery: Iterable[tuple[str, str, int]],
        heaters: Iterable[UtilityPlacement] = (),
        coolers: Iterable[UtilityPlacement] = (),
        hot_utility: str | None = None,
        cold_utility: str | None = None,
        stage_count: int | None = None,
        period_id: str = "0",
    ) -> Self:
        """Build a zero-duty network structure from stream names.

        ``recovery`` lists ``(hot_stream, cold_stream, stage)`` matches with
        one-based stages. ``heaters`` lists cold streams and ``coolers`` hot
        streams; each entry is a stream name (served by ``hot_utility`` or
        ``cold_utility`` at the stream end), ``(stream, utility)``, or
        ``(stream, utility, stage)`` for an exchanger just after the stream
        leaves that stage. Exchanger ids follow the solver result convention:
        ``recovery:<hot>-><cold>:S<stage>``, ``hot-utility:<utility>-><cold>``
        and ``cold-utility:<hot>-><utility>``, with ``:S<stage>`` appended to a
        staged utility exchanger.
        """

        heater_entries = [
            _utility_placement(entry, hot_utility, "hot_utility", "heaters")
            for entry in heaters
        ]
        cooler_entries = [
            _utility_placement(entry, cold_utility, "cold_utility", "coolers")
            for entry in coolers
        ]

        def state() -> tuple[_HeatExchangerPeriodState, ...]:
            return (
                _HeatExchangerPeriodState(
                    period_id=period_id, period_idx=0, duty=0.0, active=False
                ),
            )

        def suffix(stage: int | None) -> str:
            return "" if stage is None else f":S{stage}"

        exchangers = [
            HeatExchanger(
                exchanger_id=f"recovery:{hot}->{cold}:S{int(stage)}",
                kind=HeatExchangerKind.RECOVERY,
                source_stream=hot,
                sink_stream=cold,
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Process,
                stage=int(stage),
                period_states=state(),
            )
            for hot, cold, stage in recovery
        ]
        exchangers.extend(
            HeatExchanger(
                exchanger_id=f"hot-utility:{utility}->{cold}{suffix(stage)}",
                kind=HeatExchangerKind.HOT_UTILITY,
                source_stream=utility,
                sink_stream=cold,
                source_stream_role=StreamID.Utility,
                sink_stream_role=StreamID.Process,
                stage=stage,
                period_states=state(),
            )
            for cold, utility, stage in heater_entries
        )
        exchangers.extend(
            HeatExchanger(
                exchanger_id=f"cold-utility:{hot}->{utility}{suffix(stage)}",
                kind=HeatExchangerKind.COLD_UTILITY,
                source_stream=hot,
                sink_stream=utility,
                source_stream_role=StreamID.Process,
                sink_stream_role=StreamID.Utility,
                stage=stage,
                period_states=state(),
            )
            for hot, utility, stage in cooler_entries
        )
        return cls(exchangers=tuple(exchangers), stage_count=stage_count)

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
            # Duty belongs to the period; area and cost belong to the installed
            # exchanger, which counts if it runs in any period.
            if active_only and attribute == "duty" and not state.active:
                continue
            if (
                active_only
                and attribute != "duty"
                and not any(item.active for item in exchanger.period_states)
            ):
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
