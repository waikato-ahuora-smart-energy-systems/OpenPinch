"""OpenPinch-native heat exchanger design records."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._heat_exchanger import area as _area
from ._heat_exchanger import rules as _rules
from ._heat_exchanger.area import HeatExchangerAreaSlice as _HeatExchangerAreaSlice
from ._heat_exchanger.period_state import (
    HeatExchangerPeriodState as _HeatExchangerPeriodState,
)
from .enums import HeatExchangerKind, StreamID


class HeatExchanger(BaseModel):
    """One labelled heat-transfer link in a heat exchanger network."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    exchanger_id: str | None = None
    kind: HeatExchangerKind
    source_stream: str
    sink_stream: str
    source_stream_role: StreamID
    sink_stream_role: StreamID
    stage: int | None = None
    period_states: tuple[_HeatExchangerPeriodState, ...] = Field(min_length=1)
    area: float | None = None
    match_allowed: bool = True
    capital_cost: float | None = None
    segment_area_contributions: tuple[_HeatExchangerAreaSlice, ...] = Field(
        default_factory=tuple
    )
    solver_metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )

    @field_validator("exchanger_id", "source_stream", "sink_stream")
    @classmethod
    def _validate_identity(cls, value: str | None) -> str | None:
        return _rules.optional_identity(value, _rules.EXCHANGER_IDENTITY_MESSAGE)

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stage must be a positive integer when supplied")
        return value

    @field_validator(
        "area",
        "capital_cost",
    )
    @classmethod
    def _validate_non_negative_finite(cls, value: float | None) -> float | None:
        return _rules.optional_non_negative_finite(
            value, _rules.EXCHANGER_NUMERIC_MESSAGE
        )

    @model_validator(mode="after")
    def _validate_period_states(self) -> Self:
        _rules.check_period_states(self.period_states)
        return self

    @model_validator(mode="after")
    def _validate_direction_semantics(self) -> Self:
        _rules.check_direction_semantics(
            kind=self.kind,
            source_stream=self.source_stream,
            sink_stream=self.sink_stream,
            source_stream_role=self.source_stream_role,
            sink_stream_role=self.sink_stream_role,
            stage=self.stage,
        )
        return self

    @model_validator(mode="after")
    def _validate_segment_design_area(self) -> Self:
        design_area = _area.validate_segment_design_area(
            self.area,
            self.segment_area_contributions,
        )
        if self.area is None and design_area is not None:
            object.__setattr__(self, "area", design_area)
        return self

    @property
    def has_segment_area_contributions(self) -> bool:
        """Return whether exact local segment-area slices are available."""

        return bool(self.segment_area_contributions)

    @property
    def segment_duty_by_period(self) -> dict[str, float]:
        """Return local slice duty totals grouped by operating period."""
        return _area.segment_duty_by_period(self.segment_area_contributions)

    @property
    def segment_area_by_period(self) -> dict[str, float]:
        """Return local slice area totals grouped by operating period."""
        return _area.segment_area_by_period(self.segment_area_contributions)

    @property
    def segment_design_area(self) -> float | None:
        """Return the maximum period-total slice area when slices are available."""
        return _area.segment_design_area(self.segment_area_contributions)

    @property
    def period_ids(self) -> tuple[str, ...]:
        """Return ordered operating-period identities for this exchanger."""

        return tuple(state.period_id for state in self.period_states)

    def state(self, period_id: str | None = None) -> _HeatExchangerPeriodState:
        """Return one period state, requiring identity for multiperiod results."""

        if period_id is None:
            if len(self.period_states) != 1:
                raise ValueError(
                    "period_id is required when an exchanger has multiple period states"
                )
            return self.period_states[0]
        for state in self.period_states:
            if state.period_id == period_id:
                return state
        raise ValueError(
            f"unknown period_id {period_id!r}; expected one of {self.period_ids!r}"
        )

    def involves_stream(self, stream_id: str) -> bool:
        """Return whether this exchanger uses ``stream_id`` as source or sink."""
        return self.source_stream == stream_id or self.sink_stream == stream_id

    def matches(
        self,
        *,
        source_stream: str,
        sink_stream: str,
        stage: int | None = None,
    ) -> bool:
        """Return whether this exchanger matches a labelled stream-stage link."""
        if self.source_stream != source_stream or self.sink_stream != sink_stream:
            return False
        return stage is None or self.stage == stage


__all__ = [
    "HeatExchanger",
]
