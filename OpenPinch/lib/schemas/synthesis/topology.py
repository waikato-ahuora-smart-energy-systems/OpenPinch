"""HEN synthesis topology restriction schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from .common import _validate_non_negative_finite, _validate_optional_identity


class HeatExchangerNetworkTopologyRestriction(BaseModel):
    """OpenPinch stream-link topology inherited by downstream synthesis tasks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_stream: str
    sink_stream: str
    stage: int
    duty: float

    @field_validator("source_stream", "sink_stream")
    @classmethod
    def _validate_stream_identity(cls, value: str) -> str:
        validated = _validate_optional_identity(value)
        if validated is None:
            raise ValueError("stream identities must be non-empty strings")
        return validated

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("stage must be a positive integer")
        return int(value)

    @field_validator("duty")
    @classmethod
    def _validate_duty(cls, value: float) -> float:
        return _validate_non_negative_finite(value)


HeatExchangerNetworkTopologyRestriction.model_rebuild()
