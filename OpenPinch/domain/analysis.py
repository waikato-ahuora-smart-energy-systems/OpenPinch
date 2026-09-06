"""Immutable provenance shared by otherwise independent analysis families."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalysisProvenance(BaseModel):
    """Identity and effective inputs of a completed analysis, without live state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner_id: str
    method_id: str
    zone_address: str
    period_ids: tuple[str, ...]
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    effective_settings_json: str = "{}"
    prerequisite_ids: tuple[str, ...] = ()

    @field_validator("effective_settings_json")
    @classmethod
    def _settings_object(cls, value):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("effective settings must be a JSON object")
        return json.dumps(
            parsed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def effective_settings(self):
        """Read-only detached values; nested changes cannot modify provenance."""
        return MappingProxyType(json.loads(self.effective_settings_json))

    @property
    def identity(self) -> str:
        """Stable identity within an owner, independent of execution scheduling."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()
