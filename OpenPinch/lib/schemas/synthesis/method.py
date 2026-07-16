"""Method-level HEN synthesis schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ....classes.heat_exchanger_network import HeatExchangerNetwork
from .common import (
    HeatExchangerNetworkSynthesisManifest,
    SynthesisMethod,
    SynthesisTaskStatus,
    _validate_non_negative_finite,
    _validate_optional_identity,
    _validate_positive_finite,
    _validate_run_id,
)
from .topology import HeatExchangerNetworkTopologyRestriction


class HeatExchangerNetworkSynthesisMethodInput(BaseModel):
    """Validated input for one PDM, TDM, or evolution method run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: str | None = None
    run_id: str
    method: SynthesisMethod
    approach_temperature: float
    derivative_threshold: float | None = None
    stage_count: int | None = None
    problem_id: str | None = None
    workspace_variant: str | None = None
    period_id: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    seed_network: HeatExchangerNetwork | None = None
    seed_network_index: int | None = None
    parent_task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    topology_restrictions: tuple["HeatExchangerNetworkTopologyRestriction", ...] = (
        Field(default_factory=tuple)
    )

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        return _validate_run_id(value)

    @field_validator(
        "task_id",
        "problem_id",
        "workspace_variant",
        "period_id",
        "parent_task_id",
    )
    @classmethod
    def _validate_optional_identity(cls, value: str | None) -> str | None:
        return _validate_optional_identity(value)

    @field_validator("approach_temperature", "derivative_threshold")
    @classmethod
    def _validate_positive_float(cls, value: float | None) -> float | None:
        if value is None:
            return value
        return _validate_positive_finite(value)

    @field_validator("stage_count")
    @classmethod
    def _validate_stage_count(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stage_count must be a positive integer when supplied")
        return value

    @field_validator("seed_network_index")
    @classmethod
    def _validate_seed_network_index(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("seed_network_index must be non-negative when supplied")
        return value

    @model_validator(mode="after")
    def _ensure_task_id(self) -> Self:
        if self.task_id is None:
            self.task_id = self.generate_task_id()
        return self

    def generate_task_id(self) -> str:
        """Return the deterministic identifier for this task definition."""
        task_key_data = {
            "approach_temperature": self.approach_temperature,
            "derivative_threshold": self.derivative_threshold,
            "method": self.method,
            "parent_task_id": self.parent_task_id,
            "problem_id": self.problem_id,
            "run_id": self.run_id,
            "stage_count": self.stage_count,
            "settings": self.settings,
            "period_id": self.period_id,
            "topology_restrictions": [
                restriction.model_dump(mode="json")
                for restriction in self.topology_restrictions
            ],
            "workspace_variant": self.workspace_variant,
        }
        if self.seed_network_index is not None:
            task_key_data["seed_network_index"] = self.seed_network_index
        encoded = json.dumps(
            task_key_data,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"hens-task-{digest}"


class HeatExchangerNetworkSynthesisMethodOutput(BaseModel):
    """Validated output for one PDM, TDM, or evolution method run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    method: SynthesisMethod | None = None
    task: HeatExchangerNetworkSynthesisMethodInput | None = None
    status: SynthesisTaskStatus
    network: HeatExchangerNetwork | None = None
    accepted_networks: tuple[HeatExchangerNetwork, ...] = Field(default_factory=tuple)
    ranked_networks: tuple[HeatExchangerNetwork, ...] = Field(default_factory=tuple)
    task_manifest: HeatExchangerNetworkSynthesisManifest | None = None
    objective_value: float | None = None
    solver_status: str | None = None
    error: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    diagnostic_references: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("objective_value")
    @classmethod
    def _validate_objective_value(cls, value: float | None) -> float | None:
        if value is None:
            return value
        return _validate_non_negative_finite(value)

    @field_validator("solver_status", "error")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        return _validate_optional_identity(value)

    @field_validator("diagnostic_references")
    @classmethod
    def _validate_diagnostic_references(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(_validate_optional_identity(item) for item in value)

    @model_validator(mode="after")
    def _fill_method_from_task(self) -> Self:
        if self.method is None and self.task is not None:
            self.method = self.task.method
        return self


HeatExchangerNetworkSynthesisMethodInput.model_rebuild()
HeatExchangerNetworkSynthesisMethodOutput.model_rebuild()
