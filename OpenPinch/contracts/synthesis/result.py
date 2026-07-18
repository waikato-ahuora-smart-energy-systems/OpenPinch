"""Result-level HEN synthesis schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...domain.enums import HeatExchangerNetworkDesignMethod
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .common import (
    HeatExchangerNetworkSynthesisManifest,
    _validate_non_negative_finite,
    _validate_optional_identity,
    _validate_run_id,
)
from .task import HeatExchangerNetworkSynthesisTaskOutcome


class HeatExchangerNetworkSynthesisResult(BaseModel):
    """Problem-owned heat exchanger network synthesis result data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    network: HeatExchangerNetwork
    run_id: str
    task_id: str | None = None
    problem_id: str | None = None
    workspace_variant: str | None = None
    period_id: str | None = None
    solver_name: str | None = None
    solver_status: str | None = None
    design_method: HeatExchangerNetworkDesignMethod | None = None
    method: HeatExchangerNetworkDesignMethod | None = None
    stage_count: int | None = None
    objective_values: dict[str, float] = Field(default_factory=dict)
    ranked_networks: tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...] = Field(
        default_factory=tuple,
    )
    manifest: HeatExchangerNetworkSynthesisManifest | None = None
    diagnostic_references: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        return _validate_run_id(value)

    @field_validator(
        "task_id",
        "problem_id",
        "workspace_variant",
        "period_id",
        "solver_name",
        "solver_status",
    )
    @classmethod
    def _validate_optional_identity(cls, value: str | None) -> str | None:
        return _validate_optional_identity(value)

    @field_validator("stage_count")
    @classmethod
    def _validate_stage_count(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stage_count must be a positive integer when supplied")
        return value

    @field_validator("objective_values")
    @classmethod
    def _validate_objective_values(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        for objective_name, objective_value in value.items():
            _validate_optional_identity(objective_name)
            _validate_non_negative_finite(objective_value)
        return {str(name): float(metric) for name, metric in value.items()}

    @field_validator("diagnostic_references")
    @classmethod
    def _validate_diagnostic_references(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(_validate_optional_identity(item) for item in value)


HeatExchangerNetworkSynthesisResult.model_rebuild()
