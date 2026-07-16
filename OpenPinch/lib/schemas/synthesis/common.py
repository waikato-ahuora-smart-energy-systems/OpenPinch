"""Shared HEN synthesis schema types, records, and validators."""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...enums import HeatExchangerNetworkDesignMethod

SynthesisMethod = HeatExchangerNetworkDesignMethod
SynthesisDesignMethod = HeatExchangerNetworkDesignMethod
SynthesisTaskStatus = Literal["pending", "success", "failed", "skipped"]
SynthesisOutputFormat = Literal["json", "csv", "xlsx"]

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_run_id(value: str) -> str:
    value = _validate_optional_identity(value)
    if value is None or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, underscores, hyphens, or periods"
        )
    return value


def _validate_optional_identity(value: str | None) -> str | None:
    if value is None:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("identity fields must be non-empty strings")
    return value.strip()


def _validate_positive_finite(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("value must be finite and positive")
    return float(value)


def _validate_non_negative_finite(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("value must be finite and non-negative")
    return float(value)


def _validate_positive_grid(value: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        raise ValueError("grid values must contain at least one entry")
    return tuple(_validate_positive_finite(item) for item in value)


class HeatExchangerNetworkSynthesisExportRecord(BaseModel):
    """One optional export view generated from an in-memory design result."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str
    format: SynthesisOutputFormat
    path: str
    record_id: str | None = None
    content_type: str | None = None

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        return _validate_run_id(value)

    @field_validator("path", "record_id", "content_type")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        return _validate_optional_identity(value)


class HeatExchangerNetworkSynthesisManifest(BaseModel):
    """Run manifest for exports and diagnostic records."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str
    approach_temperatures: tuple[float, ...]
    derivative_thresholds: tuple[float, ...]
    stage_selection: tuple[int, ...]
    method_sequence: tuple[SynthesisMethod, ...] = (
        HeatExchangerNetworkDesignMethod.PinchDesign,
        HeatExchangerNetworkDesignMethod.ThermalDerivative,
        HeatExchangerNetworkDesignMethod.NetworkEvolution,
    )
    export_formats: tuple[SynthesisOutputFormat, ...] = Field(default_factory=tuple)
    solve_tolerance: float = 1e-3
    best_solutions_to_save: int = 1
    synthesis_quality_tier: int = 1
    pdm_stage_pair_limit: int | None = None
    tdm_parent_limit: int | None = None
    stage_packing: str = "auto"
    evm_n_ad_branches: int = 1
    evm_n_rm_branches: int = 1
    task_ids: tuple[str, ...] = Field(default_factory=tuple)
    problem_id: str | None = None
    workspace_variant: str | None = None
    period_id: str | None = None
    design_method: SynthesisDesignMethod | None = None
    selected_pathway_id: str | None = None
    selected_pathway_kind: str | None = None
    selected_pdm_mode: str | None = None
    selected_tier_origin: int | None = None
    selected_protected_pathway: bool = False
    task_count_by_method: dict[str, int] = Field(default_factory=dict)
    export_records: tuple[HeatExchangerNetworkSynthesisExportRecord, ...] = Field(
        default_factory=tuple,
    )

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        return _validate_run_id(value)

    @field_validator("approach_temperatures", "derivative_thresholds")
    @classmethod
    def _validate_positive_grid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _validate_positive_grid(value)

    @field_validator("stage_selection")
    @classmethod
    def _validate_stage_selection(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("stage_selection must contain at least one stage")
        if any(stage <= 0 for stage in value):
            raise ValueError("stage_selection values must be positive integers")
        if len(set(value)) != len(value):
            raise ValueError("stage_selection values must be unique")
        return tuple(int(stage) for stage in value)

    @field_validator("solve_tolerance")
    @classmethod
    def _validate_solve_tolerance(cls, value: float) -> float:
        return _validate_positive_finite(value)

    @field_validator(
        "best_solutions_to_save",
        "evm_n_ad_branches",
        "evm_n_rm_branches",
    )
    @classmethod
    def _validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be a positive integer")
        return value

    @field_validator("synthesis_quality_tier")
    @classmethod
    def _validate_synthesis_quality_tier(cls, value: int) -> int:
        if value < 0 or value > 5:
            raise ValueError("synthesis_quality_tier must be between 0 and 5")
        return value

    @field_validator("pdm_stage_pair_limit")
    @classmethod
    def _validate_pdm_stage_pair_limit(cls, value: int | None) -> int | None:
        if value is not None and (value < 0 or value > 12):
            raise ValueError("pdm_stage_pair_limit must be between 0 and 12")
        return value

    @field_validator("tdm_parent_limit")
    @classmethod
    def _validate_tdm_parent_limit(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("tdm_parent_limit must be positive when supplied")
        return value

    @field_validator("stage_packing")
    @classmethod
    def _validate_stage_packing(cls, value: str) -> str:
        if value not in {"auto", "none", "pdm", "tdm", "all"}:
            raise ValueError("stage_packing must be one of auto, none, pdm, tdm, all")
        return value

    @field_validator(
        "task_ids",
        "problem_id",
        "workspace_variant",
        "period_id",
        "selected_pathway_id",
        "selected_pathway_kind",
        "selected_pdm_mode",
    )
    @classmethod
    def _validate_id_collection_or_optional_text(cls, value):
        if isinstance(value, tuple):
            return tuple(_validate_optional_identity(item) for item in value)
        return _validate_optional_identity(value)

    @field_validator("selected_tier_origin")
    @classmethod
    def _validate_selected_tier_origin(cls, value: int | None) -> int | None:
        if value is not None and (value < 0 or value > 5):
            raise ValueError("selected_tier_origin must be between 0 and 5")
        return value

    @field_validator("task_count_by_method")
    @classmethod
    def _validate_task_count_by_method(cls, value: dict[str, int]) -> dict[str, int]:
        validated = {}
        for method, count in value.items():
            method_key = _validate_optional_identity(method)
            if method_key is None:
                raise ValueError("task_count_by_method keys must be non-empty")
            if count < 0:
                raise ValueError("task_count_by_method counts must be non-negative")
            validated[method_key] = int(count)
        return validated


HeatExchangerNetworkSynthesisExportRecord.model_rebuild()
HeatExchangerNetworkSynthesisManifest.model_rebuild()
