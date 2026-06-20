"""OpenPinch-native heat exchanger network synthesis schemas."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...classes.heat_exchanger_network import HeatExchangerNetwork

SynthesisMethod = Literal[
    "pinch_decomposition",
    "topology_design",
    "energy_stage_refinement",
]
SynthesisTaskStatus = Literal["pending", "success", "failed", "skipped"]
SynthesisOutputFormat = Literal["json", "csv", "xlsx"]

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


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
    state_id: str | None = None
    seed_network_index: int | None = None
    parent_task_id: str | None = None
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
        "state_id",
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
        """Return the deterministic identifier for this task payload."""
        payload = {
            "approach_temperature": self.approach_temperature,
            "derivative_threshold": self.derivative_threshold,
            "method": self.method,
            "parent_task_id": self.parent_task_id,
            "problem_id": self.problem_id,
            "run_id": self.run_id,
            "seed_network_index": self.seed_network_index,
            "stage_count": self.stage_count,
            "state_id": self.state_id,
            "topology_restrictions": [
                restriction.model_dump(mode="json")
                for restriction in self.topology_restrictions
            ],
            "workspace_variant": self.workspace_variant,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"hens-task-{digest}"


class HeatExchangerNetworkSynthesisTask(HeatExchangerNetworkSynthesisMethodInput):
    """One deterministic OpenPinch heat exchanger network synthesis task record."""


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
        "pinch_decomposition",
        "topology_design",
        "energy_stage_refinement",
    )
    export_formats: tuple[SynthesisOutputFormat, ...] = Field(default_factory=tuple)
    solve_tolerance: float = 1e-3
    best_solutions_to_save: int = 1
    task_ids: tuple[str, ...] = Field(default_factory=tuple)
    problem_id: str | None = None
    workspace_variant: str | None = None
    state_id: str | None = None
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

    @field_validator("best_solutions_to_save")
    @classmethod
    def _validate_best_solutions_to_save(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("best_solutions_to_save must be a positive integer")
        return value

    @field_validator(
        "task_ids",
        "problem_id",
        "workspace_variant",
        "state_id",
    )
    @classmethod
    def _validate_id_collection_or_optional_text(cls, value):
        if isinstance(value, tuple):
            return tuple(_validate_optional_identity(item) for item in value)
        return _validate_optional_identity(value)


class HeatExchangerNetworkSynthesisMethodOutput(BaseModel):
    """Validated output for one PDM, TDM, or evolution method run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task: HeatExchangerNetworkSynthesisMethodInput
    status: SynthesisTaskStatus
    network: HeatExchangerNetwork | None = None
    objective_value: float | None = None
    solver_status: str | None = None
    error: str | None = None
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


class HeatExchangerNetworkSynthesisTaskOutcome(
    HeatExchangerNetworkSynthesisMethodOutput,
):
    """Outcome for one OpenPinch heat exchanger network synthesis task."""

    task: HeatExchangerNetworkSynthesisTask


class HeatExchangerNetworkSynthesisResult(BaseModel):
    """Problem-owned heat exchanger network synthesis result payload."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    network: HeatExchangerNetwork
    run_id: str
    task_id: str | None = None
    problem_id: str | None = None
    workspace_variant: str | None = None
    state_id: str | None = None
    solver_name: str | None = None
    solver_status: str | None = None
    method: SynthesisMethod | None = None
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
        "state_id",
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

    def grid_diagram(
        self,
        solution_rank: int = 1,
        *,
        stream_line_width: float = 5.0,
        temperature_scaled: bool = False,
    ):
        """Return an OpenHENS-style grid diagram for one ranked solution."""
        from ...services.network_grid_diagram import (
            build_grid_diagram,
        )

        if solution_rank < 1:
            raise IndexError("solution_rank is 1-based and must be at least 1")

        ranked = self.get_n_best_networks()
        if not ranked:
            if solution_rank == 1:
                network = self.network
            else:
                raise IndexError(
                    "solution_rank 2 is unavailable; only 1 network is available"
                )
        elif solution_rank > len(ranked):
            raise IndexError(
                f"solution_rank {solution_rank} is unavailable; only "
                f"{len(ranked)} network(s) are available"
            )
        else:
            selected = ranked[solution_rank - 1]
            if selected.network is None:
                raise ValueError(
                    "selected ranked network outcome is missing network output"
                )
            network = selected.network

        return build_grid_diagram(
            network,
            stream_line_width=stream_line_width,
            temperature_scaled=temperature_scaled,
        )

    def get_n_best_networks(self, n: int | None = None):
        """Return the best ranked network outcomes with duplicates removed."""
        from ...services.heat_exchanger_network_synthesis.ranking import (
            rank_unique_network_outcomes,
        )

        return rank_unique_network_outcomes(self, limit=n)

    def select_network(self, solution_rank: int = 1) -> Self:
        """Select ``network`` from the ranked network list and return this result."""
        ranked = self.get_n_best_networks()
        if solution_rank < 1:
            raise IndexError("solution_rank is 1-based and must be at least 1")
        if not ranked:
            if solution_rank == 1:
                return self
            raise IndexError(
                "solution_rank 2 is unavailable; only 1 network is available"
            )
        if solution_rank > len(ranked):
            raise IndexError(
                f"solution_rank {solution_rank} is unavailable; only "
                f"{len(ranked)} network(s) are available"
            )

        selected = ranked[solution_rank - 1]
        if selected.network is None:
            raise ValueError(
                "selected ranked network outcome is missing network output"
            )

        network = selected.network
        self.ranked_networks = ranked
        self.network = network
        self.task_id = selected.task.task_id
        self.solver_status = selected.solver_status
        self.method = selected.task.method
        self.stage_count = network.stage_count or selected.task.stage_count
        self.objective_values = {
            key: value
            for key, value in {
                "total_annual_cost": network.total_annual_cost,
                "utility_cost": network.utility_cost,
                "capital_cost": network.capital_cost,
            }.items()
            if value is not None
        }
        return self


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


__all__ = [
    "HeatExchangerNetworkSynthesisExportRecord",
    "HeatExchangerNetworkSynthesisManifest",
    "HeatExchangerNetworkSynthesisMethodInput",
    "HeatExchangerNetworkSynthesisMethodOutput",
    "HeatExchangerNetworkSynthesisResult",
    "HeatExchangerNetworkSynthesisTask",
    "HeatExchangerNetworkSynthesisTaskOutcome",
    "HeatExchangerNetworkTopologyRestriction",
    "SynthesisMethod",
    "SynthesisOutputFormat",
    "SynthesisTaskStatus",
]


HeatExchangerNetworkTopologyRestriction.model_rebuild()
HeatExchangerNetworkSynthesisMethodInput.model_rebuild()
HeatExchangerNetworkSynthesisTask.model_rebuild()
HeatExchangerNetworkSynthesisExportRecord.model_rebuild()
HeatExchangerNetworkSynthesisManifest.model_rebuild()
HeatExchangerNetworkSynthesisMethodOutput.model_rebuild()
HeatExchangerNetworkSynthesisTaskOutcome.model_rebuild()
HeatExchangerNetworkSynthesisResult.model_rebuild()
