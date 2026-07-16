"""Result-level HEN synthesis schemas."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ....classes.heat_exchanger_network import HeatExchangerNetwork
from .common import (
    HeatExchangerNetworkSynthesisManifest,
    SynthesisDesignMethod,
    SynthesisMethod,
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
    design_method: SynthesisDesignMethod | None = None
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

    def grid_diagram(
        self,
        solution_rank: int = 1,
        *,
        period_id: str | None = None,
        stream_line_width: float = 5.0,
        temperature_scaled: bool = False,
    ):
        """Return an OpenHENS-style grid diagram for one ranked solution."""
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

        return network.build_grid_diagram(
            period_id=period_id,
            stream_line_width=stream_line_width,
            temperature_scaled=temperature_scaled,
        )

    def get_n_best_networks(self, n: int | None = None):
        """Return the best ranked network outcomes with duplicates removed."""
        from ....services.heat_exchanger_network_synthesis.common.reporting import (
            ranking,
        )

        return ranking.rank_unique_network_outcomes(self, limit=n)

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


HeatExchangerNetworkSynthesisResult.model_rebuild()
