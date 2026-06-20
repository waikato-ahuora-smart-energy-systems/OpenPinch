"""Resolved HEN synthesis workflow settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .....lib.enums import HeatExchangerNetworkDesignMethod, HENDesignMethod
from .....lib.schemas.synthesis import SynthesisMethod


@dataclass(frozen=True)
class SynthesisWorkflowSettings:
    """Resolved synthesis controls read from a prepared OpenPinch problem."""

    run_id: str
    approach_temperatures: tuple[float, ...]
    derivative_thresholds: tuple[float, ...]
    stage_selection: tuple[int, ...]
    method_sequence: tuple[SynthesisMethod, ...]
    output_formats: tuple[str, ...]
    solve_tolerance: float
    best_solutions_to_save: int
    max_parallel: int
    pdm_solver: str
    tdm_solver: str
    esm_solver: str
    pdm_solver_options: dict[str, Any]
    tdm_solver_options: dict[str, Any]
    esm_solver_options: dict[str, Any]
    problem_id: str | None = None
    workspace_variant: str | None = None
    state_id: str | None = None
    design_method: HeatExchangerNetworkDesignMethod = HENDesignMethod.OpenHENS

    def solver_for(self, method: SynthesisMethod | None) -> str | None:
        """Return the configured solver name for one workflow method."""
        if method == "pinch_design_method":
            return self.pdm_solver
        if method == "thermal_derivative_method":
            return self.tdm_solver
        if method == "network_evolution_method":
            return self.esm_solver
        return None

    def solver_options_for(self, method: SynthesisMethod | None) -> dict[str, Any]:
        """Return user-provided solver options for one workflow method."""
        if method == "pinch_design_method":
            return dict(self.pdm_solver_options)
        if method == "thermal_derivative_method":
            return dict(self.tdm_solver_options)
        if method == "network_evolution_method":
            return dict(self.esm_solver_options)
        return {}


def workflow_settings_from_problem(
    problem,
    *,
    state_id: str | None = None,
    workspace_variant: str | None = None,
) -> SynthesisWorkflowSettings:
    """Read persistent synthesis controls from a prepared problem configuration."""
    zone = problem.master_zone
    if zone is None:
        raise RuntimeError(
            "heat exchanger network synthesis requires a loaded PinchProblem."
        )
    config = zone.config
    return SynthesisWorkflowSettings(
        run_id=str(config.HENS_RUN_ID),
        approach_temperatures=tuple(
            float(value) for value in config.HENS_APPROACH_TEMPERATURES
        ),
        derivative_thresholds=tuple(
            float(value) for value in config.HENS_DERIVATIVE_THRESHOLDS
        ),
        stage_selection=tuple(int(value) for value in config.HENS_STAGE_SELECTION),
        method_sequence=tuple(
            HeatExchangerNetworkDesignMethod(value)
            for value in config.HENS_METHOD_SEQUENCE
        ),
        output_formats=tuple(config.HENS_OUTPUT_FORMATS),
        solve_tolerance=float(config.HENS_SOLVE_TOLERANCE),
        best_solutions_to_save=int(config.HENS_BEST_SOLUTIONS_TO_SAVE),
        max_parallel=int(config.HENS_MAX_PARALLEL),
        pdm_solver=str(config.HENS_PDM_SOLVER),
        tdm_solver=str(config.HENS_TDM_SOLVER),
        esm_solver=str(config.HENS_ESM_SOLVER),
        pdm_solver_options=dict(config.HENS_PDM_SOLVER_OPTIONS),
        tdm_solver_options=dict(config.HENS_TDM_SOLVER_OPTIONS),
        esm_solver_options=dict(config.HENS_ESM_SOLVER_OPTIONS),
        problem_id=problem.project_name,
        workspace_variant=workspace_variant,
        state_id=state_id,
        design_method=HENDesignMethod.OpenHENS,
    )


__all__ = ["SynthesisWorkflowSettings", "workflow_settings_from_problem"]
