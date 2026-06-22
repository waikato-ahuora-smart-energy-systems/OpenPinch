"""Resolved HEN synthesis workflow settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .....lib.enums import HeatExchangerNetworkDesignMethod, HENDesignMethod
from .....lib.schemas.synthesis import SynthesisMethod


@dataclass(frozen=True, init=False)
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
    evm_solver: str
    pdm_solver_options: dict[str, Any]
    tdm_solver_options: dict[str, Any]
    evm_solver_options: dict[str, Any]
    problem_id: str | None = None
    workspace_variant: str | None = None
    state_id: str | None = None
    design_method: HeatExchangerNetworkDesignMethod = HENDesignMethod.OpenHENS

    def __init__(
        self,
        *,
        run_id: str,
        approach_temperatures: tuple[float, ...],
        derivative_thresholds: tuple[float, ...],
        stage_selection: tuple[int, ...],
        method_sequence: tuple[SynthesisMethod, ...],
        output_formats: tuple[str, ...],
        solve_tolerance: float,
        best_solutions_to_save: int,
        max_parallel: int,
        pdm_solver: str,
        tdm_solver: str,
        pdm_solver_options: dict[str, Any],
        tdm_solver_options: dict[str, Any],
        evm_solver: str | None = None,
        evm_solver_options: dict[str, Any] | None = None,
        esm_solver: str | None = None,
        esm_solver_options: dict[str, Any] | None = None,
        problem_id: str | None = None,
        workspace_variant: str | None = None,
        state_id: str | None = None,
        design_method: HeatExchangerNetworkDesignMethod = HENDesignMethod.OpenHENS,
    ) -> None:
        resolved_evm_solver = evm_solver if evm_solver is not None else esm_solver
        if resolved_evm_solver is None:
            raise ValueError("evm_solver must be provided.")
        resolved_evm_solver_options = (
            evm_solver_options if evm_solver_options is not None else esm_solver_options
        )
        for name, value in {
            "run_id": run_id,
            "approach_temperatures": approach_temperatures,
            "derivative_thresholds": derivative_thresholds,
            "stage_selection": stage_selection,
            "method_sequence": method_sequence,
            "output_formats": output_formats,
            "solve_tolerance": solve_tolerance,
            "best_solutions_to_save": best_solutions_to_save,
            "max_parallel": max_parallel,
            "pdm_solver": pdm_solver,
            "tdm_solver": tdm_solver,
            "evm_solver": resolved_evm_solver,
            "pdm_solver_options": pdm_solver_options,
            "tdm_solver_options": tdm_solver_options,
            "evm_solver_options": dict(resolved_evm_solver_options or {}),
            "problem_id": problem_id,
            "workspace_variant": workspace_variant,
            "state_id": state_id,
            "design_method": design_method,
        }.items():
            object.__setattr__(self, name, value)

    def solver_for(self, method: SynthesisMethod | None) -> str | None:
        """Return the configured solver name for one workflow method."""
        if method == "pinch_design_method":
            return self.pdm_solver
        if method == "thermal_derivative_method":
            return self.tdm_solver
        if method == "network_evolution_method":
            return self.evm_solver
        return None

    def solver_options_for(self, method: SynthesisMethod | None) -> dict[str, Any]:
        """Return user-provided solver options for one workflow method."""
        if method == "pinch_design_method":
            return dict(self.pdm_solver_options)
        if method == "thermal_derivative_method":
            return dict(self.tdm_solver_options)
        if method == "network_evolution_method":
            return dict(self.evm_solver_options)
        return {}

    @property
    def esm_solver(self) -> str:
        """Backward-compatible alias for the network evolution solver."""
        return self.evm_solver

    @property
    def esm_solver_options(self) -> dict[str, Any]:
        """Backward-compatible alias for network evolution solver options."""
        return dict(self.evm_solver_options)


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
    hens = config.hens
    return SynthesisWorkflowSettings(
        run_id=str(hens.run_id),
        approach_temperatures=tuple(
            float(value) for value in hens.approach_temperatures
        ),
        derivative_thresholds=tuple(
            float(value) for value in hens.derivative_thresholds
        ),
        stage_selection=tuple(int(value) for value in hens.stage_selection),
        method_sequence=tuple(
            HeatExchangerNetworkDesignMethod(value) for value in hens.method_sequence
        ),
        output_formats=tuple(hens.output_formats),
        solve_tolerance=float(hens.solve_tolerance),
        best_solutions_to_save=int(hens.best_solutions_to_save),
        max_parallel=int(hens.max_parallel),
        pdm_solver=str(hens.solver_pdm),
        tdm_solver=str(hens.solver_tdm),
        evm_solver=str(hens.solver_evm),
        pdm_solver_options=dict(hens.solver_options_pdm),
        tdm_solver_options=dict(hens.solver_options_tdm),
        evm_solver_options=dict(hens.solver_options_evm),
        problem_id=problem.project_name,
        workspace_variant=workspace_variant,
        state_id=state_id,
        design_method=HENDesignMethod.OpenHENS,
    )


__all__ = ["SynthesisWorkflowSettings", "workflow_settings_from_problem"]
