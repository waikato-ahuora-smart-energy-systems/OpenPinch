"""Resolved HEN synthesis workflow settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....domain.enums import HeatExchangerNetworkDesignMethod
from .pathways import tier_evm_branch_breadth, tier_pdm_multipliers

StagePackingScope = str

_MAX_PDM_STAGE_PAIR_COUNT = 12
_MAX_TDM_PARENT_LIMIT = 10
_OPEN_HENS_METHOD_SEQUENCE = (
    HeatExchangerNetworkDesignMethod.PinchDesign,
    HeatExchangerNetworkDesignMethod.ThermalDerivative,
    HeatExchangerNetworkDesignMethod.NetworkEvolution,
)


@dataclass(frozen=True, init=False)
class SynthesisWorkflowSettings:
    """Resolved synthesis controls read from a prepared OpenPinch problem."""

    run_id: str
    approach_temperatures: tuple[float, ...]
    dt_cont_multipliers: tuple[float, ...] | None
    derivative_thresholds: tuple[float, ...]
    stage_selection: tuple[int, ...]
    method_sequence: tuple[HeatExchangerNetworkDesignMethod, ...]
    output_formats: tuple[str, ...]
    solve_tolerance: float
    best_solutions_to_save: int
    max_parallel: int
    synthesis_quality_tier: int
    pdm_stage_pair_limit: int | None
    tdm_parent_limit: int | None
    stage_packing: StagePackingScope
    evm_n_ad_branches: int | None
    evm_n_rm_branches: int | None
    user_dt_cont_multipliers: bool
    pdm_solver: str
    tdm_solver: str
    evm_solver: str
    pdm_solver_options: dict[str, Any]
    tdm_solver_options: dict[str, Any]
    evm_solver_options: dict[str, Any]
    problem_id: str | None = None
    workspace_variant: str | None = None
    period_id: str | None = None
    design_method: HeatExchangerNetworkDesignMethod = (
        HeatExchangerNetworkDesignMethod.OpenHENS
    )

    def __init__(
        self,
        *,
        run_id: str,
        approach_temperatures: tuple[float, ...],
        dt_cont_multipliers: tuple[float, ...] | None = None,
        derivative_thresholds: tuple[float, ...],
        stage_selection: tuple[int, ...],
        method_sequence: tuple[HeatExchangerNetworkDesignMethod, ...],
        output_formats: tuple[str, ...],
        solve_tolerance: float,
        best_solutions_to_save: int,
        max_parallel: int,
        pdm_solver: str,
        tdm_solver: str,
        pdm_solver_options: dict[str, Any],
        tdm_solver_options: dict[str, Any],
        evm_solver: str,
        evm_solver_options: dict[str, Any] | None = None,
        synthesis_quality_tier: int = 1,
        pdm_stage_pair_limit: int | None = None,
        tdm_parent_limit: int | None = None,
        stage_packing: StagePackingScope = "auto",
        user_dt_cont_multipliers: bool = False,
        evm_n_ad_branches: int | None = None,
        evm_n_rm_branches: int | None = None,
        problem_id: str | None = None,
        workspace_variant: str | None = None,
        period_id: str | None = None,
        design_method: HeatExchangerNetworkDesignMethod = (
            HeatExchangerNetworkDesignMethod.OpenHENS
        ),
    ) -> None:
        for name, value in {
            "run_id": run_id,
            "approach_temperatures": approach_temperatures,
            "dt_cont_multipliers": dt_cont_multipliers,
            "derivative_thresholds": derivative_thresholds,
            "stage_selection": stage_selection,
            "method_sequence": method_sequence,
            "output_formats": output_formats,
            "solve_tolerance": solve_tolerance,
            "best_solutions_to_save": best_solutions_to_save,
            "max_parallel": max_parallel,
            "synthesis_quality_tier": synthesis_quality_tier,
            "pdm_stage_pair_limit": pdm_stage_pair_limit,
            "tdm_parent_limit": tdm_parent_limit,
            "stage_packing": stage_packing,
            "evm_n_ad_branches": evm_n_ad_branches,
            "evm_n_rm_branches": evm_n_rm_branches,
            "user_dt_cont_multipliers": user_dt_cont_multipliers,
            "pdm_solver": pdm_solver,
            "tdm_solver": tdm_solver,
            "evm_solver": evm_solver,
            "pdm_solver_options": pdm_solver_options,
            "tdm_solver_options": tdm_solver_options,
            "evm_solver_options": dict(evm_solver_options or {}),
            "problem_id": problem_id,
            "workspace_variant": workspace_variant,
            "period_id": period_id,
            "design_method": design_method,
        }.items():
            object.__setattr__(self, name, value)

    def solver_for(self, method: HeatExchangerNetworkDesignMethod | None) -> str | None:
        """Return the configured solver name for one workflow method."""
        if method == "pinch_design_method":
            return self.pdm_solver
        if method == "thermal_derivative_method":
            return self.tdm_solver
        if method == "network_evolution_method":
            return self.evm_solver
        return None

    def solver_options_for(
        self, method: HeatExchangerNetworkDesignMethod | None
    ) -> dict[str, Any]:
        """Return user-provided solver options for one workflow method."""
        if method == "pinch_design_method":
            return dict(self.pdm_solver_options)
        if method == "thermal_derivative_method":
            return dict(self.tdm_solver_options)
        if method == "network_evolution_method":
            return dict(self.evm_solver_options)
        return {}

    @property
    def quality_fraction(self) -> float:
        """Return the tier-2..5 search breadth on a 0..1 scale."""

        if self.synthesis_quality_tier <= 1:
            return 0.0
        return min(1.0, max(0.0, (int(self.synthesis_quality_tier) - 1) / 4.0))

    @property
    def is_standard_quality_tier(self) -> bool:
        """Return whether the workflow should use exact standard OpenHENS."""

        return int(self.synthesis_quality_tier) == 1

    @property
    def skips_thermal_derivative_method(self) -> bool:
        """Return whether this tier runs PDM directly into EVM."""

        return int(self.synthesis_quality_tier) == 0

    @property
    def quality_dt_cont_multipliers(self) -> tuple[float, ...]:
        """Return tier-generated PDM multiplier candidates for expanded paths."""

        if self.synthesis_quality_tier <= 1:
            return ()
        if self.user_dt_cont_multipliers:
            return tuple(float(value) for value in self.dt_cont_multipliers or ())
        return tier_pdm_multipliers(int(self.synthesis_quality_tier))

    @property
    def quality_pdm_approach_temperatures(self) -> tuple[float, ...]:
        """Return concrete PDM dTmin values for tier multiplier candidates."""

        if not self.quality_dt_cont_multipliers:
            return ()
        base_approach = float(self.approach_temperatures[0])
        return tuple(
            dict.fromkeys(
                base_approach * float(multiplier)
                for multiplier in self.quality_dt_cont_multipliers
            )
        )

    @property
    def quality_pdm_stage_pair_count(self) -> int:
        """Return explicitly requested stage-pair PDM candidates."""

        if self.pdm_stage_pair_limit is not None:
            return max(0, int(self.pdm_stage_pair_limit))
        return 0

    @property
    def quality_derivative_thresholds(self) -> tuple[float, ...]:
        """Return configured TDM thresholds without tier-generated sweeps."""

        return tuple(
            dict.fromkeys(float(value) for value in self.derivative_thresholds)
        )

    @property
    def quality_tdm_parent_limit(self) -> int:
        """Return explicit parent limit metadata for reporting."""

        if self.tdm_parent_limit is not None:
            return max(1, int(self.tdm_parent_limit))
        return max(1, int(self.best_solutions_to_save))

    @property
    def effective_evm_n_ad_branches(self) -> int:
        """Return derived or explicitly overridden add-branch breadth."""

        if self.evm_n_ad_branches is not None:
            return max(1, int(self.evm_n_ad_branches))
        return tier_evm_branch_breadth(self.synthesis_quality_tier)

    @property
    def effective_evm_n_rm_branches(self) -> int:
        """Return derived or explicitly overridden remove-branch breadth."""

        if self.evm_n_rm_branches is not None:
            return max(1, int(self.evm_n_rm_branches))
        return tier_evm_branch_breadth(self.synthesis_quality_tier)


def workflow_settings_from_problem(
    problem,
    *,
    period_id: str | None = None,
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
    dt_cont_multipliers = getattr(hens, "dt_cont_multipliers", None)
    return SynthesisWorkflowSettings(
        run_id=str(hens.run_id),
        approach_temperatures=tuple(
            float(value) for value in hens.approach_temperatures
        ),
        dt_cont_multipliers=(
            None
            if dt_cont_multipliers is None
            else tuple(float(value) for value in dt_cont_multipliers)
        ),
        derivative_thresholds=tuple(
            float(value) for value in hens.derivative_thresholds
        ),
        stage_selection=tuple(int(value) for value in hens.stage_selection),
        method_sequence=_OPEN_HENS_METHOD_SEQUENCE,
        output_formats=tuple(hens.output_formats),
        solve_tolerance=float(hens.solve_tolerance),
        best_solutions_to_save=int(hens.best_solutions_to_save),
        max_parallel=int(hens.max_parallel),
        synthesis_quality_tier=int(hens.synthesis_quality_tier),
        pdm_stage_pair_limit=(
            None
            if hens.pdm_stage_pair_limit is None
            else int(hens.pdm_stage_pair_limit)
        ),
        tdm_parent_limit=(
            None if hens.tdm_parent_limit is None else int(hens.tdm_parent_limit)
        ),
        stage_packing=_stage_packing_scope(hens),
        evm_n_ad_branches=(
            None if hens.evm_n_ad_branches is None else int(hens.evm_n_ad_branches)
        ),
        evm_n_rm_branches=(
            None if hens.evm_n_rm_branches is None else int(hens.evm_n_rm_branches)
        ),
        user_dt_cont_multipliers=dt_cont_multipliers is not None,
        pdm_solver=str(hens.solver_pdm),
        tdm_solver=str(hens.solver_tdm),
        evm_solver=str(hens.solver_evm),
        pdm_solver_options=dict(hens.solver_options_pdm),
        tdm_solver_options=dict(hens.solver_options_tdm),
        evm_solver_options=dict(hens.solver_options_evm),
        problem_id=problem.project_name,
        workspace_variant=workspace_variant,
        period_id=period_id,
        design_method=HeatExchangerNetworkDesignMethod.OpenHENS,
    )


def _stage_packing_scope(hens) -> StagePackingScope:
    return str(getattr(hens, "stage_packing", "auto"))


__all__ = [
    "StagePackingScope",
    "SynthesisWorkflowSettings",
    "workflow_settings_from_problem",
]
