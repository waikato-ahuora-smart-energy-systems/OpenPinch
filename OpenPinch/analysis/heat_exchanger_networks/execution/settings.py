"""Resolved HEN synthesis workflow settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....domain.enums import HeatExchangerNetworkDesignMethod
from .search import DEPTH_LEVELS, resolve_search

StagePackingScope = str

_MAX_PDM_STAGE_PAIR_COUNT = 12
_MAX_TDM_PARENT_LIMIT = 10
_OPEN_HENS_METHOD_SEQUENCE = (
    HeatExchangerNetworkDesignMethod.PinchDesign,
    HeatExchangerNetworkDesignMethod.ThermalDerivative,
    HeatExchangerNetworkDesignMethod.NetworkEvolution,
)

# The heuristic pinch design method realises its networks with the PDM
# side models, so it runs on the PDM solver settings.
_PINCH_DESIGN_METHODS = (
    HeatExchangerNetworkDesignMethod.PinchDesign,
    HeatExchangerNetworkDesignMethod.HeuristicPinchDesign,
)


@dataclass(frozen=True, kw_only=True)
class SynthesisWorkflowSettings:
    """Resolved synthesis controls read from a prepared OpenPinch problem."""

    run_id: str
    dt_cont_multipliers: tuple[float, ...] = (1.0,)
    derivative_thresholds: tuple[float, ...]
    stage_selection: tuple[int, ...]
    method_sequence: tuple[HeatExchangerNetworkDesignMethod, ...]
    output_formats: tuple[str, ...]
    solve_tolerance: float
    best_solutions_to_save: int
    max_parallel: int
    # The search levels; workflow_settings_from_problem resolves the options
    # they set into the fields below.
    search_width: int = 1
    search_depth: int = 2
    pdm_stage_pair_limit: int | None = None
    tdm_parent_limit: int | None = None
    stage_packing: StagePackingScope = "auto"
    evm_n_ad_branches: int | None = None
    evm_n_rm_branches: int | None = None
    # Cap on heuristic pinch design networks per multiplier; 0 runs none.
    heuristic_pdm_networks: int = 10
    pdm_milp: bool = False
    pdm_solver: str
    tdm_solver: str
    evm_solver: str
    pdm_solver_options: dict[str, Any]
    tdm_solver_options: dict[str, Any]
    evm_solver_options: dict[str, Any] | None = None
    problem_id: str | None = None
    workspace_variant: str | None = None
    period_id: str | None = None
    design_method: HeatExchangerNetworkDesignMethod = (
        HeatExchangerNetworkDesignMethod.OpenHENS
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evm_solver_options", dict(self.evm_solver_options or {})
        )

    def solver_for(self, method: HeatExchangerNetworkDesignMethod | None) -> str | None:
        """Return the configured solver name for one workflow method."""
        if method in _PINCH_DESIGN_METHODS:
            return self.pdm_solver
        if method == HeatExchangerNetworkDesignMethod.ThermalDerivative:
            return self.tdm_solver
        if method == HeatExchangerNetworkDesignMethod.NetworkEvolution:
            return self.evm_solver
        return None

    def solver_options_for(
        self, method: HeatExchangerNetworkDesignMethod | None
    ) -> dict[str, Any]:
        """Return user-provided solver options for one workflow method."""
        if method in _PINCH_DESIGN_METHODS:
            return dict(self.pdm_solver_options)
        if method == HeatExchangerNetworkDesignMethod.ThermalDerivative:
            return dict(self.tdm_solver_options)
        if method == HeatExchangerNetworkDesignMethod.NetworkEvolution:
            return dict(self.evm_solver_options)
        return {}

    @property
    def runs_evolution(self) -> bool:
        """Whether network evolution runs (search depth 1 and above)."""

        return DEPTH_LEVELS[self.search_depth].evolution

    @property
    def runs_thermal_derivative_method(self) -> bool:
        """Whether MILP networks go through the thermal derivative method."""

        return DEPTH_LEVELS[self.search_depth].thermal_derivative

    @property
    def milp_networks_seed_evolution(self) -> bool:
        """Whether every MILP network also seeds evolution directly."""

        return DEPTH_LEVELS[self.search_depth].milp_seeds_evolution

    @property
    def quality_pdm_stage_pair_count(self) -> int:
        """Return explicitly requested stage-pair PDM candidates."""

        if self.pdm_stage_pair_limit is not None:
            return max(0, int(self.pdm_stage_pair_limit))
        return 0

    @property
    def runs_milp_pinch_design(self) -> bool:
        """Whether the MILP pinch design method runs.

        The heuristic is the default pinch design method; the MILP runs when
        asked for, or when the heuristic makes no networks.
        """

        return self.pdm_milp or self.heuristic_pdm_networks <= 0

    @property
    def quality_derivative_thresholds(self) -> tuple[float, ...]:
        """Return the distinct configured TDM thresholds."""

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
        return DEPTH_LEVELS[self.search_depth].evolution_breadth

    @property
    def effective_evm_n_rm_branches(self) -> int:
        """Return derived or explicitly overridden remove-branch breadth."""

        if self.evm_n_rm_branches is not None:
            return max(1, int(self.evm_n_rm_branches))
        return DEPTH_LEVELS[self.search_depth].evolution_breadth


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
    search = resolve_search(hens)
    return SynthesisWorkflowSettings(
        run_id=str(hens.run_id),
        dt_cont_multipliers=search.dt_cont_multipliers,
        derivative_thresholds=tuple(
            float(value) for value in hens.derivative_thresholds
        ),
        stage_selection=tuple(int(value) for value in hens.stage_selection),
        method_sequence=_OPEN_HENS_METHOD_SEQUENCE,
        output_formats=tuple(hens.output_formats),
        solve_tolerance=float(hens.solve_tolerance),
        best_solutions_to_save=int(hens.best_solutions_to_save),
        max_parallel=int(hens.max_parallel),
        search_width=search.width,
        search_depth=search.depth,
        pdm_stage_pair_limit=(
            None
            if hens.pdm_stage_pair_limit is None
            else int(hens.pdm_stage_pair_limit)
        ),
        tdm_parent_limit=(
            None if hens.tdm_parent_limit is None else int(hens.tdm_parent_limit)
        ),
        stage_packing=_stage_packing_scope(hens),
        evm_n_ad_branches=search.evm_n_ad_branches,
        evm_n_rm_branches=search.evm_n_rm_branches,
        heuristic_pdm_networks=search.heuristic_networks,
        pdm_milp=search.milp,
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
