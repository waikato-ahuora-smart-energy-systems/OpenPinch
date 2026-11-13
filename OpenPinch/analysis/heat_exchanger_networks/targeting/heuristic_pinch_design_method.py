"""Heuristic (Linnhoff rules) pinch design method orchestration."""

from __future__ import annotations

from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.settings import SynthesisWorkflowSettings
from ..results.assembly import SynthesisWorkflowResult
from ._stage import run_single_method_workflow, run_stage

# Networks per multiplier for the standalone method when none is configured.
STANDALONE_NETWORK_LIMIT = 10


def _execute_heuristic_pinch_design_method_workflow(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Execute only the heuristic pinch design method."""
    return run_single_method_workflow(
        settings,
        HeatExchangerNetworkDesignMethod.HeuristicPinchDesign,
        lambda method_settings: execute_heuristic_pinch_design_method_stage(
            problem,
            method_settings,
            executor=executor,
            standalone=True,
        ),
    )


def execute_heuristic_pinch_design_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
    standalone: bool = False,
):
    """Build and execute one heuristic pinch design stage."""
    return run_stage(
        build_heuristic_pinch_design_method_tasks(settings, standalone=standalone),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
    )


def build_heuristic_pinch_design_method_tasks(
    settings: SynthesisWorkflowSettings,
    *,
    standalone: bool = False,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """One heuristic task per configured dt_cont multiplier.

    Each task carries its network cap, ``HENS_HEURISTIC_PDM_NETWORKS``; the
    standalone method uses ``STANDALONE_NETWORK_LIMIT`` when that is 0.
    """
    limit = int(settings.heuristic_pdm_networks)
    if standalone and limit <= 0:
        limit = STANDALONE_NETWORK_LIMIT
    return tuple(
        HeatExchangerNetworkSynthesisTask(
            run_id=settings.run_id,
            method=HeatExchangerNetworkDesignMethod.HeuristicPinchDesign,
            dt_cont_multiplier=multiplier,
            problem_id=settings.problem_id,
            workspace_variant=settings.workspace_variant,
            period_id=settings.period_id,
            settings={"heuristic_networks": limit},
        )
        for multiplier in dict.fromkeys(settings.dt_cont_multipliers)
    )


__all__ = [
    "STANDALONE_NETWORK_LIMIT",
    "_execute_heuristic_pinch_design_method_workflow",
    "build_heuristic_pinch_design_method_tasks",
    "execute_heuristic_pinch_design_method_stage",
]
