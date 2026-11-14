"""Heuristic (Linnhoff rules) pinch design method orchestration."""

from __future__ import annotations

from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ..execution.executor import LocalSynthesisExecutor, SynthesisExecutor
from ..execution.settings import SynthesisWorkflowSettings
from ._stage import run_stage


def execute_heuristic_pinch_design_method_stage(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
):
    """Build and execute one heuristic pinch design stage."""
    return run_stage(
        build_heuristic_pinch_design_method_tasks(settings),
        problem=problem,
        settings=settings,
        executor=executor,
        default_executor=LocalSynthesisExecutor,
    )


def build_heuristic_pinch_design_method_tasks(
    settings: SynthesisWorkflowSettings,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """One heuristic task per configured dt_cont multiplier.

    Each task carries its network cap, ``HENS_HEURISTIC_PDM_NETWORKS``.
    """
    limit = int(settings.heuristic_pdm_networks)
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
    "build_heuristic_pinch_design_method_tasks",
    "execute_heuristic_pinch_design_method_stage",
]
