"""The effective topology of a solved vapour compression design."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....contracts.hpr import HeatPumpTargetInputs, HPRActiveStage, HPREffectiveTopology
from ..optimisation_adapter import NEGLIGIBLE_STAGE_WORK_FRACTION

__all__ = ["effective_topology", "topology_from_stages"]


def effective_topology(model: Any, args: HeatPumpTargetInputs) -> HPREffectiveTopology:
    """Return the requested layout and the stages the solved ``model`` uses.

    ``model`` is a solved cascade, parallel or VC+MVR model. A stage is active
    when it does more than ``NEGLIGIBLE_STAGE_WORK_FRACTION`` of the total work
    and lifts at least ``args.min_stage_lift``. Stages are indexed in the
    model's order: hottest first for a cascade, then MVR stages after the
    vapour compression stages of a VC+MVR cascade.
    """
    return topology_from_stages(_stages(model), args)


def topology_from_stages(
    stages: list[HPRActiveStage], args: HeatPumpTargetInputs
) -> HPREffectiveTopology:
    """Return the requested layout and which of the configured ``stages`` work.

    The same rule as :func:`effective_topology`, for a design described by
    its stages rather than a solved model (a TESPy target).
    """
    total_work = sum(abs(stage.work) for stage in stages)
    min_lift = float(getattr(args, "min_stage_lift", 0.0) or 0.0)
    active = tuple(
        stage
        for stage in stages
        if abs(stage.work) > NEGLIGIBLE_STAGE_WORK_FRACTION * total_work
        and stage.condensing_temperature - stage.evaporating_temperature >= min_lift
    )
    return HPREffectiveTopology.from_stages(
        cycle=str(args.hpr_type),
        condensers=int(args.n_cond),
        evaporators=int(args.n_evap),
        configured_stages=len(stages),
        active_stages=active,
    )


def _stages(model: Any) -> list[HPRActiveStage]:
    if hasattr(model, "vc_cycles"):
        stages = [_vc_stage(i, cycle) for i, cycle in enumerate(model.vc_cycles)]
        offset = len(stages)
        for i, cycle in enumerate(model.mvr_cycles):
            stages.append(
                HPRActiveStage(
                    index=offset + i,
                    role="mvr",
                    fluid="Water",
                    evaporating_temperature=float(np.ravel(model.T_evap_mvr)[i]),
                    condensing_temperature=float(np.ravel(model.T_cond_mvr)[i]),
                    pressure_ratio=float(cycle.Ps[1] / cycle.Ps[0]),
                    work=float(cycle.work),
                    process_heat=float(cycle.process_heat),
                )
            )
        return stages
    return [_vc_stage(i, cycle) for i, cycle in enumerate(model._subcycles)]


def _vc_stage(index: int, cycle: Any) -> HPRActiveStage:
    return HPRActiveStage(
        index=index,
        role="vapour_compression",
        fluid=str(cycle.refrigerant),
        evaporating_temperature=float(cycle.T_evap_sat_vap),
        condensing_temperature=float(cycle.T_cond_sat_liq),
        pressure_ratio=float(cycle.Ps[1] / cycle.Ps[0]),
        work=float(cycle.work),
        process_heat=float(cycle.Q_heat or 0.0),
        process_cooling=float(cycle.Q_cool or 0.0),
    )
