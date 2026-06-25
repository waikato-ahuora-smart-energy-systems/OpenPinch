"""Stage-wise HEN model with explicit stage packing constraints."""

from __future__ import annotations

from .stage_packing import add_recovery_stage_packing_constraints
from .stagewise import StageWiseModel


class StagePackedStageWiseModel(StageWiseModel):
    """StageWise model with integer-stage symmetry reduced for TDM solves."""

    def set_stage_wise_superstructure(self) -> None:
        super().set_stage_wise_superstructure()
        add_recovery_stage_packing_constraints(self)


__all__ = ["StagePackedStageWiseModel"]
