"""Pinch-decomposition model with explicit stage packing constraints."""

from __future__ import annotations

from .pinch_design import PinchDecompModel
from .stage_packing import add_recovery_stage_packing_constraints


class StagePackedPinchDecompModel(PinchDecompModel):
    """PDM slice with packed recovery stages to reduce stage-index symmetry."""

    def set_stage_wise_superstructure(self) -> None:
        super().set_stage_wise_superstructure()
        add_recovery_stage_packing_constraints(self)


__all__ = ["StagePackedPinchDecompModel"]
