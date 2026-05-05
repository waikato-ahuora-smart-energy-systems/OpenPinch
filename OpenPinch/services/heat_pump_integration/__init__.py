"""Heat-pump and refrigeration targeting services."""

from .heat_pump_and_refrigeration_entry import (
    compute_direct_heat_pump_or_refrigeration_target,
    compute_indirect_heat_pump_or_refrigeration_target,
    plot_multi_hp_profiles_from_results,
)

__all__ = [
    "compute_direct_heat_pump_or_refrigeration_target",
    "compute_indirect_heat_pump_or_refrigeration_target",
    "plot_multi_hp_profiles_from_results",
]
