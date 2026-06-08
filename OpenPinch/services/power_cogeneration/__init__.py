"""Power cogeneration analysis service exports."""

from .power_cogeneration_entry import (
    get_power_cogeneration_above_pinch,
    get_power_cogeneration_below_pinch,
)

__all__ = [
    "get_power_cogeneration_above_pinch",
    "get_power_cogeneration_below_pinch",
]
