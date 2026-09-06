"""Immutable composition of explicitly owned result-family adapters."""

from types import MappingProxyType

from .specialist import SPECIALIST_RESULT_ADAPTERS
from .thermal import THERMAL_RESULT_ADAPTERS

RESULT_ADAPTERS = MappingProxyType(THERMAL_RESULT_ADAPTERS | SPECIALIST_RESULT_ADAPTERS)


def adapt_analysis_result(target, *, is_total=False):
    """Adapt by declared result family; no numerical execution occurs here."""
    if getattr(target, "reportable", True):
        for family in type(target).__mro__:
            adapter = RESULT_ADAPTERS.get(family)
            if adapter is not None:
                return adapter(target, is_total=is_total)
    raise NotImplementedError(f"Reporting is not defined for {type(target).__name__}.")
