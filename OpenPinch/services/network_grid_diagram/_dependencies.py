"""Optional dependency guards for network grid diagram rendering."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

GRAPHING_EXTRA = "synthesis"


class MissingNetworkGridDiagramDependencyError(ImportError):
    """Raised when an optional grid diagram dependency is unavailable."""


def require_network_grid_diagram_dependency(
    import_name: str,
    *,
    package: str | None = None,
    purpose: str | None = None,
) -> ModuleType:
    """Import one optional grid diagram dependency or raise an actionable error."""
    package_name = package or import_name.split(".", maxsplit=1)[0]
    purpose_text = f" for {purpose}" if purpose else ""

    try:
        return import_module(import_name)
    except ImportError as exc:
        raise MissingNetworkGridDiagramDependencyError(
            f"{package_name} is required{purpose_text}. Install the optional "
            "heat exchanger network synthesis dependencies with "
            f"'python -m pip install \"openpinch[{GRAPHING_EXTRA}]\"'."
        ) from exc


__all__ = [
    "MissingNetworkGridDiagramDependencyError",
    "require_network_grid_diagram_dependency",
]
