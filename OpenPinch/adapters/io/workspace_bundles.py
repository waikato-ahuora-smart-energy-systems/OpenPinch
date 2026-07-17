"""JSON persistence adapter for workspace bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ...contracts.workspace import PinchWorkspaceBundle

if TYPE_CHECKING:
    from .problem_sources import PathLike


def load_workspace_bundle(path: "PathLike") -> PinchWorkspaceBundle:
    """Read and validate a workspace bundle from disk."""
    return PinchWorkspaceBundle.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def save_workspace_bundle(
    path: "PathLike",
    bundle: PinchWorkspaceBundle,
) -> Path:
    """Write one validated workspace bundle and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(bundle.model_dump(mode="python"), indent=2),
        encoding="utf-8",
    )
    return destination


__all__ = ["load_workspace_bundle", "save_workspace_bundle"]
