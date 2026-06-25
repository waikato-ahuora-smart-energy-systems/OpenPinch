"""Shared optional dependency error messages."""

from __future__ import annotations

from collections.abc import Iterable


def optional_dependency_error(
    *,
    package: str,
    purpose: str,
    extras: str | Iterable[str],
    docs: str | None = None,
) -> str:
    """Return a consistent message for unavailable optional dependencies."""
    extras_list = [extras] if isinstance(extras, str) else list(extras)
    installs = " or ".join(
        f'python -m pip install "openpinch[{extra}]"' for extra in extras_list
    )
    docs_text = f" See {docs}." if docs else ""
    return (
        f"{package} is required for {purpose}. "
        f"Install the optional dependency with {installs}.{docs_text}"
    )


__all__ = ["optional_dependency_error"]
