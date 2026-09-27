"""Structural view of the prepared problem state HEN synthesis reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...contracts.output import TargetOutput
    from ...domain.zone import Zone


@runtime_checkable
class PreparedProblem(Protocol):
    """The ``PinchProblem`` attributes used by HEN arrays, seeds and exports.

    Defined here so analysis modules can accept a live problem without
    importing the application layer. ``isinstance`` checks only that the
    attributes exist (looked up statically, so properties are not evaluated).
    """

    @property
    def master_zone(self) -> Zone | None: ...

    @property
    def results(self) -> TargetOutput | None: ...

    @property
    def project_name(self) -> str: ...


__all__ = ["PreparedProblem"]
