"""Closed lazy factory for concrete HPR point simulators."""

from __future__ import annotations

from .errors import HprSimulatorFailure
from .protocols import HprPointSimulator


def get_hpr_point_simulator(backend: str) -> HprPointSimulator:
    """Create one fresh explicitly selected concrete simulator session."""
    normalized = backend.strip().lower()
    if normalized == "coolprop":
        from .adapters.coolprop import CoolPropHprPointSimulator

        return CoolPropHprPointSimulator()
    if normalized == "tespy":
        try:
            from .adapters.tespy import TespyHprPointSimulator
        except ImportError as exc:
            raise HprSimulatorFailure(
                "dependency_unavailable",
                "TESPy simulation requires the optional 'tespy' extra",
                session_fatal=True,
                cause=exc,
            ) from exc
        return TespyHprPointSimulator()
    raise HprSimulatorFailure(
        "unsupported_backend",
        "simulation backend must be 'coolprop' or 'tespy'",
        session_fatal=True,
    )


__all__ = ["get_hpr_point_simulator"]
