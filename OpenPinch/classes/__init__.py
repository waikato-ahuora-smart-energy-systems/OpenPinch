"""Core domain classes used throughout OpenPinch."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .heat_exchanger import (
        HeatExchanger,
        HeatExchangerKind,
        HeatExchangerStreamRole,
    )
    from .heat_exchanger_network import HeatExchangerNetwork
    from .pinch_problem import PinchProblem
    from .pinch_workspace import PinchWorkspace
    from .problem_table import ProblemTable
    from .stream import Stream
    from .stream_collection import StreamCollection
    from .value import Value
    from .zone import Zone

__all__ = [
    "HeatExchanger",
    "HeatExchangerKind",
    "HeatExchangerNetwork",
    "HeatExchangerStreamRole",
    "ProblemTable",
    "Stream",
    "StreamCollection",
    "Value",
    "Zone",
    "PinchProblem",
    "PinchWorkspace",
]

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "HeatExchanger": ("OpenPinch.classes.heat_exchanger", "HeatExchanger"),
    "HeatExchangerKind": ("OpenPinch.classes.heat_exchanger", "HeatExchangerKind"),
    "HeatExchangerStreamRole": (
        "OpenPinch.classes.heat_exchanger",
        "HeatExchangerStreamRole",
    ),
    "HeatExchangerNetwork": (
        "OpenPinch.classes.heat_exchanger_network",
        "HeatExchangerNetwork",
    ),
    "PinchProblem": ("OpenPinch.classes.pinch_problem", "PinchProblem"),
    "PinchWorkspace": ("OpenPinch.classes.pinch_workspace", "PinchWorkspace"),
    "ProblemTable": ("OpenPinch.classes.problem_table", "ProblemTable"),
    "Stream": ("OpenPinch.classes.stream", "Stream"),
    "StreamCollection": ("OpenPinch.classes.stream_collection", "StreamCollection"),
    "Value": ("OpenPinch.classes.value", "Value"),
    "Zone": ("OpenPinch.classes.zone", "Zone"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS))
