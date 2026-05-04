"""OpenPinch public API."""

import warnings as _warnings

from .classes.pinch_problem import PinchProblem
from .lib import *  # noqa: F401,F403
from .main import extract_results, get_targets, pinch_analysis_service
from .utils.stream_linearisation import get_piecewise_linearisation_for_streams

_warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"^CoolProp\.Plots\.SimpleCycles(Expansion|Compression)$",
)

__all__ = [name for name in globals() if not name.startswith("_")]
