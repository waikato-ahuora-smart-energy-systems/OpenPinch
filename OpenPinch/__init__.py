"""OpenPinch public API surface.

The package exposes a small set of high-level helpers for running Pinch Analysis
and working with the structured results.  Detailed configuration and schema
objects are re-exported from :mod:`OpenPinch.lib` for convenience so downstream
code can construct validated inputs.
"""

from .classes import PinchProblem
from .main import get_targets, get_visualise, pinch_analysis_service
from .analysis.stream_linearisation import get_piecewise_linearisation_for_streams
from .lib import *  # noqa: F401,F403 - re-export schema/config/enum types

__all__ = [
    "PinchProblem",
    "pinch_analysis_service",
    "get_targets",
    "get_visualise",
    "get_piecewise_linearisation_for_streams",
]
