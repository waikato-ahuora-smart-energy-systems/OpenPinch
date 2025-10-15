"""OpenPinch public API surface.

The package exposes a small set of high-level helpers for running Pinch Analysis
and working with the structured results.  Detailed configuration and schema
objects are re-exported from :mod:`OpenPinch.lib` for convenience so downstream
code can construct validated inputs.
"""

from .classes import PinchProblem
from .lib import *
from .main import pinch_analysis_service
from .scales.entry import get_targets, get_visualise
from .utils.stream_linearisation import get_piecewise_linearisation_for_streams

__all__ = [
    "PinchProblem",
    "pinch_analysis_service",
    "get_targets",
    "get_visualise",
    "get_piecewise_linearisation_for_streams",
]
