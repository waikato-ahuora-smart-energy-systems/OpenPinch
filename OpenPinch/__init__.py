"""OpenPinch public API surface.

The package exposes a small set of high-level helpers for running Pinch Analysis
and working with the structured results.  Detailed configuration and schema
objects are re-exported from :mod:`OpenPinch.lib` for convenience so downstream
code can construct validated inputs.
"""

import warnings

# Python 3.14 emits SyntaxWarning for legacy escape sequences in CoolProp docstrings.
# Keep warnings visible elsewhere, but silence this known third-party noise.
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"^CoolProp\.Plots\.SimpleCycles(Expansion|Compression)$",
)

from .classes import PinchProblem
from .lib import *
from .main import pinch_analysis_service, get_targets, get_visualise, extract_results
from .utils.stream_linearisation import get_piecewise_linearisation_for_streams
