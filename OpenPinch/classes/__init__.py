"""Core domain classes used throughout OpenPinch.

The objects re-exported here model streams, zones, target results, and helper
containers that compose the Pinch Analysis workflow.  They form the backbone of
the public API and are designed to be interoperable with the higher-level
services in :mod:`OpenPinch.main`.
"""

from .stream import Stream
from .zone import Zone
from .target import EnergyTarget
from .value import Value
from .stream_collection import StreamCollection
from .problem_table import ProblemTable
from .pinch_problem import PinchProblem
