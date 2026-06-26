"""OpenPinch public API."""

from __future__ import annotations

import warnings as _warnings

from . import lib as lib
from .classes.pinch_problem import PinchProblem
from .classes.pinch_workspace import PinchWorkspace
from .lib.config import Configuration
from .lib.enums import GraphType, HPRcycle, StreamType, TargetType, ZoneType
from .lib.schemas.io import (
    StreamSchema,
    TargetInput,
    TargetOutput,
    UtilitySchema,
    ZoneTreeSchema,
)
from .lib.schemas.reporting import GraphAvailability, ProblemReport, ReportMetric
from .lib.schemas.workspace import ValidationIssue, ValidationReport
from .main import pinch_analysis_service
from .resources import (
    NotebookMetadata,
    SampleCaseMetadata,
    copy_notebook,
    copy_sample_case,
    list_notebooks,
    list_sample_cases,
    notebook_metadata,
    read_sample_case,
    sample_case_metadata,
)
from .utils.stream_linearisation import get_piecewise_linearisation_for_streams

_warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"^CoolProp\.Plots\.SimpleCycles(Expansion|Compression)$",
)


def config_options():
    """Return metadata for supported flat ``TargetInput.options`` keys."""
    return Configuration.options_catalog()


__all__ = [
    "Configuration",
    "GraphAvailability",
    "GraphType",
    "HPRcycle",
    "NotebookMetadata",
    "PinchProblem",
    "PinchWorkspace",
    "ProblemReport",
    "ReportMetric",
    "SampleCaseMetadata",
    "StreamSchema",
    "StreamType",
    "TargetInput",
    "TargetOutput",
    "TargetType",
    "UtilitySchema",
    "ValidationIssue",
    "ValidationReport",
    "ZoneTreeSchema",
    "ZoneType",
    "config_options",
    "copy_notebook",
    "copy_sample_case",
    "get_piecewise_linearisation_for_streams",
    "list_notebooks",
    "list_sample_cases",
    "notebook_metadata",
    "pinch_analysis_service",
    "read_sample_case",
    "sample_case_metadata",
]
