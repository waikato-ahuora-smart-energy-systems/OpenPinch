"""Private helpers for the :mod:`OpenPinch.classes.pinch_problem` surface."""

from ._component_accessor import _ComponentAccessorDescriptor
from ._design_accessor import _DesignAccessorDescriptor
from ._loading import (
    JsonDict,
    PathLike,
    _LoadedProblemSource,
    _ProblemSourceAdapters,
    load_problem_source,
    prepare_in_memory_problem_source,
)
from ._output import (
    build_graph_payload,
    build_problem_summary_frame,
    locate_summary_row,
)
from ._plot_accessor import _PlotAccessorDescriptor
from ._result_extraction import extract_results
from ._target_accessor import _TargetAccessorDescriptor
from ._target_dispatch import run_targeting_for_zone_and_subzones
from ._validation import (
    _validate_problem_semantics,
    format_schema_validation_error,
)

__all__ = [
    "JsonDict",
    "_LoadedProblemSource",
    "_DesignAccessorDescriptor",
    "PathLike",
    "_ProblemSourceAdapters",
    "_ComponentAccessorDescriptor",
    "_PlotAccessorDescriptor",
    "_TargetAccessorDescriptor",
    "_validate_problem_semantics",
    "build_graph_payload",
    "build_problem_summary_frame",
    "extract_results",
    "format_schema_validation_error",
    "load_problem_source",
    "locate_summary_row",
    "prepare_in_memory_problem_source",
    "run_targeting_for_zone_and_subzones",
]
