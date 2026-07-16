"""Workspace problem-table view shaping."""

from __future__ import annotations

from ....lib.schemas.workspace import ProblemTableView
from ....streamlit_webviewer.web_graphing import (
    collect_targets,
    problem_table_to_dataframe,
)
from ...pinch_problem import PinchProblem
from .common import dataframe_to_table_view


def problem_table_views(problem: PinchProblem) -> list[ProblemTableView]:
    """Build serializable problem-table views for one solved problem."""
    if problem.master_zone is None:
        return []

    tables = []
    targets = collect_targets(problem.master_zone)
    for target_name in sorted(targets):
        target = targets[target_name]
        for table_kind, attr_name in (("shifted", "pt"), ("real", "pt_real")):
            frame = problem_table_to_dataframe(
                getattr(target, attr_name, None),
                round_decimals=2,
            )
            if frame.empty:
                continue
            tables.append(
                ProblemTableView(
                    table_id=f"{target_name}::{table_kind}",
                    target_id=target_name,
                    target_name=target_name,
                    table_kind=table_kind,
                    table=dataframe_to_table_view(frame),
                )
            )
    return tables
