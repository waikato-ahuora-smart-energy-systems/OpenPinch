"""Workspace problem-table view shaping."""

from __future__ import annotations

from ....contracts.workspace import ProblemTableView
from ....presentation.reporting.problem_table import problem_table_frame
from ...problem import PinchProblem
from .serialization import dataframe_to_table_view


def problem_table_views(problem: PinchProblem) -> list[ProblemTableView]:
    """Build serializable problem-table views for one solved problem."""
    if problem.master_zone is None:
        return []

    tables = []
    targets = _collect_targets(problem.master_zone)
    for target_name in sorted(targets):
        target = targets[target_name]
        for table_kind, attr_name in (("shifted", "pt"), ("real", "pt_real")):
            frame = problem_table_frame(
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


def _collect_targets(zone) -> dict[str, object]:
    targets = {target.name: target for target in zone.targets.values()}
    for subzone in zone.subzones.values():
        targets.update(_collect_targets(subzone))
    return targets
