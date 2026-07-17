"""Deterministic workspace variant comparison assembly."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ...contracts.workspace import ScenarioComparisonView, ScenarioVariantView
from .views import problem_table_diffs, summary_metric_deltas

if TYPE_CHECKING:
    from ..workspace import PinchWorkspace


def compare_workspace_variants(
    workspace: "PinchWorkspace",
    variant_names: Iterable[str] | None,
    *,
    base: str | None,
) -> ScenarioComparisonView:
    """Resolve, solve, and compare a stable ordered set of variants."""
    names = list(variant_names or workspace.list_variants())
    if not names:
        raise ValueError("At least one variant is required for comparison.")
    base_name = base or workspace.baseline_name
    if base_name not in names:
        names.insert(0, base_name)
    views = {name: workspace._ensure_solved_view(name) for name in names}
    return build_variant_comparison(
        names=names,
        base_name=base_name,
        views=views,
    )


def build_variant_comparison(
    *,
    names: list[str],
    base_name: str,
    views: dict[str, ScenarioVariantView],
) -> ScenarioComparisonView:
    """Build one comparison from already solved, ordered variant views."""
    base_view = views[base_name]
    metric_deltas = []
    problem_diffs = []
    for name in names:
        if name == base_name:
            continue
        metric_deltas.extend(
            summary_metric_deltas(base_name, base_view, name, views[name])
        )
        problem_diffs.extend(
            problem_table_diffs(base_name, base_view, name, views[name])
        )

    return ScenarioComparisonView(
        base_variant=base_name,
        variant_names=names,
        metric_deltas=metric_deltas,
        graph_catalogs={name: views[name].graph_catalog for name in names},
        problem_table_diffs=problem_diffs,
    )
