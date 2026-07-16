"""Deterministic workspace variant comparison assembly."""

from __future__ import annotations

from ...lib.schemas.workspace import ScenarioComparisonView, ScenarioVariantView
from .views import problem_table_diffs, summary_metric_deltas


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
