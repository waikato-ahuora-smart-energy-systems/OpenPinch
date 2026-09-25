"""Properties for pure helpers embedded in generated HPR notebooks."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from OpenPinch.contracts.hpr import (
    HPRFailureCategory,
    HPRFailureDiagnostic,
    HPRFailureSummary,
    HPRSearchBudget,
    HPRTargetingError,
)

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "OpenPinch" / "tutorials" / "notebooks"


def _notebook_functions(notebook_name: str, function_names: tuple[str, ...]):
    notebook = json.loads((NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source, filename=notebook_name)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {node.name for node in functions} == set(function_names), (
        notebook_name,
        function_names,
    )
    namespace: dict[str, object] = {"HPRTargetingError": HPRTargetingError}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            notebook_name,
            "exec",
        ),
        namespace,
    )
    return namespace


def _notebook_function(notebook_name: str, function_name: str):
    return _notebook_functions(notebook_name, (function_name,))[function_name]


@st.composite
def _failure_summaries(draw) -> HPRFailureSummary:
    category = draw(st.sampled_from(list(HPRFailureCategory)))
    diagnostic_count = draw(st.integers(min_value=0, max_value=16))
    diagnostic = HPRFailureDiagnostic(
        category=category,
        reason_code=draw(st.from_regex(r"[a-z][a-z0-9_.-]{0,31}", fullmatch=True)),
        summary=draw(
            st.text(
                alphabet=st.characters(
                    blacklist_categories=("Cs",),
                    blacklist_characters="\r\n",
                ),
                min_size=1,
                max_size=64,
            )
        ),
        fluid=draw(st.one_of(st.none(), st.sampled_from(["Water", "Ammonia"]))),
        period_id=draw(
            st.one_of(st.none(), st.sampled_from(["turndown", "base", "peak"]))
        ),
    )
    evaluated_count = draw(st.integers(min_value=diagnostic_count, max_value=500))
    return HPRFailureSummary(
        simulation_backend=draw(st.sampled_from(["coolprop", "tespy"])),
        cycle=draw(st.sampled_from(["single-stage", "cascade", "vc-mvr"])),
        evaluated_count=evaluated_count,
        category_counts={category: evaluated_count},
        representative_failures=(diagnostic,) * diagnostic_count,
        budget=HPRSearchBudget(
            maximum_iterations=draw(st.integers(min_value=1, max_value=20)),
            maximum_evaluations=draw(st.integers(min_value=1, max_value=50)),
        ),
        warm_start_evaluated=False,
        warm_start_viable=False,
    )


@seed(20260715)
@given(summary=_failure_summaries())
def test_notebook_failure_summary_is_bounded_plain_json(
    summary: HPRFailureSummary,
) -> None:
    summarize = _notebook_function(
        "10_multiperiod_heat_pumps.ipynb", "summarize_hpr_failure"
    )
    result = summarize(HPRTargetingError("bounded failure", diagnostics=summary))
    round_trip = json.loads(json.dumps(result))

    assert round_trip == result
    assert result["status"] == "typed infeasible"
    assert result["reason"] == "bounded failure"
    assert result["diagnostics"] == summary.model_dump(mode="json")
    assert len(result["diagnostics"]["representative_failures"]) <= 16


@pytest.mark.parametrize(
    "notebook_name",
    [
        "08_carnot_heat_pump_and_refrigeration.ipynb",
        "09_vapour_compression_and_brayton.ipynb",
        "10_multiperiod_heat_pumps.ipynb",
        "11_process_mvr_and_cascade.ipynb",
    ],
)
@seed(20260715)
@given(
    period_ids=st.permutations(("turndown", "base", "peak")),
    design_vector=st.lists(
        st.floats(
            min_value=-1.0e6,
            max_value=1.0e6,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=8,
    ),
    selected=st.floats(
        min_value=0.0,
        max_value=1.0e6,
        allow_nan=False,
        allow_infinity=False,
    ),
    achieved=st.floats(
        min_value=0.0,
        max_value=1.0e6,
        allow_nan=False,
        allow_infinity=False,
    ),
    objective=st.floats(
        min_value=-1.0e12,
        max_value=1.0e12,
        allow_nan=False,
        allow_infinity=False,
    ),
    loop_count=st.integers(min_value=0, max_value=8),
)
def test_notebook_target_summaries_are_finite_plain_and_preserve_periods(
    notebook_name: str,
    period_ids: list[str],
    design_vector: list[float],
    selected: float,
    achieved: float,
    objective: float,
    loop_count: int,
) -> None:
    summarize = _notebook_function(notebook_name, "summarize_hpr_target")
    weights = [1.0 / len(period_ids)] * len(period_ids)
    target = SimpleNamespace(
        hpr_cycle="single-stage",
        hpr_load=SimpleNamespace(selected=selected, achieved=achieved),
        hpr_details=SimpleNamespace(
            simulation_backend="coolprop",
            target_simulation_record=SimpleNamespace(loops=[None] * loop_count),
            design_vector=design_vector,
            period_ids=period_ids,
            period_weights=weights,
            obj=objective,
        ),
    )

    result = summarize("generated target", target)

    assert json.loads(json.dumps(result)) == result
    assert result["status"] == "feasible"
    assert result["period_ids"] == period_ids
    assert set(result["period_ids"]) == {"turndown", "base", "peak"}
    assert result["design_vector"] == design_vector
    assert result["period_weights"] == weights
    assert result["loop_count"] == loop_count
    assert all(
        math.isfinite(value)
        for value in (
            result["selected_load"],
            result["achieved_load"],
            result["objective"],
            *result["design_vector"],
            *result["period_weights"],
        )
    )


def test_notebook_09_optional_status_vocabulary_is_distinct() -> None:
    namespace = _notebook_functions(
        "09_vapour_compression_and_brayton.ipynb",
        (
            "summarize_hpr_target",
            "summarize_hpr_failure",
            "screen_optional_hpr",
        ),
    )
    screen = namespace["screen_optional_hpr"]

    def missing_dependency(**_arguments):
        raise ImportError("TESPy is not installed")

    def unavailable_method(**_arguments):
        raise NotImplementedError("Brayton is unavailable")

    def typed_infeasible(**_arguments):
        raise HPRTargetingError(
            "bounded failure",
            diagnostics=HPRFailureSummary(
                simulation_backend="coolprop",
                cycle="single-stage",
                evaluated_count=0,
                category_counts={},
                representative_failures=(),
                budget=HPRSearchBudget(
                    maximum_iterations=1,
                    maximum_evaluations=1,
                ),
                warm_start_evaluated=False,
                warm_start_viable=False,
            ),
        )

    results = [
        screen(missing_dependency),
        screen(unavailable_method),
        screen(typed_infeasible),
    ]

    assert [result["status"] for result in results] == [
        "optional dependency unavailable",
        "method unavailable",
        "typed infeasible",
    ]
    assert all(json.loads(json.dumps(result)) == result for result in results)
