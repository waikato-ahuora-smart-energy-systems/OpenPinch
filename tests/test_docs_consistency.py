"""Consistency checks for the user-facing documentation surface."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
GETTING_STARTED = REPO_ROOT / "docs" / "getting-started.rst"
QUICKSTART = REPO_ROOT / "docs" / "user-guide" / "quickstart.rst"
NOTEBOOKS = REPO_ROOT / "docs" / "user-guide" / "notebooks.rst"
HEAT_PUMP_TARGETING = REPO_ROOT / "docs" / "user-guide" / "heat-pump-targeting.rst"
INTERPRETING_RESULTS = REPO_ROOT / "docs" / "user-guide" / "interpreting-results.rst"
API_CLASSES = REPO_ROOT / "docs" / "reference" / "api-classes.rst"
API_LIB = REPO_ROOT / "docs" / "reference" / "api-lib.rst"
API_ANALYSIS = REPO_ROOT / "docs" / "reference" / "api-analysis.rst"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_highlights_current_cli_workflow():
    text = _read(README)
    assert "openpinch sample" in text
    assert "openpinch run" in text
    assert "openpinch graph" in text
    assert "openpinch heat-pump" in text
    assert "openpinch validate" in text
    assert "openpinch notebook" in text


def test_docs_highlight_current_pinchproblem_methods():
    combined = "\n".join(
        [
            _read(README),
            _read(GETTING_STARTED),
            _read(QUICKSTART),
            _read(NOTEBOOKS),
            _read(HEAT_PUMP_TARGETING),
            _read(INTERPRETING_RESULTS),
        ]
    )
    assert "run()" in combined
    assert "summary_frame()" in combined
    assert "export_excel" in combined
    assert "show_dashboard()" in combined
    assert "plot_grand_composite_curve" in combined
    assert "evaluate_heat_pump_integration" in combined


def test_docs_do_not_reference_stale_workflow_names():
    combined = "\n".join(
        [
            _read(README),
            _read(GETTING_STARTED),
            _read(QUICKSTART),
            _read(NOTEBOOKS),
        ]
    )
    assert "problem.export(" not in combined
    assert "Python 3.11 or newer" not in combined


def test_docs_highlight_interpretation_and_heat_pump_integration():
    combined = "\n".join(
        [
            _read(README),
            _read(QUICKSTART),
            _read(NOTEBOOKS),
            _read(HEAT_PUMP_TARGETING),
            _read(INTERPRETING_RESULTS),
        ]
    )
    assert "Interpreting Results" in combined
    assert "heat-pump targeting and integration" in combined
    assert "heat_pump_targeting.json" in combined


def test_reference_docs_match_current_heat_pump_and_schema_surface():
    api_classes = _read(API_CLASSES)
    api_lib = _read(API_LIB)
    api_analysis = _read(API_ANALYSIS)

    assert "OpenPinch.classes.vapour_compression_cycle" in api_classes
    assert "OpenPinch.classes.parallel_vapour_compression_cycles" in api_classes
    assert "OpenPinch.classes.cascade_vapour_compression_cycle" in api_classes
    assert "OpenPinch.classes.simple_heat_pump" not in api_classes
    assert "OpenPinch.classes.multi_simple_heat_pump" not in api_classes
    assert "OpenPinch.classes.cascade_heat_pump" not in api_classes
    assert "state_ids" in api_classes
    assert "weights" in api_classes

    assert "HeatPumpIntegrationScenario" in api_lib
    assert "HeatPumpIntegrationComparison" in api_lib

    assert "get_direct_heat_pump_and_refrigeration_target" in api_analysis
    assert "get_indirect_heat_pump_and_refrigeration_target" in api_analysis
