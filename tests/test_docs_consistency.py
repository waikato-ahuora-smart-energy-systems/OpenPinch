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
GUIDE_FIRST_SOLVE_CLI = REPO_ROOT / "docs" / "guides" / "first-solve-cli.rst"
GUIDE_EXPORTING_RESULTS = REPO_ROOT / "docs" / "guides" / "exporting-results.rst"
GUIDE_GRAPHING = REPO_ROOT / "docs" / "guides" / "graphing-and-interpretation.rst"
FUNDAMENTALS_GRAPHS = REPO_ROOT / "docs" / "fundamentals" / "graphs-and-interpretation.rst"
OVERVIEW_CAPABILITY_MATRIX = REPO_ROOT / "docs" / "overview" / "capability-matrix.rst"
API_PINCHWORKSPACE = REPO_ROOT / "docs" / "api" / "pinchworkspace.rst"
API_PACKAGE_ROOT = REPO_ROOT / "docs" / "api" / "package-root.rst"
API_CLASSES = REPO_ROOT / "docs" / "reference" / "api-classes.rst"
API_LIB = REPO_ROOT / "docs" / "reference" / "api-lib.rst"
API_ANALYSIS = REPO_ROOT / "docs" / "reference" / "api-analysis.rst"
REFERENCE_INDEX = REPO_ROOT / "docs" / "reference" / "index.rst"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    assert "target()" in combined
    assert "summary_frame()" in combined
    assert "export_excel" in combined
    assert "show_dashboard()" in combined
    assert "plot.grand_composite_curve" in combined
    assert "problem.target.direct_heat_pump" in combined


def test_docs_do_not_reference_stale_workflow_names():
    combined = "\n".join(
        [
            _read(README),
            _read(GETTING_STARTED),
            _read(QUICKSTART),
            _read(NOTEBOOKS),
            _read(GUIDE_FIRST_SOLVE_CLI),
            _read(GUIDE_EXPORTING_RESULTS),
            _read(GUIDE_GRAPHING),
            _read(FUNDAMENTALS_GRAPHS),
            _read(OVERVIEW_CAPABILITY_MATRIX),
        ]
    )
    assert "problem.export(" not in combined
    assert "openpinch graph" not in combined
    assert "openpinch run" not in combined
    assert "openpinch validate" not in combined
    assert "openpinch sample" not in combined
    assert "Python 3.11 or newer" not in combined
    assert "openpinch notebook -o notebooks" in combined


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
    assert "heat pump" in combined
    assert "heat_pump_targeting.json" in combined
    assert "03_carnot_hpr_comparison.ipynb" in combined


def test_docs_reference_the_current_three_notebook_series():
    combined = "\n".join([_read(README), _read(QUICKSTART), _read(NOTEBOOKS)])
    assert "01_basic_pinch_and_dtcont_sensitivity.ipynb" in combined
    assert "02_total_site_targets_and_sugcc.ipynb" in combined
    assert "03_carnot_hpr_comparison.ipynb" in combined
    assert "01_basic_pinch_analysis.ipynb" not in combined
    assert "06_target_services_workflow.ipynb" not in combined


def test_docs_explain_base_and_notebook_installs():
    readme = _read(README)
    getting_started = _read(GETTING_STARTED)
    quickstart = _read(QUICKSTART)
    notebooks = _read(NOTEBOOKS)

    assert "python -m pip install openpinch" in readme
    assert 'python -m pip install "openpinch[notebook]"' in readme

    assert "python -m pip install openpinch" in getting_started
    assert 'python -m pip install "openpinch[notebook]"' in getting_started

    notebook_guides = "\n".join([readme, getting_started, quickstart, notebooks])
    assert notebook_guides.count('python -m pip install "openpinch[notebook]"') >= 4
    assert 'python -m pip install "openpinch[dashboard]"' in readme
    assert 'python -m pip install "openpinch[dashboard]"' in getting_started
    assert 'python -m pip install "openpinch[dashboard]"' in quickstart
    assert 'python -m pip install "openpinch[brayton_cycle]"' in readme
    assert 'python -m pip install "openpinch[brayton_cycle]"' in getting_started
    assert "Optional: Jupyter" not in getting_started


def test_docs_expose_pinchworkspace_as_the_named_study_surface():
    combined = "\n".join(
        [
            _read(README),
            _read(GETTING_STARTED),
            _read(API_PACKAGE_ROOT),
            _read(API_PINCHWORKSPACE),
        ]
    )

    assert "PinchWorkspace" in combined
    assert 'source="crude_preheat_train.json"' in combined
    assert 'project_name="crude_preheat_train"' in combined
    assert "compare_cases" in combined


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

    assert "compute_direct_heat_pump_or_refrigeration_target" in api_analysis
    assert "compute_indirect_heat_pump_or_refrigeration_target" in api_analysis
    assert "HeatPumpTargetOutputs" in api_lib
    assert "OpenPinch.services.input_data_processing.data_preparation" in api_analysis
    assert "OpenPinch.services.data_preparation" not in api_analysis


def test_reference_docs_show_uv_docs_build_command():
    reference_index = _read(REFERENCE_INDEX)

    assert "uv run scripts/build_docs.py" in reference_index
    assert "sphinx-build -b html docs/ docs/_build/html" not in reference_index


def test_reference_docs_use_current_data_preparation_module_path():
    combined = "\n".join(
        [
            _read(REPO_ROOT / "docs" / "reference" / "api.rst"),
            _read(REPO_ROOT / "docs" / "reference" / "architecture.rst"),
            _read(API_CLASSES),
            _read(API_ANALYSIS),
        ]
    )

    assert "OpenPinch.services.input_data_processing.data_preparation" in combined
    assert "OpenPinch.services.data_preparation" not in combined
