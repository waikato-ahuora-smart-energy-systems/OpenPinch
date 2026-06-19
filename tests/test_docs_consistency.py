"""Consistency checks for the user-facing documentation surface."""

from __future__ import annotations

from pathlib import Path

from OpenPinch.resources import list_notebooks

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
GETTING_STARTED = REPO_ROOT / "docs" / "getting-started.rst"
QUICKSTART = REPO_ROOT / "docs" / "user-guide" / "quickstart.rst"
NOTEBOOKS = REPO_ROOT / "docs" / "user-guide" / "notebooks.rst"
HEAT_PUMP_TARGETING = REPO_ROOT / "docs" / "user-guide" / "heat-pump-targeting.rst"
INTERPRETING_RESULTS = REPO_ROOT / "docs" / "user-guide" / "interpreting-results.rst"
GUIDE_FIRST_SOLVE_CLI = REPO_ROOT / "docs" / "guides" / "first-solve-cli.rst"
GUIDE_NOTEBOOKS_AND_CASES = (
    REPO_ROOT / "docs" / "guides" / "notebooks-and-sample-cases.rst"
)
GUIDE_EXPORTING_RESULTS = REPO_ROOT / "docs" / "guides" / "exporting-results.rst"
GUIDE_HEN_SYNTHESIS = (
    REPO_ROOT / "docs" / "guides" / "heat-exchanger-network-synthesis.rst"
)
GUIDE_GRAPHING = REPO_ROOT / "docs" / "guides" / "graphing-and-interpretation.rst"
FUNDAMENTALS_GRAPHS = (
    REPO_ROOT / "docs" / "fundamentals" / "graphs-and-interpretation.rst"
)
EXAMPLE_NOTEBOOK_SERIES = REPO_ROOT / "docs" / "examples" / "notebook-series.rst"
EXAMPLE_SAMPLE_CASES = REPO_ROOT / "docs" / "examples" / "sample-cases.rst"
OVERVIEW_CAPABILITY_MATRIX = REPO_ROOT / "docs" / "overview" / "capability-matrix.rst"
OVERVIEW_SUPPORT = REPO_ROOT / "docs" / "overview" / "support-and-stability.rst"
API_PINCHWORKSPACE = REPO_ROOT / "docs" / "api" / "pinchworkspace.rst"
API_PACKAGE_ROOT = REPO_ROOT / "docs" / "api" / "package-root.rst"
API_CLI_RESOURCES = REPO_ROOT / "docs" / "api" / "cli-and-resources.rst"
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
    assert 'state_id="peak"' in combined or 'state_id="winter"' in combined


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
            _read(OVERVIEW_SUPPORT),
            _read(API_CLI_RESOURCES),
        ]
    )
    assert "problem.export(" not in combined
    assert "openpinch graph" not in combined
    assert "openpinch run" not in combined
    assert "openpinch validate" not in combined
    assert "openpinch sample" not in combined
    assert "Python 3.11 or newer" not in combined
    assert "openpinch notebook -o notebooks" in combined


def test_support_and_cli_docs_match_the_current_command_surface():
    combined = "\n".join([_read(OVERVIEW_SUPPORT), _read(API_CLI_RESOURCES)])

    assert "openpinch notebook" in combined
    assert "``notebook`` for copying packaged example notebooks" in combined
    assert "``run`` for end-to-end analysis" not in combined
    assert "``graph`` for HTML graph export" not in combined
    assert "``validate`` for payload preflight checks" not in combined
    assert "``sample`` for copying packaged sample cases" not in combined
    assert "``heat-pump`` for evaluating" not in combined


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


def test_docs_reference_the_current_notebook_series():
    combined = "\n".join(
        [
            _read(README),
            _read(QUICKSTART),
            _read(NOTEBOOKS),
            _read(GUIDE_FIRST_SOLVE_CLI),
            _read(GUIDE_NOTEBOOKS_AND_CASES),
            _read(EXAMPLE_NOTEBOOK_SERIES),
        ]
    )
    assert "01_basic_pinch_and_dtcont_sensitivity.ipynb" in combined
    assert "02_total_site_targets_and_sugcc.ipynb" in combined
    assert "03_carnot_hpr_comparison.ipynb" in combined
    assert "04_multistate_targeting_and_state_comparison.ipynb" in combined
    assert "05_schema_service_and_output_workflows.ipynb" in combined
    assert "06_energy_transfer_analysis.ipynb" in combined
    assert "07_vapour_compression_mvr_cascade_hpr.ipynb" in combined
    assert "08_direct_gas_stream_mvr.ipynb" in combined
    for notebook_name in list_notebooks():
        assert notebook_name in combined
    assert "multistate" in combined or 'state_id="' in combined
    assert "direct gas/vapour" in combined
    assert "add_component.process_mvr" in combined


def test_docs_reference_current_packaged_sample_cases():
    combined = "\n".join(
        [
            _read(GUIDE_NOTEBOOKS_AND_CASES),
            _read(EXAMPLE_SAMPLE_CASES),
        ]
    )

    assert "basic_pinch.json" in combined
    assert "crude_preheat_train.json" in combined
    assert "crude_preheat_train_multistate.json" in combined
    assert "zonal_site.json" in combined
    assert "zonal_site_multistate.json" in combined
    assert "pulp_mill.json" in combined
    assert "heat_pump_targeting.json" in combined
    assert "chocolate_factory.json" in combined


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


def test_docs_explain_hen_synthesis_public_workflow_and_cutover():
    guide = _read(GUIDE_HEN_SYNTHESIS)

    assert "problem.design.heat_exchanger_network_synthesis()" in guide
    assert 'workflow="heat_exchanger_network_synthesis"' in guide
    assert "heat_exchanger_network_synthesis_service(problem)" in guide
    assert "internal and" in guide
    assert "TargetInput.options" in guide
    assert "``Configuration``" in guide
    assert "TargetOutput.design" in guide
    assert "export_heat_exchanger_network_synthesis_results" in guide
    assert "CaseStudy.from_csv" in guide
    assert "OpenPinch does not provide runtime import" in guide
    assert "aliases, OpenHENS field aliases" in guide
    assert "OpenHENS field aliases" in guide
    assert "command parity" in guide
    assert "Source OpenHENS CSV files are migration" in guide
    assert "source material only" in guide
    assert 'python -m pip install "openpinch[synthesis]"' in guide
    assert 'pytest -m "not synthesis and not solver"' in guide
    assert "pytest -m synthesis" in guide
    assert "pytest -m solver" in guide


def test_reference_docs_match_current_heat_pump_and_schema_surface():
    api_classes = _read(API_CLASSES)
    api_lib = _read(API_LIB)
    api_analysis = _read(API_ANALYSIS)

    assert (
        "OpenPinch.services.heat_pump_integration.unit_models.vapour_compression_cycle"
        in api_classes
    )
    assert (
        "OpenPinch.services.heat_pump_integration.unit_models.parallel_vapour_compression_cycles"
        in api_classes
    )
    assert (
        "OpenPinch.services.heat_pump_integration.unit_models.cascade_vapour_compression_cycle"
        in api_classes
    )
    assert (
        "OpenPinch.services.heat_pump_integration.unit_models.mechanical_vapour_recompression_cycle"
        in api_classes
    )
    assert (
        "OpenPinch.services.heat_pump_integration.unit_models.vapour_compression_mvr_cascade"
        in api_classes
    )
    assert (
        "OpenPinch.services.heat_pump_integration.unit_models.brayton_heat_pump"
        in api_classes
    )
    assert (
        "OpenPinch.services.power_cogeneration.unit_models.multi_stage_steam_turbine"
        in api_classes
    )
    assert "OpenPinch.services.heat_pump_integration.targeting_services" in _read(
        REPO_ROOT / "docs" / "reference" / "api-heat-pump.rst"
    )
    assert "OpenPinch.services.components.process_mvr" in api_classes
    assert "OpenPinch.services.components.direct_mvr.direct_gas_mvr" in api_classes
    assert "OpenPinch.classes.simple_heat_pump" not in api_classes
    assert "OpenPinch.classes.parallel_heat_pump" not in api_classes
    assert "OpenPinch.classes.cascade_heat_pump" not in api_classes
    assert "OpenPinch.classes.brayton_heat_pump" not in api_classes
    assert "OpenPinch.classes.multi_stage_steam_turbine" not in api_classes
    assert "state_ids" in api_classes
    assert "weights" in api_classes
    assert "OpenPinch.classes.heat_exchanger" in api_classes
    assert "OpenPinch.classes.heat_exchanger_network" in api_classes

    assert "compute_direct_heat_pump_or_refrigeration_target" in api_analysis
    assert "compute_indirect_heat_pump_or_refrigeration_target" in api_analysis
    assert "HeatPumpTargetOutputs" in api_lib
    assert "OpenPinch.lib.schemas.synthesis" in api_lib
    assert "OpenPinch.services.input_data_processing.data_preparation" in api_analysis
    assert "OpenPinch.services.data_preparation" not in api_analysis


def test_reference_docs_mark_partial_analysis_modules_as_partial():
    api_analysis = _read(API_ANALYSIS)

    assert "Experimental or Partial Analysis Modules" in api_analysis
    assert "they should not be read as stable production workflows" in api_analysis
    assert "OpenPinch.services.exergy_analysis.exergy_targeting_entry" in api_analysis


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
