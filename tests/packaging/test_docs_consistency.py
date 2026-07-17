"""Structural checks for the user-facing documentation surface."""

from __future__ import annotations

from pathlib import Path

import OpenPinch
from OpenPinch.resources import list_notebooks, list_sample_cases
from tests.support.paths import REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
DOCS_ROOT = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"
RTD_CONFIG = REPO_ROOT / ".readthedocs.yaml"

GUIDES_ROOT = DOCS_ROOT / "guides"
API_ROOT = DOCS_ROOT / "api"
EXAMPLES_ROOT = DOCS_ROOT / "examples"

STALE_DOC_STRINGS = (
    "openpinch run",
    "openpinch graph",
    "openpinch validate",
    "openpinch sample",
    "``run`` for end-to-end analysis",
    "``graph`` for HTML graph export",
    "``validate`` for payload preflight checks",
    "``sample`` for copying packaged sample cases",
)

GUIDE_REQUIRED_HEADINGS = (
    "Purpose\n-------",
    "Prerequisites\n-------------",
    "Runnable Workflow\n-----------------",
    "Expected Output\n---------------",
    "Interpretation\n--------------",
    "Next Steps\n----------",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _docs_text(paths: list[Path] | None = None) -> str:
    if paths is None:
        paths = [*DOCS_ROOT.rglob("*.rst"), README]
    return "\n".join(_read(path) for path in paths)


def test_rtd_configuration_keeps_strict_python314_build():
    config = _read(RTD_CONFIG)

    assert 'python: "3.14"' in config
    assert "configuration: docs/conf.py" in config
    assert "fail_on_warning: true" in config
    assert "requirements: docs/requirements.txt" in config


def test_top_level_navigation_uses_user_first_information_architecture():
    index = _read(DOCS_ROOT / "index.rst")

    for phrase in (
        "I want to solve a case",
        "I need to understand the method",
        "I am building a reusable study",
        "getting-started",
        "overview/index",
        "fundamentals/index",
        "guides/index",
        "api/index",
        "examples/index",
        "developer/index",
    ):
        assert phrase in index

    assert index.index("getting-started") < index.index("overview/index")


def test_getting_started_is_canonical_first_run_page():
    page = _read(DOCS_ROOT / "getting-started.rst")

    assert ":orphan:" not in page
    assert "from OpenPinch import PinchProblem" in page
    assert "problem.target.all_heat_integration()" in page
    assert "PinchWorkspace" in page
    assert "Observation" in page or "observe" in page


def test_guides_use_the_public_root_workflow_and_current_vocabulary():
    guide_pages = sorted(
        path for path in GUIDES_ROOT.glob("*.rst") if path.name != "index.rst"
    )

    assert guide_pages
    for path in guide_pages:
        text = _read(path)
        assert "from OpenPinch.application" not in text
        assert "pinch_analysis_service" not in text
        assert "problem.target()" not in text
        assert "problem.add_component" not in text


def test_packaged_assets_are_documented_in_examples_and_guides():
    combined = _docs_text(
        [
            EXAMPLES_ROOT / "notebook-series.rst",
            EXAMPLES_ROOT / "sample-cases.rst",
            GUIDES_ROOT / "notebooks-and-sample-cases.rst",
            GUIDES_ROOT / "first-solve-cli.rst",
            README,
        ]
    )

    for notebook_name in list_notebooks():
        assert notebook_name in combined
    for sample_case_name in list_sample_cases():
        assert sample_case_name in combined


def test_notebook_docs_keep_numbered_order_and_coverage_link():
    text = _read(EXAMPLES_ROOT / "notebook-series.rst")

    last_position = -1
    for notebook_name in list_notebooks():
        position = text.index(notebook_name)
        assert position > last_position, (
            f"notebook series has {notebook_name} out of order"
        )
        last_position = position
    assert "tutorial-coverage-map" in text


def test_public_python_surface_is_covered_by_curated_api_docs():
    combined_api = _docs_text(sorted(API_ROOT.glob("*.rst")))
    contract_page = _read(API_ROOT / "package-root.rst")

    assert OpenPinch.__all__ == ["PinchProblem", "PinchWorkspace"]
    assert "pinch_analysis_service" not in combined_api
    assert "from OpenPinch import PinchProblem, PinchWorkspace" in contract_page
    assert "process-engineer workflows begin" in contract_page.lower()
    assert "tutorial-coverage-map" in contract_page


def test_workflow_guides_do_not_recommend_concrete_application_imports():
    combined_guides = _docs_text(sorted(GUIDES_ROOT.glob("*.rst")))

    assert "from OpenPinch.application.problem import" not in combined_guides
    assert "from OpenPinch.application.workspace import" not in combined_guides


def test_architecture_docs_define_owner_directions_and_shared_optimisation():
    page = _read(DOCS_ROOT / "developer" / "architecture.rst")

    for owner in (
        "``domain``",
        "``contracts``",
        "``optimisation``",
        "``adapters``",
        "``analysis``",
        "``application``",
        "``presentation``",
    ):
        assert owner in page
    assert "Dependencies point toward domain and contracts" in page
    assert "Other services can reuse" in page
    assert "without importing any HPR module" in page
    assert "Hypothesis with seed ``20260715``" in page


def test_docs_define_stability_and_optional_dependency_boundaries():
    combined = _docs_text(
        [
            DOCS_ROOT / "overview" / "capability-matrix.rst",
            DOCS_ROOT / "overview" / "support-and-stability.rst",
            DOCS_ROOT / "developer" / "docs-conventions.rst",
            DOCS_ROOT / "developer" / "build-and-coverage.rst",
        ]
    )

    for phrase in (
        "Stable",
        "Advanced",
        "Experimental / partial",
        "openpinch[notebook]",
        "openpinch[dashboard]",
        "openpinch[synthesis]",
        "optional local audit",
    ):
        assert phrase in combined


def test_docs_do_not_present_removed_cli_surfaces():
    combined = _docs_text()

    for stale in STALE_DOC_STRINGS:
        assert stale not in combined

    assert "openpinch notebook" in combined


def test_hen_synthesis_docs_keep_internal_cutover_and_dependency_notes():
    guide = _read(GUIDES_ROOT / "heat-exchanger-network-synthesis.rst")
    service_layer = _read(API_ROOT / "service-layer.rst")
    schemas_config = _read(API_ROOT / "schemas-and-config.rst")

    for phrase in (
        'python -m pip install "openpinch[synthesis]"',
        "idaes get-extensions",
        "problem.design.enhanced_heat_exchanger_network(quality_tier=2)",
        "problem.design.open_hens()",
        "problem.design.network_evolution(",
        "TargetOutput.design",
        'design.result.model_dump(mode="json")',
        "design.selected_network",
        "pytest -m synthesis",
        "pytest -m solver",
    ):
        assert phrase in guide

    assert "OpenPinch.analysis.heat_exchanger_networks.service" in service_layer
    assert "HeatExchangerNetworkDesignMethod" in schemas_config


def test_segmented_stream_docs_cover_input_targeting_and_hen_contracts():
    input_guide = _read(GUIDES_ROOT / "input-formats-and-validation.rst")
    hen_guide = _read(GUIDES_ROOT / "heat-exchanger-network-synthesis.rst")
    domain_model = _read(API_ROOT / "domain-model.rst")
    capability_matrix = _read(DOCS_ROOT / "overview" / "capability-matrix.rst")
    normalized_input_guide = " ".join(input_guide.split())
    normalized_hen_guide = " ".join(hen_guide.split())

    for phrase in (
        "Variable Heat-Capacity Streams",
        "segments",
        "profile",
        "target temperature must equal the next segment supply temperature",
    ):
        assert phrase in normalized_input_guide

    for phrase in (
        "Segmented Variable-Heat-Capacity Streams",
        "one physical parent",
        "does not silently substitute an average parent ``CP``",
        "segment_area_contributions",
        "maximum period-total slice area",
        "Chen area surrogate",
        "continuous NLP path",
    ):
        assert phrase in normalized_hen_guide

    assert "Runtime segment record classes are private" in domain_model
    assert "Variable heat-capacity streams" in capability_matrix


def test_heat_pump_docs_keep_advanced_workflow_boundaries():
    guide = _read(GUIDES_ROOT / "heat-pump-workflows.rst")
    fundamentals = _read(
        DOCS_ROOT / "fundamentals" / "heat-pump-and-refrigeration-methods.rst"
    )

    for phrase in (
        "problem.target.carnot_heat_pump",
        "problem.target.vapour_compression_heat_pump",
        "process.components.add_process_mvr",
        "Cascade Carnot cycles",
        "Parallel vapour compression cycles",
        "Vapour compression with MVR cascade",
        "08_carnot_heat_pump_and_refrigeration.ipynb",
        "11_process_mvr_and_cascade.ipynb",
    ):
        assert phrase in guide

    assert "Simulated-cycle integration accounting" in fundamentals
