"""Structural checks for the user-facing documentation surface."""

from __future__ import annotations

from pathlib import Path

import OpenPinch
from OpenPinch.resources import list_notebooks, list_sample_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"
RTD_CONFIG = REPO_ROOT / ".readthedocs.yaml"

GUIDES_ROOT = DOCS_ROOT / "guides"
API_ROOT = DOCS_ROOT / "api"
EXAMPLES_ROOT = DOCS_ROOT / "examples"

LEGACY_PAGE_TARGETS = {
    DOCS_ROOT / "user-guide" / "quickstart.rst": (
        "../guides/first-solve-python",
        "../guides/notebooks-and-sample-cases",
    ),
    DOCS_ROOT / "user-guide" / "notebooks.rst": (
        "../guides/notebooks-and-sample-cases",
        "../examples/notebook-series",
    ),
    DOCS_ROOT / "user-guide" / "heat-pump-targeting.rst": (
        "../guides/heat-pump-workflows",
        "../fundamentals/heat-pump-and-refrigeration-methods",
    ),
    DOCS_ROOT / "user-guide" / "interpreting-results.rst": (
        "../guides/graphing-and-interpretation",
        "../fundamentals/graphs-and-interpretation",
    ),
    DOCS_ROOT / "reference" / "api.rst": (
        "../api/package-root",
        "../api/generated-index",
    ),
    DOCS_ROOT / "reference" / "architecture.rst": (
        "../overview/workflow-map",
        "../fundamentals/pinch-analysis",
    ),
    DOCS_ROOT / "reference" / "index.rst": (
        "../api/index",
        "../developer/index",
    ),
}

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
        "I am integrating or extending OpenPinch",
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
    assert "python -m pip install openpinch" in page
    assert 'PinchProblem("basic_pinch.json"' in page
    assert "PinchWorkspace" in page
    assert "openpinch notebook -o notebooks" in page
    assert "Use the CLI Only for Notebook Assets" in page


def test_guides_follow_the_standard_task_structure():
    guide_pages = sorted(
        path for path in GUIDES_ROOT.glob("*.rst") if path.name != "index.rst"
    )

    assert guide_pages
    for path in guide_pages:
        text = _read(path)
        for heading in GUIDE_REQUIRED_HEADINGS:
            assert heading in text, f"{path} is missing {heading!r}"
        assert (
            "Sample Case" in text or "Sample Asset" in text or "Sample Cases" in text
        ), f"{path} is missing a sample section"


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


def test_notebook_docs_keep_user_paths_and_numbered_order():
    pages = [
        GUIDES_ROOT / "notebooks-and-sample-cases.rst",
        EXAMPLES_ROOT / "notebook-series.rst",
    ]

    for path in pages:
        text = _read(path)
        for phrase in (
            "I want to solve a case with advanced methods",
            "I need to understand the method",
            "I am integrating or extending OpenPinch",
        ):
            assert phrase in text, f"{path} missing user path {phrase!r}"

        last_position = -1
        for notebook_name in list_notebooks():
            position = text.index(notebook_name)
            assert position > last_position, f"{path} has notebooks out of order"
            last_position = position


def test_public_root_exports_are_covered_by_curated_api_docs():
    combined_api = _docs_text(sorted(API_ROOT.glob("*.rst")))

    for export_name in OpenPinch.__all__:
        assert export_name in combined_api, (
            f"{export_name} missing from curated API docs"
        )


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


def test_old_url_pages_remain_orphan_transition_stubs():
    for path, targets in LEGACY_PAGE_TARGETS.items():
        text = _read(path)
        assert ":orphan:" in text
        assert "Use these pages instead" in text or "Use these sections instead" in text
        for target in targets:
            assert target in text, f"{path} missing transition target {target}"


def test_docs_do_not_present_removed_cli_surfaces():
    combined = _docs_text()

    for stale in STALE_DOC_STRINGS:
        assert stale not in combined

    assert "openpinch notebook" in combined


def test_hen_synthesis_docs_keep_public_cutover_and_dependency_notes():
    guide = _read(GUIDES_ROOT / "heat-exchanger-network-synthesis.rst")
    service_layer = _read(API_ROOT / "service-layer.rst")
    schemas_config = _read(API_ROOT / "schemas-and-config.rst")

    for phrase in (
        'python -m pip install "openpinch[synthesis]"',
        "idaes get-extensions",
        "problem.design.enhanced_synthesis_method(quality_tier=2)",
        "problem.design.open_hens_method()",
        "problem.design.network_evolution_method(initial_networks=",
        "TargetOutput.design",
        "old import paths",
        "OpenHENS field aliases",
        "pytest -m synthesis",
        "pytest -m solver",
    ):
        assert phrase in guide

    assert (
        "OpenPinch.services.heat_exchanger_network_synthesis."
        "heat_exchanger_network_synthesis_entry" in service_layer
    )
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
        "problem.target.direct_heat_pump",
        "problem.target.indirect_heat_pump",
        "problem.add_component.process_mvr",
        "Cascade Carnot cycles",
        "Parallel vapour compression cycles",
        "Vapour compression with MVR cascade",
        "04_carnot_heat_pump_screening.ipynb",
        "05_direct_gas_stream_mvr_scenarios.ipynb",
    ):
        assert phrase in guide

    assert "Simulated-cycle integration accounting" in fundamentals
