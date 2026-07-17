"""Static dependency rules for the owner-oriented package layers."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import OpenPinch
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value
from OpenPinch.domain.zone import Zone
from tests.support.paths import REPOSITORY_ROOT, TESTS_ROOT

PACKAGE_DIR = Path(OpenPinch.__file__).parent
LAYER_ALLOWED_ROOTS = {
    "domain": {"domain"},
    "contracts": {"contracts", "domain"},
    "optimisation": {"optimisation"},
    "adapters": {"adapters", "contracts", "domain", "resources"},
    "analysis": {"adapters", "analysis", "contracts", "domain", "optimisation"},
    "application": {
        "adapters",
        "analysis",
        "application",
        "contracts",
        "domain",
        "presentation",
    },
    "presentation": {"adapters", "analysis", "contracts", "domain", "presentation"},
}
LAYER_BOUNDARY_EXCEPTIONS = {
    "analysis/heat_exchanger_networks/context.py": {"application"},
    "analysis/heat_exchanger_networks/reporting/exports.py": {"application"},
    "analysis/heat_exchanger_networks/results/seeds.py": {"application"},
    "analysis/heat_exchanger_networks/service.py": {"application"},
    "analysis/heat_exchanger_networks/solver/arrays.py": {"application"},
    "analysis/heat_exchanger_networks/solver/pinch_design_decomposition.py": {
        "application"
    },
    "analysis/heat_pumps/components.py": {"application"},
    "analysis/heat_pumps/process_mvr.py": {"application"},
    "presentation/graphs/problem.py": {"application"},
}
OWNER_PACKAGE_MARKERS = {
    "OpenPinch",
    *(f"OpenPinch.{owner}" for owner in LAYER_ALLOWED_ROOTS),
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_DIR).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("OpenPinch", *parts))


def _import_target(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    module_name = _module_name(path)
    package = (
        module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    )
    relative_name = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative_name, package)


def _openpinch_import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            targets.append(_import_target(path, node))
        elif isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        for target in targets:
            if not target.startswith("OpenPinch."):
                continue
            parts = target.split(".")
            if len(parts) > 1:
                roots.add(parts[1])
    return roots


def _import_targets(path: Path) -> set[str]:
    targets: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            targets.add(_import_target(path, node))
        elif isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
    return targets


def test_layer_dependencies_follow_explicit_allowed_directions() -> None:
    observed_exceptions: dict[str, set[str]] = {}
    for layer, allowed_roots in LAYER_ALLOWED_ROOTS.items():
        for path in (PACKAGE_DIR / layer).rglob("*.py"):
            relative = str(path.relative_to(PACKAGE_DIR))
            imported_roots = _openpinch_import_roots(path)
            exceptions = LAYER_BOUNDARY_EXCEPTIONS.get(relative, set())
            unexpected = imported_roots - allowed_roots - exceptions
            assert not unexpected, f"{relative}: unexpected dependencies {unexpected}"
            observed = imported_roots - allowed_roots
            if observed:
                observed_exceptions[relative] = observed

    assert observed_exceptions == LAYER_BOUNDARY_EXCEPTIONS


def test_source_imports_concrete_modules_instead_of_parent_barrels() -> None:
    offenders: list[str] = []
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _import_target(path, node)
            if target in OWNER_PACKAGE_MARKERS:
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{node.lineno}")

    assert offenders == []


def test_repository_entrypoints_do_not_import_retired_packages() -> None:
    retired_roots = {
        "OpenPinch.classes",
        "OpenPinch.lib",
        "OpenPinch.services",
        "OpenPinch.streamlit_webviewer",
        "OpenPinch.utils",
    }
    paths = [
        *sorted((REPOSITORY_ROOT / "examples").rglob("*.py")),
        *sorted((REPOSITORY_ROOT / "scripts").rglob("*.py")),
        REPOSITORY_ROOT / "streamlit_app.py",
    ]
    offenders: dict[str, list[str]] = {}
    for path in paths:
        imported: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module)
        retired = sorted(
            target
            for target in imported
            if any(
                target == root or target.startswith(f"{root}.")
                for root in retired_roots
            )
        )
        if retired:
            offenders[str(path.relative_to(REPOSITORY_ROOT))] = retired

    assert offenders == {}


def test_concrete_module_exports_are_defined_by_their_owner() -> None:
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_names: set[str] = set()
        defined_names: set[str] = set()
        exported_names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                imported_names.update(
                    alias.asname or alias.name for alias in node.names
                )
            elif isinstance(node, ast.Import):
                imported_names.update(
                    alias.asname or alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                exported_names.update(ast.literal_eval(node.value))

        forwarded = sorted((exported_names & imported_names) - defined_names)
        if forwarded:
            offenders[str(path.relative_to(PACKAGE_DIR))] = forwarded

    assert offenders == {}


def test_test_modules_are_grouped_by_observable_owner_layer() -> None:
    expected_test_layers = {
        "adapters",
        "analysis",
        "application",
        "architecture",
        "contracts",
        "domain",
        "e2e",
        "optimisation",
        "packaging",
        "presentation",
    }
    observed_test_layers = {
        path.relative_to(TESTS_ROOT).parts[0] for path in TESTS_ROOT.rglob("test_*.py")
    }

    assert observed_test_layers == expected_test_layers
    assert not list(TESTS_ROOT.glob("test_*.py"))
    for retired in ("test_analysis", "test_classes", "test_lib", "test_utils"):
        assert not list((TESTS_ROOT / retired).rglob("*.py"))


def test_domain_has_no_outward_package_dependencies() -> None:
    forbidden = {
        "adapters",
        "analysis",
        "application",
        "classes",
        "contracts",
        "presentation",
        "services",
        "streamlit_webviewer",
        "utils",
    }
    for path in (PACKAGE_DIR / "domain").rglob("*.py"):
        assert _openpinch_import_roots(path).isdisjoint(forbidden), path


def test_contracts_depend_only_on_domain_and_contract_peers() -> None:
    forbidden = {
        "adapters",
        "analysis",
        "application",
        "classes",
        "optimisation",
        "presentation",
        "services",
        "streamlit_webviewer",
        "utils",
    }
    for path in (PACKAGE_DIR / "contracts").rglob("*.py"):
        assert _openpinch_import_roots(path).isdisjoint(forbidden), path


def test_core_domain_classes_have_concrete_domain_owners() -> None:
    expected = {
        Value: "OpenPinch.domain.value",
        Stream: "OpenPinch.domain.stream",
        StreamCollection: "OpenPinch.domain.stream_collection",
        ProblemTable: "OpenPinch.domain.problem_table",
        Zone: "OpenPinch.domain.zone",
        HeatExchanger: "OpenPinch.domain.heat_exchanger",
        HeatExchangerNetwork: "OpenPinch.domain.heat_exchanger_network",
    }
    assert {owner: owner.__module__ for owner in expected} == expected


def test_application_use_cases_have_concrete_application_owners() -> None:
    assert PinchProblem.__module__ == "OpenPinch.application.problem"
    assert PinchWorkspace.__module__ == "OpenPinch.application.workspace"
    assert not (PACKAGE_DIR / "classes" / "pinch_problem.py").exists()
    assert not (PACKAGE_DIR / "classes" / "pinch_workspace.py").exists()
    assert not (PACKAGE_DIR / "classes" / "_pinch_problem").exists()
    assert not (PACKAGE_DIR / "classes" / "_pinch_workspace").exists()


def test_presentation_helpers_have_concrete_owners() -> None:
    assert not (PACKAGE_DIR / "streamlit_webviewer").exists()
    assert not (PACKAGE_DIR / "services" / "network_grid_diagram").exists()
    assert (PACKAGE_DIR / "presentation" / "dashboard" / "rendering.py").is_file()
    assert (PACKAGE_DIR / "presentation" / "graphs" / "plotly.py").is_file()
    assert (PACKAGE_DIR / "presentation" / "network_grid" / "geometry.py").is_file()
    assert (PACKAGE_DIR / "presentation" / "network_grid" / "labels.py").is_file()
    assert (PACKAGE_DIR / "presentation" / "network_grid" / "temperatures.py").is_file()


def test_foundational_analysis_and_io_retired_legacy_owners() -> None:
    retired_service_packages = {
        "common",
        "direct_heat_integration",
        "energy_transfer_analysis",
        "exergy_analysis",
        "indirect_heat_integration",
        "input_data_processing",
        "network_grid_diagram",
        "power_cogeneration",
    }
    for package_name in retired_service_packages:
        assert not (PACKAGE_DIR / "services" / package_name).exists()

    retired_utility_modules = {
        "_tabular_input.py",
        "costing.py",
        "csv_to_json.py",
        "export.py",
        "heat_exchanger.py",
        "input_validation.py",
        "optional_dependencies.py",
        "plots.py",
        "value_resolution.py",
        "water_properties.py",
        "wkbook_to_json.py",
    }
    for module_name in retired_utility_modules:
        assert not (PACKAGE_DIR / "utils" / module_name).exists()


def test_foundational_analysis_does_not_depend_on_legacy_layers() -> None:
    roots = (
        PACKAGE_DIR / "analysis" / "energy_transfer",
        PACKAGE_DIR / "analysis" / "exergy",
        PACKAGE_DIR / "analysis" / "graphs",
        PACKAGE_DIR / "analysis" / "power",
        PACKAGE_DIR / "analysis" / "targeting",
        PACKAGE_DIR / "analysis" / "thermodynamics",
    )
    for root in roots:
        for path in root.rglob("*.py"):
            assert _openpinch_import_roots(path).isdisjoint({"services", "utils"}), path


def test_heat_pump_analysis_has_owner_packages_without_compatibility_facades() -> None:
    heat_pumps = PACKAGE_DIR / "analysis" / "heat_pumps"
    assert (heat_pumps / "optimisation_adapter.py").is_file()
    assert (heat_pumps / "_multiperiod" / "preparation.py").is_file()
    assert (heat_pumps / "_multiperiod" / "aggregation.py").is_file()
    assert (heat_pumps / "_multiperiod" / "execution.py").is_file()
    assert (heat_pumps / "_process_mvr" / "replacement_streams.py").is_file()
    assert (heat_pumps / "direct_mvr" / "thermodynamics.py").is_file()
    assert not (heat_pumps / "targeting" / "multiperiod.py").exists()
    assert not (heat_pumps / "targeting" / "_multiperiod").exists()
    assert not (heat_pumps / "direct_mvr" / "direct_gas_mvr.py").exists()
    assert not (heat_pumps / "direct_mvr" / "_direct_gas_mvr").exists()
    assert not (PACKAGE_DIR / "services" / "components").exists()
    assert not (PACKAGE_DIR / "services" / "heat_pump_integration").exists()
    assert not (PACKAGE_DIR / "utils" / "blackbox_minimisers.py").exists()

    for path in heat_pumps.rglob("*.py"):
        assert _openpinch_import_roots(path).isdisjoint({"services", "utils"}), path


def test_hen_analysis_has_concrete_owner_packages_without_facades() -> None:
    hens = PACKAGE_DIR / "analysis" / "heat_exchanger_networks"
    expected_files = {
        "controllability.py",
        "context.py",
        "errors.py",
        "execution/executor.py",
        "extraction/metadata.py",
        "extraction/period_state.py",
        "extraction/recovery.py",
        "extraction/segment_area.py",
        "extraction/utility.py",
        "models/_base/parameters.py",
        "models/_base/piecewise.py",
        "models/_stagewise/equations.py",
        "models/_stagewise/evolution.py",
        "models/_stagewise/warm_start.py",
        "models/_pinch_design/amalgamation.py",
        "models/_pinch_design/equations.py",
        "models/pinch_decomposition.py",
        "reporting/verification.py",
        "results/selection.py",
        "solver/backend.py",
    }
    assert all((hens / relative_path).is_file() for relative_path in expected_files)
    assert not (hens / "common").exists()
    assert not (hens / "models" / "pinch_design.py").exists()
    assert not (hens / "solver" / "extraction.py").exists()
    assert not (
        PACKAGE_DIR / "services" / "heat_exchanger_network_controllability"
    ).exists()
    assert not (PACKAGE_DIR / "services" / "heat_exchanger_network_synthesis").exists()

    model_barrel = hens / "models" / "__init__.py"
    assert not any(
        isinstance(node, ast.Import | ast.ImportFrom)
        for node in ast.parse(model_barrel.read_text()).body
    )
    for path in hens.rglob("*.py"):
        assert _openpinch_import_roots(path).isdisjoint({"services", "utils"}), path


def test_application_has_no_concrete_ui_filesystem_or_solver_imports() -> None:
    forbidden_roots = {
        "OpenPinch.classes",
        "OpenPinch.lib",
        "OpenPinch.optimisation.backends",
        "OpenPinch.streamlit_webviewer",
        "idaes",
        "pathlib",
        "plotly",
        "pyomo",
        "streamlit",
    }
    for path in (PACKAGE_DIR / "application").rglob("*.py"):
        targets = _import_targets(path)
        assert not {
            target
            for target in targets
            if any(
                target == forbidden or target.startswith(f"{forbidden}.")
                for forbidden in forbidden_roots
            )
        }, path
