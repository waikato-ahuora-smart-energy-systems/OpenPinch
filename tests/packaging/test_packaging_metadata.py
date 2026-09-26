"""Checks for packaging metadata declared in ``pyproject.toml``."""

from __future__ import annotations

import re
import runpy
import subprocess
import tomllib

from tests.support.paths import REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
BUMPVERSION = REPO_ROOT / ".bumpversion.toml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
PYTHON_VERSION = REPO_ROOT / ".python-version"
PYTEST_INI = REPO_ROOT / "pytest.ini"
UPDATE_TOOLCHAIN = REPO_ROOT / "scripts" / "update_toolchain.py"
WORKFLOWS = [
    REPO_ROOT / ".github" / "workflows" / "ci-develop.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-pull-request.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-publish.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-validation.yml",
    REPO_ROOT / ".github" / "workflows" / "ci-main.yml",
]
UPLOAD_ARTIFACT_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
DOWNLOAD_ARTIFACT_SHA = "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
TESPY_REQUIREMENT = "tespy>=0.10.1.post2"


def _read_pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def _read_uv_lock() -> dict:
    with UV_LOCK.open("rb") as handle:
        return tomllib.load(handle)


def _read_bumpversion() -> dict:
    with BUMPVERSION.open("rb") as handle:
        return tomllib.load(handle)


def _optional_deps() -> dict:
    return _read_pyproject()["project"]["optional-dependencies"]


def _dependency_groups() -> dict:
    return _read_pyproject()["dependency-groups"]


def _dependency_name(requirement: str) -> str:
    for separator in ("<", ">", "=", "!", "~", "[", ";"):
        requirement = requirement.split(separator, maxsplit=1)[0]
    return requirement.strip().lower().replace("_", "-")


def _minimum_python_version() -> str:
    requires_python = _read_pyproject()["project"]["requires-python"]
    assert requires_python.startswith(">=")
    return requires_python.removeprefix(">=")


def test_python_package_sources_are_not_gitignored():
    sources = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "OpenPinch").rglob("*.py")
    )
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(sources),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode in {0, 1}, completed.stderr
    assert completed.stdout == "", (
        "Python package sources are hidden by .gitignore:\n" + completed.stdout
    )


def test_notebook_extra_declares_jupyter_runtime_dependencies():
    assert _optional_deps()["notebook"] == [
        "ipykernel>=7.2.0",
        "nbformat>=5.10.4",
        "plotly",
        "openpyxl",
        "pyxlsb",
    ]


def test_dashboard_brayton_and_tespy_extras_are_declared():
    optional_deps = _optional_deps()
    assert optional_deps["dashboard"] == [
        "streamlit",
        "plotly",
        "openpyxl",
        "pyxlsb",
    ]
    assert optional_deps["brayton_cycle"] == [TESPY_REQUIREMENT]
    assert optional_deps["tespy"] == [TESPY_REQUIREMENT]


def test_hpr_runtime_requires_coolprop_8_and_dev_retains_tespy_floor():
    project = _read_pyproject()["project"]
    coolprop_entries = [
        dependency
        for dependency in project["dependencies"]
        if _dependency_name(dependency) == "coolprop"
    ]
    tespy_dev_entries = [
        dependency
        for dependency in _dependency_groups()["dev"]
        if _dependency_name(dependency) == "tespy"
    ]

    assert coolprop_entries == ["CoolProp>=8"]
    assert tespy_dev_entries == [TESPY_REQUIREMENT]


def test_synthesis_extra_declares_optional_solver_stack_only():
    optional_deps = _optional_deps()

    assert optional_deps["synthesis"] == [
        "pyomo>=6.10.0",
        "gekko>=1.3.2",
        "plotly>=6.8.0",
        "kaleido>=1.3.0",
        "openpyxl>=3.1.5",
        "wakepy>=1.0.0",
        "idaes-pse>=2.11.0",
    ]

    synthesis_only = {"pyomo", "gekko", "kaleido", "wakepy", "idaes-pse"}
    core_deps = {
        _dependency_name(dep) for dep in _read_pyproject()["project"]["dependencies"]
    }
    full_deps = {_dependency_name(dep) for dep in optional_deps["full"]}
    unrelated_optional_deps = {
        _dependency_name(dep)
        for extra_name, deps in optional_deps.items()
        if extra_name not in {"synthesis", "full"}
        for dep in deps
    }

    assert synthesis_only.isdisjoint(core_deps)
    assert synthesis_only <= full_deps
    assert synthesis_only.isdisjoint(unrelated_optional_deps)


def test_full_extra_aggregates_every_optional_runtime_surface_without_duplicates():
    optional_deps = _optional_deps()
    full = optional_deps["full"]
    full_names = [_dependency_name(dep) for dep in full]
    expected_names = {
        _dependency_name(dep)
        for extra_name, dependencies in optional_deps.items()
        if extra_name != "full"
        for dep in dependencies
    }

    assert set(full_names) == expected_names
    assert len(full_names) == len(set(full_names))


def test_dev_dependency_group_retains_notebook_dependencies():
    dev_group = _dependency_groups()["dev"]

    assert "ipykernel>=7.2.0" in dev_group
    assert "nbformat>=5.10.4" in dev_group


def test_dev_dependency_group_has_one_ruff_entry():
    dev_group = _dependency_groups()["dev"]
    ruff_entries = [entry for entry in dev_group if entry.startswith("ruff")]

    assert ruff_entries == ["ruff>=0.15.8"]


def test_jsonschema_is_a_development_only_contract_verifier():
    project = _read_pyproject()["project"]
    runtime_entries = [
        *project["dependencies"],
        *(
            dependency
            for dependencies in project["optional-dependencies"].values()
            for dependency in dependencies
        ),
    ]
    jsonschema_runtime_entries = [
        dependency
        for dependency in runtime_entries
        if _dependency_name(dependency) == "jsonschema"
    ]
    jsonschema_dev_entries = [
        dependency
        for dependency in _dependency_groups()["dev"]
        if _dependency_name(dependency) == "jsonschema"
    ]

    assert jsonschema_runtime_entries == []
    assert jsonschema_dev_entries == ["jsonschema>=4.25.1"]


def test_requires_python_matches_python_version_files_and_ci():
    minimum_version = _minimum_python_version()

    assert minimum_version == PYTHON_VERSION.read_text(encoding="utf-8").strip()

    update_toolchain = UPDATE_TOOLCHAIN.read_text(encoding="utf-8")
    assert "_read_python_minor" in update_toolchain
    assert "requires-python" in update_toolchain

    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        if "actions/setup-python@" in text:
            assert (
                f'PYTHON_VERSION: "{minimum_version}"' in text
                or f'python-version: "{minimum_version}"' in text
            )


def test_update_toolchain_uses_minor_selector_for_python_install():
    namespace = runpy.run_path(str(UPDATE_TOOLCHAIN))

    assert namespace["_read_python_version"](REPO_ROOT) == "3.14.2"
    assert namespace["_read_python_minor"](REPO_ROOT) == "3.14"
    assert namespace["_python_minor_from_version"]("3.14.2") == "3.14"


def test_requires_python_classifier_matches_minimum_version():
    project = _read_pyproject()["project"]

    assert project["requires-python"] == ">=3.14.2"
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]


def test_pytest_marker_policy_declares_optional_and_solver_tiers():
    pytest_ini = PYTEST_INI.read_text(encoding="utf-8")

    assert (
        "synthesis: optional heat exchanger network synthesis tests that require the synthesis extra"
        in pytest_ini
    )
    assert (
        "solver: tests that require external solver binaries such as Couenne or IPOPT"
        in pytest_ini
    )
    assert (
        "tespy: tests that require the optional TESPy HPR simulation extra"
        in pytest_ini
    )
    assert "performance: bounded convergence and runtime benchmark tests" in pytest_ini
    assert "docs: warning-strict documentation build tests" in pytest_ini


def test_lockfile_project_version_matches_pyproject():
    project_version = _read_pyproject()["project"]["version"]
    lock = _read_uv_lock()
    package = next(
        package
        for package in lock["package"]
        if package["name"] == "openpinch" and package["source"] == {"editable": "."}
    )

    assert package["version"] == project_version


def test_bumpversion_updates_lockfile_project_version():
    config = _read_bumpversion()["tool"]["bumpversion"]
    assert config["tag"] is False
    uv_lock_entries = [
        entry
        for entry in config["files"]
        if entry["filename"] in {"uv.lock", "./uv.lock"}
    ]

    assert len(uv_lock_entries) == 1

    entry = uv_lock_entries[0]
    current_search = entry["search"].replace(
        "{current_version}", config["current_version"]
    )

    assert 'name = "openpinch"' in entry["search"]
    assert "source = " in entry["search"]
    assert "{ editable" not in entry["search"]
    assert 'version = "{new_version}"' in entry["replace"]
    assert current_search in UV_LOCK.read_text(encoding="utf-8")


def test_ci_measures_branch_coverage_with_the_documented_hypothesis_seed():
    for workflow in [REPO_ROOT / ".github/workflows/ci-validation.yml"]:
        text = workflow.read_text(encoding="utf-8")
        assert "coverage run --branch --source=OpenPinch" in text
        assert "--hypothesis-seed=20260715" in text
        assert "coverage report --fail-under=95" in text


def test_every_external_action_is_pinned_to_an_immutable_commit():
    action_ref = re.compile(r"^\s*(?:- )?uses: ([^\s]+)$", re.MULTILINE)

    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        for reference in action_ref.findall(workflow):
            assert reference.startswith("./.github/workflows/") or re.fullmatch(
                r"[^@]+@[0-9a-f]{40}", reference
            ), (
                workflow_path,
                reference,
            )


def test_installed_wheel_smoke_uses_only_the_root_workflow_contract():
    smoke = (REPO_ROOT / "scripts" / "artifact_install_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "from OpenPinch import PinchProblem, PinchWorkspace" in smoke
    assert "Installed wheel failed the PinchProblem workflow" in smoke
    assert "Unexpected root exports" in smoke
    assert "Installed wheel contains retired packages" in smoke
    assert 'choices=("core", "tespy")' in smoke
    assert "load_tespy_compressor_characteristic" in smoke
    assert 'get_hpr_point_simulator("tespy")' in smoke
    assert "HprTargetSimulationRecord" in smoke
    assert "_exercise_tespy_public_target_and_map" in smoke
    assert "hpr_performance_map" in smoke


def test_core_dependencies_have_required_coolprop_floor_and_other_major_ceilings():
    dependencies = _read_pyproject()["project"]["dependencies"]

    assert dependencies == [
        "numpy<3",
        "pint<1",
        "pandas<3",
        "CoolProp>=8",
        "pydantic<3",
        "scipy<2",
    ]
