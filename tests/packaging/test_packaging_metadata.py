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
]
UPLOAD_ARTIFACT_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
DOWNLOAD_ARTIFACT_SHA = "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"


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


def test_dashboard_and_brayton_cycle_extras_are_declared():
    optional_deps = _optional_deps()
    assert optional_deps["dashboard"] == [
        "streamlit",
        "plotly",
        "openpyxl",
        "pyxlsb",
    ]
    assert optional_deps["brayton_cycle"] == [
        "tespy",
    ]


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


def test_requires_python_matches_python_version_files_and_ci():
    minimum_version = _minimum_python_version()

    assert minimum_version == PYTHON_VERSION.read_text(encoding="utf-8").strip()

    update_toolchain = UPDATE_TOOLCHAIN.read_text(encoding="utf-8")
    assert "_read_python_minor" in update_toolchain
    assert "requires-python" in update_toolchain

    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        assert f'PYTHON_VERSION: "{minimum_version}"' in text
        assert "python-version: ${{ env.PYTHON_VERSION }}" in text


def test_update_toolchain_uses_minor_selector_for_python_install():
    namespace = runpy.run_path(str(UPDATE_TOOLCHAIN))

    assert namespace["_read_python_version"](REPO_ROOT) == "3.14.2"
    assert namespace["_read_python_minor"](REPO_ROOT) == "3.14"
    assert namespace["_python_minor_from_version"]("3.14.2") == "3.14"


def test_requires_python_classifier_matches_minimum_version():
    project = _read_pyproject()["project"]

    assert project["requires-python"] == ">=3.14.2"
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]


def test_pytest_marker_policy_declares_synthesis_and_solver_tiers():
    pytest_ini = PYTEST_INI.read_text(encoding="utf-8")

    assert (
        "synthesis: optional heat exchanger network synthesis tests that require the synthesis extra"
        in pytest_ini
    )
    assert (
        "solver: tests that require external solver binaries such as Couenne or IPOPT"
        in pytest_ini
    )


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


def test_ci_workflows_check_lockfile_project_version():
    command = "python scripts/check_lockfile_version.py"

    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        assert command in text

    pull_request_workflow = (
        REPO_ROOT / ".github" / "workflows" / "ci-pull-request.yml"
    ).read_text(encoding="utf-8")
    assert pull_request_workflow.count(command) >= 2


def test_ci_measures_branch_coverage_with_the_documented_hypothesis_seed():
    for workflow in WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")
        assert "coverage run --branch --source=OpenPinch" in text
        assert "--hypothesis-seed=20260715" in text
        assert "coverage report --fail-under=95" in text


def test_workflows_use_frozen_uv_environment_and_read_only_default_permissions():
    setup_uv_ref = re.compile(r"astral-sh/setup-uv@[0-9a-f]{40}")

    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "permissions:\n  contents: read" in workflow
        assert setup_uv_ref.search(workflow)
        assert "uv sync --frozen" in workflow
        assert "pip install --group dev" not in workflow
        assert "python -m pip install --upgrade pip" not in workflow


def test_every_external_action_is_pinned_to_an_immutable_commit():
    action_ref = re.compile(r"^\s*- uses: ([^\s]+)$", re.MULTILINE)

    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        for reference in action_ref.findall(workflow):
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", reference), (
                workflow_path,
                reference,
            )


def test_workflows_use_current_node24_artifact_actions():
    upload_ref = f"actions/upload-artifact@{UPLOAD_ARTIFACT_SHA}"
    download_ref = f"actions/download-artifact@{DOWNLOAD_ARTIFACT_SHA}"

    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert upload_ref in workflow
        assert download_ref in workflow


def test_testpypi_and_pypi_publish_the_same_release_artifacts_retry_safely():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )
    testpypi_block = workflow.split("preflight-testpypi:", 1)[1].split(
        "publish-release:", 1
    )[0]
    release_block = workflow.split("create-draft-release:", 1)[1].split(
        "publish-testpypi:", 1
    )[0]
    validation_block = workflow.split("validate-published-release:", 1)[1].split(
        "publish-pypi:", 1
    )[0]
    pypi_block = workflow.split("preflight-pypi:", 1)[1]

    assert "skip-existing: true" in testpypi_block
    assert "skip-existing: true" in pypi_block
    assert testpypi_block.count("scripts/check_package_index_release.py") >= 2
    assert pypi_block.count("scripts/check_package_index_release.py") >= 2
    assert "--allow-partial" in testpypi_block
    assert "--allow-partial" in pypi_block
    assert "--require-complete" in testpypi_block
    assert "--require-complete" in pypi_block
    assert (
        testpypi_block.count("artifact-ids: ${{ needs.build.outputs.artifact_id }}")
        == 3
    )
    assert "dist/* SHA256SUMS --draft" in release_block
    assert 'gh release download "${TAG_NAME}"' in validation_block
    assert "sha256sum --check ../SHA256SUMS" in validation_block
    assert "name: openpinch-release-dist" in validation_block
    assert (
        pypi_block.count(
            "artifact-ids: ${{ needs.validate-published-release.outputs.artifact_id }}"
        )
        == 3
    )
    assert "build_dist.py" not in validation_block
    assert "build_dist.py" not in pypi_block


def test_release_artifacts_are_anchored_to_a_verified_immutable_source_run():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )
    build_block = workflow.split("  build:", 1)[1].split(
        "  artifact-install-smoke:", 1
    )[0]
    release_block = workflow.split("  create-draft-release:", 1)[1].split(
        "  publish-testpypi:", 1
    )[0]
    dispatch_block = workflow.split("  dispatch-pypi:", 1)[1].split(
        "  validate-published-release:", 1
    )[0]
    validation_block = workflow.split("  validate-published-release:", 1)[1].split(
        "  publish-pypi:", 1
    )[0]

    assert "artifact_id: ${{ steps.release_artifact.outputs.artifact-id }}" in (
        build_block
    )
    assert "artifact_digest: ${{ steps.release_artifact.outputs.artifact-digest }}" in (
        build_block
    )
    assert "artifact_attempt: ${{ steps.artifact_identity.outputs.run_attempt }}" in (
        build_block
    )
    assert "id: release_artifact" in build_block
    assert "contents: write" not in build_block

    assert "needs: [release-check, artifact-install-smoke, build]" in release_block
    assert "SOURCE_ARTIFACT_ID: ${{ needs.build.outputs.artifact_id }}" in release_block
    assert "artifact-ids: ${{ needs.build.outputs.artifact_id }}" in release_block

    assert "needs: [release-check, publish-release, build]" in dispatch_block
    for input_name in (
        "source_run_id",
        "source_artifact_attempt",
        "source_artifact_id",
        "source_artifact_digest",
    ):
        assert f'-f {input_name}="${{{input_name.upper()}}}"' in dispatch_block

    assert "manifest_sha256" not in workflow
    assert "SOURCE_RUN_ID: ${{ inputs.source_run_id }}" in validation_block
    assert "SOURCE_ARTIFACT_ATTEMPT: ${{ inputs.source_artifact_attempt }}" in (
        validation_block
    )
    assert "SOURCE_ARTIFACT_ID: ${{ inputs.source_artifact_id }}" in validation_block
    assert "SOURCE_ARTIFACT_DIGEST: ${{ inputs.source_artifact_digest }}" in (
        validation_block
    )
    assert "/actions/workflows/${source_workflow_id}" in validation_block
    assert 'source_workflow_path="$(gh api' in validation_block
    assert "/actions/runs/${SOURCE_RUN_ID}/jobs" in validation_block
    assert "filter=latest" in validation_block
    assert "/attempts/${SOURCE_ARTIFACT_ATTEMPT}/jobs" not in validation_block
    assert "artifact-ids: ${{ inputs.source_artifact_id }}" in validation_block
    assert "github-token: ${{ github.token }}" in validation_block
    assert "actions: read" in validation_block
    assert 'cmp "source-dist/${expected_wheel}" "dist/${expected_wheel}"' in (
        validation_block
    )


def test_publish_workflow_hands_off_from_public_release_to_tag_ref_pypi():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )

    assert 'branches: ["main"]' in workflow
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in workflow
    assert "tags:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "reject-invalid-dispatch:" in workflow
    assert "github.ref_type != 'tag'" in workflow
    assert "Production publication must be dispatched at a release tag." in workflow
    assert "release_tag:" in workflow
    assert "source_run_id:" in workflow
    assert "python scripts/check_release_version.py" in workflow
    assert "solver-tests:" in workflow
    assert 'pytest --hypothesis-seed=20260715 -m "solver"' in workflow
    assert "create-draft-release:" in workflow
    assert 'git tag -a "${TAG_NAME}"' in workflow
    assert 'gh release create "${TAG_NAME}"' in workflow
    assert "--draft" in workflow
    assert "publish-testpypi:" in workflow
    assert "publish-release:" in workflow
    assert "needs: [release-check, verify-testpypi]" in workflow
    assert 'gh release edit "${TAG_NAME}" --draft=false --latest' in workflow
    assert "dispatch-pypi:" in workflow
    assert "needs: [release-check, publish-release, build]" in workflow
    assert 'gh workflow run ci-publish.yml --ref "${TAG_NAME}"' in workflow
    assert '-f release_tag="${TAG_NAME}"' in workflow
    assert "validate-published-release:" in workflow
    assert "github.ref_type == 'tag'" in workflow
    assert "TAG_NAME: ${{ inputs.release_tag }}" in workflow
    assert '"${TAG_NAME}" != "${GITHUB_REF_NAME}"' in workflow
    assert "publish-pypi:" in workflow
    assert "needs: [validate-published-release, preflight-pypi]" in workflow
    assert "finalize-release:" not in workflow
    assert "coverage report --fail-under=95" in workflow
    assert "surface: [core, dashboard, notebook, brayton_cycle, synthesis]" in workflow
    assert "os: [ubuntu-latest, windows-latest, macos-latest]" in workflow


def test_publish_solver_gate_uses_supported_runner_and_probes_binaries():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )
    solver_block = workflow.split("solver-tests:", 1)[1].split("\n  build:", 1)[0]

    assert "runs-on: ubuntu-22.04" in solver_block
    assert "Verify IDAES solver binaries" in solver_block
    assert "SolverFactory(name).available(exception_flag=False)" in solver_block
    assert "('couenne', 'ipopt')" in solver_block


def test_pr_workflow_bumps_same_repository_main_pr_before_release_validation():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-pull-request.yml").read_text(
        encoding="utf-8"
    )
    bump_block = workflow.split("  bump-version:", 1)[1].split("  release-version:", 1)[
        0
    ]
    release_block = workflow.split("  release-version:", 1)[1].split("  docs:", 1)[0]

    assert 'branches: ["main", "develop"]' in workflow
    assert "edited" in workflow
    assert "pull-requests: write" not in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in (
        bump_block
    )
    assert "github.event.pull_request.base.ref == 'main'" in workflow
    assert "contents: write" in bump_block
    assert "persist-credentials: true" in bump_block
    assert "bump-my-version==1.2.3" in bump_block
    assert 'BUMP_PART="patch"' in bump_block
    assert 'NORMALIZED_LABELS=",${PR_LABELS,,},"' in bump_block
    assert '[[ "${PR_TITLE,,}" =~ \\[(major|minor|patch)\\] ]]' in bump_block
    assert 'if [ "${CURRENT_VERSION}" = "${BASE_VERSION}" ]; then' in bump_block
    assert "elif ! python scripts/check_release_version.py" in bump_block
    assert '--base-pyproject "${RUNNER_TEMP}/base-pyproject.toml"' in bump_block
    assert 'bump "${BUMP_PART}" --no-tag' in bump_block
    assert "python scripts/check_lockfile_version.py" in bump_block
    assert "git push origin" in bump_block
    assert "needs: bump-version" in release_block
    assert "needs.bump-version.result" in release_block
    assert "repository: ${{ github.event.pull_request.head.repo.full_name }}" in (
        release_block
    )
    assert "ref: ${{ github.event.pull_request.head.ref }}" in release_block
    assert "persist-credentials: false" in release_block
    assert "python scripts/check_release_version.py --base-pyproject" in release_block
    assert "coverage report --fail-under=95" in workflow
    assert "surface: [core, dashboard, notebook, brayton_cycle, synthesis]" in workflow
    assert "solver-tests:" in workflow
    assert 'pytest --hypothesis-seed=20260715 -m "solver"' in workflow
    assert "pr-gate:" in workflow
    gate_block = workflow.split("  pr-gate:", 1)[1]
    assert "- bump-version" in gate_block
    assert "BUMP_VERSION_RESULT: ${{ needs.bump-version.result }}" in gate_block
    assert (
        "HEAD_REPO: ${{ github.event.pull_request.head.repo.full_name }}" in gate_block
    )
    assert "REPOSITORY: ${{ github.repository }}" in gate_block


def test_develop_workflow_defers_to_an_open_develop_to_main_pull_request():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-develop.yml").read_text(
        encoding="utf-8"
    )

    assert "detect-main-pr:" in workflow
    assert "pull-requests: read" in workflow
    assert 'head="${GITHUB_REPOSITORY_OWNER}:develop"' in workflow
    assert 'base="main"' in workflow
    assert workflow.count("needs: detect-main-pr") >= 4
    assert "needs.detect-main-pr.outputs.should_run == 'true'" in workflow


def test_parallel_jobs_restore_uv_caches_without_competing_to_save_them():
    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        setup_count = workflow.count("astral-sh/setup-uv@")
        save_disabled_count = workflow.count("save-cache: false")

        assert setup_count > 0
        assert save_disabled_count >= setup_count - 3


def test_release_jobs_are_privilege_separated_from_project_code_execution():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )

    create_block = workflow.split("  create-draft-release:", 1)[1].split(
        "  preflight-testpypi:", 1
    )[0]
    publish_block = workflow.split("  publish-release:", 1)[1].split(
        "  dispatch-pypi:", 1
    )[0]
    dispatch_block = workflow.split("  dispatch-pypi:", 1)[1].split(
        "  validate-published-release:", 1
    )[0]
    assert "contents: write" in create_block
    assert "contents: write" in publish_block
    assert "actions: write" in dispatch_block
    assert "contents: write" not in dispatch_block
    assert "GH_REPO: ${{ github.repository }}" in publish_block
    assert "GH_REPO: ${{ github.repository }}" in dispatch_block
    assert "uv run" not in create_block
    assert "python scripts/" not in create_block
    assert "uv run" not in publish_block
    assert "python scripts/" not in publish_block
    assert "uv run" not in dispatch_block
    assert "python scripts/" not in dispatch_block


def test_release_creation_never_clobbers_a_public_release():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )
    release_check = workflow.split("  release-check:", 1)[1].split("  test:", 1)[0]
    create_block = workflow.split("  create-draft-release:", 1)[1].split(
        "  publish-testpypi:", 1
    )[0]

    assert "--clobber" not in create_block
    assert "isDraft" in release_check
    assert "already public" in release_check
    assert "isDraft" in create_block
    assert "existing release assets" in create_block
    assert "--json assets --jq '.assets[].name'" in create_block
    assert "expected_asset_names" in create_block
    assert 'git cat-file -t "refs/tags/${TAG_NAME}"' in create_block


def test_pypi_environment_is_reached_only_from_a_verified_public_tag_release():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )
    validation_block = workflow.split("  validate-published-release:", 1)[1].split(
        "  publish-pypi:", 1
    )[0]
    pypi_block = workflow.split("  publish-pypi:", 1)[1].split("  verify-pypi:", 1)[0]

    assert "github.event_name == 'workflow_dispatch'" in validation_block
    assert "github.ref_type == 'tag'" in validation_block
    assert "TAG_NAME: ${{ inputs.release_tag }}" in validation_block
    assert '"${GITHUB_REF_TYPE}" != "tag"' in validation_block
    assert '"${TAG_NAME}" != "${GITHUB_REF_NAME}"' in validation_block
    assert '"${release_is_draft}" != "false"' in validation_block
    assert '"${release_is_prerelease}" != "false"' in validation_block
    assert "--json assets --jq '.assets[].name'" in validation_block
    assert "release_asset_names" in validation_block
    assert "environment:" not in validation_block
    assert "needs: [validate-published-release, preflight-pypi]" in pypi_block
    assert "name: pypi" in pypi_block
    assert "id-token: write" in pypi_block


def test_pypi_verification_is_an_independently_retryable_unprivileged_job():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci-publish.yml").read_text(
        encoding="utf-8"
    )
    pypi_block = workflow.split("  publish-pypi:", 1)[1].split("  verify-pypi:", 1)[0]
    verification_block = workflow.split("  verify-pypi:", 1)[1]

    assert "Verify the published version on PyPI" not in pypi_block
    assert "curl --fail" not in pypi_block
    assert "needs: [validate-published-release, publish-pypi]" in verification_block
    assert "contents: read" in verification_block
    assert "environment:" not in verification_block
    assert "id-token: write" not in verification_block
    assert "Verify the published version on PyPI" in verification_block
    assert "scripts/check_package_index_release.py" in verification_block
    assert "--require-complete" in verification_block


def test_installed_wheel_smoke_uses_only_the_root_workflow_contract():
    smoke = (REPO_ROOT / "scripts" / "artifact_install_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "from OpenPinch import PinchProblem, PinchWorkspace" in smoke
    assert "Installed wheel failed the PinchProblem workflow" in smoke
    assert "Unexpected root exports" in smoke
    assert "Installed wheel contains retired packages" in smoke


def test_core_dependencies_have_incompatible_major_release_ceilings():
    dependencies = _read_pyproject()["project"]["dependencies"]

    assert dependencies == [
        "numpy<3",
        "pint<1",
        "pandas<3",
        "coolprop<8",
        "pydantic<3",
        "scipy<2",
    ]
