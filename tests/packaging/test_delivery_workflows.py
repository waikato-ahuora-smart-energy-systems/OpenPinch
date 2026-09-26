"""Parsed workflow contracts plus executable gate and orchestration regressions."""

import json
import os
import subprocess

import pytest
import yaml

from scripts.ci_policy import POLICY, required_jobs
from tests.support.paths import REPOSITORY_ROOT


def workflow(name):
    # BaseLoader preserves GitHub's `on` key instead of YAML 1.1's bool coercion.
    return yaml.load(
        (REPOSITORY_ROOT / ".github/workflows" / name).read_text(),
        Loader=yaml.BaseLoader,
    )


def runs(job):
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def test_branch_workflows_only_call_read_only_shared_validation():
    for name, branch, profile in [
        ("ci-develop.yml", "develop", "integration"),
        ("ci-main.yml", "main", "full"),
    ]:
        data = workflow(name)
        assert data["on"] == {"push": {"branches": [branch]}}
        assert data["permissions"] == {"contents": "read"}
        assert data["jobs"] == {
            "validation": {
                "uses": "./.github/workflows/ci-validation.yml",
                "with": {"candidate": "${{ github.sha }}", "profile": profile},
            }
        }


def test_shared_validation_has_every_lane_and_bounded_reports():
    data = workflow("ci-validation.yml")
    jobs = data["jobs"]
    assert set(jobs["gate"]["needs"]) == required_jobs("full", expanded=False)
    assert jobs["gate"]["name"] == POLICY
    assert "always()" in jobs["gate"]["if"]
    assert jobs["solver-tests"]["runs-on"] == "ubuntu-22.04"
    assert "SolverFactory(name).available(exception_flag=False)" in runs(
        jobs["solver-tests"]
    )
    assert "coverage report --fail-under=95" in runs(jobs["test"])
    assert (
        jobs["artifact-build"]["outputs"]["artifact_digest"]
        == "${{ format('sha256:{0}', steps.distribution.outputs.artifact-digest) }}"
    )
    assert "check_lockfile_version.py" in runs(jobs["test"])
    for lane in [
        "test",
        "docs",
        "hpr-tespy-tests",
        "performance-tests",
        "solver-tests",
    ]:
        assert f"selector --lane {lane}" in runs(jobs[lane])
        assert "--junitxml=" in runs(jobs[lane])
        uploads = [
            s
            for s in jobs[lane]["steps"]
            if s.get("uses", "").startswith("actions/upload-artifact@")
        ]
        assert len(uploads) == 1
        assert uploads[0]["with"]["retention-days"] == "30"
        assert "always()" in uploads[0]["if"]
    for job in jobs.values():
        assert "write" not in job.get("permissions", {}).values()
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["ref"] == "${{ inputs.candidate }}"
                assert step["with"]["persist-credentials"] == "false"
    assert "--check" in runs(jobs["workflow-lint"])
    assert "actionlint" in runs(jobs["workflow-lint"])


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "failed", [None, *sorted(required_jobs("full", expanded=False))]
)
def test_executable_shared_gate_never_hides_failed_lane(
    monkeypatch, capsys, reuse, failed
):
    from scripts.ci_policy import main

    results = {
        name: {"result": "success"} for name in required_jobs("full", expanded=False)
    }
    if reuse:
        for name in required_jobs("integration", expanded=False):
            results[name]["result"] = "skipped"
    if failed:
        results[failed]["result"] = "failure"
    monkeypatch.setenv("LANE_RESULTS", json.dumps(results))
    monkeypatch.setenv("REUSE_INTEGRATION", str(reuse).lower())
    monkeypatch.setattr("sys.argv", ["policy", "gate", "--profile", "full"])
    assert main() == int(failed is not None)
    assert POLICY in capsys.readouterr().out


@pytest.mark.parametrize("validation", ["success", "failure", "cancelled", "skipped"])
def test_actual_top_level_gate_script_requires_validation(validation, tmp_path):
    jobs = workflow("ci-pull-request.yml")["jobs"]
    env = {
        **os.environ,
        "PLAN_RESULT": "success",
        "VALIDATION_RESULT": validation,
        "RUN_VALIDATION": "true",
        "CANDIDATE": "a" * 40,
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }
    result = subprocess.run(
        ["bash", "-e", "-c", runs(jobs["pr-gate"])],
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert (result.returncode == 0) == (validation == "success")
    assert jobs["test"]["needs"] == "pr-gate"
    assert "GATE_RESULT" in runs(jobs["test"])
    assert "bump-version" not in jobs
    assert all(
        "write" not in job.get("permissions", {}).values() for job in jobs.values()
    )


def test_explicit_release_uses_one_bundle_and_verifies_before_finalizing():
    data = workflow("ci-publish.yml")
    assert set(data["on"]) == {"workflow_dispatch"}
    assert data["concurrency"] == {
        "group": "openpinch-release",
        "cancel-in-progress": "false",
    }
    jobs = data["jobs"]
    assert jobs["validation"]["with"]["profile"] == "full"
    assert "refs/heads/main" in runs(jobs["request"])
    assert "check_release_version.py" in runs(jobs["request"])
    assert "check_lockfile_version.py" in runs(jobs["request"])
    assert "dispatch-pypi" not in jobs
    assert jobs["finalize-release"]["needs"] == ["bundle", "verify-pypi"]
    assert jobs["preflight-pypi"]["needs"] == ["bundle", "verify-testpypi"]
    assert jobs["publish-pypi"]["environment"]["name"] == "pypi"
    for destination in ["testpypi", "pypi"]:
        publisher = jobs[f"publish-{destination}"]
        assert publisher["permissions"]["id-token"] == "write"
        assert "contents: write" not in json.dumps(publisher)
        assert "scripts.release_manifest verify" in runs(publisher)
        action = next(
            s for s in publisher["steps"] if s.get("uses", "").startswith("pypa/")
        )
        assert action["with"]["skip-existing"] == "true"
        assert action["with"]["packages-dir"] == "upload/"
        assert "--allow-partial" in runs(jobs[f"preflight-{destination}"])
        verifier = jobs[f"verify-{destination}"]
        assert "--require-complete" in runs(verifier)
        assert "id-token" not in verifier["permissions"]
        assert "environment" not in verifier
        assert verifier["timeout-minutes"] == "15"
    for name, job in jobs.items():
        assert (
            "actions" not in job.get("permissions", {})
            or job["permissions"]["actions"] == "read"
        )
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/download-artifact@"):
                assert "artifact-ids" in step["with"]
                assert "run-id" in step["with"]
                assert "github-token" in step["with"]
        if name not in {"validation", "request"}:
            assert "build_dist.py" not in runs(job)
            assert "scripts.release_manifest download --directory bundle" in runs(job)
            assert "SOURCE_ARTIFACT_ID" in job["env"]
            assert "SOURCE_ARTIFACT_DIGEST" in job["env"]
            if name != "bundle":
                assert "!cancelled()" in job["if"]


def test_external_actions_stay_pinned_and_artifact_versions_are_current():
    import re

    for path in (REPOSITORY_ROOT / ".github/workflows").glob("*.yml"):
        data = workflow(path.name)
        for job in data["jobs"].values():
            for item in [job, *job.get("steps", [])]:
                if "uses" not in item or item["uses"].startswith("./"):
                    continue
                assert re.fullmatch(r"[^@]+@[a-f0-9]{40}", item["uses"])
                if item["uses"].startswith("actions/upload-artifact@"):
                    assert item["uses"].endswith(
                        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
                    )
                if item["uses"].startswith("actions/download-artifact@"):
                    assert item["uses"].endswith(
                        "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
                    )
