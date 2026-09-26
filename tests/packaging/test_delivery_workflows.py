"""Parsed workflow contracts plus executable gate and orchestration regressions."""

import json
import os
import re
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


def test_metadata_marker_only_identifies_ready_non_base_edits():
    job = workflow("ci-pull-request.yml")["jobs"]["review-metadata-only"]
    assert job["if"] == (
        "${{ github.event.action == 'edited' && !github.event.changes.base "
        "&& !github.event.pull_request.draft }}"
    )
    assert job["permissions"] == {}
    assert all("uses" not in step for step in job["steps"])


def test_preparation_authority_and_evidence_contracts():
    prepare = workflow("ci-prepare-release.yml")
    assert prepare["concurrency"]["cancel-in-progress"] == "false"
    job = prepare["jobs"]["prepare"]
    assert job["permissions"] == {
        "contents": "write",
        "pull-requests": "write",
        "actions": "read",
    }
    assert job["steps"][0]["with"]["ref"] == "refs/heads/main"
    assert job["steps"][0]["with"]["persist-credentials"] == "false"
    assert "reconcile" in runs(job)
    notification = workflow("ci-review-notification.yml")
    assert notification["permissions"] == {}
    assert all("uses" not in s for s in notification["jobs"]["notify"]["steps"])
    jobs = workflow("ci-validation.yml")["jobs"]
    assert "reusable" not in runs(jobs["test"])
    assert "Fresh version-sensitive" in str(jobs["test"]["steps"])
    for name in (
        "artifact-build",
        "artifact-install-smoke",
        "artifact-install-tespy-smoke",
    ):
        assert "!cancelled()" in jobs[name]["if"]
    assert "REUSE_REVIEW" in str(jobs["gate"])
    assert "review-proof" in jobs["gate"]["needs"]
    assert "scripts.release_preparation gate" in runs(
        workflow("ci-pull-request.yml")["jobs"]["pr-gate"]
    )


def test_branch_workflows_only_call_read_only_shared_validation():
    for name, branch, profile in [
        ("ci-develop.yml", "develop", "integration"),
        ("ci-main.yml", "main", "full"),
    ]:
        data = workflow(name)
        assert data["on"] == {"push": {"branches": [branch]}}
        assert data["permissions"] == {"contents": "read", "actions": "read"}
        assert data["jobs"] == {
            "validation": {
                "uses": "./.github/workflows/ci-validation.yml",
                "with": {"candidate": "${{ github.sha }}", "profile": profile},
            }
        }


def test_shared_validation_has_every_lane_and_bounded_reports():
    data = workflow("ci-validation.yml")
    jobs = data["jobs"]
    assert set(jobs["gate"]["needs"]) == required_jobs("full", expanded=False) | {
        "review-proof"
    }
    assert (
        jobs["gate"]["name"]
        == "${{ needs.review-proof.outputs.reuse == 'true' && 'delivery-v2' || 'delivery-v1' }}"
    )
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
        [
            "bash",
            "-e",
            "-c",
            next(
                s["run"]
                for s in jobs["pr-gate"]["steps"]
                if s.get("name") == "Require complete validation"
            ),
        ],
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
    assert set(data["on"]) == {"workflow_dispatch", "workflow_run"}
    assert data["on"]["workflow_run"] == {
        "workflows": ["CI Main"],
        "types": ["completed"],
        "branches": ["main"],
    }
    assert data["concurrency"] == {
        "group": "openpinch-release",
        "cancel-in-progress": "false",
    }
    jobs = data["jobs"]
    assert jobs["request"]["permissions"] == {"contents": "read", "actions": "read"}
    assert "python -m scripts.plan_release" in runs(jobs["request"])
    assert (
        jobs["validation"]["if"]
        == "${{ github.event_name == 'workflow_dispatch' && inputs.mode == 'new' }}"
    )
    assert "needs.request.outputs.action == 'publish'" in jobs["bundle"]["if"]
    assert "github.event_name == 'workflow_run'" in jobs["bundle"]["if"]
    assert (
        jobs["bundle"]["outputs"]["version"] == "${{ needs.request.outputs.version }}"
    )
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
                assert job["env"]["VERSION"] == "${{ needs.bundle.outputs.version }}"


def test_external_actions_stay_pinned_and_artifact_versions_are_current():
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


@pytest.mark.parametrize("mode", ["automatic", "new", "resume"])
@pytest.mark.parametrize("decision", ["publish", "complete", "ignore"])
@pytest.mark.parametrize("request_result", ["success", "failure"])
@pytest.mark.parametrize("validation", ["success", "failure", "cancelled", "skipped"])
def test_bundle_condition_cannot_bypass_planning_or_required_validation(
    mode, decision, request_result, validation
):
    expression = workflow("ci-publish.yml")["jobs"]["bundle"]["if"][3:-3]
    values = {
        "needs.request.result": request_result,
        "needs.request.outputs.action": decision,
        "needs.validation.result": validation,
        "github.event_name": "workflow_run"
        if mode == "automatic"
        else "workflow_dispatch",
        "inputs.mode": "" if mode == "automatic" else mode,
    }
    for key, value in values.items():
        expression = expression.replace(key, repr(value))
    expression = (
        expression.replace("!cancelled()", "True")
        .replace("&&", "and")
        .replace("||", "or")
    )
    # Evaluate only the checked-in boolean condition after replacing its entire
    # context; no service response or user input becomes executable code.
    actual = eval(expression, {"__builtins__": {}}, {})
    expected = (
        request_result == "success"
        and decision == "publish"
        and (validation == "success" or (mode != "new" and validation == "skipped"))
    )
    assert actual == expected
