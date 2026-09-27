"""Contracts for the two GitHub Actions workflows (ci.yml and release.yml)."""

import yaml

from tests.support.paths import REPOSITORY_ROOT

WORKFLOW_DIR = REPOSITORY_ROOT / ".github" / "workflows"


def workflow(name):
    # BaseLoader keeps GitHub's `on` key instead of YAML 1.1's bool coercion.
    return yaml.load((WORKFLOW_DIR / name).read_text(), Loader=yaml.BaseLoader)


def runs(job):
    return "\n".join(step.get("run", "") for step in job.get("steps", []))


def test_only_the_two_pipeline_workflows_exist():
    assert sorted(p.name for p in WORKFLOW_DIR.iterdir()) == ["ci.yml", "release.yml"]


def test_ci_triggers_and_single_required_check():
    ci = workflow("ci.yml")
    assert set(ci["on"]) == {"pull_request", "push", "workflow_dispatch", "workflow_call"}
    assert ci["on"]["pull_request"]["branches"] == ["develop", "main"]
    assert ci["on"]["push"]["branches"] == ["develop"]
    # Metadata edits must never re-run (and so never re-report) the gate.
    assert "types" not in ci["on"]["pull_request"]
    assert ci["permissions"] == {"contents": "read"}

    gate = ci["jobs"]["ci-ok"]
    assert gate["name"] == "CI OK"
    assert gate["if"] == "${{ always() }}"
    assert set(gate["needs"]) == set(ci["jobs"]) - {"ci-ok"}
    assert '"solver"' in runs(gate) and '"version"' in runs(gate)


def test_main_only_lanes():
    jobs = workflow("ci.yml")["jobs"]
    assert "refs/heads/main" in jobs["solver"]["if"]
    assert "github.base_ref == 'main'" in jobs["solver"]["if"]
    assert jobs["version"]["if"] == (
        "${{ github.event_name == 'pull_request' && github.base_ref == 'main' }}"
    )
    assert "check_release_version.py --base-pyproject" in runs(jobs["version"])


def test_every_ci_lane_is_present():
    jobs = workflow("ci.yml")["jobs"]
    lanes = {row["lane"] for row in jobs["tests"]["strategy"]["matrix"]["include"]}
    assert lanes == {"unit", "docs", "tespy", "performance"}
    smoke = jobs["smoke"]["strategy"]["matrix"]
    assert set(smoke["surface"]) == {
        "core", "dashboard", "notebook", "brayton_cycle", "tespy", "synthesis"
    }
    assert {row["os"] for row in smoke["include"]} == {"windows-latest", "macos-latest"}
    assert "SOURCE_DATE_EPOCH" in runs(jobs["build"])


def test_release_publishes_the_bytes_ci_built():
    release = workflow("release.yml")
    assert release["on"]["push"]["branches"] == ["main"]
    assert release["permissions"] == {"contents": "read"}
    assert release["concurrency"]["cancel-in-progress"] == "false"
    jobs = release["jobs"]
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["plan"]["needs"] == "ci"
    assert "scripts/release_plan.py" in runs(jobs["plan"])

    pypi = jobs["pypi"]
    assert pypi["environment"]["name"] == "pypi"
    assert pypi["permissions"] == {"id-token": "write"}
    assert "actions/checkout" not in str(pypi["steps"])  # no repo code runs here
    downloads = [s for s in pypi["steps"] if "download-artifact" in s.get("uses", "")]
    assert downloads[0]["with"]["name"] == "dist"
    publish = [s for s in pypi["steps"] if "pypi-publish" in s.get("uses", "")]
    assert publish[0]["with"] == {"skip-existing": "true"}

    github_release = jobs["github-release"]
    assert github_release["needs"] == ["plan", "pypi"]
    assert github_release["permissions"] == {"contents": "write"}
    assert '--target "$GITHUB_SHA"' in runs(github_release)
    # An interrupted release is completed on re-run, then verified.
    assert "--draft=false" in runs(github_release)
    assert "gh release view" in runs(github_release)


def test_no_job_keeps_checkout_credentials():
    for name in ("ci.yml", "release.yml"):
        for job in workflow(name)["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] == "false"
