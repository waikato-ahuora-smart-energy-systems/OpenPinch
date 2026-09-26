"""CI reuse must prove the exact source and all validation, or run fresh checks."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy

import pytest
from hypothesis import given, seed
from hypothesis import strategies as st

from scripts import reuse_develop_ci as reuse

SHA = "a" * 40
REPOSITORY = "example/OpenPinch"


def event():
    return {
        "pull_request": {
            "draft": False,
            "head": {"ref": "develop", "sha": SHA, "repo": {"full_name": REPOSITORY}},
            "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
        }
    }


def successful_run():
    return {
        "id": 123,
        "head_sha": SHA,
        "head_branch": "develop",
        "event": "push",
        "path": ".github/workflows/ci-develop.yml",
        "status": "completed",
        "conclusion": "success",
    }


def successful_jobs():
    return [
        {"name": name, "conclusion": "success"} for name in sorted(reuse.REQUIRED_JOBS)
    ]


def test_only_ready_same_repository_develop_to_main_is_eligible():
    assert reuse.eligible_pull_request(event(), REPOSITORY)
    assert not reuse.eligible_pull_request({}, REPOSITORY)
    for path, value in [
        (("draft",), True),
        (("head", "ref"), "feature"),
        (("base", "ref"), "develop"),
        (("head", "repo", "full_name"), "fork/OpenPinch"),
    ]:
        changed = event()
        parent = changed["pull_request"]
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        assert not reuse.eligible_pull_request(changed, REPOSITORY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_sha", "b" * 40),
        ("head_branch", "main"),
        ("event", "pull_request"),
        ("path", ".github/workflows/other.yml"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("conclusion", "cancelled"),
    ],
)
def test_wrong_commit_or_run_is_not_reused(field, value):
    run = successful_run()
    run[field] = value
    assert not reuse.successful_develop_run(run, successful_jobs(), SHA)


@seed(20260918)
@given(
    st.sampled_from(sorted(reuse.REQUIRED_JOBS)),
    st.sampled_from(["failure", "skipped", "cancelled", "timed_out", None, "missing"]),
)
def test_any_missing_or_unsuccessful_required_job_prevents_reuse(name, conclusion):
    jobs = successful_jobs()
    assert reuse.successful_develop_run(successful_run(), jobs, SHA)
    if conclusion == "missing":
        jobs = [job for job in jobs if job["name"] != name]
    else:
        next(job for job in jobs if job["name"] == name)["conclusion"] = conclusion
    assert not reuse.successful_develop_run(successful_run(), jobs, SHA)


def test_duplicate_job_results_are_not_accepted():
    jobs = successful_jobs()
    jobs.append(deepcopy(jobs[0]))
    assert not reuse.successful_develop_run(successful_run(), jobs, SHA)


def test_wrong_job_attempt_invalidates_reuse():
    run = {**successful_run(), "run_attempt": 2}
    jobs = [
        {**job, "name": "validation / " + job["name"], "run_attempt": 2}
        for job in successful_jobs()
    ]
    assert reuse.successful_develop_run(run, jobs, SHA)
    jobs[0]["run_attempt"] = 1
    assert not reuse.successful_develop_run(run, jobs, SHA)


def test_excessive_pagination_is_not_complete_evidence(monkeypatch):
    monkeypatch.setattr(
        reuse.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps([{"jobs": []}] * 101)
        ),
    )
    with pytest.raises(ValueError, match="pagination"):
        reuse.github_json(
            "repos/example/OpenPinch/actions/runs/123/jobs", paginate=True
        )


def test_exact_source_and_paginated_successful_jobs_are_reused(monkeypatch):
    calls = []
    jobs = successful_jobs()

    def api(endpoint, *, paginate=False):
        calls.append((endpoint, paginate))
        if "ci-develop.yml/runs" in endpoint:
            return {"workflow_runs": [successful_run()]}
        return [{"jobs": jobs[:5]}, {"jobs": jobs[5:]}]

    monkeypatch.setattr(reuse, "github_json", api)
    monkeypatch.setattr(reuse, "git_tree", lambda ref: "same-tree")
    accepted, reason = reuse.find_reusable_run(event(), REPOSITORY)
    assert accepted
    assert "/actions/runs/123" in reason
    assert f"head_sha={SHA}" in calls[0][0]
    assert calls[1][1] is True
    assert "filter=latest" in calls[1][0]


def test_changed_merge_tree_never_uses_develop(monkeypatch):
    monkeypatch.setattr(reuse, "git_tree", lambda ref: ref)
    monkeypatch.setattr(
        reuse, "github_json", lambda *args, **kwargs: pytest.fail("API lookup")
    )
    accepted, reason = reuse.find_reusable_run(event(), REPOSITORY)
    assert not accepted
    assert "merge tree differs" in reason


@pytest.mark.parametrize(
    "state", ["absent", "stale", "running", "failed", "skipped_jobs"]
)
def test_unproven_develop_results_run_normal_validation(monkeypatch, state):
    latest = successful_run()
    latest["id"] = 124
    if state == "running":
        latest.update(status="in_progress", conclusion=None)
    if state == "failed":
        latest["conclusion"] = "failure"
    runs = [successful_run(), latest]
    if state == "absent":
        runs = []
    if state == "stale":
        runs = [{**latest, "head_sha": "b" * 40}]
    jobs = [{"name": name, "conclusion": "skipped"} for name in reuse.REQUIRED_JOBS]
    monkeypatch.setattr(reuse, "git_tree", lambda ref: "same-tree")
    monkeypatch.setattr(
        reuse,
        "github_json",
        lambda endpoint, **kwargs: (
            {"workflow_runs": runs}
            if "ci-develop.yml/runs" in endpoint
            else [{"jobs": jobs}]
        ),
    )
    assert reuse.find_reusable_run(event(), REPOSITORY)[0] is False


@pytest.mark.parametrize("error", [OSError("unavailable"), ValueError("malformed")])
def test_lookup_error_emits_false_and_allows_fresh_validation(
    monkeypatch, tmp_path, error
):
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(event()))
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    for name, value in {
        "GITHUB_EVENT_PATH": str(event_file),
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
    }.items():
        monkeypatch.setenv(name, value)

    def fail(*args):
        raise error

    monkeypatch.setattr(reuse, "find_reusable_run", fail)
    assert reuse.main() == 0
    assert output.read_text() == "reuse=false\n"
    assert "could not be verified" in summary.read_text()


def test_github_request_uses_pagination_and_bounded_timeout(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout='[{"jobs": []}]')

    monkeypatch.setattr(reuse.subprocess, "run", run)
    assert reuse.github_json(
        "repos/example/OpenPinch/actions/runs/123/jobs", paginate=True
    ) == [{"jobs": []}]
    assert calls[0][0][-2:] == ["--paginate", "--slurp"]
    assert calls[0][1]["timeout"] == 90
