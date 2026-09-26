"""Prove whether a main PR can reuse validation of its exact develop commit."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlencode

from scripts.ci_policy import POLICY, gate_errors, required_jobs

REQUIRED_JOBS = required_jobs("integration") | {POLICY}


def eligible_pull_request(event: dict, repository: str) -> bool:
    pr = event.get("pull_request", {})
    return bool(
        pr
        and not pr.get("draft", True)
        and pr.get("base", {}).get("ref") == "main"
        and pr.get("head", {}).get("ref") == "develop"
        and pr.get("head", {}).get("repo", {}).get("full_name") == repository
        and pr.get("base", {}).get("repo", {}).get("full_name") == repository
    )


def successful_develop_run(run: dict, jobs: list[dict], head_sha: str) -> bool:
    """A green workflow with skipped or missing validation is not proof."""
    if not (
        run.get("head_sha") == head_sha
        and run.get("head_branch") == "develop"
        and run.get("event") == "push"
        and run.get("path", "").split("@", 1)[0] == ".github/workflows/ci-develop.yml"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
    ):
        return False
    # The caller job is deliberately named validation in every branch workflow.
    normalized = [
        {**job, "name": job.get("name", "").removeprefix("validation / ")}
        for job in jobs
    ]
    for name in REQUIRED_JOBS:
        matches = [job for job in normalized if job.get("name") == name]
        if len(matches) != 1 or matches[0].get("conclusion") != "success":
            return False
        if "run_attempt" in matches[0] and matches[0]["run_attempt"] != run.get(
            "run_attempt"
        ):
            return False
    return not gate_errors(normalized, "integration")


def github_json(endpoint: str, *, paginate: bool = False):
    command = ["gh", "api", "--method", "GET", endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=90
    )
    if len(result.stdout) > 10_000_000:
        raise ValueError("Evidence response exceeds size bound")
    payload = json.loads(result.stdout)
    if paginate and (not isinstance(payload, list) or len(payload) > 100):
        raise ValueError("Evidence pagination exceeds bound")
    return payload


def git_tree(ref: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def find_reusable_run(event: dict, repository: str) -> tuple[bool, str]:
    if not eligible_pull_request(event, repository):
        return (
            False,
            "Normal validation: not a ready same-repository develop-to-main PR.",
        )
    head_sha = event["pull_request"]["head"]["sha"]
    if git_tree("HEAD") != git_tree(head_sha):
        return False, "Normal validation: the PR merge tree differs from develop."

    endpoint = f"repos/{repository}/actions/workflows/ci-develop.yml/runs"
    query = urlencode(
        dict(branch="develop", event="push", head_sha=head_sha, per_page=100)
    )
    payload = github_json(f"{endpoint}?{query}")
    runs = [run for run in payload["workflow_runs"] if run.get("head_sha") == head_sha]
    if not runs:
        return False, "Normal validation: no develop run exists for this commit."
    # Never hide a newer failed or running attempt behind an older green run.
    run = max(runs, key=lambda item: item["id"])
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        return False, "Normal validation: the latest develop run has not passed."
    pages = github_json(
        f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100",
        paginate=True,
    )
    jobs = [job for page in pages for job in page["jobs"]]
    if not successful_develop_run(run, jobs, head_sha):
        return False, "Normal validation: develop did not pass every required job."
    return (
        True,
        f"Reusing https://github.com/{repository}/actions/runs/{run['id']} "
        f"for {head_sha}.",
    )


def main() -> int:
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        reuse, reason = find_reusable_run(event, os.environ["GITHUB_REPOSITORY"])
    except OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError:
        reuse = False
        reason = "Normal validation: develop proof could not be verified."
    print(reason)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"reuse={'true' if reuse else 'false'}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(f"{reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
