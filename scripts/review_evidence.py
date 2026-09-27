"""Verify review evidence against immutable build bytes before reusing test lanes."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from scripts import release_preparation as prep
from scripts.ci_policy import required_jobs

# Package/metadata and documentation checks always execute for the new version.
REUSABLE = frozenset(
    {"hpr-tespy-tests", "performance-tests", "solver-tests", "workflow-lint"}
)


def find_review_run(repo: str, pr: int, head: str) -> int:
    runs = [
        r
        for page in prep.pages(
            f"repos/{repo}/actions/workflows/ci-pull-request.yml/runs?event=pull_request&head_sha={head}&per_page=100"
        )
        for r in page["workflow_runs"]
        if r["head_sha"] == head
        and r.get("head_branch") == "develop"
        and (
            not r.get("pull_requests")
            or any(p["number"] == pr for p in r["pull_requests"])
        )
    ]
    if not runs:
        raise ValueError("No matching review validation")
    for run in sorted(runs, key=lambda r: r["id"], reverse=True):
        if not _metadata_only_run(repo, run):
            return run["id"]
    raise ValueError("No matching review validation")


def _metadata_only_run(repo: str, run: dict) -> bool:
    """Ignore only explicit completed metadata events, never missing proof."""
    if run.get("status") != "completed" or run.get("conclusion") not in {
        "success",
        "failure",
    }:
        return False
    jobs = [
        job
        for page in prep.pages(
            f"repos/{repo}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100"
        )
        for job in page["jobs"]
    ]
    for name in ("plan", "review-metadata-only"):
        matches = [job for job in jobs if job["name"] == name]
        if (
            len(matches) != 1
            or matches[0].get("conclusion") != "success"
            or matches[0].get("run_attempt") != run.get("run_attempt")
            or run.get("run_attempt") is None
        ):
            return False
    return all(
        job.get("conclusion") == "skipped"
        for job in jobs
        if job["name"] == "validation" or job["name"].startswith("validation / ")
    )


def verify_review_run(
    repo: str, run_id: int, pr: int, head: str, base: str
) -> tuple[str, str]:
    from scripts import release_manifest as release

    run = prep.api(f"repos/{repo}/actions/runs/{run_id}")
    if (
        run.get("status") != "completed"
        or run.get("conclusion") not in {"success", "failure"}
        or run.get("event") != "pull_request"
        or run.get("path", "").split("@")[0] != ".github/workflows/ci-pull-request.yml"
        or run.get("head_sha") != head
        or run.get("head_branch") != "develop"
        or run.get("head_repository", {}).get("full_name") != repo
        or (
            run.get("pull_requests")
            and not any(
                p["number"] == pr
                and p["head"]["sha"] == head
                and p["base"]["sha"] == base
                for p in run.get("pull_requests", [])
            )
        )
    ):
        raise ValueError("Review run identity mismatch")
    if find_review_run(repo, pr, head) != run_id:
        raise ValueError("Newer review validation exists")
    jobs = [
        j
        for page in prep.pages(
            f"repos/{repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100"
        )
        for j in page["jobs"]
    ]
    for lane in required_jobs("full") | {"delivery-v1"}:
        matches = [j for j in jobs if j["name"] == f"validation / {lane}"]
        if (
            len(matches) != 1
            or matches[0].get("conclusion") != "success"
            or matches[0].get("run_attempt", run["run_attempt"]) != run["run_attempt"]
        ):
            raise ValueError(f"Review lacks concrete successful lane: {lane}")
    artifacts = [
        a
        for page in prep.pages(
            f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
        )
        for a in page["artifacts"]
        if a["name"] == f"openpinch-dist-{run_id}-{run['run_attempt']}"
    ]
    if len(artifacts) != 1 or artifacts[0].get("expired") is not False:
        raise ValueError("Review build artifact missing or ambiguous")
    artifact = artifacts[0]
    if (
        artifact.get("workflow_run", {}).get("id") != run_id
        or artifact.get("workflow_run", {}).get("head_sha") != head
    ):
        raise ValueError("Review artifact belongs to another workflow candidate")
    with tempfile.TemporaryDirectory() as temporary:
        release.download(Path(temporary), artifact["id"], artifact["digest"])
        manifest = release.load(Path(temporary) / release.MANIFEST)
    if (
        manifest["repository"] != repo
        or manifest["run_id"] != run_id
        or manifest["run_attempt"] != run["run_attempt"]
    ):
        raise ValueError("Review artifact identity mismatch")
    source = manifest["source_sha"]
    prep.command("git", "fetch", "--no-tags", "origin", source)
    parents = prep.command("git", "rev-list", "--parents", "-n", "1", source).split()[
        1:
    ]
    if (
        parents != [base, head]
        or prep.command("git", "rev-parse", f"{source}^{{tree}}")
        != manifest["source_tree"]
    ):
        raise ValueError("Review artifact is not the tested PR merge tree")
    return source, manifest["source_tree"]


def reusable(ref: str, repo: str) -> dict:
    record = prep.verify_prepared(ref, repo)
    source, tree = verify_review_run(
        repo, record["run"], record["pr"], record["head"], record["base"]
    )
    if tree != record["tree"] or source != record["source"]:
        raise ValueError("Recorded review tree mismatch")
    # Never borrow proof across policy/workflow changes relative to trusted base.
    if prep.command(
        "git",
        "diff",
        "--name-only",
        record["base"],
        tree,
        "--",
        ".github/workflows",
        "scripts",
    ):
        raise ValueError("Delivery policy changed: execute fresh validation")
    return record


def main() -> int:
    reuse = False
    try:
        reusable(prep.command("git", "rev-parse", "HEAD"), prep.repository())
        reuse = True
        reason = "Verified review evidence: reuse unaffected expensive lanes"
    except ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError:
        reason = "No reusable review evidence: execute normal validation"
    print(reason)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"reuse-review={'true' if reuse else 'false'}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(reason + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
