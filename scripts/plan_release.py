"""Read-only automatic release planning from authenticated main-run evidence."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from scripts import release_manifest as release
from scripts.check_package_index_release import inspect_release

INDEXES = ("https://test.pypi.org/pypi", "https://pypi.org/pypi")
COORDINATOR = ".github/workflows/ci-prepare-release.yml"


def eligible_event(run: dict, repository: str) -> bool:
    """Only a successful main push from this repository can request publication."""
    return (
        run.get("conclusion") == "success"
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
        and run.get("head_repository", {}).get("full_name") == repository
    )


def pages(endpoint: str) -> list:
    result = json.loads(release.command("gh", "api", "--paginate", "--slurp", endpoint))
    if not isinstance(result, list) or len(result) > 100:
        raise ValueError("Release evidence pagination exceeds bound")
    return result


def positive_id(value) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid source identity")
    return value


def source_artifact(run_id: int, attempt: int | None = None) -> dict:
    repository = os.environ["GITHUB_REPOSITORY"]
    positive_id(run_id)
    candidates = [
        item
        for page in pages(
            f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100"
        )
        for item in page["artifacts"]
        if re.fullmatch(rf"openpinch-dist-{run_id}-[1-9][0-9]*", item.get("name", ""))
        and (attempt is None or item["name"] == f"openpinch-dist-{run_id}-{attempt}")
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Missing or ambiguous original artifact; verified recovery required"
        )
    artifact = candidates[0]
    positive_id(artifact["id"])
    if artifact.get("expired") is not False or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", artifact.get("digest", "")
    ):
        raise ValueError(
            "Expired or invalid original artifact; verified recovery required"
        )
    return artifact


def verified_bundle(artifact: dict, directory: Path) -> dict:
    release.download(directory, artifact["id"], artifact["digest"])
    manifest = release.load(directory / release.MANIFEST)
    release.verify_source(manifest, artifact["id"], artifact["digest"])
    return manifest


def require_stable(existing: dict) -> None:
    if existing.get("draft") is not False or existing.get("prerelease") is not False:
        raise ValueError(
            "Existing release is not stable and complete; verified recovery required"
        )


def coordinator_bootstrap(source: str, version: str) -> bool:
    """Prove this main merge only installs the coordinator over an old release."""
    if not release.SHA.fullmatch(source):
        raise ValueError("Invalid bootstrap source identity")
    current = release.command(
        "git", "ls-tree", "-r", "--name-only", source, "--", COORDINATOR
    )
    parents = release.command("git", "rev-list", "--parents", "-n", "1", source).split()
    # Limit the exception to a normal two-parent PR merge. This avoids guessing
    # the previous main identity for squash/rebase histories after the fact.
    if current != COORDINATOR or len(parents) != 3 or parents[0] != source:
        return False
    previous_main = parents[1]
    if not release.SHA.fullmatch(previous_main):
        raise ValueError("Invalid bootstrap parent identity")
    if release.command(
        "git", "ls-tree", "-r", "--name-only", previous_main, "--", COORDINATOR
    ):
        return False
    try:
        release.command(
            "git", "merge-base", "--is-ancestor", f"refs/tags/v{version}", previous_main
        )
    except subprocess.CalledProcessError:
        return False
    return True


def index_states(manifest: dict, directory: Path) -> list[str]:
    release.verify_files(manifest, directory)
    return [
        inspect_release(
            index_url=endpoint,
            project="OpenPinch",
            version=manifest["version"],
            expected_files=manifest["files"],
            timeout=20,
        )
        for endpoint in INDEXES
    ]


def completed_release(existing: dict, version: str, root: Path) -> None:
    """Prove completion against the original bundle, not today's same-version build."""
    require_stable(existing)
    assets = existing["assets"]
    names = [item["name"] for item in assets]
    if len(names) != 3 or len(set(names)) != 3 or release.MANIFEST not in names:
        raise ValueError(
            "Missing or conflicting release assets; verified recovery required"
        )
    if any(
        type(item.get("size")) is not int or not 0 < item["size"] <= 50_000_000
        for item in assets
    ):
        raise ValueError("Invalid release asset size")
    saved = root / "released"
    saved.mkdir()
    tag = f"v{version}"
    release.command(
        "gh",
        "release",
        "download",
        tag,
        "--pattern",
        release.MANIFEST,
        "--dir",
        str(saved),
    )
    manifest = release.load(saved / release.MANIFEST)
    if (
        manifest["version"] != version
        or manifest["repository"] != os.environ["GITHUB_REPOSITORY"]
    ):
        raise ValueError("Original release identity conflict")
    if set(names) != set(manifest["files"]) | {release.MANIFEST}:
        raise ValueError("Original release asset names conflict")
    artifact = source_artifact(manifest["run_id"], manifest["run_attempt"])
    bundle = root / "original"
    original = verified_bundle(artifact, bundle)
    if original != manifest:
        raise ValueError("Original release manifest conflict")
    release.verify_tag(original)
    for name in manifest["files"]:
        release.command(
            "gh", "release", "download", tag, "--pattern", name, "--dir", str(saved)
        )
    if any(
        (saved / name).read_bytes() != (bundle / name).read_bytes() for name in names
    ):
        raise ValueError("Original release asset bytes conflict")
    if index_states(original, bundle) != ["complete", "complete"]:
        raise ValueError("Incomplete package indexes; verified recovery required")


def plan(event: dict, root: Path) -> dict:
    repository = os.environ["GITHUB_REPOSITORY"]
    trigger = event.get("workflow_run", {})
    if not eligible_event(trigger, repository):
        return {
            "action": "ignore",
            "reason": "Not a successful same-repository main push",
        }
    run_id = positive_id(trigger["id"])
    run = release.api(f"repos/{repository}/actions/runs/{run_id}")
    if (
        not eligible_event(run, repository)
        or run.get("status") != "completed"
        or run.get("path") != ".github/workflows/ci-main.yml"
        or run.get("repository", {}).get("full_name") != repository
        or run.get("id") != run_id
        or run.get("head_sha") != trigger.get("head_sha")
        or run.get("run_attempt") != trigger.get("run_attempt")
    ):
        raise ValueError("Source run no longer matches successful main evidence")
    artifact = source_artifact(run_id)
    directory = root / "candidate"
    manifest = verified_bundle(artifact, directory)
    if (
        manifest["source_sha"] != run["head_sha"]
        or manifest["run_id"] != run_id
        or manifest["run_attempt"] > positive_id(run["run_attempt"])
    ):
        raise ValueError("Source bundle does not match triggering run")
    version = manifest["version"]
    matches = [
        item
        for page in pages(f"repos/{repository}/releases?per_page=100")
        for item in page
        if item["tag_name"] == f"v{version}"
    ]
    if len(matches) > 1:
        raise ValueError("Duplicate release identity")
    if matches:
        completed_release(matches[0], version, root)
        return {
            "action": "complete",
            "version": version,
            "reason": "Original stable release and both indexes verified",
        }
    if release.command("git", "tag", "--list", f"v{version}"):
        if coordinator_bootstrap(manifest["source_sha"], version):
            return {
                "action": "ignore",
                "version": version,
                "reason": "Coordinator bootstrap retains the existing release version",
            }
        raise ValueError("Version already tagged; verified recovery required")
    if index_states(manifest, directory) != ["absent", "absent"]:
        raise ValueError(
            "Version already present on a package index; verified recovery required"
        )
    return {
        "action": "publish",
        "version": version,
        "source_sha": manifest["source_sha"],
        "run_id": run_id,
        "run_attempt": manifest["run_attempt"],
        "artifact_id": artifact["id"],
        "artifact_digest": artifact["digest"],
        "reason": "Verified new version from successful main validation",
    }


def main() -> int:
    try:
        if os.environ.get("GITHUB_EVENT_NAME") != "workflow_run":
            raise ValueError("Automatic planner requires workflow_run")
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        with tempfile.TemporaryDirectory() as temporary:
            result = plan(event, Path(temporary))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            for key, value in result.items():
                if key != "reason":
                    if not re.fullmatch(r"[a-zA-Z0-9:._-]+", str(value)):
                        raise ValueError("Unsafe release output")
                    output.write(f"{key}={value}\n")
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write(
                f"Automatic release: {result['action']} — {result['reason']}\n"
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        # Never echo subprocess output or service responses containing credentials.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(
            f"Automatic release blocked: {message}. "
            "Inspect evidence and use verified recovery.",
            file=sys.stderr,
        )
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a") as summary:
                summary.write(
                    "Automatic release: blocked — inspect logs "
                    "and use verified recovery.\n"
                )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
