"""Immutable release manifests, provenance checks, and safe release operations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from scripts.check_package_index_release import expected_distribution_hashes
from scripts.check_release_version import read_project_version
from scripts.ci_policy import POLICY, gate_errors

MANIFEST = "release-manifest.json"
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\Z")


def extract_verified_bundle(archive: bytes, digest: str, directory: Path) -> None:
    """Verify the actual immutable archive digest before trusting its manifest."""
    if len(archive) > 50_000_000:
        raise ValueError("Artifact size exceeds download bound")
    if "sha256:" + hashlib.sha256(archive).hexdigest() != digest:
        raise ValueError("Downloaded artifact digest mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        infos = bundle.infolist()
        names = [item.filename for item in infos]
        if (
            len(names) != 3
            or len(set(names)) != 3
            or MANIFEST not in names
            or any(Path(name).name != name or "\\" in name for name in names)
            or sum(item.file_size for item in infos) > 50_000_000
        ):
            raise ValueError("Unsafe or unexpected artifact contents")
        manifest = validate(
            json.loads(bundle.read(MANIFEST), object_pairs_hook=no_duplicate_keys)
        )
        if set(names) != set(manifest["files"]) | {MANIFEST}:
            raise ValueError("Archive contains unexpected distributions")
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError("Bundle destination must be empty")
        for name in names:
            (directory / name).write_bytes(bundle.read(name))
    verify_files(manifest, directory)


def download(directory: Path) -> None:
    repository = os.environ["GITHUB_REPOSITORY"]
    artifact_id = int(os.environ["SOURCE_ARTIFACT_ID"])
    digest = os.environ["SOURCE_ARTIFACT_DIGEST"]
    if artifact_id < 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Invalid artifact download identity")
    metadata = api(f"repos/{repository}/actions/artifacts/{artifact_id}")
    if metadata.get("digest") != digest or metadata.get("expired") is not False:
        raise ValueError("Artifact digest mismatch or expired artifact")
    if not 0 < metadata.get("size_in_bytes", 0) <= 50_000_000:
        raise ValueError("Artifact size exceeds download bound")
    archive = subprocess.run(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"],
        check=True,
        capture_output=True,
        timeout=90,
    ).stdout
    extract_verified_bundle(archive, digest, directory)


def command(*args: str) -> str:
    output = subprocess.run(
        args, check=True, text=True, capture_output=True, timeout=90
    ).stdout.strip()
    if len(output) > 10_000_000:
        raise ValueError("Command response exceeds evidence bound")
    return output


def api(endpoint: str):
    return json.loads(command("gh", "api", "--method", "GET", endpoint))


def validate(manifest: dict) -> dict:
    """Validate an untrusted manifest before using any field in a command."""
    fields = {
        "schema",
        "policy",
        "repository",
        "source_sha",
        "source_tree",
        "run_id",
        "run_attempt",
        "version",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("Invalid manifest fields")
    if (
        type(manifest["schema"]) is not int
        or manifest["schema"] != 1
        or manifest["policy"] != POLICY
    ):
        raise ValueError("Unsupported manifest policy/schema")
    for name in ("source_sha", "source_tree"):
        if not isinstance(manifest[name], str) or not SHA.fullmatch(manifest[name]):
            raise ValueError(f"Invalid {name}")
    if not isinstance(manifest["repository"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest["repository"]
    ):
        raise ValueError("Invalid repository")
    if not isinstance(manifest["version"], str) or not VERSION.fullmatch(
        manifest["version"]
    ):
        raise ValueError("Invalid release version")
    for name in ("run_id", "run_attempt"):
        if type(manifest[name]) is not int or manifest[name] < 1:
            raise ValueError(f"Invalid {name}")
    version = manifest["version"]
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        f"openpinch-{version}-py3-none-any.whl",
        f"openpinch-{version}.tar.gz",
    }:
        raise ValueError("Invalid distribution set")
    if any(
        not isinstance(value, str) or not DIGEST.fullmatch(value)
        for value in files.values()
    ):
        raise ValueError("Invalid distribution digest")
    return manifest


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key")
        result[key] = value
    return result


def load(path: Path) -> dict:
    return validate(json.loads(path.read_text(), object_pairs_hook=no_duplicate_keys))


def verify_files(manifest: dict, directory: Path) -> None:
    validate(manifest)
    actual = expected_distribution_hashes(
        directory, project="OpenPinch", version=manifest["version"]
    )
    if actual != manifest["files"]:
        raise ValueError("Distribution hashes differ from original manifest")
    if any((directory / name).is_symlink() for name in manifest["files"]):
        raise ValueError("Symlink distributions are not permitted")


def verify_origin(
    manifest: dict,
    run: dict,
    artifact: dict,
    repository: str,
    artifact_id: int,
    artifact_digest: str,
) -> None:
    """Bind bytes to an immutable artifact and trusted source run, not its name."""
    validate(manifest)
    if manifest["repository"] != repository:
        raise ValueError("Repository mismatch")
    expected_name = f"openpinch-dist-{manifest['run_id']}-{manifest['run_attempt']}"
    if (
        artifact.get("id") != artifact_id
        or artifact.get("digest") != artifact_digest
        or artifact.get("name") != expected_name
        or artifact.get("expired") is not False
        or artifact.get("workflow_run", {}).get("id") != manifest["run_id"]
        or artifact.get("workflow_run", {}).get("head_sha") != manifest["source_sha"]
    ):
        raise ValueError("Artifact provenance mismatch or expired artifact")
    if (
        run.get("id") != manifest["run_id"]
        or run.get("head_sha") != manifest["source_sha"]
        or run.get("run_attempt") != manifest["run_attempt"]
        or run.get("head_branch") != "main"
        or (run.get("event"), run.get("path", "").split("@", 1)[0])
        not in {
            ("push", ".github/workflows/ci-main.yml"),
            ("workflow_dispatch", ".github/workflows/ci-publish.yml"),
        }
    ):
        raise ValueError("Untrusted source run")


def next_transition(test_index: str, production: str, finalized: bool) -> str:
    """Pure recovery decision; conflicts can never authorize mutation."""
    if test_index not in {"absent", "partial", "complete"} or production not in {
        "absent",
        "partial",
        "complete",
    }:
        raise ValueError("Conflicting index state")
    if test_index != "complete":
        return "testpypi"
    if production != "complete":
        return "pypi"
    return "complete" if finalized else "finalize"


def create(directory: Path) -> dict:
    version = read_project_version(Path("pyproject.toml"))
    manifest = validate(
        {
            "schema": 1,
            "policy": POLICY,
            "repository": os.environ["GITHUB_REPOSITORY"],
            "source_sha": command("git", "rev-parse", "HEAD"),
            "source_tree": command("git", "rev-parse", "HEAD^{tree}"),
            "run_id": int(os.environ["GITHUB_RUN_ID"]),
            "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
            "version": version,
            "files": expected_distribution_hashes(
                directory, project="OpenPinch", version=version
            ),
        }
    )
    (directory / MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest


def verify_tag(manifest: dict) -> None:
    tag = f"v{manifest['version']}"
    if command("git", "cat-file", "-t", f"refs/tags/{tag}") != "tag":
        raise ValueError("Release tag must be annotated")
    if (
        command("git", "rev-parse", f"refs/tags/{tag}^{{commit}}")
        != manifest["source_sha"]
    ):
        raise ValueError("Tag points to a different source")


def stage(manifest: dict, directory: Path) -> bool:
    """Create missing immutable tag/assets, accepting only byte-identical state."""
    verify_files(manifest, directory)
    tag = f"v{manifest['version']}"
    refs = command("git", "tag", "--list", tag)
    if refs:
        verify_tag(manifest)
    else:
        command(
            "git",
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "tag",
            "-a",
            tag,
            manifest["source_sha"],
            "-m",
            f"OpenPinch {tag}",
        )
        command("git", "push", "origin", f"refs/tags/{tag}")
    # List by tag with a successful API response; network/auth failures are not absence.
    releases = json.loads(
        command(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{manifest['repository']}/releases?per_page=100",
        )
    )
    matching = [
        release for page in releases for release in page if release["tag_name"] == tag
    ]
    if len(matching) > 1:
        raise ValueError("Duplicate release identity")
    expected = set(manifest["files"]) | {MANIFEST}
    if not matching:
        command(
            "gh",
            "release",
            "create",
            tag,
            "--verify-tag",
            "--draft",
            "--generate-notes",
            "--title",
            f"OpenPinch {tag}",
        )
        assets = []
        draft = True
    else:
        if matching[0].get("prerelease") is not False:
            raise ValueError("Existing release must explicitly be a stable release")
        assets = matching[0]["assets"]
        draft = matching[0]["draft"]
    names = [item["name"] for item in assets]
    if len(names) != len(set(names)) or set(names) - expected:
        raise ValueError("Unexpected release assets")
    with tempfile.TemporaryDirectory() as temporary:
        for name in names:
            command(
                "gh", "release", "download", tag, "--pattern", name, "--dir", temporary
            )
            if (Path(temporary) / name).read_bytes() != (directory / name).read_bytes():
                raise ValueError("Existing release assets differ from original bundle")
    missing = expected - set(names)
    if missing and not draft:
        raise ValueError("Public release has missing assets; refusing modification")
    # No --clobber: concurrent/conflicting writes fail rather than overwrite.
    for name in sorted(missing):
        command("gh", "release", "upload", tag, str(directory / name))
    return draft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["create", "download", "verify", "stage", "finalize"]
    )
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    args = parser.parse_args()
    try:
        if args.command == "download":
            download(args.directory)
            return 0
        manifest = (
            create(args.directory)
            if args.command == "create"
            else load(args.directory / MANIFEST)
        )
        verify_files(manifest, args.directory)
        if args.command != "create":
            repository = os.environ["GITHUB_REPOSITORY"]
            artifact_id = int(os.environ["SOURCE_ARTIFACT_ID"])
            digest = os.environ["SOURCE_ARTIFACT_DIGEST"]
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("Invalid artifact digest")
            run = api(
                f"repos/{repository}/actions/runs/{manifest['run_id']}/attempts/{manifest['run_attempt']}"
            )
            artifact = api(f"repos/{repository}/actions/artifacts/{artifact_id}")
            verify_origin(manifest, run, artifact, repository, artifact_id, digest)
            pages = json.loads(
                command(
                    "gh",
                    "api",
                    "--paginate",
                    "--slurp",
                    f"repos/{repository}/actions/runs/{manifest['run_id']}/jobs?filter=latest&per_page=100",
                )
            )
            if len(pages) > 100:
                raise ValueError("Source evidence pagination exceeds bound")
            jobs = [
                {**job, "name": job.get("name", "").removeprefix("validation / ")}
                for page in pages
                for job in page["jobs"]
            ]
            # A failed validation lane may be rerun without rebuilding. GitHub's
            # latest filter combines retained successful jobs with their newest
            # replacements. Never fall back to old success behind a new failure.
            if (
                gate_errors(jobs, "full")
                or sum(
                    job["name"] == POLICY and job.get("conclusion") == "success"
                    for job in jobs
                )
                != 1
            ):
                raise ValueError(
                    "Source artifact lacks complete full-profile validation"
                )
            command(
                "git",
                "merge-base",
                "--is-ancestor",
                manifest["source_sha"],
                "origin/main",
            )
            if (
                command("git", "rev-parse", f"{manifest['source_sha']}^{{tree}}")
                != manifest["source_tree"]
            ):
                raise ValueError("Source tree mismatch")
        draft = False
        if args.command in {"stage", "finalize"}:
            draft = stage(manifest, args.directory)
        if args.command == "finalize":
            # Repeat exact external verification immediately before finalization.
            from scripts.check_package_index_release import main as verify_index

            for endpoint in ("https://test.pypi.org/pypi", "https://pypi.org/pypi"):
                if verify_index(
                    [
                        "--index-url",
                        endpoint,
                        "--version",
                        manifest["version"],
                        "--dist-dir",
                        str(args.directory),
                        "--require-complete",
                    ]
                ):
                    raise ValueError("Index verification failed before finalization")
            if draft:
                # Leave latest selection to GitHub: recovery may publish an
                # older version after a newer release has already completed.
                command(
                    "gh",
                    "release",
                    "edit",
                    f"v{manifest['version']}",
                    "--draft=false",
                )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        # Avoid command output containing tokens or credential-bearing URLs.
        print(
            f"Release operation failed: {type(exc).__name__}: "
            + (
                str(exc)
                if isinstance(exc, ValueError)
                else "check source/artifact identity and service access"
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
