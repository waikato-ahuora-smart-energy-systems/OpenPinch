"""Immutable bundle contracts and stateful interrupted-release recovery."""

import hashlib
import io
import json
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from scripts import release_manifest as release
from scripts.ci_policy import POLICY, required_jobs
from scripts.release_manifest import (
    next_transition,
    no_duplicate_keys,
    validate,
    verify_origin,
)
from tests.strategies.delivery import manifests


def write_bundle(directory, manifest):
    manifest = deepcopy(manifest)
    for name in manifest["files"]:
        data = name.encode()
        (directory / name).write_bytes(data)
        manifest["files"][name] = hashlib.sha256(data).hexdigest()
    (directory / release.MANIFEST).write_text(json.dumps(manifest))
    return manifest


@given(manifests(), st.integers(min_value=0, max_value=3))
@settings(
    deadline=2000
)  # Real temporary-file I/O includes variable filesystem latency.
def test_real_staging_resumes_interrupted_uploads_without_replacing_bytes(
    manifest, interruption
):
    with (
        tempfile.TemporaryDirectory() as temporary,
        pytest.MonkeyPatch.context() as patch,
    ):
        directory = Path(temporary)
        manifest = write_bundle(directory, manifest)
        assets = {}
        exists = False
        interrupted = False

        def command(*args):
            nonlocal exists, interrupted
            if args[:3] == ("git", "tag", "--list"):
                return "v" + manifest["version"]
            if args[:3] == ("git", "cat-file", "-t"):
                return "tag"
            if args[:2] == ("git", "rev-parse"):
                return manifest["source_sha"]
            if args[:2] == ("gh", "api"):
                return json.dumps(
                    [
                        [
                            {
                                "tag_name": "v" + manifest["version"],
                                "draft": True,
                                "assets": [{"name": name} for name in assets],
                            }
                        ]
                        if exists
                        else []
                    ]
                )
            if args[:3] == ("gh", "release", "create"):
                exists = True
            elif args[:3] == ("gh", "release", "upload"):
                if len(assets) == interruption and not interrupted:
                    interrupted = True
                    raise OSError("simulated interruption")
                path = Path(args[4])
                assert path.name not in assets
                assets[path.name] = path.read_bytes()
            elif args[:3] == ("gh", "release", "download"):
                (Path(args[7]) / args[5]).write_bytes(assets[args[5]])
            else:
                raise AssertionError(args)
            return ""

        patch.setattr(release, "command", command)
        try:
            release.stage(manifest, directory)
        except OSError:
            pass
        release.stage(manifest, directory)
        before = dict(assets)
        release.stage(manifest, directory)
        assert assets == before == {p.name: p.read_bytes() for p in directory.iterdir()}
        assets[release.MANIFEST] = b"conflict"
        with pytest.raises(ValueError, match="differ"):
            release.stage(manifest, directory)


@given(manifests())
@settings(deadline=2000)  # Archive extraction exercises real filesystem boundaries.
def test_verified_archive_round_trip_and_tampering(manifest):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        manifest = write_bundle(source, manifest)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            for path in source.iterdir():
                zipped.writestr(path.name, path.read_bytes())
        data = archive.getvalue()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        release.extract_verified_bundle(data, digest, root / "target")
        assert release.load(root / "target" / release.MANIFEST) == manifest
        with pytest.raises(ValueError, match="digest mismatch"):
            release.extract_verified_bundle(data + b"changed", digest, root / "other")
        with pytest.raises(ValueError, match="empty"):
            release.extract_verified_bundle(data, digest, root / "target")


@pytest.mark.parametrize(
    "bad_name", ["../escape", "/absolute", "sub/file", "sub\\file"]
)
def test_unsafe_archive_rejected_before_extraction(tmp_path, bad_name):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in [release.MANIFEST, "file.whl", bad_name]:
            zipped.writestr(name, b"{}")
    data = archive.getvalue()
    with pytest.raises(ValueError, match="Unsafe"):
        release.extract_verified_bundle(
            data, "sha256:" + hashlib.sha256(data).hexdigest(), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "metadata",
    [
        {"expired": True, "digest": "sha256:" + "a" * 64, "size_in_bytes": 10},
        {"expired": False, "digest": "sha256:" + "b" * 64, "size_in_bytes": 10},
        {"expired": False, "digest": "sha256:" + "a" * 64, "size_in_bytes": 50_000_001},
    ],
)
def test_download_rejects_untrusted_metadata_before_fetch(
    monkeypatch, tmp_path, metadata
):
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/OpenPinch")
    monkeypatch.setenv("SOURCE_ARTIFACT_ID", "99")
    monkeypatch.setenv("SOURCE_ARTIFACT_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setattr(release, "api", lambda _: metadata)
    monkeypatch.setattr(
        release.subprocess, "run", lambda *a, **k: pytest.fail("unexpected download")
    )
    with pytest.raises(ValueError):
        release.download(tmp_path)


@given(manifests())
def test_manifest_round_trip_and_exact_fields(manifest):
    assert (
        validate(json.loads(json.dumps(manifest), object_pairs_hook=no_duplicate_keys))
        == manifest
    )
    changed = {**manifest, "unexpected": True}
    with pytest.raises(ValueError):
        validate(changed)


@given(manifests())
def test_provenance_cannot_substitute_identity(manifest):
    run = {
        "id": manifest["run_id"],
        "run_attempt": manifest["run_attempt"],
        "head_sha": manifest["source_sha"],
        "head_branch": "main",
        "event": "workflow_dispatch",
        "path": ".github/workflows/ci-publish.yml",
    }
    artifact = {
        "id": 99,
        "digest": "sha256:" + "a" * 64,
        "expired": False,
        "name": f"openpinch-dist-{manifest['run_id']}-{manifest['run_attempt']}",
        "workflow_run": {"id": manifest["run_id"], "head_sha": manifest["source_sha"]},
    }
    verify_origin(manifest, run, artifact, "example/OpenPinch", 99, artifact["digest"])
    for key, value in [
        ("expired", True),
        ("id", 100),
        ("digest", "wrong"),
        ("name", "other"),
    ]:
        changed = deepcopy(artifact)
        changed[key] = value
        with pytest.raises(ValueError):
            verify_origin(
                manifest, run, changed, "example/OpenPinch", 99, artifact["digest"]
            )
    with pytest.raises(ValueError):
        verify_origin(
            manifest,
            {**run, "head_branch": "feature"},
            artifact,
            "example/OpenPinch",
            99,
            artifact["digest"],
        )


@given(
    st.sampled_from(["absent", "partial", "complete"]),
    st.sampled_from(["absent", "partial", "complete"]),
    st.booleans(),
)
def test_recovery_matches_reference(test, production, finalized):
    expected = (
        "testpypi"
        if test != "complete"
        else "pypi"
        if production != "complete"
        else "complete"
        if finalized
        else "finalize"
    )
    assert next_transition(test, production, finalized) == expected
    assert next_transition(test, production, finalized) == next_transition(
        test, production, finalized
    )


def test_conflicts_and_duplicate_keys_fail_closed():
    with pytest.raises(ValueError):
        next_transition("conflict", "complete", False)
    with pytest.raises(ValueError):
        json.loads('{"schema": 1, "schema": 1}', object_pairs_hook=no_duplicate_keys)


@pytest.mark.parametrize("failed_lane", [None, "solver-tests", "test", POLICY])
@pytest.mark.parametrize("already_public", [False, True])
@pytest.mark.parametrize(
    "operation,index_failure",
    [("verify", None), ("finalize", None), ("finalize", 0), ("finalize", 1)],
)
def test_cli_proof_uses_latest_validation_without_rebuilding_original_attempt(
    tmp_path, monkeypatch, failed_lane, operation, index_failure, already_public
):
    manifest = write_bundle(
        tmp_path,
        {
            "schema": 1,
            "policy": POLICY,
            "repository": "example/OpenPinch",
            "source_sha": "a" * 40,
            "source_tree": "b" * 40,
            "run_id": 123,
            "run_attempt": 1,
            "version": "1.2.3",
            "files": {
                "openpinch-1.2.3-py3-none-any.whl": "0" * 64,
                "openpinch-1.2.3.tar.gz": "0" * 64,
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv", ["release", operation, "--directory", str(tmp_path)]
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", manifest["repository"])
    monkeypatch.setenv("SOURCE_ARTIFACT_ID", "99")
    monkeypatch.setenv("SOURCE_ARTIFACT_DIGEST", "sha256:" + "c" * 64)

    def api(endpoint):
        if endpoint.endswith("/attempts/1"):
            return {
                "id": 123,
                "run_attempt": 1,
                "head_sha": manifest["source_sha"],
                "head_branch": "main",
                "event": "workflow_dispatch",
                "path": ".github/workflows/ci-publish.yml",
            }
        assert endpoint.endswith("/artifacts/99")
        return {
            "id": 99,
            "digest": "sha256:" + "c" * 64,
            "expired": False,
            "name": "openpinch-dist-123-1",
            "workflow_run": {"id": 123, "head_sha": manifest["source_sha"]},
        }

    published = []
    verified = []

    def verify_index(args):
        verified.append(args)
        return int(len(verified) - 1 == index_failure)

    monkeypatch.setattr("scripts.check_package_index_release.main", verify_index)
    monkeypatch.setattr(release, "stage", lambda *args: not already_public)

    def command(*args):
        if args[:3] == ("gh", "release", "edit"):
            assert len(verified) == 2
            published.append(args)
            return ""
        if args[:2] == ("gh", "api"):
            assert args[-1].endswith("/runs/123/jobs?filter=latest&per_page=100")
            return json.dumps(
                [
                    {
                        "jobs": [
                            {
                                "name": "validation / " + name,
                                "conclusion": "failure"
                                if name == failed_lane
                                else "success",
                                "run_attempt": 2
                                if name in {"solver-tests", POLICY}
                                else 1,
                            }
                            for name in required_jobs("full") | {POLICY}
                        ]
                    }
                ]
            )
        if args[:2] == ("git", "merge-base"):
            return ""
        assert args[:2] == ("git", "rev-parse")
        return manifest["source_tree"]

    monkeypatch.setattr(release, "api", api)
    monkeypatch.setattr(release, "command", command)
    failed = failed_lane is not None or index_failure is not None
    assert release.main() == failed
    assert bool(published) == (
        operation == "finalize" and not failed and not already_public
    )
    if failed_lane is not None:
        assert not verified


class RecoveryMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.test = self.production = "absent"
        self.finalized = False

    @rule()
    def resume(self):
        action = next_transition(self.test, self.production, self.finalized)
        if action == "testpypi":
            self.test = "complete"
        elif action == "pypi":
            self.production = "complete"
        elif action == "finalize":
            self.finalized = True

    @rule()
    def interruption(self):
        before = (self.test, self.production, self.finalized)
        next_transition(*before)
        assert before == (self.test, self.production, self.finalized)

    @invariant()
    def no_premature_finalization(self):
        assert not self.finalized or self.test == self.production == "complete"


TestRecovery = RecoveryMachine.TestCase
