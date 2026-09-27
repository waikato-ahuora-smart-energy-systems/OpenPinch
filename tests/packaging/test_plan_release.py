"""Automatic publication accepts only trusted main evidence and complete states."""

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts import plan_release as planner
from scripts.ci_policy import POLICY, required_jobs

REPOSITORY = "example/OpenPinch"


@pytest.fixture
def evidence(monkeypatch):
    """Fake external boundaries; real planner, manifest and proof logic execute."""
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    manifests = {}
    bundles = {}
    artifacts = {}
    for run_id in (90, 100):
        files = {
            "openpinch-1.2.3-py3-none-any.whl": f"wheel-{run_id}".encode(),
            "openpinch-1.2.3.tar.gz": f"sdist-{run_id}".encode(),
        }
        manifest = {
            "schema": 1,
            "policy": POLICY,
            "repository": REPOSITORY,
            "source_sha": ("a" if run_id == 100 else "c") * 40,
            "source_tree": "b" * 40,
            "run_id": run_id,
            "run_attempt": 1,
            "version": "1.2.3",
            "files": {
                name: hashlib.sha256(data).hexdigest() for name, data in files.items()
            },
        }
        bundles[run_id] = {
            **files,
            planner.release.MANIFEST: json.dumps(manifest).encode(),
        }
        manifests[run_id] = manifest
        artifacts[run_id] = {
            "id": run_id + 1,
            "name": f"openpinch-dist-{run_id}-1",
            "digest": "sha256:" + "d" * 64,
            "expired": False,
            "workflow_run": {"id": run_id, "head_sha": manifest["source_sha"]},
        }
    run = {
        "id": 100,
        "run_attempt": 1,
        "head_sha": "a" * 40,
        "event": "push",
        "head_branch": "main",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/ci-main.yml",
        "head_repository": {"full_name": REPOSITORY},
        "repository": {"full_name": REPOSITORY},
    }
    state = {
        "run": run,
        "event": {"workflow_run": deepcopy(run)},
        "releases": [],
        "tag": False,
        "index": ["absent", "absent"],
        "artifacts": artifacts,
        "bundles": bundles,
        "manifests": manifests,
        "failed_lane": None,
        "downloads": [],
        "extra_artifact": False,
    }

    def api(endpoint):
        if endpoint.endswith("/runs/100"):
            return state["run"]
        if "/attempts/" in endpoint:
            run_id = int(endpoint.split("/runs/")[1].split("/")[0])
            return {
                **run,
                "id": run_id,
                "head_sha": manifests[run_id]["source_sha"],
                "run_attempt": 1,
            }
        if "/artifacts/" in endpoint:
            return state["artifacts"][int(endpoint.rsplit("/", 1)[1]) - 1]
        pytest.fail(endpoint)

    def command(*args):
        if args[:2] == ("gh", "api"):
            endpoint = args[-1]
            if "/releases?" in endpoint:
                return json.dumps([state["releases"]])
            if "/artifacts?" in endpoint:
                run_id = int(endpoint.split("/runs/")[1].split("/")[0])
                items = [state["artifacts"][run_id]]
                if state["extra_artifact"]:
                    items.append(
                        {**items[0], "id": 999, "name": f"openpinch-dist-{run_id}-2"}
                    )
                return json.dumps([{"artifacts": items}])
            if "/jobs?" in endpoint:
                return json.dumps(
                    [
                        {
                            "jobs": [
                                {
                                    "name": "validation / " + name,
                                    "conclusion": "failure"
                                    if name == state["failed_lane"]
                                    else "success",
                                }
                                for name in required_jobs("full") | {POLICY}
                            ]
                        }
                    ]
                )
        if args[:3] == ("gh", "release", "download"):
            name = args[5]
            (Path(args[7]) / name).write_bytes(
                state.get("released_bytes", bundles[90])[name]
            )
            return ""
        if args[:3] == ("git", "tag", "--list"):
            return "v1.2.3" if state["tag"] else ""
        if args[:2] == ("git", "merge-base"):
            return ""
        if args[:3] == ("git", "cat-file", "-t"):
            return "tag"
        if args[:2] == ("git", "rev-parse"):
            return "c" * 40 if args[-1].endswith("^{commit}") else "b" * 40
        pytest.fail(f"Unexpected mutation or command: {args}")

    def download(directory, artifact_id, digest):
        run_id = artifact_id - 1
        assert digest == state["artifacts"][run_id]["digest"]
        state["downloads"].append(run_id)
        directory.mkdir()
        for name, data in bundles[run_id].items():
            (directory / name).write_bytes(data)

    def inspect(**kwargs):
        # This is deliberately version-specific, not candidate-byte-specific:
        # the completed path must use the original bundle's expected hashes.
        state.setdefault("index_hashes", []).append(kwargs["expected_files"])
        return state["index"][planner.INDEXES.index(kwargs["index_url"])]

    monkeypatch.setattr(planner.release, "api", api)
    monkeypatch.setattr(planner.release, "command", command)
    monkeypatch.setattr(planner.release, "download", download)
    monkeypatch.setattr(planner, "inspect_release", inspect)
    return state


def public_release(state):
    state["releases"] = [
        {
            "tag_name": "v1.2.3",
            "draft": False,
            "prerelease": False,
            "assets": [
                {"name": name, "size": len(data)}
                for name, data in state["bundles"][90].items()
            ],
        }
    ]
    state["tag"] = True
    state["index"] = ["complete", "complete"]


def test_new_version_uses_triggering_bundle_not_listener_head(evidence, tmp_path):
    result = planner.plan(evidence["event"], tmp_path)
    assert result["action"] == "publish"
    assert result["run_id"] == 100
    assert result["artifact_id"] == 101
    assert result["source_sha"] == "a" * 40
    assert evidence["downloads"] == [100]


def test_later_same_version_commit_verifies_original_before_noop(evidence, tmp_path):
    public_release(evidence)
    result = planner.plan(evidence["event"], tmp_path)
    assert result["action"] == "complete"
    assert evidence["downloads"] == [100, 90]
    assert evidence["index_hashes"] == [evidence["manifests"][90]["files"]] * 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("conclusion", "failure"),
        ("status", "in_progress"),
        ("path", ".github/workflows/ci-develop.yml"),
        ("head_sha", "e" * 40),
        ("repository", {"full_name": "other/repo"}),
        ("run_attempt", 2),
    ],
)
def test_api_identity_must_match_event(evidence, tmp_path, field, value):
    evidence["run"][field] = value
    with pytest.raises(ValueError, match="evidence"):
        planner.plan(evidence["event"], tmp_path)
    assert evidence["downloads"] == []


@pytest.mark.parametrize(
    "failure",
    [
        "tag",
        "partial",
        "draft",
        "prerelease",
        "expired",
        "ambiguous",
        "lane",
        "bytes",
        "legacy",
    ],
)
def test_unsafe_release_states_block(evidence, tmp_path, failure):
    if failure in {"draft", "prerelease", "bytes", "legacy"}:
        public_release(evidence)
    if failure == "tag":
        evidence["tag"] = True
    elif failure == "partial":
        evidence["index"] = ["complete", "absent"]
    elif failure in {"draft", "prerelease"}:
        evidence["releases"][0][failure] = True
    elif failure == "expired":
        evidence["artifacts"][100]["expired"] = True
    elif failure == "ambiguous":
        evidence["extra_artifact"] = True
    elif failure == "lane":
        evidence["failed_lane"] = "solver-tests"
    elif failure == "bytes":
        evidence["released_bytes"] = {
            **evidence["bundles"][90],
            "openpinch-1.2.3.tar.gz": b"conflict",
        }
    elif failure == "legacy":
        evidence["releases"][0]["assets"] = []
    with pytest.raises(ValueError):
        planner.plan(evidence["event"], tmp_path)


def test_failed_job_retry_retains_original_build(evidence, tmp_path):
    evidence["event"]["workflow_run"]["run_attempt"] = 2
    evidence["run"]["run_attempt"] = 2
    assert planner.plan(evidence["event"], tmp_path)["run_attempt"] == 1


def test_api_outage_is_not_absence(evidence, monkeypatch, tmp_path):
    def fail(_):
        raise OSError("unavailable")

    monkeypatch.setattr(planner.release, "api", fail)
    with pytest.raises(OSError):
        planner.plan(evidence["event"], tmp_path)


@given(st.booleans(), st.booleans(), st.booleans(), st.booleans())
def test_event_authorization_matches_independent_predicate(success, push, main, own):
    run = {
        "conclusion": "success" if success else "failure",
        "event": "push" if push else "pull_request",
        "head_branch": "main" if main else "develop",
        "head_repository": {"full_name": REPOSITORY if own else "fork/repo"},
    }
    assert planner.eligible_event(run, REPOSITORY) == all((success, push, main, own))


def test_duplicate_completed_checks_then_conflict_never_mutate(evidence, tmp_path):
    public_release(evidence)
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        assert planner.plan(evidence["event"], root)["action"] == "complete"
    evidence["releases"][0]["prerelease"] = True
    root = tmp_path / "conflict"
    root.mkdir()
    with pytest.raises(ValueError, match="recovery"):
        planner.plan(evidence["event"], root)


def test_generated_observation_sequences_match_completion_oracle(evidence):
    baseline = deepcopy(evidence)

    @given(
        st.lists(
            st.sampled_from(["complete", "draft", "partial", "expired"]), max_size=6
        )
    )
    # Real filesystem operations vary with host load; this is a state/provenance
    # property, not a latency benchmark. Sequence/example counts remain bounded,
    # shrinking stays enabled, and the CI job supplies the outer time limit.
    @settings(max_examples=30, deadline=None)
    def check(sequence):
        evidence.clear()
        evidence.update(deepcopy(baseline))
        public_release(evidence)
        for observation in sequence:
            evidence["releases"][0]["draft"] = observation == "draft"
            evidence["index"] = [
                "complete",
                "partial" if observation == "partial" else "complete",
            ]
            evidence["artifacts"][90]["expired"] = observation == "expired"
            # Fake command boundary rejects every mutation. Repeat each check
            # to prove idempotence after each generated external state change.
            for _ in range(2):
                with tempfile.TemporaryDirectory() as directory:
                    if observation == "complete":
                        assert (
                            planner.plan(evidence["event"], Path(directory))["action"]
                            == "complete"
                        )
                    else:
                        with pytest.raises(ValueError):
                            planner.plan(evidence["event"], Path(directory))

    check()


@pytest.mark.parametrize("mode", ["publish", "complete", "ignore", "block"])
def test_cli_outputs_and_blocked_exit(evidence, tmp_path, monkeypatch, capsys, mode):
    if mode == "complete":
        public_release(evidence)
    if mode == "ignore":
        evidence["event"]["workflow_run"]["conclusion"] = "failure"
    if mode == "block":
        evidence["tag"] = True
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps(evidence["event"]))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    assert planner.main() == int(mode == "block")
    if mode == "block":
        assert not (tmp_path / "output").exists()
        assert "verified recovery" in capsys.readouterr().err
        assert "blocked" in (tmp_path / "summary").read_text()
    else:
        assert f"action={mode}\n" in (tmp_path / "output").read_text()
        assert mode in (tmp_path / "summary").read_text()


@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped", None])
def test_unsuccessful_events_do_not_publish(status):
    assert planner.eligible_event({"conclusion": status}, "example/OpenPinch") is False


def test_completed_state_requires_stable_public_release():
    for draft, prerelease in [(True, False), (False, True), (False, None)]:
        with pytest.raises(ValueError, match="recovery"):
            planner.require_stable({"draft": draft, "prerelease": prerelease})
