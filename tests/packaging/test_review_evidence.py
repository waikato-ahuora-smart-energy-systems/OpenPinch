"""Adversarial review provenance and independent lane-coverage oracles."""

import copy

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts import release_manifest as release
from scripts import release_preparation as prep
from scripts import review_evidence as evidence
from scripts.ci_policy import required_jobs
from tests.packaging.test_release_preparation import record


@pytest.fixture
def proof(monkeypatch):
    item = record()
    run = dict(
        id=10,
        status="completed",
        conclusion="failure",
        event="pull_request",
        path=".github/workflows/ci-pull-request.yml",
        head_sha=item["head"],
        head_branch="develop",
        head_repository=dict(full_name=item["repository"]),
        run_attempt=1,
        pull_requests=[
            dict(number=1, head=dict(sha=item["head"]), base=dict(sha=item["base"]))
        ],
    )
    jobs = [
        dict(name=f"validation / {name}", conclusion="success", run_attempt=1)
        for name in sorted(required_jobs("full") | {"delivery-v1"})
    ]
    artifact = dict(
        name="openpinch-dist-10-1",
        id=100,
        expired=False,
        digest="sha256:" + "a" * 64,
        workflow_run={"id": 10, "head_sha": item["head"]},
    )
    manifest = dict(
        repository=item["repository"],
        run_id=10,
        run_attempt=1,
        source_sha=item["source"],
        source_tree=item["tree"],
    )
    state = dict(run=run, jobs=jobs, artifact=artifact, manifest=manifest)

    def pages(endpoint):
        if "/workflows/" in endpoint:
            return [{"workflow_runs": [state["run"]]}]
        if "/jobs?" in endpoint:
            return [{"jobs": state["jobs"]}]
        if "/artifacts?" in endpoint:
            return [{"artifacts": [state["artifact"]]}]
        raise AssertionError(endpoint)

    def command(*args):
        if args[:2] == ("git", "fetch"):
            return ""
        if args[:2] == ("git", "rev-list"):
            return f"{item['source']} {item['base']} {item['head']}"
        if args[:2] == ("git", "rev-parse"):
            return item["tree"]
        raise AssertionError(args)

    monkeypatch.setenv("GITHUB_REPOSITORY", item["repository"])
    monkeypatch.setattr(prep, "api", lambda *_: state["run"])
    monkeypatch.setattr(prep, "pages", pages)
    monkeypatch.setattr(prep, "command", command)
    # Archive integrity is separately exercised by real release manifest tests.
    monkeypatch.setattr(release, "download", lambda *_: None)
    monkeypatch.setattr(release, "load", lambda *_: state["manifest"])
    return item, state


def verify(item):
    return evidence.verify_review_run(
        item["repository"], item["run"], item["pr"], item["head"], item["base"]
    )


def test_review_lanes_can_pass_before_preparation_merge_gate(proof):
    item, _ = proof
    assert verify(item) == (item["source"], item["tree"])


def metadata_jobs():
    return [
        dict(name=name, conclusion="success", run_attempt=1)
        for name in ("plan", "review-metadata-only")
    ] + [dict(name="validation", conclusion="skipped", run_attempt=1)]


def review_history(monkeypatch, state, newer, jobs):
    original = prep.pages

    def pages(endpoint):
        if "/workflows/" in endpoint:
            return [{"workflow_runs": [state["run"], newer]}]
        if "/runs/11/jobs?" in endpoint:
            return [{"jobs": jobs}]
        return original(endpoint)

    monkeypatch.setattr(prep, "pages", pages)


def test_metadata_edit_preserves_original_verified_review(proof, monkeypatch):
    item, state = proof
    newer = dict(state["run"], id=11)
    review_history(monkeypatch, state, newer, metadata_jobs())
    assert verify(item) == (item["source"], item["tree"])


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "attempt", "plan", "validation", "running", "cancelled"],
)
def test_ambiguous_or_actual_newer_runs_block_old_proof(proof, monkeypatch, mutation):
    item, state = proof
    newer = dict(state["run"], id=11)
    jobs = metadata_jobs()
    if mutation == "missing":
        jobs.pop(1)
    elif mutation == "duplicate":
        jobs.append(dict(jobs[1]))
    elif mutation == "attempt":
        jobs[1]["run_attempt"] = 2
    elif mutation == "plan":
        jobs[0]["conclusion"] = "failure"
    elif mutation == "validation":
        jobs.append(dict(name="validation / test", conclusion="failure"))
    elif mutation == "running":
        newer["status"] = "in_progress"
    else:
        newer["conclusion"] = "cancelled"
    review_history(monkeypatch, state, newer, jobs)
    with pytest.raises(ValueError, match="Newer review validation"):
        verify(item)


@given(st.lists(st.booleans(), min_size=1, max_size=20))
@settings(max_examples=40)
def test_generated_review_history_selects_latest_nonmetadata(flags):
    runs = [
        dict(
            id=i + 1,
            head_sha="a" * 40,
            head_branch="develop",
            status="completed",
            conclusion="failure",
            run_attempt=1,
        )
        for i in range(len(flags))
    ]
    with pytest.MonkeyPatch.context() as patch:

        def pages(endpoint):
            if "/workflows/" in endpoint:
                return [{"workflow_runs": runs[::2]}, {"workflow_runs": runs[1::2]}]
            run_id = int(endpoint.split("/runs/")[1].split("/")[0])
            return [{"jobs": metadata_jobs() if flags[run_id - 1] else []}]

        patch.setattr(prep, "pages", pages)
        candidates = [i + 1 for i, metadata in enumerate(flags) if not metadata]
        if candidates:
            assert evidence.find_review_run("owner/repo", 1, "a" * 40) == candidates[-1]
        else:
            with pytest.raises(ValueError, match="No matching"):
                evidence.find_review_run("owner/repo", 1, "a" * 40)


def test_empty_github_pr_association_still_requires_merge_proof(proof):
    item, state = proof
    state["run"]["pull_requests"] = []
    assert verify(item) == (item["source"], item["tree"])
    state["manifest"]["source_tree"] = "f" * 40
    with pytest.raises(ValueError, match="merge tree"):
        verify(item)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "in_progress"),
        ("event", "push"),
        ("path", "evil.yml"),
        ("head_sha", "e" * 40),
        ("head_repository", {"full_name": "fork/repo"}),
        ("head_branch", "feature"),
    ],
)
def test_untrusted_review_identity_rejected(proof, field, value):
    item, state = proof
    state["run"][field] = value
    with pytest.raises(ValueError):
        verify(item)


@given(
    st.dictionaries(
        st.sampled_from(sorted(required_jobs("full"))),
        st.sampled_from(["success", "failure", "cancelled", "skipped"]),
    )
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_generated_lane_coverage_matches_independent_oracle(proof, results):
    item, state = proof
    state["jobs"] = [
        dict(name=f"validation / {k}", conclusion=v, run_attempt=1)
        for k, v in results.items()
    ]
    state["jobs"].append(
        dict(name="validation / delivery-v1", conclusion="success", run_attempt=1)
    )
    expected = set(results) == set(required_jobs("full")) and set(results.values()) == {
        "success"
    }
    if expected:
        assert verify(item) == (item["source"], item["tree"])
    else:
        with pytest.raises(ValueError):
            verify(item)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "attempt", "expired", "manifest", "parents", "latest"]
)
def test_review_evidence_conflicts_fail_closed(proof, monkeypatch, mutation):
    item, state = proof
    if mutation == "duplicate":
        state["jobs"].append(copy.deepcopy(state["jobs"][0]))
    elif mutation == "attempt":
        state["jobs"][0]["run_attempt"] = 2
    elif mutation == "expired":
        state["artifact"]["expired"] = True
    elif mutation == "manifest":
        state["manifest"]["run_id"] = 11
    elif mutation == "parents":
        original = prep.command
        monkeypatch.setattr(
            prep,
            "command",
            lambda *a: "evil parents" if a[1] == "rev-list" else original(*a),
        )
    else:
        monkeypatch.setattr(evidence, "find_review_run", lambda *_: 11)
    with pytest.raises(ValueError):
        verify(item)


def test_reuse_cli_falls_back_without_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evidence, "reusable", lambda *_: (_ for _ in ()).throw(ValueError("stale"))
    )
    monkeypatch.setattr(prep, "command", lambda *_: "a" * 40)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    assert evidence.main() == 0
    assert (tmp_path / "output").read_text() == "reuse-review=false\n"


def test_v2_manifest_binds_preparation_to_release():
    item = record()
    manifest = dict(
        schema=2,
        policy="delivery-v2",
        repository=item["repository"],
        source_sha="e" * 40,
        source_tree="f" * 40,
        run_id=11,
        run_attempt=1,
        version="0.6.11",
        preparation=item,
        files={
            "openpinch-0.6.11-py3-none-any.whl": "a" * 64,
            "openpinch-0.6.11.tar.gz": "b" * 64,
        },
    )
    assert release.validate(manifest) == manifest
    manifest["preparation"] = {**item, "target": "0.6.12"}
    manifest["preparation"]["identity"] = prep.identity(manifest["preparation"])
    with pytest.raises(ValueError):
        release.validate(manifest)


@pytest.mark.parametrize("bad_lane", [None, "test", "docs", "solver-tests"])
def test_publisher_independently_verifies_reused_lanes(monkeypatch, bad_lane):
    item = record()
    manifest = dict(
        schema=2,
        policy="delivery-v2",
        repository=item["repository"],
        source_sha="e" * 40,
        source_tree="f" * 40,
        run_id=11,
        run_attempt=1,
        version="0.6.11",
        preparation=item,
        files={
            "openpinch-0.6.11-py3-none-any.whl": "a" * 64,
            "openpinch-0.6.11.tar.gz": "b" * 64,
        },
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", item["repository"])

    def api(endpoint):
        if "/attempts/" in endpoint:
            return dict(
                id=11,
                head_sha="e" * 40,
                run_attempt=1,
                head_branch="main",
                event="push",
                path=".github/workflows/ci-main.yml",
            )
        return dict(
            id=12,
            digest="sha256:" + "a" * 64,
            name="openpinch-dist-11-1",
            expired=False,
            workflow_run=dict(id=11, head_sha="e" * 40),
        )

    checked = []

    def reusable(ref, repo):
        checked.append((ref, repo))
        return item

    def command(*args):
        if args[:2] == ("gh", "api"):
            import json

            jobs = [
                dict(
                    name="validation / " + name,
                    conclusion="failure"
                    if name == bad_lane
                    else "skipped"
                    if name in evidence.REUSABLE
                    else "success",
                )
                for name in required_jobs("full") | {"delivery-v2"}
            ]
            return json.dumps([dict(jobs=jobs)])
        if args[:2] == ("git", "merge-base"):
            return ""
        assert args[:2] == ("git", "rev-parse")
        return "f" * 40

    monkeypatch.setattr(release, "api", api)
    monkeypatch.setattr(release, "command", command)
    monkeypatch.setattr(evidence, "reusable", reusable)
    if bad_lane:
        with pytest.raises(ValueError):
            release.verify_source(manifest, 12, "sha256:" + "a" * 64)
    else:
        release.verify_source(manifest, 12, "sha256:" + "a" * 64)
    assert checked == [("e" * 40, item["repository"])]
