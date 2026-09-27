"""Version preparation properties and real-Git/fake-GitHub lifecycle regressions."""

import base64
import copy
import json
import subprocess

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scripts import release_preparation as prep
from scripts import review_evidence as evidence

VERSIONS = st.tuples(*(st.integers(min_value=0, max_value=1000) for _ in range(3)))


def blobs(v="0.6.10"):
    return {
        "pyproject.toml": f'[project]\nname = "OpenPinch"\ndependencies=["numpy"]\nversion = "{v}"\n',
        ".bumpversion.toml": f'[tool.bumpversion]\ncurrent_version = "{v}"\n',
        "uv.lock": f'[[package]]\nname = "openpinch"\nversion = "{v}"\nsource = {{ editable = "." }}\n',
    }


def record(v="0.6.10"):
    item = dict(
        schema=1,
        repository="owner/repo",
        pr=1,
        head="a" * 40,
        base="b" * 40,
        source="c" * 40,
        tree="d" * 40,
        baseline=v,
        target=prep.allocate(v, v),
        run=10,
    )
    item["identity"] = prep.identity(item)
    return item


@given(VERSIONS)
def test_version_allocation_codec_and_edit_properties(parts):
    original = ".".join(map(str, parts))
    target = ".".join(map(str, (*parts[:2], parts[2] + 1)))
    assert prep.allocate(original, original) == target
    assert prep.version(original) == parts
    updated = prep.bumped(blobs(original), target)
    assert prep.metadata_version(updated) == target
    assert prep.bumped(updated, original) == blobs(original)
    item = record(original)
    assert prep.decode(prep.canonical(item)) == item
    assert prep.bumped(updated, target) == updated


@given(VERSIONS, st.integers(min_value=1, max_value=100))
def test_explicit_minor_is_preserved(parts, delta):
    old = ".".join(map(str, parts))
    new = f"{parts[0]}.{parts[1] + delta}.0"
    assert prep.allocate(old, new) == new


@pytest.mark.parametrize(
    "value", ["01.2.3", "1.2", "1.2.3rc1", "v1.2.3", "1.2.3\n", True]
)
def test_noncanonical_versions_rejected(value):
    with pytest.raises(ValueError):
        prep.version(value)


def test_unknown_record_fields_and_duplicate_keys_rejected():
    item = record()
    with pytest.raises(ValueError):
        prep.validate({**item, "execute": "command"})
    with pytest.raises(ValueError):
        prep.decode(prep.canonical(item)[:-1] + ',"pr":1}')
    with pytest.raises(ValueError):
        prep.decode(" " * 65537)


@pytest.fixture
def local_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

    def git(*args, input=None):
        return subprocess.run(
            ["git", *args], input=input, text=True, check=True, capture_output=True
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    for path, content in blobs().items():
        (tmp_path / path).write_text(content)
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "feature.txt").write_text("reviewed feature\n")
    git("add", ".")
    git("commit", "-m", "feature")
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    source = git("commit-tree", tree, "-p", base, "-p", head, "-m", "review merge")
    git("update-ref", "refs/pull/1/merge", source)
    git("remote", "add", "origin", str(tmp_path))
    return tmp_path, git, base, head, tree, source


@pytest.fixture
def coordinator(local_git, monkeypatch):
    root, git, base, head, tree, source = local_git
    pr = dict(
        number=1,
        state="open",
        draft=False,
        head=dict(ref="develop", sha=head, repo=dict(full_name="owner/repo")),
        base=dict(ref="main", sha=base, repo=dict(full_name="owner/repo")),
    )
    state = {"branches": [], "prs": [], "writes": [], "interrupt": False}

    def api(endpoint, method="GET", payload=None):
        if endpoint == "graphql":
            return {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewDecision": "APPROVED",
                            "headRefOid": pr["head"]["sha"],
                            "baseRefOid": pr["base"]["sha"],
                        }
                    }
                }
            }
        if method == "GET":
            assert endpoint.endswith("/pulls/1")
            return copy.deepcopy(pr)
        state["writes"].append((endpoint, payload))
        if endpoint.endswith("/git/blobs"):
            return {
                "sha": git(
                    "hash-object",
                    "-w",
                    "--stdin",
                    input=base64.b64decode(payload["content"]).decode(),
                )
            }
        if endpoint.endswith("/git/trees"):
            git("read-tree", payload["base_tree"])
            for entry in payload["tree"]:
                git(
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"100644,{entry['sha']},{entry['path']}",
                )
            result = git("write-tree")
            # GitHub tree creation does not modify the caller's index/worktree.
            # Restore our temporary construction index to model that boundary.
            git("read-tree", "HEAD")
            return {"sha": result}
        if endpoint.endswith("/git/commits"):
            return {
                "sha": git(
                    "commit-tree",
                    payload["tree"],
                    "-p",
                    payload["parents"][0],
                    "-m",
                    payload["message"],
                )
            }
        if endpoint.endswith("/git/refs"):
            git("update-ref", payload["ref"], payload["sha"])
            state["branches"].append(
                {
                    "name": payload["ref"].removeprefix("refs/heads/"),
                    "commit": {"sha": payload["sha"]},
                }
            )
            return {}
        assert endpoint.endswith("/pulls")
        if state["interrupt"]:
            state["interrupt"] = False
            raise OSError("PR creation interrupted")
        result = dict(
            number=2,
            state="open",
            head=dict(ref=payload["head"], repo=dict(full_name="owner/repo")),
        )
        state["prs"].append(result)
        return result

    def pages(endpoint):
        if "/branches?" in endpoint:
            return [state["branches"]]
        if "/pulls?" in endpoint:
            return [state["prs"]]
        raise AssertionError(endpoint)

    monkeypatch.setattr(prep, "api", api)
    monkeypatch.setattr(prep, "pages", pages)
    monkeypatch.setattr(prep, "target_absent", lambda *_: None)
    monkeypatch.setattr(evidence, "find_review_run", lambda *_: 10)
    monkeypatch.setattr(evidence, "verify_review_run", lambda *_: (source, tree))
    return root, git, pr, state


def test_prepare_retry_is_idempotent_and_only_edits_version(coordinator):
    root, git, pr, state = coordinator
    first = prep.prepare("owner/repo", pr)
    writes = len(state["writes"])
    assert prep.prepare("owner/repo", pr) == first
    assert len(state["writes"]) == writes
    assert len(state["branches"]) == len(state["prs"]) == 1
    commit = state["branches"][0]["commit"]["sha"]
    result = prep.verify_prepared(commit, "owner/repo")
    assert result["target"] == "0.6.11"
    assert prep.metadata_version(prep.metadata(pr["head"]["sha"])) == "0.6.10"
    assert git("show", f"{commit}:feature.txt") == "reviewed feature"


def test_partial_write_recovery_and_closed_pr(coordinator):
    _, _, pr, state = coordinator
    state["interrupt"] = True
    with pytest.raises(OSError):
        prep.prepare("owner/repo", pr)
    assert len(state["branches"]) == 1 and not state["prs"]
    prep.prepare("owner/repo", pr)
    assert len(state["branches"]) == len(state["prs"]) == 1
    state["prs"][0]["state"] = "closed"
    writes = len(state["writes"])
    with pytest.raises(ValueError, match="closed"):
        prep.prepare("owner/repo", pr)
    assert len(state["writes"]) == writes


def test_main_gate_uses_pre_push_baseline_for_rebase_merge(coordinator, monkeypatch):
    root, git, pr, state = coordinator
    prep.prepare("owner/repo", pr)
    bump = state["branches"][0]["commit"]["sha"]
    git("checkout", "--detach", bump)
    (root / "feature.txt").write_text("later reviewed change\n")
    git("add", ".")
    git("commit", "-m", "later reviewed change")
    event = root / "event.json"
    event.write_text(json.dumps({"before": pr["base"]["sha"]}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setattr("sys.argv", ["prepare", "gate"])
    original = prep.command

    def command(*args):
        if (
            args[:2] == ("git", "ls-tree")
            and args[-1] == ".github/workflows/ci-prepare-release.yml"
        ):
            return "100644 blob exists"
        return original(*args)

    monkeypatch.setattr(prep, "command", command)
    assert prep.main() == 0
    event.write_text(json.dumps({"before": bump}))
    assert prep.main() == 1


@given(st.lists(st.sampled_from(["retry", "interrupt", "close"]), max_size=8))
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_stateful_preparation_model(coordinator, sequence):
    # Reset fake remote observations per example. Git objects are immutable and
    # deterministic; no checkout/source changes occur in these commands.
    _, _, pr, state = coordinator
    state.update(branches=[], prs=[], writes=[], interrupt=False)
    allocated = False
    has_pr = False
    closed = False
    for action in sequence:
        if action == "close" and has_pr:
            state["prs"][0]["state"] = "closed"
            closed = True
        before = len(state["writes"])
        if closed:
            with pytest.raises(ValueError):
                prep.prepare("owner/repo", pr)
            assert len(state["writes"]) == before
        elif action == "interrupt" and not has_pr:
            state["interrupt"] = True
            with pytest.raises(OSError):
                prep.prepare("owner/repo", pr)
            allocated = True
        else:
            prep.prepare("owner/repo", pr)
            allocated = has_pr = True
        assert len(state["branches"]) == int(allocated)
        assert len(state["prs"]) == int(has_pr)


@pytest.mark.parametrize(
    "path,content",
    [
        ("feature.txt", "unreviewed"),
        ("uv.lock", '\n[[package]]\nname="evil"\nversion="1"\n'),
        ("pyproject.toml", '\n[build-system]\nrequires=["evil"]\n'),
    ],
)
def test_substantive_changes_cannot_reuse_proof(coordinator, path, content):
    root, git, pr, state = coordinator
    prep.prepare("owner/repo", pr)
    commit = state["branches"][0]["commit"]["sha"]
    git("checkout", "--detach", commit)
    with (root / path).open("a") as handle:
        handle.write(content)
    git("add", ".")
    git("commit", "-m", "unreviewed change")
    with pytest.raises(ValueError):
        prep.verify_prepared(git("rev-parse", "HEAD"), "owner/repo")
