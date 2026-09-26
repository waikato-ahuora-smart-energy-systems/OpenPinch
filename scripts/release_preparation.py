"""Prepare reviewed version PRs using inert Git blobs and bounded GitHub APIs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.request import urlopen

from scripts.check_package_index_release import _read_index_response
from scripts.release_manifest import command, no_duplicate_keys

RECORD = ".github/release-preparation.json"
FILES = ("pyproject.toml", ".bumpversion.toml", "uv.lock")
SHA = re.compile(r"[0-9a-f]{40}\Z")
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")


def version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise ValueError("Noncanonical version")
    return tuple(map(int, value.split(".")))


def allocate(baseline: str, reviewed: str) -> str:
    base, current = version(baseline), version(reviewed)
    if base == current:
        return f"{base[0]}.{base[1]}.{base[2] + 1}"
    if current > base and current[:2] > base[:2]:
        return reviewed
    raise ValueError("Unexplained version change; inspect pending preparation")


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def identity(record: dict) -> str:
    return hashlib.sha256(
        canonical(
            {k: v for k, v in record.items() if k not in {"identity", "run"}}
        ).encode()
    ).hexdigest()


def validate(record: dict) -> dict:
    fields = {
        "schema",
        "repository",
        "pr",
        "head",
        "base",
        "tree",
        "source",
        "baseline",
        "target",
        "run",
        "identity",
    }
    if (
        not isinstance(record, dict)
        or set(record) != fields
        or type(record["schema"]) is not int
        or record["schema"] != 1
    ):
        raise ValueError("Invalid preparation schema")
    if not isinstance(record["repository"], str) or not re.fullmatch(
        r"[\w.-]+/[\w.-]+", record["repository"], re.ASCII
    ):
        raise ValueError("Invalid repository")
    for key in ("head", "base", "tree", "source"):
        if not isinstance(record[key], str) or not SHA.fullmatch(record[key]):
            raise ValueError("Invalid source identity")
    for key in ("pr", "run"):
        if type(record[key]) is not int or record[key] < 1:
            raise ValueError("Invalid evidence identity")
    if version(record["target"]) <= version(record["baseline"]):
        raise ValueError("Version must increase")
    if record["identity"] != identity(record):
        raise ValueError("Preparation identity mismatch")
    return record


def decode(text: str) -> dict:
    if len(text.encode()) > 65536:
        raise ValueError("Preparation record too large")
    return validate(json.loads(text, object_pairs_hook=no_duplicate_keys))


def git_blob(ref: str, path: str) -> str:
    # Only validated object identities and constant paths reach Git revision syntax.
    if not SHA.fullmatch(ref) or path not in (*FILES, RECORD):
        raise ValueError("Invalid blob request")
    mode = command("git", "ls-tree", ref, "--", path).split()
    if not mode or mode[0] != "100644":
        raise ValueError("Missing or non-regular metadata file")
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    ).stdout
    if len(result) > 10_000_000:
        raise ValueError("Blob exceeds bound")
    return result


def metadata(ref: str) -> dict[str, str]:
    return {path: git_blob(ref, path) for path in FILES}


def metadata_version(blobs: dict[str, str]) -> str:
    project = tomllib.loads(blobs[FILES[0]])["project"]["version"]
    bump = tomllib.loads(blobs[FILES[1]])["tool"]["bumpversion"]["current_version"]
    packages = [
        p for p in tomllib.loads(blobs[FILES[2]])["package"] if p["name"] == "openpinch"
    ]
    if (
        len(packages) != 1
        or packages[0].get("source") != {"editable": "."}
        or packages[0]["version"] != project
        or bump != project
    ):
        raise ValueError("Version metadata is inconsistent")
    version(project)
    return project


def bumped(blobs: dict[str, str], target: str) -> dict[str, str]:
    old = metadata_version(blobs)
    version(target)
    result = {}
    patterns = {
        FILES[
            0
        ]: rf'(?ms)(^\[project\]\n(?:(?!^\[).)*?^version\s*=\s*"){re.escape(old)}(")',
        FILES[1]: rf'(\bcurrent_version\s*=\s*"){re.escape(old)}(")',
        FILES[2]: rf'(name = "openpinch"\nversion = "){re.escape(old)}(")',
    }
    for path in FILES:
        result[path], count = re.subn(
            patterns[path], lambda m: m[1] + target + m[2], blobs[path], count=0
        )
        if count != 1:
            raise ValueError("Ambiguous version edit")
    if metadata_version(result) != target:
        raise ValueError("Version edit failed")
    return result


def record_at(ref: str) -> dict | None:
    if not command("git", "ls-tree", ref, "--", RECORD):
        return None
    return decode(git_blob(ref, RECORD))


def verify_prepared(ref: str, repository: str, *, exact: bool = True) -> dict:
    record = record_at(ref)
    if not record or record["repository"] != repository:
        raise ValueError("Missing preparation for this repository")
    command("git", "fetch", "--no-tags", "origin", record["source"])
    if command("git", "rev-parse", f"{record['source']}^{{tree}}") != record[
        "tree"
    ] or command("git", "rev-list", "--parents", "-n", "1", record["source"]).split()[
        1:
    ] != [record["base"], record["head"]]:
        raise ValueError("Invalid recorded merge source")
    if metadata_version(metadata(record["base"])) != record["baseline"]:
        raise ValueError("Baseline mismatch")
    original = metadata(record["head"])
    if allocate(record["baseline"], metadata_version(original)) != record["target"]:
        raise ValueError("Allocation mismatch")
    if metadata_version(metadata(ref)) != record["target"]:
        raise ValueError("Prepared version mismatch")
    command("git", "merge-base", "--is-ancestor", record["base"], ref)
    if not exact:
        # Allocation survives subsequently reviewed source changes. Those changes
        # cannot reuse tests: reusable() always requests exact equivalence.
        return record
    # Squash merges need not preserve the develop head as an ancestor. The
    # reviewed tree + exact transformation is the authoritative content proof.
    if metadata(ref) != bumped(metadata(record["tree"]), record["target"]):
        raise ValueError("Not an exact version-only transformation")
    changes = command(
        "git", "diff", "--name-only", record["tree"], ref, "--"
    ).splitlines()
    if set(changes) - {*FILES, RECORD}:
        raise ValueError("Candidate differs from reviewed source")
    raw = command("git", "diff", "--raw", record["tree"], ref, "--", *FILES)
    if any(line.split()[0:2] != [":100644", "100644"] for line in raw.splitlines()):
        raise ValueError("Metadata mode changed")
    return record


def api(endpoint: str, method: str = "GET", payload: dict | None = None):
    args = ["gh", "api", "--method", method, endpoint]
    if payload is not None:
        args += ["--input", "-"]
    result = subprocess.run(
        args,
        input=None if payload is None else json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    if len(result.stdout) > 10_000_000:
        raise ValueError("API response exceeds bound")
    return json.loads(result.stdout) if result.stdout.strip() else None


def pages(endpoint: str) -> list:
    result = json.loads(command("gh", "api", "--paginate", "--slurp", endpoint))
    if not isinstance(result, list) or len(result) > 100:
        raise ValueError("Pagination exceeds bound")
    return result


def repository() -> str:
    value = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value, re.ASCII):
        raise ValueError("Invalid repository")
    return value


def reviewed(repo: str, pr: dict) -> bool:
    if (
        pr.get("draft")
        or pr.get("state") != "open"
        or pr["head"]["ref"] != "develop"
        or pr["base"]["ref"] != "main"
        or any(pr[k]["repo"]["full_name"] != repo for k in ("head", "base"))
    ):
        return False
    owner, name = repo.split("/")
    response = api(
        "graphql",
        "POST",
        {
            "query": (
                "query($owner:String!,$name:String!,$number:Int!){"
                "repository(owner:$owner,name:$name){pullRequest(number:$number){"
                "reviewDecision headRefOid baseRefOid}}}"
            ),
            "variables": {"owner": owner, "name": name, "number": pr["number"]},
        },
    )
    state = response["data"]["repository"]["pullRequest"]
    return state == {
        "reviewDecision": "APPROVED",
        "headRefOid": pr["head"]["sha"],
        "baseRefOid": pr["base"]["sha"],
    }


def target_absent(repo: str, target: str) -> None:
    tags = [t for page in pages(f"repos/{repo}/tags?per_page=100") for t in page]
    releases = [
        r for page in pages(f"repos/{repo}/releases?per_page=100") for r in page
    ]
    if any(t["name"] == f"v{target}" for t in tags) or any(
        r["tag_name"] == f"v{target}" for r in releases
    ):
        raise ValueError("Target already reserved; resume original release")
    for index in ("https://test.pypi.org/pypi", "https://pypi.org/pypi"):
        if (
            _read_index_response(
                f"{index}/OpenPinch/{target}/json", opener=urlopen, timeout=20
            )
            is not None
        ):
            raise ValueError("Target exists on package index")


def prepare(repo: str, pr: dict) -> str:
    from scripts.review_evidence import find_review_run, verify_review_run

    if not reviewed(repo, pr):
        return "waiting for current review approval"
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    for ref in (head, base):
        if not SHA.fullmatch(ref):
            raise ValueError("Invalid PR source")
    command("git", "fetch", "--no-tags", "origin", head, base)
    command("git", "fetch", "--no-tags", "origin", f"refs/pull/{pr['number']}/merge")
    merge = command("git", "rev-parse", "FETCH_HEAD")
    old = record_at(head)
    if (
        old
        and old["target"] == metadata_version(metadata(head))
        and old["target"] != metadata_version(metadata(base))
    ):
        verify_prepared(merge, repo, exact=False)
        return "already prepared"
    run = find_review_run(repo, pr["number"], head)
    source, tree = verify_review_run(repo, run, pr["number"], head, base)
    baseline = metadata_version(metadata(base))
    target = allocate(baseline, metadata_version(metadata(head)))
    record = dict(
        schema=1,
        repository=repo,
        pr=pr["number"],
        head=head,
        base=base,
        source=source,
        tree=tree,
        baseline=baseline,
        target=target,
        run=run,
    )
    record["identity"] = identity(record)
    validate(record)
    branch = f"codex/release-{record['identity']}"
    branches = [
        b for page in pages(f"repos/{repo}/branches?per_page=100") for b in page
    ]
    existing = next((b for b in branches if b["name"] == branch), None)
    all_prs = [
        p
        for page in pages(f"repos/{repo}/pulls?state=all&base=develop&per_page=100")
        for p in page
    ]
    matching = [
        p
        for p in all_prs
        if p["head"]["ref"] == branch
        and p["head"]["repo"]
        and p["head"]["repo"]["full_name"] == repo
    ]
    if len(matching) > 1:
        raise ValueError("Duplicate preparation PRs")
    if matching and matching[0]["state"] != "open":
        raise ValueError("Preparation PR closed; maintainer action required")
    for p in all_prs:
        if (
            p["state"] == "open"
            and p["head"]["ref"].startswith("codex/release-")
            and p["head"]["ref"] != branch
        ):
            raise ValueError("Another preparation is pending; resolve stale PR first")
    target_absent(repo, target)
    fresh = api(f"repos/{repo}/pulls/{pr['number']}")
    if (
        fresh["head"]["sha"] != head
        or fresh["base"]["sha"] != base
        or not reviewed(repo, fresh)
    ):
        raise ValueError("Review or branch changed during preparation")
    edits = bumped(metadata(head), target)
    edits[RECORD] = canonical(record) + "\n"
    if existing:
        commit = existing["commit"]["sha"]
        command("git", "fetch", "--no-tags", "origin", commit)
        saved = record_at(commit)
        if saved and {k: v for k, v in saved.items() if k != "run"} == {
            k: v for k, v in record.items() if k != "run"
        }:
            # A new successful run must not rewrite an existing version commit.
            # If old evidence is no longer reusable, normal CI executes again.
            record = saved
            edits[RECORD] = canonical(record) + "\n"
        if command("git", "rev-parse", f"{commit}^") != head or any(
            git_blob(commit, p) != s for p, s in edits.items()
        ):
            raise ValueError("Existing preparation branch conflicts")
        changed = set(command("git", "diff", "--name-only", head, commit).splitlines())
        if changed - set(edits):
            raise ValueError("Existing branch has unrelated changes")
    else:
        entries = []
        for path, content in edits.items():
            blob = api(
                f"repos/{repo}/git/blobs",
                "POST",
                {
                    "content": base64.b64encode(content.encode()).decode(),
                    "encoding": "base64",
                },
            )
            entries.append(
                {"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]}
            )
        tree_result = api(
            f"repos/{repo}/git/trees",
            "POST",
            {
                "base_tree": command("git", "rev-parse", f"{head}^{{tree}}"),
                "tree": entries,
            },
        )
        commit = api(
            f"repos/{repo}/git/commits",
            "POST",
            {
                "message": f"Prepare OpenPinch {target}",
                "tree": tree_result["sha"],
                "parents": [head],
            },
        )["sha"]
        # Create-only ref: no update/force operation exists in this adapter.
        api(
            f"repos/{repo}/git/refs",
            "POST",
            {"ref": f"refs/heads/{branch}", "sha": commit},
        )
    if not matching:
        fresh = api(f"repos/{repo}/pulls/{pr['number']}")
        if (
            fresh["head"]["sha"] != head
            or fresh["base"]["sha"] != base
            or not reviewed(repo, fresh)
        ):
            raise ValueError("Source changed; retained branch needs reconciliation")
        api(
            f"repos/{repo}/pulls",
            "POST",
            {
                "title": f"Prepare OpenPinch {target}",
                "head": branch,
                "base": "develop",
                "body": (
                    f"Version preparation for #{pr['number']}. "
                    f"Identity: {record['identity']}. Merge this PR, then the "
                    "reviewed develop-to-main PR. Normal protections apply."
                ),
            },
        )
    return f"prepared {target} via {branch}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reconcile", "gate"))
    args = parser.parse_args()
    repo = repository()
    try:
        if args.mode == "gate":
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            pr = event.get("pull_request", {})
            if pr.get("base", {}).get("ref") == "develop" and pr.get("head", {}).get(
                "ref", ""
            ).startswith("codex/release-"):
                ref = command("git", "rev-parse", "HEAD")
                record = record_at(ref)
                if (
                    not record
                    or record["repository"] != repo
                    or pr["base"]["sha"] != record["head"]
                ):
                    raise ValueError("Stale or invalid bump PR")
                if metadata(ref) != bumped(metadata(record["head"]), record["target"]):
                    raise ValueError("Bump PR changes more than version fields")
                changes = command(
                    "git", "diff", "--name-only", record["head"], ref
                ).splitlines()
                if set(changes) - {*FILES, RECORD}:
                    raise ValueError("Bump PR has unrelated changes")
                return 0
            if not pr and os.environ.get("GITHUB_REF") == "refs/heads/main":
                ref = command("git", "rev-parse", "HEAD")
                # Push.before is main before the entire merge, not necessarily
                # HEAD^ (rebase merges can contain several versioned commits).
                parent = event.get("before")
                if parent is None:
                    saved = record_at(ref)
                    parent = (
                        saved["base"] if saved else command("git", "rev-parse", "HEAD^")
                    )
                if not isinstance(parent, str) or not SHA.fullmatch(parent):
                    raise ValueError("Invalid previous main identity")
                if command(
                    "git",
                    "ls-tree",
                    parent,
                    "--",
                    ".github/workflows/ci-prepare-release.yml",
                ):
                    record = verify_prepared(ref, repo, exact=False)
                    if metadata_version(metadata(parent)) != record["baseline"]:
                        raise ValueError(
                            "Main source does not advance its baseline version"
                        )
                return 0
            if pr.get("base", {}).get("ref") != "main":
                return 0
            base = pr["base"]["sha"]
            if not SHA.fullmatch(base):
                raise ValueError("Invalid main base")
            # One-time rollout: only a base predating this coordinator may omit
            # preparation. No input or label can enable this on an active base.
            if not command(
                "git", "ls-tree", base, "--", ".github/workflows/ci-prepare-release.yml"
            ):
                print("Coordinator bootstrap: existing main predates preparation")
                return 0
            ref = command("git", "rev-parse", "HEAD")
            record = verify_prepared(ref, repo, exact=False)
            if metadata_version(metadata(base)) != record["baseline"]:
                raise ValueError("A new main release requires new version preparation")
            print("Version preparation verified")
        else:
            prs = [
                p
                for page in pages(
                    f"repos/{repo}/pulls?state=open&base=main&per_page=100"
                )
                for p in page
                if p["head"]["ref"] == "develop"
            ]
            if len(prs) > 1:
                raise ValueError("Multiple develop-to-main candidates")
            for pr in prs:
                print(prepare(repo, pr))
        return 0
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Preparation blocked: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
