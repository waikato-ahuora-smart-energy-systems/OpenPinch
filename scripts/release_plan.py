"""Decide whether a push to main should publish its project version.

Reads the tags and published GitHub releases gathered by ``release.yml`` and
prints ``publish=true|false`` for ``$GITHUB_OUTPUT``. It fails (exit 1) when
publishing would be unsafe:

- the version is not higher than every existing ``vX.Y.Z`` tag, or
- ``v<version>`` already exists on a different commit.

A version counts as released only when its GitHub release is published and
has both distribution files attached; anything less is resumed, not skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


class UnsafeRelease(ValueError):
    """Publishing this version would be wrong; a maintainer must intervene."""


def parse(tag: str) -> tuple[int, int, int] | None:
    match = TAG.fullmatch(tag)
    return tuple(int(part) for part in match.groups()) if match else None


def expected_files(version: str) -> set[str]:
    return {f"openpinch-{version}-py3-none-any.whl", f"openpinch-{version}.tar.gz"}


def decide(
    *,
    version: str,
    tags: list[str],
    releases: list[dict],
    tag_sha: str,
    head_sha: str,
) -> tuple[bool, str]:
    """Return ``(publish, reason)`` or raise :class:`UnsafeRelease`."""
    tag = f"v{version}"
    current = parse(tag)
    if current is None:
        raise UnsafeRelease(f"{version!r} is not an X.Y.Z version.")

    published = [r for r in releases if r["tag"] == tag]
    if published and expected_files(version) <= set(published[0]["assets"]):
        return False, f"{tag} is already released; nothing to publish."

    newer = sorted((t for t in tags if (p := parse(t)) and p > current), key=parse)
    if newer:
        raise UnsafeRelease(
            f"{tag} is lower than the existing tag {newer[-1]}. "
            "Bump the version above it before releasing."
        )
    if tag_sha and tag_sha != head_sha:
        raise UnsafeRelease(
            f"{tag} already points to {tag_sha[:12]}, not this commit "
            f"{head_sha[:12]}. Bump the version to release this commit."
        )
    if published or tag_sha:
        return True, f"Resuming the incomplete release of {tag}."
    return True, f"Releasing {tag}."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--tags-file", type=Path, required=True)
    parser.add_argument("--releases-file", type=Path, required=True)
    parser.add_argument("--tag-sha", default="")
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args(argv)

    tags = [
        line.strip().removeprefix("refs/tags/")
        for line in args.tags_file.read_text().splitlines()
        if line.strip()
    ]
    releases = [
        json.loads(line)
        for line in args.releases_file.read_text().splitlines()
        if line.strip()
    ]
    try:
        publish, reason = decide(
            version=args.version,
            tags=tags,
            releases=releases,
            tag_sha=args.tag_sha,
            head_sha=args.head_sha,
        )
    except UnsafeRelease as exc:
        print(f"::error title=Release blocked::{exc}", file=sys.stderr)
        return 1
    print(f"publish={'true' if publish else 'false'}")
    print(reason, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
