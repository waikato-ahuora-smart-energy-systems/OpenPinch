"""Compare local distributions with one release on a PEP 503 package index."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ReleaseValidationError(ValueError):
    """Raised when published release files do not match local distributions."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_distribution_hashes(
    dist_dir: Path, *, project: str, version: str
) -> dict[str, str]:
    """Return the exact wheel and source-distribution hashes for one release."""
    wheel_stem = re.sub(r"[-_.]+", "_", project).lower()
    sdist_stem = re.sub(r"[-_.]+", "-", project).lower()
    expected_names = {
        f"{wheel_stem}-{version}-py3-none-any.whl",
        f"{sdist_stem}-{version}.tar.gz",
    }
    actual_files = {
        path.name: path
        for path in dist_dir.iterdir()
        if path.is_file()
        and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
    }
    if set(actual_files) != expected_names:
        raise ReleaseValidationError(
            "Local distributions must be exactly "
            f"{sorted(expected_names)!r}; found {sorted(actual_files)!r}."
        )
    return {name: _sha256(actual_files[name]) for name in sorted(actual_files)}


def _read_index_response(
    url: str,
    *,
    opener: Callable[..., BinaryIO],
    attempts: int,
    retry_delay: float,
) -> Mapping[str, object] | None:
    request = Request(url, headers={"User-Agent": "OpenPinch-release-verifier/1"})
    for attempt in range(1, attempts + 1):
        try:
            with opener(request, timeout=20.0) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ReleaseValidationError("Package-index response is not an object.")
            return payload
        except HTTPError as exc:
            if exc.code == 404:
                return None
            transient = exc.code in {408, 425, 429} or exc.code >= 500
            if not transient or attempt == attempts:
                raise ReleaseValidationError(
                    f"Package-index request failed with HTTP {exc.code}."
                ) from exc
        except (TimeoutError, URLError) as exc:
            if attempt == attempts:
                raise ReleaseValidationError(
                    f"Package-index request failed after {attempts} attempts: {exc}"
                ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReleaseValidationError(
                "Package-index response is not valid JSON."
            ) from exc
        time.sleep(retry_delay)
    raise AssertionError("unreachable package-index retry state")


def inspect_release(
    *,
    index_url: str,
    project: str,
    version: str,
    expected_files: Mapping[str, str],
    opener: Callable[..., BinaryIO] = urlopen,
    attempts: int = 5,
    retry_delay: float = 2.0,
) -> str:
    """Return ``absent``, ``partial``, or ``complete`` for an exact release."""
    endpoint = (
        f"{index_url.rstrip('/')}/{quote(project, safe='')}/"
        f"{quote(version, safe='')}/json"
    )
    payload = _read_index_response(
        endpoint,
        opener=opener,
        attempts=attempts,
        retry_delay=retry_delay,
    )
    if payload is None:
        return "absent"

    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise ReleaseValidationError("Package-index response has no file list.")

    published: dict[str, str] = {}
    for entry in urls:
        if not isinstance(entry, dict):
            raise ReleaseValidationError("Package-index file entry is invalid.")
        filename = entry.get("filename")
        digests = entry.get("digests")
        sha256 = digests.get("sha256") if isinstance(digests, dict) else None
        if not isinstance(filename, str) or not isinstance(sha256, str):
            raise ReleaseValidationError(
                "Package-index file entry lacks a filename or SHA-256 digest."
            )
        if filename in published:
            raise ReleaseValidationError(
                f"Package-index response repeats filename {filename!r}."
            )
        published[filename] = sha256.lower()

    unexpected = set(published) - set(expected_files)
    if unexpected:
        raise ReleaseValidationError(
            f"Package index contains unexpected release files: {sorted(unexpected)!r}."
        )
    mismatched = [
        name
        for name, digest in published.items()
        if digest != expected_files[name].lower()
    ]
    if mismatched:
        raise ReleaseValidationError(
            f"Package index contains mismatched files: {sorted(mismatched)!r}."
        )
    if not published:
        return "absent"
    if set(published) == set(expected_files):
        return "complete"
    return "partial"


def wait_for_complete_release(
    *,
    inspect: Callable[[], str],
    attempts: int = 6,
    retry_delay: float = 10.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> str:
    """Retry exact release inspection while an index propagates new files."""
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    for attempt in range(1, attempts + 1):
        status = inspect()
        if status == "complete" or attempt == attempts:
            return status
        sleeper(retry_delay)
    raise AssertionError("unreachable package-index verification retry state")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--project", default="OpenPinch")
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-partial", action="store_true")
    mode.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate local files against one package-index release."""
    args = build_parser().parse_args(argv)
    try:
        expected = expected_distribution_hashes(
            args.dist_dir,
            project=args.project,
            version=args.version,
        )

        def inspect() -> str:
            return inspect_release(
                index_url=args.index_url,
                project=args.project,
                version=args.version,
                expected_files=expected,
                opener=urlopen,
            )

        status = (
            wait_for_complete_release(inspect=inspect)
            if args.require_complete
            else inspect()
        )
        if args.require_complete and status != "complete":
            raise ReleaseValidationError(
                f"Published release is {status}; expected a complete exact release."
            )
    except (OSError, ReleaseValidationError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
