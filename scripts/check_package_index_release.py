"""Compare local distributions with one release on a PEP 503 package index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import ssl
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


class ReleaseValidationError(ValueError):
    """Raised when published release files do not match local distributions."""


class TransientIndexError(ReleaseValidationError):
    """A retryable request failure, never evidence of an absent release."""

    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


def retry_after_seconds(value: str | None) -> float:
    """Parse Retry-After without allowing nonfinite or negative delays."""
    if value is None:
        return 0
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (
                parsedate_to_datetime(value) - datetime.now(timezone.utc)
            ).total_seconds()
        except ValueError, TypeError, OverflowError:
            return 0
    return max(0, delay) if math.isfinite(delay) else 0


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
    timeout: float,
) -> Mapping[str, object] | None:
    request = Request(url, headers={"User-Agent": "OpenPinch-release-verifier/1"})
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ReleaseValidationError("Package-index response exceeds size limit.")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ReleaseValidationError("Package-index response is not an object.")
        return payload
    except HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code in {408, 425, 429} or 500 <= exc.code < 600:
            raise TransientIndexError(
                f"Package-index HTTP {exc.code}",
                retry_after_seconds(exc.headers.get("Retry-After")),
            ) from exc
        raise ReleaseValidationError(f"Package-index HTTP {exc.code}") from exc
    except (ssl.SSLError, URLError, TimeoutError, ConnectionError) as exc:
        if isinstance(exc, ssl.SSLError) or isinstance(
            getattr(exc, "reason", None), ssl.SSLError
        ):
            raise ReleaseValidationError(
                "Package-index TLS verification failed."
            ) from exc
        raise TransientIndexError("Package-index transport unavailable.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReleaseValidationError(
            "Package-index response is not valid JSON."
        ) from exc


def inspect_release(
    *,
    index_url: str,
    project: str,
    version: str,
    expected_files: Mapping[str, str],
    opener: Callable[..., BinaryIO] = urlopen,
    timeout: float = 20.0,
) -> str:
    """Return ``absent``, ``partial``, or ``complete`` for an exact release."""
    parsed = urlsplit(index_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ReleaseValidationError(
            "Index URL must be HTTPS without credentials, query, or fragment."
        )
    endpoint = (
        f"{index_url.rstrip('/')}/{quote(project, safe='')}/"
        f"{quote(version, safe='')}/json"
    )
    print(f"Destination={parsed.hostname}; version={version}", file=sys.stderr)
    payload = _read_index_response(
        endpoint,
        opener=opener,
        timeout=timeout,
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
    print(
        f"Missing files: {sorted(set(expected_files) - set(published))}",
        file=sys.stderr,
    )
    return "partial"


def wait_for_complete_release(
    *,
    inspect: Callable[[float], str],
    budget: float = 300.0,
    retry_delay: float = 10.0,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    require_complete: bool = True,
) -> str:
    """One deadline owns all visibility and transport retries; no nested sleeps."""
    if not math.isfinite(budget) or not 1 <= budget <= 600:
        raise ValueError("budget must be finite and between 1 and 600 seconds")
    if not math.isfinite(retry_delay) or retry_delay <= 0:
        raise ValueError("retry_delay must be positive and finite")
    started = clock()
    deadline = started + budget
    attempt = 0
    status = "unobserved"
    while (remaining := deadline - clock()) > 0:
        attempt += 1
        delay = retry_delay
        try:
            status = inspect(min(20.0, remaining))
            if status not in {"absent", "partial", "complete"}:
                raise ReleaseValidationError("Unknown index observation")
        except TransientIndexError as exc:
            status = f"transport exhaustion ({exc})"
            delay = max(delay, exc.retry_after)
        elapsed = clock() - started
        print(
            f"Attempt {attempt}; elapsed={elapsed:.1f}s; state={status}",
            file=sys.stderr,
        )
        if clock() >= deadline:
            break
        if status == "complete" or (
            not require_complete and status in {"absent", "partial"}
        ):
            return status
        sleeper(min(delay, max(0, deadline - clock())))
    raise ReleaseValidationError(
        f"Verification deadline exhausted after {budget:g}s; last state: {status}."
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", required=True)
    parser.add_argument("--project", default="OpenPinch")
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Total budget, 1-600 seconds (postflight 300; preflight 60).",
    )
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

        def inspect(timeout: float) -> str:
            return inspect_release(
                index_url=args.index_url,
                project=args.project,
                version=args.version,
                expected_files=expected,
                opener=urlopen,
                timeout=timeout,
            )

        print(
            f"Verifying project={args.project} version={args.version}", file=sys.stderr
        )
        status = wait_for_complete_release(
            inspect=inspect,
            budget=args.timeout
            if args.timeout is not None
            else (300 if args.require_complete else 60),
            require_complete=args.require_complete,
        )
        if args.require_complete and status != "complete":
            raise ReleaseValidationError(
                f"Published release is {status}; expected a complete exact release."
            )
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
