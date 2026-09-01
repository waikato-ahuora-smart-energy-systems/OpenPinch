"""Validate the strict project release version and optional forward advance."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

_VERSION_PART = r"(?:0|[1-9]\d*)"
RELEASE_VERSION_PATTERN = re.compile(
    rf"(?P<major>{_VERSION_PART})\."
    rf"(?P<minor>{_VERSION_PART})\."
    rf"(?P<patch>{_VERSION_PART})\Z"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_project_version(pyproject: Path) -> str:
    """Return one canonical ``X.Y.Z`` project version."""
    with pyproject.open("rb") as handle:
        version = str(tomllib.load(handle)["project"]["version"])
    if RELEASE_VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(
            f"Project version {version!r} must use the exact form X.Y.Z."
        )
    return version


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = RELEASE_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"Project version {version!r} must use the exact form X.Y.Z.")
    return tuple(int(match.group(name)) for name in ("major", "minor", "patch"))


def validate_version_advance(pyproject: Path, base_pyproject: Path) -> str:
    """Return the current version when it strictly exceeds the base version."""
    current = read_project_version(pyproject)
    base = read_project_version(base_pyproject)
    if _version_tuple(current) <= _version_tuple(base):
        raise ValueError(
            f"Release version {current!r} must be greater than base version {base!r}."
        )
    return current


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=REPO_ROOT / "pyproject.toml",
        help="Project metadata containing the candidate release version.",
    )
    parser.add_argument(
        "--base-pyproject",
        type=Path,
        default=None,
        help="Optional base-branch metadata that the candidate must exceed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate and print the candidate release version."""
    args = build_parser().parse_args(argv)
    try:
        version = (
            read_project_version(args.pyproject)
            if args.base_pyproject is None
            else validate_version_advance(args.pyproject, args.base_pyproject)
        )
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(exc)
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
