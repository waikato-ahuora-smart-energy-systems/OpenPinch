"""Validate that a release tag exactly matches the project version."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

_VERSION_PART = r"(?:0|[1-9]\d*)"
RELEASE_TAG_PATTERN = re.compile(
    rf"v(?P<version>{_VERSION_PART}\.{_VERSION_PART}\.{_VERSION_PART})\Z"
)


def expected_release_tag(pyproject: Path) -> str:
    """Return the release tag declared by the project metadata."""
    with pyproject.open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    return f"v{version}"


def validate_release_tag(tag: str, pyproject: Path) -> None:
    """Raise ``ValueError`` unless ``tag`` is valid and current."""
    if RELEASE_TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"Release tag {tag!r} must use the exact form vX.Y.Z.")

    expected = expected_release_tag(pyproject)
    if tag != expected:
        raise ValueError(
            f"Release tag {tag!r} does not match the project version; "
            f"expected {expected!r}."
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Git tag to validate, normally github.ref_name.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
        help="Project metadata file used to resolve the expected version.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the requested release tag."""
    args = build_parser().parse_args(argv)
    try:
        validate_release_tag(args.tag, args.pyproject)
    except ValueError as exc:
        print(exc)
        return 1
    print(f"Release tag {args.tag!r} matches the project version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
