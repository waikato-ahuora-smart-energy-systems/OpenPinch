"""Check that uv.lock carries the current editable project version."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main() -> int:
    project_version = _read_toml(REPO_ROOT / "pyproject.toml")["project"]["version"]
    lock = _read_toml(REPO_ROOT / "uv.lock")

    package = next(
        (
            package
            for package in lock["package"]
            if package["name"] == "openpinch"
            and package.get("source") == {"editable": "."}
        ),
        None,
    )
    if package is None:
        print(
            "uv.lock does not contain the editable openpinch package entry.",
            file=sys.stderr,
        )
        return 1

    lock_version = package["version"]
    if lock_version != project_version:
        print(
            "uv.lock openpinch version "
            f"{lock_version!r} does not match pyproject.toml version "
            f"{project_version!r}.",
            file=sys.stderr,
        )
        print("Run `uv lock` after changing the project version.", file=sys.stderr)
        return 1

    print(f"uv.lock openpinch version matches pyproject.toml ({project_version}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
