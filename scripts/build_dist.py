"""Build the local source and wheel distributions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import which

_DIST_ARTIFACT_PATTERNS = (
    "openpinch-*.whl",
    "openpinch-*.tar.gz",
    "openpinch-*.whl.publish.attestation",
    "openpinch-*.tar.gz.publish.attestation",
)


def _has_hatchling_backend() -> bool:
    """Return whether the active interpreter can import ``hatchling.build``."""
    try:
        return find_spec("hatchling.build") is not None
    except ModuleNotFoundError:
        return False


def _build_command(output_dir: Path) -> list[str] | None:
    """Return the preferred build command for this environment."""
    if find_spec("build") is not None:
        command = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--outdir",
            str(output_dir),
        ]
        if _has_hatchling_backend():
            command.insert(-2, "--no-isolation")
        return command

    uv_bin = which("uv")
    if uv_bin is None:
        return None

    return [
        uv_bin,
        "run",
        "--group",
        "dev",
        "python",
        "-m",
        "build",
        "--wheel",
        "--sdist",
        "--no-isolation",
        "--outdir",
        str(output_dir),
    ]


def _clean_output_dir(output_dir: Path) -> None:
    """Remove stale OpenPinch distribution artifacts before building."""
    for pattern in _DIST_ARTIFACT_PATTERNS:
        for child in output_dir.glob(pattern):
            if child.is_file() or child.is_symlink():
                child.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for distribution builds."""
    parser = argparse.ArgumentParser(
        prog="build_dist.py",
        description="Build the OpenPinch wheel and source distribution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for wheel and sdist artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the package into ``dist/`` by default."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir is not None else repo_root / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_output_dir(output_dir)

    command = _build_command(output_dir)
    if command is None:
        print(
            "Unable to build distributions: install the 'build' module in the "
            "active interpreter or make 'uv' available on PATH.",
            file=sys.stderr,
        )
        return 1

    proc = subprocess.run(command, cwd=repo_root)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
