"""Release artifact checks for packaging boundaries."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from zipfile import ZipFile

import pytest


def _build_command(out_dir: Path) -> list[str] | None:
    """Return a build command for the active interpreter or the uv dev env."""
    if find_spec("build") is not None:
        return [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(out_dir),
        ]

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
        str(out_dir),
    ]


def _build_artifacts(tmp_path: Path) -> tuple[Path, list[str], list[str]]:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    command = _build_command(out_dir)
    if command is None:
        pytest.skip("release artifact build requires the 'build' module or uv")

    proc = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    wheel_paths = sorted(out_dir.glob("*.whl"))
    sdist_paths = sorted(out_dir.glob("*.tar.gz"))
    assert wheel_paths, (
        f"no wheel created in {out_dir}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert sdist_paths, (
        f"no sdist created in {out_dir}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    wheel_path = wheel_paths[0]
    sdist_path = sdist_paths[0]

    with ZipFile(wheel_path) as wheel:
        wheel_names = wheel.namelist()

    with tarfile.open(sdist_path, "r:gz") as sdist:
        sdist_names = sdist.getnames()

    return wheel_path, wheel_names, sdist_names


def _assert_common_release_boundary(names: list[str], *, root_prefix: str = "") -> None:
    prefix = f"{root_prefix}/" if root_prefix else ""

    assert f"{prefix}OpenPinch/__init__.py" in names
    assert f"{prefix}__init__.py" not in names

    forbidden_prefixes = [
        f"{prefix}examples/",
        f"{prefix}tests/",
        f"{prefix}Excel_Version/",
    ]
    forbidden_fragments = [
        "streamlit_app.py",
    ]

    for disallowed_prefix in forbidden_prefixes:
        assert not any(name.startswith(disallowed_prefix) for name in names)
    for fragment in forbidden_fragments:
        assert not any(fragment in name for name in names)

    assert any(
        name.endswith("OpenPinch/data/sample_cases/basic_pinch.json") for name in names
    )
    assert any(
        name.endswith("OpenPinch/data/sample_cases/heat_pump_targeting.json")
        for name in names
    )
    assert any(
        name.endswith(
            "OpenPinch/data/notebooks/01_basic_pinch_and_dtcont_sensitivity.ipynb"
        )
        for name in names
    )


def test_release_artifacts_exclude_repo_only_assets(tmp_path: Path):
    _, wheel_names, sdist_names = _build_artifacts(tmp_path)
    _assert_common_release_boundary(wheel_names)

    sdist_root = next(
        name.split("/", 1)[0] for name in sdist_names if "/" in name
    )
    _assert_common_release_boundary(sdist_names, root_prefix=sdist_root)
