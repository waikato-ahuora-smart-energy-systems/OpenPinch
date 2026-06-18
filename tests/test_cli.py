"""CLI integration tests for the user-facing notebook-copy workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

import OpenPinch.__main__ as cli
from OpenPinch.resources import list_notebooks


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--help"], ["notebook"]),
        (["notebook", "--help"], ["--name", "--output"]),
    ],
)
def test_help_surfaces_supported_commands_and_flags(argv, expected, capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.main(argv)

    captured = capsys.readouterr()
    for marker in expected:
        assert marker in captured.out
    assert "run" not in captured.out
    assert "validate" not in captured.out
    assert "sample" not in captured.out


def test_notebook_command_copies_full_series(tmp_path: Path, capsys):
    notebook_dir = tmp_path / "notebooks"

    assert cli.main(["notebook", "-o", str(notebook_dir)]) == 0

    captured = capsys.readouterr()
    assert f"Copied {len(list_notebooks())} notebook(s)" in captured.out
    copied = sorted(path.name for path in notebook_dir.glob("*.ipynb"))
    assert copied == list_notebooks()


def test_notebook_command_copies_named_notebook_to_existing_directory(
    tmp_path: Path, capsys
):
    destination = tmp_path / "notebooks"
    destination.mkdir()
    notebook_name = list_notebooks()[0]

    assert cli.main(["notebook", "--name", notebook_name, "-o", str(destination)]) == 0

    captured = capsys.readouterr()
    assert "Copied notebook to" in captured.out
    assert (destination / notebook_name).exists()


def test_notebook_command_copies_named_notebook_to_explicit_file(
    tmp_path: Path, capsys
):
    notebook_name = list_notebooks()[0]
    destination = tmp_path / "custom-name.ipynb"

    assert cli.main(["notebook", "--name", notebook_name, "-o", str(destination)]) == 0

    captured = capsys.readouterr()
    assert "Copied notebook to" in captured.out
    assert destination.exists()
    assert destination.name == "custom-name.ipynb"


@pytest.mark.parametrize("removed_command", ["run", "validate", "sample"])
def test_removed_commands_are_rejected(removed_command, capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main([removed_command, "--help"])

    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_notebook_command_rejects_invalid_name(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main(["notebook", "--name", "missing.ipynb"])

    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_notebook_command_wraps_copy_errors(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        cli,
        "copy_notebook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main(["notebook", "--name", list_notebooks()[0], "-o", str(tmp_path)])

    captured = capsys.readouterr()
    assert "OpenPinch error: disk full" in captured.err


def test_notebook_command_debug_raises_original_exception(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        cli,
        "copy_notebook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        cli.main(
            [
                "--debug",
                "notebook",
                "--name",
                list_notebooks()[0],
                "-o",
                str(tmp_path),
            ]
        )
