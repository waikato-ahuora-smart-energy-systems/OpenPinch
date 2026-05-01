"""CLI integration tests for the user-facing OpenPinch workflows."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

import OpenPinch.__main__ as cli
from OpenPinch.resources import copy_sample_case, list_notebooks


def test_validate_command_accepts_packaged_sample(tmp_path: Path):
    case_path = copy_sample_case("basic_pinch.json", tmp_path / "basic_pinch.json")
    assert cli.main(["validate", str(case_path), "--quiet"]) == 0


def test_run_command_writes_summary_excel_json_and_graphs(tmp_path: Path, capsys):
    case_path = copy_sample_case("basic_pinch.json", tmp_path / "basic_pinch.json")
    json_path = tmp_path / "results.json"
    graph_dir = tmp_path / "graphs"
    excel_dir = tmp_path / "excel"

    assert (
        cli.main(
            [
                "run",
                str(case_path),
                "-o",
                str(excel_dir),
                "--json-output",
                str(json_path),
                "--graph-output",
                str(graph_dir),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "Hot Utility Target" in captured.out
    assert json.loads(json_path.read_text(encoding="utf-8"))["name"]
    assert any(graph_dir.glob("*.html"))
    assert any(excel_dir.glob("*.xlsx"))


def test_graph_command_filters_and_exports_html(tmp_path: Path):
    case_path = copy_sample_case("basic_pinch.json", tmp_path / "basic_pinch.json")
    graph_dir = tmp_path / "graphs"

    assert (
        cli.main(
            [
                "graph",
                str(case_path),
                "--graph-type",
                "gcc",
                "-o",
                str(graph_dir),
            ]
        )
        == 0
    )

    written = list(graph_dir.glob("*.html"))
    assert written
    assert all("grand_composite_curve" in path.stem for path in written)


def test_sample_and_notebook_commands_copy_packaged_assets(tmp_path: Path):
    sample_out = tmp_path / "sample.json"
    notebook_dir = tmp_path / "notebooks"

    assert cli.main(["sample", "--name", "basic_pinch.json", "-o", str(sample_out)]) == 0
    assert sample_out.exists()

    assert cli.main(["notebook", "-o", str(notebook_dir)]) == 0
    copied = sorted(path.name for path in notebook_dir.glob("*.ipynb"))
    assert copied == list_notebooks()


def test_validate_command_surfaces_user_facing_error(tmp_path: Path, capsys):
    bad_case = tmp_path / "bad.json"
    bad_case.write_text(
        json.dumps(
            {
                "streams": [
                    {
                        "zone": "Zone A",
                        "name": "Hot Feed",
                        "t_supply": 150.0,
                        "heat_flow": 100.0,
                        "dt_cont": 10.0,
                        "htc": 1.0,
                    }
                ],
                "utilities": [],
                "options": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="1"):
        cli.main(["validate", str(bad_case)])

    captured = capsys.readouterr()
    assert "OpenPinch error: Input validation failed" in captured.err
    assert "Hot Feed" in captured.err
    assert "t_target" in captured.err


def test_validate_command_debug_raises_original_exception(tmp_path: Path):
    bad_case = tmp_path / "bad.json"
    bad_case.write_text(
        json.dumps(
            {
                "streams": [
                    {
                        "zone": "Zone A",
                        "name": "Hot Feed",
                        "t_supply": 150.0,
                        "heat_flow": 100.0,
                        "dt_cont": 10.0,
                        "htc": 1.0,
                    }
                ],
                "utilities": [],
                "options": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Input validation failed"):
        cli.main(["--debug", "validate", str(bad_case)])
