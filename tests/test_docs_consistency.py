"""Consistency checks for the user-facing documentation surface."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
GETTING_STARTED = REPO_ROOT / "docs" / "getting-started.rst"
QUICKSTART = REPO_ROOT / "docs" / "user-guide" / "quickstart.rst"
NOTEBOOKS = REPO_ROOT / "docs" / "user-guide" / "notebooks.rst"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_highlights_current_cli_workflow():
    text = _read(README)
    assert "openpinch sample" in text
    assert "openpinch run" in text
    assert "openpinch graph" in text
    assert "openpinch validate" in text
    assert "openpinch notebook" in text


def test_docs_highlight_current_pinchproblem_methods():
    combined = "\n".join(
        [
            _read(README),
            _read(GETTING_STARTED),
            _read(QUICKSTART),
            _read(NOTEBOOKS),
        ]
    )
    assert "run()" in combined
    assert "summary_frame()" in combined
    assert "export_excel" in combined
    assert "show_dashboard()" in combined
    assert "plot_grand_composite_curve" in combined


def test_docs_do_not_reference_stale_workflow_names():
    combined = "\n".join(
        [
            _read(README),
            _read(GETTING_STARTED),
            _read(QUICKSTART),
        ]
    )
    assert "problem.export(" not in combined
    assert "Python 3.11 or newer" not in combined
