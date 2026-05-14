"""Content checks for the packaged OpenPinch notebook series."""

from __future__ import annotations

import json
from pathlib import Path

from OpenPinch.resources import copy_notebook, list_notebooks


EXPECTED_NOTEBOOKS = [
    "01_basic_pinch_and_dtcont_sensitivity.ipynb",
    "02_total_site_targets_and_sugcc.ipynb",
    "03_carnot_hpr_comparison.ipynb",
]


def _load_notebook(path: Path) -> dict:
    """Return the notebook JSON document from ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))


def _combined_source(notebook: dict) -> str:
    """Return all notebook cell sources concatenated into one string."""
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_packaged_notebook_series_is_present():
    """Keep the packaged notebook inventory synchronized with the docs."""
    assert list_notebooks() == EXPECTED_NOTEBOOKS


def test_notebook_2_covers_total_site_graph_payloads_and_cogeneration(
    tmp_path: Path,
):
    """Verify the total-site notebook still teaches the intended stable workflow."""
    notebook_path = copy_notebook(
        "02_total_site_targets_and_sugcc.ipynb",
        tmp_path / "02_total_site_targets_and_sugcc.ipynb",
    )
    notebook = _load_notebook(notebook_path)
    combined_source = _combined_source(notebook)
    lead_markdown = "".join(notebook["cells"][0].get("source", []))

    assert "total site" in lead_markdown.lower()
    assert "pulp_mill.json" in combined_source
    assert "workspace = PinchWorkspace(" in combined_source
    assert "catalog = baseline.plot.catalog()" in combined_source
    assert "baseline.plot.get_graph_data()" in combined_source
    assert "baseline.plot.total_site_profiles" in combined_source
    assert "baseline.plot.site_utility_grand_composite_curve" in combined_source
    assert "Total Site Profiles" in combined_source
    assert "Site Utility Grand Composite Curve" in combined_source
    assert '"base_target_type": "Total Site Target"' in combined_source
    assert "Power Cogeneration Target" in combined_source


def test_packaged_notebooks_use_pinch_workspace_without_local_helpers(tmp_path: Path):
    """The packaged notebooks should use the library API directly."""
    for notebook_name in EXPECTED_NOTEBOOKS:
        notebook_path = copy_notebook(notebook_name, tmp_path / notebook_name)
        notebook = _load_notebook(notebook_path)
        combined_source = _combined_source(notebook)

        assert "PinchWorkspace" in combined_source
        assert "read_sample_case" not in combined_source
        assert "json.loads(" not in combined_source
        assert "def " not in combined_source
