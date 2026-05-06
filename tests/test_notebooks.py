# """Smoke and content tests for the packaged OpenPinch notebook series."""

# from __future__ import annotations

# import json
# from pathlib import Path

# import pytest

# from OpenPinch.resources import copy_notebook, list_notebooks


# EXPECTED_NOTEBOOKS = [
#     "01_basic_pinch_and_dtcont_sensitivity.ipynb",
#     "02_total_site_targets_and_sugcc.ipynb",
#     "03_carnot_hpr_comparison.ipynb",
# ]


# def _load_notebook(path: Path) -> dict:
#     return json.loads(path.read_text(encoding="utf-8"))


# def _combined_source(notebook: dict) -> str:
#     return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


# def _execute_notebook(path: Path) -> None:
#     notebook = _load_notebook(path)
#     namespace = {"__name__": "__main__"}
#     for cell in notebook["cells"]:
#         if cell.get("cell_type") != "code":
#             continue
#         source = "".join(cell.get("source", []))
#         exec(compile(source, str(path), "exec"), namespace)


# def test_packaged_notebook_series_is_present():
#     assert list_notebooks() == EXPECTED_NOTEBOOKS


# @pytest.mark.parametrize("notebook_name", EXPECTED_NOTEBOOKS)
# def test_packaged_notebooks_execute_smoke(
#     tmp_path: Path, monkeypatch, notebook_name: str
# ):
#     notebook_path = copy_notebook(notebook_name, tmp_path / notebook_name)
#     monkeypatch.chdir(tmp_path)
#     _execute_notebook(notebook_path)


# def test_notebook_1_uses_real_case_and_dtcont_sensitivity(tmp_path: Path):
#     notebook_path = copy_notebook(
#         "01_basic_pinch_and_dtcont_sensitivity.ipynb",
#         tmp_path / "01_basic_pinch_and_dtcont_sensitivity.ipynb",
#     )
#     notebook = _load_notebook(notebook_path)
#     combined_source = _combined_source(notebook)
#     lead_markdown = "".join(notebook["cells"][0].get("source", []))

#     assert "basic pinch analysis" in lead_markdown.lower()
#     assert "`dt_cont`" in combined_source
#     assert "crude_preheat_train.json" in combined_source
#     assert "plot_composite_curve" in combined_source
#     assert "variant=\"shifted\"" in combined_source
#     assert "plot_grand_composite_curve" in combined_source
#     assert "multipliers =" in combined_source
#     assert "dt_cont multiplier" in combined_source
#     assert "display_plotly" in combined_source
#     assert "PLOT_WIDTH = 720" in combined_source
#     assert "PLOT_HEIGHT = 540" in combined_source
#     assert "figure.update_layout(width=width, height=height, autosize=False)" in combined_source
#     assert "figure.to_html" in combined_source
#     assert "make_subplots" in combined_source
#     assert "sensitivity_fig" in combined_source


# def test_notebook_2_covers_total_site_sugcc_and_cogeneration(tmp_path: Path):
#     notebook_path = copy_notebook(
#         "02_total_site_targets_and_sugcc.ipynb",
#         tmp_path / "02_total_site_targets_and_sugcc.ipynb",
#     )
#     notebook = _load_notebook(notebook_path)
#     combined_source = _combined_source(notebook)
#     lead_markdown = "".join(notebook["cells"][0].get("source", []))

#     assert "total site" in lead_markdown.lower()
#     assert "pulp_mill.json" in combined_source
#     assert "display_plotly" in combined_source
#     assert "PLOT_WIDTH = 720" in combined_source
#     assert "PLOT_HEIGHT = 540" in combined_source
#     assert "figure.update_layout(width=width, height=height, autosize=False)" in combined_source
#     assert "figure.to_html" in combined_source
#     assert "Site Utility Grand Composite Curve" in combined_source
#     assert "Grand Composite Curve" in combined_source
#     assert "target_cogeneration" in combined_source
#     assert "\"base_target_type\": \"Total Site Target\"" in combined_source
#     assert "Total Site Profiles" in combined_source
#     assert "Power Cogeneration Target" in combined_source


# def test_notebook_3_covers_direct_and_indirect_carnot_hpr(tmp_path: Path):
#     notebook_path = copy_notebook(
#         "03_carnot_hpr_comparison.ipynb",
#         tmp_path / "03_carnot_hpr_comparison.ipynb",
#     )
#     notebook = _load_notebook(notebook_path)
#     combined_source = _combined_source(notebook)
#     lead_markdown = "".join(notebook["cells"][0].get("source", []))

#     assert "carnot hpr comparison" in lead_markdown.lower()
#     assert "chocolate_factory.json" in combined_source
#     assert "target_direct_heat_pump" in combined_source
#     assert "target_direct_refrigeration" in combined_source
#     assert "target_indirect_heat_pump" in combined_source
#     assert "target_indirect_refrigeration" in combined_source
#     assert "load_fractions = [0.25, 0.5]" in combined_source
