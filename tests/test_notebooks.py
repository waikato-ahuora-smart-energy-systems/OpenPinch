"""Content checks for the packaged OpenPinch notebook series."""

from __future__ import annotations

import json
from pathlib import Path

from OpenPinch.resources import copy_notebook, list_notebooks

EXPECTED_NOTEBOOKS = [
    "01_basic_pinch_and_dtcont_sensitivity.ipynb",
    "02_total_site_targets_and_sugcc.ipynb",
    "03_carnot_hpr_comparison.ipynb",
    "04_multistate_targeting_and_state_comparison.ipynb",
    "05_schema_service_and_output_workflows.ipynb",
    "06_energy_transfer_analysis.ipynb",
    "07_vapour_compression_mvr_cascade_hpr.ipynb",
    "08_direct_gas_stream_mvr.ipynb",
]


def _load_notebook(path: Path) -> dict:
    """Return the notebook JSON document from ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))


def _combined_source(notebook: dict) -> str:
    """Return all notebook cell sources concatenated into one string."""
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _copied_notebook(tmp_path: Path, notebook_name: str) -> dict:
    return _load_notebook(copy_notebook(notebook_name, tmp_path / notebook_name))


def test_packaged_notebook_series_is_present():
    """Keep the packaged notebook inventory synchronized with the docs."""
    assert list_notebooks() == EXPECTED_NOTEBOOKS


def test_packaged_notebooks_are_output_free_and_use_library_surfaces(tmp_path: Path):
    """The packaged notebooks should stay source-only and API-forward."""
    for notebook_name in EXPECTED_NOTEBOOKS:
        notebook = _copied_notebook(tmp_path, notebook_name)
        combined_source = _combined_source(notebook)

        assert "read_sample_case" not in combined_source
        assert "json.loads(" not in combined_source
        assert "def " not in combined_source
        assert all(cell.get("execution_count") is None for cell in notebook["cells"])
        assert all(
            not cell.get("outputs")
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )


def test_notebook_1_covers_single_case_and_workspace_sensitivity(tmp_path: Path):
    notebook = _copied_notebook(
        tmp_path,
        "01_basic_pinch_and_dtcont_sensitivity.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "PinchProblem" in combined_source
    assert "PinchWorkspace" in combined_source
    assert ".validate()" in combined_source
    assert "summary_frame()" in combined_source
    assert "plot.catalog()" in combined_source
    assert "plot.composite_curve" in combined_source
    assert "plot.shifted_composite_curve" in combined_source
    assert "plot.grand_composite_curve" in combined_source
    assert "target.area_cost()" in combined_source
    assert "copy_case(" in combined_source
    assert "set_dt_cont_multiplier(" in combined_source
    assert "compare_cases(" in combined_source
    assert 'state_id="0"' not in combined_source


def test_notebook_2_uses_only_packaged_pulp_mill_assets_and_real_zones(
    tmp_path: Path,
):
    notebook = _copied_notebook(
        tmp_path,
        "02_total_site_targets_and_sugcc.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "pulp_mill.json" in combined_source
    assert "p_Varbanov et al.json" not in combined_source
    assert 'source="pulp_mill.json"' in combined_source
    assert 'zone_name="Bleaching"' in combined_source
    assert "plot.get_graph_data()" in combined_source
    assert "plot.total_site_profiles" in combined_source
    assert "plot.site_utility_grand_composite_curve" in combined_source
    assert '"base_target_type": "Total Site Target"' in combined_source
    assert '"base_target_type": "Direct Integration"' in combined_source
    assert "Power Cogeneration Target" in combined_source
    assert 'state_id="0"' not in combined_source


def test_notebook_3_compares_direct_indirect_and_carnot_backends(
    tmp_path: Path,
):
    notebook = _copied_notebook(
        tmp_path,
        "03_carnot_hpr_comparison.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "plot_multi_hp_profiles_from_results" not in combined_source
    assert "direct_heat_pump" in combined_source
    assert "indirect_heat_pump" in combined_source
    assert "direct_refrigeration" not in combined_source
    assert "indirect_refrigeration" not in combined_source
    assert "HPRcycle.CascadeCarnot" in combined_source
    assert "HPRcycle.ParallelCarnot" in combined_source
    assert "SOLVED_TARGETS" in combined_source
    assert "plot.catalog()" in combined_source
    assert "selected_problem.plot.net_load_profiles_with_heat_pump(" in combined_source
    assert (
        "selected_problem.plot.grand_composite_curve_with_heat_pump("
        in combined_source
    )
    assert "compare_cases(" in combined_source


def test_notebook_4_covers_real_multistate_targeting(tmp_path: Path):
    notebook = _copied_notebook(
        tmp_path,
        "04_multistate_targeting_and_state_comparison.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "crude_preheat_train_multistate.json" in combined_source
    assert "zonal_site_multistate.json" in combined_source
    assert "state_ids" in combined_source
    assert "target_all_states(" in combined_source
    assert "direct_heat_integration(state_id=" in combined_source
    assert "indirect_heat_integration(state_id=" in combined_source
    assert "turndown" in combined_source
    assert "summer" in combined_source


def test_notebook_5_covers_service_boundary_and_output_workflows(tmp_path: Path):
    notebook = _copied_notebook(
        tmp_path,
        "05_schema_service_and_output_workflows.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "copy_sample_case" in combined_source
    assert "TargetInput" in combined_source
    assert "pinch_analysis_service" in combined_source
    assert "export_excel" in combined_source
    assert "plot.export" in combined_source
    assert "payload_view(" in combined_source
    assert "validate_variant(" in combined_source
    assert "solve_variant(" in combined_source
    assert "compare_variants(" in combined_source
    assert "save_bundle(" in combined_source
    assert "load_bundle(" in combined_source


def test_notebook_6_covers_energy_transfer_analysis(tmp_path: Path):
    notebook = _copied_notebook(
        tmp_path,
        "06_energy_transfer_analysis.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "pulp_mill.json" in combined_source
    assert "target.energy_transfer(" in combined_source
    assert "heat_surplus_deficit_table" in combined_source
    assert "energy_transfer_diagram" in combined_source
    assert "plot.energy_transfer_diagram" in combined_source
    assert '"base_target_type": "Total Site Target"' in combined_source
    assert '"base_target_type": "Direct Integration"' in combined_source
    assert 'zone_name="Bleaching"' in combined_source


def test_notebook_7_covers_vapour_compression_mvr_cascade_hpr(tmp_path: Path):
    notebook = _copied_notebook(
        tmp_path,
        "07_vapour_compression_mvr_cascade_hpr.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "HPRcycle.VapourCompMVR" in combined_source
    assert "VapourCompressionMvrCascade" in combined_source
    assert "MechanicalVapourRecompressionCycle" in combined_source
    assert "target.direct_heat_pump()" in combined_source
    assert "plot.net_load_profiles_with_heat_pump(" in combined_source
    assert "plot.grand_composite_curve_with_heat_pump(" in combined_source


def test_notebook_8_covers_direct_gas_stream_mvr(tmp_path: Path):
    notebook = _copied_notebook(
        tmp_path,
        "08_direct_gas_stream_mvr.ipynb",
    )
    combined_source = _combined_source(notebook)

    assert "PinchWorkspace" in combined_source
    assert "copy_case(" in combined_source
    assert "workspace.case(" in combined_source
    assert "workspace.compare_cases(" in combined_source
    assert "add_component.process_mvr(" in combined_source
    assert "after_mvr_liquid_injection" in combined_source
    assert "mvr_liquid_injection" in combined_source
    assert "liquid_injection=True" in combined_source
    assert "problem.process_components" in combined_source
    assert "mvr.deactivate()" in combined_source
    assert "mvr.activate()" in combined_source
    assert "Evaporator vapour" in combined_source
    assert "mvr.replacement_streams" in combined_source
    assert "stage_results_by_state" in combined_source
    assert "target.direct_heat_integration(" in combined_source
    assert "target.indirect_heat_integration(" in combined_source
    assert "summary_frame(case_name=" in combined_source
    assert "plot.grand_composite_curve(" in combined_source
    assert "return_graph_data=True" in combined_source
