"""Helpers for accessing packaged OpenPinch sample cases and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

_SAMPLE_CASE_ROOT = files("OpenPinch.data.sample_cases")
_NOTEBOOK_ROOT = files("OpenPinch.data.notebooks")


@dataclass(frozen=True)
class SampleCaseMetadata:
    """Description of one packaged sample case."""

    name: str
    title: str
    description: str
    topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class NotebookMetadata:
    """Description of one packaged notebook."""

    name: str
    title: str
    description: str
    topics: tuple[str, ...] = ()


_SAMPLE_CASE_METADATA: dict[str, SampleCaseMetadata] = {
    "basic_pinch.json": SampleCaseMetadata(
        name="basic_pinch.json",
        title="Basic Pinch",
        description="Small single-zone problem for first solves and API examples.",
        topics=("pinch", "quickstart"),
    ),
    "chocolate_factory.json": SampleCaseMetadata(
        name="chocolate_factory.json",
        title="Chocolate Factory",
        description="Process heat-pump targeting case with refrigeration examples.",
        topics=("heat pump", "refrigeration"),
    ),
    "crude_preheat_train.json": SampleCaseMetadata(
        name="crude_preheat_train.json",
        title="Crude Preheat Train",
        description="Single-state refinery preheat example for integration targets.",
        topics=("pinch", "refinery"),
    ),
    "crude_preheat_train_multiperiod.json": SampleCaseMetadata(
        name="crude_preheat_train_multiperiod.json",
        title="Crude Preheat Train Multiperiod",
        description="Multiperiod refinery case for period-specific targeting.",
        topics=("multiperiod", "refinery"),
    ),
    "heat_pump_targeting.json": SampleCaseMetadata(
        name="heat_pump_targeting.json",
        title="Heat Pump Targeting",
        description="Focused case for direct and indirect heat-pump workflows.",
        topics=("heat pump",),
    ),
    "pulp_mill.json": SampleCaseMetadata(
        name="pulp_mill.json",
        title="Pulp Mill",
        description="Zonal total-site problem with cogeneration and SUGCC examples.",
        topics=("total site", "cogeneration", "zonal"),
    ),
    "zonal_site.json": SampleCaseMetadata(
        name="zonal_site.json",
        title="Zonal Site",
        description="Multi-zone site model for total-site targeting examples.",
        topics=("zonal", "total site"),
    ),
    "zonal_site_multiperiod.json": SampleCaseMetadata(
        name="zonal_site_multiperiod.json",
        title="Zonal Site Multiperiod",
        description="Multi-zone, multiperiod site model for scenario comparisons.",
        topics=("zonal", "multiperiod"),
    ),
    "Four-stream-Yee-and-Grossmann-1990-1.json": SampleCaseMetadata(
        name="Four-stream-Yee-and-Grossmann-1990-1.json",
        title="Four Stream Yee and Grossmann",
        description="Classic four-stream heat-exchanger-network synthesis benchmark.",
        topics=("synthesis", "benchmark"),
    ),
}


_NOTEBOOK_METADATA: dict[str, NotebookMetadata] = {
    "01_first_solve_summary_graphs.ipynb": NotebookMetadata(
        name="01_first_solve_summary_graphs.ipynb",
        title="First Solve, Summary, and Graphs",
        description=(
            "Foundation notebook for solving one case, reading summary metrics, "
            "graphing, area/cost, and dt_cont workspace sensitivity."
        ),
        topics=("solve", "quickstart", "graphs", "workspace"),
    ),
    "02_total_site_sugcc_interpretation.ipynb": NotebookMetadata(
        name="02_total_site_sugcc_interpretation.ipynb",
        title="Total Site and SUGCC Interpretation",
        description=(
            "Method notebook for local-vs-site targeting, Total Site profiles, "
            "SUGCC, and cogeneration context."
        ),
        topics=("method", "total site", "sugcc", "cogeneration"),
    ),
    "03_multiperiod_workspace_scenarios.ipynb": NotebookMetadata(
        name="03_multiperiod_workspace_scenarios.ipynb",
        title="Multiperiod Workspace Scenarios",
        description=(
            "Scenario notebook for named periods, period-specific targeting, "
            "target_all_periods, and workspace comparison."
        ),
        topics=("solve", "multiperiod", "workspace"),
    ),
    "04_carnot_heat_pump_screening.ipynb": NotebookMetadata(
        name="04_carnot_heat_pump_screening.ipynb",
        title="Carnot Heat Pump Screening",
        description=(
            "Advanced-method notebook for direct/indirect Carnot Heat Pump "
            "screening and HPR graph interpretation."
        ),
        topics=("solve", "advanced", "heat pump", "comparison"),
    ),
    "05_direct_gas_stream_mvr_scenarios.ipynb": NotebookMetadata(
        name="05_direct_gas_stream_mvr_scenarios.ipynb",
        title="Direct Gas Stream MVR Scenarios",
        description=(
            "Advanced-method notebook for process-component MVR, replacement "
            "streams, and baseline-vs-MVR case comparison."
        ),
        topics=("solve", "advanced", "mvr", "workspace"),
    ),
    "06_vapour_compression_mvr_cascade_hpr.ipynb": NotebookMetadata(
        name="06_vapour_compression_mvr_cascade_hpr.ipynb",
        title="Vapour Compression MVR Cascade HPR",
        description=(
            "Method notebook for VC+MVR cascade mechanics, standalone MVR "
            "thermodynamics, stream profiles, and HPR targeting."
        ),
        topics=("method", "heat pump", "mvr", "vapour compression"),
    ),
    "07_heat_exchanger_network_synthesis.ipynb": NotebookMetadata(
        name="07_heat_exchanger_network_synthesis.ipynb",
        title="Heat Exchanger Network Synthesis",
        description=(
            "Advanced-method notebook for public HEN design accessors, ranked "
            "networks, manifests, and grid diagrams."
        ),
        topics=("solve", "advanced", "synthesis", "design"),
    ),
    "08_energy_transfer_analysis.ipynb": NotebookMetadata(
        name="08_energy_transfer_analysis.ipynb",
        title="Energy Transfer Analysis",
        description=(
            "Method notebook for heat-surplus/deficit tables, energy-transfer "
            "diagrams, and target-family selection."
        ),
        topics=("method", "energy transfer", "graphs"),
    ),
    "09_schema_service_exports_and_bundles.ipynb": NotebookMetadata(
        name="09_schema_service_exports_and_bundles.ipynb",
        title="Schema Service Exports and Bundles",
        description=(
            "Integrator notebook for TargetInput, pinch_analysis_service, "
            "exports, workspace variant views, and bundles."
        ),
        topics=("integrator", "schemas", "exports", "workspace"),
    ),
    "10_multiperiod_hpr_shared_design.ipynb": NotebookMetadata(
        name="10_multiperiod_hpr_shared_design.ipynb",
        title="Multiperiod HPR Shared Design",
        description=(
            "Focused notebook for opt-in weighted multi-period HPR design, "
            "HPR detail inspection, and weighted summary modes."
        ),
        topics=("multiperiod", "heat pump", "hpr", "weighted summary"),
    ),
}


def list_sample_cases() -> list[str]:
    """Return the packaged sample-case filenames."""
    return sorted(
        item.name
        for item in _SAMPLE_CASE_ROOT.iterdir()
        if item.is_file() and item.name.endswith(".json")
    )


def list_notebooks() -> list[str]:
    """Return the packaged notebook filenames."""
    return sorted(
        item.name
        for item in _NOTEBOOK_ROOT.iterdir()
        if item.is_file() and item.name.endswith(".ipynb")
    )


def sample_case_metadata(name: str | None = None):
    """Return metadata for one or all packaged sample cases."""
    if name is not None:
        _resolve_resource(name, list_sample_cases(), "sample case")
        return _SAMPLE_CASE_METADATA.get(
            name,
            SampleCaseMetadata(name=name, title=Path(name).stem, description=""),
        )
    return [
        _SAMPLE_CASE_METADATA.get(
            item,
            SampleCaseMetadata(name=item, title=Path(item).stem, description=""),
        )
        for item in list_sample_cases()
    ]


def notebook_metadata(name: str | None = None):
    """Return metadata for one or all packaged notebooks."""
    if name is not None:
        _resolve_resource(name, list_notebooks(), "notebook")
        return _NOTEBOOK_METADATA.get(
            name,
            NotebookMetadata(name=name, title=Path(name).stem, description=""),
        )
    return [
        _NOTEBOOK_METADATA.get(
            item,
            NotebookMetadata(name=item, title=Path(item).stem, description=""),
        )
        for item in list_notebooks()
    ]


def read_sample_case(name: str) -> str:
    """Return the text of a packaged sample case."""
    resolved = _resolve_resource(name, list_sample_cases(), "sample case")
    return _SAMPLE_CASE_ROOT.joinpath(resolved).read_text(encoding="utf-8")


def copy_sample_case(name: str, destination: str | Path) -> Path:
    """Copy a packaged sample case to ``destination``."""
    resolved = _resolve_resource(name, list_sample_cases(), "sample case")
    source = _SAMPLE_CASE_ROOT.joinpath(resolved)
    dest_path = Path(destination)
    if dest_path.is_dir():
        dest_path = dest_path / resolved
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path


def copy_notebook(name: str, destination: str | Path) -> Path:
    """Copy a packaged notebook to ``destination``."""
    resolved = _resolve_resource(name, list_notebooks(), "notebook")
    source = _NOTEBOOK_ROOT.joinpath(resolved)
    dest_path = Path(destination)
    if dest_path.is_dir():
        dest_path = dest_path / resolved
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_path


def _resolve_resource(name: str, available: list[str], resource_type: str) -> str:
    """Return a valid packaged resource name or raise a friendly error."""
    if name in available:
        return name
    available_text = ", ".join(available)
    raise FileNotFoundError(
        f"Unknown OpenPinch {resource_type} {name!r}. "
        f"Available {resource_type}s: {available_text}."
    )


__all__ = [
    "NotebookMetadata",
    "SampleCaseMetadata",
    "copy_notebook",
    "copy_sample_case",
    "list_notebooks",
    "list_sample_cases",
    "notebook_metadata",
    "read_sample_case",
    "sample_case_metadata",
]
