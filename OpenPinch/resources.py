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
    "process_mvr.json": SampleCaseMetadata(
        name="process_mvr.json",
        title="Process MVR",
        description="Pressure-defined gas stream for direct process MVR studies.",
        topics=("MVR", "components", "heat pump"),
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


_NOTEBOOK_CATALOG = (
    (
        "01_first_solve_and_core_curves.ipynb",
        "First Solve and Core Curves",
        ("quickstart", "plots"),
    ),
    (
        "02_focused_direct_and_total_site.ipynb",
        "Focused Direct and Total Site",
        ("direct", "total site"),
    ),
    ("03_multisegment_streams.ipynb", "Multi-Segment Streams", ("streams", "segments")),
    (
        "04_workspace_cases_and_scenarios.ipynb",
        "Workspace Cases and Scenarios",
        ("workspace", "cases"),
    ),
    (
        "05_workspace_persistence.ipynb",
        "Workspace Data and Persistence",
        ("workspace", "persistence"),
    ),
    (
        "06_multiperiod_heat_integration.ipynb",
        "Multiperiod Heat Integration",
        ("multiperiod", "targets"),
    ),
    (
        "07_area_cost_and_exergy.ipynb",
        "Area Cost and Exergy",
        ("area", "cost", "exergy"),
    ),
    (
        "08_carnot_heat_pump_and_refrigeration.ipynb",
        "Carnot Heat Pump and Refrigeration",
        ("heat pump", "refrigeration"),
    ),
    (
        "09_vapour_compression_and_brayton.ipynb",
        "Vapour Compression and Brayton HPR",
        ("heat pump", "Brayton"),
    ),
    (
        "10_multiperiod_heat_pumps.ipynb",
        "Multiperiod Heat Pumps",
        ("multiperiod", "heat pump"),
    ),
    (
        "11_process_mvr_and_cascade.ipynb",
        "Process MVR and VC Cascade",
        ("MVR", "components"),
    ),
    ("12_cogeneration.ipynb", "Cogeneration", ("cogeneration",)),
    (
        "13_multiperiod_cogeneration.ipynb",
        "Multiperiod Cogeneration",
        ("multiperiod", "cogeneration"),
    ),
    ("14_energy_transfer.ipynb", "Energy Transfer", ("energy transfer", "plots")),
    (
        "15_hen_synthesis_and_selection.ipynb",
        "HEN Synthesis and Selection",
        ("HEN", "synthesis"),
    ),
    (
        "16_advanced_hen_methods.ipynb",
        "Advanced HEN Methods",
        ("HEN", "advanced design"),
    ),
    (
        "17_multiperiod_hen_synthesis.ipynb",
        "Multiperiod HEN Synthesis",
        ("multiperiod", "HEN"),
    ),
    (
        "18_results_plots_reports_exports.ipynb",
        "Results, Plots, Reports, and Exports",
        ("results", "exports"),
    ),
)

_NOTEBOOK_METADATA: dict[str, NotebookMetadata] = {
    name: NotebookMetadata(
        name=name,
        title=title,
        description=f"Process-engineer tutorial for {title.lower()}.",
        topics=topics,
    )
    for name, title, topics in _NOTEBOOK_CATALOG
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
