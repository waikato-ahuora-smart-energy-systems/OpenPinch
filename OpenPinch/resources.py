"""Helpers for accessing packaged OpenPinch sample cases and notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

_SAMPLE_CASE_ROOT = files("OpenPinch.tutorials.sample_cases")
_NOTEBOOK_ROOT = files("OpenPinch.tutorials.notebooks")
_HPR_PERFORMANCE_MAP_CONTRACT_ROOT = files(
    "OpenPinch.contracts.resources.hpr_performance_map"
)
_HPR_PERFORMANCE_MAP_CONTRACT_RESOURCES = (
    "heat-pump-1.0.json",
    "refrigeration-1.0.json",
    "schema-1.0.json",
)


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


def list_hpr_performance_map_contract_resources() -> list[str]:
    """Return the closed HPR performance-map contract resource catalog."""
    return list(_HPR_PERFORMANCE_MAP_CONTRACT_RESOURCES)


def read_hpr_performance_map_contract_resource(name: str) -> str:
    """Return one packaged HPR contract schema or golden fixture as text."""
    resolved = _resolve_resource(
        name,
        list_hpr_performance_map_contract_resources(),
        "HPR performance-map contract resource",
    )
    return _HPR_PERFORMANCE_MAP_CONTRACT_ROOT.joinpath(resolved).read_text(
        encoding="utf-8"
    )


def load_hpr_performance_map_contract_resource(name: str) -> dict[str, Any]:
    """Return one packaged HPR contract resource as detached plain JSON data."""
    value = json.loads(read_hpr_performance_map_contract_resource(name))
    if not isinstance(value, dict):
        raise ValueError(f"HPR performance-map resource {name!r} is not an object")
    return value


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


def _notebook_metadata(name: str) -> NotebookMetadata:
    """Read one notebook's ``metadata["openpinch"]`` block."""
    notebook = json.loads(_NOTEBOOK_ROOT.joinpath(name).read_text(encoding="utf-8"))
    meta = notebook.get("metadata", {}).get("openpinch", {})
    title = meta.get("title")
    if title is None:
        return NotebookMetadata(name=name, title=Path(name).stem, description="")
    return NotebookMetadata(
        name=name,
        title=title,
        description=f"Process-engineer tutorial for {title.lower()}.",
        topics=tuple(meta.get("topics", ())),
    )


def notebook_metadata(name: str | None = None):
    """Return metadata for one or all packaged notebooks.

    Each notebook's ``metadata["openpinch"]`` block is the single source for
    its title, topics, level, expected runtime, execution profile and extras.
    """
    if name is not None:
        return _notebook_metadata(_resolve_resource(name, list_notebooks(), "notebook"))
    return [_notebook_metadata(item) for item in list_notebooks()]


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
    "list_hpr_performance_map_contract_resources",
    "list_notebooks",
    "list_sample_cases",
    "load_hpr_performance_map_contract_resource",
    "notebook_metadata",
    "read_hpr_performance_map_contract_resource",
    "read_sample_case",
    "sample_case_metadata",
]
