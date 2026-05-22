"""High-level convenience wrapper around the OpenPinch targeting service."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from pydantic import ValidationError

from ..lib.schemas.io import TargetInput, TargetOutput
from ..lib.schemas.targets import BaseTargetModel
from ..resources import list_sample_cases, read_sample_case
from ..services import data_preprocessing_service
from ..services.input_data_processing._canonicalization import canonical_problem_payload
from ..streamlit_webviewer.web_graphing import (
    render_streamlit_dashboard as _render_streamlit_dashboard,
)
from ..utils.csv_to_json import get_problem_from_csv
from ..utils.export import (
    build_summary_dataframe,
    export_target_summary_to_excel_with_units,
)
from ..utils.multiscale_targeting import (
    extract_results,
    get_targets_for_zone_and_sub_zones,
)
from ..utils.wkbook_to_json import get_problem_from_excel
from ._problem.loading import (
    JsonDict,
    LoadedProblemSource,
    PathLike,
    ProblemSourceAdapters,
    load_problem_source,
    prepare_in_memory_problem_source,
)
from ._problem.loading import (
    find_zone_tree_node as _find_zone_tree_node,
)
from ._problem.output import (
    build_graph_payload,
    build_problem_summary_frame,
)
from ._problem.output import (
    locate_summary_row as _locate_summary_row,
)
from ._problem.validation import (
    _validate_problem_semantics,
)
from ._problem.validation import (
    format_schema_validation_error as _format_schema_validation_error,
)
from .accessors.plot_accessor import _PlotAccessorDescriptor
from .accessors.target_accessor import _TargetAccessorDescriptor
from .zone import Zone

ZoneService = Callable[["Zone"], "Zone"]


class PinchProblem:
    """Typed orchestrator for loading input data and running targeting."""

    results_dir: Optional[Path]
    _problem_filepath: Optional[Path]
    _problem_data: Optional[JsonDict | TargetInput]
    _project_name: str
    _results: Optional[TargetOutput]
    _validated_data: Optional[TargetInput]
    _master_zone: Optional["Zone"]
    _input_source_kind: str
    _validation_context: Optional[dict[str, list[dict[str, Any]]]]
    plot = _PlotAccessorDescriptor()
    target = _TargetAccessorDescriptor()

    def __init__(
        self,
        source: (
            TargetInput
            | JsonDict
            | PathLike
            | tuple[PathLike, PathLike]
            | None
        ) = None,
        *,
        project_name: Optional[str] = "Site",
    ) -> None:
        self._project_name = project_name
        self._input_source_kind = "unknown"
        self._validation_context = None
        self._problem_filepath = None
        self._problem_data = None
        self._results = None
        self._validated_data = None
        self._master_zone = None
        self.results_dir = None

        if source is not None:
            self.load(source=source)

    def load(
        self,
        source: (
            TargetInput
            | JsonDict
            | PathLike
            | tuple[PathLike, PathLike]
            | None
        ) = None,
    ) -> Optional[Zone]:
        """Load input data from JSON, Excel, CSV, or an in-memory payload."""
        if source is None:
            if self.problem_filepath is None:
                return None
            source = Path(self.problem_filepath)

        loaded_source = load_problem_source(
            source,
            current_project_name=self._project_name,
            adapters=self._problem_source_adapters(),
        )
        self._apply_loaded_source(loaded_source)
        return self._rebuild_problem_state()

    def _run_targeting_for_zone_and_subzones(
        self,
        zone: Optional[Zone] = None,
        direct_service_func: Optional[ZoneService] = None,
        indirect_service_func: Optional[ZoneService] = None,
    ) -> TargetOutput:
        """Run the targeting analysis against the loaded input and cache the result."""
        if self._master_zone is None:
            if self._problem_data is None:
                raise RuntimeError("No input loaded. Call load(...) first.")
            self.load(self._problem_data)
        if not isinstance(zone, Zone):
            zone = self._master_zone
        get_targets_for_zone_and_sub_zones(
            zone=zone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
        )
        self._results = TargetOutput.model_validate(extract_results(zone))
        return self._results

    def _execute_targeting(
        self,
        *,
        target_id: str,
        application_zone: Optional[str | Zone],
        options: Optional[dict[str, Any]],
        include_subzones: bool,
        direct_service_func: Optional[ZoneService] = None,
        indirect_service_func: Optional[ZoneService] = None,
    ) -> BaseTargetModel:
        zone = self._resolve_target_zone(application_zone)
        if include_subzones:
            self._run_targeting_for_zone_and_subzones(
                zone=zone,
                direct_service_func=direct_service_func,
                indirect_service_func=indirect_service_func,
            )
        else:
            if direct_service_func is not None:
                direct_service_func(zone, options)
            if indirect_service_func is not None:
                indirect_service_func(zone, options)
            self._refresh_results_from_master_zone()

        try:
            return zone.targets[target_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Targeting did not produce target {target_id!r} "
                f"for zone {zone.name!r}."
            ) from exc

    def _resolve_target_zone(self, application_zone: Optional[str] = None) -> "Zone":
        if self._master_zone is None:
            raise RuntimeError("Load source first before targeting.")
        if isinstance(application_zone, Zone):
            return application_zone
        if application_zone is None:
            return self._master_zone
        return self._master_zone.get_subzone(application_zone)

    def validate(self) -> TargetInput:
        """Validate the currently loaded problem data without running targeting."""
        if self._problem_data is None:
            raise RuntimeError("No input loaded. Call load(...) first.")
        try:
            input_data = TargetInput.model_validate(self._problem_data)
        except ValidationError as exc:
            raise ValueError(
                _format_schema_validation_error(
                    exc,
                    problem_data=self._problem_data,
                    context=self._validation_context or {},
                )
            ) from exc

        _validate_problem_semantics(
            input_data,
            context=self._validation_context or {},
        )
        return input_data

    def summary_frame(self, *, detailed: bool = False) -> pd.DataFrame:
        """Return the solved target summary as a pandas DataFrame."""
        results = self._results if self._results is not None else self.target()
        if detailed:
            return build_summary_dataframe(results.targets)
        return build_problem_summary_frame(results, detailed=False)

    def export_to_Excel(self, results_dir: Optional[PathLike] = None) -> Path:
        """Export the solved target summary and problem tables to an Excel file."""
        if results_dir is not None:
            self.results_dir = Path(results_dir)
        if self.results_dir is None:
            raise ValueError("No results_dir set. Provide a path to export results.")
        if self._results is None:
            self.target()

        output_path = export_target_summary_to_excel_with_units(
            target_response=self._results,
            master_zone=self._master_zone,
            out_dir=self.results_dir,
        )
        return Path(output_path)

    def export_excel(self, results_dir: Optional[PathLike] = None) -> Path:
        """Alias for :meth:`export_to_Excel` with a conventional snake_case name."""
        return self.export_to_Excel(results_dir)

    def compare_to(
        self,
        other_problem: "PinchProblem",
        *,
        target_name: Optional[str] = None,
        base_label: str = "Base case",
        other_label: str = "Scenario",
    ) -> pd.DataFrame:
        """Compare the compact summaries of two solved problems."""
        base_frame = self.summary_frame()
        other_frame = other_problem.summary_frame()

        base_row = _locate_summary_row(base_frame, target_name=target_name)
        other_row = _locate_summary_row(
            other_frame,
            target_name=target_name or str(base_row["Target"]),
        )

        columns = [
            "Hot Utility Target",
            "Cold Utility Target",
            "Heat Recovery",
            "Hot Pinch",
            "Cold Pinch",
        ]
        comparison = pd.DataFrame(
            [
                pd.Series({col: base_row.get(col) for col in columns}, name=base_label),
                pd.Series(
                    {col: other_row.get(col) for col in columns},
                    name=other_label,
                ),
            ]
        )
        comparison.loc["Change"] = (
            comparison.loc[other_label] - comparison.loc[base_label]
        )
        comparison.insert(0, "Target", str(base_row["Target"]))
        comparison.loc["Change", "Target"] = str(base_row["Target"])
        return comparison

    def _data_preprocessing(self) -> "Zone":
        if isinstance(self._validated_data, TargetInput) and isinstance(
            self._project_name, str
        ):
            return data_preprocessing_service(
                input_data=self._validated_data,
                project_name=self._project_name,
            )
        raise ValueError("No validated data load. Try ``load(source)``.")

    @property
    def problem_filepath(self) -> Optional[Path]:
        """Return the filepath of the problem that was loaded or supplied."""
        return self._problem_filepath

    @property
    def problem_data(self) -> Optional[TargetInput | JsonDict]:
        """Return the raw problem definition that was loaded or supplied."""
        return self._problem_data

    @property
    def results(self) -> Optional[TargetOutput]:
        """Return the cached targeting results, if targeting has been executed."""
        return self._results

    @property
    def master_zone(self) -> Optional["Zone"]:
        """Return the analysed Zone hierarchy after a successful ``target()`` run."""
        return self._master_zone

    @property
    def project_name(self) -> str:
        """Return the project label used for the root zone and exports."""
        return self._project_name

    @project_name.setter
    def project_name(self, value: str):
        """Update the root project label and mirror it onto the loaded root zone."""
        self._project_name = value
        if isinstance(self._master_zone, Zone):
            self._master_zone.name = value

    @classmethod
    def from_json(cls, data: JsonDict) -> "PinchProblem":
        """Build from an in-memory mapping and apply the normal input cleaners."""
        obj = cls()
        obj._apply_loaded_source(
            prepare_in_memory_problem_source(data, source_kind="in_memory")
        )
        return obj

    def to_problem_json(self, *, canonical: bool = False) -> JsonDict:
        """Return the currently loaded problem payload."""
        if self._problem_data is None:
            raise RuntimeError(
                "No problem_data available. Did you call load(...) or from_json(...)?"
            )
        if canonical:
            return self._canonical_problem_payload()
        return self._problem_data

    def canonical_problem_json(self) -> JsonDict:
        """Return a canonical mutable payload including an explicit zone tree."""
        return self.to_problem_json(canonical=True)

    def set_dt_cont_multiplier(
        self,
        value: float,
        *,
        zone_name: Optional[str] = None,
    ) -> Zone:
        """Update one zone-tree multiplier and rebuild the prepared analysis state."""
        resolved_value = float(value)
        if not math.isfinite(resolved_value) or resolved_value < 0.0:
            raise ValueError("dt_cont_multiplier must be a finite non-negative value.")

        payload = self._canonical_problem_payload()
        zone_tree = payload.get("zone_tree")
        if not isinstance(zone_tree, dict):
            raise RuntimeError("No zone_tree is available to update.")

        target_zone = zone_name or str(zone_tree.get("name") or self.project_name)
        zone_node = _find_zone_tree_node(zone_tree, target_zone)
        zone_node["dt_cont_multiplier"] = resolved_value

        self._replace_problem_payload(payload)
        return self._master_zone

    def update_options(
        self,
        options: Dict[str, Any],
        *,
        replace: bool = False,
    ) -> Zone:
        """Update the problem options in-place and rebuild the analysis state."""
        if not isinstance(options, dict):
            raise TypeError("options must be provided as a dict.")

        payload = self.canonical_problem_json()
        current_options = payload.get("options") or {}
        payload["options"] = (
            deepcopy(options)
            if replace
            else {**deepcopy(current_options), **deepcopy(options)}
        )
        self._replace_problem_payload(payload)
        return self._master_zone

    def _canonical_problem_payload(self) -> JsonDict:
        """Return a canonical mutable payload including an explicit zone tree."""
        validated = self.validate()
        return canonical_problem_payload(validated, project_name=self.project_name)

    def __repr__(self) -> str:
        """Return a compact summary of the source and cached result state."""
        src = (
            str(self._problem_filepath)
            if self._problem_filepath is not None
            else "<in-memory or CSV tuple>"
        )
        export = str(self.results_dir) if self.results_dir is not None else "<unset>"
        has_results = "yes" if self._results is not None else "no"
        return f"PinchProblem(source={src}, export={export}, results={has_results})"

    def render_streamlit_dashboard(
        self,
        *,
        zone: Optional["Zone"] = None,
        graph_payload: Optional[Dict[str, Any]] = None,
        page_title: Optional[str] = "OpenPinch Dashboard",
        value_rounding: int = 2,
    ) -> None:
        """Launch the Streamlit dashboard for the analysed problem."""
        active_zone = zone or self._master_zone
        if active_zone is None:
            raise RuntimeError(
                "No analysed zone is available. Run target() before rendering."
            )

        payload = graph_payload
        if payload is None and self._results is not None:
            payload = build_graph_payload(self._results)

        _render_streamlit_dashboard(
            active_zone,
            graph_payload=payload,
            page_title=page_title,
            value_rounding=value_rounding,
        )

    def show_dashboard(
        self,
        *,
        zone: Optional["Zone"] = None,
        graph_payload: Optional[Dict[str, Any]] = None,
        page_title: Optional[str] = "OpenPinch Dashboard",
        value_rounding: int = 2,
    ) -> None:
        """Alias for :meth:`render_streamlit_dashboard`."""
        self.render_streamlit_dashboard(
            zone=zone,
            graph_payload=graph_payload,
            page_title=page_title,
            value_rounding=value_rounding,
        )

    def _refresh_results_from_master_zone(self) -> TargetOutput:
        if self._master_zone is None:
            raise RuntimeError("No analysed zone is available. Run target() first.")
        self._results = TargetOutput.model_validate(extract_results(self._master_zone))
        return self._results

    def _problem_source_adapters(self) -> ProblemSourceAdapters:
        """Build source adapters lazily so tests can monkeypatch module symbols."""
        return ProblemSourceAdapters(
            get_problem_from_excel=get_problem_from_excel,
            get_problem_from_csv=get_problem_from_csv,
            list_sample_cases=list_sample_cases,
            read_sample_case=read_sample_case,
        )

    def _apply_loaded_source(self, loaded_source: LoadedProblemSource) -> None:
        """Apply one normalized source bundle to this problem instance."""
        self._problem_data = loaded_source.payload
        self._input_source_kind = loaded_source.source_kind
        self._validation_context = loaded_source.validation_context
        self._problem_filepath = loaded_source.problem_filepath
        if loaded_source.project_name is not None:
            self._project_name = loaded_source.project_name

    def _rebuild_problem_state(self) -> Zone:
        """Revalidate, reconstruct the zone tree, and clear cached results."""
        self._validated_data = self.validate()
        self._master_zone = self._data_preprocessing()
        self._results = None
        return self._master_zone

    def _replace_problem_payload(self, payload: JsonDict) -> Zone:
        """Replace the current payload and rebuild prepared analysis state."""
        current_filepath = self._problem_filepath
        loaded_source = prepare_in_memory_problem_source(
            payload,
            source_kind=self._input_source_kind or "target_input",
        )
        self._apply_loaded_source(loaded_source)
        self._problem_filepath = current_filepath
        return self._rebuild_problem_state()
