"""High-level convenience wrapper around the OpenPinch targeting service.

``PinchProblem`` provides a script-friendly interface for loading validated
inputs from JSON/Excel/CSV sources, running analysis, exporting results, and
launching the Streamlit dashboard.
"""

import json
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
from pydantic import ValidationError

from ..lib.enums import ST
from ..lib.schemas.io import TargetInput, TargetOutput
from ..lib.schemas.targets import BaseTargetModel
from ..services import data_preprocessing_service
from .accessors.target_accessor import _TargetAccessorDescriptor
from .accessors.plot_accessor import _PlotAccessorDescriptor
from ..services.input_data_processing.data_preparation import (
    _validate_zone_tree_structure,
)
from ..streamlit_webviewer.web_graphing import (
    render_streamlit_dashboard as _render_streamlit_dashboard,
)
from ..utils.csv_to_json import get_problem_from_csv
from ..utils.export import (
    build_summary_dataframe,
    export_target_summary_to_excel_with_units,
)
from ..utils.input_validation import validate_stream_data, validate_utility_data
from ..utils.miscellaneous import get_value
from ..utils.multiscale_targeting import (
    extract_results,
    get_targets_for_zone_and_sub_zones,
)
from ..utils.wkbook_to_json import get_problem_from_excel
from .zone import Zone

JsonDict = Dict[str, Any]
PathLike = Union[str, Path]
ZoneService = Callable[["Zone"], "Zone"]


@dataclass
class PinchProblem:
    """Typed orchestrator for loading input data, running targeting, and exporting results.

    Supports the following input formats out of the box:

    - JSON problem files
    - Excel problem files (use Excel_Version/Data_input_template.xlsx)
    - CSV bundles: either a directory containing ``streams.csv`` and ``utilities.csv``
      or an explicit ``(streams_csv_path, utilities_csv_path)`` tuple

    The object caches both the raw problem definition and the solved analysis
    result so subsequent export, dashboard, or inspection steps do not need to
    rerun the numerical workflow unless the inputs change.
    """

    problem_filepath: Optional[Path] = None
    results_dir: Optional[Path] = None
    problem_data: Optional[JsonDict] = None
    _project_name: str = "Site"
    _results: Optional[TargetOutput] = None
    _validated_data: TargetInput = None
    _master_zone: Optional["Zone"] = None
    _input_source_kind: str = "unknown"
    _validation_context: Optional[dict[str, list[dict[str, Any]]]] = None
    plot = _PlotAccessorDescriptor()
    target = _TargetAccessorDescriptor()

    def __init__(
        self,
        source: Union[
            TargetInput, JsonDict, PathLike, Tuple[PathLike, PathLike]
        ] = None,
        *,
        project_name: Optional[str] = "Site",
    ) -> None:
        """Initialise the orchestrator and optionally load and solve a case.

        Parameters
        ----------
        source:
            In-memory :class:`TargetInput`, mapping payload, problem filepath, or
            ``(streams.csv, utilities.csv)`` tuple to pass through :meth:`load`.
        project_name:
            Root zone / project label used in generated results.
        """
        self._project_name = project_name
        self._input_source_kind = "unknown"
        self._validation_context = None
        self._problem_filepath = None
        self._problem_data = None
        self._results = None
        self.results_dir = None

        if source is not None:
            self.load(source=source)

    # ----------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------

    def load(
        self,
        source: Union[
            TargetInput, JsonDict, PathLike, Tuple[PathLike, PathLike]
        ] = None,
    ) -> Zone:
        """Load input data from one of:

        - JSON file path (``*.json``)
        - Excel file path (``*.xlsx``, ``*.xls``, ``*.xlsb``, ``*.xlsm``)
        - CSV bundle: either a directory containing ``streams.csv`` and
          ``utilities.csv`` or a ``(streams_csv_path, utilities_csv_path)`` tuple

        Returns
        -------
        Zone
            Prepared in-memory zone hierarchy ready for targeting.
        """
        if source is None:
            if self.problem_filepath is None:
                return None
            source = Path(self.problem_filepath)

        if not isinstance(source, (TargetInput, dict)):
            try:
                source = TargetInput.model_validate(source)
            except ValidationError:
                pass

        if isinstance(source, TargetInput):
            self._problem_data = source
            self._input_source_kind = "target_input"
            self._validation_context = _build_validation_context(
                source.model_dump() if hasattr(source, "model_dump") else source,
                source_kind=self._input_source_kind,
            )

        elif isinstance(source, dict):
            self._problem_data = source
            if isinstance(self._problem_data.get("streams"), list):
                self._problem_data["streams"] = validate_stream_data(
                    self._problem_data["streams"]
                )
            if isinstance(self._problem_data.get("utilities"), list):
                self._problem_data["utilities"] = validate_utility_data(
                    self._problem_data["utilities"]
                )
            self._input_source_kind = "target_input"
            self._validation_context = _build_validation_context(
                self._problem_data,
                source_kind=self._input_source_kind,
            )

        elif isinstance(source, tuple) and len(source) == 2:
            # CSV tuple form
            streams_csv, utilities_csv = map(Path, source)
            self._problem_data = get_problem_from_csv(
                streams_csv, utilities_csv, output_json=None
            )
            self._input_source_kind = "csv"
            self._validation_context = _build_validation_context(
                self._problem_data,
                source_kind=self._input_source_kind,
            )
            self._problem_filepath = None  # Not a single-file source

        else:
            src_path = Path(source)
            if self._project_name == "Untitled":
                self._project_name = src_path.stem

            # 1. JSON
            if src_path.suffix.lower() == ".json":
                try:
                    with src_path.open("r", encoding="utf-8") as f:
                        self._problem_data = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    raise ValueError(
                        f"Failed to parse JSON from {src_path}: {e}"
                    ) from e
                if isinstance(self._problem_data, dict) and isinstance(
                    self._problem_data.get("streams"), list
                ):
                    self._problem_data["streams"] = validate_stream_data(
                        self._problem_data["streams"]
                    )
                if isinstance(self._problem_data, dict) and isinstance(
                    self._problem_data.get("utilities"), list
                ):
                    self._problem_data["utilities"] = validate_utility_data(
                        self._problem_data["utilities"]
                    )
                self._input_source_kind = "json"
                self._validation_context = _build_validation_context(
                    self._problem_data,
                    source_kind=self._input_source_kind,
                )
                self._problem_filepath = src_path

            # 2. Excel
            elif src_path.suffix.lower() in {".xlsx", ".xls", ".xlsb", ".xlsm"}:
                # Reuse your existing Excel reader; writes options, streams, utilities
                self._problem_data = get_problem_from_excel(src_path, output_json=None)
                self._input_source_kind = "excel"
                self._validation_context = _build_validation_context(
                    self._problem_data,
                    source_kind=self._input_source_kind,
                )
                self._problem_filepath = src_path

            # 3. CSV bundle via directory lookup
            elif src_path.is_dir():
                streams_csv = src_path / "streams.csv"
                utilities_csv = src_path / "utilities.csv"
                if not streams_csv.exists() or not utilities_csv.exists():
                    raise FileNotFoundError(
                        f"CSV directory '{src_path}' must contain 'streams.csv' and 'utilities.csv'."
                    )
                self._problem_data = get_problem_from_csv(
                    streams_csv, utilities_csv, output_json=None
                )
                self._input_source_kind = "csv"
                self._validation_context = _build_validation_context(
                    self._problem_data,
                    source_kind=self._input_source_kind,
                )
                self._problem_filepath = src_path

            else:
                raise ValueError(
                    f"Unrecognized source '{src_path}'. Provide a JSON/Excel file, "
                    f"a directory with 'streams.csv' and 'utilities.csv', or a (streams, utilities) tuple."
                )
        self._validated_data = self.validate()
        self._master_zone = self._data_preprocessing()
        return self._master_zone

    def _run_targeting_for_zone_and_subzones(
        self,
        zone: Zone = None,
        direct_service_func: Optional[ZoneService] = None,
        indirect_service_func: Optional[ZoneService] = None,
    ) -> TargetOutput:
        """Run the targeting analysis against the loaded input and cache the result."""
        if self._master_zone is None:
            if self._problem_data is None:
                raise RuntimeError("No input loaded. Call load(...) first.")
            else:
                self.load(self._problem_data)
        if not isinstance(zone, Zone):
            zone = self._master_zone
        # Perform advanced targeting analysis on the master zone and all subzones
        get_targets_for_zone_and_sub_zones(
            zone=zone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
        )
        # Extract the core results from the master zone
        return_data = extract_results(zone)
        # Validate response data
        self._results = TargetOutput.model_validate(return_data)
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
                f"Targeting did not produce target {target_id!r} for zone {zone.name!r}."
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
        results = self.target()
        if detailed:
            return build_summary_dataframe(results.targets)

        rows = []
        for target in results.targets:
            rows.append(
                {
                    "Target": target.name,
                    "Hot Utility Target": _maybe_get_value(target.Qh),
                    "Cold Utility Target": _maybe_get_value(target.Qc),
                    "Heat Recovery": _maybe_get_value(target.Qr),
                    "Hot Pinch": _maybe_get_value(target.temp_pinch.hot_temp),
                    "Cold Pinch": _maybe_get_value(target.temp_pinch.cold_temp),
                    "Hot Utilities": ", ".join(
                        _format_utility(utility.name, utility.heat_flow)
                        for utility in target.hot_utilities
                    ),
                    "Cold Utilities": ", ".join(
                        _format_utility(utility.name, utility.heat_flow)
                        for utility in target.cold_utilities
                    ),
                }
            )
        return pd.DataFrame(rows)

    def export_to_Excel(self, results_dir: Optional[PathLike] = None) -> Path:
        """Export the solved target summary and problem tables to an Excel file."""
        if results_dir is not None:
            self.results_dir = Path(results_dir)

        if self.results_dir is None:
            raise ValueError("No results_dir set. Provide a path to export results.")

        # Ensure results exist
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
        self._project_name = value
        if isinstance(self._master_zone, Zone):
            self._master_zone.name = value

    # ----------------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------------

    @classmethod
    def from_json(cls, data: JsonDict) -> "PinchProblem":
        """Build from an in-memory mapping and apply the normal input cleaners."""
        obj = cls()
        obj._problem_data = data
        if isinstance(obj._problem_data, dict) and isinstance(
            obj._problem_data.get("streams"), list
        ):
            obj._problem_data["streams"] = validate_stream_data(
                obj._problem_data["streams"]
            )
        if isinstance(obj._problem_data, dict) and isinstance(
            obj._problem_data.get("utilities"), list
        ):
            obj._problem_data["utilities"] = validate_utility_data(
                obj._problem_data["utilities"]
            )
        obj._input_source_kind = "in_memory"
        obj._validation_context = _build_validation_context(
            obj._problem_data,
            source_kind=obj._input_source_kind,
        )
        return obj

    def to_problem_json(self) -> JsonDict:
        """Return the currently loaded problem payload in its canonical mapping form."""
        if self._problem_data is None:
            raise RuntimeError(
                "No problem_data available. Did you call load(...) or from_json(...)?"
            )
        return self._problem_data

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

        self._problem_data = payload
        self._validation_context = _build_validation_context(
            self._problem_data,
            source_kind=self._input_source_kind,
        )
        self._validated_data = self.validate()
        self._master_zone = self._data_preprocessing()
        self._results = None
        return self._master_zone

    def _canonical_problem_payload(self) -> JsonDict:
        """Return a canonical mutable payload including an explicit zone tree."""
        validated = self.validate()
        payload = deepcopy(validated.model_dump(mode="python"))
        stream_models = [stream.model_copy(deep=True) for stream in validated.streams]
        zone_tree = (
            validated.zone_tree.model_copy(deep=True)
            if validated.zone_tree is not None
            else None
        )
        canonical_zone_tree = _validate_zone_tree_structure(
            zone_tree,
            stream_models,
            self.project_name,
        )
        payload["streams"] = [
            stream.model_dump(mode="python") for stream in stream_models
        ]
        payload["zone_tree"] = canonical_zone_tree.model_dump(mode="python")
        return payload

    def __repr__(self) -> str:
        """Machine-readable summary capturing source, export target, and result cache state."""
        src = (
            str(self._problem_filepath)
            if self._problem_filepath is not None
            else "<in-memory or CSV tuple>"
        )
        tgt = str(self.results_dir) if self.results_dir is not None else "<unset>"
        has_results = "yes" if self._results is not None else "no"
        return f"PinchProblem(source={src}, export={tgt}, results={has_results})"

    # ----------------------------------------------------------------------------
    # Visualisation helpers
    # ----------------------------------------------------------------------------

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
            graphs = getattr(self._results, "graphs", None)
            if graphs:
                payload = {
                    key: value.model_dump()
                    if hasattr(value, "model_dump")
                    else dict(value)
                    for key, value in graphs.items()
                }

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


def _maybe_get_value(value):
    if value is None:
        return None
    return get_value(value)


def _format_utility(name: str, heat_flow) -> str:
    value = _maybe_get_value(heat_flow)
    if value is None:
        return f"{name}: n/a"
    return f"{name}: {value:.2f}"


def _locate_summary_row(
    frame: pd.DataFrame,
    *,
    target_name: Optional[str] = None,
) -> pd.Series:
    if "Target" not in frame.columns or frame.empty:
        raise ValueError("Summary frame is empty or missing the 'Target' column.")

    targets = frame["Target"].astype(str)
    if target_name is not None:
        exact_match = frame.loc[targets == str(target_name)]
        if not exact_match.empty:
            return exact_match.iloc[0]

        suffix = str(target_name).split("/", 1)[-1]
        suffix_match = frame.loc[targets.str.endswith(suffix)]
        if not suffix_match.empty:
            return suffix_match.iloc[0]

        raise KeyError(f"Target {target_name!r} was not found in the summary output.")

    preferred_targets = [
        "Plant/Direct Integration",
    ]
    for preferred in preferred_targets:
        preferred_match = frame.loc[targets == preferred]
        if not preferred_match.empty:
            return preferred_match.iloc[0]

    direct_match = frame.loc[targets.str.endswith("/Direct Integration")]
    if not direct_match.empty:
        return direct_match.iloc[0]
    return frame.iloc[0]


def _find_zone_tree_node(
    zone_tree: dict[str, Any],
    zone_name: str,
) -> dict[str, Any]:
    root_name = str(zone_tree.get("name") or "")
    path_parts = [part.strip() for part in str(zone_name).split("/") if part.strip()]
    if not path_parts:
        raise ValueError("zone_name must identify a zone in the zone_tree.")

    if path_parts[0] == root_name:
        path_parts = path_parts[1:]

    node = zone_tree
    if not path_parts:
        return node

    for part in path_parts:
        children = node.get("children") or []
        next_node = next(
            (child for child in children if str(child.get("name")) == part),
            None,
        )
        if next_node is None:
            raise ValueError(f"Zone {zone_name!r} was not found in the zone_tree.")
        node = next_node
    return node


def _build_validation_context(
    problem_data: Any,
    *,
    source_kind: str,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(problem_data, dict):
        return {}

    context: dict[str, list[dict[str, Any]]] = {}
    for section in ("streams", "utilities"):
        records = problem_data.get(section)
        if not isinstance(records, list):
            continue
        context[section] = [
            _build_record_context(section, index, record, source_kind=source_kind)
            for index, record in enumerate(records)
        ]
    return context


def _build_record_context(
    section: str,
    index: int,
    record: Any,
    *,
    source_kind: str,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "index": index,
        "section": section,
    }
    if isinstance(record, dict):
        details["name"] = record.get("name")
        details["zone"] = record.get("zone")

    if source_kind in {"excel", "csv"}:
        details["sheet"] = "Stream Data" if section == "streams" else "Utility Data"
        details["row"] = index + 3
    elif source_kind == "json":
        details["entry"] = index + 1
    return details


def _format_schema_validation_error(
    exc: ValidationError,
    *,
    problem_data: Any,
    context: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        f"Input validation failed with {len(exc.errors())} issue(s):",
    ]
    for error in exc.errors():
        lines.append(
            _format_single_validation_error(
                error,
                problem_data=problem_data,
                context=context,
            )
        )
    return "\n".join(lines)


def _format_single_validation_error(
    error: dict[str, Any],
    *,
    problem_data: Any,
    context: dict[str, list[dict[str, Any]]],
) -> str:
    loc = tuple(error.get("loc", ()))
    message = error.get("msg", "Invalid value.")
    section = loc[0] if loc else None
    record_index = loc[1] if len(loc) > 1 and isinstance(loc[1], int) else None
    field_path = ".".join(str(part) for part in loc[2:]) if len(loc) > 2 else ""

    prefix = "Input"
    if isinstance(section, str) and record_index is not None:
        record_context = _lookup_record_context(context, section, record_index)
        prefix = _describe_record(section, record_index, record_context)
    elif loc:
        prefix = f"Field '{'.'.join(str(part) for part in loc)}'"

    rendered = f"- {prefix}"
    if field_path:
        rendered += f": field '{field_path}'"
    rendered += f" - {message}"
    return rendered


def _lookup_record_context(
    context: dict[str, list[dict[str, Any]]],
    section: str,
    record_index: int,
) -> dict[str, Any]:
    records = context.get(section, [])
    if 0 <= record_index < len(records):
        return records[record_index]
    return {"index": record_index, "section": section}


def _describe_record(
    section: str, record_index: int, record_context: dict[str, Any]
) -> str:
    label = (
        "Stream"
        if section == "streams"
        else "Utility"
        if section == "utilities"
        else section
    )
    name = record_context.get("name")
    row = record_context.get("row")
    sheet = record_context.get("sheet")
    entry = record_context.get("entry")

    description = f"{label} {record_index + 1}"
    if name not in (None, ""):
        description += f" '{name}'"
    if sheet and row:
        description += f" ({sheet} row {row})"
    elif entry:
        description += f" (entry {entry})"
    return description


def _validate_problem_semantics(
    payload: TargetInput,
    *,
    context: dict[str, list[dict[str, Any]]],
) -> None:
    fatal_issues = []
    warnings_issues = []

    if len(payload.streams) == 0:
        fatal_issues.append("- At least one stream must be provided.")

    for index, stream in enumerate(payload.streams):
        if _maybe_get_value(stream.t_supply) == _maybe_get_value(stream.t_target):
            fatal_issues.append(
                _format_semantic_issue(
                    "streams",
                    index,
                    context,
                    "field 't_supply/t_target' - supply and target temperatures must differ.",
                )
            )

        value = _maybe_get_value(getattr(stream, "heat_flow"))
        if value is not None and value < 0:
            fatal_issues.append(
                _format_semantic_issue(
                    "streams",
                    index,
                    context,
                    f"field '{'heat_flow'}' - value must be non-negative.",
                )
            )

        value = _maybe_get_value(getattr(stream, "dt_cont"))
        if value is not None and value < 0:
            warnings_issues.append(
                _format_semantic_issue(
                    "streams",
                    index,
                    context,
                    f"Warning: field '{'dt_cont'}' - value should be non-negative.",
                )
            )

        value = _maybe_get_value(getattr(stream, "htc"))
        if value is not None and value <= 0:
            fatal_issues.append(
                _format_semantic_issue(
                    "streams",
                    index,
                    context,
                    f"field '{'htc'}' - value must be non-negative.",
                )
            )

    for index, utility in enumerate(payload.utilities):
        utility_type = str(utility.type)
        t_supply = _maybe_get_value(utility.t_supply)
        t_target = _maybe_get_value(utility.t_target)
        if (
            utility_type == ST.Hot.value
            and t_supply is not None
            and t_target is not None
            and t_supply < t_target
        ):
            warnings_issues.append(
                _format_semantic_issue(
                    "utilities",
                    index,
                    context,
                    "field 't_supply/t_target' - hot utilities should have t_supply >= t_target.",
                )
            )
        if (
            utility_type == ST.Cold.value
            and t_supply is not None
            and t_target is not None
            and t_supply > t_target
        ):
            warnings_issues.append(
                _format_semantic_issue(
                    "utilities",
                    index,
                    context,
                    "field 't_supply/t_target' - cold utilities should have t_supply <= t_target.",
                )
            )

        for field_name in ("dt_cont", "price", "heat_flow"):
            value = _maybe_get_value(getattr(utility, field_name))
            if value is not None and value < 0:
                warnings_issues.append(
                    _format_semantic_issue(
                        "utilities",
                        index,
                        context,
                        f"field '{field_name}' - value should be non-negative.",
                    )
                )

        value = _maybe_get_value(getattr(utility, "htc"))
        if value is not None and value <= 0:
            fatal_issues.append(
                _format_semantic_issue(
                    "utilities",
                    index,
                    context,
                    f"field '{'htc'}' - value must be non-negative.",
                )
            )

    if fatal_issues:
        raise ValueError(
            "Input validation failed with "
            f"{len(fatal_issues)} issue(s):\n" + "\n".join(fatal_issues)
        )

    if warnings_issues:
        warnings.warn(
            "Input validation reported "
            f"{len(warnings_issues)} warning(s):\n" + "\n".join(warnings_issues),
            UserWarning,
            stacklevel=2,
        )


def _format_semantic_issue(
    section: str,
    record_index: int,
    context: dict[str, list[dict[str, Any]]],
    message: str,
) -> str:
    record_context = _lookup_record_context(context, section, record_index)
    return f"- {_describe_record(section, record_index, record_context)}: {message}"
