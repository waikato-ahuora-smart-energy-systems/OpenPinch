"""High-level convenience wrapper around the OpenPinch targeting service.

``PinchProblem`` provides a script-friendly interface for loading validated
inputs from JSON/Excel/CSV sources, running analysis, exporting results, and
launching the Streamlit dashboard.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import pandas as pd
from pydantic import ValidationError

from ..services.common.graph_data import get_output_graph_data
from ..lib.enums import GT
from ..lib.schema import (
    HeatPumpIntegrationComparison,
    HeatPumpIntegrationScenario,
    TargetInput,
    TargetOutput,
)
from ..utils.csv_to_json import get_problem_from_csv
from ..utils.export import build_summary_dataframe
from ..utils.export import export_target_summary_to_excel_with_units
from ..utils.input_validation import validate_stream_data, validate_utility_data
from ..utils.miscellaneous import get_value
from ..utils.wkbook_to_json import get_problem_from_excel
from ..streamlit_webviewer.web_graphing import (
    _build_plotly_graph,
    render_streamlit_dashboard as _render_streamlit_dashboard,
)
from ..main import pinch_analysis_service

if TYPE_CHECKING:
    from .zone import Zone

JsonDict = Dict[str, Any]
PathLike = Union[str, Path]
GraphPayload = Dict[str, Dict[str, Any]]

_GRAPH_TYPE_ALIASES = {
    "cc": GT.CC.value,
    "composite": GT.CC.value,
    "composite curve": GT.CC.value,
    "composite curves": GT.CC.value,
    "scc": GT.SCC.value,
    "shifted": GT.SCC.value,
    "shifted composite": GT.SCC.value,
    "shifted composite curve": GT.SCC.value,
    "shifted composite curves": GT.SCC.value,
    "bcc": GT.BCC.value,
    "balanced": GT.BCC.value,
    "balanced composite": GT.BCC.value,
    "balanced composite curve": GT.BCC.value,
    "balanced composite curves": GT.BCC.value,
    "gcc": GT.GCC.value,
    "grand composite": GT.GCC.value,
    "grand composite curve": GT.GCC.value,
    "tsp": GT.TSP.value,
    "total site": GT.TSP.value,
    "total site profiles": GT.TSP.value,
    "sugcc": GT.SUGCC.value,
    "site utility grand composite curve": GT.SUGCC.value,
}


@dataclass
class HeatPumpIntegrationEvaluation:
    """Container returned by :meth:`PinchProblem.evaluate_heat_pump_integration`."""

    scenario: HeatPumpIntegrationScenario
    comparison: HeatPumpIntegrationComparison
    comparison_frame: pd.DataFrame
    integrated_problem: "PinchProblem"


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

    # Internal state
    problem_data: Optional[JsonDict] = None
    _project_name: str = "Untitled"
    _results: Optional[TargetOutput] = None
    _master_zone: Optional["Zone"] = None
    _input_source_kind: str = "unknown"
    _validation_context: Optional[dict[str, list[dict[str, Any]]]] = None


    def __init__(
        self,
        problem_filepath: Optional[PathLike] = None,
        results_dir: Optional[PathLike] = None,
        run: bool = False,
    ) -> None:
        """Initialise the orchestrator and optionally run the full targeting workflow.

        Parameters
        ----------
        problem_filepath:
            Path to a JSON or Excel problem definition handled by :meth:`load`.
        results_dir:
            Destination directory for exported Excel summaries. May be ``None`` if export
            is handled later.
        run:
            When ``True``, execute targeting immediately after construction and
            export results if ``results_dir`` is provided.
        """
        self._input_source_kind = "unknown"
        self._validation_context = None
        if problem_filepath is not None:
            self.load(source=Path(problem_filepath))
        else:
            self._problem_filepath = None
            self._problem_data = None

        self.results_dir = Path(results_dir) if results_dir is not None else None
        self._results = None
        self._master_zone = None

        if run:
            try:
                self.target()
            except Exception as exc:
                raise ValueError(
                    "Targeting analysis failed. Check input data format. "
                    "Report persistent bugs via GitHub issues."
                ) from exc

            if self.results_dir is not None:
                self.export_to_Excel(self.results_dir)

    # ----------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------


    def load(
        self,
        source: Union[TargetInput, PathLike, Tuple[PathLike, PathLike]],
    ) -> TargetInput | JsonDict:
        """Load input data from one of:

        - JSON file path (``*.json``)
        - Excel file path (``*.xlsx``, ``*.xls``, ``*.xlsb``, ``*.xlsm``)
        - CSV bundle: either a directory containing ``streams.csv`` and
          ``utilities.csv`` or a ``(streams_csv_path, utilities_csv_path)`` tuple

        Returns
        -------
        TargetInput or dict
            The loaded input structure ready for targeting.
        """
        if isinstance(source, TargetInput):
            self._problem_data = source
            self._input_source_kind = "target_input"
            self._validation_context = _build_validation_context(
                source.model_dump() if hasattr(source, "model_dump") else source,
                source_kind=self._input_source_kind,
            )
            return self._problem_data

        if isinstance(source, tuple) and len(source) == 2:
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
            return self._problem_data

        src_path = Path(source)
        self._project_name = src_path.stem

        # 1. JSON
        if src_path.suffix.lower() == ".json":
            try:
                with src_path.open("r", encoding="utf-8") as f:
                    self._problem_data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from {src_path}: {e}") from e
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
            return self._problem_data

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
            return self._problem_data

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
            return self._problem_data

        raise ValueError(
            f"Unrecognized source '{src_path}'. Provide a JSON/Excel file, "
            f"a directory with 'streams.csv' and 'utilities.csv', or a (streams, utilities) tuple."
        )


    def target(self) -> TargetOutput:
        """Run the targeting analysis against the loaded input and cache the result."""
        if self._problem_data is None:
            raise RuntimeError("No input loaded. Call load(...) first.")
        if self._results is None:
            self._results, self._master_zone = pinch_analysis_service(
                data=self._problem_data,
                project_name=self._project_name,
                is_return_full_results=True,
            )
        return self._results


    def run(self) -> TargetOutput:
        """Run the targeting workflow and return the cached result."""
        self.validate()
        return self.target()

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
        results = self.run()
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


    def graph_data(self) -> GraphPayload:
        """Return the serialized graph payload for the solved problem."""
        self.run()

        graphs = getattr(self._results, "graphs", None)
        if graphs:
            return {
                key: value.model_dump() if hasattr(value, "model_dump") else dict(value)
                for key, value in graphs.items()
            }

        if self._master_zone is None:
            raise RuntimeError("No analysed zone is available. Run target() first.")

        return get_output_graph_data(self._master_zone)

    def graph_catalog(self) -> pd.DataFrame:
        """Return a table describing the available graph outputs."""
        rows = []
        for zone_name, graph_set in self.graph_data().items():
            for index, graph in enumerate(graph_set.get("graphs", [])):
                rows.append(
                    {
                        "Zone": zone_name,
                        "Graph Type": graph.get("type"),
                        "Graph Name": graph.get("name", f"Graph {index + 1}"),
                        "Index": index,
                    }
                )
        return pd.DataFrame(rows)


    def plot(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
        index: int = 0,
    ):
        """Build a Plotly figure for one graph from the solved result set."""
        graph = self._select_graph(
            zone_name=zone_name,
            graph_type=graph_type,
            index=index,
        )
        return _build_plotly_graph(graph)


    def plot_composite_curve(
        self,
        *,
        zone_name: Optional[str] = None,
        variant: str = "composite",
    ):
        """Build a composite-curve Plotly figure for the selected zone."""
        selector = {
            "composite": GT.CC.value,
            "shifted": GT.SCC.value,
            "balanced": GT.BCC.value,
        }.get(variant.strip().lower())
        if selector is None:
            raise ValueError(
                "variant must be one of: 'composite', 'shifted', or 'balanced'."
            )
        return self.plot(zone_name=zone_name, graph_type=selector)


    def plot_grand_composite_curve(self, *, zone_name: Optional[str] = None):
        """Build the grand composite curve Plotly figure for the selected zone."""
        return self.plot(zone_name=zone_name, graph_type=GT.GCC.value)


    def export_graphs(
        self,
        output_dir: PathLike,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
    ) -> list[Path]:
        """Write selected graph outputs as standalone HTML files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        selected = self._select_graphs(zone_name=zone_name, graph_type=graph_type)
        written_paths = []
        for idx, (graph_zone_name, graph) in enumerate(selected, start=1):
            figure = _build_plotly_graph(graph)
            stem = _slugify(f"{graph_zone_name}_{graph.get('type', 'graph')}_{idx}")
            destination = output_path / f"{stem}.html"
            figure.write_html(destination)
            written_paths.append(destination)
        return written_paths


    def export_to_Excel(self, results_dir: Optional[PathLike] = None) -> Path:
        """Export the solved target summary and problem tables to an Excel file."""
        if results_dir is not None:
            self.results_dir = Path(results_dir)

        if self.results_dir is None:
            print(self._results)
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


    def build_heat_pump_integration_problem(
        self,
        scenario: HeatPumpIntegrationScenario | dict[str, Any],
    ) -> tuple[HeatPumpIntegrationScenario, "PinchProblem"]:
        """Build a new :class:`PinchProblem` with an integrated heat-pump scenario."""
        if self._problem_data is None:
            raise RuntimeError("No input loaded. Call load(...) first.")

        validated_scenario = HeatPumpIntegrationScenario.model_validate(scenario)
        base_problem_data = self.to_problem_json()
        scenario_data = deepcopy(
            base_problem_data.model_dump()
            if hasattr(base_problem_data, "model_dump")
            else base_problem_data
        )
        scenario_data.setdefault("streams", []).extend(
            _build_heat_pump_stream_payloads(validated_scenario)
        )

        integrated_problem = PinchProblem.from_json(scenario_data)
        integrated_problem._project_name = f"{self._project_name}_with_hp"
        return validated_scenario, integrated_problem


    def evaluate_heat_pump_integration(
        self,
        scenario: HeatPumpIntegrationScenario | dict[str, Any],
        *,
        target_name: Optional[str] = None,
        base_label: str = "Base case",
        scenario_label: str = "Integrated heat-pump scenario",
    ) -> HeatPumpIntegrationEvaluation:
        """Solve and compare a candidate integrated heat-pump scenario."""
        validated_scenario, integrated_problem = (
            self.build_heat_pump_integration_problem(scenario)
        )
        comparison_frame = self.compare_to(
            integrated_problem,
            target_name=target_name,
            base_label=base_label,
            other_label=scenario_label,
        )

        approximate_power_input = (
            validated_scenario.condenser_duty - validated_scenario.evaporator_duty
        )
        comparison_frame["Approx. HP Power Input"] = [
            None,
            approximate_power_input,
            None,
        ]

        target_value = str(comparison_frame.loc[base_label, "Target"])
        comparison_summary = HeatPumpIntegrationComparison(
            target=target_value,
            base_case_name=self._project_name,
            scenario_case_name=integrated_problem._project_name,
            hot_utility_target_delta=float(
                comparison_frame.loc["Change", "Hot Utility Target"]
            ),
            cold_utility_target_delta=float(
                comparison_frame.loc["Change", "Cold Utility Target"]
            ),
            heat_recovery_delta=float(comparison_frame.loc["Change", "Heat Recovery"]),
            hot_pinch_delta=_coerce_optional_float(
                comparison_frame.loc["Change", "Hot Pinch"]
            ),
            cold_pinch_delta=_coerce_optional_float(
                comparison_frame.loc["Change", "Cold Pinch"]
            ),
            approximate_power_input=float(approximate_power_input),
        )
        return HeatPumpIntegrationEvaluation(
            scenario=validated_scenario,
            comparison=comparison_summary,
            comparison_frame=comparison_frame,
            integrated_problem=integrated_problem,
        )

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

    # ----------------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------------

    @classmethod
    def from_json(cls, data: JsonDict) -> "PinchProblem":
        """Build from an in-memory mapping and apply the normal input cleaners."""
        obj = cls(problem_filepath=None, results_dir=None, run=False)
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

    # ----------------------------------------------------------------------------
    # Internal graph helpers
    # ----------------------------------------------------------------------------

    def _select_graph(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
        index: int = 0,
    ) -> dict:
        graphs = self._select_graphs(zone_name=zone_name, graph_type=graph_type)
        if not graphs:
            raise ValueError("No graphs matched the requested selection.")
        try:
            return graphs[index][1]
        except IndexError as exc:
            raise IndexError(
                f"Graph index {index} is out of range for the selected graphs."
            ) from exc

    def _select_graphs(
        self,
        *,
        zone_name: Optional[str] = None,
        graph_type: Optional[str] = None,
    ) -> list[tuple[str, dict]]:
        payload = self.graph_data()
        selected_graph_type = _normalise_graph_type_selector(graph_type)

        zone_items = payload.items()
        if zone_name is not None:
            resolved_zone_key = zone_name
            if resolved_zone_key not in payload:
                resolved_zone_key = str(zone_name).split("/", 1)[-1]

            try:
                zone_items = [(resolved_zone_key, payload[resolved_zone_key])]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown zone {zone_name!r}. Available zones: {', '.join(payload)}"
                ) from exc

        selected = []
        for zone_key, graph_set in zone_items:
            for graph in graph_set.get("graphs", []):
                if (
                    selected_graph_type is not None
                    and graph.get("type") != selected_graph_type
                ):
                    continue
                selected.append((zone_key, graph))
        return selected


def _normalise_graph_type_selector(graph_type: Optional[str]) -> Optional[str]:
    if graph_type is None:
        return None
    text = str(graph_type).strip()
    return _GRAPH_TYPE_ALIASES.get(text.lower(), text)


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "graph"


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


def _build_heat_pump_stream_payloads(
    scenario: HeatPumpIntegrationScenario,
) -> list[dict[str, Any]]:
    return [
        {
            "zone": scenario.zone,
            "name": scenario.condenser_name,
            "t_supply": {"value": scenario.condenser_temperature, "units": "degC"},
            "t_target": {
                "value": scenario.condenser_temperature - scenario.dt_phase_change,
                "units": "degC",
            },
            "heat_flow": {"value": scenario.condenser_duty, "units": "kW"},
            "dt_cont": {"value": scenario.dt_cont, "units": "degC"},
            "htc": {"value": scenario.htc, "units": "kW/m^2/degC"},
        },
        {
            "zone": scenario.zone,
            "name": scenario.evaporator_name,
            "t_supply": {
                "value": scenario.evaporator_temperature - scenario.dt_phase_change,
                "units": "degC",
            },
            "t_target": {"value": scenario.evaporator_temperature, "units": "degC"},
            "heat_flow": {"value": scenario.evaporator_duty, "units": "kW"},
            "dt_cont": {"value": scenario.dt_cont, "units": "degC"},
            "htc": {"value": scenario.htc, "units": "kW/m^2/degC"},
        },
    ]


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


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
    issues = []

    if len(payload.streams) == 0:
        issues.append("- At least one stream must be provided.")

    for index, stream in enumerate(payload.streams):
        if _maybe_get_value(stream.t_supply) == _maybe_get_value(stream.t_target):
            issues.append(
                _format_semantic_issue(
                    "streams",
                    index,
                    context,
                    "field 't_supply/t_target' - supply and target temperatures must differ.",
                )
            )

        for field_name in ("heat_flow", "dt_cont", "htc"):
            value = _maybe_get_value(getattr(stream, field_name))
            if value is not None and value < 0:
                issues.append(
                    _format_semantic_issue(
                        "streams",
                        index,
                        context,
                        f"field '{field_name}' - value must be non-negative.",
                    )
                )

    for index, utility in enumerate(payload.utilities):
        utility_type = str(utility.type)
        t_supply = _maybe_get_value(utility.t_supply)
        t_target = _maybe_get_value(utility.t_target)
        if (
            utility_type == "Hot"
            and t_supply is not None
            and t_target is not None
            and t_supply < t_target
        ):
            issues.append(
                _format_semantic_issue(
                    "utilities",
                    index,
                    context,
                    "field 't_supply/t_target' - hot utilities must have t_supply >= t_target.",
                )
            )
        if (
            utility_type == "Cold"
            and t_supply is not None
            and t_target is not None
            and t_supply > t_target
        ):
            issues.append(
                _format_semantic_issue(
                    "utilities",
                    index,
                    context,
                    "field 't_supply/t_target' - cold utilities must have t_supply <= t_target.",
                )
            )

        for field_name in ("dt_cont", "htc", "price", "heat_flow"):
            value = _maybe_get_value(getattr(utility, field_name))
            if value is not None and value < 0:
                issues.append(
                    _format_semantic_issue(
                        "utilities",
                        index,
                        context,
                        f"field '{field_name}' - value must be non-negative.",
                    )
                )

    if issues:
        raise ValueError(
            "Input validation failed with "
            f"{len(issues)} issue(s):\n" + "\n".join(issues)
        )


def _format_semantic_issue(
    section: str,
    record_index: int,
    context: dict[str, list[dict[str, Any]]],
    message: str,
) -> str:
    record_context = _lookup_record_context(context, section, record_index)
    return f"- {_describe_record(section, record_index, record_context)}: {message}"
