"""High-level convenience wrapper around the OpenPinch targeting service.

``PinchProblem`` provides a script-friendly interface for loading validated
inputs from JSON/Excel/CSV sources, running analysis, exporting results, and
launching the Streamlit dashboard.
"""

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
from pydantic import ValidationError

from ..services.common.graph_data import get_output_graph_data
from ..services import (
    data_preprocessing_service,
    area_cost_targeting_service,
    direct_heat_integration_service,
    direct_heat_pump_service,
    direct_refrigeration_service,
    indirect_heat_integration_service,
    indirect_heat_pump_service,
    indirect_refrigeration_service,
    power_cogeneration_service,
)
from ..lib.enums import GT, TT, ZT
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
from .zone import Zone
from .energy_target import EnergyTarget
from ..services.services_entry import data_preprocessing_service
from ..utils.multiscale_targeting import (
    get_targets, 
    extract_results,
)


if TYPE_CHECKING:
    from .energy_target import EnergyTarget
    

JsonDict = Dict[str, Any]
PathLike = Union[str, Path]
GraphPayload = Dict[str, Dict[str, Any]]
ZoneService = Callable[["Zone"], "Zone"]

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
    _validated_data: TargetInput = None
    _master_zone: Optional["Zone"] = None
    _input_source_kind: str = "unknown"
    _validation_context: Optional[dict[str, list[dict[str, Any]]]] = None

    def __init__(
        self,
        problem_filepath: Optional[PathLike] = None,
        results_dir: Optional[PathLike] = None,
        project_name: Optional[str] = "Untitled",        
    ) -> None:
        """Initialise the orchestrator and optionally run the full targeting workflow.

        Parameters
        ----------
        problem_filepath:
            Path to a JSON or Excel problem definition handled by :meth:`load`.
        results_dir:
            Destination directory for exported Excel summaries. May be ``None`` if export
            is handled later.
        """
        self._input_source_kind = "unknown"
        self._validation_context = None
        self._project_name = project_name
        if problem_filepath is not None:
            self.load(source=Path(problem_filepath))
        else:
            self._problem_filepath = None
            self._problem_data = None

        self.results_dir = Path(results_dir) if results_dir is not None else None
        self._results = None


    # ----------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------

    def load(
        self,
        source: Union[TargetInput, PathLike, Tuple[PathLike, PathLike]] = None,
    ) -> TargetInput:
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
        if source is None:
            source = Path(self.problem_filepath)

        try:
            source = TargetInput.model_validate(source)
        except:
            pass

        if isinstance(source, TargetInput):
            self._problem_data = source
            self._input_source_kind = "target_input"
            self._validation_context = _build_validation_context(
                source.model_dump() if hasattr(source, "model_dump") else source,
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


    def target(self) -> TargetOutput:
        """Run the targeting analysis against the loaded input and cache the result."""
        if self._master_zone is None:
            if self._problem_data is None:
                raise RuntimeError("No input loaded. Call load(...) first.")
            else:
                self.load(self._problem_data)
        
        # Perform advanced targeting analysis on the master zone and all subzones
        master_zone = get_targets(self._master_zone)
        # Extract the core results from the master zone
        return_data = extract_results(master_zone)
        # Validate response data
        self._results = TargetOutput.model_validate(return_data)
        return self._results


    def target_direct_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Run direct heat-pump targeting on the selected solved zone."""
        return self._execute_zone_service(
            direct_heat_integration_service,
            target_id=TT.DI.value,
            zone_name=zone_name,
            options=options,
        )
    

    def target_indirect_heat_integration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Run direct heat-pump targeting on the selected solved zone."""
        return self._execute_zone_service(
            indirect_heat_integration_service,
            target_id=TT.TS.value,
            zone_name=zone_name,
            options=options,
        )


    def target_direct_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Run direct heat-pump targeting on the selected solved zone."""
        return self._execute_zone_service(
            direct_heat_pump_service,
            target_id=TT.DHP.value,
            zone_name=zone_name,
            options=options,
        )


    def target_indirect_heat_pump(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Run indirect heat-pump targeting on the selected solved zone."""
        return self._execute_zone_service(
            indirect_heat_pump_service,
            target_id=TT.IHP.value,
            zone_name=zone_name,
            options=options,
        )


    def target_direct_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Run direct refrigeration targeting on the selected solved zone."""
        return self._execute_zone_service(
            direct_refrigeration_service,
            target_id=TT.DR.value,
            zone_name=zone_name,
            options=options,
        )


    def target_indirect_refrigeration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Run indirect refrigeration targeting on the selected solved zone."""
        return self._execute_zone_service(
            indirect_refrigeration_service,
            target_id=TT.IR.value,
            zone_name=zone_name,
            options=options,
        )


    def target_cogeneration(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Run cogeneration targeting on the selected solved zone."""
        target_id = TT.DI.value
        if options and "base_target_type" in options:
            target_id = str(options["base_target_type"])
        return self._execute_zone_service(
            power_cogeneration_service,
            target_id=target_id,
            zone_name=zone_name,
            options=options,
        )


    def target_area_cost(
        self,
        *,
        zone_name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> "EnergyTarget":
        """Recompute direct targets with area and cost targeting enabled."""
        return self._execute_zone_service(
            area_cost_targeting_service,
            target_id=TT.DI.value,
            zone_name=zone_name,
            options=options,
        )


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


    def graph_data(self) -> GraphPayload:
        """Return the serialized graph payload for the solved problem."""
        self.target()

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


    def plot_grand_composite_curve_with_heat_pump(self, *, zone_name: Optional[str] = None):
        """Build the heat pump load profile curve Plotly figure for the selected zone."""
        return self.plot(zone_name=zone_name, graph_type=GT.GCC_HP.value)
    

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


    def _data_preprocessing(self) -> "Zone":
        if isinstance(self._validated_data, TargetInput) and isinstance(self._project_name, str):
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
        """Return the analysed Zone hierarchy after a successful ``target()`` run."""
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
        obj = cls(problem_filepath=None, results_dir=None)
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


    def _execute_zone_service(
        self,
        service: ZoneService,
        *,
        target_id: str,
        zone_name: Optional[str],
        options: Optional[dict[str, Any]],
    ) -> "EnergyTarget":
        if self._master_zone is None:
            self.target()

        zone = self._resolve_target_zone(zone_name)
        service(zone, options)
        self._refresh_results_from_master_zone()

        try:
            return zone.targets[target_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Service {service.__name__!r} did not produce target {target_id!r}."
            ) from exc


    def _resolve_target_zone(self, zone_name: Optional[str]) -> "Zone":
        if self._master_zone is None:
            raise RuntimeError("No analysed zone is available. Run target() first.")
        if zone_name is None:
            return self._master_zone
        return self._master_zone.get_subzone(zone_name)


    def _refresh_results_from_master_zone(self) -> TargetOutput:
        if self._master_zone is None:
            raise RuntimeError("No analysed zone is available. Run target() first.")
        self._results = TargetOutput.model_validate(extract_results(self._master_zone))
        return self._results


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
                    f"field '{"heat_flow"}' - value must be non-negative.",
                )
            )

        value = _maybe_get_value(getattr(stream, "dt_cont"))
        if value is not None and value < 0:
            warnings_issues.append(
                _format_semantic_issue(
                    "streams",
                    index,
                    context,
                    f"Warning: field '{"dt_cont"}' - value should be non-negative.",
                )
            )

        value = _maybe_get_value(getattr(stream, "htc"))
        if value is not None and value <= 0:
            fatal_issues.append(
                _format_semantic_issue(
                    "streams",
                    index,
                    context,
                    f"field '{"htc"}' - value must be non-negative.",
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
            warnings_issues.append(
                _format_semantic_issue(
                    "utilities",
                    index,
                    context,
                    "field 't_supply/t_target' - hot utilities should have t_supply >= t_target.",
                )
            )
        if (
            utility_type == "Cold"
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
                    f"field '{"htc"}' - value must be non-negative.",
                )
            )

    if fatal_issues:
        raise ValueError(
            "Input validation failed with "
            f"{len(fatal_issues)} issue(s):\n" + "\n".join(fatal_issues)
        )
    
    if warnings_issues:
        warnings.warn(
            "Input validation failed with "
            f"{len(fatal_issues)} issue(s):\n" + "\n".join(fatal_issues),
            UserWarning,
        )    


def _format_semantic_issue(
    section: str,
    record_index: int,
    context: dict[str, list[dict[str, Any]]],
    message: str,
) -> str:
    record_context = _lookup_record_context(context, section, record_index)
    return f"- {_describe_record(section, record_index, record_context)}: {message}"
