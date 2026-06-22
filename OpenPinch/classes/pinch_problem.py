"""High-level convenience wrapper around the OpenPinch targeting service."""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from pydantic import ValidationError

from ..lib.schemas.io import TargetInput, TargetOutput
from ..lib.schemas.targets import BaseTargetModel
from ..resources import list_sample_cases, read_sample_case
from ..services import data_preprocessing_service
from ..services.common.miscellaneous import get_state_index
from ..services.input_data_processing._canonicalization import canonical_problem_inputs
from ..streamlit_webviewer.web_graphing import (
    render_streamlit_dashboard as _render_streamlit_dashboard,
)
from ..utils.csv_to_json import get_problem_from_csv
from ..utils.export import (
    build_summary_dataframe,
    export_target_summary_to_excel_with_units,
)
from ..utils.wkbook_to_json import get_problem_from_excel
from ._problem import (
    JsonDict,
    PathLike,
    _ComponentAccessorDescriptor,
    _DesignAccessorDescriptor,
    _LoadedProblemSource,
    _PlotAccessorDescriptor,
    _ProblemSourceAdapters,
    _TargetAccessorDescriptor,
    _validate_problem_semantics,
    build_graph_payload,
    build_problem_summary_frame,
    extract_results,
    load_problem_source,
    prepare_in_memory_problem_source,
    run_targeting_for_zone_and_subzones,
)
from ._problem import (
    format_schema_validation_error as _format_schema_validation_error,
)
from ._problem import (
    locate_summary_row as _locate_summary_row,
)
from .stream_collection import StreamCollection
from .value import Value
from .zone import Zone

ZoneService = Callable[["Zone", Optional[dict[str, Any]]], "Zone"]


class PinchProblem:
    """Typed orchestrator for loading input data and running targeting."""

    results_dir: Optional[Path]
    _problem_filepath: Optional[Path]
    _problem_data: Optional[JsonDict | TargetInput]
    _project_name: str
    _results: Optional[TargetOutput]
    _validated_data: Optional[TargetInput]
    _master_zone: Optional["Zone"]
    _process_components: dict[str, Any]
    _input_source_kind: str
    _validation_context: Optional[dict[str, list[dict[str, Any]]]]
    add_component = _ComponentAccessorDescriptor()
    design = _DesignAccessorDescriptor()
    plot = _PlotAccessorDescriptor()
    target = _TargetAccessorDescriptor()

    def __init__(
        self,
        source: (
            TargetInput | JsonDict | PathLike | tuple[PathLike, PathLike] | None
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
        self._process_components = {}
        self.results_dir = None

        if source is not None:
            self.load(source=source)

    def load(
        self,
        source: (
            TargetInput | JsonDict | PathLike | tuple[PathLike, PathLike] | None
        ) = None,
    ) -> Optional[Zone]:
        """Load problem inputs from JSON, Excel, CSV, or an in-memory object."""
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
        options: Optional[dict[str, Any]] = None,
        sid: str = None,
    ) -> TargetOutput:
        """Run the targeting analysis against the loaded input and cache the result."""
        if not isinstance(zone, Zone):
            zone = self._build_execution_master_zone()
        runtime_options, sid = self._resolve_runtime_state_options(
            options,
            zone=zone,
        )
        run_targeting_for_zone_and_subzones(
            zone=zone,
            direct_service_func=direct_service_func,
            indirect_service_func=indirect_service_func,
            args=runtime_options,
        )
        self._attach_process_component_work_targets(zone, runtime_options)
        self._results = TargetOutput.model_validate(extract_results(zone, state_id=sid))
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
        sid: str = None,
    ) -> BaseTargetModel:
        execution_master_zone = self._build_execution_master_zone()
        runtime_options, sid = self._resolve_runtime_state_options(
            options,
            zone=execution_master_zone,
        )
        zone = self._resolve_target_zone(
            application_zone, master_zone=execution_master_zone
        )
        if include_subzones:
            self._run_targeting_for_zone_and_subzones(
                zone=zone,
                direct_service_func=direct_service_func,
                indirect_service_func=indirect_service_func,
                options=runtime_options,
                sid=sid,
            )
        else:
            if direct_service_func is not None:
                direct_service_func(zone, runtime_options)
            if indirect_service_func is not None:
                indirect_service_func(zone, runtime_options)
            self._attach_process_component_work_targets(
                execution_master_zone,
                runtime_options,
            )
            self._results = TargetOutput.model_validate(
                extract_results(execution_master_zone, state_id=sid)
            )

        try:
            return zone.targets[target_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Targeting did not produce target {target_id!r} "
                f"for zone {zone.name!r}."
            ) from exc

    def _execute_cogeneration_targeting(
        self,
        *,
        application_zone: Optional[str | Zone],
        options: Optional[dict[str, Any]],
        include_subzones: bool,
        service_func: Optional[ZoneService] = None,
        sid: str = None,
    ) -> BaseTargetModel:
        """Run cogeneration targeting and return the family selected at runtime."""
        execution_master_zone = self._build_execution_master_zone()
        runtime_options, sid = self._resolve_runtime_state_options(
            options,
            zone=execution_master_zone,
        )
        zone = self._resolve_target_zone(
            application_zone, master_zone=execution_master_zone
        )
        if include_subzones:
            self._run_targeting_for_zone_and_subzones(
                zone=zone,
                direct_service_func=service_func,
                options=runtime_options,
                sid=sid,
            )
        else:
            if service_func is not None:
                service_func(zone, runtime_options)
            self._attach_process_component_work_targets(
                execution_master_zone,
                runtime_options,
            )
            self._results = TargetOutput.model_validate(
                extract_results(execution_master_zone, state_id=sid)
            )

        selected_target_type = getattr(zone, "_selected_cogeneration_target_type", None)
        if not isinstance(selected_target_type, str):
            raise RuntimeError(
                "Cogeneration did not select a compatible target "
                f"for zone {zone.name!r}."
            )
        try:
            return zone.targets[selected_target_type]
        except KeyError as exc:
            raise RuntimeError(
                "Cogeneration selected target "
                f"{selected_target_type!r} for zone {zone.name!r}, "
                "but that target was not available on the zone."
            ) from exc

    def _run_exergy_targeting_for_zone_and_subzones(
        self,
        *,
        zone: "Zone",
        service_func: Optional[ZoneService],
        options: Optional[dict[str, Any]],
    ) -> None:
        """Run exergy targeting in post-order so site targets see solved children."""
        child_options = dict(options or {})
        child_options.pop("base_target_type", None)
        for subzone in zone.subzones.values():
            self._run_exergy_targeting_for_zone_and_subzones(
                zone=subzone,
                service_func=service_func,
                options=child_options,
            )
        if service_func is not None:
            service_func(zone, options)

    def _execute_exergy_targeting(
        self,
        *,
        application_zone: Optional[str | Zone],
        options: Optional[dict[str, Any]],
        include_subzones: bool,
        service_func: Optional[ZoneService] = None,
        sid: str = None,
    ) -> BaseTargetModel:
        """Apply exergy targeting and return the compatible target family selected."""
        execution_master_zone = self._build_execution_master_zone()
        runtime_options, sid = self._resolve_runtime_state_options(
            options,
            zone=execution_master_zone,
        )
        zone = self._resolve_target_zone(
            application_zone, master_zone=execution_master_zone
        )

        if include_subzones:
            self._run_exergy_targeting_for_zone_and_subzones(
                zone=zone,
                service_func=service_func,
                options=runtime_options,
            )
        elif service_func is not None:
            service_func(zone, runtime_options)

        self._attach_process_component_work_targets(
            execution_master_zone,
            runtime_options,
        )
        self._results = TargetOutput.model_validate(
            extract_results(execution_master_zone, state_id=sid)
        )

        selected_target_type = getattr(zone, "_selected_exergy_target_type", None)
        if not isinstance(selected_target_type, str):
            raise RuntimeError(
                "Exergy targeting did not select a compatible target "
                f"for zone {zone.name!r}."
            )

        try:
            return zone.targets[selected_target_type]
        except KeyError as exc:
            raise RuntimeError(
                "Exergy targeting selected target "
                f"{selected_target_type!r} for zone {zone.name!r}, "
                "but that target was not available on the zone."
            ) from exc

    def _resolve_target_zone(
        self,
        application_zone: Optional[str] = None,
        *,
        master_zone: Optional["Zone"] = None,
    ) -> "Zone":
        selected_master_zone = master_zone or self._master_zone
        if selected_master_zone is None:
            raise RuntimeError("Load problem source data first before targeting.")
        if isinstance(application_zone, Zone):
            return application_zone
        if application_zone is None:
            return selected_master_zone
        return selected_master_zone.get_subzone(application_zone)

    def _attach_process_component_work_targets(
        self,
        zone: "Zone",
        runtime_options: Optional[dict[str, Any]],
    ) -> None:
        if not self._process_components:
            return
        state_id = (runtime_options or {}).get("state_id")
        state_idx = (runtime_options or {}).get("idx")
        for current_zone in self._walk_zone_tree(zone):
            component_work = self._process_component_work_for_zone(
                current_zone,
                state_id=state_id,
                state_idx=state_idx,
            )
            for target in current_zone.targets.values():
                if hasattr(target, "process_component_work_target"):
                    target.process_component_work_target = component_work
                if (
                    component_work > 0.0
                    and hasattr(target, "work_target")
                    and getattr(target, "work_target", None) is None
                ):
                    target.work_target = component_work

    def _process_component_work_for_zone(
        self,
        zone: "Zone",
        *,
        state_id: str | None,
        state_idx: int | None,
    ) -> float:
        total = 0.0
        for component in self._process_components.values():
            work_for_zone = getattr(component, "work_for_zone", None)
            if work_for_zone is None:
                continue
            total += float(work_for_zone(zone, state_id=state_id, state_idx=state_idx))
        return total

    def _walk_zone_tree(self, zone: "Zone"):
        yield zone
        for subzone in zone.subzones.values():
            yield from self._walk_zone_tree(subzone)

    def _build_execution_master_zone(self) -> "Zone":
        if self._problem_data is None and self._master_zone is None:
            raise RuntimeError("No input loaded. Call load(...) first.")
        if self._master_zone is None:
            self.load(self._problem_data)
        return self._master_zone

    @property
    def state_ids(self) -> dict[str, int]:
        """Return the canonical ``state_id -> idx`` lookup for the loaded problem."""
        master_zone = self._require_prepared_root_zone()
        return master_zone.state_ids

    def target_all_states(
        self,
        *,
        parallel: bool | str = False,
        max_workers: int | None = None,
        preserve_cached_results: bool = True,
    ) -> dict[str, TargetOutput]:
        """Run default targeting once per canonical state id.

        Parameters
        ----------
        parallel:
            ``False`` runs serially. ``True`` and ``"process"`` use a process pool,
            while ``"thread"`` uses a thread pool which is suitable for no-GIL
            Python builds.
        max_workers:
            Optional executor worker limit for parallel runs.
        preserve_cached_results:
            Restore the original ``results`` cache after the batch run when ``True``.
        """
        state_ids = list(self.state_ids.keys())
        if not state_ids:
            raise ValueError("This problem has no canonical state_ids to target.")

        previous_results = self._results
        try:
            if parallel in (False, None):
                results_by_requested_state = {
                    state_id: self._solve_target_for_state(state_id)
                    for state_id in state_ids
                }
                return self._order_state_results(
                    state_ids=state_ids,
                    results_by_requested_state=results_by_requested_state,
                )
            return self._target_all_states_parallel(
                state_ids=state_ids,
                backend="thread" if parallel == "thread" else "process",
                max_workers=max_workers,
            )
        finally:
            if preserve_cached_results:
                self._results = previous_results

    def _solve_target_for_state(self, state_id: str) -> TargetOutput:
        result = self.target(state_id=state_id)
        return TargetOutput.model_validate(result.model_dump(mode="python"))

    def _resolve_runtime_state_options(
        self,
        options: Optional[dict[str, Any]],
        *,
        zone: "Zone",
    ) -> tuple[dict[str, Any], str | None]:
        runtime_options = dict(options or {})
        idx, sid = get_state_index(state_ids=zone.state_ids, args=runtime_options)
        runtime_options["idx"] = idx
        if sid is not None:
            runtime_options["state_id"] = sid
        return runtime_options, sid

    def _state_result_key(
        self,
        result: TargetOutput,
        *,
        requested_state_id: str,
    ) -> str:
        return (
            str(result.state_id) if result.state_id is not None else requested_state_id
        )

    def _order_state_results(
        self,
        *,
        state_ids: list[str],
        results_by_requested_state: dict[str, TargetOutput],
    ) -> dict[str, TargetOutput]:
        ordered_results: dict[str, TargetOutput] = {}
        for requested_state_id in state_ids:
            result = results_by_requested_state[requested_state_id]
            ordered_results[
                self._state_result_key(
                    result,
                    requested_state_id=requested_state_id,
                )
            ] = result
        return ordered_results

    def _target_all_states_parallel(
        self,
        *,
        state_ids: list[str],
        backend: str,
        max_workers: int | None,
    ) -> dict[str, TargetOutput]:
        problem_inputs = self.canonical_problem_json()
        executor_cls = (
            ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        )
        results_by_requested_state: dict[str, TargetOutput] = {}
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _solve_default_target_for_state,
                    problem_inputs,
                    self.project_name,
                    state_id,
                ): state_id
                for state_id in state_ids
            }
            for future in as_completed(futures):
                state_id = futures[future]
                results_by_requested_state[state_id] = TargetOutput.model_validate(
                    future.result()
                )
        return self._order_state_results(
            state_ids=state_ids,
            results_by_requested_state=results_by_requested_state,
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
        results = self._results if self._results is not None else self.target()
        if detailed:
            return build_summary_dataframe(results.targets)
        return build_problem_summary_frame(results, detailed=False)

    def export_excel(self, results_dir: Optional[PathLike] = None) -> Path:
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
        unit_columns = {col: f"{col} (unit)" for col in columns}

        row_columns = [*columns, *unit_columns.values()]
        base_payload = {
            "Target": str(base_row["Target"]),
            **{col: base_row.get(col) for col in row_columns},
        }
        other_payload = {
            "Target": str(other_row["Target"]),
            **{col: other_row.get(col) for col in row_columns},
        }
        change_row: dict[str, object] = {"Target": str(base_row["Target"])}
        for col in columns:
            unit_col = unit_columns[col]
            base_unit = base_row.get(unit_col)
            other_unit = other_row.get(unit_col)
            base_value = base_row.get(col)
            other_value = other_row.get(col)
            try:
                base_value = Value(base_value, base_unit)
                other_value = Value(other_value, other_unit).to(base_unit)
                other_unit = base_unit
                change_row[col] = float(other_value) - float(base_value)
                change_row[unit_col] = base_unit
            except (TypeError, ValueError):
                change_row[col] = None
                change_row[unit_col] = None

        return pd.DataFrame.from_dict(
            {
                base_label: base_payload,
                other_label: other_payload,
                "Change": change_row,
            },
            orient="index",
            columns=["Target", *row_columns],
        )

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
        """Return the prepared root zone after a successful ``load()`` pass."""
        return self._master_zone

    @property
    def process_components(self) -> dict[str, Any]:
        """Memory-only process components applied to the prepared model."""
        return self._process_components

    @property
    def hot_streams(self) -> StreamCollection:
        """Hot process streams on the root analysis zone."""
        return self._require_prepared_root_zone().hot_streams

    @property
    def cold_streams(self) -> StreamCollection:
        """Cold process streams on the root analysis zone."""
        return self._require_prepared_root_zone().cold_streams

    @property
    def hot_utilities(self) -> StreamCollection:
        """Hot utility streams on the root analysis zone."""
        return self._require_prepared_root_zone().hot_utilities

    @property
    def cold_utilities(self) -> StreamCollection:
        """Cold utility streams on the root analysis zone."""
        return self._require_prepared_root_zone().cold_utilities

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
        """Return the currently loaded problem inputs."""
        if self._problem_data is None:
            raise RuntimeError(
                "No problem_data available. Did you call load(...) or from_json(...)?"
            )
        if canonical:
            return self._canonical_problem_inputs()
        return self._problem_data

    def canonical_problem_json(self) -> JsonDict:
        """Return canonical mutable problem inputs with an explicit zone tree."""
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
            warnings.warn(
                "dt_cont_multiplier must be a finite non-negative value. "
                "Used default value of 1.0 instead.",
                UserWarning,
            )
            resolved_value = 1.0
        self._master_zone.get_subzone(zone_name).dt_cont_multiplier = resolved_value
        self._results = None  # Clear cached results since multipliers have changed
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

        problem_inputs = self.canonical_problem_json()
        current_options = problem_inputs.get("options") or {}
        problem_inputs["options"] = (
            deepcopy(options)
            if replace
            else {**deepcopy(current_options), **deepcopy(options)}
        )
        self._replace_problem_inputs(problem_inputs)
        return self._master_zone

    def _canonical_problem_inputs(self) -> JsonDict:
        """Return canonical mutable problem inputs with an explicit zone tree."""
        validated = self.validate()
        return canonical_problem_inputs(validated, project_name=self.project_name)

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

    def show_dashboard(
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

    def _refresh_results_from_master_zone(self) -> TargetOutput:
        if self._master_zone is None:
            raise RuntimeError("No analysed zone is available. Run target() first.")
        self._results = TargetOutput.model_validate(extract_results(self._master_zone))
        return self._results

    def _require_prepared_root_zone(self) -> Zone:
        """Return the prepared root zone, rebuilding it lazily when possible."""
        if self._master_zone is None:
            if self._problem_data is None:
                raise RuntimeError("No input loaded. Call load(...) first.")
            return self._rebuild_problem_state()
        return self._master_zone

    def _problem_source_adapters(self) -> _ProblemSourceAdapters:
        """Build source adapters lazily so tests can monkeypatch module symbols."""
        return _ProblemSourceAdapters(
            get_problem_from_excel=get_problem_from_excel,
            get_problem_from_csv=get_problem_from_csv,
            list_sample_cases=list_sample_cases,
            read_sample_case=read_sample_case,
        )

    def _apply_loaded_source(self, loaded_source: _LoadedProblemSource) -> None:
        """Apply one normalized source bundle to this problem instance."""
        self._problem_data = loaded_source.input_data
        self._input_source_kind = loaded_source.source_kind
        self._validation_context = loaded_source.validation_context
        self._problem_filepath = loaded_source.problem_filepath
        if loaded_source.project_name is not None:
            self._project_name = loaded_source.project_name

    def _rebuild_problem_state(self) -> Zone:
        """Revalidate, reconstruct the zone tree, and clear cached results."""
        self._validated_data = self.validate()
        self._master_zone = self._data_preprocessing()
        self._process_components = {}
        self._results = None
        return self._master_zone

    def _replace_problem_inputs(self, problem_inputs: JsonDict) -> Zone:
        """Replace the current problem inputs and rebuild analysis state."""
        current_filepath = self._problem_filepath
        loaded_source = prepare_in_memory_problem_source(
            problem_inputs,
            source_kind=self._input_source_kind or "target_input",
        )
        self._apply_loaded_source(loaded_source)
        self._problem_filepath = current_filepath
        return self._rebuild_problem_state()


def _solve_default_target_for_state(
    problem_inputs: JsonDict,
    project_name: str,
    state_id: str,
) -> dict[str, Any]:
    problem = PinchProblem(source=problem_inputs, project_name=project_name)
    result = problem.target(state_id=state_id)
    return result.model_dump(mode="python")
