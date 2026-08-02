"""Multi-case orchestration built around real :class:`PinchProblem` instances."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from ..adapters.io.workspace_bundles import (
    load_workspace_bundle,
    save_workspace_bundle,
)
from ..contracts.input import TargetInput
from ..contracts.workspace import (
    PinchWorkspaceBundle,
    WorkspaceCaseBundleEntry,
    validate_workspace_case_name,
)
from ._problem.input.validation import build_validation_report
from ._workspace import state as _workspace_state
from ._workspace.case_inputs import (
    JsonDict,
    PathLike,
    canonical_case_input_from_source,
    merge_case_inputs,
    normalise_case_input,
)
from .problem import PinchProblem


@dataclass(frozen=True)
class CaseBatchResult:
    """Ordered successes and failures from one explicit batch operation."""

    results: Mapping[str, Any]
    errors: Mapping[str, Exception]


class _CaseBatchAccessor:
    def __init__(self, batch: "_CaseBatch", surface: str) -> None:
        self._batch = batch
        self._surface = surface

    def _run(self, method: str, **kwargs) -> CaseBatchResult:
        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        for name in self._batch.names:
            try:
                accessor = self._batch.workspace.case(name)
                for segment in self._surface.split("."):
                    accessor = getattr(accessor, segment)
                results[name] = getattr(accessor, method)(**kwargs)
            except Exception as exc:  # batch isolation is the public contract
                errors[name] = exc
        return CaseBatchResult(
            results=MappingProxyType(results),
            errors=MappingProxyType(errors),
        )


class _CaseBatchTargetAccessor(_CaseBatchAccessor):
    """Mirror focused target workflows over an ordered case selection."""

    @property
    def all_periods(self) -> "_CaseBatchAllPeriodsTargetAccessor":
        return _CaseBatchAllPeriodsTargetAccessor(self._batch, "target.all_periods")

    def direct_heat_integration(self, **kwargs):
        return self._run("direct_heat_integration", **kwargs)

    def indirect_heat_integration(self, **kwargs):
        return self._run("indirect_heat_integration", **kwargs)

    def total_site_heat_integration(self, **kwargs):
        return self._run("total_site_heat_integration", **kwargs)

    def all_heat_integration(self, **kwargs):
        return self._run("all_heat_integration", **kwargs)

    def heat_exchanger_area_and_cost(self, **kwargs):
        return self._run("heat_exchanger_area_and_cost", **kwargs)

    def carnot_heat_pump(self, **kwargs):
        return self._run("carnot_heat_pump", **kwargs)

    def carnot_refrigeration(self, **kwargs):
        return self._run("carnot_refrigeration", **kwargs)

    def vapour_compression_heat_pump(self, **kwargs):
        return self._run("vapour_compression_heat_pump", **kwargs)

    def vapour_compression_refrigeration(self, **kwargs):
        return self._run("vapour_compression_refrigeration", **kwargs)

    def brayton_heat_pump(self, **kwargs):
        return self._run("brayton_heat_pump", **kwargs)

    def brayton_refrigeration(self, **kwargs):
        return self._run("brayton_refrigeration", **kwargs)

    def mvr_heat_pump(self, **kwargs):
        return self._run("mvr_heat_pump", **kwargs)

    def cogeneration(self, **kwargs):
        return self._run("cogeneration", **kwargs)

    def sun_smith_cogeneration(self, **kwargs):
        return self._run("sun_smith_cogeneration", **kwargs)

    def varbanov_cogeneration(self, **kwargs):
        return self._run("varbanov_cogeneration", **kwargs)

    def isentropic_cogeneration(self, **kwargs):
        return self._run("isentropic_cogeneration", **kwargs)

    def exergy(self, **kwargs):
        return self._run("exergy", **kwargs)

    def energy_transfer(self, **kwargs):
        return self._run("energy_transfer", **kwargs)


class _CaseBatchAllPeriodsTargetAccessor(_CaseBatchAccessor):
    """Mirror supported all-period target workflows over selected cases."""

    def direct_heat_integration(self, **kwargs):
        return self._run("direct_heat_integration", **kwargs)

    def indirect_heat_integration(self, **kwargs):
        return self._run("indirect_heat_integration", **kwargs)

    def total_site_heat_integration(self, **kwargs):
        return self._run("total_site_heat_integration", **kwargs)

    def all_heat_integration(self, **kwargs):
        return self._run("all_heat_integration", **kwargs)

    def heat_exchanger_area_and_cost(self, **kwargs):
        return self._run("heat_exchanger_area_and_cost", **kwargs)

    def carnot_heat_pump(self, **kwargs):
        return self._run("carnot_heat_pump", **kwargs)

    def carnot_refrigeration(self, **kwargs):
        return self._run("carnot_refrigeration", **kwargs)

    def vapour_compression_heat_pump(self, **kwargs):
        return self._run("vapour_compression_heat_pump", **kwargs)

    def vapour_compression_refrigeration(self, **kwargs):
        return self._run("vapour_compression_refrigeration", **kwargs)

    def mvr_heat_pump(self, **kwargs):
        return self._run("mvr_heat_pump", **kwargs)

    def cogeneration(self, **kwargs):
        return self._run("cogeneration", **kwargs)

    def sun_smith_cogeneration(self, **kwargs):
        return self._run("sun_smith_cogeneration", **kwargs)

    def varbanov_cogeneration(self, **kwargs):
        return self._run("varbanov_cogeneration", **kwargs)

    def isentropic_cogeneration(self, **kwargs):
        return self._run("isentropic_cogeneration", **kwargs)

    def exergy(self, **kwargs):
        return self._run("exergy", **kwargs)

    def energy_transfer(self, **kwargs):
        return self._run("energy_transfer", **kwargs)


class _CaseBatchDesignAccessor(_CaseBatchAccessor):
    """Mirror HEN design workflows over an ordered case selection."""

    def heat_exchanger_network(self, **kwargs):
        return self._run("heat_exchanger_network", **kwargs)

    def enhanced_heat_exchanger_network(self, **kwargs):
        return self._run("enhanced_heat_exchanger_network", **kwargs)

    def multiperiod_heat_exchanger_network(self, **kwargs):
        return self._run("multiperiod_heat_exchanger_network", **kwargs)

    def open_hens(self, **kwargs):
        return self._run("open_hens", **kwargs)

    def pinch_design(self, **kwargs):
        return self._run("pinch_design", **kwargs)

    def thermal_derivative(self, **kwargs):
        return self._run("thermal_derivative", **kwargs)

    def network_evolution(self, **kwargs):
        return self._run("network_evolution", **kwargs)


class _CaseBatch:
    def __init__(self, workspace: "PinchWorkspace", names: Iterable[str]) -> None:
        self.workspace = workspace
        self.names = tuple(workspace._resolve_case_name(name) for name in names)
        if not self.names:
            raise ValueError("cases requires at least one case name.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("case names must be unique.")
        self.target = _CaseBatchTargetAccessor(self, "target")
        self.design = _CaseBatchDesignAccessor(self, "design")

    def _run_problem_method(self, method: str, **kwargs) -> CaseBatchResult:
        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        for name in self.names:
            try:
                results[name] = getattr(self.workspace.case(name), method)(**kwargs)
            except Exception as exc:  # batch isolation is the public contract
                errors[name] = exc
        return CaseBatchResult(
            MappingProxyType(results),
            MappingProxyType(errors),
        )

    def summary_frames(self, **kwargs) -> CaseBatchResult:
        """Return ordered summary frames for solved cases."""
        return self._run_problem_method("summary_frame", **kwargs)

    def metrics(self, **kwargs) -> CaseBatchResult:
        """Return ordered typed metrics for solved cases."""
        return self._run_problem_method("metrics", **kwargs)

    def reports(self, **kwargs) -> CaseBatchResult:
        """Return ordered typed reports for solved cases."""
        return self._run_problem_method("report", **kwargs)

    def export_excel(self, destination: PathLike, **kwargs) -> CaseBatchResult:
        """Export each selected case into a distinct case subdirectory."""
        output_dir = os.fspath(destination).rstrip("/\\")
        if not output_dir:
            raise ValueError("destination is required for batch Excel export.")
        export_root = os.path.realpath(output_dir)
        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        for name in self.names:
            try:
                validate_workspace_case_name(name)
                case_directory = os.path.realpath(os.path.join(export_root, name))
                if os.path.commonpath((export_root, case_directory)) != export_root:
                    raise ValueError(
                        f"case name {name!r} resolves outside the batch export "
                        "destination"
                    )
                results[name] = self.workspace.case(name).export_excel(
                    case_directory,
                    **kwargs,
                )
            except Exception as exc:  # batch isolation is the public contract
                errors[name] = exc
        return CaseBatchResult(
            MappingProxyType(results),
            MappingProxyType(errors),
        )


class PinchWorkspace:
    """Manage multiple named :class:`PinchProblem` cases with a script-native API."""

    def __init__(
        self,
        source: (
            TargetInput
            | JsonDict
            | PathLike
            | tuple[PathLike, PathLike]
            | PinchProblem
            | None
        ) = None,
        *,
        project_name: Optional[str] = "Site",
        baseline_name: str = "baseline",
    ) -> None:
        self.baseline_name = validate_workspace_case_name(baseline_name)
        self.project_name = project_name
        self._case_inputs: dict[str, JsonDict] = {}
        self._case_cache: dict[str, PinchProblem] = {}
        self._active_case_name: Optional[str] = None

        if source is not None:
            self.load(source, case_name=baseline_name, activate=True)

    @classmethod
    def load_bundle(cls, path: PathLike) -> "PinchWorkspace":
        """Load a previously persisted workspace bundle."""
        bundle = load_workspace_bundle(path)
        workspace = cls(
            project_name=bundle.project_name,
            baseline_name=bundle.baseline_name,
        )
        workspace._case_inputs = {
            name: deepcopy(entry.case_input) for name, entry in bundle.cases.items()
        }
        workspace._active_case_name = workspace._default_case_name()
        return workspace

    def __repr__(self) -> str:
        active = self._active_case_name or "<unset>"
        return (
            f"PinchWorkspace(cases={self.list_cases()}, "
            f"active_case={active!r}, project_name={self.project_name!r})"
        )

    def load(
        self,
        source: (
            TargetInput
            | JsonDict
            | PathLike
            | tuple[PathLike, PathLike]
            | PinchProblem
            | None
        ),
        *,
        case_name: Optional[str] = None,
        activate: bool = True,
        project_name: Optional[str] = None,
    ) -> Optional[PinchProblem]:
        """Load or replace a named case and return a live validated case."""
        if source is None:
            return self.case(case_name)

        name = (
            validate_workspace_case_name(case_name)
            if case_name is not None
            else self._active_case_name or self.baseline_name
        )
        name = validate_workspace_case_name(name)
        case_input, resolved_project_name = canonical_case_input_from_source(
            source,
            project_name=project_name,
            workspace_project_name=self.project_name,
        )

        self.project_name = resolved_project_name
        self._case_inputs[name] = case_input
        self._invalidate_case_state(name)

        if activate or self._active_case_name is None:
            self._active_case_name = name

        if build_validation_report(case_input).valid:
            return self.case(name)
        return None

    def validation_report(self, case_name: Optional[str] = None):
        """Return a structured validation report for one case input."""
        return build_validation_report(
            self._get_case_input(self._resolve_case_name(case_name))
        )

    def _set_case_input(
        self,
        name: str,
        case_input: TargetInput | JsonDict,
        *,
        base: Optional[str] = None,
    ) -> JsonDict:
        """Create or replace one stored case input."""
        name = validate_workspace_case_name(name)
        normalized = normalise_case_input(case_input)
        if base is not None:
            base_case_input = self._get_case_input(base)
            normalized = merge_case_inputs(base_case_input, normalized)
        self._case_inputs[name] = normalized
        if self._active_case_name is None:
            self._active_case_name = name
        self._invalidate_case_state(name)
        return deepcopy(normalized)

    def list_cases(self) -> list[str]:
        """Return the loaded case names in stable insertion order."""
        return list(self._case_inputs)

    def cases(self, names: Iterable[str] | None = None) -> _CaseBatch:
        """Return an ordered batch view over selected cases."""
        return _CaseBatch(self, self.list_cases() if names is None else names)

    def case(self, name: Optional[str] = None) -> PinchProblem:
        """Return the live :class:`PinchProblem` for one named case."""
        return _workspace_state.case_for_name(self, name)

    def use_case(self, name: str) -> PinchProblem:
        """Activate one named case and return it."""
        self._active_case_name = self._resolve_case_name(name)
        return self.case(self._active_case_name)

    def _create_case_from_base(
        self,
        *,
        source_name: str = "baseline",
        new_name: str = "new",
        activate: bool = False,
    ) -> PinchProblem:
        """Clone one existing case into a new named case."""
        data_source = self.to_problem_json(case_name=source_name)
        return self.load(data_source, case_name=new_name, activate=activate)

    def scenario(
        self,
        name: str,
        *,
        base: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
        replace_options: bool = False,
        dt_cont_multiplier: float | None = None,
        activate: bool = False,
    ) -> PinchProblem:
        """Create and return an unsolved named scenario."""
        source_name = base or self.baseline_name
        case = self._create_case_from_base(
            source_name=source_name,
            new_name=name,
            activate=activate,
        )
        if options:
            case.update_options(options, replace=replace_options)
        if dt_cont_multiplier is not None:
            case.set_dt_cont_multiplier(dt_cont_multiplier)
        self._sync_case_input(name)
        return self.case(name)

    def to_problem_json(
        self,
        *,
        case_name: Optional[str] = None,
    ) -> JsonDict:
        """Return canonical problem input for one named case."""
        resolved_name = self._resolve_case_name(case_name)
        self._sync_case_input(resolved_name)
        return deepcopy(self._case_inputs[resolved_name])

    @property
    def active_case_name(self) -> Optional[str]:
        """Return the currently active case name."""
        return self._active_case_name

    @property
    def target(self):
        """Delegate the ``target`` accessor to the active case."""
        return self.case().target

    @property
    def plot(self):
        """Delegate the ``plot`` accessor to the active case."""
        return self.case().plot

    @property
    def design(self):
        """Delegate the ``design`` accessor to the active case."""
        return self.case().design

    @property
    def components(self):
        """Delegate the ``components`` accessor to the active case."""
        return self.case().components

    @property
    def config(self):
        """Return the active case's read-only configuration view."""
        return self.case().config

    @property
    def problem_data(self):
        """Return the active case input."""
        return self.case().problem_data

    @property
    def problem_filepath(self):
        """Return the active case filepath when available."""
        return self.case().problem_filepath

    @property
    def results(self):
        """Return the active case results when available."""
        return self.case().results

    @property
    def master_zone(self):
        """Return the active case master zone when available."""
        return self.case().master_zone

    def validate(self, case_name: Optional[str] = None):
        """Validate one case input."""
        return self.case(case_name).validate()

    def summary_frame(
        self,
        *,
        case_name: Optional[str] = None,
        detailed: bool = False,
        include_periods: bool = False,
        include_weighted_average: bool = False,
    ) -> pd.DataFrame:
        """Return the solved summary for one case."""
        return self.case(case_name).summary_frame(
            detailed=detailed,
            include_periods=include_periods,
            include_weighted_average=include_weighted_average,
        )

    def metrics(
        self,
        *,
        case_name: Optional[str] = None,
        include_periods: bool = False,
        include_weighted_average: bool = False,
    ):
        """Return typed metrics for one case."""
        return self.case(case_name).metrics(
            include_periods=include_periods,
            include_weighted_average=include_weighted_average,
        )

    def report(
        self,
        *,
        case_name: Optional[str] = None,
        include_periods: bool = False,
        include_weighted_average: bool = False,
    ):
        """Return a typed report for one case."""
        return self.case(case_name).report(
            include_periods=include_periods,
            include_weighted_average=include_weighted_average,
        )

    def export_excel(
        self,
        destination: PathLike,
        *,
        case_name: Optional[str] = None,
        include_periods: bool = False,
        include_weighted_average: bool = False,
    ) -> Any:
        """Export one case to an Excel workbook."""
        return self.case(case_name).export_excel(
            destination,
            include_periods=include_periods,
            include_weighted_average=include_weighted_average,
        )

    def set_dt_cont_multiplier(
        self,
        value: float,
        *,
        zone_name: Optional[str] = None,
        case_name: Optional[str] = None,
    ):
        """Update one case multiplier and keep the stored case input in sync."""
        resolved_name = self._resolve_case_name(case_name)
        result = self.case(resolved_name).set_dt_cont_multiplier(
            value,
            zone_name=zone_name,
        )
        self._sync_case_input(resolved_name)
        return result

    def update_options(
        self,
        options: dict[str, Any],
        *,
        case_name: Optional[str] = None,
        replace: bool = False,
    ) -> PinchProblem:
        """Update one case's options and keep the stored case input in sync."""
        resolved_name = self._resolve_case_name(case_name)
        problem = self.case(resolved_name)
        problem.update_options(options, replace=replace)
        self._sync_case_input(resolved_name)
        return problem

    def show_dashboard(
        self,
        *,
        case_name: Optional[str] = None,
        zone=None,
        graph_data: Optional[dict[str, Any]] = None,
        page_title: Optional[str] = "OpenPinch Dashboard",
        value_rounding: int = 2,
    ) -> None:
        """Launch the dashboard for one case."""
        self.case(case_name).show_dashboard(
            zone=zone,
            graph_data=graph_data,
            page_title=page_title,
            value_rounding=value_rounding,
        )

    def compare_to(
        self,
        other_problem: PinchProblem | "PinchWorkspace",
        *,
        case_name: Optional[str] = None,
        other_case_name: Optional[str] = None,
        scope: Optional[str] = None,
        zone_type: Optional[str] = None,
        integration_type: Optional[str] = None,
        target_method: Optional[str] = None,
        base_label: str = "Base case",
        other_label: str = "Scenario",
    ) -> pd.DataFrame:
        """Compare one workspace case to another problem or workspace case."""
        base_problem = self.case(case_name)
        if isinstance(other_problem, PinchWorkspace):
            comparison_problem = other_problem.case(other_case_name)
        else:
            comparison_problem = other_problem
        return base_problem.compare_to(
            comparison_problem,
            scope=scope,
            zone_type=zone_type,
            integration_type=integration_type,
            target_method=target_method,
            base_label=base_label,
            other_label=other_label,
        )

    def compare_cases(
        self,
        base_case: str,
        other_case: str,
        *,
        scope: Optional[str] = None,
        zone_type: Optional[str] = None,
        integration_type: Optional[str] = None,
        target_method: Optional[str] = None,
        base_label: Optional[str] = None,
        other_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compare two cases in the same workspace."""
        return self.case(base_case).compare_to(
            self.case(other_case),
            scope=scope,
            zone_type=zone_type,
            integration_type=integration_type,
            target_method=target_method,
            base_label=base_label or base_case,
            other_label=other_label or other_case,
        )

    def save_bundle(self, path: PathLike) -> Any:
        """Persist the current workspace, syncing any live case edits first."""
        self._sync_all_cases()
        bundle = PinchWorkspaceBundle(
            schema_version="3",
            project_name=self.project_name,
            baseline_name=self.baseline_name,
            cases={
                name: WorkspaceCaseBundleEntry(
                    case_input=deepcopy(self._get_case_input(name)),
                )
                for name in self.list_cases()
            },
        )
        return save_workspace_bundle(path, bundle)

    def _resolve_case_name(self, name: Optional[str]) -> str:
        return _workspace_state.resolve_case_name(self, name)

    def _default_case_name(self) -> Optional[str]:
        return _workspace_state.default_case_name(self)

    def _get_case_input(self, name: str) -> JsonDict:
        self._sync_case_input(name)
        try:
            return self._case_inputs[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown case {name!r}. Available cases: "
                f"{', '.join(self.list_cases())}"
            ) from exc

    def _invalidate_case_state(self, name: str) -> None:
        """Drop cached case and view state for one variant case input."""
        _workspace_state.invalidate_case_state(self, name)

    def _sync_case_input(self, name: str) -> None:
        _workspace_state.sync_case_input(self, name)

    def _sync_all_cases(self) -> None:
        for name in list(self._case_cache):
            self._sync_case_input(name)


__all__ = ["PinchWorkspace"]
