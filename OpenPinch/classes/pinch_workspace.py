"""Multi-case orchestration built around real :class:`PinchProblem` instances."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from ..lib.schemas.io import TargetInput
from ..lib.schemas.workspace import (
    ConfigurationFieldMetadata,
    PinchWorkspaceBundle,
    ScenarioComparisonView,
    ScenarioVariantBundleEntry,
    ScenarioVariantView,
    ScenarioWorkflowConfig,
    VariantPayloadView,
)
from ._problem._validation import build_validation_report
from ._workspace.execution import (
    WorkspaceExecutionError,
    run_problem_workflow,
    workflow_support_level,
    workflow_warnings,
)
from ._workspace.payloads import (
    JsonDict,
    PathLike,
    canonical_payload_from_source,
    merge_payloads,
    normalise_payload,
    project_name_from_payload,
)
from ._workspace.views import (
    configuration_field_metadata as _configuration_field_metadata,
)
from ._workspace.views import (
    error_variant_view,
    invalid_variant_view,
    json_safe,
    problem_table_diffs,
    problem_to_variant_view,
    record_views,
    summary_metric_deltas,
    zone_tree_view,
)
from .pinch_problem import PinchProblem


class PinchWorkspace:
    """Manage multiple named :class:`PinchProblem` cases with a script-native API."""

    def __init__(
        self,
        source: TargetInput
        | JsonDict
        | PathLike
        | tuple[PathLike, PathLike]
        | PinchProblem
        | None = None,
        *,
        project_name: Optional[str] = "Site",
        baseline_name: str = "baseline",
    ) -> None:
        self.baseline_name = baseline_name
        self.project_name = project_name
        self._variant_payloads: dict[str, JsonDict] = {}
        self._variant_workflows: dict[str, ScenarioWorkflowConfig] = {}
        self._cached_views: dict[str, ScenarioVariantView] = {}
        self._case_cache: dict[str, PinchProblem] = {}
        self._active_case_name: Optional[str] = None

        if source is not None:
            self.load(source, case_name=baseline_name, activate=True)

    @classmethod
    def from_json(
        cls,
        data: JsonDict,
        *,
        baseline_name: str = "baseline",
        project_name: Optional[str] = None,
    ) -> "PinchWorkspace":
        return cls(
            data,
            baseline_name=baseline_name,
            project_name=project_name,
        )

    @classmethod
    def load_bundle(cls, path: PathLike) -> "PinchWorkspace":
        """Load a previously persisted workspace bundle."""
        bundle = PinchWorkspaceBundle.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
        workspace = cls(
            project_name=bundle.project_name,
            baseline_name=bundle.baseline_name,
        )
        workspace._variant_payloads = {
            name: deepcopy(entry.payload) for name, entry in bundle.variants.items()
        }
        workspace._variant_workflows = {
            name: entry.workflow.model_copy(deep=True)
            for name, entry in bundle.variants.items()
        }
        workspace._cached_views = {
            name: entry.cached_view.model_copy(deep=True)
            for name, entry in bundle.variants.items()
            if entry.cached_view is not None
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
        source: TargetInput
        | JsonDict
        | PathLike
        | tuple[PathLike, PathLike]
        | PinchProblem
        | None,
        *,
        case_name: Optional[str] = None,
        activate: bool = True,
        project_name: Optional[str] = None,
    ) -> Optional[PinchProblem]:
        """Load or replace a named case and return a live validated case."""
        if source is None:
            return self.case(case_name)

        name = case_name or self._active_case_name or self.baseline_name
        payload, resolved_project_name = canonical_payload_from_source(
            source,
            project_name=project_name,
            workspace_project_name=self.project_name,
        )

        self.project_name = resolved_project_name
        self._variant_payloads[name] = payload
        self._variant_workflows[name] = ScenarioWorkflowConfig()
        self._invalidate_variant_state(name)

        if activate or self._active_case_name is None:
            self._active_case_name = name

        if build_validation_report(payload).valid:
            return self.case(name)
        return None

    def list_variants(self) -> list[str]:
        """Return the case names in stable insertion order."""
        return list(self._variant_payloads)

    def get_variant_payload(self, name: str) -> JsonDict:
        """Return a defensive copy of one stored payload."""
        return deepcopy(self._get_variant_payload(name))

    def payload_view(self, name: str) -> VariantPayloadView:
        """Return a frontend-friendly editable payload view."""
        payload = self._get_variant_payload(name)
        zone_tree = payload.get("zone_tree")
        return VariantPayloadView(
            variant_name=name,
            zones=zone_tree_view(zone_tree),
            streams=record_views(payload.get("streams"), section="streams"),
            utilities=record_views(payload.get("utilities"), section="utilities"),
            options=json_safe(payload.get("options") or {}),
        )

    def validate_variant(self, name: str):
        """Return a structured validation report for one payload."""
        return build_validation_report(self._get_variant_payload(name))

    def set_variant_payload(
        self,
        name: str,
        payload: TargetInput | JsonDict,
        *,
        base: Optional[str] = None,
    ) -> JsonDict:
        """Create or replace one stored payload."""
        normalized = normalise_payload(payload)
        if base is not None:
            base_payload = self._get_variant_payload(base)
            normalized = merge_payloads(base_payload, normalized)
        self._variant_payloads[name] = normalized
        if name not in self._variant_workflows:
            self._variant_workflows[name] = ScenarioWorkflowConfig()
        if self._active_case_name is None:
            self._active_case_name = name
        self._invalidate_variant_state(name)
        return deepcopy(normalized)

    def solve_variant(
        self,
        name: str,
        *,
        workflow: str = "target",
        workflow_options: Optional[dict[str, Any]] = None,
    ) -> ScenarioVariantView:
        """Solve one case and return a serializable frontend-facing view."""
        payload = self._get_variant_payload(name)
        validation = build_validation_report(payload)
        resolved_options = deepcopy(workflow_options or {})
        support_level = workflow_support_level(workflow)
        warnings_list = workflow_warnings(workflow, support_level)
        self._variant_workflows[name] = ScenarioWorkflowConfig(
            workflow=workflow,
            workflow_options=resolved_options,
        )

        if not validation.valid:
            view = invalid_variant_view(
                variant_name=name,
                workflow=workflow,
                workflow_options=resolved_options,
                validation=validation,
                support_level=support_level,
                warnings_list=warnings_list,
            )
            self._cached_views[name] = view
            return view

        try:
            problem = self.case(name)
            run_problem_workflow(
                problem,
                workflow,
                resolved_options,
                workspace_variant=name,
            )
        except WorkspaceExecutionError as exc:
            view = error_variant_view(
                variant_name=name,
                workflow=workflow,
                workflow_options=resolved_options,
                validation=validation,
                support_level=support_level,
                warnings_list=warnings_list,
                error_message=str(exc),
                error_category=exc.category,
            )
            self._cached_views[name] = view
            return view
        except Exception as exc:
            view = error_variant_view(
                variant_name=name,
                workflow=workflow,
                workflow_options=resolved_options,
                validation=validation,
                support_level=support_level,
                warnings_list=warnings_list,
                error_message=str(exc),
                error_category="unexpected_error",
            )
            self._cached_views[name] = view
            return view

        view = problem_to_variant_view(
            problem,
            variant_name=name,
            workflow=workflow,
            workflow_options=resolved_options,
            validation=validation,
            support_level=support_level,
            warnings_list=warnings_list,
        )
        self._cached_views[name] = view
        self._sync_case_payload(name)
        return view

    def compare_variants(
        self,
        variant_names: Optional[Iterable[str]] = None,
        *,
        base: Optional[str] = None,
    ) -> ScenarioComparisonView:
        """Return deterministic comparison payloads across solved variants."""
        names = list(variant_names or self.list_variants())
        if not names:
            raise ValueError("At least one variant is required for comparison.")

        base_name = base or self.baseline_name
        if base_name not in names:
            names.insert(0, base_name)

        views = {name: self._ensure_solved_view(name) for name in names}
        base_view = views[base_name]
        metric_deltas = []
        problem_diffs = []

        for name in names:
            if name == base_name:
                continue
            metric_deltas.extend(
                summary_metric_deltas(base_name, base_view, name, views[name])
            )
            problem_diffs.extend(
                problem_table_diffs(base_name, base_view, name, views[name])
            )

        return ScenarioComparisonView(
            base_variant=base_name,
            variant_names=names,
            metric_deltas=metric_deltas,
            graph_catalogs={name: views[name].graph_catalog for name in names},
            problem_table_diffs=problem_diffs,
        )

    def list_cases(self) -> list[str]:
        """Return the loaded case names in stable insertion order."""
        return self.list_variants()

    def case(self, name: Optional[str] = None) -> PinchProblem:
        """Return the live :class:`PinchProblem` for one named case."""
        resolved_name = self._resolve_case_name(name)
        cached = self._case_cache.get(resolved_name)
        if cached is not None:
            if self.project_name:
                cached.project_name = self.project_name
            return cached

        payload = deepcopy(self._variant_payloads[resolved_name])
        project_name = self.project_name or project_name_from_payload(payload) or "Site"
        problem = PinchProblem(source=payload, project_name=project_name)
        if self.project_name:
            problem.project_name = self.project_name
        self._case_cache[resolved_name] = problem
        return problem

    def use_case(self, name: str) -> PinchProblem:
        """Activate one named case and return it."""
        self._active_case_name = self._resolve_case_name(name)
        return self.case(self._active_case_name)

    def copy_case(
        self,
        *,
        source_name: str = "baseline",
        new_name: str = "new",
        activate: bool = False,
    ) -> PinchProblem:
        """Clone one existing case into a new named case."""
        data_source = self.get_case_payload(source_name, canonical=True)
        return self.load(data_source, case_name=new_name, activate=activate)

    def get_case_payload(
        self,
        name: Optional[str] = None,
        *,
        canonical: bool = True,
    ) -> JsonDict:
        """Return one case payload, optionally normalised to canonical form."""
        resolved_name = self._resolve_case_name(name)
        if canonical:
            self._sync_case_payload(resolved_name)
        return deepcopy(self._variant_payloads[resolved_name])

    def to_problem_json(
        self,
        *,
        case_name: Optional[str] = None,
        canonical: bool = True,
    ) -> JsonDict:
        """Return the payload for one case using :class:`PinchProblem` naming."""
        return self.get_case_payload(case_name, canonical=canonical)

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
    def problem_data(self):
        """Return the active case payload."""
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
        """Validate one case payload."""
        return self.case(case_name).validate()

    def summary_frame(
        self,
        *,
        case_name: Optional[str] = None,
        detailed: bool = False,
    ) -> pd.DataFrame:
        """Return the solved summary for one case."""
        return self.case(case_name).summary_frame(detailed=detailed)

    def export_excel(
        self,
        results_dir: Optional[PathLike] = None,
        *,
        case_name: Optional[str] = None,
    ) -> Path:
        """Export one case to an Excel workbook."""
        return self.case(case_name).export_excel(results_dir)

    def set_dt_cont_multiplier(
        self,
        value: float,
        *,
        zone_name: Optional[str] = None,
        case_name: Optional[str] = None,
    ):
        """Update one case multiplier and keep the stored payload in sync."""
        resolved_name = self._resolve_case_name(case_name)
        result = self.case(resolved_name).set_dt_cont_multiplier(
            value,
            zone_name=zone_name,
        )
        self._sync_case_payload(resolved_name)
        return result

    def update_options(
        self,
        options: dict[str, Any],
        *,
        case_name: Optional[str] = None,
        replace: bool = False,
    ) -> PinchProblem:
        """Update one case's options and keep the stored payload in sync."""
        resolved_name = self._resolve_case_name(case_name)
        problem = self.case(resolved_name)
        problem.update_options(options, replace=replace)
        self._sync_case_payload(resolved_name)
        return problem

    def show_dashboard(
        self,
        *,
        case_name: Optional[str] = None,
        zone=None,
        graph_payload: Optional[dict[str, Any]] = None,
        page_title: Optional[str] = "OpenPinch Dashboard",
        value_rounding: int = 2,
    ) -> None:
        """Launch the dashboard for one case."""
        self.case(case_name).show_dashboard(
            zone=zone,
            graph_payload=graph_payload,
            page_title=page_title,
            value_rounding=value_rounding,
        )

    def compare_to(
        self,
        other_problem: PinchProblem | "PinchWorkspace",
        *,
        case_name: Optional[str] = None,
        other_case_name: Optional[str] = None,
        target_name: Optional[str] = None,
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
            target_name=target_name,
            base_label=base_label,
            other_label=other_label,
        )

    def compare_cases(
        self,
        base_case: str,
        other_case: str,
        *,
        target_name: Optional[str] = None,
        base_label: Optional[str] = None,
        other_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compare two cases in the same workspace."""
        return self.case(base_case).compare_to(
            self.case(other_case),
            target_name=target_name,
            base_label=base_label or base_case,
            other_label=other_label or other_case,
        )

    def save_bundle(self, path: PathLike) -> Path:
        """Persist the current workspace, syncing any live case edits first."""
        self._sync_all_cases()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        bundle = PinchWorkspaceBundle(
            project_name=self.project_name,
            baseline_name=self.baseline_name,
            variants={
                name: ScenarioVariantBundleEntry(
                    payload=self.get_variant_payload(name),
                    workflow=self._variant_workflows.get(
                        name,
                        ScenarioWorkflowConfig(),
                    ),
                    cached_view=self._cached_views.get(name),
                )
                for name in self.list_variants()
            },
        )
        destination.write_text(
            json.dumps(bundle.model_dump(mode="python"), indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def configuration_field_metadata(cls) -> list[ConfigurationFieldMetadata]:
        """Return declarative metadata for editable configuration fields."""
        return _configuration_field_metadata()

    def _resolve_case_name(self, name: Optional[str]) -> str:
        if name is None:
            default_name = self._default_case_name()
            if default_name is None:
                raise KeyError("No cases are loaded in this PinchWorkspace.")
            return default_name

        if name not in self._variant_payloads:
            available = ", ".join(self.list_cases())
            raise KeyError(f"Unknown case {name!r}. Available cases: {available}")
        return name

    def _default_case_name(self) -> Optional[str]:
        if self._active_case_name in self._variant_payloads:
            return self._active_case_name
        if self.baseline_name in self._variant_payloads:
            self._active_case_name = self.baseline_name
            return self._active_case_name
        if self._variant_payloads:
            self._active_case_name = next(iter(self._variant_payloads))
            return self._active_case_name
        return None

    def _get_variant_payload(self, name: str) -> JsonDict:
        self._sync_case_payload(name)
        try:
            return self._variant_payloads[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown variant {name!r}. Available variants: "
                f"{', '.join(self.list_variants())}"
            ) from exc

    def _ensure_solved_view(self, name: str) -> ScenarioVariantView:
        if name in self._cached_views:
            view = self._cached_views[name]
        else:
            workflow_config = self._variant_workflows.get(
                name,
                ScenarioWorkflowConfig(),
            )
            view = self.solve_variant(
                name,
                workflow=workflow_config.workflow,
                workflow_options=workflow_config.workflow_options,
            )

        if view.status != "solved":
            raise ValueError(
                f"Variant {name!r} is not solved and cannot be compared "
                f"(status={view.status!r})."
            )
        return view

    def _invalidate_variant_state(self, name: str) -> None:
        """Drop cached case and view state for one variant payload."""
        self._cached_views.pop(name, None)
        self._case_cache.pop(name, None)

    def _sync_case_payload(self, name: str) -> None:
        problem = self._case_cache.get(name)
        if problem is None:
            return

        payload = problem.canonical_problem_json()
        if self._variant_payloads.get(name) != payload:
            self._variant_payloads[name] = payload
            self._cached_views.pop(name, None)

    def _sync_all_cases(self) -> None:
        for name in list(self._case_cache):
            self._sync_case_payload(name)


__all__ = ["PinchWorkspace"]
