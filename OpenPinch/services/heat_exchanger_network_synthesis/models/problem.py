"""Internal HEN problem shell behind the OpenPinch synthesis service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ....classes.heat_exchanger_network import HeatExchangerNetwork
from ....lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from ..array_adapter import PreparedSolverArrays
from .extraction import extract_heat_exchanger_network, extract_network_synthesis_result

FrameworkName = Literal["PDM", "TDM", "ESM"]


class ModelSliceUnavailableError(NotImplementedError):
    """Raised when HENS-06 is asked to run deferred equation-model slices."""


@dataclass
class InternalHeatExchangerNetworkProblem:
    """OpenPinch-owned replacement for the source ``HeatExchangerNetworkProblem``.

    This object is private solver state. HENS-06 records the source
    task-to-model arguments and emits OpenPinch network/result payloads at the
    extraction boundary. Concrete PDM/TDM/ESM construction and solving stay
    unavailable until the later migration slices move their source semantics.
    """

    solver_arrays: PreparedSolverArrays
    name: str = ""
    framework: FrameworkName = "TDM"
    solver: str = "couenne"
    dTmin: float = 0.1
    min_dqda: float = 0.0
    z_restriction: list | None = None
    minimisation_goal: str = "hot utility"
    non_isothermal_model: bool = False
    integers: bool = True
    parent: "InternalHeatExchangerNetworkProblem | None" = None
    tol: float = 1e-3
    stage_selection: str | list[str] = "automated"
    stages: int | None = None
    synthesis_task_id: str | None = None

    def load_model(
        self,
        *,
        model_factories: Mapping[str, Any] | None = None,
    ) -> None:
        """Reject concrete model loading until deferred source semantics move."""

        self._raise_deferred_model_error(model_factories=model_factories)

    def get_solution(
        self,
        *,
        print_output: bool = True,
        evolution: bool | None = None,
        model_factories: Mapping[str, Any] | None = None,
    ) -> Any:
        """Reject concrete solves until PDM/stagewise behavior is fully moved."""

        del print_output, evolution
        self._raise_deferred_model_error(model_factories=model_factories)

    def extract_network(self, *, run_id: str) -> HeatExchangerNetwork:
        """Convert the solved private case into an OpenPinch network result."""

        return extract_heat_exchanger_network(
            self.case,
            self.solver_arrays,
            run_id=run_id,
            task_id=self.synthesis_task_id,
            method=self.framework,
            stage_count=getattr(self.case, "S", self.stages),
        )

    def extract_result(
        self,
        *,
        run_id: str,
        problem_id: str | None = None,
        workspace_variant: str | None = None,
        state_id: str | None = None,
    ) -> HeatExchangerNetworkSynthesisResult:
        """Return the serializable result payload for the service boundary."""

        return extract_network_synthesis_result(
            self.case,
            self.solver_arrays,
            run_id=run_id,
            task_id=self.synthesis_task_id,
            problem_id=problem_id,
            workspace_variant=workspace_variant,
            state_id=state_id,
            solver_name=self.solver,
            method=self.framework,
            stage_count=getattr(self.case, "S", self.stages),
        )

    def _raise_deferred_model_error(
        self,
        *,
        model_factories: Mapping[str, Any] | None,
    ) -> None:
        if model_factories:
            attempted = ", ".join(sorted(model_factories))
            detail = f" Factory registrations were ignored: {attempted}."
        else:
            detail = ""
        raise ModelSliceUnavailableError(
            "HENS-06 moved only the base equation-kernel boundary, backend "
            "guards, private problem shell, and solution extraction. Concrete "
            "PDM/TDM/ESM model construction and solving remain unavailable "
            "until HENS-07 moves the source PDM/stagewise semantics and "
            "HENS-08 moves stage reduction/topology evolution unchanged."
            f"{detail}"
        )
