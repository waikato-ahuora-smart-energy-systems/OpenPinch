"""Internal HEN problem shell behind the OpenPinch synthesis service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ....classes.heat_exchanger_network import HeatExchangerNetwork
from ....lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from ..array_adapter import PreparedSolverArrays
from ..pinch_decomposition import PinchDecompositionSnapshot
from .extraction import extract_heat_exchanger_network, extract_network_synthesis_result

FrameworkName = Literal["PDM", "TDM", "ESM"]


class ModelSliceUnavailableError(NotImplementedError):
    """Raised when a later migration slice is asked to run too early."""


@dataclass
class InternalHeatExchangerNetworkProblem:
    """OpenPinch-owned replacement for source ``HeatExchangerNetworkProblem``.

    This object is private solver state. HENS-07 constructs moved PDM and
    StageWise models from OpenPinch-prepared solver arrays and emits OpenPinch
    network/result payloads at the extraction boundary. HENS-08 still owns
    stage reduction and topology evolution.
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
    pinch_snapshots: Mapping[str, PinchDecompositionSnapshot] | None = None

    def load_model(
        self,
        *,
        model_factories: Mapping[str, Any] | None = None,
    ) -> None:
        """Construct the private PDM or StageWise model for this task."""

        self.args = {
            "name": self.name,
            "framework": self.framework,
            "solver": self.solver,
            "solver_arrays": self.solver_arrays,
            "dTmin": self.dTmin,
            "min_dqda": self.min_dqda,
            "z_restriction": self.z_restriction,
            "minimisation_goal": self.minimisation_goal,
            "non_isothermal_model": self.non_isothermal_model,
            "integers": self.integers,
            "tol": self.tol,
        }
        if self.framework == "PDM":
            self._build_pdm(model_factories=model_factories)
            return
        self._build_stage_wise(model_factories=model_factories)

    def get_solution(
        self,
        *,
        print_output: bool = True,
        evolution: bool | None = None,
        model_factories: Mapping[str, Any] | None = None,
    ) -> Any:
        """Load, solve, and return the private solved model for this task."""

        if evolution:
            raise ModelSliceUnavailableError(
                "HENS-07 moved PDM and StageWise model construction only. "
                "Stage reduction and topology evolution are reserved for HENS-08."
            )
        try:
            self.load_model(model_factories=model_factories)
            if self.framework == "PDM":
                self._solve_pdm(print_output=print_output)
            else:
                self.case.optimise(print_output=print_output)
            return self.case
        except ValueError as exc:
            self.solution_failure_reason = str(exc)
            return None

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

    def _build_pdm(
        self,
        *,
        model_factories: Mapping[str, Any] | None,
    ) -> None:
        """Construct source PDM above/below models with private snapshots."""

        snapshots = dict(self.pinch_snapshots or {})
        if "above" not in snapshots or "below" not in snapshots:
            raise ValueError(
                "PDM construction requires above and below pinch decomposition "
                "snapshots from the OpenPinch targeting boundary."
            )
        factory = self._model_factory(
            model_factories,
            "pinch_decomposition",
            default="PinchDecompModel",
        )
        base_args = dict(self.args)
        self.above = factory(
            **(
                base_args
                | {
                    "name": f"above pinch {self.dTmin}",
                    "pinch_loc": "above",
                    "minimisation_goal": "hot utility",
                    "pinch_snapshot": snapshots["above"],
                    "stage_selection": self.stage_selection,
                }
            )
        )
        self.below = factory(
            **(
                base_args
                | {
                    "name": f"below pinch {self.dTmin}",
                    "pinch_loc": "below",
                    "minimisation_goal": "cold utility",
                    "pinch_snapshot": snapshots["below"],
                    "stage_selection": self.stage_selection,
                }
            )
        )

    def _build_stage_wise(
        self,
        *,
        model_factories: Mapping[str, Any] | None,
    ) -> None:
        """Construct a TDM/ESM model from explicit or parent stage state."""

        stages = self.stages
        if stages is None and self.parent is not None:
            stages = getattr(
                self.parent.case,
                "stages",
                getattr(self.parent.case, "S", None),
            )
        if stages is None:
            raise ValueError(
                "Stage-wise models require an explicit stage count or a parent "
                "solution."
            )
        factory = self._model_factory(
            model_factories,
            "stagewise",
            default="StageWiseModel",
        )
        self.case = factory(**self.args, stages=stages)
        if self.parent is not None:
            self.case.set_initial_values_for_variables(self.parent.case)

    def _solve_pdm(self, *, print_output: bool = True) -> None:
        """Solve PDM sides and amalgamate them without HENS-08 reduction."""

        if self.above.HU_target > 0:
            self.above.optimise(print_output=print_output)
        if self.below.CU_target > 0:
            self.below.optimise(print_output=print_output)
        self.case = self.above.amalgamate_networks(
            below_case=self.below,
            above_case=self.above,
        )
        self.case.get_post_process()
        if print_output:
            self.case.output_to_cmd_line()
        self.args.update(
            {
                "name": f"PDM amalgamated {self.dTmin}",
                "minimisation_goal": "total utility",
                "tol": self.tol,
            }
        )

    def _model_factory(
        self,
        model_factories: Mapping[str, Any] | None,
        factory_name: str,
        *,
        default: str,
    ) -> Any:
        if model_factories and factory_name in model_factories:
            return model_factories[factory_name]
        if default == "PinchDecompModel":
            from .pinch_decomposition import PinchDecompModel

            return PinchDecompModel
        if default == "StageWiseModel":
            from .stagewise import StageWiseModel

            return StageWiseModel
        raise ValueError(f"Unknown HEN model factory {default!r}.")
