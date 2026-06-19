"""Internal HEN problem shell behind the OpenPinch synthesis service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ....classes.heat_exchanger_network import HeatExchangerNetwork
from ....lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
from ..array_adapter import PreparedSolverArrays
from ..pinch_decomposition import PinchDecompositionSnapshot
from .extraction import extract_heat_exchanger_network, extract_network_synthesis_result
from .stagewise import StageWiseModel

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
    solver_options: Mapping[str, Any] | Sequence[str] | None = None
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
            "solver_options": self.solver_options,
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

        try:
            self.load_model(model_factories=model_factories)
            if self.framework == "PDM":
                self._solve_pdm(print_output=print_output)
            else:
                self.case.optimise(print_output=print_output)
            if self.framework in {"PDM", "TDM"}:
                self.case = self.remove_unused_stages(self.case)
            if evolution:
                evolved = self.case.get_net_benefit_evolution(
                    print_output=print_output
                )
                if evolved is not None:
                    self.case = evolved
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
        """Solve PDM sides and amalgamate them before stage reduction."""

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

    def remove_unused_stages(self, case) -> StageWiseModel:
        """Apply the source stage-utilisation reduction after PDM/TDM solves."""

        if case.mSuccess != 1:
            return case

        active_stages = self._get_active_stages(case)
        if len(active_stages) == case.S:
            return case

        active_locations = [
            [[None for _k in range(len(active_stages))] for _j in range(case.J)]
            for _i in range(case.I)
        ]
        for new_k, old_k in enumerate(active_stages):
            for i in range(case.I):
                for j in range(case.J):
                    active_locations[i][j][new_k] = (
                        1 if case.Q_r[i][j][old_k][0] > case.tol else 0
                    )

        f_case = StageWiseModel(
            name=f"reduced-{case.name}",
            framework=case.framework,
            solver="apopt",
            solver_arrays=case.solver_arrays,
            stages=len(active_stages),
            dTmin=case.dTmin,
            z_restriction=[active_locations, None, None],
            min_dqda=case.min_dqda,
            minimisation_goal=case.minimisation_goal,
            non_isothermal_model=case.non_isothermal_model,
            integers=False,
            tol=case.tol,
        )

        for new_k, old_k in enumerate(active_stages):
            for i in range(case.I):
                for j in range(case.J):
                    q_val = (
                        case.Q_r[i][j][old_k][0]
                        if case.Q_r[i][j][old_k][0] > case.tol
                        else 0.0
                    )
                    _assign(f_case.Q_r[i][j][new_k], q_val)
                    _assign_binary(f_case.z[i][j][new_k], q_val)
                    _assign(f_case.theta_1[i][j][new_k], case.theta_1[i][j][old_k][0])
                    _assign(f_case.theta_2[i][j][new_k], case.theta_2[i][j][old_k][0])

                    if case.non_isothermal_model:
                        _assign(f_case.X[i][j][new_k], case.X[i][j][old_k][0])
                        _assign(f_case.Y[j][i][new_k], case.Y[j][i][old_k][0])
                        _assign(
                            f_case.T_h_out_x[i][j][new_k],
                            case.T_h_out_x[i][j][old_k][0],
                        )
                        _assign(
                            f_case.T_c_out_y[j][i][new_k],
                            case.T_c_out_y[j][i][old_k][0],
                        )

        reduced_boundaries = [0] + [stage + 1 for stage in active_stages]
        for i in range(case.I):
            for new_k, old_k in enumerate(reduced_boundaries):
                _assign(f_case.T_h[i][new_k], case.T_h[i][old_k][0])

        for j in range(case.J):
            for new_k, old_k in enumerate(reduced_boundaries):
                _assign(f_case.T_c[j][new_k], case.T_c[j][old_k][0])

        for i in range(case.I):
            _assign(f_case.Q_c[i], case.Q_c[i][0])
            _assign_binary(f_case.z_cu[i], case.Q_c[i][0])

        for j in range(case.J):
            _assign(f_case.Q_h[j], case.Q_h[j][0])
            _assign_binary(f_case.z_hu[j], case.Q_h[j][0])

        f_case.TAC_model = case.TAC_model
        f_case.TAC = case.TAC
        f_case.solve_time = case.solve_time
        f_case.mSuccess = case.mSuccess

        f_case.optimise(print_output=False)
        if f_case.mSuccess != 1:
            return case
        return f_case

    def _get_active_stages(self, case) -> list[int]:
        q_total = sum(
            case.Q_r[i][j][k][0]
            for i in range(case.I)
            for j in range(case.J)
            for k in range(case.S)
        )
        q_per_stage = [
            sum(
                case.Q_r[i][j][k][0]
                for i in range(case.I)
                for j in range(case.J)
            )
            for k in range(case.S)
        ]
        threshold = self._utilisation_threshold(case.S)
        return [
            k
            for k, q_stage in enumerate(q_per_stage)
            if q_stage / q_total * 100 >= threshold
        ]

    def _utilisation_threshold(self, stage_count: int) -> float:
        if stage_count <= 3:
            return 0.1
        if stage_count <= 5:
            return 5.0
        if stage_count <= 8:
            return 8.0
        if stage_count <= 10:
            return 10.0
        return 0.0

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
            return StageWiseModel
        raise ValueError(f"Unknown HEN model factory {default!r}.")


def _assign(variable: Any, value: float) -> None:
    if type(variable).__name__ != "GKParameter":
        value = max(variable.lower, min(variable.upper, value))
    variable.VALUE.value = value


def _assign_binary(variable: Any, value: float) -> None:
    variable.VALUE.value = 1 if value > 0 else 0
