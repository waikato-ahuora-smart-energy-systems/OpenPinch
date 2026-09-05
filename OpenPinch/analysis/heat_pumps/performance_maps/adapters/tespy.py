"""Optional TESPy HPR simulator for one single-stage refrigerant loop."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import CoolProp
from tespy.components import Compressor, CycleCloser, SimpleHeatExchanger, Valve
from tespy.connections import Connection
from tespy.networks import Network
from tespy.tools.characteristics import CharLine

from OpenPinch.domain.fluids import build_coolprop_abstract_state

from .....contracts.hpr_performance_map import HprPerformanceMapRequest
from ..context import build_hpr_map_generation_context
from ..errors import HprSimulatorFailure
from ..fluids import to_tespy_fluid_token
from ..models import (
    HprMapGenerationContext,
    HprOperatingPoint,
    HprPointSimulation,
    HprSimulatorMetadata,
    HprTargetMapBasis,
)
from ..resources import load_tespy_compressor_characteristic
from ..settings import TESPY_CONVERGENCE_SETTINGS
from ..targeting import PreparedTespyHprTargeting
from ..targeting_models import (
    HprTargetEvaluatorError,
    HprTargetEvaluatorMetadata,
    HprTargetThermodynamicRequest,
    HprTargetThermodynamicResult,
    HprThermalProfilePoint,
)

_WATTS_PER_KILOWATT = 1_000.0


def _tespy_version() -> str:
    try:
        return version("tespy")
    except PackageNotFoundError:
        return "unknown"


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except TypeError, ValueError:
        return None
    return result if math.isfinite(result) else None


def _result_value(container: object) -> float | None:
    value = getattr(container, "val_SI", None)
    if value is None:
        value = getattr(container, "val", None)
    return _finite_float(value)


def _iteration_count(network: Network) -> int | None:
    problem = getattr(network, "problem", None)
    value = getattr(problem, "iter", None)
    if value is None:
        value = getattr(network, "iter", None)
    result = _finite_float(value)
    return None if result is None else int(result)


def _network_converged(network: Network) -> bool:
    if not bool(getattr(network, "converged", False)):
        return False
    problem = getattr(network, "problem", None)
    linearly_dependent = getattr(problem, "lin_dep", None)
    if linearly_dependent is None:
        linearly_dependent = getattr(network, "lin_dep", False)
    return not bool(linearly_dependent)


class TespyHprPointSimulator:
    """One isolated TESPy design/offdesign simulation session."""

    def __init__(self) -> None:
        self._context: HprMapGenerationContext | None = None
        self._state = "fresh"
        self._temporary_directory: TemporaryDirectory[str] | None = None
        self._design_state_path: Path | None = None
        self._design_restore_count = 0
        self._network: Network | None = None
        self._compressor: Compressor | None = None
        self._condenser: SimpleHeatExchanger | None = None
        self._evaporator: SimpleHeatExchanger | None = None
        self._compressor_inlet: Connection | None = None
        self._condenser_outlet: Connection | None = None

    @staticmethod
    def _saturation_pressures(
        context: HprMapGenerationContext,
        evaporating_temperature: float,
        condensing_temperature: float,
    ) -> tuple[float, float]:
        state = build_coolprop_abstract_state(context.working_fluid.source_spec)
        state.update(
            CoolProp.QT_INPUTS,
            1.0,
            evaporating_temperature + 273.15,
        )
        evaporating_pressure = float(state.p())
        state.update(
            CoolProp.QT_INPUTS,
            0.0,
            condensing_temperature + 273.15,
        )
        condensing_pressure = float(state.p())
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (evaporating_pressure, condensing_pressure)
        ):
            raise ValueError("saturation pressures must be finite and positive")
        return evaporating_pressure, condensing_pressure

    def _build_network(self, context: HprMapGenerationContext) -> None:
        characteristic = load_tespy_compressor_characteristic()
        x_values, y_values = zip(*characteristic.points, strict=True)

        network = Network()
        cycle_closer = CycleCloser("cycle closer")
        compressor = Compressor("compressor")
        condenser = SimpleHeatExchanger("condenser")
        valve = Valve("expansion valve")
        evaporator = SimpleHeatExchanger("evaporator")
        compressor_inlet = Connection(
            cycle_closer,
            "out1",
            compressor,
            "in1",
            label="compressor inlet",
        )
        compressor_outlet = Connection(
            compressor,
            "out1",
            condenser,
            "in1",
            label="compressor outlet",
        )
        condenser_outlet = Connection(
            condenser,
            "out1",
            valve,
            "in1",
            label="condenser outlet",
        )
        valve_outlet = Connection(
            valve,
            "out1",
            evaporator,
            "in1",
            label="valve outlet",
        )
        evaporator_outlet = Connection(
            evaporator,
            "out1",
            cycle_closer,
            "in1",
            label="evaporator outlet",
        )
        network.add_conns(
            compressor_inlet,
            compressor_outlet,
            condenser_outlet,
            valve_outlet,
            evaporator_outlet,
        )
        compressor.set_attr(
            eta_s=context.basis.compressor_isentropic_efficiency,
            eta_s_char=CharLine(x=list(x_values), y=list(y_values)),
            design=["eta_s"],
            offdesign=["eta_s_char"],
        )
        condenser.set_attr(pr=1.0)
        evaporator.set_attr(pr=1.0)
        compressor_inlet.set_attr(
            fluid={to_tespy_fluid_token(context.working_fluid): 1.0}
        )

        self._network = network
        self._compressor = compressor
        self._condenser = condenser
        self._evaporator = evaporator
        self._compressor_inlet = compressor_inlet
        self._condenser_outlet = condenser_outlet

    def _apply_condition(
        self,
        *,
        evaporating_temperature: float,
        condensing_temperature: float,
        useful_duty: float,
    ) -> None:
        assert self._context is not None
        assert self._compressor_inlet is not None
        assert self._condenser_outlet is not None
        assert self._condenser is not None
        assert self._evaporator is not None
        context = self._context
        evaporating_pressure, condensing_pressure = self._saturation_pressures(
            context,
            evaporating_temperature,
            condensing_temperature,
        )
        self._compressor_inlet.set_attr(
            p=evaporating_pressure,
            T=(evaporating_temperature + context.basis.superheat + 273.15),
        )
        self._condenser_outlet.set_attr(
            p=condensing_pressure,
            T=(condensing_temperature - context.basis.subcooling + 273.15),
        )
        if context.basis.mode == "heat_pump":
            self._evaporator.set_attr(Q=None)
            self._condenser.set_attr(Q=-useful_duty * _WATTS_PER_KILOWATT)
        else:
            self._condenser.set_attr(Q=None)
            self._evaporator.set_attr(Q=useful_duty * _WATTS_PER_KILOWATT)

    def _solve(self, mode: str, **paths: Any) -> None:
        assert self._network is not None
        self._network.solve(
            mode,
            **paths,
            **dict(TESPY_CONVERGENCE_SETTINGS.solve_arguments),
        )

    def prepare(self, context: HprMapGenerationContext) -> HprSimulatorMetadata:
        if self._state != "fresh":
            raise HprSimulatorFailure(
                "prepare_failed",
                "TESPy simulator can only be prepared once",
                session_fatal=True,
            )
        if context.basis.simulation_backend.strip().lower() != "tespy":
            raise HprSimulatorFailure(
                "prepare_failed",
                "TESPy simulator received a context for a different backend",
                session_fatal=True,
            )
        if context.basis.internal_hx_gas_temperature_change != 0.0:
            raise HprSimulatorFailure(
                "unsupported_model",
                (
                    "the TESPy single-stage model does not include an internal "
                    "heat exchanger"
                ),
                session_fatal=True,
            )

        self._context = context
        self._temporary_directory = TemporaryDirectory(prefix="openpinch-hpr-tespy-")
        self._design_state_path = (
            Path(self._temporary_directory.name) / "design-state.json"
        )
        characteristic = load_tespy_compressor_characteristic()
        try:
            self._build_network(context)
            self._apply_condition(
                evaporating_temperature=(context.basis.nominal_evaporating_temperature),
                condensing_temperature=(context.basis.nominal_condensing_temperature),
                useful_duty=context.reference_capacity,
            )
        except Exception as exc:
            self._state = "fatal"
            raise HprSimulatorFailure(
                "unsupported_working_fluid",
                "TESPy could not construct the selected working-fluid state",
                session_fatal=True,
                cause=exc,
            ) from exc

        try:
            self._solve("design")
        except Exception as exc:
            self._state = "fatal"
            raise HprSimulatorFailure(
                "prepare_failed",
                "TESPy raised while solving the design condition",
                session_fatal=True,
                cause=exc,
            ) from exc
        assert self._network is not None
        if not _network_converged(self._network):
            self._state = "fatal"
            raise HprSimulatorFailure(
                "non_converged",
                "TESPy did not converge at the design condition",
                session_fatal=True,
            )
        try:
            self._network.save(self._design_state_path)
        except Exception as exc:
            self._state = "fatal"
            raise HprSimulatorFailure(
                "prepare_failed",
                "TESPy could not persist its private design state",
                session_fatal=True,
                cause=exc,
            ) from exc
        if not self._design_state_path.is_file():
            self._state = "fatal"
            raise HprSimulatorFailure(
                "prepare_failed",
                "TESPy did not create its private design state",
                session_fatal=True,
            )

        self._state = "prepared"
        return HprSimulatorMetadata(
            backend="tespy",
            engine_version=_tespy_version(),
            model_id=context.basis.model_id,
            characteristic_set_id=characteristic.characteristic_set_id,
            design_converged=True,
            design_details={
                "topology": "single_stage_refrigerant_only",
                "convergence_settings": (TESPY_CONVERGENCE_SETTINGS.as_provenance()),
                "characteristic_abscissa": characteristic.abscissa,
                "characteristic_ordinate": characteristic.ordinate,
                "characteristic_points": [
                    [x_value, y_value] for x_value, y_value in characteristic.points
                ],
                "characteristic_sha256": characteristic.sha256,
                "design_iterations": _iteration_count(self._network),
                "fixed_external_approaches": True,
                "heat_exchanger_characteristics": "not_applicable",
            },
        )

    def simulate(self, point: HprOperatingPoint) -> HprPointSimulation:
        if self._state != "prepared" or self._context is None:
            raise HprSimulatorFailure(
                "point_exception",
                "TESPy simulator is not in a prepared state",
                session_fatal=True,
            )
        if self._design_state_path is None or not self._design_state_path.is_file():
            self._state = "fatal"
            raise HprSimulatorFailure(
                "restore_failed",
                "TESPy private design state is unavailable",
                session_fatal=True,
            )

        try:
            self._apply_condition(
                evaporating_temperature=point.evaporating_temperature,
                condensing_temperature=point.condensing_temperature,
                useful_duty=point.requested_useful_duty,
            )
        except Exception as exc:
            raise HprSimulatorFailure(
                "unsupported_working_fluid",
                "TESPy could not construct the requested working-fluid state",
                session_fatal=False,
                cause=exc,
            ) from exc

        snapshot: str | Path
        if self._design_restore_count % 2:
            snapshot = str(self._design_state_path)
        else:
            snapshot = self._design_state_path
        self._design_restore_count += 1
        try:
            self._solve(
                "offdesign",
                design_path=snapshot,
                init_path=snapshot,
            )
        except FileNotFoundError as exc:
            self._state = "fatal"
            raise HprSimulatorFailure(
                "restore_failed",
                "TESPy could not restore its private design state",
                session_fatal=True,
                cause=exc,
            ) from exc
        except Exception as exc:
            raise HprSimulatorFailure(
                "point_exception",
                "TESPy raised at the requested offdesign condition",
                session_fatal=False,
                cause=exc,
            ) from exc

        assert self._network is not None
        if not _network_converged(self._network):
            raise HprSimulatorFailure(
                "non_converged",
                "TESPy did not converge at the requested offdesign condition",
                session_fatal=False,
            )
        assert self._compressor is not None
        assert self._condenser is not None
        assert self._evaporator is not None
        q_source_w = _result_value(self._evaporator.Q)
        q_sink_w = _result_value(self._condenser.Q)
        compressor_power_w = _result_value(self._compressor.P)
        if any(value is None for value in (q_source_w, q_sink_w, compressor_power_w)):
            raise HprSimulatorFailure(
                "non_converged",
                "TESPy did not return finite offdesign duties and power",
                session_fatal=False,
            )
        return HprPointSimulation(
            q_source=abs(q_source_w) / _WATTS_PER_KILOWATT,
            q_sink=abs(q_sink_w) / _WATTS_PER_KILOWATT,
            compressor_power=compressor_power_w / _WATTS_PER_KILOWATT,
            converged=True,
            engine_details={
                "iterations": _iteration_count(self._network),
                "design_restore_count": self._design_restore_count,
            },
        )

    def close(self) -> None:
        if self._state == "closed":
            raise HprSimulatorFailure(
                "cleanup_failed",
                "TESPy simulator was closed more than once",
                session_fatal=True,
            )
        temporary_directory = self._temporary_directory
        self._context = None
        self._network = None
        self._compressor = None
        self._condenser = None
        self._evaporator = None
        self._compressor_inlet = None
        self._condenser_outlet = None
        self._design_state_path = None
        self._temporary_directory = None
        self._state = "closed"
        if temporary_directory is not None:
            temporary_directory.cleanup()


class TespyHprTargetEvaluator:
    """Evaluate every optimizer candidate as one fresh TESPy design solve."""

    def __init__(self, prepared: PreparedTespyHprTargeting) -> None:
        self._prepared = prepared
        self._state = "created"
        self._design_solve_count = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def design_solve_count(self) -> int:
        return self._design_solve_count

    def open(self) -> HprTargetEvaluatorMetadata:
        if self._state != "created":
            raise HprTargetEvaluatorError(
                "evaluator_open_failure",
                "TESPy target evaluator can open only once",
                session_fatal=True,
            )
        characteristic = load_tespy_compressor_characteristic()
        self._state = "ready"
        return HprTargetEvaluatorMetadata(
            backend="tespy",
            engine_version=_tespy_version(),
            model_id=self._prepared.model_id,
            assumptions={
                "topology": "single_stage_refrigerant_only",
                "candidate_solve_mode": "design",
                "convergence_settings": TESPY_CONVERGENCE_SETTINGS.as_provenance(),
                "characteristic_set_id": characteristic.characteristic_set_id,
                "characteristic_sha256": characteristic.sha256,
                "fixed_external_approaches": True,
                "modeled_auxiliaries": [],
            },
        )

    def evaluate(
        self,
        request: HprTargetThermodynamicRequest,
    ) -> HprTargetThermodynamicResult:
        if self._state != "ready":
            raise HprTargetEvaluatorError(
                "evaluator_state_failure",
                "TESPy target evaluator is not ready",
                session_fatal=True,
            )
        self._validate_request_identity(request)
        self._state = "evaluating"
        simulator = TespyHprPointSimulator()
        result: HprTargetThermodynamicResult | None = None
        local_failure: HprTargetEvaluatorError | None = None
        try:
            result = self._evaluate_fresh_design(simulator, request)
        except HprTargetEvaluatorError as exc:
            local_failure = exc
        except Exception as exc:
            local_failure = HprTargetEvaluatorError(
                "candidate_design_failure",
                "TESPy could not evaluate the nominal target candidate",
                session_fatal=False,
                cause=exc,
            )

        try:
            simulator.close()
        except Exception as exc:
            self._state = "fatal"
            raise HprTargetEvaluatorError(
                "cleanup_failure",
                "TESPy candidate resources could not be released",
                session_fatal=True,
                cause=exc,
            ) from exc

        self._state = "ready"
        if local_failure is not None:
            raise local_failure
        assert result is not None
        return result

    def _validate_request_identity(
        self,
        request: HprTargetThermodynamicRequest,
    ) -> None:
        if (
            request.mode != self._prepared.mode
            or request.cycle_id != self._prepared.cycle_id
            or request.model_id != self._prepared.model_id
            or request.working_fluid != self._prepared.working_fluid
        ):
            self._state = "fatal"
            raise HprTargetEvaluatorError(
                "candidate_request_mismatch",
                "TESPy candidate identity does not match prepared targeting intent",
                session_fatal=True,
            )
        if request.internal_hx_gas_temperature_change != 0.0:
            self._state = "fatal"
            raise HprTargetEvaluatorError(
                "unsupported_model",
                (
                    "TESPy single-stage targeting does not model an internal "
                    "heat exchanger"
                ),
                session_fatal=True,
            )

    def _evaluate_fresh_design(
        self,
        simulator: TespyHprPointSimulator,
        request: HprTargetThermodynamicRequest,
    ) -> HprTargetThermodynamicResult:
        context = _target_generation_context(request)
        simulator._context = context
        simulator._build_network(context)
        simulator._apply_condition(
            evaporating_temperature=request.evaporating_temperature,
            condensing_temperature=request.condensing_temperature,
            useful_duty=request.useful_duty,
        )
        self._design_solve_count += 1
        simulator._solve("design")
        if simulator._network is None or not _network_converged(simulator._network):
            raise HprTargetEvaluatorError(
                "non_converged",
                "TESPy did not converge for the nominal target candidate",
                session_fatal=False,
            )
        if (
            simulator._compressor is None
            or simulator._condenser is None
            or simulator._evaporator is None
        ):
            raise HprTargetEvaluatorError(
                "candidate_result_unavailable",
                "TESPy target candidate result components are unavailable",
                session_fatal=False,
            )
        q_source_w = _result_value(simulator._evaporator.Q)
        q_sink_w = _result_value(simulator._condenser.Q)
        compressor_power_w = _result_value(simulator._compressor.P)
        if any(value is None for value in (q_source_w, q_sink_w, compressor_power_w)):
            raise HprTargetEvaluatorError(
                "candidate_result_unavailable",
                "TESPy did not return finite target duties and compressor power",
                session_fatal=False,
            )
        q_source = abs(q_source_w) / _WATTS_PER_KILOWATT
        q_sink = abs(q_sink_w) / _WATTS_PER_KILOWATT
        compressor_power = compressor_power_w / _WATTS_PER_KILOWATT
        useful_duty = q_sink if request.mode == "heat_pump" else q_source
        cop = useful_duty / compressor_power if compressor_power > 0.0 else math.inf
        return HprTargetThermodynamicResult(
            backend="tespy",
            model_id=request.model_id,
            working_fluid=request.working_fluid,
            converged=True,
            q_source=q_source,
            q_sink=q_sink,
            compressor_power=compressor_power,
            cop=cop,
            source_profile=(
                HprThermalProfilePoint(
                    temperature=(
                        request.evaporating_temperature
                        + request.source_approach_temperature
                    ),
                    enthalpy=0.0,
                ),
                HprThermalProfilePoint(
                    temperature=request.evaporating_temperature,
                    enthalpy=q_source,
                ),
            ),
            sink_profile=(
                HprThermalProfilePoint(
                    temperature=(
                        request.condensing_temperature
                        - request.sink_approach_temperature
                    ),
                    enthalpy=0.0,
                ),
                HprThermalProfilePoint(
                    temperature=request.condensing_temperature,
                    enthalpy=q_sink,
                ),
            ),
            engine_version=_tespy_version(),
            design_details={
                "iterations": _iteration_count(simulator._network),
                "solve_mode": "design",
                "convergence_settings": TESPY_CONVERGENCE_SETTINGS.as_provenance(),
            },
        )

    def close(self) -> None:
        if self._state == "closed":
            raise HprTargetEvaluatorError(
                "cleanup_failure",
                "TESPy target evaluator was closed more than once",
                session_fatal=True,
            )
        self._state = "closed"


def _target_generation_context(
    request: HprTargetThermodynamicRequest,
) -> HprMapGenerationContext:
    basis = HprTargetMapBasis(
        target_id=request.candidate_id,
        mode=request.mode,
        simulation_backend="tespy",
        cycle_id=request.cycle_id,
        model_id=request.model_id,
        refrigerant_spec=request.working_fluid.source_spec,
        nominal_evaporating_temperature=request.evaporating_temperature,
        nominal_condensing_temperature=request.condensing_temperature,
        nominal_useful_duty=request.useful_duty,
        source_approach_temperature=request.source_approach_temperature,
        sink_approach_temperature=request.sink_approach_temperature,
        compressor_isentropic_efficiency=(request.compressor_isentropic_efficiency),
        superheat=request.superheat,
        subcooling=request.subcooling,
        internal_hx_gas_temperature_change=(request.internal_hx_gas_temperature_change),
        source_provenance={"purpose": "targeting_candidate_design"},
    )
    external_source = (
        request.evaporating_temperature + request.source_approach_temperature
    )
    external_sink = request.condensing_temperature - request.sink_approach_temperature
    map_request = HprPerformanceMapRequest(
        map_id=f"target-candidate-{request.candidate_id}",
        source_temperatures=(external_source,),
        sink_temperatures=(external_sink,),
        load_fractions=(1.0,),
        reference_capacity=request.useful_duty,
    )
    return build_hpr_map_generation_context(basis, map_request)


__all__ = ["TespyHprPointSimulator", "TespyHprTargetEvaluator"]
