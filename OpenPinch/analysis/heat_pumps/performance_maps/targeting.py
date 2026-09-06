"""Engine-neutral helpers shared by HPR targeting backend integrations."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal, cast

from CoolProp.CoolProp import PropsSI

from ....contracts.hpr import HeatPumpTargetInputs, MultiPeriodHPRTargetInputs
from ....domain.enums import HeatPumpAndRefrigerationCycle
from .fluids import (
    resolve_hpr_working_fluid,
    to_tespy_fluid_token,
)
from .models import HprWorkingFluidSpec
from .targeting_models import (
    HprTargetEvaluationFailure,
    HprTargetEvaluator,
    HprTargetEvaluatorError,
    HprTargetEvaluatorMetadata,
    HprTargetThermodynamicRequest,
    HprTargetThermodynamicResult,
    HprThermalProfilePoint,
)

HprSimulationBackend = Literal["coolprop", "tespy"]
_SUPPORTED_HPR_SIMULATION_BACKENDS = frozenset(("coolprop", "tespy"))


class HprTargetCompatibilityError(ValueError):
    """Stable fail-fast error for unsupported HPR targeting selections."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PreparedTespyHprTargeting:
    """Validated scalar TESPy topology and working-fluid preparation facts."""

    simulation_backend: Literal["tespy"]
    mode: Literal["heat_pump", "refrigeration"]
    cycle_id: Literal["single_stage_vapour_compression"]
    model_id: str
    working_fluid: HprWorkingFluidSpec
    tespy_fluid_token: str
    evaporator_count: int
    condenser_count: int
    period_idx: int
    probe_evaporating_temperature: float
    probe_condensing_temperature: float
    power_boundary: Literal["compressor_only"] = "compressor_only"


@dataclass(frozen=True, slots=True)
class HprTargetValidationTolerances:
    """Separately named scalar tolerances for candidate-result acceptance."""

    energy_balance_absolute: float = 1e-5
    energy_balance_relative: float = 1e-8
    useful_duty_absolute: float = 1e-5
    useful_duty_relative: float = 1e-8
    cop_absolute: float = 1e-6
    cop_relative: float = 1e-8
    profile_duty_absolute: float = 1e-5
    profile_duty_relative: float = 1e-8
    profile_order_absolute: float = 1e-9


@dataclass(frozen=True, slots=True)
class HprTargetCacheStats:
    """Immutable snapshot of one call-local exact candidate cache."""

    callbacks: int
    hits: int
    misses: int
    solves: int
    insertions: int
    evictions: int


class HprTargetEvaluatorCoordinator:
    """Own one evaluator lifecycle and partition local from fatal failures."""

    def __init__(self, evaluator: HprTargetEvaluator) -> None:
        self._evaluator = evaluator
        self._state = "created"
        self._metadata: HprTargetEvaluatorMetadata | None = None
        self._close_attempted = False
        self._cache: OrderedDict[
            HprTargetThermodynamicRequest,
            HprTargetThermodynamicResult | HprTargetEvaluationFailure,
        ] = OrderedDict()
        self._callbacks = 0
        self._hits = 0
        self._misses = 0
        self._solves = 0
        self._insertions = 0
        self._evictions = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def metadata(self) -> HprTargetEvaluatorMetadata | None:
        return self._metadata

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def cached_requests(self) -> tuple[HprTargetThermodynamicRequest, ...]:
        return tuple(self._cache)

    @property
    def cache_stats(self) -> HprTargetCacheStats:
        return HprTargetCacheStats(
            callbacks=self._callbacks,
            hits=self._hits,
            misses=self._misses,
            solves=self._solves,
            insertions=self._insertions,
            evictions=self._evictions,
        )

    def open(self) -> HprTargetEvaluatorMetadata:
        if self._state != "created":
            raise RuntimeError("HPR target evaluator can open only from created state")
        try:
            metadata = self._evaluator.open()
        except HprTargetEvaluatorError:
            self._state = "fatal"
            raise
        except Exception as exc:
            self._state = "fatal"
            raise HprTargetEvaluatorError(
                "evaluator_open_failure",
                "HPR target evaluator failed to open",
                session_fatal=True,
                cause=exc,
            ) from exc
        if not isinstance(metadata, HprTargetEvaluatorMetadata):
            self._state = "fatal"
            raise HprTargetEvaluatorError(
                "invalid_evaluator_metadata",
                "HPR target evaluator returned invalid metadata",
                session_fatal=True,
            )
        self._metadata = metadata
        self._state = "ready"
        return metadata

    def evaluate(
        self,
        request: HprTargetThermodynamicRequest,
    ) -> HprTargetThermodynamicResult | HprTargetEvaluationFailure:
        if self._state != "ready":
            raise RuntimeError(
                f"HPR target evaluator is {self._state}, not ready for evaluation"
            )
        self._callbacks += 1
        cached = self._cache.pop(request, None)
        if cached is not None:
            self._hits += 1
            self._cache[request] = cached
            return cached
        self._misses += 1
        self._state = "evaluating"
        self._solves += 1
        try:
            result = self._evaluator.evaluate(request)
        except HprTargetEvaluatorError as exc:
            if exc.session_fatal:
                self._state = "fatal"
                raise
            self._state = "ready"
            failure = exc.to_failure()
            self._insert_cache_value(request, failure)
            return failure
        except Exception as exc:
            self._state = "fatal"
            raise HprTargetEvaluatorError(
                "unexpected_evaluator_failure",
                "HPR target evaluator raised an unexpected exception",
                session_fatal=True,
                cause=exc,
            ) from exc

        expected_backend = "tespy" if self._metadata is None else self._metadata.backend
        validated = validate_hpr_target_result(
            request,
            result,
            expected_backend=expected_backend,
        )
        self._state = "ready"
        self._insert_cache_value(request, validated)
        return validated

    def _insert_cache_value(
        self,
        request: HprTargetThermodynamicRequest,
        value: HprTargetThermodynamicResult | HprTargetEvaluationFailure,
    ) -> None:
        self._cache[request] = value
        self._insertions += 1
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
            self._evictions += 1

    def close(self) -> None:
        if self._close_attempted:
            return
        self._close_attempted = True
        self._state = "closing"
        try:
            self._evaluator.close()
        except HprTargetEvaluatorError:
            self._state = "cleanup_failed"
            raise
        except Exception as exc:
            self._state = "cleanup_failed"
            raise HprTargetEvaluatorError(
                "cleanup_failure",
                "HPR target evaluator cleanup failed",
                session_fatal=True,
                cause=exc,
            ) from exc
        else:
            self._state = "closed"
        finally:
            self._cache.clear()


def validate_hpr_target_result(
    request: HprTargetThermodynamicRequest,
    result: HprTargetThermodynamicResult,
    *,
    expected_backend: HprSimulationBackend = "tespy",
    tolerances: HprTargetValidationTolerances = HprTargetValidationTolerances(),
) -> HprTargetThermodynamicResult | HprTargetEvaluationFailure:
    """Return an accepted result or one detached candidate-local failure."""
    if not isinstance(result, HprTargetThermodynamicResult):
        return _target_failure(
            "invalid_result_type",
            "target evaluator did not return HprTargetThermodynamicResult",
        )
    if result.backend != expected_backend:
        return _target_failure(
            "backend_mismatch",
            "target result backend does not match the selected evaluator",
        )
    if result.model_id != request.model_id:
        return _target_failure(
            "model_mismatch",
            "target result model does not match the candidate request",
        )
    if result.working_fluid != request.working_fluid:
        return _target_failure(
            "working_fluid_mismatch",
            "target result working fluid does not match the candidate request",
        )
    if not result.converged:
        return _target_failure("not_converged", "target candidate did not converge")

    quantities = (
        result.q_source,
        result.q_sink,
        result.compressor_power,
        result.cop,
    )
    if not all(math.isfinite(value) for value in quantities):
        return _target_failure(
            "nonfinite_result",
            "target result quantities must be finite",
        )
    if result.q_source < 0.0 or result.q_sink < 0.0:
        return _target_failure(
            "invalid_duty",
            "target source and sink duties must be nonnegative",
        )
    if result.compressor_power <= 0.0:
        return _target_failure(
            "invalid_power",
            "target compressor power must be positive",
        )
    if not _within_tolerance(
        result.q_sink,
        result.q_source + result.compressor_power,
        absolute=tolerances.energy_balance_absolute,
        relative=tolerances.energy_balance_relative,
    ):
        return _target_failure(
            "energy_balance_error",
            "target result does not close q_sink = q_source + compressor_power",
        )

    useful_duty = result.q_sink if request.mode == "heat_pump" else result.q_source
    if not _within_tolerance(
        useful_duty,
        request.useful_duty,
        absolute=tolerances.useful_duty_absolute,
        relative=tolerances.useful_duty_relative,
    ):
        return _target_failure(
            "useful_duty_mismatch",
            "target useful duty does not match the candidate request",
        )
    expected_cop = useful_duty / result.compressor_power
    if not _within_tolerance(
        result.cop,
        expected_cop,
        absolute=tolerances.cop_absolute,
        relative=tolerances.cop_relative,
    ):
        return _target_failure(
            "cop_mismatch",
            "target COP does not match useful duty divided by compressor power",
        )

    source_error = _profile_error(
        result.source_profile,
        expected_duty=result.q_source,
        direction="decreasing",
        tolerances=tolerances,
    )
    if source_error is not None:
        return _target_failure("invalid_source_profile", source_error)
    sink_error = _profile_error(
        result.sink_profile,
        expected_duty=result.q_sink,
        direction="increasing",
        tolerances=tolerances,
    )
    if sink_error is not None:
        return _target_failure("invalid_sink_profile", sink_error)
    return result


def _profile_error(
    profile: tuple[HprThermalProfilePoint, ...],
    *,
    expected_duty: float,
    direction: Literal["increasing", "decreasing"],
    tolerances: HprTargetValidationTolerances,
) -> str | None:
    if len(profile) < 2 or not all(
        isinstance(point, HprThermalProfilePoint) for point in profile
    ):
        return "thermal profile must contain at least two valid points"
    enthalpies = [point.enthalpy for point in profile]
    if any(
        later + tolerances.profile_order_absolute < earlier
        for earlier, later in zip(enthalpies, enthalpies[1:])
    ):
        return "thermal profile enthalpy coordinates must be ordered"
    temperatures = [point.temperature for point in profile]
    if direction == "increasing":
        out_of_order = any(
            later + tolerances.profile_order_absolute < earlier
            for earlier, later in zip(temperatures, temperatures[1:])
        )
    else:
        out_of_order = any(
            later > earlier + tolerances.profile_order_absolute
            for earlier, later in zip(temperatures, temperatures[1:])
        )
    if out_of_order:
        return f"thermal profile temperatures must be {direction}"
    profile_duty = enthalpies[-1] - enthalpies[0]
    if not _within_tolerance(
        profile_duty,
        expected_duty,
        absolute=tolerances.profile_duty_absolute,
        relative=tolerances.profile_duty_relative,
    ):
        return "thermal profile enthalpy span does not match exchanger duty"
    return None


def _within_tolerance(
    left: float,
    right: float,
    *,
    absolute: float,
    relative: float,
) -> bool:
    return abs(left - right) <= max(
        absolute,
        relative * max(abs(left), abs(right)),
    )


def _target_failure(code: str, message: str) -> HprTargetEvaluationFailure:
    return HprTargetEvaluationFailure(
        code=code,
        message=message,
        session_fatal=False,
    )


def normalize_hpr_simulation_backend(value: object) -> HprSimulationBackend:
    """Return one closed, case-insensitive HPR simulation backend selector."""
    if not isinstance(value, str):
        raise TypeError("simulation_backend must be a string: 'coolprop' or 'tespy'.")
    normalized = value.strip().lower()
    if normalized not in _SUPPORTED_HPR_SIMULATION_BACKENDS:
        raise ValueError(
            f"simulation_backend must be one of: coolprop, tespy; received {value!r}."
        )
    return cast(HprSimulationBackend, normalized)


def get_hpr_target_evaluator(
    prepared: PreparedTespyHprTargeting,
) -> HprTargetEvaluator:
    """Create one fresh concrete evaluator through the closed lazy boundary."""
    if not isinstance(prepared, PreparedTespyHprTargeting):
        raise TypeError("prepared targeting input must be PreparedTespyHprTargeting")
    try:
        from .adapters.tespy import TespyHprTargetEvaluator
    except ImportError as exc:
        raise HprTargetEvaluatorError(
            "dependency_unavailable",
            "TESPy targeting requires the optional 'tespy' extra",
            session_fatal=True,
            cause=exc,
        ) from exc
    return TespyHprTargetEvaluator(prepared)


def preflight_tespy_hpr_targeting(
    args: HeatPumpTargetInputs | MultiPeriodHPRTargetInputs,
) -> PreparedTespyHprTargeting:
    """Validate the initial TESPy targeting envelope without creating a network."""
    if isinstance(args, MultiPeriodHPRTargetInputs):
        raise HprTargetCompatibilityError(
            "unsupported_multiperiod",
            "TESPy targeting does not support shared-vector multi-period optimization.",
        )
    if not isinstance(args, HeatPumpTargetInputs):
        raise HprTargetCompatibilityError(
            "invalid_target_inputs",
            "TESPy targeting requires scalar HeatPumpTargetInputs.",
        )
    if normalize_hpr_simulation_backend(args.simulation_backend) != "tespy":
        raise HprTargetCompatibilityError(
            "backend_mismatch",
            "TESPy compatibility preflight requires simulation_backend='tespy'.",
        )
    if args.hpr_type != HeatPumpAndRefrigerationCycle.CascadeVapourComp.value:
        raise HprTargetCompatibilityError(
            "unsupported_cycle",
            "TESPy targeting initially supports only single-stage vapour compression.",
        )
    if int(args.n_evap) != 1 or int(args.n_cond) != 1:
        raise HprTargetCompatibilityError(
            "unsupported_topology",
            "TESPy targeting requires exactly one evaporator and one condenser.",
        )
    if bool(args.allow_integrated_expander):
        raise HprTargetCompatibilityError(
            "integrated_expander_unsupported",
            "TESPy targeting does not support the integrated-expander option.",
        )
    if any(float(value) != 0.0 for value in (getattr(args, "dt_hp_ihx", 0.0),)):
        raise HprTargetCompatibilityError(
            "unsupported_model",
            "TESPy single-stage targeting does not include an internal heat exchanger.",
        )
    if find_spec("tespy") is None:
        raise HprTargetCompatibilityError(
            "dependency_unavailable",
            "TESPy targeting requires the optional 'tespy' extra.",
        )

    refrigerant = _select_single_loop_refrigerant(args)
    evaporating_temperature, condensing_temperature = _property_probe_temperatures(args)
    try:
        working_fluid = resolve_hpr_working_fluid(
            refrigerant,
            evaporating_temperature,
            condensing_temperature,
        )
    except ValueError as exc:
        code = (
            "property_backend_unsupported"
            if "REFPROP" in str(exc)
            else "working_fluid_unsupported"
        )
        raise HprTargetCompatibilityError(code, str(exc)) from exc

    return PreparedTespyHprTargeting(
        simulation_backend="tespy",
        mode="heat_pump" if args.is_heat_pumping else "refrigeration",
        cycle_id="single_stage_vapour_compression",
        model_id="openpinch-tespy-single-stage-v1",
        working_fluid=working_fluid,
        tespy_fluid_token=to_tespy_fluid_token(working_fluid),
        evaporator_count=1,
        condenser_count=1,
        period_idx=int(args.period_idx),
        probe_evaporating_temperature=evaporating_temperature,
        probe_condensing_temperature=condensing_temperature,
    )


def _select_single_loop_refrigerant(args: HeatPumpTargetInputs) -> str:
    refrigerants = [str(value).strip() for value in args.refrigerant_ls]
    refrigerants = [value for value in refrigerants if value]
    if not refrigerants:
        return "water"
    if args.do_refrigerant_sort:
        try:
            refrigerants.sort(
                key=lambda value: float(PropsSI("Tcrit", value)),
                reverse=True,
            )
        except Exception as exc:
            raise HprTargetCompatibilityError(
                "working_fluid_unsupported",
                "TESPy targeting could not rank the configured working fluids.",
            ) from exc
    return refrigerants[0]


def _property_probe_temperatures(
    args: HeatPumpTargetInputs,
) -> tuple[float, float]:
    evap_values = [float(value) for value in args.T_hot]
    cond_values = [float(value) for value in args.T_cold]
    if (
        not evap_values
        or not cond_values
        or not all(math.isfinite(value) for value in evap_values + cond_values)
    ):
        raise HprTargetCompatibilityError(
            "unsupported_state",
            "TESPy targeting requires finite evaporation and condensation bounds.",
        )
    minimum_lift = max(float(args.dtcont_hp), 1.0)
    evap_min, evap_max = min(evap_values), max(evap_values)
    cond_min, cond_max = min(cond_values), max(cond_values)
    evaporating_temperature = max(
        evap_min,
        min(evap_max, cond_min - minimum_lift),
    )
    condensing_temperature = max(
        cond_min,
        evaporating_temperature + minimum_lift,
    )
    if condensing_temperature > cond_max:
        raise HprTargetCompatibilityError(
            "unsupported_state",
            "TESPy targeting bounds contain no positive-lift scalar state.",
        )
    return evaporating_temperature, condensing_temperature


__all__ = [
    "HprSimulationBackend",
    "HprTargetCompatibilityError",
    "HprTargetCacheStats",
    "HprTargetEvaluatorCoordinator",
    "HprTargetValidationTolerances",
    "PreparedTespyHprTargeting",
    "get_hpr_target_evaluator",
    "normalize_hpr_simulation_backend",
    "preflight_tespy_hpr_targeting",
    "validate_hpr_target_result",
]
