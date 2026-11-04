"""Public data models returned by direct gas MVR solves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ....domain.stream_collection import StreamCollection

DEFAULT_MVR_COMP_EFFICIENCY = 0.7
DEFAULT_MOTOR_EFFICIENCY = 0.95
DEFAULT_DIRECT_MVR_STAGES = 1
DEFAULT_TEMPERATURE_UNIT = "degC"
DEFAULT_PRESSURE_UNIT = "kPa"
DEFAULT_ENTHALPY_UNIT = "kJ/kg"
DEFAULT_HEAT_FLOW_UNIT = "kW"


@dataclass(frozen=True)
class DirectGasMVROutputUnits:
    """Units used for public direct-MVR outputs."""

    temperature: str = DEFAULT_TEMPERATURE_UNIT
    pressure: str = DEFAULT_PRESSURE_UNIT
    enthalpy: str = DEFAULT_ENTHALPY_UNIT
    heat_flow: str = DEFAULT_HEAT_FLOW_UNIT


@dataclass
class DirectGasMVRSettings:
    """User-facing settings for one direct gas MVR solve."""

    n_stages: int = DEFAULT_DIRECT_MVR_STAGES
    mvr_stage_t_lift: float | None = None
    mvr_stage_pressure_ratio: float | None = None
    liquid_injection: bool = False
    eta_mvr_comp: float = DEFAULT_MVR_COMP_EFFICIENCY
    eta_motor: float = DEFAULT_MOTOR_EFFICIENCY
    dt_diff_max: float = 0.1


@dataclass(frozen=True)
class DirectGasMVRFallbackDiagnostic:
    """Detached evidence that a named direct-MVR fallback was applied."""

    code: Literal["dry_stage", "reduced_profile"]
    summary: str
    source_stream: str
    period_index: int
    stage_index: int
    fluid: str

    def __post_init__(self) -> None:
        if self.code not in {"dry_stage", "reduced_profile"}:
            raise ValueError("Unsupported direct-MVR fallback code.")
        if (
            not self.summary
            or len(self.summary) > 160
            or any(c in self.summary for c in "\r\n")
        ):
            raise ValueError(
                "Direct-MVR fallback summary must be 1-160 single-line characters."
            )
        if (
            type(self.period_index) is not int
            or type(self.stage_index) is not int
            or self.period_index < 0
            or self.stage_index < 1
        ):
            raise ValueError(
                "Direct-MVR fallback indices must be non-negative/positive."
            )
        object.__setattr__(self, "source_stream", _bounded_context(self.source_stream))
        object.__setattr__(self, "fluid", _bounded_context(self.fluid))


PHASE_TARGET_BELOW_SATURATION = "phase.target_below_saturation"
PHASE_LIQUID_STAGE_INLET = "phase.liquid_stage_inlet"


class DirectGasMVRStageError(ValueError):
    """A bounded required-state failure with direct-MVR stage context."""

    def __init__(
        self,
        *,
        reason_code: str,
        source_stream: str,
        period_index: int,
        stage_index: int,
        fluid: str,
    ) -> None:
        self.reason_code = reason_code
        self.source_stream = source_stream
        self.period_index = period_index
        self.stage_index = stage_index
        self.fluid = fluid
        display_stream = _bounded_context(source_stream)
        display_fluid = _bounded_context(fluid)
        super().__init__(
            "Direct MVR stage could not evaluate a required thermodynamic state "
            f"for stream {display_stream!r}, period {period_index}, stage "
            f"{stage_index}, fluid {display_fluid!r} ({reason_code})."
        )

    def __reduce__(self):
        return (
            _restore_direct_mvr_error,
            (
                dict(
                    reason_code=self.reason_code,
                    source_stream=self.source_stream,
                    period_index=self.period_index,
                    stage_index=self.stage_index,
                    fluid=self.fluid,
                ),
            ),
        )


class DirectGasMVRPhaseError(DirectGasMVRStageError):
    """A stage whose fluid would be liquid where the model needs vapour."""

    def __init__(
        self,
        *,
        reason_code: str,
        source_stream: str,
        period_index: int,
        stage_index: int,
        fluid: str,
        t_target: float,
        t_saturation: float,
        pressure_kpa: float,
        above_critical_pressure: bool = False,
    ) -> None:
        super().__init__(
            reason_code=reason_code,
            source_stream=source_stream,
            period_index=period_index,
            stage_index=stage_index,
            fluid=fluid,
        )
        self.t_target = t_target
        self.t_saturation = t_saturation
        self.pressure_kpa = pressure_kpa
        self.above_critical_pressure = above_critical_pressure
        limit = "critical" if above_critical_pressure else "saturation"
        context = (
            f"Direct MVR stream {_bounded_context(source_stream)!r} "
            f"({_bounded_context(fluid)}), period {period_index}, stage {stage_index}: "
            f"target {t_target:.2f} degC is below the {limit} temperature "
            f"{t_saturation:.2f} degC at {pressure_kpa:.2f} kPa"
        )
        if reason_code == PHASE_LIQUID_STAGE_INLET:
            detail = (
                "so the previous stage leaves liquid and this stage would start from "
                "liquid. Multi-stage MVR of a condensing stream is not modelled; use "
                "n_stages=1."
            )
        else:
            detail = (
                "so the stream would condense although its source target is vapour. "
                "Raise t_target above that temperature or use a smaller stage lift."
            )
        self.args = (f"{context}, {detail}",)

    def __reduce__(self):
        return (
            _restore_direct_mvr_phase_error,
            (
                dict(
                    reason_code=self.reason_code,
                    source_stream=self.source_stream,
                    period_index=self.period_index,
                    stage_index=self.stage_index,
                    fluid=self.fluid,
                    t_target=self.t_target,
                    t_saturation=self.t_saturation,
                    pressure_kpa=self.pressure_kpa,
                    above_critical_pressure=self.above_critical_pressure,
                ),
            ),
        )


def _restore_direct_mvr_error(context):
    return DirectGasMVRStageError(**context)


def _restore_direct_mvr_phase_error(context):
    return DirectGasMVRPhaseError(**context)


def _bounded_context(value: str, limit: int = 80) -> str:
    normalized = str(value).replace("\r", " ").replace("\n", " ")
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


@dataclass
class DirectGasMVRStageResult:
    """Solved accounting for one direct gas MVR stage."""

    source_stream: str
    stage_index: int
    p_in: float
    p_out: float
    t_in: float
    t_discharge: float
    t_hot_supply: float
    t_target: float
    heat_flow: float
    work: float
    h_hot_supply: float
    h_target: float
    th_curve: np.ndarray = field(repr=False)
    linearised_profile: np.ndarray = field(repr=False)
    q_liquid_injection: float = 0.0
    liquid_injection_applied: bool = False
    temperature_unit: str = DEFAULT_TEMPERATURE_UNIT
    pressure_unit: str = DEFAULT_PRESSURE_UNIT
    enthalpy_unit: str = DEFAULT_ENTHALPY_UNIT
    heat_flow_unit: str = DEFAULT_HEAT_FLOW_UNIT
    source_mass_flow: float = 0.0
    hot_mass_flow: float = 0.0
    liquid_injection_ratio: float = 0.0
    fallback_diagnostics: tuple[DirectGasMVRFallbackDiagnostic, ...] = ()


@dataclass
class DirectGasMVRStreamSolveResult:
    """Solved direct gas MVR streams for one source stream at one period index."""

    replacement_streams: StreamCollection
    stage_results: list[DirectGasMVRStageResult] = field(default_factory=list)
