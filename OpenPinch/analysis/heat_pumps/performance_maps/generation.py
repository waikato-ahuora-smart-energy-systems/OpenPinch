"""Atomic engine-neutral orchestration of complete HPR performance maps."""

from __future__ import annotations

import math

from ....contracts.hpr_performance_map import (
    HprPerformanceMap,
    HprPerformanceMapRequest,
    HprPerformanceMapUnits,
    HprPerformancePoint,
)
from .context import build_hpr_map_generation_context
from .errors import (
    HprMapGenerationError,
    HprSimulatorFailure,
    diagnostic_from_failure,
)
from .factory import get_hpr_point_simulator
from .models import (
    HprMapGenerationContext,
    HprOperatingPoint,
    HprPointSimulation,
    HprSimulationDiagnostic,
    HprSimulatorMetadata,
    HprTargetMapBasis,
)
from .points import iter_hpr_operating_points
from .protocols import HprPointSimulatorFactory
from .provenance import build_hpr_map_provenance

_RELATIVE_TOLERANCE = 1e-9


def _diagnostic(
    code: str,
    message: str,
    context: HprMapGenerationContext,
    point: HprOperatingPoint | None,
    *,
    details: dict[str, object] | None = None,
) -> HprSimulationDiagnostic:
    return diagnostic_from_failure(
        HprSimulatorFailure(
            code,
            message,
            session_fatal=False,
            details=details,
        ),
        backend=context.basis.simulation_backend.strip().lower(),
        model_id=context.basis.model_id,
        point=point,
    )


def _normalize_point(
    context: HprMapGenerationContext,
    point: HprOperatingPoint,
    simulation: HprPointSimulation,
) -> tuple[HprPerformancePoint | None, HprSimulationDiagnostic | None]:
    if not simulation.converged:
        return None, _diagnostic(
            "non_converged",
            "selected simulator did not converge at the requested operating point",
            context,
            point,
        )
    values = (
        simulation.q_source,
        simulation.q_sink,
        simulation.compressor_power,
    )
    if (
        not all(math.isfinite(value) for value in values)
        or simulation.q_source < 0.0
        or simulation.q_sink < 0.0
        or simulation.compressor_power <= 0.0
    ):
        return None, _diagnostic(
            "invalid_simulation",
            "simulator duties and compressor power are outside their valid domains",
            context,
            point,
        )

    tolerance = context.energy_balance_tolerance
    if not math.isclose(
        simulation.q_sink,
        simulation.q_source + simulation.compressor_power,
        rel_tol=_RELATIVE_TOLERANCE,
        abs_tol=tolerance,
    ):
        return None, _diagnostic(
            "energy_balance",
            "simulated source, sink, and compressor power do not close",
            context,
            point,
        )
    useful_duty = (
        simulation.q_sink
        if context.reference_capacity_basis == "q_sink"
        else simulation.q_source
    )
    if not math.isclose(
        useful_duty,
        point.requested_useful_duty,
        rel_tol=_RELATIVE_TOLERANCE,
        abs_tol=tolerance,
    ):
        return None, _diagnostic(
            "useful_duty_mismatch",
            "simulated useful duty does not match the requested useful duty",
            context,
            point,
        )
    cop = useful_duty / simulation.compressor_power
    return (
        HprPerformancePoint(
            name=point.name,
            curve_id=point.curve_id,
            source_temperature=point.source_temperature,
            sink_temperature=point.sink_temperature,
            load_fraction=point.load_fraction,
            q_source=simulation.q_source,
            q_sink=simulation.q_sink,
            electric_power=simulation.compressor_power,
            cop=cop,
        ),
        None,
    )


def _raise_generation_error(
    diagnostics: list[HprSimulationDiagnostic],
    cause: BaseException | None,
) -> None:
    error = HprMapGenerationError(tuple(diagnostics))
    if cause is not None:
        raise error from cause
    raise error


def generate_hpr_performance_map(
    basis: HprTargetMapBasis,
    request: HprPerformanceMapRequest,
    *,
    simulator_factory: HprPointSimulatorFactory = get_hpr_point_simulator,
) -> HprPerformanceMap:
    """Generate one complete map or raise one ordered aggregate failure."""
    context = build_hpr_map_generation_context(basis, request)
    backend = basis.simulation_backend.strip().lower()
    diagnostics: list[HprSimulationDiagnostic] = []
    complete_points: list[HprPerformancePoint] = []
    simulator = None
    metadata: HprSimulatorMetadata | None = None
    primary_cause: BaseException | None = None
    fatal = False

    try:
        try:
            simulator = simulator_factory(backend)
            metadata = simulator.prepare(context)
            if (
                not metadata.design_converged
                or metadata.backend != backend
                or metadata.model_id != basis.model_id
            ):
                raise HprSimulatorFailure(
                    "prepare_failed",
                    "simulator metadata does not match the prepared context",
                    session_fatal=True,
                )
        except HprSimulatorFailure as failure:
            primary_cause = failure.cause
            diagnostics.append(
                diagnostic_from_failure(
                    failure,
                    backend=backend,
                    model_id=basis.model_id,
                )
            )
        except Exception as exc:
            primary_cause = exc
            diagnostics.append(
                _diagnostic(
                    "prepare_failed",
                    "selected simulator raised while preparing its design state",
                    context,
                    None,
                )
            )

        if metadata is not None:
            for point in iter_hpr_operating_points(context):
                if fatal:
                    diagnostics.append(
                        _diagnostic(
                            "session_unavailable",
                            "a prior fatal simulator failure prevents this solve",
                            context,
                            point,
                        )
                    )
                    continue
                if not point.is_valid:
                    diagnostics.append(
                        _diagnostic(
                            "invalid_operating_point",
                            point.validation_error or "invalid operating point",
                            context,
                            point,
                        )
                    )
                    continue
                try:
                    simulation = simulator.simulate(point)
                except HprSimulatorFailure as failure:
                    if primary_cause is None:
                        primary_cause = failure.cause
                    diagnostics.append(
                        diagnostic_from_failure(
                            failure,
                            backend=backend,
                            model_id=basis.model_id,
                            point=point,
                        )
                    )
                    fatal = failure.session_fatal
                    continue
                except Exception as exc:
                    if primary_cause is None:
                        primary_cause = exc
                    diagnostics.append(
                        _diagnostic(
                            "point_exception",
                            (
                                "selected simulator raised at the requested "
                                "operating point"
                            ),
                            context,
                            point,
                        )
                    )
                    fatal = True
                    continue
                normalized, failure = _normalize_point(context, point, simulation)
                if failure is not None:
                    diagnostics.append(failure)
                elif normalized is not None:
                    complete_points.append(normalized)
    finally:
        if simulator is not None:
            try:
                simulator.close()
            except Exception as exc:
                if primary_cause is None:
                    primary_cause = exc
                diagnostics.append(
                    _diagnostic(
                        "cleanup_failed",
                        "selected simulator failed while releasing session resources",
                        context,
                        None,
                    )
                )

    if diagnostics:
        _raise_generation_error(diagnostics, primary_cause)
    assert metadata is not None
    provenance = build_hpr_map_provenance(
        context,
        metadata,
        point_count=len(complete_points),
    )
    return HprPerformanceMap(
        schema_version="1.0",
        map_id=request.map_id,
        mode=basis.mode,
        units=HprPerformanceMapUnits(
            source_temperature="degC",
            sink_temperature="degC",
            q_source="kW",
            q_sink="kW",
            electric_power="kW",
        ),
        reference_capacity=context.reference_capacity,
        reference_capacity_basis=context.reference_capacity_basis,
        interpolation_topology="ordered_part_load_curve",
        thermodynamic_backend=backend,
        model_id=basis.model_id,
        provenance=provenance,
        points=tuple(complete_points),
        cop_convention=context.cop_convention,
        energy_balance_tolerance=context.energy_balance_tolerance,
        temperature_match_tolerance=context.temperature_match_tolerance,
    )


__all__ = ["generate_hpr_performance_map"]
