"""Validation shell for accepted heat exchanger network synthesis design data."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ....contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger import HeatExchanger
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..indexing import ordered_mapping_keys

_DUTY_ABS_TOL = 1.0
_DUTY_REL_TOL = 1e-4
_COST_ABS_TOL = 1.0
_COST_REL_TOL = 1e-4
_TEMPERATURE_ABS_TOL = 1e-3
_AREA_ABS_TOL = 1e-3
_AREA_REL_TOL = 1e-4


def verify_synthesis_result(
    result: HeatExchangerNetworkSynthesisResult,
) -> tuple[str, ...]:
    """Return contract failures for an accepted design result."""
    failures: list[str] = []
    if not result.network.exchangers:
        failures.append(
            "accepted heat exchanger network design must contain at least one exchanger"
        )
    if result.network.run_id != result.run_id:
        failures.append("network run_id must match the enclosing design result")
    if result.task_id is not None and result.network.task_id != result.task_id:
        failures.append("network task_id must match the accepted task id")
    if (
        result.network.total_annual_cost is None
        and "total_annual_cost" not in result.objective_values
    ):
        failures.append(
            "accepted heat exchanger network design must include total annual "
            "cost metadata"
        )
    failures.extend(
        f"accepted network infeasible: {reason}"
        for reason in verify_network_feasibility(result.network)
    )
    for outcome in result.ranked_networks:
        if outcome.status == "success" and outcome.network is None:
            failures.append(
                f"successful task {outcome.task.task_id} is missing network output"
            )
        if outcome.status == "success" and outcome.network is not None:
            failures.extend(
                f"successful task {outcome.task.task_id} infeasible: {reason}"
                for reason in verify_network_feasibility(outcome.network)
            )
    return tuple(failures)


def verify_network_feasibility(
    network: HeatExchangerNetwork | None,
) -> tuple[str, ...]:
    """Return feasibility failures for one extracted HEN candidate.

    The checks intentionally operate on the service-boundary network data,
    not the private solver model. They therefore catch under-costed or
    heat-balance-inconsistent EVM candidates after solution extraction and before
    ranking/selection.
    """

    if network is None:
        return ("network output is missing",)

    failures: list[str] = []
    period_ids = network.period_ids
    if len(period_ids) <= 1:
        failures.extend(_check_summary_totals(network))
    failures.extend(_check_cost_consistency(network))
    for period_id in period_ids:
        failures.extend(_check_temperature_order(network.exchangers, period_id))
        failures.extend(_check_minimum_approach(network, period_id))
        failures.extend(_check_exchanger_duty_balances(network, period_id))
        failures.extend(_check_area_presence(network.exchangers, period_id))
        failures.extend(_check_area_consistency(network, period_id))
    if len(period_ids) <= 1:
        failures.extend(_check_stream_heat_balances(network))
    return tuple(failures)


def is_network_feasible(network: HeatExchangerNetwork | None) -> bool:
    """Return whether one extracted HEN candidate passes feasibility checks."""

    return not verify_network_feasibility(network)


def _check_summary_totals(network: HeatExchangerNetwork) -> list[str]:
    failures: list[str] = []
    metrics = network.summary_metrics
    checks = {
        "hot_utility_load": network.total_duty(kind=HeatExchangerKind.HOT_UTILITY),
        "cold_utility_load": network.total_duty(kind=HeatExchangerKind.COLD_UTILITY),
        "recovery_load": network.total_duty(kind=HeatExchangerKind.RECOVERY),
        "hot_utility_units": _count_active(
            network.exchangers, HeatExchangerKind.HOT_UTILITY
        ),
        "cold_utility_units": _count_active(
            network.exchangers, HeatExchangerKind.COLD_UTILITY
        ),
        "recovery_units": _count_active(network.exchangers, HeatExchangerKind.RECOVERY),
        "total_units": _count_active(network.exchangers, None),
    }
    for metric_name, observed in checks.items():
        expected = metrics.get(metric_name)
        if expected is None:
            continue
        if metric_name.endswith("units"):
            if int(expected) != int(observed):
                failures.append(
                    f"{metric_name} summary {expected} does not match active "
                    f"network count {observed}"
                )
        elif not _close(float(expected), float(observed), abs_tol=_DUTY_ABS_TOL):
            failures.append(
                f"{metric_name} summary {float(expected):.6g} does not match "
                f"exchanger duty total {float(observed):.6g}"
            )
    return failures


def _check_cost_consistency(network: HeatExchangerNetwork) -> list[str]:
    if (
        network.total_annual_cost is None
        or network.utility_cost is None
        or network.capital_cost is None
    ):
        return []
    component_total = network.utility_cost + network.capital_cost
    if _close(
        network.total_annual_cost,
        component_total,
        rel_tol=_COST_REL_TOL,
        abs_tol=_COST_ABS_TOL,
    ):
        return []
    return [
        "total annual cost "
        f"{network.total_annual_cost:.6g} does not match utility + capital "
        f"{component_total:.6g}"
    ]


def _check_temperature_order(
    exchangers: Iterable[HeatExchanger],
    period_id: str,
) -> list[str]:
    failures: list[str] = []
    for exchanger in exchangers:
        state = exchanger.state(period_id)
        if not state.active:
            continue
        if (
            state.source_inlet_temperature is not None
            and state.source_outlet_temperature is not None
        ):
            if (
                state.source_outlet_temperature
                > state.source_inlet_temperature + _TEMPERATURE_ABS_TOL
            ):
                failures.append(
                    f"{exchanger.exchanger_id or exchanger.kind.value} source "
                    "temperature increases across a heat source"
                )
        if (
            state.sink_inlet_temperature is not None
            and state.sink_outlet_temperature is not None
        ):
            if (
                state.sink_outlet_temperature + _TEMPERATURE_ABS_TOL
                < state.sink_inlet_temperature
            ):
                failures.append(
                    f"{exchanger.exchanger_id or exchanger.kind.value} sink "
                    "temperature decreases across a heat sink"
                )
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            if (
                state.source_inlet_temperature is not None
                and state.sink_outlet_temperature is not None
                and state.source_inlet_temperature + _TEMPERATURE_ABS_TOL
                < state.sink_outlet_temperature
            ):
                failures.append(
                    f"{exchanger.exchanger_id or exchanger.kind.value} hot inlet "
                    "is colder than cold outlet"
                )
            if (
                state.source_outlet_temperature is not None
                and state.sink_inlet_temperature is not None
                and state.source_outlet_temperature + _TEMPERATURE_ABS_TOL
                < state.sink_inlet_temperature
            ):
                failures.append(
                    f"{exchanger.exchanger_id or exchanger.kind.value} hot outlet "
                    "is colder than cold inlet"
                )
    return failures


def _check_minimum_approach(
    network: HeatExchangerNetwork,
    period_id: str,
) -> list[str]:
    d_tmin = _optional_float(network.source_metadata.get("solver_dTmin"))
    if d_tmin is None:
        return []

    failures: list[str] = []
    for exchanger in network.exchangers:
        state = exchanger.state(period_id)
        if not state.active or exchanger.kind is not HeatExchangerKind.RECOVERY:
            continue
        approaches = state.approach_temperatures
        if not approaches:
            approaches = _recovery_terminal_approaches(state)
        if approaches and min(approaches) + _TEMPERATURE_ABS_TOL < d_tmin:
            failures.append(
                f"{exchanger.exchanger_id or exchanger.kind.value} minimum "
                f"approach {min(approaches):.6g} is below dTmin {d_tmin:.6g}"
            )
    return failures


def _check_exchanger_duty_balances(
    network: HeatExchangerNetwork,
    period_id: str,
) -> list[str]:
    metadata = network.source_metadata
    segment_failures = _check_segment_contribution_balances(
        network.exchangers,
        period_id,
    )
    if _uses_isothermal_stage_boundaries(metadata):
        return segment_failures
    hot_cp = _period_stream_value_map(
        network,
        "hot_process_streams",
        metadata=metadata,
        period_id=period_id,
        scalar_key="hot_stream_heat_capacity_flowrates",
        period_key="hot_stream_heat_capacity_flowrates_by_period",
    )
    cold_cp = _period_stream_value_map(
        network,
        "cold_process_streams",
        metadata=metadata,
        period_id=period_id,
        scalar_key="cold_stream_heat_capacity_flowrates",
        period_key="cold_stream_heat_capacity_flowrates_by_period",
    )
    if not hot_cp and not cold_cp:
        return segment_failures

    failures: list[str] = list(segment_failures)
    for exchanger in network.exchangers:
        state = exchanger.state(period_id)
        if not state.active:
            continue
        if exchanger.segment_area_contributions:
            continue
        if exchanger.kind in {
            HeatExchangerKind.RECOVERY,
            HeatExchangerKind.COLD_UTILITY,
        }:
            cp = hot_cp.get(exchanger.source_stream)
            if cp is not None:
                expected = _heat_removed(
                    cp
                    * (
                        state.source_split_fraction
                        if state.source_split_fraction is not None
                        else 1.0
                    ),
                    state.source_inlet_temperature,
                    state.source_outlet_temperature,
                )
                failures.extend(_duty_failure(exchanger, state, "source", expected))
        if exchanger.kind in {
            HeatExchangerKind.RECOVERY,
            HeatExchangerKind.HOT_UTILITY,
        }:
            cp = cold_cp.get(exchanger.sink_stream)
            if cp is not None:
                expected = _heat_added(
                    cp
                    * (
                        state.sink_split_fraction
                        if state.sink_split_fraction is not None
                        else 1.0
                    ),
                    state.sink_inlet_temperature,
                    state.sink_outlet_temperature,
                )
                failures.extend(_duty_failure(exchanger, state, "sink", expected))
    return failures


def _check_stream_heat_balances(network: HeatExchangerNetwork) -> list[str]:
    metadata = network.source_metadata
    hot_cp = _stream_value_map(
        network,
        "hot_process_streams",
        metadata.get("hot_stream_heat_capacity_flowrates"),
    )
    cold_cp = _stream_value_map(
        network,
        "cold_process_streams",
        metadata.get("cold_stream_heat_capacity_flowrates"),
    )
    hot_total_duties = _stream_value_map(
        network,
        "hot_process_streams",
        metadata.get("hot_stream_total_duties"),
    )
    cold_total_duties = _stream_value_map(
        network,
        "cold_process_streams",
        metadata.get("cold_stream_total_duties"),
    )
    if not hot_cp and not cold_cp and not hot_total_duties and not cold_total_duties:
        return []

    failures: list[str] = []
    hot_supply = _stream_value_map(
        network,
        "hot_process_streams",
        metadata.get("hot_stream_supply_temperatures"),
    )
    hot_target = _stream_value_map(
        network,
        "hot_process_streams",
        metadata.get("hot_stream_target_temperatures"),
    )
    cold_supply = _stream_value_map(
        network,
        "cold_process_streams",
        metadata.get("cold_stream_supply_temperatures"),
    )
    cold_target = _stream_value_map(
        network,
        "cold_process_streams",
        metadata.get("cold_stream_target_temperatures"),
    )

    for stream in hot_cp.keys() | hot_total_duties.keys():
        expected = hot_total_duties.get(stream)
        if expected is None:
            cp = hot_cp.get(stream)
            if cp is None:
                continue
            expected = _heat_removed(cp, hot_supply.get(stream), hot_target.get(stream))
        if expected is None:
            continue
        observed = network.total_duty(
            stream=stream,
            kind=HeatExchangerKind.RECOVERY,
        ) + network.total_duty(stream=stream, kind=HeatExchangerKind.COLD_UTILITY)
        if not _close(expected, observed, abs_tol=_DUTY_ABS_TOL):
            failures.append(
                f"hot stream {stream} heat removed {observed:.6g} does not "
                f"match required heat load {expected:.6g}"
            )

    for stream in cold_cp.keys() | cold_total_duties.keys():
        expected = cold_total_duties.get(stream)
        if expected is None:
            cp = cold_cp.get(stream)
            if cp is None:
                continue
            expected = _heat_added(cp, cold_supply.get(stream), cold_target.get(stream))
        if expected is None:
            continue
        observed = network.total_duty(
            stream=stream,
            kind=HeatExchangerKind.RECOVERY,
        ) + network.total_duty(stream=stream, kind=HeatExchangerKind.HOT_UTILITY)
        if not _close(expected, observed, abs_tol=_DUTY_ABS_TOL):
            failures.append(
                f"cold stream {stream} heat added {observed:.6g} does not "
                f"match required heat load {expected:.6g}"
            )
    return failures


def _check_area_presence(
    exchangers: Iterable[HeatExchanger],
    period_id: str,
) -> list[str]:
    failures = []
    for exchanger in exchangers:
        state = exchanger.state(period_id)
        if not state.active or state.duty <= _DUTY_ABS_TOL:
            continue
        if exchanger.area is None or exchanger.area <= 0.0:
            failures.append(
                f"{exchanger.exchanger_id or exchanger.kind.value} has positive "
                "duty but no positive area"
            )
    return failures


def _check_area_consistency(
    network: HeatExchangerNetwork,
    period_id: str,
) -> list[str]:
    metadata = network.source_metadata
    failures = _check_segment_contribution_areas(network.exchangers, period_id)
    if _uses_isothermal_stage_boundaries(metadata):
        return failures
    hot_htc = _stream_value_map(
        network,
        "hot_process_streams",
        metadata.get("hot_stream_heat_transfer_coefficients"),
    )
    cold_htc = _stream_value_map(
        network,
        "cold_process_streams",
        metadata.get("cold_stream_heat_transfer_coefficients"),
    )
    hot_utility_htc = _stream_value_map(
        network,
        "hot_utilities",
        metadata.get("hot_utility_heat_transfer_coefficients"),
    )
    cold_utility_htc = _stream_value_map(
        network,
        "cold_utilities",
        metadata.get("cold_utility_heat_transfer_coefficients"),
    )
    if not any((hot_htc, cold_htc, hot_utility_htc, cold_utility_htc)):
        return failures

    for exchanger in network.exchangers:
        state = exchanger.state(period_id)
        if not state.active or state.duty <= _DUTY_ABS_TOL or exchanger.area is None:
            continue
        if exchanger.segment_area_contributions:
            continue
        overall_htc = _overall_heat_transfer_coefficient(
            exchanger,
            hot_htc=hot_htc,
            cold_htc=cold_htc,
            hot_utility_htc=hot_utility_htc,
            cold_utility_htc=cold_utility_htc,
        )
        terminal_lmtd = _terminal_lmtd(state)
        if overall_htc is None or terminal_lmtd is None:
            continue
        required_area = state.duty / overall_htc / terminal_lmtd
        if exchanger.area + _AREA_ABS_TOL < required_area and not _close(
            required_area,
            exchanger.area,
            rel_tol=_AREA_REL_TOL,
            abs_tol=_AREA_ABS_TOL,
        ):
            failures.append(
                f"{exchanger.exchanger_id or exchanger.kind.value} area "
                f"{exchanger.area:.6g} is below required area "
                f"{required_area:.6g}"
            )
    return failures


def _check_segment_contribution_balances(
    exchangers: Iterable[HeatExchanger],
    period_id: str,
) -> list[str]:
    failures = []
    for exchanger in exchangers:
        state = exchanger.state(period_id)
        contributions = exchanger.segment_area_contributions
        if not state.active or not contributions:
            continue
        period_duties = exchanger.segment_duty_by_period
        if period_id in period_duties:
            contribution_duty = period_duties[period_id]
            if not _close(
                contribution_duty,
                state.duty,
                rel_tol=_DUTY_REL_TOL,
                abs_tol=_DUTY_ABS_TOL,
            ):
                failures.append(
                    f"{exchanger.exchanger_id or exchanger.kind.value} segment "
                    f"duty {contribution_duty:.6g} does not match parent duty "
                    f"{state.duty:.6g} in period {period_id!r}"
                )
    return failures


def _check_segment_contribution_areas(
    exchangers: Iterable[HeatExchanger],
    period_id: str,
) -> list[str]:
    failures = []
    for exchanger in exchangers:
        state = exchanger.state(period_id)
        contributions = exchanger.segment_area_contributions
        if not state.active or not contributions or exchanger.area is None:
            continue
        for contribution in contributions:
            if contribution.period != period_id:
                continue
            required = contribution.duty / contribution.overall_htc / contribution.lmtd
            if not _close(
                required,
                contribution.area,
                rel_tol=_AREA_REL_TOL,
                abs_tol=_AREA_ABS_TOL,
            ):
                failures.append(
                    f"{exchanger.exchanger_id or exchanger.kind.value} segment "
                    f"area {contribution.area:.6g} does not match local duty/U/LMTD "
                    f"{required:.6g}"
                )
        design_area = exchanger.segment_design_area
        if design_area is None:
            continue
        if not _close(
            design_area,
            exchanger.area,
            rel_tol=_AREA_REL_TOL,
            abs_tol=_AREA_ABS_TOL,
        ):
            failures.append(
                f"{exchanger.exchanger_id or exchanger.kind.value} area "
                f"{exchanger.area:.6g} does not match maximum period-total "
                f"segment area {design_area:.6g}"
            )
    return failures


def _overall_heat_transfer_coefficient(
    exchanger: HeatExchanger,
    *,
    hot_htc: dict[str, float],
    cold_htc: dict[str, float],
    hot_utility_htc: dict[str, float],
    cold_utility_htc: dict[str, float],
) -> float | None:
    if exchanger.kind is HeatExchangerKind.RECOVERY:
        source_htc = hot_htc.get(exchanger.source_stream)
        sink_htc = cold_htc.get(exchanger.sink_stream)
    elif exchanger.kind is HeatExchangerKind.HOT_UTILITY:
        source_htc = hot_utility_htc.get(exchanger.source_stream)
        sink_htc = cold_htc.get(exchanger.sink_stream)
    elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
        source_htc = hot_htc.get(exchanger.source_stream)
        sink_htc = cold_utility_htc.get(exchanger.sink_stream)
    else:
        return None
    if source_htc is None or sink_htc is None or source_htc <= 0.0 or sink_htc <= 0.0:
        return None
    return 1.0 / (1.0 / source_htc + 1.0 / sink_htc)


def _terminal_lmtd(state) -> float | None:
    approaches = _recovery_terminal_approaches(state)
    if len(approaches) != 2:
        return None
    delta_1, delta_2 = approaches
    if delta_1 <= 0.0 or delta_2 <= 0.0:
        return None
    if _close(delta_1, delta_2, abs_tol=_TEMPERATURE_ABS_TOL):
        return (delta_1 + delta_2) / 2.0
    return (delta_1 - delta_2) / math.log(delta_1 / delta_2)


def _count_active(
    exchangers: Iterable[HeatExchanger],
    kind: HeatExchangerKind | None,
) -> int:
    return sum(
        1
        for exchanger in exchangers
        if any(state.active for state in exchanger.period_states)
        and (kind is None or exchanger.kind is kind)
    )


def _stream_value_map(
    network: HeatExchangerNetwork,
    axis_name: str,
    values,
) -> dict[str, float]:
    stream_order = _stream_order(network, axis_name)
    if not stream_order or not isinstance(values, list | tuple):
        return {}
    return {
        stream: float(values[index])
        for index, stream in enumerate(stream_order)
        if index < len(values) and _finite(values[index])
    }


def _period_stream_value_map(
    network: HeatExchangerNetwork,
    axis_name: str,
    *,
    metadata: dict,
    period_id: str,
    scalar_key: str,
    period_key: str,
) -> dict[str, float]:
    period_values = metadata.get(period_key)
    if isinstance(period_values, list | tuple):
        period_idx = network.period_ids.index(period_id)
        if period_idx < len(period_values):
            return _stream_value_map(network, axis_name, period_values[period_idx])
    return _stream_value_map(network, axis_name, metadata.get(scalar_key))


def _stream_order(
    network: HeatExchangerNetwork,
    axis_name: str,
) -> tuple[str, ...]:
    axis_map = network.solver_axis_metadata.get("axis_maps", {}).get(axis_name, {})
    if not axis_map:
        return ()
    return ordered_mapping_keys(axis_map)


def _uses_isothermal_stage_boundaries(metadata: dict) -> bool:
    """Return whether extraction lacks per-match outlet temperatures.

    PDM and TDM use the source OpenHENS isothermal stage-wise seed model. Those
    intermediate networks expose stage-boundary temperatures, not unique outlet
    temperatures for every match inside a mixed stage. Stream-level heat
    balances still apply, while per-exchanger duty and LMTD checks are reserved
    for non-isothermal EVM candidates or hand-built network data.
    """

    if metadata.get("solver_non_isothermal_model", True):
        return False
    return metadata.get("solver_framework") in {"PDM", "TDM"}


def _heat_removed(
    heat_capacity_flowrate: float,
    inlet_temperature: float | None,
    outlet_temperature: float | None,
) -> float | None:
    if inlet_temperature is None or outlet_temperature is None:
        return None
    return heat_capacity_flowrate * (inlet_temperature - outlet_temperature)


def _heat_added(
    heat_capacity_flowrate: float,
    inlet_temperature: float | None,
    outlet_temperature: float | None,
) -> float | None:
    if inlet_temperature is None or outlet_temperature is None:
        return None
    return heat_capacity_flowrate * (outlet_temperature - inlet_temperature)


def _recovery_terminal_approaches(state) -> tuple[float, ...]:
    values = (
        _temperature_difference(
            state.source_inlet_temperature,
            state.sink_outlet_temperature,
        ),
        _temperature_difference(
            state.source_outlet_temperature,
            state.sink_inlet_temperature,
        ),
    )
    return tuple(value for value in values if value is not None)


def _temperature_difference(
    hot_temperature: float | None,
    cold_temperature: float | None,
) -> float | None:
    if hot_temperature is None or cold_temperature is None:
        return None
    return hot_temperature - cold_temperature


def _duty_failure(
    exchanger: HeatExchanger,
    state,
    side: str,
    expected: float | None,
) -> list[str]:
    if expected is None:
        return []
    if _close(expected, state.duty, abs_tol=_DUTY_ABS_TOL):
        return []
    return [
        f"{exchanger.exchanger_id or exchanger.kind.value} {side} heat balance "
        f"{expected:.6g} does not match duty {state.duty:.6g}"
    ]


def _close(
    expected: float,
    observed: float,
    *,
    rel_tol: float = _DUTY_REL_TOL,
    abs_tol: float,
) -> bool:
    return math.isclose(expected, observed, rel_tol=rel_tol, abs_tol=abs_tol)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except TypeError, ValueError:
        return False


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


__all__ = [
    "is_network_feasible",
    "verify_network_feasibility",
    "verify_synthesis_result",
]
