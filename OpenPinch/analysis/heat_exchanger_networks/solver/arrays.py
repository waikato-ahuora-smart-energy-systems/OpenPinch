"""Private adapter from prepared :class:`PinchProblem` state to solver arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ....application.problem import PinchProblem
from ....domain._stream.value_state import resolve_period_weights
from ....domain.configuration import tol
from ....domain.stream import Stream
from ....domain.value import Value
from ....domain.zone import Zone
from ..indexing import ordered_mapping_keys

_PreparedItem = tuple[str, Stream, Any | None]
SEGMENT_PROFILE_VERSION = 2


@dataclass(frozen=True)
class PreparedSolverArrays:
    """Immutable private solver-array data for migrated equation models."""

    arrays: dict[str, np.ndarray]
    axis_maps: dict[str, dict[str, int]]
    unit_conventions: dict[str, str]
    stream_identities: dict[str, list[str]]
    utility_identities: dict[str, list[str]]
    configuration: dict[str, Any]
    preparation: dict[str, Any]

    @property
    def array_shapes(self) -> dict[str, list[int]]:
        """Return JSON-compatible shapes for each private array."""
        return {name: list(values.shape) for name, values in self.arrays.items()}

    def to_json_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "arrays": {
                name: self.arrays[name].tolist() for name in sorted(self.arrays)
            },
            "array_shapes": {
                name: self.array_shapes[name] for name in sorted(self.array_shapes)
            },
            "axis_maps": self.axis_maps,
            "configuration": self.configuration,
            "preparation": self.preparation,
            "stream_identities": self.stream_identities,
            "unit_conventions": self.unit_conventions,
            "utility_identities": self.utility_identities,
        }


def problem_to_solver_arrays(
    problem: PinchProblem,
    dTmin: float,
) -> PreparedSolverArrays:
    """Build the private solver-array data from a prepared ``PinchProblem``.

    The adapter intentionally accepts only a live ``PinchProblem`` so raw fixture
    rows, raw ``TargetInput`` dictionaries, heat exchanger network schemas, cached
    array dictionaries, and standalone DTOs cannot bypass OpenPinch validation
    and preparation.
    """

    if not isinstance(problem, PinchProblem):
        raise TypeError(
            "problem_to_solver_arrays requires a prepared PinchProblem; "
            f"got {type(problem).__name__}."
        )
    if not isinstance(problem.master_zone, Zone):
        raise RuntimeError(
            "problem_to_solver_arrays requires PinchProblem.load(...) and "
            "prepare_problem(...) to create a prepared root Zone first."
        )
    if not np.isfinite(dTmin) or dTmin <= 0.0:
        raise ValueError("dTmin must be finite and positive.")

    zone = problem.master_zone
    hot_items = _ordered_stream_items(problem, zone.hot_streams.items())
    cold_items = _ordered_stream_items(problem, zone.cold_streams.items())
    hot_utility_items = _ordered_utility_items(problem, zone.hot_utilities.items())
    cold_utility_items = _ordered_utility_items(problem, zone.cold_utilities.items())

    if not hot_items or not cold_items:
        raise ValueError("prepared PinchProblem must contain hot and cold streams.")
    if not hot_utility_items or not cold_utility_items:
        raise ValueError("prepared PinchProblem must contain hot and cold utilities.")

    config = zone.config
    costing = config.costing
    hens = config.hens
    period_ids, period_weights = _period_ids_and_weights(zone)
    num_periods = len(period_ids)
    arrays = _solver_array_mapping(
        cold_items=cold_items,
        cold_utility_items=cold_utility_items,
        costing=costing,
        dTmin=dTmin,
        hot_items=hot_items,
        hot_utility_items=hot_utility_items,
        num_periods=num_periods,
        period_ids=period_ids,
        period_weights=period_weights,
        zone=zone,
    )
    arrays.update(_segment_profile_arrays("hot", hot_items, num_periods, dTmin))
    arrays.update(_segment_profile_arrays("cold", cold_items, num_periods, dTmin))
    arrays.update(
        _segment_profile_arrays("hot_utility", hot_utility_items, num_periods, dTmin)
    )
    arrays.update(
        _segment_profile_arrays("cold_utility", cold_utility_items, num_periods, dTmin)
    )

    axis_maps = {
        "cold_process_streams": {
            key: index for index, (key, _, _) in enumerate(cold_items)
        },
        "cold_utilities": {
            key: index for index, (key, _, _) in enumerate(cold_utility_items)
        },
        "hot_process_streams": {
            key: index for index, (key, _, _) in enumerate(hot_items)
        },
        "hot_utilities": {
            key: index for index, (key, _, _) in enumerate(hot_utility_items)
        },
        "stages": {
            str(stage): index for index, stage in enumerate(hens.stage_selection)
        },
        "periods": {period_id: index for index, period_id in enumerate(period_ids)},
        "hot_segments": _segment_axis_map(hot_items),
        "cold_segments": _segment_axis_map(cold_items),
        "cold_utility_segments": _segment_axis_map(cold_utility_items),
        "hot_utility_segments": _segment_axis_map(hot_utility_items),
    }

    return PreparedSolverArrays(
        arrays=arrays,
        axis_maps=axis_maps,
        configuration={
            "HENS_APPROACH_TEMPERATURES": list(hens.approach_temperatures),
            "HENS_BEST_SOLUTIONS_TO_SAVE": hens.best_solutions_to_save,
            "HENS_DERIVATIVE_THRESHOLDS": list(hens.derivative_thresholds),
            "HENS_DT_CONT_MULTIPLIERS": getattr(
                hens,
                "dt_cont_multipliers",
                None,
            ),
            "HENS_SYNTHESIS_QUALITY_TIER": hens.synthesis_quality_tier,
            "HENS_PDM_STAGE_PAIR_LIMIT": hens.pdm_stage_pair_limit,
            "HENS_TDM_PARENT_LIMIT": hens.tdm_parent_limit,
            "HENS_STAGE_PACKING": hens.stage_packing,
            "HENS_LOG_LEVEL": hens.log_level,
            "HENS_MAX_PARALLEL": hens.max_parallel,
            "HENS_EVM_N_AD_BRANCHES": hens.evm_n_ad_branches,
            "HENS_EVM_N_RM_BRANCHES": hens.evm_n_rm_branches,
            "HENS_METHOD_SEQUENCE": list(hens.method_sequence),
            "HENS_OUTPUT_FOLDER": hens.output_folder,
            "HENS_OUTPUT_FORMATS": list(hens.output_formats),
            "HENS_RUN_ID": hens.run_id,
            "HENS_SOLVE_TOLERANCE": hens.solve_tolerance,
            "HENS_SOLVER_EVM": hens.solver_evm,
            "HENS_SOLVER_OPTIONS_EVM": dict(hens.solver_options_evm),
            "HENS_SOLVER_OPTIONS_PDM": dict(hens.solver_options_pdm),
            "HENS_SOLVER_OPTIONS_TDM": dict(hens.solver_options_tdm),
            "HENS_SOLVER_PDM": hens.solver_pdm,
            "HENS_SOLVER_TDM": hens.solver_tdm,
            "HENS_STAGE_SELECTION": list(hens.stage_selection),
            "active_dTmin": float(dTmin),
            "segment_profile_version": SEGMENT_PROFILE_VERSION,
            "active_dt_cont_multiplier": float(dTmin),
            "period_ids": list(period_ids),
            "period_weights": [float(weight) for weight in period_weights],
            "costing": {
                "COSTING_HX_AREA_COEFF": costing.hx_area_coeff,
                "COSTING_HX_AREA_EXP": costing.hx_area_exp,
                "COSTING_HX_UNIT_COST": costing.hx_unit_cost,
            },
        },
        preparation={
            "pinch_problem_class": type(problem).__name__,
            "prepared_zone_class": type(zone).__name__,
            "process_stream_collection_class": type(zone.process_streams).__name__,
            "stream_class": Stream.__name__,
            "zone_name": zone.name,
        },
        stream_identities={
            "cold_process_streams": [key for key, _, _ in cold_items],
            "hot_process_streams": [key for key, _, _ in hot_items],
            "cold_segments": _segment_identity_list(cold_items),
            "hot_segments": _segment_identity_list(hot_items),
            "cold_utility_segments": _segment_identity_list(cold_utility_items),
            "hot_utility_segments": _segment_identity_list(hot_utility_items),
        },
        unit_conventions={
            "cost_coefficients": (
                "OpenPinch costing config: COSTING_HX_UNIT_COST, "
                "COSTING_HX_AREA_COEFF, COSTING_HX_AREA_EXP"
            ),
            "heat_capacity_flowrate": "kW/K",
            "heat_transfer_coefficient": "kW/m^2/K",
            "temperature": "K",
            "temperature_contribution": (
                "K, using prepared stream dt_cont multiplied by the active HEN "
                "sweep value and falling back to dTmin / 2 when prepared "
                "contribution is absent"
            ),
            "utility_price": (
                "numeric UtilitySchema.price passed through for OpenHENS "
                "annual cost equations"
            ),
        },
        utility_identities={
            "cold_utilities": [key for key, _, _ in cold_utility_items],
            "hot_utilities": [key for key, _, _ in hot_utility_items],
        },
    )


# fmt: off
def _solver_array_mapping(
    *,
    cold_items: list[_PreparedItem],
    cold_utility_items: list[_PreparedItem],
    costing: Any,
    dTmin: float,
    hot_items: list[_PreparedItem],
    hot_utility_items: list[_PreparedItem],
    num_periods: int,
    period_ids: tuple[str, ...],
    period_weights: tuple[float, ...],
    zone: Zone,
) -> dict[str, np.ndarray]:
    temp_unit = "K"
    price_unit = "$/MW/h"
    htc_unit = "kW/m^2/delta_degC"
    periods = range(num_periods)

    def period_values(items: list[_PreparedItem], getter) -> np.ndarray:
        return _state_float_matrix(
            (getter(stream, record, n) for _, stream, record in items)
            for n in periods
        )

    def stream_attr(attr: str, unit: str):
        def getter(stream: Stream, _record, n: int) -> float:
            return _value(getattr(stream, attr), unit, period_idx=n)

        return getter

    def temperature_contribution(stream: Stream, _record, n: int) -> float:
        return _temperature_contribution(stream, dTmin, period_idx=n)

    def utility_solver_target(stream: Stream, _record, n: int) -> float:
        return _utility_solver_target(stream, zone, period_idx=n)

    def heat_capacity_flowrate(stream: Stream, record, n: int) -> float:
        return _stream_heat_capacity_flowrate(stream, record, period_idx=n)

    cu = cold_utility_items
    hu = hot_utility_items

    return {
        "A_coeff": _float_array([costing.hx_area_coeff]),
        "A_exp": _float_array([costing.hx_area_exp]),
        "period_ids": _str_array(period_ids),
        "period_weights": _float_array(period_weights),
        "T_c_cont_period": period_values(cold_items, temperature_contribution),
        "T_c_in_period": period_values(cold_items, stream_attr("t_supply", temp_unit)),
        "T_c_out_period": period_values(cold_items, stream_attr("t_target", temp_unit)),
        "T_cu_in_period": period_values(cu, stream_attr("t_supply", temp_unit)),
        "T_cu_out_period": period_values(cold_utility_items, utility_solver_target),
        "T_cu_cont_period": period_values(cold_utility_items, temperature_contribution),
        "T_h_cont_period": period_values(hot_items, temperature_contribution),
        "T_h_in_period": period_values(hot_items, stream_attr("t_supply", temp_unit)),
        "T_h_out_period": period_values(hot_items, stream_attr("t_target", temp_unit)),
        "T_hu_in_period": period_values(hu, stream_attr("t_supply", temp_unit)),
        "T_hu_out_period": period_values(hot_utility_items, utility_solver_target),
        "T_hu_cont_period": period_values(hot_utility_items, temperature_contribution),
        "c_cost_period": period_values(cold_items, stream_attr("price", price_unit)),
        "cold_names": _str_array(stream.name for _, stream, _ in cold_items),
        "cu_coeff": _float_array([costing.hx_area_coeff]),
        "cu_cost_period": period_values(cu, stream_attr("price", price_unit)),
        "cu_exp": _float_array([costing.hx_area_exp]),
        "cu_unit_cost": _float_array([costing.hx_unit_cost]),
        "f_c_period": period_values(cold_items, heat_capacity_flowrate),
        "f_h_period": period_values(hot_items, heat_capacity_flowrate),
        "h_cost_period": period_values(hot_items, stream_attr("price", price_unit)),
        "hot_names": _str_array(stream.name for _, stream, _ in hot_items),
        "htc_c_period": period_values(cold_items, stream_attr("htc", htc_unit)),
        "htc_cu_period": period_values(
            cold_utility_items, stream_attr("htc", htc_unit)
        ),
        "htc_h_period": period_values(hot_items, stream_attr("htc", htc_unit)),
        "htc_hu_period": period_values(
            hot_utility_items, stream_attr("htc", htc_unit)
        ),
        "hu_coeff": _float_array([costing.hx_area_coeff]),
        "hu_cost_period": period_values(hu, stream_attr("price", price_unit)),
        "hu_exp": _float_array([costing.hx_area_exp]),
        "hu_unit_cost": _float_array([costing.hx_unit_cost]),
        "unit_cost": _float_array([costing.hx_unit_cost]),
    }
# fmt: on


def _segment_profile_arrays(
    prefix: str,
    items: list[_PreparedItem],
    num_periods: int,
    dTmin: float,
) -> dict[str, np.ndarray]:
    """Build padded segment tensors while keeping the solver parent axes intact."""
    segment_rows = [_segments_for_solver(stream) for _, stream, _ in items]
    max_segments = max(len(row) for row in segment_rows)
    shape = (num_periods, len(items), max_segments)
    temperatures_in = np.zeros(shape, dtype=float)
    temperatures_out = np.zeros(shape, dtype=float)
    duties = np.zeros(shape, dtype=float)
    heat_capacity_flowrates = np.zeros(shape, dtype=float)
    heat_transfer_coefficients = np.zeros(shape, dtype=float)
    prices = np.zeros(shape, dtype=float)
    temperature_contributions = np.zeros(shape, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    cumulative_duties = np.zeros(
        (num_periods, len(items), max_segments + 1), dtype=float
    )
    identities = np.full((len(items), max_segments), "", dtype=object)

    for parent_index, ((parent_key, _stream, _record), segments) in enumerate(
        zip(items, segment_rows)
    ):
        for segment_index, segment in enumerate(segments):
            identities[parent_index, segment_index] = _segment_identity(
                parent_key,
                segment_index,
            )
            for period_index in range(num_periods):
                mask[period_index, parent_index, segment_index] = True
                temperatures_in[period_index, parent_index, segment_index] = _value(
                    segment.t_supply, "K", period_idx=period_index
                )
                temperatures_out[period_index, parent_index, segment_index] = _value(
                    segment.t_target, "K", period_idx=period_index
                )
                duties[period_index, parent_index, segment_index] = _value(
                    segment.heat_flow, "kW", period_idx=period_index
                )
                heat_capacity_flowrates[period_index, parent_index, segment_index] = (
                    _value(segment.CP, "kW/delta_degC", period_idx=period_index)
                )
                heat_transfer_coefficients[
                    period_index, parent_index, segment_index
                ] = _value(
                    segment.htc,
                    "kW/m^2/delta_degC",
                    period_idx=period_index,
                )
                prices[period_index, parent_index, segment_index] = _value(
                    segment.price,
                    "$/MW/h",
                    period_idx=period_index,
                )
                temperature_contributions[period_index, parent_index, segment_index] = (
                    _temperature_contribution(
                        segment,
                        dTmin,
                        period_idx=period_index,
                    )
                )
        cumulative_duties[:, parent_index, 1:] = np.cumsum(
            duties[:, parent_index, :], axis=1
        )

    return {
        f"{prefix}_segment_count": np.array(
            [len(row) for row in segment_rows], dtype=int
        ),
        f"{prefix}_segment_mask_period": mask,
        f"{prefix}_segment_t_in_period": temperatures_in,
        f"{prefix}_segment_t_out_period": temperatures_out,
        f"{prefix}_segment_duty_period": duties,
        f"{prefix}_segment_cp_period": heat_capacity_flowrates,
        f"{prefix}_segment_htc_period": heat_transfer_coefficients,
        f"{prefix}_segment_price_period": prices,
        f"{prefix}_segment_dt_cont_period": temperature_contributions,
        f"{prefix}_segment_cumulative_duty_period": cumulative_duties,
        f"{prefix}_segment_identities": identities,
        f"{prefix}_parent_segmented": np.asarray(
            [stream.has_segments for _, stream, _ in items],
            dtype=bool,
        ),
    }


def _segments_for_solver(stream: Stream) -> tuple[Stream, ...]:
    return stream.segments if stream.has_segments else (stream,)


def _segment_axis_map(items: list[_PreparedItem]) -> dict[str, int]:
    return {
        identity: flat_index
        for flat_index, identity in enumerate(_segment_identity_list(items))
    }


def _segment_identity_list(items: list[_PreparedItem]) -> list[str]:
    identities = []
    for parent_key, stream, _record in items:
        identities.extend(
            _segment_identity(parent_key, segment_index)
            for segment_index in range(len(_segments_for_solver(stream)))
        )
    return identities


def _segment_identity(parent_key: str, segment_index: int) -> str:
    """Return a canonical collision-free identity stable across periods."""
    return f"{parent_key}.S{segment_index + 1}"


def _period_ids_and_weights(zone: Zone) -> tuple[tuple[str, ...], tuple[float, ...]]:
    period_lookup = zone.period_ids or {"0": 0}
    ordered_period_ids = ordered_mapping_keys(period_lookup)
    period_weights = tuple(
        float(weight)
        for weight in resolve_period_weights(
            ordered_period_ids,
            getattr(zone, "weights", None),
        )
    )
    return ordered_period_ids, period_weights


def _temperature_contribution(
    stream: Stream,
    dTmin: float,
    *,
    period_idx: int = 0,
) -> float:
    contribution = _value(stream.dt_cont, "delta_degC", period_idx=period_idx)
    if contribution > tol:
        return contribution * float(dTmin)
    return float(dTmin) / 2.0


def _utility_solver_target(
    stream: Stream,
    zone: Zone,
    *,
    period_idx: int = 0,
) -> float:
    supply = _value(stream.t_supply, "K", period_idx=period_idx)
    target = _value(stream.t_target, "K", period_idx=period_idx)
    if abs(supply - target) <= zone.config.thermal.dt_phase_change + tol:
        return supply
    return target


def _stream_heat_capacity_flowrate(
    stream: Stream,
    record,
    *,
    period_idx: int = 0,
) -> float:
    if getattr(stream, "has_segments", False):
        # Parent axes retain this legacy array only for shape compatibility.
        # A zero sentinel prevents segmented equations from accidentally using
        # the parent's effective CP instead of the segment-profile tensors.
        return 0.0
    raw_flowrate = getattr(record, "heat_capacity_flowrate", None)
    if raw_flowrate is not None:
        parsed = _optional_value(
            raw_flowrate,
            "kW/delta_degC",
            period_idx=period_idx,
        )
        if parsed is not None:
            return parsed
    return _value(stream.CP, "kW/delta_degC", period_idx=period_idx)


def _ordered_stream_items(problem: PinchProblem, items) -> list[_PreparedItem]:
    item_list = list(items)
    input_streams = getattr(getattr(problem, "_validated_data", None), "streams", ())
    return _items_in_input_order(item_list, input_streams)


def _ordered_utility_items(problem: PinchProblem, items) -> list[_PreparedItem]:
    item_list = list(items)
    input_utilities = getattr(
        getattr(problem, "_validated_data", None), "utilities", ()
    )
    ordered = _items_in_input_order(item_list, input_utilities)
    if len(ordered) <= 1:
        return ordered
    real_utilities = [
        item
        for item in ordered
        if not (item[1].name == "HU" and item[0].endswith(".HU"))
    ]
    return real_utilities or ordered


def _items_in_input_order(
    items: list[tuple[str, Stream]],
    input_records,
) -> list[_PreparedItem]:
    if not input_records:
        return [(key, stream, None) for key, stream in items]

    remaining = list(items)
    ordered: list[_PreparedItem] = []
    for record in input_records:
        match_index = _matching_item_index(remaining, record)
        if match_index is None:
            continue
        key, stream = remaining.pop(match_index)
        ordered.append((key, stream, record))
    ordered.extend((key, stream, None) for key, stream in remaining)
    return ordered


def _matching_item_index(
    items: list[tuple[str, Stream]],
    record,
) -> int | None:
    record_name = str(getattr(record, "name", "") or "").strip()
    record_zone = str(getattr(record, "zone", "") or "").strip()
    zone_leaf = record_zone.split("/")[-1] if record_zone else ""
    for index, (key, stream) in enumerate(items):
        if stream.name != record_name:
            continue
        if not zone_leaf or key.startswith(f"{zone_leaf}."):
            return index
    for index, (_key, stream) in enumerate(items):
        if stream.name == record_name:
            return index
    return None


def _value(value: Any, unit: str, *, period_idx: int = 0) -> float:
    parsed = _optional_value(value, unit, period_idx=period_idx)
    if parsed is None:
        return 0.0
    return parsed


def _optional_value(value: Any, unit: str, *, period_idx: int = 0) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to"):
        converted = value.to(unit)
        if getattr(converted, "num_periods", 1) > 1:
            return float(converted[period_idx].value)
        return float(converted.value)
    if hasattr(value, "values"):
        raw_values = getattr(value, "values")
        if raw_values is None:
            return None
        raw_value = raw_values[period_idx] if len(raw_values) > 1 else raw_values[0]
        raw_unit = getattr(value, "unit", None)
        if raw_unit:
            return float(Value(raw_value, raw_unit).to(unit).value)
        return float(raw_value)
    if hasattr(value, "value"):
        raw_value = getattr(value, "value")
        if raw_value is None:
            return None
        if isinstance(raw_value, (list, tuple, np.ndarray)):
            raw_value = raw_value[period_idx]
        raw_unit = getattr(value, "unit", None)
        if raw_unit:
            return float(Value(raw_value, raw_unit).to(unit).value)
        return float(raw_value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[period_idx])
    return float(value)


def _float_array(values) -> np.ndarray:
    return np.array(list(values), dtype=float)


def _state_float_matrix(rows) -> np.ndarray:
    return np.array([list(row) for row in rows], dtype=float)


def _str_array(values) -> np.ndarray:
    return np.array(list(values), dtype=str)


__all__ = [
    "SEGMENT_PROFILE_VERSION",
    "PreparedSolverArrays",
    "problem_to_solver_arrays",
]
