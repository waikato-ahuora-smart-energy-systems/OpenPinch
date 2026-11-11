"""Cascade vapour-compression HP targeting."""

from __future__ import annotations

from functools import partial

import CoolProp
import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRActiveStage,
    HPRBackendResult,
    HPREffectiveTopology,
    HPREvaluationMode,
    HPRParsedState,
    HPRTopologyIdentifier,
)
from ....domain.configuration import tol
from ....domain.fluids import build_coolprop_abstract_state
from ....domain.stream import Stream
from ....domain.stream_collection import StreamCollection
from ..common._shared.streams import get_Q_vals_at_T_hpr_from_bckgrd_profile
from ..common.encoding import (
    DutyAllocation,
    DutyAllocationRequest,
    decode_available_fractions,
    encode_available_fractions,
    limit_available_duty,
    map_T_arr_to_x_arr,
    map_x_arr_to_DT_arr,
    map_x_arr_to_T_arr,
    require_stage_duty_allocation,
)
from ..common.layout import HPRoptVectorLayout, clip_to_bounds
from ..common.multi_vc_objective import _evaluate_multi_vc_objective
from ..common.shared import (
    cap_stage_condensing_temperatures,
    condensing_temperature_search_range,
    cycle_limits,
    evaluate_vapour_hpr_result,
    floor_stage_evaporating_temperatures,
    limit_cascade_pressure_ratios,
    raise_cascade_lifts_to_floor,
    subcritical_stage_condensing_temperatures,
    validate_vapour_hp_refrigerant_ls,
)
from ..common.topology import topology_from_stages
from ..cycles.cascade_vapour_compression_cycle import CascadeVapourCompressionCycle
from ..optimisation_adapter import initialise_hpr_seed, solve_hpr_placement
from ..performance_maps.coolprop_preflight import (
    preflight_coolprop_fluid_names,
    preflight_coolprop_hpr_targeting,
    representative_hpr_point,
)
from ..performance_maps.target_records import (
    build_coolprop_target_simulation_record,
    build_tespy_target_simulation_record,
)
from ..performance_maps.targeting import (
    HprTargetEvaluatorCoordinator,
    PreparedTespyHprTargeting,
    get_hpr_target_evaluator,
    preflight_tespy_hpr_targeting,
)
from ..performance_maps.targeting_models import (
    HprTargetEvaluationFailure,
    HprTargetThermodynamicRequest,
    HprTargetThermodynamicResult,
    HprThermalProfilePoint,
)
from .cascade_carnot import (
    optimise_cascade_carnot_heat_pump_placement,
)

__all__ = [
    "optimise_cascade_heat_pump_placement",
]

_TESPY_TARGET_SUPERHEAT = 5.0


################################################################################
# Public API
################################################################################


def optimise_cascade_heat_pump_placement(
    args: HeatPumpTargetInputs,
    *,
    prepared_tespy: PreparedTespyHprTargeting | None = None,
) -> HPRBackendResult:
    """Optimise a cascade vapour-compression placement for the prepared HPR case."""
    if getattr(args, "simulation_backend", "coolprop") == "tespy":
        prepared = (
            preflight_tespy_hpr_targeting(args)
            if prepared_tespy is None
            else prepared_tespy
        )
        return _optimise_tespy_cascade_heat_pump_placement(args, prepared)

    num_stages = int(args.n_cond + args.n_evap - 1)
    preflight_coolprop_fluid_names(args, _cascade_topology_id(args))
    init_res = initialise_hpr_seed(optimise_cascade_carnot_heat_pump_placement, args)

    args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(num_stages, args)
    x0_ls, bnds = _get_cascade_hp_opt_setup(init_res, args)
    warm_state = _parse_cascade_hp_state_variables(
        representative_hpr_point(x0_ls, bnds), args
    )
    preflight_coolprop_hpr_targeting(
        args=args,
        state=_cascade_stage_preflight_state(warm_state, args),
        topology_id=_cascade_topology_id(args),
    )
    return solve_hpr_placement(
        f_obj=_compute_cascade_hp_system_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


def _optimise_tespy_cascade_heat_pump_placement(
    args: HeatPumpTargetInputs,
    prepared: PreparedTespyHprTargeting,
) -> HPRBackendResult:
    """Run one scalar placement with a call-owned TESPy evaluator session."""
    init_res = initialise_hpr_seed(optimise_cascade_carnot_heat_pump_placement, args)
    args.refrigerant_ls = [prepared.working_fluid.source_spec]
    x0_ls, bnds = _get_cascade_hp_opt_setup(init_res, args)
    coordinator = HprTargetEvaluatorCoordinator(get_hpr_target_evaluator(prepared))
    coordinator.open()
    try:
        return solve_hpr_placement(
            f_obj=partial(
                _compute_tespy_cascade_hp_system_obj,
                coordinator=coordinator,
                prepared=prepared,
            ),
            x0_ls=x0_ls,
            bnds=bnds,
            args=args,
        )
    finally:
        coordinator.close()


################################################################################
# Helper Functions
################################################################################


def _cascade_topology_id(args: HeatPumpTargetInputs) -> HPRTopologyIdentifier:
    if int(args.n_cond) == 1 and int(args.n_evap) == 1:
        return HPRTopologyIdentifier.SINGLE_STAGE_VAPOUR_COMPRESSION
    return HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION


def _get_cascade_hp_opt_setup(
    init_res: HPRBackendResult | None,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray | None, list]:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    n_cond = int(args.n_cond)
    n_evap = int(args.n_evap)
    n_heat, n_cool = _cascade_process_control_counts(
        n_cond=n_cond,
        n_evap=n_evap,
        is_heat_pumping=is_heat_pumping,
    )
    layout = HPRoptVectorLayout(
        n_cond=n_cond,
        n_evap=n_evap,
        n_subcool=n_cond,
        n_heat_split=n_heat,
        n_cool_split=n_cool,
        n_ihx=n_cond + n_evap - 1,
    )
    bnds = layout.build_bounds(
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_subcool=(0.0, 1.0),
        x_heat_split=(0.0, 1.0),
        x_cool_split=(0.0, 1.0),
        x_ihx=(0.0, 1.0),
    )
    if init_res is None:
        return None, bnds

    # The Carnot warm start knows nothing about critical points, so clip it
    # into the refrigerant's subcritical condensing range.
    x_cond = np.clip(
        map_T_arr_to_x_arr(init_res.T_cond, *condensing_temperature_search_range(args)),
        0.0,
        1.0,
    ).tolist()
    x_evap = map_T_arr_to_x_arr(
        init_res.T_evap[::-1], args.T_hot[-1], args.T_hot[0]
    ).tolist()
    # Nor about pressure ratios, which the search penalises: seed within the
    # limit so even a small search starts from an unpenalised design.
    T_cond_seed, T_evap_seed = limit_cascade_pressure_ratios(
        subcritical_stage_condensing_temperatures(
            map_x_arr_to_T_arr(
                np.asarray(x_cond), *condensing_temperature_search_range(args)
            ),
            args,
        ),
        floor_stage_evaporating_temperatures(
            map_x_arr_to_T_arr(np.asarray(x_evap), args.T_hot[-1], args.T_hot[0]),
            args,
            first_stage=n_cond - 1,
        ),
        args,
    )
    x_cond = np.clip(
        map_T_arr_to_x_arr(T_cond_seed, *condensing_temperature_search_range(args)),
        0.0,
        1.0,
    ).tolist()
    x_evap = map_T_arr_to_x_arr(
        T_evap_seed[::-1], args.T_hot[-1], args.T_hot[0]
    ).tolist()
    x_subcool = [0.0] * int(args.n_cond)
    init_heat = init_res.Q_cond if is_heat_pumping else init_res.Q_cond[1:]
    init_cool = init_res.Q_evap[:n_cool] if is_heat_pumping else init_res.Q_evap
    x_ihx = [0.0] * (int(args.n_cond) + int(args.n_evap) - 1)
    pack_kwargs = {
        "x_cond": x_cond,
        "x_evap": x_evap,
        "x_subcool": x_subcool,
        "x_ihx": x_ihx,
    }
    if n_heat:
        pack_kwargs["x_heat_split"] = np.ones(n_heat, dtype=float)
    if n_cool:
        pack_kwargs["x_cool_split"] = np.ones(n_cool, dtype=float)
    # Encode the seed duties against the availability at the seed's own
    # temperatures, exactly as the search will decode them.
    state = _parse_cascade_hp_state_variables(layout.pack(**pack_kwargs), args)
    if n_heat:
        pack_kwargs["x_heat_split"] = encode_available_fractions(
            np.asarray(init_heat, dtype=float)[:n_heat], state.Q_heat_available
        )
    if n_cool:
        pack_kwargs["x_cool_split"] = encode_available_fractions(
            np.asarray(init_cool, dtype=float)[:n_cool], state.Q_cool_available
        )
    return clip_to_bounds(layout.pack(**pack_kwargs), bnds), bnds


def _parse_cascade_hp_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    n_cond = int(args.n_cond)
    n_evap = int(args.n_evap)
    n_heat, n_cool = _cascade_process_control_counts(
        n_cond=n_cond,
        n_evap=n_evap,
        is_heat_pumping=is_heat_pumping,
    )
    parts = HPRoptVectorLayout(
        n_cond=n_cond,
        n_evap=n_evap,
        n_subcool=n_cond,
        n_heat_split=n_heat,
        n_cool_split=n_cool,
        n_ihx=n_cond + n_evap - 1,
    ).unpack(x)
    x_cond = parts["x_cond"]
    x_evap = parts["x_evap"]
    x_subcool = parts["x_subcool"]
    x_heat_split = parts["x_heat_split"]
    x_cool_split = parts["x_cool_split"]
    x_ihx = parts["x_ihx"]

    T_cond = cap_stage_condensing_temperatures(
        map_x_arr_to_T_arr(x_cond, *condensing_temperature_search_range(args)), args
    )
    T_evap = floor_stage_evaporating_temperatures(
        map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0]),
        args,
        # Process evaporators sit on the coldest stages.
        first_stage=n_cond - 1,
    )
    if getattr(args, "stage_lift_floor", None) is not None:
        # A shared multiperiod machine cannot turn its lift below its floor.
        # The floors are physical lifts; the levels are process levels, which
        # the cascade exchangers widen.
        T_cond, T_evap = raise_cascade_lifts_to_floor(
            T_cond,
            T_evap,
            np.asarray(args.stage_lift_floor, dtype=float)
            - _cascade_exchanger_lift(T_cond.size, T_evap.size, args),
        )
    dT_subcool = map_x_arr_to_DT_arr(
        x_subcool,
        T_cond,
        _condenser_stage_evaporating_temperatures(T_cond, T_evap, args),
    )
    Q_heat_available = (
        limit_available_duty(
            get_Q_vals_at_T_hpr_from_bckgrd_profile(
                T_cond if is_heat_pumping else T_cond[1:],
                args.T_cold,
                args.H_cold,
                is_cond=True,
            ),
            args.Q_heat_max,
        )
        if n_heat
        else None
    )
    Q_cool_available = (
        limit_available_duty(
            get_Q_vals_at_T_hpr_from_bckgrd_profile(
                T_evap[:n_cool] if is_heat_pumping else T_evap,
                args.T_hot,
                args.H_hot,
                is_cond=False,
            ),
            args.Q_cool_max,
        )
        if n_cool
        else None
    )
    Q_heat_base = (
        float(decode_available_fractions(x_heat_split, Q_heat_available).sum())
        if n_heat
        else None
    )
    Q_cool_base = (
        float(decode_available_fractions(x_cool_split, Q_cool_available).sum())
        if n_cool
        else None
    )
    T_cond_all = np.sort(np.concatenate([T_cond - dT_subcool, T_evap[:-1]]))[::-1]
    T_evap_all = np.sort(np.concatenate([(T_cond - dT_subcool)[1:], T_evap]))[::-1]
    dT_ihx_gas_side = map_x_arr_to_DT_arr(x_ihx, T_cond_all, T_evap_all)
    return HPRParsedState(
        T_cond=T_cond,
        dT_subcool=dT_subcool,
        T_evap=T_evap,
        dT_ihx_gas_side=dT_ihx_gas_side,
        Q_heat_base=Q_heat_base,
        Q_cool_base=Q_cool_base,
        x_heat_split=x_heat_split if n_heat else None,
        x_cool_split=x_cool_split if n_cool else None,
        Q_heat_available=Q_heat_available,
        Q_cool_available=Q_cool_available,
    )


def _cascade_stage_lifts(x: np.ndarray, args: HeatPumpTargetInputs) -> np.ndarray:
    """Each stage's physical lift as ``x`` decodes, hottest stage first.

    That is the gap between its process levels plus any cascade exchanger
    approach it lifts across.
    """
    state = _parse_cascade_hp_state_variables(x, args)
    levels = np.concatenate([state.T_cond, state.T_evap])
    return (
        levels[:-1]
        - levels[1:]
        + _cascade_exchanger_lift(state.T_cond.size, state.T_evap.size, args)
    )


def _cascade_exchanger_lift(
    n_cond: int, n_evap: int, args: HeatPumpTargetInputs
) -> np.ndarray:
    """The cascade exchanger approach each stage lifts across, hottest first.

    A stage below a process condensing level evaporates ``dt_cascade_hx``
    below it; a stage above a process evaporating level condenses that much
    above it. The stage between the condensing and evaporating levels has no
    cascade exchanger. See ``_cascade_stage_preflight_state``.
    """
    stage = np.arange(n_cond + n_evap - 1)
    crosses = (stage + 1 < n_cond) | (stage >= n_cond)
    return np.where(crosses, float(getattr(args, "dt_cascade_hx", 0.0)), 0.0)


def _condenser_stage_evaporating_temperatures(
    T_cond: np.ndarray,
    T_evap: np.ndarray,
    args: HeatPumpTargetInputs,
) -> np.ndarray:
    """Evaporating temperature of each process condenser's own stage."""
    dt_cascade_hx = float(getattr(args, "dt_cascade_hx", 0.0))
    return np.concatenate([T_cond[1:] - dt_cascade_hx, T_evap[:1]])


def _cascade_process_control_counts(
    *,
    n_cond: int,
    n_evap: int,
    is_heat_pumping: bool,
) -> tuple[int, int]:
    if is_heat_pumping:
        return n_cond, max(n_evap - 1, 0)
    return max(n_cond - 1, 0), n_evap


def _cascade_stage_preflight_state(
    state: HPRParsedState,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    """Expand process temperature controls to physical cascade stage states."""
    T_cond = np.asarray(state.T_cond, dtype=float)
    T_evap = np.asarray(state.T_evap, dtype=float)
    return state.model_copy(
        update={
            "T_cond": np.sort(
                np.concatenate([T_cond, T_evap[:-1] + args.dt_cascade_hx])
            ),
            "T_evap": np.sort(
                np.concatenate([T_cond[1:] - args.dt_cascade_hx, T_evap])
            ),
        }
    )


def _compute_cascade_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    return _evaluate_multi_vc_objective(
        x,
        args,
        parse_state=_parse_cascade_hp_state_variables,
        build_and_solve=_build_and_solve_cascade_cycle,
        topology_id=_cascade_topology_id,
        label="Cascade",
        evaluate_result=evaluate_vapour_hpr_result,
        build_record=build_coolprop_target_simulation_record,
        debug=debug,
        artifact_mode=artifact_mode,
    )


def _build_and_solve_cascade_cycle(
    state_vars: HPRParsedState,
    args: HeatPumpTargetInputs,
    is_heat_pumping: bool,
) -> CascadeVapourCompressionCycle:
    hp = CascadeVapourCompressionCycle()
    hp.solve(
        T_evap=state_vars.T_evap,
        T_cond=state_vars.T_cond,
        dtcont=args.dtcont_hp,
        duty_allocation=DutyAllocationRequest.from_state(state_vars),
        dT_subcool=state_vars.dT_subcool,
        dT_ihx_gas_side=state_vars.dT_ihx_gas_side,
        dt_ihx=args.dt_hp_ihx,
        limits=cycle_limits(args),
        eta_comp=args.eta_comp,
        eta_motor=args.eta_motor,
        refrigerant=args.refrigerant_ls,
        dt_cascade_hx=args.dt_cascade_hx,
        is_heat_pump=is_heat_pumping,
    )
    if hp.solved:
        state_vars.dT_subcool = hp.dT_subcool[: state_vars.dT_subcool.size]
    return hp


def _compute_tespy_cascade_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    coordinator: HprTargetEvaluatorCoordinator,
    prepared: PreparedTespyHprTargeting,
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    """Evaluate one optimizer candidate through the normalized TESPy boundary."""
    state = _parse_cascade_hp_state_variables(x, args)
    allocation = _tespy_primary_duty_allocation(state, args)
    useful_duty = float(allocation.Q_model[0])
    if useful_duty <= tol:
        return _tespy_candidate_failure(
            state,
            "TESPy target candidate requires a positive allocated useful duty.",
        )

    try:
        request = HprTargetThermodynamicRequest(
            mode=prepared.mode,
            cycle_id=prepared.cycle_id,
            model_id=prepared.model_id,
            working_fluid=prepared.working_fluid,
            evaporating_temperature=float(state.T_evap[0]),
            condensing_temperature=float(state.T_cond[0]),
            useful_duty=useful_duty,
            source_approach_temperature=float(args.dtcont_hp),
            sink_approach_temperature=float(args.dtcont_hp),
            compressor_isentropic_efficiency=float(args.eta_comp),
            superheat=_TESPY_TARGET_SUPERHEAT,
            subcooling=float(state.dT_subcool[0]),
            internal_hx_gas_temperature_change=_tespy_ihx_gas_rise(state, args),
            internal_hx_approach_temperature=float(getattr(args, "dt_hp_ihx", 0.0)),
            candidate_id=_tespy_candidate_id(x),
        )
    except (TypeError, ValueError) as exc:
        return _tespy_candidate_failure(state, str(exc))

    evaluated = coordinator.evaluate(request)
    if isinstance(evaluated, HprTargetEvaluationFailure):
        return _tespy_candidate_failure(
            state,
            f"{evaluated.code}: {evaluated.message}",
        )
    if not isinstance(evaluated, HprTargetThermodynamicResult):
        raise TypeError("HPR target evaluator coordinator returned an invalid value")

    hpr_streams = _build_tespy_hpr_streams(evaluated, args)
    # TESPy returns shaft power; the motor losses leave the cycle.
    work = evaluated.compressor_power / args.eta_motor
    result = evaluate_vapour_hpr_result(
        args=args,
        state=state,
        work=work,
        work_arr=np.array([work], dtype=float),
        Q_heat=np.array([evaluated.q_sink], dtype=float),
        Q_cool=np.array([evaluated.q_source], dtype=float),
        cop_h=evaluated.cop * args.eta_motor,
        hpr_streams=hpr_streams,
        model=None,
        penalty_terms=[],
        dT_subcool=state.dT_subcool,
        dT_superheat=np.array([request.superheat], dtype=float),
        debug=debug,
        artifact_mode=artifact_mode,
    )
    metadata = coordinator.metadata
    if metadata is None:
        raise RuntimeError("TESPy evaluator metadata is unavailable after open")
    record = (
        build_tespy_target_simulation_record(
            request=request,
            result=evaluated,
            metadata=metadata,
        )
        if artifact_mode is HPREvaluationMode.FINAL
        else None
    )
    return result.with_updates(
        simulation_backend="tespy"
        if artifact_mode is HPREvaluationMode.FINAL
        else None,
        target_simulation_record=record,
        effective_topology=(
            _tespy_effective_topology(request, result, args)
            if artifact_mode is HPREvaluationMode.FINAL
            else None
        ),
    )


def _tespy_effective_topology(
    request: HprTargetThermodynamicRequest,
    result: HPRBackendResult,
    args: HeatPumpTargetInputs,
) -> HPREffectiveTopology:
    """The single stage of a TESPy target, as a CoolProp target reports it."""
    state = build_coolprop_abstract_state(request.working_fluid.source_spec)
    state.update(CoolProp.QT_INPUTS, 1.0, request.evaporating_temperature + 273.15)
    evaporating_pressure = state.p()
    state.update(CoolProp.QT_INPUTS, 0.0, request.condensing_temperature + 273.15)
    stage = HPRActiveStage(
        index=0,
        role="vapour_compression",
        fluid=request.working_fluid.source_spec,
        evaporating_temperature=request.evaporating_temperature,
        condensing_temperature=request.condensing_temperature,
        pressure_ratio=state.p() / evaporating_pressure,
        work=float(result.w_net),
        process_heat=float(result.Q_heat[0]),
        process_cooling=float(result.Q_cool[0]),
    )
    return topology_from_stages([stage], args)


def _tespy_ihx_gas_rise(state: HPRParsedState, args: HeatPumpTargetInputs) -> float:
    """The IHX gas-side rise, held ``dt_hp_ihx`` below the liquid entering it.

    The same limit as the CoolProp cycle: the gas leaves at most the liquid's
    temperature (condensing less subcooling) less the approach.
    """
    ceiling = (
        float(state.T_cond[0])
        - float(state.dT_subcool[0])
        - float(state.T_evap[0])
        - _TESPY_TARGET_SUPERHEAT
        - float(getattr(args, "dt_hp_ihx", 0.0))
    )
    return max(min(float(state.dT_ihx_gas_side[0]), ceiling), 0.0)


def _tespy_primary_duty_allocation(
    state: HPRParsedState,
    args: HeatPumpTargetInputs,
) -> DutyAllocation:
    if args.is_heat_pumping:
        return require_stage_duty_allocation(
            x_split=state.x_heat_split,
            Q_available=state.Q_heat_available,
            duty_name="heat",
        )
    return require_stage_duty_allocation(
        x_split=state.x_cool_split,
        Q_available=state.Q_cool_available,
        duty_name="cool",
    )


def _build_tespy_hpr_streams(
    result: HprTargetThermodynamicResult,
    args: HeatPumpTargetInputs,
) -> StreamCollection:
    """Convert detached source/sink profiles into ordinary HPR utility streams."""
    streams = StreamCollection()
    _add_profile_streams(
        streams,
        result.source_profile,
        prefix="TESPy Evaporator",
        heat_flow=result.q_source,
        is_hot=False,
        args=args,
    )
    _add_profile_streams(
        streams,
        result.sink_profile,
        prefix="TESPy Condenser",
        heat_flow=result.q_sink,
        is_hot=True,
        args=args,
    )
    return streams


def _add_profile_streams(
    streams: StreamCollection,
    profile: tuple[HprThermalProfilePoint, ...],
    *,
    prefix: str,
    heat_flow: float,
    is_hot: bool,
    args: HeatPumpTargetInputs,
) -> None:
    enthalpy_span = float(profile[-1].enthalpy - profile[0].enthalpy)
    for index, (start, end) in enumerate(zip(profile, profile[1:], strict=False), 1):
        segment_duty = heat_flow * (end.enthalpy - start.enthalpy) / enthalpy_span
        if segment_duty <= tol:
            continue
        low_temperature = min(start.temperature, end.temperature)
        high_temperature = max(start.temperature, end.temperature)
        if high_temperature - low_temperature <= tol:
            high_temperature = low_temperature + float(args.dt_phase_change)
        streams.add(
            Stream(
                name=f"{prefix} {index}",
                supply_temperature=(high_temperature if is_hot else low_temperature),
                target_temperature=(low_temperature if is_hot else high_temperature),
                heat_flow=segment_duty,
                # HPR_DT_CONT is applied once, to the background profiles;
                # cycle streams carry none of their own, as Carnot streams.
                delta_t_contribution=0.0,
                is_process_stream=False,
            )
        )


def _tespy_candidate_id(x: np.ndarray) -> str:
    return "candidate:" + ",".join(float(value).hex() for value in np.asarray(x))


def _tespy_candidate_failure(
    state: HPRParsedState,
    reason: str,
) -> HPRBackendResult:
    return HPRBackendResult.failure(
        reason=reason,
        Q_amb_hot=state.Q_amb_hot,
        Q_amb_cold=state.Q_amb_cold,
    ).with_updates(simulation_backend="tespy")
