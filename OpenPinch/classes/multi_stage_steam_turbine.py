"""Multi-stage steam turbine targeting utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..lib.config import tol
from ..lib.enums import TurbineModel
from ..lib.schema import TurbineSolveResult, TurbineStageResult
from ..utils.water_properties import Tsat_p, h_ps, h_pT, hL_p, hV_p, psat_T, s_ph

__all__ = [
    "MultiStageSteamTurbine"
]


def _normalise_model_name(model: str | TurbineModel) -> str:
    return model.value if isinstance(model, TurbineModel) else str(model)


def _apply_efficiency_limits(
    work: float,
    work_isentropic: float,
    min_eff: float,
) -> tuple[float, float]:
    if work_isentropic <= tol:
        return 0.0, min_eff

    efficiency = work / work_isentropic
    if efficiency <= min_eff:
        return min_eff * work_isentropic, min_eff

    return work, efficiency


def _segment_enthalpy(
    h_prev: float,
    work: float,
    mass_flow: float,
    mech_eff: float,
) -> float:
    if mass_flow <= tol or mech_eff <= tol:
        return h_prev
    return h_prev - work / (mass_flow * mech_eff)


def _predict_stage_work(
    *,
    model: str,
    pressure_in: float,
    enthalpy_in: float,
    pressure_out: float,
    saturation_enthalpy: float,
    mass_flow: float,
    mass_flow_max: float,
    dh_isentropic: float,
    mech_eff: float,
    min_eff: float,
) -> float:
    if mass_flow <= tol or dh_isentropic <= tol:
        return 0.0

    if model == TurbineModel.SUN_SMITH.value:
        if mass_flow_max <= tol:
            return 0.0
        return _work_SunModel(
            pressure_in,
            enthalpy_in,
            pressure_out,
            saturation_enthalpy,
            mass_flow,
            mass_flow_max,
            dh_isentropic,
            mech_eff,
        )
    if model == TurbineModel.MEDINA_FLORES.value:
        return _work_MedinaModel(pressure_in, mass_flow, dh_isentropic)
    if model == TurbineModel.VARBANOV.value:
        return _work_THM(
            pressure_in,
            enthalpy_in,
            pressure_out,
            saturation_enthalpy,
            mass_flow,
            dh_isentropic,
            mech_eff,
        )
    if model == TurbineModel.ISENTROPIC.value:
        return mass_flow * dh_isentropic * min_eff
    return 0.0


class _TurbineState:
    """Mutable turbine calculation state for above-pinch extraction solving."""

    def __init__(self, params: dict, data: dict):
        self.P_out = data["P_out"]
        self.Q_users = data["Q_users"]
        self.w_k = data["w_k"]
        self.w_isen_k = data["w_isen_k"]
        self.m_k = data["m_k"]
        self.eff_k = data["eff_k"]
        self.dh_is_k = data["dh_is_k"]
        self.h_out = data["h_out"]
        self.h_tar = data["h_tar"]
        self.h_sat = data["h_sat"]
        self.stage_temperatures = data["stage_temperatures"]
        self.source_indices = data["source_indices"]
        self.s = data["s"]

        self.model = params["model"]
        self.load_frac = params["load_frac"]
        self.n_mech = params["mech_eff"]
        self.min_eff = params["min_eff"]
        self.flash_correction = params["flash_correction"]

        self.m_in_est = data["m_in_est"]
        self.mass_flow_in = [0.0] * self.s
        self.mass_flow_out = [0.0] * self.s

    def max_mass_flow(self, mass_flow: float) -> float:
        return mass_flow / self.load_frac if self.load_frac > tol else 0.0


def _segment_work(
    state: _TurbineState,
    *,
    index: int,
    mass_flow: float,
    mass_flow_max: float,
) -> float:
    return _predict_stage_work(
        model=state.model,
        pressure_in=state.P_out[index - 1],
        enthalpy_in=state.h_out[index - 1],
        pressure_out=state.P_out[index],
        saturation_enthalpy=state.h_sat[index],
        mass_flow=mass_flow,
        mass_flow_max=mass_flow_max,
        dh_isentropic=state.dh_is_k[index],
        mech_eff=state.n_mech,
        min_eff=state.min_eff,
    )


def _segment_mass_flow(state: _TurbineState, *, index: int, mass_flow: float) -> float:
    if mass_flow <= tol:
        return 0.0

    if state.flash_correction:
        q_flash = state.m_k[index - 1] * (state.h_tar[index - 1] - state.h_tar[index])
        q_stage = max(state.Q_users[index] - q_flash, 0.0)
    else:
        q_stage = state.Q_users[index]

    dh_cond = state.h_out[index] - state.h_tar[index]
    if dh_cond <= tol:
        return 0.0

    return q_stage / dh_cond


def _iterate_turbine_state(state: _TurbineState) -> None:
    m_in_remaining = state.m_in_est
    state.m_in_est = 0.0

    for j in range(1, state.s):
        m_in_remaining -= state.m_k[j - 1]
        m_in_remaining = max(m_in_remaining, 0.0)
        state.mass_flow_in[j] = m_in_remaining
        state.dh_is_k[j] = state.h_out[j - 1] - h_ps(
            state.P_out[j],
            s_ph(state.P_out[j - 1], state.h_out[j - 1]),
        )
        state.w_isen_k[j] = m_in_remaining * state.dh_is_k[j]
        m_max = state.max_mass_flow(m_in_remaining)
        work_guess = _segment_work(
            state,
            index=j,
            mass_flow=m_in_remaining,
            mass_flow_max=m_max,
        )
        work, efficiency = _apply_efficiency_limits(
            work_guess,
            state.w_isen_k[j],
            state.min_eff,
        )
        state.w_k[j] = work
        state.eff_k[j] = efficiency
        state.h_out[j] = _segment_enthalpy(
            state.h_out[j - 1],
            state.w_k[j],
            m_in_remaining,
            state.n_mech,
        )
        state.m_k[j] = _segment_mass_flow(
            state,
            index=j,
            mass_flow=m_in_remaining,
        )
        state.mass_flow_out[j] = max(m_in_remaining - state.m_k[j], 0.0)
        state.m_in_est += state.m_k[j]


def _work_MedinaModel(P_in, m, dh_is):
    """Determine power generation using Medina-Flores & Picon-Nunez (2010)."""
    A0 = 185.4 + 43.3 * (P_in * 0.1)
    b0 = 1.2057 + 0.0075 * (P_in * 0.1)
    return (m * dh_is - A0) / b0


def _work_SunModel(P_in, h_in, P_out, h_sat, m, m_max, dh_is, n_mech, t_type=1):
    """Determine power generation using Sun & Smith (2015)."""
    coeff = {
        "BPST": {
            "a": [1.18795366, -0.00029564, 0.004647288],
            "b": [449.9767142, 5.670176939, -11.5045814],
            "c": [0.205149333, -0.000695171, 0.002844611],
        },
        "CT": {
            "a": [1.314991261, -0.001634725, -0.367975103],
            "b": [-437.7746025, 29.00736723, 10.35902331],
            "c": [0.07886297, 0.000528327, -0.703153891],
        },
    }

    if t_type in (1, "BPST"):
        t_type_key = "BPST"
    elif t_type in (2, "CT"):
        t_type_key = "CT"
    else:
        raise ValueError("Unsupported Sun model turbine type.")

    A0 = (
        coeff[t_type_key]["a"][0]
        + coeff[t_type_key]["a"][1] * P_in
        + coeff[t_type_key]["a"][2] * P_out
    )
    b0 = (
        coeff[t_type_key]["b"][0]
        + coeff[t_type_key]["b"][1] * P_in
        + coeff[t_type_key]["b"][2] * P_out
    )
    c0 = (
        coeff[t_type_key]["c"][0]
        + coeff[t_type_key]["c"][1] * P_in
        + coeff[t_type_key]["c"][2] * P_out
    )

    W_int = c0 / A0 * (m_max * dh_is - b0)
    n = (1 + c0) / A0 * (dh_is - b0 / m_max)
    w_act = n * m - W_int
    h_out = h_in - w_act / (n_mech * m)

    if h_out <= h_sat + tol and t_type_key == "BPST":
        w_act = _work_SunModel(P_in, h_in, P_out, h_sat, m, m_max, dh_is, n_mech, "CT")
    return w_act


def _work_THM(P_in, h_in, P_out, h_sat, m, dh_is, n_mech, t_size=1, t_type=1):
    """Determine power generation using Varbanov et al. (2004)."""
    coeff = {
        "BPST": {
            "<2MW": [0, 0.00108, 1.097, 0.00172],
            ">2MW": [0, 0.00423, 1.155, 0.000538],
        },
        "CT": {
            "<2MW": [0, 0.000662, 1.191, 0.000759],
            ">2MW": [-0.463, 0.00353, 1.22, 0.000148],
        },
    }

    if t_type in (1, "BPST"):
        t_type_key = "BPST"
    elif t_type in (2, "CT"):
        t_type_key = "CT"
    else:
        raise ValueError("Unsupported THM turbine type.")

    if t_size in (1, "<2MW"):
        t_size_key = "<2MW"
    elif t_size in (2, ">2MW"):
        t_size_key = ">2MW"
    else:
        raise ValueError("Unsupported THM turbine size.")

    dT_sat = Tsat_p(P_in) - Tsat_p(P_out)
    a = (
        coeff[t_type_key][t_size_key][0] + coeff[t_type_key][t_size_key][1] * dT_sat
    ) * 1000
    b = coeff[t_type_key][t_size_key][2] + coeff[t_type_key][t_size_key][3] * dT_sat
    w_max = (dh_is * m - a) / b

    if w_max > 2000 and t_size_key == "<2MW":
        w_max = _work_THM(P_in, h_in, P_out, h_sat, m, dh_is, n_mech, ">2MW", t_type_key)

    h_out = h_in - w_max / (n_mech * m)
    if h_out <= h_sat + tol and t_type_key == "BPST":
        w_max = _work_THM(P_in, h_in, P_out, h_sat, m, dh_is, n_mech, t_size_key, "CT")

    return w_max


class MultiStageSteamTurbine:
    """Stateful multi-stage steam turbine solver for pinch targeting."""

    def __init__(self):
        self._solved = False
        self._result: Optional[TurbineSolveResult] = None

    @property
    def solved(self) -> bool:
        return self._solved

    @property
    def result(self) -> TurbineSolveResult:
        self._require_solution()
        return self._result

    @property
    def stages(self) -> list[TurbineStageResult]:
        self._require_solution()
        return self._result.stages

    @property
    def total_work(self) -> float:
        self._require_solution()
        return self._result.total_work

    def solve(
        self,
        temperatures: np.ndarray,
        heat_flows: np.ndarray,
        *,
        mode: str,
        T_in: float | None = None,
        P_in: float | None = None,
        T_sink: float | None = None,
        model: str | TurbineModel = TurbineModel.MEDINA_FLORES.value,
        min_eff: float = 0.1,
        load_frac: float = 1.0,
        mech_eff: float = 1.0,
        flash_correction: bool = False,
    ) -> tuple[float, dict]:
        """Solve a turbine targeting problem and return total work plus details."""
        self._solved = False
        self._result = None

        T_arr, Q_arr, source_idx = self._normalise_stage_inputs(
            temperatures, heat_flows
        )
        model_name = _normalise_model_name(model)
        params = {
            "model": model_name,
            "min_eff": min(max(float(min_eff), 0.0), 1.0),
            "load_frac": min(max(float(load_frac), 0.0), 1.0),
            "mech_eff": min(max(float(mech_eff), 0.0), 1.0),
            "flash_correction": bool(flash_correction),
        }

        if mode == "above_pinch":
            if T_in is None or P_in is None:
                raise ValueError("Above-pinch turbine solving requires T_in and P_in.")
            result = self._solve_above_pinch(
                stage_temperatures=T_arr,
                stage_heat_flows=Q_arr,
                source_indices=source_idx,
                T_in=float(T_in),
                P_in=float(P_in),
                params=params,
            )
        elif mode == "below_pinch":
            if T_sink is None:
                raise ValueError("Below-pinch turbine solving requires T_sink.")
            result = self._solve_below_pinch(
                stage_temperatures=T_arr,
                stage_heat_flows=Q_arr,
                source_indices=source_idx,
                T_sink=float(T_sink),
                params=params,
            )
        else:
            raise ValueError("mode must be 'above_pinch' or 'below_pinch'.")

        self._result = TurbineSolveResult.model_validate(result)
        self._solved = True
        return self._result.total_work, self._result.model_dump()

    def _normalise_stage_inputs(
        self,
        temperatures,
        heat_flows,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T_arr = np.asarray(temperatures, dtype=float)
        Q_arr = np.asarray(heat_flows, dtype=float)

        if T_arr.ndim == 0:
            T_arr = T_arr.reshape(1)
        if Q_arr.ndim == 0:
            Q_arr = Q_arr.reshape(1)
        if T_arr.ndim != 1 or Q_arr.ndim != 1:
            raise ValueError("Turbine stage temperatures and heat flows must be 1D.")
        if T_arr.size != Q_arr.size:
            raise ValueError("Turbine stage temperatures and heat flows must align.")
        if not np.isfinite(T_arr).all() or not np.isfinite(Q_arr).all():
            raise ValueError(
                "Turbine stage temperatures and heat flows must be finite."
            )

        valid = Q_arr > tol
        if not valid.any():
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=int),
            )

        T_arr = T_arr[valid]
        Q_arr = Q_arr[valid]
        source_idx = np.nonzero(valid)[0]
        order = np.argsort(T_arr)[::-1]
        return T_arr[order], Q_arr[order], source_idx[order]

    def _empty_result(
        self,
        *,
        mode: str,
        params: dict,
        stage_temperatures: np.ndarray,
        stage_heat_flows: np.ndarray,
        inlet_pressure: float | None = None,
        inlet_temperature: float | None = None,
        sink_pressure: float | None = None,
        sink_temperature: float | None = None,
    ) -> dict:
        return {
            "mode": mode,
            "turbine_model": params["model"],
            "load_frac": params["load_frac"],
            "mech_eff": params["mech_eff"],
            "min_eff": params["min_eff"],
            "flash_correction": params["flash_correction"],
            "total_work": 0.0,
            "total_isentropic_work": 0.0,
            "overall_efficiency": 0.0,
            "total_process_duty": float(stage_heat_flows.sum())
            if stage_heat_flows.size
            else 0.0,
            "steam_mass_flow_in": 0.0,
            "inlet_pressure": inlet_pressure,
            "inlet_temperature": inlet_temperature,
            "sink_pressure": sink_pressure,
            "sink_temperature": sink_temperature,
            "stage_temperatures": stage_temperatures.tolist(),
            "stage_heat_flows": stage_heat_flows.tolist(),
            "stages": [],
        }

    def _solve_above_pinch(
        self,
        *,
        stage_temperatures: np.ndarray,
        stage_heat_flows: np.ndarray,
        source_indices: np.ndarray,
        T_in: float,
        P_in: float,
        params: dict,
    ) -> dict:
        stage_pressures = np.asarray(
            [psat_T(T_stage) for T_stage in stage_temperatures], dtype=float
        )
        viable = stage_pressures <= P_in + tol
        if not viable.any():
            return self._empty_result(
                mode="above_pinch",
                params=params,
                stage_temperatures=np.array([], dtype=float),
                stage_heat_flows=np.array([], dtype=float),
                inlet_pressure=P_in,
                inlet_temperature=T_in,
            )

        T_stage = stage_temperatures[viable]
        Q_stage = stage_heat_flows[viable]
        P_stage = stage_pressures[viable]
        source_idx = source_indices[viable]

        h_inlet = h_pT(P_in, T_in)
        P_out = [P_in]
        Q_users = [0.0]
        w_k = [0.0]
        w_isen_k = [0.0]
        m_k = [0.0]
        eff_k = [0.0]
        dh_is_k = [0.0]
        h_out = [h_inlet]
        h_tar = [hL_p(P_in)]
        h_sat = [hV_p(P_in)]
        stage_T = [T_in]
        stage_idx = [-1]
        m_in_est = 0.0

        s_inlet = s_ph(P_in, h_inlet)
        for idx, T_target, Q_target, P_target in zip(
            source_idx, T_stage, Q_stage, P_stage
        ):
            P_out.append(float(P_target))
            Q_users.append(float(Q_target))
            w_k.append(0.0)
            w_isen_k.append(0.0)
            eff_k.append(0.0)
            stage_T.append(float(T_target))
            stage_idx.append(int(idx))

            h_sat_stage = hV_p(P_target)
            h_tar_stage = hL_p(P_target)
            h_sat.append(h_sat_stage)
            h_tar.append(h_tar_stage)
            h_out.append(h_sat_stage)

            dh_is = h_inlet - h_ps(P_target, s_inlet)
            dh_is_k.append(dh_is)
            dh_cond = h_inlet - dh_is - h_tar_stage
            m_stage = Q_target / dh_cond if dh_cond > tol else 0.0
            m_k.append(m_stage)
            m_in_est += m_stage

        data = {
            "P_out": P_out,
            "Q_users": Q_users,
            "w_k": w_k,
            "w_isen_k": w_isen_k,
            "m_k": m_k,
            "eff_k": eff_k,
            "dh_is_k": dh_is_k,
            "h_out": h_out,
            "h_tar": h_tar,
            "h_sat": h_sat,
            "stage_temperatures": stage_T,
            "source_indices": stage_idx,
            "m_in_est": m_in_est,
            "s": len(P_out),
        }
        state = _TurbineState(params, data)

        iterations = 0
        while True:
            previous_m_in = state.m_in_est
            _iterate_turbine_state(state)
            iterations += 1
            if abs(previous_m_in - state.m_in_est) < tol or iterations >= 3:
                break

        stages = []
        for j in range(1, state.s):
            stages.append(
                TurbineStageResult(
                    stage=j,
                    source_index=state.source_indices[j],
                    stage_type="extraction",
                    temperature=state.stage_temperatures[j],
                    process_duty=state.Q_users[j],
                    pressure_in=state.P_out[j - 1],
                    pressure_out=state.P_out[j],
                    mass_flow_in=state.mass_flow_in[j],
                    mass_flow_extracted=state.m_k[j],
                    mass_flow_out=state.mass_flow_out[j],
                    enthalpy_in=state.h_out[j - 1],
                    enthalpy_out=state.h_out[j],
                    condensate_enthalpy=state.h_tar[j],
                    saturation_enthalpy=state.h_sat[j],
                    dh_isentropic=state.dh_is_k[j],
                    work_actual=state.w_k[j],
                    work_isentropic=state.w_isen_k[j],
                    isentropic_efficiency=state.eff_k[j],
                    turbine_model=state.model,
                )
            )

        total_work = float(sum(state.w_k))
        total_isentropic_work = float(sum(state.w_isen_k))
        overall_efficiency = (
            total_work / total_isentropic_work if total_isentropic_work > tol else 0.0
        )

        return {
            "mode": "above_pinch",
            "turbine_model": state.model,
            "load_frac": state.load_frac,
            "mech_eff": state.n_mech,
            "min_eff": state.min_eff,
            "flash_correction": state.flash_correction,
            "total_work": total_work,
            "total_isentropic_work": total_isentropic_work,
            "overall_efficiency": overall_efficiency,
            "total_process_duty": float(np.sum(Q_stage)),
            "steam_mass_flow_in": state.mass_flow_in[1] if stages else 0.0,
            "inlet_pressure": P_in,
            "inlet_temperature": T_in,
            "sink_pressure": None,
            "sink_temperature": None,
            "stage_temperatures": T_stage.tolist(),
            "stage_heat_flows": Q_stage.tolist(),
            "stages": [stage.model_dump() for stage in stages],
        }

    def _solve_below_pinch(
        self,
        *,
        stage_temperatures: np.ndarray,
        stage_heat_flows: np.ndarray,
        source_indices: np.ndarray,
        T_sink: float,
        params: dict,
    ) -> dict:
        P_sink = psat_T(T_sink)
        viable = stage_temperatures > T_sink + tol
        if not viable.any():
            return self._empty_result(
                mode="below_pinch",
                params=params,
                stage_temperatures=np.array([], dtype=float),
                stage_heat_flows=np.array([], dtype=float),
                sink_pressure=P_sink,
                sink_temperature=T_sink,
            )

        T_stage = stage_temperatures[viable]
        Q_stage = stage_heat_flows[viable]
        source_idx = source_indices[viable]
        h_sink_liq = hL_p(P_sink)
        h_sink_sat = hV_p(P_sink)

        stages = []
        total_work = 0.0
        total_isentropic_work = 0.0
        total_mass_flow = 0.0

        for stage_no, (idx, T_source, Q_source) in enumerate(
            zip(source_idx, T_stage, Q_stage),
            start=1,
        ):
            P_stage = psat_T(T_source)
            h_in = hV_p(P_stage)
            dh_boiler = h_in - h_sink_liq
            if dh_boiler <= tol:
                continue

            m_stage = Q_source / dh_boiler
            dh_is = h_in - h_ps(P_sink, s_ph(P_stage, h_in))
            w_is = m_stage * dh_is
            m_max = m_stage / params["load_frac"] if params["load_frac"] > tol else 0.0
            w_guess = _predict_stage_work(
                model=params["model"],
                pressure_in=P_stage,
                enthalpy_in=h_in,
                pressure_out=P_sink,
                saturation_enthalpy=h_sink_sat,
                mass_flow=m_stage,
                mass_flow_max=m_max,
                dh_isentropic=dh_is,
                mech_eff=params["mech_eff"],
                min_eff=params["min_eff"],
            )
            w_act, eff = _apply_efficiency_limits(w_guess, w_is, params["min_eff"])
            h_out = _segment_enthalpy(h_in, w_act, m_stage, params["mech_eff"])

            stage = TurbineStageResult(
                stage=stage_no,
                source_index=int(idx),
                stage_type="condensing",
                temperature=float(T_source),
                process_duty=float(Q_source),
                pressure_in=float(P_stage),
                pressure_out=float(P_sink),
                mass_flow_in=float(m_stage),
                mass_flow_extracted=float(m_stage),
                mass_flow_out=0.0,
                enthalpy_in=float(h_in),
                enthalpy_out=float(h_out),
                condensate_enthalpy=float(h_sink_liq),
                saturation_enthalpy=float(h_sink_sat),
                dh_isentropic=float(dh_is),
                work_actual=float(w_act),
                work_isentropic=float(w_is),
                isentropic_efficiency=float(eff),
                turbine_model=params["model"],
            )
            stages.append(stage)
            total_work += w_act
            total_isentropic_work += w_is
            total_mass_flow += m_stage

        overall_efficiency = (
            total_work / total_isentropic_work if total_isentropic_work > tol else 0.0
        )
        return {
            "mode": "below_pinch",
            "turbine_model": params["model"],
            "load_frac": params["load_frac"],
            "mech_eff": params["mech_eff"],
            "min_eff": params["min_eff"],
            "flash_correction": params["flash_correction"],
            "total_work": float(total_work),
            "total_isentropic_work": float(total_isentropic_work),
            "overall_efficiency": float(overall_efficiency),
            "total_process_duty": float(np.sum(Q_stage)),
            "steam_mass_flow_in": float(total_mass_flow),
            "inlet_pressure": None,
            "inlet_temperature": None,
            "sink_pressure": float(P_sink),
            "sink_temperature": float(T_sink),
            "stage_temperatures": T_stage.tolist(),
            "stage_heat_flows": Q_stage.tolist(),
            "stages": [stage.model_dump() for stage in stages],
        }

    def _require_solution(self) -> None:
        if not self._solved or self._result is None:
            raise RuntimeError("Solve the turbine before accessing results.")
