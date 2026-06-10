"""Mechanical vapour recompression cycle utilities built on CoolProp."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ....classes.stream import Stream
from ....classes.stream_collection import StreamCollection
from ....utils.stream_linearisation import get_piecewise_data_points
from .vapour_compression_cycle import VapourCompressionCycle

__all__ = ["MechanicalVapourRecompressionCycle"]


class MechanicalVapourRecompressionCycle(VapourCompressionCycle):
    """Single-stage mechanical vapour recompression model.

    The open stage is represented as source vapour at the evaporating pressure,
    dry real compression to the condensing pressure, post-compression internal
    liquid-injection desuperheating, process-side condensation, and optional
    liquid subcooling.
    """

    STATECOUNT = 4
    solve = None

    @staticmethod
    def _new_cycle_states() -> list[dict[str, float]]:
        """Return empty storage for the four thermodynamic state points."""
        return [{} for _ in range(MechanicalVapourRecompressionCycle.STATECOUNT)]

    def __init__(self):
        """Initialise an unsolved MVR cycle with water as the default fluid."""
        super().__init__()
        self._cycle_states = self._new_cycle_states()
        self._eta_mvr_comp: float = 0.7
        self._eta_motor: float = 1.0
        self._m_dot_source: Optional[float] = None
        self._liquid_injection_ratio: float = 0.0
        self._shaft_work: Optional[float] = None
        self._q_shaft: Optional[float] = None
        self._q_source: Optional[float] = None
        self._q_desuperheat: Optional[float] = None
        self._q_liquid_injection: Optional[float] = None
        self._q_condense: Optional[float] = None
        self._q_latent_condense: Optional[float] = None
        self._q_subcool_process: Optional[float] = None
        self._process_split: float = 1.0
        self._process_heat_components: dict[str, float] = {
            "desuperheat": 0.0,
            "latent": 0.0,
            "subcool": 0.0,
            "total": 0.0,
        }
        self._process_m_dot_out: Optional[float] = None
        self._source_heat_is_external = False
        self._max_work: float = 0.0
        self._refrigerant = "Water"

    @property
    def Hs(self) -> Sequence[float]:
        """Specific enthalpies for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["H"] for i in range(self.STATECOUNT)]

    @property
    def Ss(self) -> Sequence[float]:
        """Specific entropies for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["S"] for i in range(self.STATECOUNT)]

    @property
    def Ts(self) -> Sequence[float]:
        """Temperatures for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["T"] for i in range(self.STATECOUNT)]

    @property
    def Ps(self) -> Sequence[float]:
        """Pressures for the solved state points."""
        self._require_solution()
        return [self._cycle_states[i]["P"] for i in range(self.STATECOUNT)]

    @property
    def eta_mvr_comp(self) -> float:
        """MVR compressor isentropic efficiency."""
        return self._eta_mvr_comp

    @property
    def eta_motor(self) -> float:
        """Motor efficiency converting shaft work to electrical work."""
        return self._eta_motor

    @property
    def shaft_work(self) -> Optional[float]:
        """Compressor shaft work before motor losses."""
        return self._shaft_work

    @property
    def source_m_dot(self) -> Optional[float]:
        """Vapour mass flow generated or received before liquid injection."""
        return self._m_dot_source

    @property
    def liquid_injection_ratio(self) -> float:
        """Injected liquid mass per unit source vapour mass."""
        return self._liquid_injection_ratio

    @property
    def q_source(self) -> Optional[float]:
        """Specific source heat needed to generate inlet vapour."""
        return self._q_source

    @property
    def q_desuperheat(self) -> Optional[float]:
        """External specific desuperheating heat after internal injection."""
        return self._q_desuperheat

    @property
    def q_liquid_injection(self) -> Optional[float]:
        """Dry-compression superheat consumed by injection per source mass."""
        return self._q_liquid_injection

    @property
    def q_condense(self) -> Optional[float]:
        """Condensation and subcooling heat per source mass."""
        return self._q_condense

    @property
    def q_latent_condense(self) -> Optional[float]:
        """Latent condensation heat per source mass."""
        return self._q_latent_condense

    @property
    def q_subcool_process(self) -> Optional[float]:
        """Process subcooling heat per source mass."""
        return self._q_subcool_process

    @property
    def process_split(self) -> float:
        """Fraction of post-injection vapour condensed for process heating."""
        self._require_solution()
        return self._process_split

    @property
    def process_heat(self) -> float:
        """External useful process heat for the stored process split."""
        self._require_solution()
        return self._process_heat_components["total"]

    @property
    def process_m_dot_out(self) -> Optional[float]:
        """Post-injection vapour mass flow sent to the next open MVR stage."""
        self._require_solution()
        return self._process_m_dot_out

    @property
    def work(self) -> Optional[float]:
        """Total electrical work, or finite infeasibility work if unsolved."""
        if self.solved:
            return self._work
        return self._max_work

    @property
    def COP_h(self) -> Optional[float]:
        """Total condenser-duty coefficient of performance based on electric work."""
        self._require_solution()
        if abs(self._work) <= 1e-9:
            raise ZeroDivisionError("COP_h is undefined when electric work is zero.")
        return self._Q_cond / self._work

    @property
    def COP_process_h(self) -> Optional[float]:
        """Useful process-heating coefficient of performance."""
        self._require_solution()
        if abs(self._work) <= 1e-9:
            raise ZeroDivisionError(
                "COP_process_h is undefined when electric work is zero."
            )
        return self.process_heat / self._work

    @property
    def COP_r(self) -> Optional[float]:
        """Evaporator-duty coefficient of performance based on electric work."""
        self._require_solution()
        if abs(self._work) <= 1e-9:
            raise ZeroDivisionError("COP_r is undefined when electric work is zero.")
        return self._Q_evap / self._work

    def solve_from_source_heat(
        self,
        T_evap: float,
        T_cond: float,
        *,
        Q_source: float,
        dT_superheat: float = 0.0,
        dT_subcool: float = 0.0,
        eta_mvr_comp: float = 0.7,
        eta_motor: float = 1.0,
        fluid: str = "Water",
        liquid_injection: bool = True,
        process_split: float = 1.0,
        source_heat_is_external: bool = True,
    ) -> float:
        """Solve an open MVR stage from source heat.

        The generated inlet vapour is saturated at ``T_evap`` plus any supplied
        ``dT_superheat``. When ``source_heat_is_external`` is false, the source
        duty is retained for cycle accounting but omitted from
        ``build_stream_collection()``.
        """
        Q_source = float(Q_source)
        if Q_source < 0.0:
            self._solved = False
            self._max_work = max(abs(Q_source), 1.0) * 1e3
            return self._max_work

        specific = self._get_state_points(
            T_evap=T_evap,
            T_cond=T_cond,
            dT_superheat=dT_superheat,
            dT_subcool=dT_subcool,
            eta_mvr_comp=eta_mvr_comp,
            eta_motor=eta_motor,
            fluid=fluid,
            liquid_injection=liquid_injection,
        )
        if specific is None:
            return self._max_work
        if specific["q_source"] <= 0.0:
            self._solved = False
            self._max_work = max(Q_source, 1.0) * 1e3
            return self._max_work

        return self._scale_open_stage_solution(
            specific,
            m_dot=Q_source / specific["q_source"],
            process_split=process_split,
            source_heat_is_external=source_heat_is_external,
        )

    def solve_from_mass_flow(
        self,
        T_evap: float,
        T_cond: float,
        *,
        m_dot: float,
        dT_superheat: float = 0.0,
        dT_subcool: float = 0.0,
        eta_mvr_comp: float = 0.7,
        eta_motor: float = 1.0,
        fluid: str = "Water",
        liquid_injection: bool = True,
        process_split: float = 1.0,
    ) -> float:
        """Solve the MVR stage from inlet vapour mass flow.

        This is primarily used by serial MVR cascades where a downstream stage
        receives the uncondensed discharge vapour from the previous stage.
        """
        m_dot = float(m_dot)
        if m_dot < 0.0:
            self._solved = False
            self._max_work = max(abs(m_dot), 1.0) * 1e3
            return self._max_work

        specific = self._get_state_points(
            T_evap=T_evap,
            T_cond=T_cond,
            dT_superheat=dT_superheat,
            dT_subcool=dT_subcool,
            eta_mvr_comp=eta_mvr_comp,
            eta_motor=eta_motor,
            fluid=fluid,
            liquid_injection=liquid_injection,
        )
        if specific is None:
            return self._max_work

        return self._scale_open_stage_solution(
            specific,
            m_dot=m_dot,
            process_split=process_split,
            source_heat_is_external=False,
        )

    def process_heat_components(
        self,
        process_split: float | None = None,
    ) -> dict[str, float]:
        """Return external MVR heat components for a process condensation split.

        When ``process_split`` is omitted, the components stored during
        ``solve_from_*`` are returned.
        """
        self._require_solution()
        if process_split is None:
            return dict(self._process_heat_components)

        process_split = float(process_split)
        if not 0.0 <= process_split <= 1.0:
            msg = "process_split must be between 0 and 1."
            raise ValueError(msg)

        source_m_dot = float(self.source_m_dot or 0.0)
        if source_m_dot <= 0.0:
            return {
                "desuperheat": 0.0,
                "latent": 0.0,
                "subcool": 0.0,
                "total": 0.0,
            }

        desuperheat = (
            source_m_dot * process_split * max(float(self.q_desuperheat or 0.0), 0.0)
        )
        latent = (
            source_m_dot
            * process_split
            * max(float(self.q_latent_condense or 0.0), 0.0)
        )
        subcool = (
            source_m_dot
            * process_split
            * max(float(self.q_subcool_process or 0.0), 0.0)
        )
        return {
            "desuperheat": desuperheat,
            "latent": latent,
            "subcool": subcool,
            "total": desuperheat + latent + subcool,
        }

    def _get_state_points(
        self,
        T_evap: float,
        T_cond: float,
        *,
        dT_superheat: float = 0.0,
        dT_subcool: float = 0.0,
        eta_mvr_comp: float = 0.7,
        eta_motor: float = 1.0,
        fluid: str = "Water",
        liquid_injection: bool = True,
    ) -> dict[str, object] | None:
        """Return one-kg/s open-stage state points and specific duties.

        State 0 is inlet vapour at the low-stage pressure, state 1 is the
        actual compressor discharge, state 2 is the condensed/subcooled process
        outlet, and state 3 is saturated vapour at the discharge pressure for
        the vapour that continues to the next stage.
        """
        self._solved = False
        self.temperature_unit = "K"
        self._cycle_states = self._new_cycle_states()
        self._penalty = np.asarray([], dtype=float)
        self._refrigerant = fluid
        self._eta_mvr_comp = float(eta_mvr_comp)
        self._eta_motor = float(eta_motor)
        self._m_dot = 0.0
        self._m_dot_source = 0.0
        self._liquid_injection_ratio = 0.0
        self._process_split = 1.0
        self._process_heat_components = {
            "desuperheat": 0.0,
            "latent": 0.0,
            "subcool": 0.0,
            "total": 0.0,
        }
        self._process_m_dot_out = 0.0
        self._source_heat_is_external = False
        self._T_evap = self._convert_C_to_K(float(T_evap))
        self._T_cond = self._convert_C_to_K(float(T_cond))
        self._dT_superheat = float(dT_superheat)
        self._dT_subcool = float(dT_subcool)
        self._max_work = 1.0

        if self._dT_superheat < 0.0:
            self._max_work *= 1e3
            return None
        if self._dT_subcool < 0.0:
            self._max_work *= 1e3
            return None
        if self._eta_mvr_comp <= 0.0 or self._eta_motor <= 0.0:
            self._max_work *= 1e3
            return None
        if self._T_cond <= self._T_evap:
            self._max_work *= 1.0 + (self._T_evap - self._T_cond)
            return None

        compression = self._compute_open_stage_compression_states(fluid)
        if compression is None:
            return None

        state0 = compression["state0"]
        state1 = compression["state1"]
        state2 = compression["state2"]
        state2_sat = compression["state2_sat"]
        state3 = compression["state3"]
        state_low_liq = compression["state_low_liq"]
        desuperheating = self._compute_liquid_injection_desuperheating(
            state1=state1,
            state3=state3,
            state_injection_liq=state2,
            liquid_injection=liquid_injection,
        )
        if desuperheating is None:
            return None

        gas_mass_factor = desuperheating["gas_mass_factor"]
        q_source = state0.hmass() - state_low_liq.hmass()
        q_desuperheat = desuperheating["q_desuperheat"]
        q_liquid_injection = desuperheating["q_liquid_injection"]
        q_latent_condense = gas_mass_factor * max(
            state3.hmass() - state2_sat.hmass(),
            0.0,
        )
        q_subcool_process = gas_mass_factor * max(
            state2_sat.hmass() - state2.hmass(),
            0.0,
        )
        q_condense = q_latent_condense + q_subcool_process
        q_cond = q_desuperheat + q_condense
        q_shaft = state1.hmass() - state0.hmass()
        if q_source <= 0.0 or q_cond <= 0.0 or q_shaft <= 0.0:
            self._max_work *= 1e3
            return None

        state_points = [
            {
                "H": float(state.hmass()),
                "S": float(state.smass()),
                "P": float(state.p()),
                "T": float(state.T()),
            }
            for state in (state0, state1, state2, state3)
        ]
        return {
            "state_points": state_points,
            "T_evap_sat_vap": float(compression["T_evap_sat_vap"]),
            "T_cond_sat_liq": float(compression["T_cond_sat_liq"]),
            "dT_subcool": float(compression["T_cond_sat_liq"] - state2.T()),
            "q_source": float(q_source),
            "q_desuperheat": float(q_desuperheat),
            "q_liquid_injection": float(q_liquid_injection),
            "q_latent_condense": float(q_latent_condense),
            "q_subcool_process": float(q_subcool_process),
            "q_condense": float(q_condense),
            "q_cond": float(q_cond),
            "q_shaft": float(q_shaft),
            "gas_mass_factor": float(gas_mass_factor),
            "liquid_injection_ratio": float(desuperheating["liquid_injection_ratio"]),
        }

    def _compute_open_stage_compression_states(
        self,
        fluid: str,
    ) -> dict[str, object] | None:
        """Compute source vapour compression before desuperheating."""
        try:
            self._validate_solve_inputs(fluid)
            P_lo = self._get_P_sat_from_T(self._T_evap, Q=1.0)
            P_hi = self._get_P_sat_from_T(self._T_cond, Q=0.0)
            if P_hi <= P_lo:
                self._max_work *= 1.0 + max(P_lo - P_hi, 0.0) / max(P_lo, 1.0)
                return None

            state0_sat = self._compute_state_from_pressure_quality(P_lo, 1.0)
            state_low_liq = self._compute_state_from_pressure_quality(P_lo, 0.0)
            T_evap_sat_vap = state0_sat.T()
            state0 = self._compute_state_from_pressure_temperature(
                P=P_lo,
                T=T_evap_sat_vap + self._dT_superheat,
                phase=1.0,
            )

            old_eta = self._eta_comp
            self._eta_comp = self._eta_mvr_comp
            try:
                state1 = self._compute_compressor_outlet_state(
                    h_in=state0.hmass(),
                    s_in=state0.smass(),
                    P_out=P_hi,
                )
            finally:
                self._eta_comp = old_eta

            state2_sat = self._compute_state_from_pressure_quality(P_hi, 0.0)
            state3 = self._compute_state_from_pressure_quality(P_hi, 1.0)
            T_cond_sat_liq = state2_sat.T()
            T_cond_out = T_cond_sat_liq - self._dT_subcool
            if T_cond_out <= 0.0:
                self._max_work *= 1e3
                return None
            state2 = self._compute_state_from_pressure_temperature(
                P=P_hi,
                T=T_cond_out,
                phase=0.0,
            )
        except Exception:
            self._max_work *= 1e3
            return None

        return {
            "state0": state0,
            "state1": state1,
            "state2": state2,
            "state2_sat": state2_sat,
            "state3": state3,
            "state_low_liq": state_low_liq,
            "T_evap_sat_vap": float(T_evap_sat_vap),
            "T_cond_sat_liq": float(T_cond_sat_liq),
        }

    def _compute_liquid_injection_desuperheating(
        self,
        *,
        state1,
        state3,
        state_injection_liq,
        liquid_injection: bool,
    ) -> dict[str, float] | None:
        """Compute post-compression liquid-injection desuperheating.

        The compressor work is based on the dry source-vapour mass flow. The
        injected liquid is assumed to be drawn from the condenser outlet state
        and evaporated after compression, removing discharge superheat and
        increasing the vapour mass available for process condensation or the
        next serial MVR stage.
        """
        dry_desuperheat = max(state1.hmass() - state3.hmass(), 0.0)
        liquid_injection_ratio = 0.0
        if liquid_injection and dry_desuperheat > 0.0:
            q_injection_evap = state3.hmass() - state_injection_liq.hmass()
            if q_injection_evap <= 0.0:
                self._max_work *= 1e3
                return None
            liquid_injection_ratio = dry_desuperheat / q_injection_evap

        q_desuperheat = 0.0 if liquid_injection_ratio > 0.0 else dry_desuperheat
        return {
            "gas_mass_factor": 1.0 + liquid_injection_ratio,
            "liquid_injection_ratio": liquid_injection_ratio,
            "q_desuperheat": q_desuperheat,
            "q_liquid_injection": dry_desuperheat - q_desuperheat,
        }

    def _scale_open_stage_solution(
        self,
        specific: dict[str, object],
        *,
        m_dot: float,
        process_split: float,
        source_heat_is_external: bool,
    ) -> float:
        """Store specific open-stage data and scale extensive quantities."""
        process_split = float(process_split)
        if process_split < 0.0 or process_split > 1.0:
            self._solved = False
            self._max_work = max(abs(float(m_dot)), 1.0) * 1e3
            return self._max_work

        self._cycle_states = list(specific["state_points"])
        self._T_evap_sat_vap = float(specific["T_evap_sat_vap"])
        self._T_cond_sat_liq = float(specific["T_cond_sat_liq"])
        self._dT_subcool = float(specific["dT_subcool"])
        self._q_source = float(specific["q_source"])
        self._q_desuperheat = float(specific["q_desuperheat"])
        self._q_liquid_injection = float(specific["q_liquid_injection"])
        self._q_latent_condense = float(specific["q_latent_condense"])
        self._q_subcool_process = float(specific["q_subcool_process"])
        self._q_condense = float(specific["q_condense"])
        self._q_cond = float(specific["q_cond"])
        self._q_evap = self._q_source
        self._q_shaft = float(specific["q_shaft"])

        self._m_dot_source = float(m_dot)
        self._liquid_injection_ratio = float(specific["liquid_injection_ratio"])
        self._m_dot = self._m_dot_source * float(specific["gas_mass_factor"])
        self._Q_evap = self._m_dot_source * self._q_source
        self._Q_cool = self._Q_evap
        self._Q_cond = self._m_dot_source * self._q_cond
        self._Q_heat = self._Q_cond
        self._shaft_work = self._m_dot_source * self._q_shaft
        self._w_net = self._q_shaft / self._eta_motor
        self._work = self._shaft_work / self._eta_motor
        self._Q_cas_heat = 0.0
        self._Q_cas_cool = 0.0
        self._q_heat = self._q_cond
        self._q_cool = self._q_source
        self._q_cas_heat = 0.0
        self._q_cas_cool = 0.0
        self._max_work = max(abs(self._work), 1.0)
        self.temperature_unit = "C"
        self._solved = True
        self._process_split = process_split
        self._source_heat_is_external = bool(source_heat_is_external)
        self._process_heat_components = self.process_heat_components(process_split)
        self._process_m_dot_out = self._m_dot * (1.0 - process_split)
        return self._work

    def build_stream_collection(
        self,
        include_cond: bool = False,
        include_evap: bool = False,
        is_process_stream: bool = False,
        dtcont: float = 0.0,
        dt_diff_max: float = 0.5,
    ) -> StreamCollection:
        """Build external MVR process-heating streams.

        ``include_evap`` emits the source/generator duty only for cycles solved
        from external source heat. Serial cascade source heat is internal and is
        not emitted by this unit model.
        """
        self._require_solution()
        self._dtcont = dtcont
        self._dt_diff_max = dt_diff_max
        streams = StreamCollection()
        if include_cond:
            streams += self._build_process_condenser_streams(dtcont=dtcont)
        if include_evap and self._source_heat_is_external:
            streams += self._build_streams(
                self._build_evaporator_profile(),
                False,
                mass_flow=float(self._m_dot_source or 0.0),
            )
        for stream in streams:
            stream.is_process_stream = is_process_stream
        return streams

    def _build_streams(
        self,
        profile: np.ndarray,
        is_condenser: bool,
        *,
        mass_flow: float,
    ) -> StreamCollection:
        sc = StreamCollection()
        for i in range(len(profile) - 1):
            h1, T1 = profile[i]
            h2, T2 = profile[i + 1]
            if abs(T1 - T2) < 0.01:
                T2 = T1 - 0.01 if is_condenser else T1 + 0.01
            sc.add(
                Stream(
                    name=f"MVR_H{i + 1}" if is_condenser else f"MVR_C{i + 1}",
                    t_supply=T1,
                    t_target=T2,
                    heat_flow=mass_flow * abs(h1 - h2),
                    dt_cont=self._dtcont,
                )
            )
        return sc

    def _build_process_condenser_streams(
        self,
        *,
        stage_index: int = 1,
        dtcont: float | None = None,
    ) -> StreamCollection:
        """Build process-heating streams for this MVR stage."""
        self._require_solution()
        streams = StreamCollection()
        components = self.process_heat_components()
        dt_cont = self._dtcont if dtcont is None else float(dtcont)
        t_discharge = self.Ts[1] - 273.15
        t_sat = self.T_cond
        if components["desuperheat"] > 0.0:
            streams.add(
                Stream(
                    name=f"MVR_desuperheat_H{stage_index}",
                    t_supply=t_discharge,
                    t_target=t_sat,
                    heat_flow=components["desuperheat"],
                    dt_cont=dt_cont,
                )
            )
        if components["latent"] > 0.0:
            streams.add(
                Stream(
                    name=f"MVR_condense_H{stage_index}",
                    t_supply=t_sat,
                    t_target=t_sat - 0.01,
                    heat_flow=components["latent"],
                    dt_cont=dt_cont,
                )
            )
        if components["subcool"] > 0.0:
            streams.add(
                Stream(
                    name=f"MVR_subcool_H{stage_index}",
                    t_supply=t_sat,
                    t_target=t_sat - self.dT_subcool,
                    heat_flow=components["subcool"],
                    dt_cont=dt_cont,
                )
            )
        return streams

    def _build_evaporator_profile(self) -> np.ndarray:
        self._require_solution()
        p_low = self.Ps[0]
        h_start = self._compute_state_from_pressure_quality(p_low, 0.0).hmass()
        h_end = self.Hs[0]
        t_h_curve_points = []
        for h in np.linspace(h_start, h_end, 61):
            state = self._compute_state_from_pressure_enthalpy(P=p_low, h=h)
            t_h_curve_points.append([h, float(state.T()) - 273.15])
        return get_piecewise_data_points(
            curve=np.asarray(t_h_curve_points),
            is_hot_stream=False,
            dt_diff_max=self._dt_diff_max,
        )
