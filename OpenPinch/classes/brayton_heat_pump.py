from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

# TESPy imports
from tespy.networks import Network
from tespy.components import (CycleCloser, Compressor, Turbine, SimpleHeatExchanger)
from tespy.connections import Connection

# Local stream API used by the simple heat pump
from .stream import Stream
from .stream_collection import StreamCollection


class SimpleBraytonHeatPumpCycle:
    """Brayton heat pump cycle using TESPy internally.

    Public API mirrors the simple Rankine `HeatPumpCycle` class so the
    object is interchangeable in downstream code.

    Notes
    -----
    - The solver uses the TESPy `Network.solve(mode='design')` call (as
      requested by the user).
    - Pressures are left to TESPy to determine (option A). The user
      provides compressor inlet/outlet temperatures and the heat duty in
      the HTHX (Q_ht). Compressor and turbine isentropic efficiencies
      must be specified.
    - The 4 cycle states are mapped as follows (matching the provided
      Brayton script):
        0: compressor inlet (C1)
        1: compressor outlet (C2)
        2: turbine inlet (C3)
        3: turbine outlet (C4)
    """

    STATECOUNT = 4

    def __init__(self):
        # Keep minimal unit-system compatibility surface (not using CoolProp
        # unit dicts here); the simple heat pump used a PropertyDict. For
        # compatibility we accept the same constructor signature.
        self.refrigerant = None
        self.unit_system = None
        self._Q_heat: Optional[float] = None
        self._Q_cool: Optional[float] = None

        # Results placeholders
        self._m_dot: Optional[float] = None
        self._work_net: Optional[float] = None
        self._solved = False

        # store TESPy network and connections after solve
        self._network: Optional[Network] = None
        self._conns = {}
        self._work: Optional[float] = None

        # simple storage for 4 states: each state will be a dict with keys 'T', 'p', 'h', 's', 'm'
        self._states = [dict(T=None, p=None, h=None, s=None, m=None) for _ in range(self.STATECOUNT)]

    # -- Properties to mimic HeatPumpCycle API -------------------------------------------------
    @property
    def cycle_states(self):
        """Expose a simple state container (list-like) for compatibility similar to a CoolProp state container.
        """
        return self._states

    @property
    def states(self):
        return self.cycle_states

    @property
    def Hs(self) -> Sequence[float]:
        self._require_solution()
        return [s['h'] for s in self._states]

    @property
    def Ts(self) -> Sequence[float]:
        self._require_solution()
        return [s['T'] for s in self._states]

    @property
    def Ps(self) -> Sequence[float]:
        self._require_solution()
        return [s['p'] for s in self._states]

    @property
    def Ss(self) -> Sequence[float]:
        self._require_solution()
        return [s.get('s') for s in self._states]

    @property
    def Q_heat(self) -> Optional[float]:
        return self._Q_heat
    
    @property
    def Q_cool(self) -> Optional[float]:
        return self._Q_cool

    @property
    def work_net(self) -> Optional[float]:
        return self._work_net
    

    # -- Solver API ---------------------------------------------------------------------------
    def solve(
            self,
            T_comp_in: float,
            T_comp_out: float,
            dT_gc: float,
            Q_h_total: float,
            eta_comp: float,
            eta_exp: float,
            is_recuperated: bool,
            refrigerant=None,
    ) -> None:
        """Solve the Brayton cycle using TESPy.

        Parameters
        ----------
        T_comp_in
            Compressor inlet temperature [°C] (T1)
        T_comp_out
            Compressor outlet temperature [°C] (T2)
        dT_gc
            Temperature difference between compressor outlet and turbine inlet:
            dT_gc = T_comp_out - T_turb_in (temperature drop in HTHX)
        Q_h_total
            Total heat duty of the HTHX (positive value for heat delivered to the process) [kW]
        eta_comp
            Compressor isentropic efficiency (fraction, e.g. 0.83)
        eta_exp
            Expander/turbine isentropic efficiency (fraction, e.g. 0.93)
        is_recuperated
            Whether the cycle includes recuperation (currently not implemented)
        refrigerant
            Working fluid name (currently fixed to air composition, not used)
        """
        # Save inputs
        self.refrigerant = refrigerant
        self._Q_heat = Q_h_total
        
        # Note: is_recuperated parameter is not currently implemented
        # Future enhancement: add a recuperator component to the cycle
        if is_recuperated:
            warnings.warn("Recuperated Brayton cycle is not yet implemented. "
                         "Proceeding with simple cycle.", UserWarning)

        # Create TESPy network and components (following original script)
        fluid_list = ['Ar', 'N2', 'CO2', 'O2']
        BraytonHP = Network(fluids=fluid_list)
        BraytonHP.set_attr(T_unit='C', p_unit='bar', h_unit='kJ / kg')

        # components
        FlowStart = CycleCloser('cycle closer')
        Comp = Compressor('compressor')
        Turb = Turbine('turbine')
        HTHX = SimpleHeatExchanger('HTHX')
        LTHX = SimpleHeatExchanger('LTHX')

        # connections (labels match the original script)
        C1 = Connection(FlowStart, 'out1', Comp, 'in1', label='s1')
        C2 = Connection(Comp, 'out1', HTHX, 'in1', label='s2')
        C3 = Connection(HTHX, 'out1', Turb, 'in1', label='s3')
        C4 = Connection(Turb, 'out1', LTHX, 'in1', label='s4')
        C5 = Connection(LTHX, 'out1', FlowStart, 'in1', label='s5')

        # set attributes as in the original script
        C1.set_attr(p=1.013, T=T_comp_in, fluid={"Air": 1})
        C2.set_attr(T=T_comp_out)

        # Set turbine inlet temperature according to dT_gc (= T_comp_out - T_turb_in)
        T_turb_in = T_comp_out - dT_gc
        C3.set_attr(T=T_turb_in)

        Comp.set_attr(eta_s=eta_comp)
        # preserve original sign convention: in the original script HTHX.Q = -x[3]
        HTHX.set_attr(pr=0.993, Q=-Q_h_total) #pr2=0.98, 
        LTHX.set_attr(pr=0.98) # pr2=0.995, ttd_l=10
        Turb.set_attr(eta_s=eta_exp)

        BraytonHP.add_conns(C1, C2, C3, C4, C5) #, C8, C9, C10, C11)
        BraytonHP.set_attr(iterinfo=False)

        # run TESPy design solve (as requested)
        BraytonHP.solve(mode='design', print_results=False)

        # Save network and connections
        self._network = BraytonHP
        self._conns = dict(s1=C1, s2=C2, s3=C3, s4=C4, s5=C5) #, s8=C8, s9=C9, s10=C10, s11=C11)

        try:
            self._states[0]['T'] = C1.T.val  # [°C]
            self._states[0]['p'] = C1.p.val * 1e5  # bar -> Pa
            self._states[0]['h'] = C1.h.val * 1000.0  # kJ/kg -> J/kg
            self._states[0]['m'] = C1.m.val

            self._states[1]['T'] = C2.T.val
            self._states[1]['p'] = C2.p.val * 1e5
            self._states[1]['h'] = C2.h.val * 1000.0
            self._states[1]['m'] = C2.m.val

            self._states[2]['T'] = C3.T.val
            self._states[2]['p'] = C3.p.val * 1e5
            self._states[2]['h'] = C3.h.val * 1000.0
            self._states[2]['m'] = C3.m.val

            self._states[3]['T'] = C4.T.val
            self._states[3]['p'] = C4.p.val * 1e5
            self._states[3]['h'] = C4.h.val * 1000.0
            self._states[3]['m'] = C4.m.val

            self._work_net = Comp.P.val + Turb.P.val  # kW (signed)
            self._m_dot = C1.m.val
            self._Q_cool = LTHX.Q.val

            self._solved = True

        except Exception as e:
            raise RuntimeError(f'Failed to extract results from TESPy network: {e}')

        return self._work_net


    def _build_hthx_profile(self) -> np.ndarray:
        """Build a simple 4-point T-h profile for the HTHX (hot side).

        Points: [superheated inlet (C2), saturated vapor/liquid approximation (not
        applicable for gas Brayton) ... bypassed; we produce a 4-point polyline
        compatible with the Rankine routine]:

        Return shape: (4,2) columns [h (J/kg), T (°C)]
        """
        self._require_solution()
        H = self.Hs
        T = self.Ts
        # create a conservative 4 point: compressor outlet -> (same) -> (same) -> turbine inlet
        profile = np.array([
            [H[1], T[1]],
            # [H[1], T[1]],
            # [H[2], T[2]],
            [H[2], T[2]],
        ], dtype=float)
        return profile


    def _build_lthx_profile(self) -> np.ndarray:
        """Build a 3-point evaporator-style profile for the LTHX (cold side).

        Return shape: (3,2) columns [h (J/kg), T (°C)]
        """
        self._require_solution()
        H = self.Hs
        T = self.Ts
        profile = np.array([
            [H[3], T[3]],
            # [H[3], T[3]],
            [H[0], T[0]],
        ], dtype=float)
        return profile


    def get_hp_th_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return condenser (HTHX) and evaporator (LTHX) T-h profiles."""
        return (self._build_hthx_profile(), self._build_lthx_profile())


    def get_hp_hot_and_cold_streams(self) -> tuple[StreamCollection, StreamCollection]:
        """Return two StreamCollections (hot, cold) similar to HeatPumpCycle.

        Hot = HTHX (compressor outlet -> turbine inlet)
        Cold = LTHX (turbine outlet -> compressor inlet)
        """
        self._require_solution()

        hot_profile = self._build_hthx_profile()
        cold_profile = self._build_lthx_profile()

        def _build_streams(profile: np.ndarray, is_hot: bool) -> StreamCollection:
            sc = StreamCollection()
            # calculate m_dot from TESPy if available
            m_dot = self._m_dot if self._m_dot is not None else 1.0
            for i in range(len(profile) - 1):
                h1, T1 = profile[i]
                h2, T2 = profile[i + 1]
                name = f"Segment_{i + 1}"
                # simple target logic similar to Rankine implementation
                if abs(T1 - T2) < 1e-6:
                    t_target = T2 + (0.001 if not is_hot else -0.001)
                else:
                    t_target = T2
                heat_flow = m_dot * abs(h1 - h2)
                s = Stream(name=name, t_supply=T1, t_target=t_target, heat_flow=heat_flow, is_process_stream=False)
                sc.add(s)
            return sc

        hot_sc = _build_streams(hot_profile, True)
        cold_sc = _build_streams(cold_profile, False)
        return hot_sc, cold_sc


    def build_stream_collection(self, include_cond: bool = False, include_evap: bool = False,
                                is_process_stream: bool = False) -> StreamCollection:

        sc = StreamCollection()
        if include_cond:
            hot, _ = self.get_hp_hot_and_cold_streams()
            sc += hot
        if include_evap:
            _, cold = self.get_hp_hot_and_cold_streams()
            sc += cold
        return sc


    def _require_solution(self) -> None:
        if not self._solved:
            raise RuntimeError('Solve the cycle before accessing results.')
