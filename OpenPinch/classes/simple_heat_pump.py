"""Simple Heat Pump cycle utilities that wrap CoolProp."""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence

import CoolProp
from CoolProp.Plots.Common import EURunits, KSIunits, PropertyDict, SIunits, process_fluid_state
from CoolProp.Plots.SimpleCycles import StateContainer
import numpy as np

from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection


__all__ = ['HeatPumpCycle']

class HeatPumpCycle:
    """Compute a simple vapor compression heat pump cycle.
    """

    STATECOUNT = 4
    UNIT_SYSTEMS = {
        'SI': SIunits(),
        'KSI': KSIunits(),
        'EUR': EURunits(),
    }

    def __init__(self, fluid_ref: str = 'Ammonia', unit_system: str | PropertyDict = 'EUR', Q_total: float = 1e6):
        self._system = self._parse_unit_system(unit_system)
        self._state = process_fluid_state(fluid_ref)
        self._cycle_states = StateContainer(unit_system=self._system)
        self._w_net: Optional[float] = None
        self._q_evap: Optional[float] = None
        self._Q_total: Optional[float] = Q_total
        self._solved = False

    @property
    def system(self) -> PropertyDict:
        return self._system

    @system.setter
    def system(self, value: str | PropertyDict) -> None:
        self._system = self._parse_unit_system(value)
        self._cycle_states.units = self._system

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value) -> None:
        self._state = process_fluid_state(value)
        self._solved = False

    @property
    def cycle_states(self) -> StateContainer:
        return self._cycle_states

    @cycle_states.setter
    def cycle_states(self, value: StateContainer) -> None:
        if len(value) != self.STATECOUNT:
            raise ValueError(f'Expected exactly {self.STATECOUNT} state points.')
        value.units = self._system
        self._cycle_states = value
        self._solved = True

    @property
    def states(self) -> StateContainer:
        """Expose state container for compatibility with plotting API."""
        return self._cycle_states

    @property
    def state_points(self) -> StateContainer:
        return self._cycle_states

    @property
    def Hs(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, 'H'] for i in self._cycle_states]

    @property
    def Ss(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, 'S'] for i in self._cycle_states]

    @property
    def Ts(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, CoolProp.iT] for i in self._cycle_states]

    @property
    def Ps(self) -> Sequence[float]:
        self._require_solution()
        return [self._cycle_states[i, CoolProp.iP] for i in self._cycle_states]

    @property
    def q_evap(self) -> Optional[float]:
        return self._q_evap

    @property
    def w_net(self) -> Optional[float]:
        return self._w_net

    def solve(
        self,
        T0: float,
        p0: float,
        T2: float,
        p2: float,
        eta_com: float,
        *,
        fluid=None,
        SI: bool = True,
    ) -> None:
        """Solve a basic four-state cycle given inlet/outlet temperatures and pressures."""
        if fluid is not None:
            self.state = fluid
        if self._state is None:
            raise ValueError('A fluid must be specified before solving the cycle.')

        T0, p0, T2, p2 = self._convert_to_SI((T0, p0, T2, p2), SI)

        cycle_states = StateContainer(unit_system=self._system)
        working_state = self._state

        # Evaporator outlet / compressor inlet
        working_state.update(CoolProp.PT_INPUTS, p0, T0)
        h0 = working_state.hmass()
        s0 = working_state.smass()
        cycle_states[0, 'H'] = h0
        cycle_states[0, 'S'] = s0
        cycle_states[0, CoolProp.iP] = p0
        cycle_states[0, CoolProp.iT] = T0

        # Compressor discharge (real)
        p1 = p2
        working_state.update(CoolProp.PSmass_INPUTS, p1, s0)
        h1_isentropic = working_state.hmass()
        h1 = h0 + (h1_isentropic - h0) / eta_com
        working_state.update(CoolProp.HmassP_INPUTS, h1, p1)
        s1 = working_state.smass()
        T1 = working_state.T()
        cycle_states[1, 'H'] = h1
        cycle_states[1, 'S'] = s1
        cycle_states[1, CoolProp.iP] = p1
        cycle_states[1, CoolProp.iT] = T1

        # Condenser outlet
        working_state.update(CoolProp.PT_INPUTS, p2, T2)
        h2 = working_state.hmass()
        s2 = working_state.smass()
        cycle_states[2, 'H'] = h2
        cycle_states[2, 'S'] = s2
        cycle_states[2, CoolProp.iP] = p2
        cycle_states[2, CoolProp.iT] = T2

        # Expansion valve outlet / evaporator inlet
        p3 = p0
        h3 = h2
        working_state.update(CoolProp.HmassP_INPUTS, h3, p3)
        s3 = working_state.smass()
        T3 = working_state.T()
        cycle_states[3, 'H'] = h3
        cycle_states[3, 'S'] = s3
        cycle_states[3, CoolProp.iP] = p3
        cycle_states[3, CoolProp.iT] = T3

        self.cycle_states = cycle_states
        self.fill_states()

        self._w_net = h0 - h1
        self._q_evap = h0 - h3

    def solve_t_dt(
        self,
        Te: float,
        Tc: float,
        dT_sh: float,
        dT_sc: float,
        eta_com: float,
        *,
        fluid=None,
        SI: bool = True,
    ) -> None:
        """Solve the cycle using saturation temperatures, superheat, and subcooling."""
        if fluid is not None:
            self.state = fluid
        if self._state is None:
            raise ValueError('A fluid must be specified before solving the cycle.')

        Te, Tc = self._convert_temperatures((Te, Tc), SI)

        self._state.update(CoolProp.QT_INPUTS, 1.0, Te)
        p0 = self._state.p()
        self._state.update(CoolProp.QT_INPUTS, 0.0, Tc)
        p2 = self._state.p()

        T0 = Te + dT_sh
        T2 = Tc - dT_sc

        self.solve(T0, p0, T2, p2, eta_com, fluid=None, SI=True)

    def solve_p_dt(
        self,
        Pe: float,
        Pc: float,
        dT_sh: float,
        dT_sc: float,
        eta_com: float,
        *,
        fluid=None,
        SI: bool = True,
    ) -> None:
        """Solve the cycle using pressures, superheat, and subcooling."""
        if fluid is not None:
            self.state = fluid
        if self._state is None:
            raise ValueError('A fluid must be specified before solving the cycle.')
        if Pe > Pc:
            raise ValueError('Evaporator pressure must be below condenser pressure.')

        p0, p2 = self._convert_to_SI((Pe, Pc), SI)
        p_crit = self._state.keyed_output(CoolProp.iP_critical)
        t_crit = self._state.keyed_output(CoolProp.iT_critical)

        if p0 < p_crit:
            self._state.update(CoolProp.PQ_INPUTS, p0, 1.0)
            Te_sat = self._state.T()
            T0 = Te_sat + dT_sh
        else:
            T0 = t_crit + dT_sh

        if p2 < p_crit:
            self._state.update(CoolProp.PQ_INPUTS, p2, 0.0)
            Tc_sat = self._state.T()
            T2 = Tc_sat - dT_sc
        else:
            T2 = t_crit - dT_sc

        self.solve(T0, p0, T2, p2, eta_com, fluid=None, SI=True)

    def fill_states(self, container: Optional[StateContainer] = None) -> StateContainer:
        """Populate missing properties for each state point where possible."""
        if container is None:
            container = self._cycle_states
            local = True
        else:
            local = False

        for i in container:
            state_point = container[i]
            if self._is_point_complete(state_point):
                continue

            if (
                state_point[CoolProp.iDmass] is not None
                and state_point[CoolProp.iT] is not None
            ):
                self._state.update(
                    CoolProp.DmassT_INPUTS,
                    state_point[CoolProp.iDmass],
                    state_point[CoolProp.iT],
                )
            elif (
                state_point[CoolProp.iP] is not None
                and state_point[CoolProp.iHmass] is not None
            ):
                self._state.update(
                    CoolProp.HmassP_INPUTS,
                    state_point[CoolProp.iHmass],
                    state_point[CoolProp.iP],
                )
            elif (
                state_point[CoolProp.iP] is not None
                and state_point[CoolProp.iSmass] is not None
            ):
                self._state.update(
                    CoolProp.PSmass_INPUTS,
                    state_point[CoolProp.iP],
                    state_point[CoolProp.iSmass],
                )
            elif (
                state_point[CoolProp.iP] is not None
                and state_point[CoolProp.iT] is not None
            ):
                try:
                    self._state.update(
                        CoolProp.PT_INPUTS,
                        state_point[CoolProp.iP],
                        state_point[CoolProp.iT],
                    )
                except:
                    self._state.update(
                        CoolProp.PQ_INPUTS,
                        state_point[CoolProp.iP],
                        1,
                    )                    
            else:
                warnings.warn(f'Insufficient data to fill state[{i}].', UserWarning)
                continue

            for prop in state_point:
                if state_point[prop] is None:
                    state_point[prop] = self._state.keyed_output(prop)

        if local:
            self._cycle_states = container
        return container

    def COP_heating(self) -> float:
        self._require_solution()
        return (self._cycle_states[1, 'H'] - self._cycle_states[2, 'H']) / (
            self._cycle_states[1, 'H'] - self._cycle_states[0, 'H']
        )

    def COP_cooling(self) -> float:
        self._require_solution()
        return (self._cycle_states[0, 'H'] - self._cycle_states[3, 'H']) / (
            self._cycle_states[1, 'H'] - self._cycle_states[0, 'H']
        )

    def get_hp_th_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return simplified condenser and evaporator T-h profiles."""
        self._require_solution()
        self.fill_states()

        condenser_profile = self._build_condenser_profile()
        condenser_sc = self._build_stream_collection(condenser_profile)
        evaporator_profile = self._build_evaporator_profile()

        return condenser_sc, evaporator_profile

    def _build_condenser_profile(self) -> np.ndarray:
        """Construct a four-point condenser T-h polyline."""
        p_high = self._cycle_states[1, CoolProp.iP]
        h_outlet = self._cycle_states[2, 'H']
        T_superheated = self._cycle_states[1, CoolProp.iT]
        h_superheated = self._cycle_states[1, 'H']
        T_subcooled = self._cycle_states[2, CoolProp.iT]

        self._state.update(CoolProp.PQ_INPUTS, p_high, 1.0)
        h_sat_vapor = self._state.hmass()
        T_sat_vapor = self._state.T()

        self._state.update(CoolProp.PQ_INPUTS, p_high, 0.0)
        h_sat_liquid = self._state.hmass()
        T_sat_liquid = self._state.T()

        condenser_profile = np.array(
            [
                [h_superheated, T_superheated],
                [h_sat_vapor, T_sat_vapor],
                [h_sat_liquid, T_sat_liquid],
                [h_outlet, T_subcooled],
            ],
            dtype=float,
        )

        return condenser_profile

    def _build_stream_collection(self,array):
        m_dot = self._Q_total / (array[0,0] - array[-1,0])
        sc = StreamCollection()
        for i in range(len(array) - 1):
            h1, T1 = array[i]
            h2, T2 = array[i + 1]

            # Do we need a name?
            name = f"Segment_{i + 1}"

            s = Stream(
                name=name,
                t_supply=T1,
                t_target=T2,
                heat_flow=m_dot*(h1 - h2),  # or m_dot * (h1 - h2), depending on your model
            )

            sc.add(s)
        return sc

    def _build_evaporator_profile(self) -> np.ndarray:
        """Construct a four-point evaporator T-h polyline."""
        p_low = self._cycle_states[0, CoolProp.iP]
        h_inlet = self._cycle_states[3, 'H']
        T_inlet = self._cycle_states[3, CoolProp.iT]

        h_superheated = self._cycle_states[0, 'H']
        T_superheated = self._cycle_states[0, CoolProp.iT]

        self._state.update(CoolProp.PQ_INPUTS, p_low, 1.0)
        h_sat_vapor = self._state.hmass()
        T_sat_vapor = self._state.T()

        evaporator_profile = np.array(
            [
                [h_inlet, T_inlet],
                [h_sat_vapor, T_sat_vapor],
                [h_superheated, T_superheated],
            ],
            dtype=float,
        )

        return evaporator_profile

    def _parse_unit_system(self, value: str | PropertyDict) -> PropertyDict:
        if isinstance(value, PropertyDict):
            return value
        if isinstance(value, str):
            key = value.upper()
            if key in self.UNIT_SYSTEMS:
                return self.UNIT_SYSTEMS[key]
        valid = ', '.join(sorted(self.UNIT_SYSTEMS.keys()))
        raise ValueError(f'Invalid unit system: {value!r}. Expected one of {valid}.')

    def _convert_to_SI(self, data: Iterable[float], already_SI: bool) -> tuple[float, float, float, float]:
        if already_SI:
            return tuple(data)
        conv_t = self._system[CoolProp.iT].to_SI
        conv_p = self._system[CoolProp.iP].to_SI
        T0, p0, T2, p2 = data
        return conv_t(T0), conv_p(p0), conv_t(T2), conv_p(p2)

    def _convert_temperatures(self, temps: Iterable[float], already_SI: bool) -> tuple[float, ...]:
        if already_SI:
            return tuple(temps)
        conv_t = self._system[CoolProp.iT].to_SI
        return tuple(conv_t(temp) for temp in temps)

    @staticmethod
    def _is_point_complete(point: PropertyDict) -> bool:
        for prop in point:
            if point[prop] is None:
                return False
        return True

    def _require_solution(self) -> None:
        if not self._solved:
            raise RuntimeError('Solve the cycle before accessing results.')
