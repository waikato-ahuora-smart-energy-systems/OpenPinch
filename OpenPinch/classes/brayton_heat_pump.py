"""Brayton-cycle heat pump model used by advanced utility targeting workflows.

The class in this module wraps a TESPy network while exposing a simplified API
compatible with other OpenPinch heat pump cycle helpers.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

try:
    from tespy.networks import Network
    from tespy.components import CycleCloser, Compressor, Turbine, SimpleHeatExchanger
    from tespy.connections import Connection
except ImportError as exc:  # pragma: no cover - optional dependency guard
    Network = None
    CycleCloser = Compressor = Turbine = SimpleHeatExchanger = Connection = None
    _TESPY_IMPORT_ERROR = exc
else:
    _TESPY_IMPORT_ERROR = None

# Local stream API used by the simple heat pump
from .stream import Stream
from .stream_collection import StreamCollection


def _require_tespy() -> None:
    if _TESPY_IMPORT_ERROR is None:
        return
    raise ImportError(
        "TESPy is required for Brayton-cycle tooling. "
        "Install it with 'pip install openpinch[cycles]'."
    ) from _TESPY_IMPORT_ERROR


class SimpleBraytonHeatPumpCycle:
    """Brayton heat pump cycle using TESPy internally.

    Public API mirrors the simple Rankine ``HeatPumpCycle`` class so the
    object is interchangeable in downstream code.

    Notes
    -----
    - The solver uses ``Network.solve(mode="design")``.
    - Pressures are left to TESPy to determine (option A). The user
      provides compressor inlet/outlet temperatures and the heat duty in
      the HTHX (Q_ht). Compressor and turbine isentropic efficiencies
      must be specified.
    - The cycle-state mapping is:
      ``0=C1 compressor inlet``, ``1=C2 compressor outlet``,
      ``2=C3 turbine inlet``, ``3=C4 turbine outlet``.
    """

    STATECOUNT = 4

    def __init__(self):
        """Initialize an unsolved Brayton heat pump cycle container."""
        _require_tespy()

        # Keep minimal unit-system compatibility surface (not using CoolProp
        # unit dicts here); the simple heat pump used a PropertyDict. For
        # compatibility we accept the same constructor signature.
        self.refrigerant = None
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
        self._states = [
            dict(T=None, p=None, h=None, s=None, m=None) for _ in range(self.STATECOUNT)
        ]

    # -- Properties to mimic HeatPumpCycle API -------------------------------------------------
    @property
    def cycle_states(self):
        """Return cycle state data in a compatibility-oriented list structure.

        Returns
        -------
        list[dict]
            State dictionaries in cycle order:
            ``0=compressor inlet``, ``1=compressor outlet``,
            ``2=turbine inlet``, ``3=turbine outlet``.
        """
        return self._states

    @property
    def states(self):
        """Alias for :attr:`cycle_states`.

        Returns
        -------
        list[dict]
            Cycle state dictionaries.
        """
        return self.cycle_states

    @property
    def Hs(self) -> Sequence[float]:
        """Return state specific enthalpies.

        Returns
        -------
        Sequence[float]
            Enthalpy values [J/kg] for states 0..3.
        """
        self._require_solution()
        return [s["h"] for s in self._states]

    @property
    def Ts(self) -> Sequence[float]:
        """Return state temperatures.

        Returns
        -------
        Sequence[float]
            Temperatures [degC] for states 0..3.
        """
        self._require_solution()
        return [s["T"] for s in self._states]

    @property
    def Ps(self) -> Sequence[float]:
        """Return state pressures.

        Returns
        -------
        Sequence[float]
            Pressures [Pa] for states 0..3.
        """
        self._require_solution()
        return [s["p"] for s in self._states]

    @property
    def Ss(self) -> Sequence[float]:
        """Return state specific entropies when available.

        Returns
        -------
        Sequence[float]
            Entropy values for states 0..3. Entries may be ``None`` when not
            populated by the underlying model.
        """
        self._require_solution()
        return [s.get("s") for s in self._states]

    @property
    def Q_heat(self) -> Optional[float]:
        """Return configured heat-delivery target.

        Returns
        -------
        float or None
            Requested gas-cooler heat duty [kW].
        """
        return self._Q_heat

    @property
    def Q_cool(self) -> Optional[float]:
        """Return low-temperature heat-rejection duty after solution.

        Returns
        -------
        float or None
            LTHX duty [kW].
        """
        return self._Q_cool

    @property
    def work_net(self) -> Optional[float]:
        """Return net shaft work after solution.

        Returns
        -------
        float or None
            Compressor plus turbine power [kW] using TESPy sign convention.
        """
        return self._work_net

    # -- Solver API ---------------------------------------------------------------------------
    def solve(
        self,
        T_comp_in: float,
        T_comp_out: float,
        dT_gc: float,
        Q_heat: float,
        eta_comp: float,
        eta_exp: float,
        is_recuperated: bool,
        refrigerant=None,
    ) -> None:
        """Solve the Brayton cycle using TESPy.

        Parameters
        ----------
        T_comp_in : float
            Compressor inlet temperature [degC] (state 1).
        T_comp_out : float
            Compressor outlet temperature [degC] (state 2).
        dT_gc : float
            Temperature difference between compressor outlet and turbine inlet:
            ``dT_gc = T_comp_out - T_turb_in``.
        Q_heat : float
            Heat delivered in the gas cooler [kW], positive for process heating.
        eta_comp : float
            Compressor isentropic efficiency (fraction).
        eta_exp : float
            Turbine/expander isentropic efficiency (fraction).
        is_recuperated : bool
            Whether recuperation is requested. Currently ignored and downgraded
            to a warning.
        refrigerant : Any, optional
            Working-fluid label accepted for API compatibility.

        Returns
        -------
        None
            The cycle object is updated in place with solved states and duties.

        Raises
        ------
        RuntimeError
            If TESPy solves but result extraction fails.
        """
        # Save inputs
        self.refrigerant = refrigerant
        self._Q_heat = Q_heat

        # Note: is_recuperated parameter is not currently implemented
        # Future enhancement: add a recuperator component to the cycle
        if is_recuperated:
            warnings.warn(
                "Recuperated Brayton cycle is not yet implemented. "
                "Proceeding with simple cycle.",
                UserWarning,
            )

        # Create TESPy network and components (following original script)
        fluid_list = ["Ar", "N2", "CO2", "O2"]
        BraytonHP = Network(fluids=fluid_list)
        BraytonHP.set_attr(T_unit="C", p_unit="bar", h_unit="kJ / kg")

        # components
        FlowStart = CycleCloser("cycle closer")
        Comp = Compressor("compressor")
        Turb = Turbine("turbine")
        HTHX = SimpleHeatExchanger("HTHX")
        LTHX = SimpleHeatExchanger("LTHX")

        # connections (labels match the original script)
        C1 = Connection(FlowStart, "out1", Comp, "in1", label="s1")
        C2 = Connection(Comp, "out1", HTHX, "in1", label="s2")
        C3 = Connection(HTHX, "out1", Turb, "in1", label="s3")
        C4 = Connection(Turb, "out1", LTHX, "in1", label="s4")
        C5 = Connection(LTHX, "out1", FlowStart, "in1", label="s5")

        # set attributes as in the original script
        C1.set_attr(p=1.013, T=T_comp_in, fluid={"Air": 1})
        C2.set_attr(T=T_comp_out)

        # Set turbine inlet temperature according to dT_gc (= T_comp_out - T_turb_in)
        T_turb_in = T_comp_out - dT_gc
        C3.set_attr(T=T_turb_in)

        Comp.set_attr(eta_s=eta_comp)
        # preserve original sign convention: in the original script HTHX.Q = -x[3]
        HTHX.set_attr(pr=0.993, Q=-Q_heat)  # pr2=0.98,
        LTHX.set_attr(pr=0.98)  # pr2=0.995, ttd_l=10
        Turb.set_attr(eta_s=eta_exp)

        BraytonHP.add_conns(C1, C2, C3, C4, C5)  # , C8, C9, C10, C11)
        BraytonHP.set_attr(iterinfo=False)

        # run TESPy design solve (as requested)
        BraytonHP.solve(mode="design", print_results=False)

        # Save network and connections
        self._network = BraytonHP
        self._conns = dict(
            s1=C1, s2=C2, s3=C3, s4=C4, s5=C5
        )  # , s8=C8, s9=C9, s10=C10, s11=C11)

        try:
            self._states[0]["T"] = C1.T.val  # [degC]
            self._states[0]["p"] = C1.p.val * 1e5  # bar -> Pa
            self._states[0]["h"] = C1.h.val * 1000.0  # kJ/kg -> J/kg
            self._states[0]["m"] = C1.m.val

            self._states[1]["T"] = C2.T.val
            self._states[1]["p"] = C2.p.val * 1e5
            self._states[1]["h"] = C2.h.val * 1000.0
            self._states[1]["m"] = C2.m.val

            self._states[2]["T"] = C3.T.val
            self._states[2]["p"] = C3.p.val * 1e5
            self._states[2]["h"] = C3.h.val * 1000.0
            self._states[2]["m"] = C3.m.val

            self._states[3]["T"] = C4.T.val
            self._states[3]["p"] = C4.p.val * 1e5
            self._states[3]["h"] = C4.h.val * 1000.0
            self._states[3]["m"] = C4.m.val

            self._work_net = Comp.P.val + Turb.P.val  # kW (signed)
            self._m_dot = C1.m.val
            self._Q_cool = LTHX.Q.val

            self._solved = True

        except Exception as e:
            raise RuntimeError(f"Failed to extract results from TESPy network: {e}")

        return self._work_net

    def _build_hthx_profile(self) -> np.ndarray:
        """Build a simplified gas-cooler T-h profile.

        Returns
        -------
        np.ndarray
            Two-point profile with columns ``[h (J/kg), T (degC)]``.
        """
        self._require_solution()
        H = self.Hs
        T = self.Ts
        # create a conservative 4 point: compressor outlet -> (same) -> (same) -> turbine inlet
        profile = np.array(
            [
                [H[1], T[1]],
                # [H[1], T[1]],
                # [H[2], T[2]],
                [H[2], T[2]],
            ],
            dtype=float,
        )
        return profile

    def _build_lthx_profile(self) -> np.ndarray:
        """Build a simplified gas-heater T-h profile.

        Returns
        -------
        np.ndarray
            Two-point profile with columns ``[h (J/kg), T (degC)]``.
        """
        self._require_solution()
        H = self.Hs
        T = self.Ts
        profile = np.array(
            [
                [H[3], T[3]],
                # [H[3], T[3]],
                [H[0], T[0]],
            ],
            dtype=float,
        )
        return profile

    def get_hp_th_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return hot- and cold-side T-h profiles.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(HTHX_profile, LTHX_profile)``.
        """
        return (self._build_hthx_profile(), self._build_lthx_profile())

    def get_hp_hot_and_cold_streams(self) -> tuple[StreamCollection, StreamCollection]:
        """Convert solved profiles to hot and cold utility stream collections.

        Returns
        -------
        tuple[StreamCollection, StreamCollection]
            Hot streams from HTHX and cold streams from LTHX.
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
                s = Stream(
                    name=name,
                    t_supply=T1,
                    t_target=t_target,
                    heat_flow=heat_flow,
                    is_process_stream=False,
                )
                sc.add(s)
            return sc

        hot_sc = _build_streams(hot_profile, True)
        cold_sc = _build_streams(cold_profile, False)
        return hot_sc, cold_sc

    def build_stream_collection(
        self,
        include_cond: bool = False,
        include_evap: bool = False,
        is_process_stream: bool = False,
    ) -> StreamCollection:
        """Build a combined stream collection for selected heat exchangers.

        Parameters
        ----------
        include_cond : bool, default=False
            Include hot-side (gas-cooler) streams.
        include_evap : bool, default=False
            Include cold-side (gas-heater) streams.
        is_process_stream : bool, default=False
            Compatibility argument retained for shared API alignment.

        Returns
        -------
        StreamCollection
            Aggregated stream collection based on selected sides.
        """

        sc = StreamCollection()
        if include_cond:
            hot, _ = self.get_hp_hot_and_cold_streams()
            sc += hot
        if include_evap:
            _, cold = self.get_hp_hot_and_cold_streams()
            sc += cold
        return sc

    def _require_solution(self) -> None:
        """Validate that the cycle has been solved before data access.

        Raises
        ------
        RuntimeError
            If ``solve`` has not been called successfully.
        """
        if not self._solved:
            raise RuntimeError("Solve the cycle before accessing results.")
