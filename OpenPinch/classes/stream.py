"""Data model representing process and utility streams."""

from typing import Optional

from ..lib.enums import ST


class Stream:
    """Generic thermal stream used for both process and utility duties.

    A :class:`Stream` stores supply/target states together with derived values
    such as hot/cold classification, shifted temperature bounds, heat-capacity
    flow rate, and simple economic attributes. The same class is reused for
    process streams, utilities, and derived net streams created during site-
    level aggregation.
    """

    def __init__(
        self,
        name: str = "Stream",
        t_supply: Optional[float] = None,
        t_target: Optional[float] = None,
        P_supply: Optional[float] = None,
        P_target: Optional[float] = None,
        h_supply: Optional[float] = None,
        h_target: Optional[float] = None,
        dt_cont: float = 0.0,
        dt_cont_act: Optional[float] = None,
        heat_flow: float | list[float, float] = 0.0,
        htc: float = 1.0,
        price: float = 0.0,
        is_process_stream: bool = True,
    ):
        """Initialise a stream and infer hot/cold classification."""
        self._name: str = name
        self._type: str = None
        self._t_supply: Optional[float] = t_supply
        self._t_target: Optional[float] = t_target
        self._P_supply: Optional[float] = P_supply
        self._P_target: Optional[float] = P_target
        self._h_supply: Optional[float] = h_supply
        self._h_target: Optional[float] = h_target
        self._dt_cont: float = dt_cont
        self._dt_cont_act: float = dt_cont if dt_cont_act is None else dt_cont_act
        self._heat_flow: float = heat_flow
        self._htc: float = htc if htc != 0.0 and isinstance(htc, float | int) else 1.0
        self._htr: float = 1 / self._htc
        self._price: float = price
        self._is_process_stream: bool = is_process_stream
        self._active = True
        self._update_attributes()

    # === Core Properties ===

    @property
    def name(self) -> str:
        """Stream name."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the display name used for reporting and graph labels."""
        self._name = value

    @property
    def is_process_stream(self) -> bool:
        """Process or utility stream."""
        return self._is_process_stream

    @is_process_stream.setter
    def is_process_stream(self, value: bool):
        """Mark whether the stream is treated as process-side or utility-side."""
        self._is_process_stream = value

    @property
    def type(self) -> Optional[str]:
        """Stream type (Hot, Cold, Both)."""
        return self._type

    @type.setter
    def type(self, value: str):
        """Override the inferred stream thermal type."""
        self._type = value

    @property
    def t_supply(self) -> Optional[float]:
        """Supply temperature (e.g., degC)."""
        return self._t_supply

    @t_supply.setter
    def t_supply(self, value: float):
        """Set the supply temperature and refresh derived stream attributes."""
        self._t_supply = value
        self._update_attributes()

    @property
    def t_target(self) -> Optional[float]:
        """Target temperature (e.g., degC)."""
        return self._t_target

    @t_target.setter
    def t_target(self, value: float):
        """Set the target temperature and refresh derived stream attributes."""
        self._t_target = value
        self._update_attributes()

    @property
    def P_supply(self) -> Optional[float]:
        """Supply pressure (e.g., kPa)."""
        return self._P_supply

    @P_supply.setter
    def P_supply(self, value: float):
        """Set the supply pressure and refresh derived stream attributes."""
        self._P_supply = value
        self._update_attributes()

    @property
    def P_target(self) -> Optional[float]:
        """Target pressure (e.g., kPa)."""
        return self._P_target

    @P_target.setter
    def P_target(self, value: float):
        """Set the target pressure and refresh derived stream attributes."""
        self._P_target = value
        self._update_attributes()

    @property
    def h_supply(self) -> Optional[float]:
        """Supply enthalpy (e.g., kJ/kg)."""
        return self._h_supply

    @h_supply.setter
    def h_supply(self, value: float):
        """Set the supply enthalpy and refresh derived stream attributes."""
        self._h_supply = value
        self._update_attributes()

    @property
    def h_target(self) -> Optional[float]:
        """Target enthalpy (e.g., kJ/kg)."""
        return self._h_target

    @h_target.setter
    def h_target(self, value: float):
        """Set the target enthalpy and refresh derived stream attributes."""
        self._h_target = value
        self._update_attributes()

    @property
    def dt_cont(self) -> float:
        """Preserved base delta-T contribution before any zone multiplier."""
        return self._dt_cont

    @dt_cont.setter
    def dt_cont(self, value: float):
        """Set the base contribution to shifted-temperature calculations."""
        self._dt_cont = value
        self._dt_cont_act = value
        self._update_attributes()

    @property
    def dt_cont_act(self) -> float:
        """Effective delta-T contribution used in shifted-temperature calculations."""
        return self._dt_cont_act

    @dt_cont_act.setter
    def dt_cont_act(self, value: float):
        """Set the effective shifted-temperature contribution in active use."""
        self._dt_cont_act = value
        self._update_attributes()

    @property
    def heat_flow(self) -> float:
        """Stream heat flow (e.g., kW)."""
        return self._heat_flow

    @heat_flow.setter
    def heat_flow(self, value: float):
        """Set the stream duty and refresh derived heat-capacity quantities."""
        self._heat_flow = value
        self._update_attributes()

    @property
    def htc(self) -> float:
        """Heat transfer coefficient (e.g., kW/m^2/K)."""
        return self._htc

    @htc.setter
    def htc(self, value: float):
        """Set the heat-transfer coefficient and refresh derived resistance."""
        self._htc = value
        self._update_attributes()

    @property
    def htr(self) -> float:
        """Heat transfer resistance (e.g., m^2.K/kW)."""
        return self._htr

    @htr.setter
    def htr(self, value: float):
        """Set the stored heat-transfer resistance explicitly."""
        self._htr = value

    @property
    def price(self) -> float:
        """Unit energy price (e.g., $/MWh)."""
        return self._price

    @price.setter
    def price(self, value: float):
        """Set the unit energy price used in utility-cost calculations."""
        self._price = value

    @property
    def ut_cost(self) -> float:
        """Utility cost (e.g., $/y)."""
        return self._ut_cost

    @ut_cost.setter
    def ut_cost(self, value: float):
        """Set the cached utility-cost figure for the stream."""
        self._ut_cost = value

    @property
    def CP(self) -> float:
        """Heat capacity flowrate (e.g., kW/K)."""
        return self._CP

    @CP.setter
    def CP(self, value: float):
        """Set the cached heat-capacity flow rate."""
        self._CP = value

    @property
    def rCP(self) -> Optional[float]:
        """Resistance-capacity product (1/heat transfer rate)."""
        return self._RCP_prod

    @rCP.setter
    def rCP(self, value: float):
        """Set the cached resistance-capacity product."""
        self._RCP_prod = value

    @property
    def active(self) -> bool:
        """Whether the stream is active in analysis."""
        return self._active

    @active.setter
    def active(self, value: bool):
        """Activate or deactivate the stream for downstream analysis."""
        self._active = value

    # === Computed Temperature Bounds ===

    @property
    def t_min(self) -> Optional[float]:
        """Minimum temperature (supply or target depending on hot/cold)."""
        return self._t_min

    @t_min.setter
    def t_min(self, value: float):
        """Set the unshifted lower temperature bound."""
        self._t_min = value

    @property
    def t_max(self) -> Optional[float]:
        """Maximum temperature (supply or target depending on hot/cold)."""
        return self._t_max

    @t_max.setter
    def t_max(self, value: float):
        """Set the unshifted upper temperature bound."""
        self._t_max = value

    @property
    def t_min_star(self) -> Optional[float]:
        """Shifted minimum temperature."""
        return self._t_min_star

    @t_min_star.setter
    def t_min_star(self, value: float):
        """Set the shifted lower temperature bound."""
        self._t_min_star = value

    @property
    def t_max_star(self) -> Optional[float]:
        """Shifted maximum temperature."""
        return self._t_max_star

    @t_max_star.setter
    def t_max_star(self, value: float):
        """Set the shifted upper temperature bound."""
        self._t_max_star = value

    # === Methods ===

    def _update_attributes(self) -> None:
        """Calculates key stream attributes based on temperatures."""
        if self._t_supply is None or self._t_target is None or self._htc is None:
            return

        if self._t_supply > self._t_target:
            # Hot stream
            self._set_hot_stream_min_max_temperatures()
        elif self._t_supply < self._t_target:
            # Cold stream
            self._set_cold_stream_min_max_temperatures()
        else:
            if isinstance(self._heat_flow, float | int):
                if self._heat_flow > 0.0:
                    # Cold stream
                    self._t_target = self._t_supply + 0.01
                    self._set_cold_stream_min_max_temperatures()
                elif self._heat_flow < 0.0:
                    # Hot stream
                    self._t_target = self._t_supply - 0.01
                    self._set_hot_stream_min_max_temperatures()
                else:
                    # Zero-duty isothermal streams stay neutral and keep their
                    # temperatures.
                    self._set_neutral_stream_min_max_temperatures()

        if isinstance(self._heat_flow, float | int):
            dt = self._t_max - self._t_min
            self._CP = self._heat_flow / dt if abs(dt) > 0.0 else 0.0
        elif isinstance(self._CP, float | int):
            self._heat_flow = self._CP * (self._t_max - self._t_min)

        self._calc_utility_cost()
        self._calc_htr_and_cp_product()

    def invert(self) -> None:
        """Flip a utility stream into its generating process-stream analogue."""
        if self._is_process_stream:
            raise ValueError(
                "Logic error: Process streams cannot be inverted; only utility "
                "streams may be inverted for generation."
            )

        ts = self._t_supply
        self._t_supply = self._t_target
        self._t_target = ts

        Ps = self._P_supply
        self._P_supply = self._P_target
        self._P_target = Ps

        hs = self._h_supply
        self._h_supply = self._h_target
        self._h_target = hs

        self._type = ST.Cold.value if self._type == ST.Hot.value else ST.Hot.value
        self._is_process_stream = True
        self._update_attributes()

    def set_heat_flow(self, value: float, units: str = "kW") -> None:
        """Sets the heat flow and updates CP and utility cost."""
        self._heat_flow = value
        self._calc_utility_cost()
        if (
            self._t_supply is not None
            and self._t_target is not None
            and abs(self._t_supply - self._t_target) > 0
        ):
            self._CP = value / abs(self._t_supply - self._t_target)
            self._RCP_prod = self._htr * self._CP

    def _calc_utility_cost(self):
        if isinstance(self._heat_flow, float | int) and isinstance(
            self._price, float | int
        ):
            self._ut_cost = (self._heat_flow / 1000) * self._price

    def _calc_htr_and_cp_product(self):
        if isinstance(self._heat_flow, float | int) and isinstance(
            self._price, float | int
        ):
            if self._htc != 0.0:
                self._htr = 1 / self._htc
                self._RCP_prod = self._CP * self._htr if self._htc > 0.0 else 0.0

    def _set_hot_stream_min_max_temperatures(self):
        self._t_min = self._t_target
        self._t_max = self._t_supply
        self._t_min_star = self._t_min - self._dt_cont_act
        self._t_max_star = self._t_max - self._dt_cont_act
        if self._type is None:
            self._type = ST.Hot.value

    def _set_cold_stream_min_max_temperatures(self):
        self._t_min = self._t_supply
        self._t_max = self._t_target
        self._t_min_star = self._t_min + self._dt_cont_act
        self._t_max_star = self._t_max + self._dt_cont_act
        if self._type is None:
            self._type = ST.Cold.value

    def _set_neutral_stream_min_max_temperatures(self):
        self._t_min = self._t_supply
        self._t_max = self._t_target
        self._t_min_star = self._t_min
        self._t_max_star = self._t_max
        self._type = ST.Both.value
