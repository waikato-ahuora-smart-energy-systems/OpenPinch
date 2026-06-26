"""Data model representing process and utility streams."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Optional

import numpy as np

from ..lib.coolprop_fluids import validate_coolprop_fluid_name
from ..lib.enums import ST, FluidPhase
from ..lib.schemas.common import MaybeVU
from .value import Value

_TEMPERATURE_EQUAL_TOL = 1e-12


class Stream:
    """Generic thermal stream used for both process and utility duties.

    A :class:`Stream` stores supply/target states together with derived values
    such as hot/cold classification, shifted temperature bounds, heat-capacity
    flow rate, and simple economic attributes. The same class is reused for
    process streams, utilities, and derived net streams created during site-
    level aggregation.
    """

    _VALUE_UNITS = {
        "_t_supply": "degC",
        "_t_target": "degC",
        "_p_supply": "kPa",
        "_p_target": "kPa",
        "_h_supply": "kJ/kg",
        "_h_target": "kJ/kg",
        "_dt_cont": "delta_degC",
        "_dt_cont_act": "delta_degC",
        "_heat_flow": "kW",
        "_htc": "kW/m^2/delta_degC",
        "_htr": "m^2*delta_degC/kW",
        "_price": "$/MW/h",
        "_cost": "$/h",
        "_cp": "kW/delta_degC",
        "_rcp_prod": "m^2",
        "_t_min": "degC",
        "_t_max": "degC",
        "_t_min_star": "degC",
        "_t_max_star": "degC",
    }
    _CORE_VALUE_ATTRS = (
        "_t_supply",
        "_t_target",
        "_p_supply",
        "_p_target",
        "_h_supply",
        "_h_target",
        "_dt_cont",
        "_heat_flow",
        "_htc",
        "_price",
    )
    _DERIVED_VALUE_ATTRS = (
        "_dt_cont_act",
        "_t_min",
        "_t_max",
        "_t_min_star",
        "_t_max_star",
        "_cp",
        "_htr",
        "_cost",
        "_rcp_prod",
    )
    _MUTABLE_VALUE_ATTRS = frozenset(
        attr.lstrip("_") for attr in _CORE_VALUE_ATTRS
    ) | frozenset(_CORE_VALUE_ATTRS)

    def __init__(
        self,
        name: str = "Stream",
        t_supply: Optional[MaybeVU] = None,
        t_target: Optional[MaybeVU] = None,
        p_supply: Optional[MaybeVU] = None,
        p_target: Optional[MaybeVU] = None,
        h_supply: Optional[MaybeVU] = None,
        h_target: Optional[MaybeVU] = None,
        dt_cont: MaybeVU = 0.0,
        dt_cont_multiplier: float = 1.0,
        heat_flow: MaybeVU = 0.0,
        htc: MaybeVU = 1.0,
        price: MaybeVU = 0.0,
        is_process_stream: bool = True,
        fluid_name: Optional[str] = None,
        fluid_phase: Optional[str | FluidPhase] = None,
    ):
        """Initialise a stream and infer hot/cold classification."""
        self._name = name
        self._is_process_stream = bool(is_process_stream)
        self._fluid_name = self._normalise_fluid_name(fluid_name)
        self._fluid_phase = self._normalise_fluid_phase(fluid_phase)
        self._active = True
        self._dt_cont_multiplier_locked = False
        self._dt_cont_multiplier = float(dt_cont_multiplier or 1.0)
        self._numeric_revision = 0

        self._period_ids: dict[str, int] | None = None
        self._weights: np.ndarray | None = None

        self._num_periods: int | None = None
        self._type: str | None = None

        self._t_supply: Value | None = None
        self._t_target: Value | None = None
        self._p_supply: Value | None = None
        self._p_target: Value | None = None
        self._h_supply: Value | None = None
        self._h_target: Value | None = None
        self._dt_cont: Value | None = None
        self._heat_flow: Value | None = None
        self._htc: Value | None = None
        self._price: Value | None = None

        self._dt_cont_act: Value | None = None
        self._t_min: Value | None = None
        self._t_max: Value | None = None
        self._t_min_star: Value | None = None
        self._t_max_star: Value | None = None
        self._cp: Value | None = None
        self._htr: Value | None = None
        self._cost: Value | None = None
        self._rcp_prod: Value | None = None

        self.set_value_attr("_t_supply", t_supply, update_derived=False)
        self.set_value_attr("_t_target", t_target, update_derived=False)
        self.set_value_attr("_p_supply", p_supply, update_derived=False)
        self.set_value_attr("_p_target", p_target, update_derived=False)
        self.set_value_attr("_h_supply", h_supply, update_derived=False)
        self.set_value_attr("_h_target", h_target, update_derived=False)
        self.set_value_attr("_dt_cont", dt_cont, update_derived=False)
        self.set_value_attr("_heat_flow", heat_flow, update_derived=False)
        self.set_value_attr("_htc", htc, update_derived=False)
        self.set_value_attr("_price", price, update_derived=False)
        self._validate_num_periods()
        self._calculate_missing_properties()
        self.update_derived_properties()

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
    def stream_name(self) -> str:
        """Alias for the stream identifier used in input schemas."""
        return self._name

    @stream_name.setter
    def stream_name(self, value: str):
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
    def fluid_name(self) -> Optional[str]:
        """CoolProp fluid name or mixture specification."""
        return self._fluid_name

    @fluid_name.setter
    def fluid_name(self, value: Optional[str]):
        self._fluid_name = self._normalise_fluid_name(value)

    @property
    def fluid_phase(self) -> Optional[str]:
        """Optional fluid-phase flag: sol, sle, liq, vle, sve, or gas."""
        return self._fluid_phase

    @fluid_phase.setter
    def fluid_phase(self, value: Optional[str | FluidPhase]):
        self._fluid_phase = self._normalise_fluid_phase(value)

    @property
    def type(self) -> Optional[str]:
        """Stream type (Hot, Cold, Both)."""
        return self._type

    @property
    def num_periods(self) -> Optional[int]:
        """Number of periods."""
        return self._num_periods

    @property
    def period_ids(self) -> dict[str, int] | None:
        return self._period_ids

    @property
    def weights(self) -> np.ndarray | None:
        return self._weights

    @property
    def t_supply(self) -> Optional[Value]:
        """Supply temperature (e.g., degC)."""
        return self._t_supply

    @t_supply.setter
    def t_supply(self, value):
        self.set_value_attr("_t_supply", value)

    @property
    def t_target(self) -> Optional[Value]:
        """Target temperature (e.g., degC)."""
        return self._t_target

    @t_target.setter
    def t_target(self, value):
        self.set_value_attr("_t_target", value)

    @property
    def p_supply(self) -> Optional[Value]:
        """Supply pressure (e.g., kPa)."""
        return self._p_supply

    @p_supply.setter
    def p_supply(self, value):
        self.set_value_attr("_p_supply", value)

    @property
    def p_target(self) -> Optional[Value]:
        """Target pressure (e.g., kPa)."""
        return self._p_target

    @p_target.setter
    def p_target(self, value):
        self.set_value_attr("_p_target", value)

    @property
    def h_supply(self) -> Optional[Value]:
        """Supply enthalpy (e.g., kJ/kg)."""
        return self._h_supply

    @h_supply.setter
    def h_supply(self, value):
        self.set_value_attr("_h_supply", value)

    @property
    def h_target(self) -> Optional[Value]:
        """Target enthalpy (e.g., kJ/kg)."""
        return self._h_target

    @h_target.setter
    def h_target(self, value):
        self.set_value_attr("_h_target", value)

    @property
    def dt_cont(self) -> Value:
        """Preserved base delta-T contribution before any zone multiplier."""
        return self._dt_cont

    @dt_cont.setter
    def dt_cont(self, value):
        self.set_value_attr("_dt_cont", value)

    @property
    def dt_cont_act(self) -> Value:
        """Effective delta-T contribution used in shifted-temperature calculations."""
        return self._dt_cont_act

    @property
    def dt_cont_multiplier(self) -> float:
        """Effective delta-T contribution used in shifted-temperature calculations."""
        return self._dt_cont_multiplier

    @dt_cont_multiplier.setter
    def dt_cont_multiplier(self, value: float):
        """Set the effective shifted-temperature contribution in active use."""
        if not self._dt_cont_multiplier_locked:
            self._dt_cont_multiplier = float(value)
            self._bump_numeric_revision()
            self.update_derived_properties()
        else:
            warnings.warn(
                "Attempted to change dt_cont_multiplier, but it is locked. "
                "No changes were made."
            )

    @property
    def dt_cont_multiplier_locked(self) -> bool:
        """Whether the delta-T contribution multiplier is locked against changes."""
        return self._dt_cont_multiplier_locked

    @dt_cont_multiplier_locked.setter
    def dt_cont_multiplier_locked(self, value: bool):
        """Lock or unlock the delta-T contribution multiplier."""
        self._dt_cont_multiplier_locked = bool(value)

    @property
    def heat_flow(self) -> Value:
        """Stream heat flow view over a scalar or multiperiod duty value."""
        return self._heat_flow

    @heat_flow.setter
    def heat_flow(self, value):
        self.set_value_attr("_heat_flow", value)

    @property
    def htc(self) -> float:
        """Heat transfer coefficient (e.g., kW/m^2/K)."""
        return self._htc

    @htc.setter
    def htc(self, value):
        self.set_value_attr("_htc", value)

    @property
    def htr(self) -> Optional[Value]:
        """Heat transfer resistance (e.g., m^2.K/kW)."""
        return self._copy_value(self._htr)

    @property
    def price(self) -> Value:
        """Unit energy price (e.g., $/MWh)."""
        return self._copy_value(self._price)

    @price.setter
    def price(self, value):
        self.set_value_attr("_price", value)

    @property
    def ut_cost(self) -> Optional[Value]:
        """Utility cost (e.g., $/y)."""
        return self._copy_value(self._cost)

    @property
    def CP(self) -> Optional[Value]:
        """Heat capacity flowrate (e.g., kW/K)."""
        return self._copy_value(self._cp)

    @property
    def rCP(self) -> Optional[Value]:
        """Resistance-capacity product (1/heat transfer rate)."""
        return self._copy_value(self._rcp_prod)

    @property
    def active(self) -> bool:
        """Whether the stream is active in analysis."""
        return self._active

    @active.setter
    def active(self, value: bool):
        """Activate or deactivate the stream for downstream analysis."""
        self._active = bool(value)
        self._bump_numeric_revision()

    # === Computed Temperature Properties ===

    @property
    def t_min(self) -> Optional[Value]:
        """Minimum temperature (supply or target depending on hot/cold)."""
        return self._copy_value(self._t_min)

    @property
    def t_max(self) -> Optional[Value]:
        """Maximum temperature (supply or target depending on hot/cold)."""
        return self._copy_value(self._t_max)

    @property
    def t_min_star(self) -> Optional[Value]:
        """Shifted minimum temperature."""
        return self._copy_value(self._t_min_star)

    @property
    def t_max_star(self) -> Optional[Value]:
        """Shifted maximum temperature."""
        return self._copy_value(self._t_max_star)

    @property
    def t_entr_mean(self) -> Optional[Value]:
        """Entropic mean temperature of supply and target temperatures."""
        return self._copy_value(self._t_entr_mean)

    # === Readable Alias Properties ===

    @property
    def stream_type(self) -> Optional[str]:
        """Alias for the stream thermal type."""
        return self.type

    @property
    def is_active(self) -> bool:
        """Alias for whether the stream participates in analysis."""
        return self.active

    @is_active.setter
    def is_active(self, value: bool):
        """Activate or deactivate the stream via a descriptive alias."""
        self.active = value

    @property
    def supply_temperature(self) -> Optional[Value]:
        """Alias for the supply temperature."""
        return self.t_supply

    @property
    def target_temperature(self) -> Optional[Value]:
        """Alias for the target temperature."""
        return self.t_target

    @property
    def minimum_temperature(self) -> Optional[Value]:
        """Alias for the minimum stream temperature."""
        return self.t_min

    @property
    def maximum_temperature(self) -> Optional[Value]:
        """Alias for the maximum stream temperature."""
        return self.t_max

    @property
    def shifted_minimum_temperature(self) -> Optional[Value]:
        """Alias for the shifted minimum stream temperature."""
        return self.t_min_star

    @property
    def entropic_mean_temperature(self) -> Optional[Value]:
        """Alias for the entropic mean temperature."""
        return self.t_entr_mean

    @property
    def shifted_maximum_temperature(self) -> Optional[Value]:
        """Alias for the shifted maximum stream temperature."""
        return self.t_max_star

    @property
    def supply_pressure(self) -> Optional[Value]:
        """Alias for the supply pressure."""
        return self.p_supply

    @property
    def target_pressure(self) -> Optional[Value]:
        """Alias for the target pressure."""
        return self.p_target

    @property
    def supply_enthalpy(self) -> Optional[Value]:
        """Alias for the supply enthalpy."""
        return self.h_supply

    @property
    def target_enthalpy(self) -> Optional[Value]:
        """Alias for the target enthalpy."""
        return self.h_target

    @property
    def delta_t_contribution(self) -> Value:
        """Alias for the base shifted-temperature contribution."""
        return self.dt_cont

    @property
    def delta_t_contribution_multiplier(self) -> float:
        """Alias for the shifted-temperature contribution multiplier."""
        return self.dt_cont_multiplier

    @property
    def effective_delta_t_contribution(self) -> Value:
        """Alias for the effective shifted-temperature contribution."""
        return self.dt_cont_act

    @property
    def heat_transfer_coefficient(self) -> Value:
        """Alias for the heat transfer coefficient."""
        return self.htc

    @property
    def heat_transfer_resistance(self) -> Optional[Value]:
        """Alias for the heat-transfer resistance."""
        return self.htr

    @property
    def utility_cost(self) -> Optional[Value]:
        """Alias for the derived utility cost."""
        return self.ut_cost

    @property
    def heat_capacity_flow_rate(self) -> Optional[Value]:
        """Alias for the derived stream heat-capacity flow rate."""
        return self.CP

    @property
    def resistance_capacity_product(self) -> Optional[Value]:
        """Alias for the stream resistance-capacity product."""
        return self.rCP

    # === Methods ===

    def set_value_attr(
        self,
        attr_name: str,
        value: float | Value | np.ndarray | Mapping | None,
        update_derived: bool = True,
    ) -> None:
        internal_name = self._resolve_attr_name(attr_name)
        if value is None:
            setattr(self, internal_name, None)
            self._bump_numeric_revision()
            if update_derived:
                self.update_derived_properties()
            return

        parsed = self._coerce_to_value(value, internal_name)
        if parsed is None:
            setattr(self, internal_name, None)
            self._bump_numeric_revision()
            if update_derived:
                self.update_derived_properties()
            return

        if parsed.num_periods == 1 and self._num_periods not in (None, 0, 1):
            parsed = Value(
                np.full(int(self._num_periods), float(parsed.value), dtype=float),
                unit=parsed.unit,
            )
        if self._weights is None or (
            len(self._weights) == 1 and len(self._weights) != parsed.num_periods
        ):
            self._num_periods = parsed.num_periods
            self._period_ids = {str(i): i for i in range(self._num_periods)}
            self._weights = np.ones(self._num_periods, dtype=float)

        if len(self._weights) > 1 and len(self._weights) != parsed.num_periods:
            raise ValueError("Weights length must match the number of periods.")

        setattr(self, internal_name, parsed.to(self._VALUE_UNITS[internal_name]))
        self._bump_numeric_revision()
        self._validate_num_periods()
        if update_derived:
            self.update_derived_properties()

    @classmethod
    def _normalise_fluid_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        validate_coolprop_fluid_name(text)
        return text

    @classmethod
    def _normalise_fluid_phase(cls, value: Optional[str | FluidPhase]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        try:
            return FluidPhase.from_code_or_description(value).name
        except ValueError as exc:
            valid = ", ".join(phase.name for phase in FluidPhase)
            raise ValueError(f"fluid_phase must be one of: {valid}.") from exc

    def set_value_attr_at_idx(
        self,
        attr_name: str,
        value: float | Value | np.ndarray = None,
        idx: int = 0,
        update_derived: bool = True,
    ):
        internal_name = self._resolve_attr_name(attr_name)
        if internal_name not in self._CORE_VALUE_ATTRS:
            raise ValueError(
                f"Attribute '{attr_name}' is not a mutable state property of Stream."
            )

        current = getattr(self, internal_name)
        if current is None:
            current = Value(0.0, unit=self._VALUE_UNITS[internal_name])
            setattr(self, internal_name, current)

        target_size = self._period_vector_size()
        if current.num_periods == 1 and target_size > 1:
            current = Value(
                np.full(target_size, float(current.value), dtype=float),
                unit=current.unit,
            )
            setattr(self, internal_name, current)

        current[idx if current.num_periods > 1 else 0] = value
        self._bump_numeric_revision()
        self._validate_num_periods()
        if update_derived:
            self.update_derived_properties()

    def _coerce_to_value(
        self, value, attr_name: str
    ) -> tuple[Value | None, dict[str, int] | None, np.ndarray | None]:
        if value is None:
            return None, None, None

        raw_value = value
        if hasattr(raw_value, "model_dump") and not isinstance(raw_value, Mapping):
            raw_value = raw_value.model_dump(mode="python")

        if isinstance(raw_value, Mapping) and self._is_period_value_payload(raw_value):
            raw_value = {
                "values": raw_value.get("values"),
                "unit": raw_value.get("unit"),
            }

        parsed = raw_value if isinstance(raw_value, Value) else Value(raw_value)
        target_unit = self._VALUE_UNITS[attr_name]
        if target_unit is None or parsed.unit == target_unit:
            return parsed
        if parsed.unit in {"", "-"}:
            return Value(parsed.value, unit=target_unit)
        return parsed.to(target_unit)

    def _calculate_missing_properties(self) -> None:
        """Calculate any missing core properties from available data."""
        if self._t_supply is None:
            if self._t_target is None:
                self._t_supply = Value(15, "degC").to(self._VALUE_UNITS["_t_supply"])
            else:
                self._t_supply = Value(self._t_target).to(
                    self._VALUE_UNITS["_t_supply"]
                )
        if self._t_target is None:
            self._t_target = Value(self._t_supply).to(self._VALUE_UNITS["_t_target"])
        if self._dt_cont is None:
            self._dt_cont = Value(0.0, unit=self._VALUE_UNITS["_dt_cont"])
        if self._heat_flow is None:
            self._heat_flow = Value(0.0, unit=self._VALUE_UNITS["_heat_flow"])
        if self._htc is None:
            self._htc = Value(1.0, unit=self._VALUE_UNITS["_htc"])
        if self._price is None:
            self._price = Value(0.0, unit=self._VALUE_UNITS["_price"])

        state_size = self._period_vector_size()
        if self._heat_flow is None:
            zeros = np.zeros(state_size, dtype=float)
            self._heat_flow = Value(
                zeros if state_size > 1 else float(zeros[0]),
                unit=self._VALUE_UNITS["_heat_flow"],
            )

        t_supply_arr = self._value_array(self._t_supply, size=state_size)
        t_target_arr = self._value_array(self._t_target, size=state_size)
        heat_flow_arr = self._value_array(self._heat_flow, size=state_size)

        equal_mask = np.isclose(
            t_supply_arr,
            t_target_arr,
            atol=_TEMPERATURE_EQUAL_TOL,
            rtol=0.0,
        )
        if np.any(equal_mask):
            adjusted_target_arr = t_target_arr.copy()
            cold_mask = equal_mask & (heat_flow_arr > 0.0)
            hot_mask = equal_mask & (heat_flow_arr < 0.0)
            adjusted_target_arr[cold_mask] = t_supply_arr[cold_mask] + 0.01
            adjusted_target_arr[hot_mask] = t_supply_arr[hot_mask] - 0.01
            self._t_target = self._build_value(
                adjusted_target_arr,
                unit=self._t_supply.unit,
            ).to(self._VALUE_UNITS["_t_target"])

    def update_derived_properties(self) -> None:
        state_size = self._period_vector_size()
        t_supply = self._value_array(self._t_supply, size=state_size)
        t_target = self._value_array(self._t_target, size=state_size)
        heat_flow = self._value_array(self._heat_flow, size=state_size)
        htc = self._value_array(self._htc, size=state_size)
        price = self._value_array(self._price, size=state_size)

        dt_cont = self._value_array(self._dt_cont, size=state_size)
        dt_cont_act = dt_cont * float(self._dt_cont_multiplier)
        self._dt_cont_act = self._build_value(
            dt_cont_act,
            unit=self._VALUE_UNITS["_dt_cont_act"],
        )

        hot_states = t_supply > t_target + _TEMPERATURE_EQUAL_TOL
        cold_states = t_supply < t_target - _TEMPERATURE_EQUAL_TOL

        if np.any(hot_states):
            self._type = ST.Hot.value
            t_min = t_target
            t_max = t_supply
            t_min_star = t_min - dt_cont_act
            t_max_star = t_max - dt_cont_act
        elif np.any(cold_states):
            self._type = ST.Cold.value
            t_min = t_supply
            t_max = t_target
            t_min_star = t_min + dt_cont_act
            t_max_star = t_max + dt_cont_act
        else:
            self._type = ST.Neutral.value
            t_min = t_supply
            t_max = t_target
            t_min_star = t_min.copy()
            t_max_star = t_max.copy()

        delta_t = t_max - t_min
        cp = np.zeros_like(delta_t, dtype=float)
        valid_dt = np.abs(delta_t) > _TEMPERATURE_EQUAL_TOL
        cp[valid_dt] = heat_flow[valid_dt] / delta_t[valid_dt]

        htr = np.zeros_like(htc, dtype=float)
        valid_htc = htc > 0.0
        htr[valid_htc] = 1.0 / htc[valid_htc]

        rcp_prod = np.zeros_like(cp, dtype=float)
        rcp_prod[valid_htc] = cp[valid_htc] * htr[valid_htc]

        cost = self._build_value(
            price * heat_flow / 1000.0,
            unit=self._VALUE_UNITS["_cost"],
        )

        self._t_min = self._build_value(t_min, unit=self._VALUE_UNITS["_t_min"])
        self._t_max = self._build_value(t_max, unit=self._VALUE_UNITS["_t_max"])
        self._t_min_star = self._build_value(
            t_min_star,
            unit=self._VALUE_UNITS["_t_min_star"],
        )
        self._t_max_star = self._build_value(
            t_max_star,
            unit=self._VALUE_UNITS["_t_max_star"],
        )
        t_supply_k = t_supply + 273.15
        t_target_k = t_target + 273.15
        with np.errstate(divide="ignore", invalid="ignore"):
            t_entr_mean = (
                (t_supply_k - t_target_k) / (np.log(t_supply_k) - np.log(t_target_k))
            ) - 273.15
        self._t_entr_mean = self._build_value(
            t_entr_mean,
            unit=self._VALUE_UNITS["_t_supply"],
        )
        self._cp = self._build_value(cp, unit=self._VALUE_UNITS["_cp"])
        self._htr = self._build_value(htr, unit=self._VALUE_UNITS["_htr"])
        self._rcp_prod = self._build_value(
            rcp_prod,
            unit=self._VALUE_UNITS["_rcp_prod"],
        )
        self._cost = cost
        self._bump_numeric_revision()

    def _validate_num_periods(self):
        counts = np.asarray(
            [
                getattr(self, attr).num_periods
                for attr in self._CORE_VALUE_ATTRS
                if isinstance(getattr(self, attr), Value)
            ],
            dtype=int,
        )
        period_value_counts = counts[counts > 1]
        if period_value_counts.size == 0:
            self._num_periods = 1
            return
        if int(period_value_counts.max()) != int(period_value_counts.min()):
            raise ValueError(
                f"Stream inputs for {self._name} have unequal period counts."
            )
        self._num_periods = int(period_value_counts.max())

    def invert(self) -> None:
        """Flip a utility stream into its generating process-stream analogue."""
        if self._is_process_stream:
            raise ValueError(
                "Logic error: Process streams cannot be inverted; only utility "
                "streams may be inverted for generation."
            )
        self._t_supply, self._t_target = self._t_target, self._t_supply
        self._p_supply, self._p_target = self._p_target, self._p_supply
        self._h_supply, self._h_target = self._h_target, self._h_supply
        self._is_process_stream = True
        self._bump_numeric_revision()
        self.update_derived_properties()

    def get_period_index(self, period_id: str | None = None) -> int:
        if self._period_ids is None or period_id is None:
            return 0
        resolved_period_id = str(period_id)
        if resolved_period_id not in self._period_ids:
            raise ValueError(
                f"Unknown period_id {resolved_period_id!r}. "
                f"Available periods: {', '.join(self._period_ids.keys())}."
            )
        return int(self._period_ids[resolved_period_id])

    def resolve_attr(self, attr_name: str, period_id: str | None = None):
        value = getattr(self, self._resolve_attr_name(attr_name))
        if isinstance(value, Value):
            return float(value[self.get_period_index(period_id)].value)
        return value

    def set_attr_for_period(
        self,
        attr_name: str,
        value,
        *,
        period_id: str | None = None,
    ) -> None:
        self.set_value_attr_at_idx(
            attr_name,
            value,
            idx=self.get_period_index(period_id),
        )

    def _get_period_context(self) -> tuple[dict[str, int] | None, np.ndarray | None]:
        return self._period_ids, self._weights

    def set_period_context(
        self,
        period_ids: dict[str, int] | list[str] | tuple[str, ...] | None,
        weights: np.ndarray | list[float] | tuple[float, ...] | None,
        num_periods: int | None,
    ) -> None:
        self._period_ids = period_ids
        self._weights = weights
        self._num_periods = num_periods
        if len(self._period_ids) != len(self._weights):
            raise ValueError(
                "Length of period_ids and weights must match eachother in length."
            )
        self._bump_numeric_revision()

    def _bump_numeric_revision(self) -> None:
        self._numeric_revision = getattr(self, "_numeric_revision", 0) + 1

    def _period_vector_size(self) -> int:
        counts = [
            value.num_periods
            for attr in self._CORE_VALUE_ATTRS
            if isinstance((value := getattr(self, attr)), Value)
        ]
        return max(counts) if counts else 1

    def _value_array(self, value: Value | None, *, size: int) -> np.ndarray:
        if value is None:
            return np.zeros(size, dtype=float)
        if value.num_periods == 1:
            return np.full(size, float(value.value), dtype=float)
        arr = value.period_values.astype(float)
        if arr.size != size:
            raise ValueError(
                f"Period count for stream '{self._name}' is inconsistent "
                f"with its values."
            )
        return arr

    def _build_value(self, magnitudes, *, unit: str) -> Value:
        arr = np.asarray(magnitudes, dtype=float).reshape(-1)
        if arr.size == 1:
            return Value(float(arr[0]), unit=unit)
        return Value(arr, unit=unit)

    def _copy_value(self, value: Value | None) -> Value | None:
        if value is None:
            return None
        return Value(value)

    def _resolve_attr_name(self, attr_name: str) -> str:
        if attr_name in self._MUTABLE_VALUE_ATTRS:
            return attr_name if attr_name.startswith("_") else f"_{attr_name}"
        if hasattr(self, attr_name):
            return attr_name
        if hasattr(self, f"_{attr_name}"):
            return f"_{attr_name}"
        raise AttributeError(f"Stream has no attribute {attr_name!r}.")

    @staticmethod
    def _is_period_value_payload(value: Mapping) -> bool:
        keys = set(value)
        return keys.issubset({"values", "period_ids", "weights", "unit"}) and (
            "values" in keys or "period_ids" in keys or "weights" in keys
        )

    @staticmethod
    def _normalise_period_ids(
        period_ids: dict[str, int] | list[str] | tuple[str, ...] | None,
    ) -> dict[str, int] | None:
        if period_ids is None:
            return None
        if isinstance(period_ids, dict):
            return {str(key): int(val) for key, val in period_ids.items()}
        return {str(period_id): idx for idx, period_id in enumerate(period_ids)}

    @staticmethod
    def _normalise_weights(
        weights,
        *,
        expected_len: int,
    ) -> np.ndarray | None:
        if weights is None:
            return None
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.size != expected_len:
            raise ValueError("weights length must match the number of periods.")
        total = float(arr.sum())
        if total > 0.0:
            arr = arr / total
        return arr
