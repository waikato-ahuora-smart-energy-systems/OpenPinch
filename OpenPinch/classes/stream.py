"""Data model representing process and utility streams."""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

from ..lib.enums import ST
from ._stream_value_view import StreamValueView
from .value import Value

_STATE_WEIGHT_RTOL = 1e-12
_STATE_WEIGHT_ATOL = 1e-12
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
        "_P_supply": "kPa",
        "_P_target": "kPa",
        "_h_supply": "kJ/kg",
        "_h_target": "kJ/kg",
        "_dt_cont": "delta_degC",
        "_dt_cont_act": "delta_degC",
        "_heat_flow": "kW",
        "_htc": "kW/m^2/K",
        "_htr": "m^2*K/kW",
        "_price": "USD/MWh",
        "_ut_cost": None,
        "_CP": "kW/K",
        "_RCP_prod": "m^2",
        "_t_min": "degC",
        "_t_max": "degC",
        "_t_min_star": "degC",
        "_t_max_star": "degC",
    }
    _CORE_STATE_FIELDS = (
        "_t_supply",
        "_t_target",
        "_P_supply",
        "_P_target",
        "_h_supply",
        "_h_target",
        "_dt_cont",
        "_dt_cont_act",
        "_heat_flow",
        "_htc",
        "_price",
    )

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
        self._dt_cont_multiplier: float = 1.0
        self._dt_cont_multiplier_locked: bool = False
        self._is_process_stream: bool = is_process_stream
        self._active = True

        self._t_supply: Value | None = None
        self._t_target: Value | None = None
        self._P_supply: Value | None = None
        self._P_target: Value | None = None
        self._h_supply: Value | None = None
        self._h_target: Value | None = None
        self._dt_cont: Value | None = None
        self._dt_cont_act: Value | None = None
        self._heat_flow: Value | None = None
        self._htc: Value | None = None
        self._htr: Value | None = None
        self._price: Value | None = None
        self._ut_cost: Value | None = None
        self._CP: Value | None = None
        self._RCP_prod: Value | None = None
        self._t_min: Value | None = None
        self._t_max: Value | None = None
        self._t_min_star: Value | None = None
        self._t_max_star: Value | None = None

        self._set_value_attribute("_t_supply", t_supply, update=False)
        self._set_value_attribute("_t_target", t_target, update=False)
        self._set_value_attribute("_P_supply", P_supply, update=False)
        self._set_value_attribute("_P_target", P_target, update=False)
        self._set_value_attribute("_h_supply", h_supply, update=False)
        self._set_value_attribute("_h_target", h_target, update=False)
        self._set_value_attribute("_dt_cont", dt_cont, update=False)
        self._dt_cont_act = (
            self._scale_value(self._dt_cont, self._dt_cont_multiplier)
            if dt_cont_act is None
            else self._coerce_to_value(dt_cont_act, "_dt_cont_act")
        )
        self._set_value_attribute("_heat_flow", heat_flow, update=False)
        self._set_value_attribute("_htc", htc, update=False)
        self._htr = self._inverse_value(self._htc, "_htr")
        self._set_value_attribute("_price", price, update=False)
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
    def t_supply(self) -> Optional[StreamValueView]:
        """Supply temperature (e.g., degC)."""
        return self._to_view(self._t_supply)

    @t_supply.setter
    def t_supply(self, value: float):
        """Set the supply temperature and refresh derived stream attributes."""
        self._set_value_attribute("_t_supply", value)

    @property
    def t_target(self) -> Optional[StreamValueView]:
        """Target temperature (e.g., degC)."""
        return self._to_view(self._t_target)

    @t_target.setter
    def t_target(self, value: float):
        """Set the target temperature and refresh derived stream attributes."""
        self._set_value_attribute("_t_target", value)

    @property
    def P_supply(self) -> Optional[StreamValueView]:
        """Supply pressure (e.g., kPa)."""
        return self._to_view(self._P_supply)

    @P_supply.setter
    def P_supply(self, value: float):
        """Set the supply pressure and refresh derived stream attributes."""
        self._set_value_attribute("_P_supply", value)

    @property
    def P_target(self) -> Optional[StreamValueView]:
        """Target pressure (e.g., kPa)."""
        return self._to_view(self._P_target)

    @P_target.setter
    def P_target(self, value: float):
        """Set the target pressure and refresh derived stream attributes."""
        self._set_value_attribute("_P_target", value)

    @property
    def h_supply(self) -> Optional[StreamValueView]:
        """Supply enthalpy (e.g., kJ/kg)."""
        return self._to_view(self._h_supply)

    @h_supply.setter
    def h_supply(self, value: float):
        """Set the supply enthalpy and refresh derived stream attributes."""
        self._set_value_attribute("_h_supply", value)

    @property
    def h_target(self) -> Optional[StreamValueView]:
        """Target enthalpy (e.g., kJ/kg)."""
        return self._to_view(self._h_target)

    @h_target.setter
    def h_target(self, value: float):
        """Set the target enthalpy and refresh derived stream attributes."""
        self._set_value_attribute("_h_target", value)

    @property
    def dt_cont(self) -> StreamValueView:
        """Preserved base delta-T contribution before any zone multiplier."""
        return self._to_view(self._dt_cont)

    @dt_cont.setter
    def dt_cont(self, value: float):
        """Set the base contribution to shifted-temperature calculations."""
        self._dt_cont = self._coerce_to_value(value, "_dt_cont")
        self._dt_cont_act = self._scale_value(self._dt_cont, self._dt_cont_multiplier)
        self._update_attributes()

    @property
    def dt_cont_act(self) -> StreamValueView:
        """Effective delta-T contribution used in shifted-temperature calculations."""
        return self._to_view(self._dt_cont_act)

    @dt_cont_act.setter
    def dt_cont_act(self, value: float):
        """Set the effective shifted-temperature contribution in active use."""
        self._dt_cont_act = self._coerce_to_value(value, "_dt_cont_act")
        self._update_attributes()

    @property
    def dt_cont_multiplier(self) -> float:
        """Effective delta-T contribution used in shifted-temperature calculations."""
        return self._dt_cont_multiplier

    @dt_cont_multiplier.setter
    def dt_cont_multiplier(self, value: float):
        """Set the effective shifted-temperature contribution in active use."""
        if not self._dt_cont_multiplier_locked:
            self._dt_cont_multiplier = value
            self._dt_cont_act = self._scale_value(self._dt_cont, value)
            self._update_attributes()
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
        self._dt_cont_multiplier_locked = value

    @property
    def heat_flow(self) -> float:
        """Stream heat flow (e.g., kW)."""
        return self._to_view(self._heat_flow)

    @heat_flow.setter
    def heat_flow(self, value: float):
        """Set the stream duty and refresh derived heat-capacity quantities."""
        self._set_value_attribute("_heat_flow", value)

    @property
    def htc(self) -> float:
        """Heat transfer coefficient (e.g., kW/m^2/K)."""
        return self._to_view(self._htc)

    @htc.setter
    def htc(self, value: float):
        """Set the heat-transfer coefficient and refresh derived resistance."""
        self._set_value_attribute("_htc", value)

    @property
    def htr(self) -> float:
        """Heat transfer resistance (e.g., m^2.K/kW)."""
        return self._to_view(self._htr)

    @htr.setter
    def htr(self, value: float):
        """Set the stored heat-transfer resistance explicitly."""
        self._htr = self._coerce_to_value(value, "_htr")

    @property
    def price(self) -> float:
        """Unit energy price (e.g., $/MWh)."""
        return self._to_view(self._price)

    @price.setter
    def price(self, value: float):
        """Set the unit energy price used in utility-cost calculations."""
        self._price = self._coerce_to_value(value, "_price")
        self._calc_utility_cost()

    @property
    def ut_cost(self) -> float:
        """Utility cost (e.g., $/y)."""
        return self._to_view(self._ut_cost)

    @ut_cost.setter
    def ut_cost(self, value: float):
        """Set the cached utility-cost figure for the stream."""
        self._ut_cost = self._coerce_to_value(value, "_ut_cost")

    @property
    def CP(self) -> float:
        """Heat capacity flowrate (e.g., kW/K)."""
        return self._to_view(self._CP)

    @CP.setter
    def CP(self, value: float):
        """Set the cached heat-capacity flow rate."""
        self._CP = self._coerce_to_value(value, "_CP")

    @property
    def rCP(self) -> Optional[float]:
        """Resistance-capacity product (1/heat transfer rate)."""
        return self._to_view(self._RCP_prod)

    @rCP.setter
    def rCP(self, value: float):
        """Set the cached resistance-capacity product."""
        self._RCP_prod = self._coerce_to_value(value, "_RCP_prod")

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
    def t_min(self) -> Optional[StreamValueView]:
        """Minimum temperature (supply or target depending on hot/cold)."""
        return self._to_view(self._t_min)

    @t_min.setter
    def t_min(self, value: float):
        """Set the unshifted lower temperature bound."""
        self._t_min = self._coerce_to_value(value, "_t_min")

    @property
    def t_max(self) -> Optional[StreamValueView]:
        """Maximum temperature (supply or target depending on hot/cold)."""
        return self._to_view(self._t_max)

    @t_max.setter
    def t_max(self, value: float):
        """Set the unshifted upper temperature bound."""
        self._t_max = self._coerce_to_value(value, "_t_max")

    @property
    def t_min_star(self) -> Optional[StreamValueView]:
        """Shifted minimum temperature."""
        return self._to_view(self._t_min_star)

    @t_min_star.setter
    def t_min_star(self, value: float):
        """Set the shifted lower temperature bound."""
        self._t_min_star = self._coerce_to_value(value, "_t_min_star")

    @property
    def t_max_star(self) -> Optional[StreamValueView]:
        """Shifted maximum temperature."""
        return self._to_view(self._t_max_star)

    @t_max_star.setter
    def t_max_star(self, value: float):
        """Set the shifted upper temperature bound."""
        self._t_max_star = self._coerce_to_value(value, "_t_max_star")

    # === Readable Alias Properties ===

    @property
    def stream_type(self) -> Optional[str]:
        """Alias for the stream thermal type."""
        return self.type

    @stream_type.setter
    def stream_type(self, value: str):
        """Set the stream thermal type via a descriptive alias."""
        self.type = value

    @property
    def process_stream(self) -> bool:
        """Alias for whether the stream is process-side."""
        return self.is_process_stream

    @process_stream.setter
    def process_stream(self, value: bool):
        """Set whether the stream is process-side via a descriptive alias."""
        self.is_process_stream = value

    @property
    def is_active(self) -> bool:
        """Alias for whether the stream participates in analysis."""
        return self.active

    @is_active.setter
    def is_active(self, value: bool):
        """Activate or deactivate the stream via a descriptive alias."""
        self.active = value

    @property
    def supply_temperature(self) -> Optional[StreamValueView]:
        """Alias for the supply temperature."""
        return self.t_supply

    @supply_temperature.setter
    def supply_temperature(self, value: float):
        """Set the supply temperature via a descriptive alias."""
        self.t_supply = value

    @property
    def target_temperature(self) -> Optional[StreamValueView]:
        """Alias for the target temperature."""
        return self.t_target

    @target_temperature.setter
    def target_temperature(self, value: float):
        """Set the target temperature via a descriptive alias."""
        self.t_target = value

    @property
    def minimum_temperature(self) -> Optional[StreamValueView]:
        """Alias for the minimum stream temperature."""
        return self.t_min

    @minimum_temperature.setter
    def minimum_temperature(self, value: float):
        """Set the minimum stream temperature via a descriptive alias."""
        self.t_min = value

    @property
    def maximum_temperature(self) -> Optional[StreamValueView]:
        """Alias for the maximum stream temperature."""
        return self.t_max

    @maximum_temperature.setter
    def maximum_temperature(self, value: float):
        """Set the maximum stream temperature via a descriptive alias."""
        self.t_max = value

    @property
    def shifted_minimum_temperature(self) -> Optional[StreamValueView]:
        """Alias for the shifted minimum stream temperature."""
        return self.t_min_star

    @shifted_minimum_temperature.setter
    def shifted_minimum_temperature(self, value: float):
        """Set the shifted minimum stream temperature via a descriptive alias."""
        self.t_min_star = value

    @property
    def shifted_maximum_temperature(self) -> Optional[StreamValueView]:
        """Alias for the shifted maximum stream temperature."""
        return self.t_max_star

    @shifted_maximum_temperature.setter
    def shifted_maximum_temperature(self, value: float):
        """Set the shifted maximum stream temperature via a descriptive alias."""
        self.t_max_star = value

    @property
    def supply_pressure(self) -> Optional[StreamValueView]:
        """Alias for the supply pressure."""
        return self.P_supply

    @supply_pressure.setter
    def supply_pressure(self, value: float):
        """Set the supply pressure via a descriptive alias."""
        self.P_supply = value

    @property
    def target_pressure(self) -> Optional[StreamValueView]:
        """Alias for the target pressure."""
        return self.P_target

    @target_pressure.setter
    def target_pressure(self, value: float):
        """Set the target pressure via a descriptive alias."""
        self.P_target = value

    @property
    def supply_enthalpy(self) -> Optional[StreamValueView]:
        """Alias for the supply enthalpy."""
        return self.h_supply

    @supply_enthalpy.setter
    def supply_enthalpy(self, value: float):
        """Set the supply enthalpy via a descriptive alias."""
        self.h_supply = value

    @property
    def target_enthalpy(self) -> Optional[StreamValueView]:
        """Alias for the target enthalpy."""
        return self.h_target

    @target_enthalpy.setter
    def target_enthalpy(self, value: float):
        """Set the target enthalpy via a descriptive alias."""
        self.h_target = value

    @property
    def delta_t_contribution(self) -> StreamValueView:
        """Alias for the base shifted-temperature contribution."""
        return self.dt_cont

    @delta_t_contribution.setter
    def delta_t_contribution(self, value: float):
        """Set the base shifted-temperature contribution via an alias."""
        self.dt_cont = value

    @property
    def effective_delta_t_contribution(self) -> StreamValueView:
        """Alias for the effective shifted-temperature contribution."""
        return self.dt_cont_act

    @effective_delta_t_contribution.setter
    def effective_delta_t_contribution(self, value: float):
        """Set the effective shifted-temperature contribution via an alias."""
        self.dt_cont_act = value

    @property
    def heat_duty(self) -> float:
        """Alias for the stream heat flow."""
        return self.heat_flow

    @heat_duty.setter
    def heat_duty(self, value: float):
        """Set the stream heat flow via a descriptive alias."""
        self.heat_flow = value

    @property
    def heat_transfer_coefficient(self) -> float:
        """Alias for the heat-transfer coefficient."""
        return self.htc

    @heat_transfer_coefficient.setter
    def heat_transfer_coefficient(self, value: float):
        """Set the heat-transfer coefficient via a descriptive alias."""
        self.htc = value

    @property
    def heat_transfer_resistance(self) -> float:
        """Alias for the heat-transfer resistance."""
        return self.htr

    @heat_transfer_resistance.setter
    def heat_transfer_resistance(self, value: float):
        """Set the heat-transfer resistance via a descriptive alias."""
        self.htr = value

    @property
    def utility_cost(self) -> float:
        """Alias for the cached utility cost."""
        return self.ut_cost

    @utility_cost.setter
    def utility_cost(self, value: float):
        """Set the cached utility cost via a descriptive alias."""
        self.ut_cost = value

    @property
    def heat_capacity_flow_rate(self) -> float:
        """Alias for the stream heat-capacity flow rate."""
        return self.CP

    @heat_capacity_flow_rate.setter
    def heat_capacity_flow_rate(self, value: float):
        """Set the heat-capacity flow rate via a descriptive alias."""
        self.CP = value

    @property
    def resistance_capacity_product(self) -> Optional[float]:
        """Alias for the stream resistance-capacity product."""
        return self.rCP

    @resistance_capacity_product.setter
    def resistance_capacity_product(self, value: float):
        """Set the resistance-capacity product via a descriptive alias."""
        self.rCP = value

    # === Methods ===

    def _set_value_attribute(self, attr_name: str, value, update: bool = True) -> None:
        setattr(self, attr_name, self._coerce_to_value(value, attr_name))
        if update:
            self._update_attributes()

    def _coerce_to_value(self, value, attr_name: str) -> Value | None:
        if value is None:
            return None
        if isinstance(value, StreamValueView):
            return value.raw_value
        parsed = Value(value)
        target_unit = self._VALUE_UNITS[attr_name]
        if target_unit is None:
            return parsed
        return Value(parsed, unit=target_unit)

    @staticmethod
    def _to_view(value):
        if isinstance(value, Value):
            return StreamValueView(value)
        return value

    def _normalise_existing_value(self, value, attr_name: str) -> Value | None:
        if value is None:
            return None
        if isinstance(value, StreamValueView):
            return value.raw_value
        if isinstance(value, Value):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool
        ):
            return Value(value, unit=self._VALUE_UNITS[attr_name])
        return None

    def _resolve_state_context(self, values_by_name: dict[str, Any]):
        stateful_values: list[tuple[str, Value]] = []
        for name, value in values_by_name.items():
            value_obj = self._normalise_existing_value(value, name)
            if value_obj is not None and value_obj.state_ids is not None:
                stateful_values.append((name, value_obj))

        if not stateful_values:
            return None, None

        ref_name, ref_value = stateful_values[0]
        ref_state_ids = ref_value.state_ids
        ref_weights = ref_value.weights

        for name, value_obj in stateful_values[1:]:
            if value_obj.state_ids != ref_state_ids:
                raise ValueError(
                    f"state_ids for {name[1:]} must align with {ref_name[1:]}."
                )
            if not np.allclose(
                value_obj.weights,
                ref_weights,
                rtol=_STATE_WEIGHT_RTOL,
                atol=_STATE_WEIGHT_ATOL,
            ):
                raise ValueError(
                    f"weights for {name[1:]} must align with {ref_name[1:]}."
                )

        return ref_state_ids, ref_weights

    def _broadcast_magnitudes(
        self, value, state_ids, attr_name: str
    ) -> np.ndarray | None:
        value_obj = self._normalise_existing_value(value, attr_name)
        if value_obj is None:
            return None
        if state_ids is None:
            return np.asarray([float(value_obj.value)], dtype=float)
        if value_obj.state_ids is None:
            return np.full(len(state_ids), float(value_obj.value), dtype=float)
        return value_obj.state_values.astype(float)

    def _value_from_array(
        self,
        magnitudes,
        *,
        attr_name: str,
        state_ids: list[str] | None,
        weights: np.ndarray | None,
    ) -> Value:
        arr = np.asarray(magnitudes, dtype=float).reshape(-1)
        if state_ids is None:
            return Value(float(arr[0]), unit=self._VALUE_UNITS[attr_name])
        return Value(
            values=arr,
            unit=self._VALUE_UNITS[attr_name],
            state_ids=state_ids,
            weights=weights,
        )

    def _scale_value(self, value: Value | None, factor: float) -> Value | None:
        if value is None:
            return None
        scaled = np.asarray(value.state_values, dtype=float) * float(factor)
        if value.state_ids is None:
            return Value(float(scaled[0]), unit=value.unit)
        return Value(
            values=scaled,
            unit=value.unit,
            state_ids=value.state_ids,
            weights=value.weights,
        )

    def _inverse_value(self, value: Value | None, attr_name: str) -> Value | None:
        if value is None:
            return None
        state_ids, weights = self._resolve_state_context({attr_name: value})
        magnitudes = self._broadcast_magnitudes(value, state_ids, attr_name)
        if magnitudes is None:
            return None
        inv = np.zeros_like(magnitudes, dtype=float)
        mask = np.abs(magnitudes) > _TEMPERATURE_EQUAL_TOL
        inv[mask] = 1.0 / magnitudes[mask]
        return self._value_from_array(
            inv,
            attr_name=attr_name,
            state_ids=state_ids,
            weights=weights,
        )

    def _update_attributes(self) -> None:
        """Calculates key stream attributes based on temperatures."""
        t_supply_value = self._normalise_existing_value(self._t_supply, "_t_supply")
        t_target_value = self._normalise_existing_value(self._t_target, "_t_target")
        htc_value = self._normalise_existing_value(self._htc, "_htc")

        if t_supply_value is None or t_target_value is None or htc_value is None:
            return

        state_ids, weights = self._resolve_state_context(
            {
                "_t_supply": t_supply_value,
                "_t_target": t_target_value,
                "_heat_flow": self._heat_flow,
                "_dt_cont": self._dt_cont,
                "_dt_cont_act": self._dt_cont_act,
                "_htc": htc_value,
                "_price": self._price,
            }
        )

        t_supply_arr = self._broadcast_magnitudes(
            t_supply_value, state_ids, "_t_supply"
        )
        t_target_arr = self._broadcast_magnitudes(
            t_target_value, state_ids, "_t_target"
        )
        heat_flow_arr = self._broadcast_magnitudes(
            self._heat_flow, state_ids, "_heat_flow"
        )
        if heat_flow_arr is None:
            heat_flow_arr = np.zeros_like(t_supply_arr, dtype=float)
        dt_cont_act_arr = self._broadcast_magnitudes(
            self._dt_cont_act, state_ids, "_dt_cont_act"
        )
        if dt_cont_act_arr is None:
            dt_cont_act_arr = np.zeros_like(t_supply_arr, dtype=float)

        equal_mask = np.isclose(
            t_supply_arr, t_target_arr, atol=_TEMPERATURE_EQUAL_TOL, rtol=0.0
        )
        if np.any(equal_mask):
            adjusted_target = t_target_arr.copy()
            cold_mask = equal_mask & (heat_flow_arr > 0.0)
            hot_mask = equal_mask & (heat_flow_arr < 0.0)
            adjusted_target[cold_mask] = t_supply_arr[cold_mask] + 0.01
            adjusted_target[hot_mask] = t_supply_arr[hot_mask] - 0.01
            if np.any(cold_mask | hot_mask):
                self._t_target = self._value_from_array(
                    adjusted_target,
                    attr_name="_t_target",
                    state_ids=state_ids,
                    weights=weights,
                )
                t_target_arr = adjusted_target

        hot_states = t_supply_arr > t_target_arr + _TEMPERATURE_EQUAL_TOL
        cold_states = t_supply_arr < t_target_arr - _TEMPERATURE_EQUAL_TOL
        neutral_states = ~(hot_states | cold_states)
        active_classes = {
            label
            for label, mask in (
                (ST.Hot.value, hot_states),
                (ST.Cold.value, cold_states),
                (ST.Both.value, neutral_states),
            )
            if np.any(mask)
        }

        if len(active_classes) > 1:
            state_labels = state_ids or [str(idx) for idx in range(len(t_supply_arr))]
            hot_state_ids = [
                sid for sid, active in zip(state_labels, hot_states) if active
            ]
            cold_state_ids = [
                sid for sid, active in zip(state_labels, cold_states) if active
            ]
            neutral_state_ids = [
                sid for sid, active in zip(state_labels, neutral_states) if active
            ]
            raise ValueError(
                "Stream states must classify consistently. "
                f"Hot={hot_state_ids}, "
                f"Cold={cold_state_ids}, "
                f"Neutral={neutral_state_ids}."
            )

        if np.any(hot_states):
            t_min_arr = t_target_arr
            t_max_arr = t_supply_arr
            t_min_star_arr = t_min_arr - dt_cont_act_arr
            t_max_star_arr = t_max_arr - dt_cont_act_arr
            if self._type is None:
                self._type = ST.Hot.value
        elif np.any(cold_states):
            t_min_arr = t_supply_arr
            t_max_arr = t_target_arr
            t_min_star_arr = t_min_arr + dt_cont_act_arr
            t_max_star_arr = t_max_arr + dt_cont_act_arr
            if self._type is None:
                self._type = ST.Cold.value
        else:
            t_min_arr = t_supply_arr
            t_max_arr = t_target_arr
            t_min_star_arr = t_min_arr.copy()
            t_max_star_arr = t_max_arr.copy()
            self._type = ST.Both.value

        self._t_min = self._value_from_array(
            t_min_arr, attr_name="_t_min", state_ids=state_ids, weights=weights
        )
        self._t_max = self._value_from_array(
            t_max_arr, attr_name="_t_max", state_ids=state_ids, weights=weights
        )
        self._t_min_star = self._value_from_array(
            t_min_star_arr,
            attr_name="_t_min_star",
            state_ids=state_ids,
            weights=weights,
        )
        self._t_max_star = self._value_from_array(
            t_max_star_arr,
            attr_name="_t_max_star",
            state_ids=state_ids,
            weights=weights,
        )

        dt_arr = t_max_arr - t_min_arr
        heat_flow_value = self._normalise_existing_value(self._heat_flow, "_heat_flow")
        cp_value = self._normalise_existing_value(self._CP, "_CP")

        if heat_flow_value is not None:
            q_arr = self._broadcast_magnitudes(heat_flow_value, state_ids, "_heat_flow")
            cp_arr = np.zeros_like(dt_arr, dtype=float)
            non_zero_dt = np.abs(dt_arr) > _TEMPERATURE_EQUAL_TOL
            cp_arr[non_zero_dt] = q_arr[non_zero_dt] / dt_arr[non_zero_dt]
            self._CP = self._value_from_array(
                cp_arr, attr_name="_CP", state_ids=state_ids, weights=weights
            )
        elif cp_value is not None:
            cp_arr = self._broadcast_magnitudes(cp_value, state_ids, "_CP")
            heat_flow_arr = cp_arr * dt_arr
            self._heat_flow = self._value_from_array(
                heat_flow_arr,
                attr_name="_heat_flow",
                state_ids=state_ids,
                weights=weights,
            )

        self._calc_utility_cost()
        self._calc_htr_and_cp_product()

    def invert(self) -> None:
        """Flip a utility stream into its generating process-stream analogue."""
        if self._is_process_stream:
            raise ValueError(
                "Logic error: Process streams cannot be inverted; only utility "
                "streams may be inverted for generation."
            )

        self._t_supply, self._t_target = self._t_target, self._t_supply
        self._P_supply, self._P_target = self._P_target, self._P_supply
        self._h_supply, self._h_target = self._h_target, self._h_supply

        self._type = ST.Cold.value if self._type == ST.Hot.value else ST.Hot.value
        self._is_process_stream = True
        self._update_attributes()

    def set_heat_flow(self, value: float, unit: str = "kW") -> None:
        """Sets the heat flow and updates CP and utility cost."""
        self._heat_flow = Value(value, unit=unit)
        self._update_attributes()

    def _calc_utility_cost(self):
        heat_flow_value = self._normalise_existing_value(self._heat_flow, "_heat_flow")
        price_value = self._normalise_existing_value(self._price, "_price")
        if heat_flow_value is None or price_value is None:
            return

        state_ids, weights = self._resolve_state_context(
            {"_heat_flow": heat_flow_value, "_price": price_value}
        )
        heat_flow_arr = self._broadcast_magnitudes(
            heat_flow_value, state_ids, "_heat_flow"
        )
        price_arr = self._broadcast_magnitudes(price_value, state_ids, "_price")
        utility_cost_arr = (heat_flow_arr / 1000.0) * price_arr
        self._ut_cost = self._value_from_array(
            utility_cost_arr,
            attr_name="_ut_cost",
            state_ids=state_ids,
            weights=weights,
        )

    def _calc_htr_and_cp_product(self):
        htc_value = self._normalise_existing_value(self._htc, "_htc")
        cp_value = self._normalise_existing_value(self._CP, "_CP")
        if htc_value is None:
            return

        state_ids, weights = self._resolve_state_context(
            {"_htc": htc_value, "_CP": cp_value}
        )
        htc_arr = self._broadcast_magnitudes(htc_value, state_ids, "_htc")
        if htc_arr is None:
            return

        htr_arr = np.zeros_like(htc_arr, dtype=float)
        positive_mask = htc_arr > 0.0
        htr_arr[positive_mask] = 1.0 / htc_arr[positive_mask]
        self._htr = self._value_from_array(
            htr_arr, attr_name="_htr", state_ids=state_ids, weights=weights
        )

        if cp_value is None:
            return

        cp_arr = self._broadcast_magnitudes(cp_value, state_ids, "_CP")
        rcp_arr = np.zeros_like(cp_arr, dtype=float)
        rcp_arr[positive_mask] = cp_arr[positive_mask] * htr_arr[positive_mask]
        self._RCP_prod = self._value_from_array(
            rcp_arr,
            attr_name="_RCP_prod",
            state_ids=state_ids,
            weights=weights,
        )
