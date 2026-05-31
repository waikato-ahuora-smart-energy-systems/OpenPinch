"""Data model representing process and utility streams."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..lib.enums import ST
from ..lib.schemas.common import MaybeVU
from ..utils.miscellaneous import extract_state_context, resolve_value_for_state
from .accessors._stream_value_accessor import StreamValueAccessor
from .value import Value

if TYPE_CHECKING:
    from .stream_collection import StreamCollection

_STATE_WEIGHT_RTOL = 1e-12
_STATE_WEIGHT_ATOL = 1e-12
_TEMPERATURE_EQUAL_TOL = 1e-12


@dataclass(frozen=True)
class _DerivedStreamState:
    state_ids: list[str] | None
    weights: np.ndarray | None
    stream_type: str
    t_min: np.ndarray
    t_max: np.ndarray
    t_min_star: np.ndarray
    t_max_star: np.ndarray
    cp: np.ndarray
    htr: np.ndarray
    ut_cost: np.ndarray
    rcp: np.ndarray


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
        heat_flow: MaybeVU = 0.0,
        htc: float = 1.0,
        price: float = 0.0,
        is_process_stream: bool = True,
    ):
        """Initialise a stream and infer hot/cold classification."""
        self._name: str = name
        self._dt_cont_multiplier: float = 1.0
        self._dt_cont_multiplier_locked: bool = False
        self._is_process_stream: bool = is_process_stream
        self._active = True
        self._state_collection: StreamCollection | None = None
        self._state_ids: list[str] | None = None
        self._weights: np.ndarray | None = None

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
        self._price: Value | None = None

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
        self._set_value_attribute("_price", price, update=False)

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
        derived = self._derive_stream_state()
        return None if derived is None else derived.stream_type

    @property
    def t_supply(self) -> Optional[StreamValueAccessor]:
        """Supply temperature (e.g., degC)."""
        return self._to_accessor(self._t_supply, "t_supply", writable=True)

    @t_supply.setter
    def t_supply(self, value: float):
        """Set the supply temperature and refresh derived stream attributes."""
        self._set_value_attribute("_t_supply", value)

    @property
    def t_target(self) -> Optional[StreamValueAccessor]:
        """Target temperature (e.g., degC)."""
        return self._to_accessor(self._t_target, "t_target", writable=True)

    @t_target.setter
    def t_target(self, value: float):
        """Set the target temperature and refresh derived stream attributes."""
        self._set_value_attribute("_t_target", value)

    @property
    def P_supply(self) -> Optional[StreamValueAccessor]:
        """Supply pressure (e.g., kPa)."""
        return self._to_accessor(self._P_supply, "P_supply", writable=True)

    @P_supply.setter
    def P_supply(self, value: float):
        """Set the supply pressure and refresh derived stream attributes."""
        self._set_value_attribute("_P_supply", value)

    @property
    def P_target(self) -> Optional[StreamValueAccessor]:
        """Target pressure (e.g., kPa)."""
        return self._to_accessor(self._P_target, "P_target", writable=True)

    @P_target.setter
    def P_target(self, value: float):
        """Set the target pressure and refresh derived stream attributes."""
        self._set_value_attribute("_P_target", value)

    @property
    def h_supply(self) -> Optional[StreamValueAccessor]:
        """Supply enthalpy (e.g., kJ/kg)."""
        return self._to_accessor(self._h_supply, "h_supply", writable=True)

    @h_supply.setter
    def h_supply(self, value: float):
        """Set the supply enthalpy and refresh derived stream attributes."""
        self._set_value_attribute("_h_supply", value)

    @property
    def h_target(self) -> Optional[StreamValueAccessor]:
        """Target enthalpy (e.g., kJ/kg)."""
        return self._to_accessor(self._h_target, "h_target", writable=True)

    @h_target.setter
    def h_target(self, value: float):
        """Set the target enthalpy and refresh derived stream attributes."""
        self._set_value_attribute("_h_target", value)

    @property
    def dt_cont(self) -> StreamValueAccessor:
        """Preserved base delta-T contribution before any zone multiplier."""
        return self._to_accessor(self._dt_cont, "dt_cont", writable=True)

    @dt_cont.setter
    def dt_cont(self, value: float):
        """Set the base contribution to shifted-temperature calculations."""
        self._dt_cont = self._coerce_to_value(value, "_dt_cont")
        self._dt_cont_act = self._scale_value(self._dt_cont, self._dt_cont_multiplier)

    @property
    def dt_cont_act(self) -> StreamValueAccessor:
        """Effective delta-T contribution used in shifted-temperature calculations."""
        return self._to_accessor(self._dt_cont_act, "dt_cont_act", writable=True)

    @dt_cont_act.setter
    def dt_cont_act(self, value: float):
        """Set the effective shifted-temperature contribution in active use."""
        self._dt_cont_act = self._coerce_to_value(value, "_dt_cont_act")

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
    def heat_flow(self) -> StreamValueAccessor:
        """Stream heat flow view over a scalar or stateful duty value."""
        return self._to_accessor(self._heat_flow, "heat_flow", writable=True)

    @heat_flow.setter
    def heat_flow(self, value: MaybeVU):
        """Set the stream duty, preserving the stream's existing state model."""
        self._heat_flow = self._coerce_to_stream_state_context(value, "_heat_flow")

    @property
    def htc(self) -> float:
        """Heat transfer coefficient (e.g., kW/m^2/K)."""
        return self._to_accessor(self._htc, "htc", writable=True)

    @htc.setter
    def htc(self, value: float):
        """Set the heat-transfer coefficient and refresh derived resistance."""
        self._set_value_attribute("_htc", value)

    @property
    def htr(self) -> float:
        """Heat transfer resistance (e.g., m^2.K/kW)."""
        return self._to_accessor(self._derived_value("_htr"), "htr", writable=False)

    @property
    def price(self) -> float:
        """Unit energy price (e.g., $/MWh)."""
        return self._to_accessor(self._price, "price", writable=True)

    @price.setter
    def price(self, value: float):
        """Set the unit energy price used in utility-cost calculations."""
        self._price = self._coerce_to_value(value, "_price")

    @property
    def ut_cost(self) -> float:
        """Utility cost (e.g., $/y)."""
        return self._to_accessor(
            self._derived_value("_ut_cost"), "ut_cost", writable=False
        )

    @property
    def CP(self) -> float:
        """Heat capacity flowrate (e.g., kW/K)."""
        return self._to_accessor(self._derived_value("_CP"), "CP", writable=False)

    @property
    def rCP(self) -> Optional[float]:
        """Resistance-capacity product (1/heat transfer rate)."""
        return self._to_accessor(
            self._derived_value("_RCP_prod"), "rCP", writable=False
        )

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
    def t_min(self) -> Optional[StreamValueAccessor]:
        """Minimum temperature (supply or target depending on hot/cold)."""
        return self._to_accessor(self._derived_value("_t_min"), "t_min", writable=False)

    @property
    def t_max(self) -> Optional[StreamValueAccessor]:
        """Maximum temperature (supply or target depending on hot/cold)."""
        return self._to_accessor(self._derived_value("_t_max"), "t_max", writable=False)

    @property
    def t_min_star(self) -> Optional[StreamValueAccessor]:
        """Shifted minimum temperature."""
        return self._to_accessor(
            self._derived_value("_t_min_star"), "t_min_star", writable=False
        )

    @property
    def t_max_star(self) -> Optional[StreamValueAccessor]:
        """Shifted maximum temperature."""
        return self._to_accessor(
            self._derived_value("_t_max_star"), "t_max_star", writable=False
        )

    # === Readable Alias Properties ===

    @property
    def stream_type(self) -> Optional[str]:
        """Alias for the stream thermal type."""
        return self.type

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
    def supply_temperature(self) -> Optional[StreamValueAccessor]:
        """Alias for the supply temperature."""
        return self.t_supply

    @supply_temperature.setter
    def supply_temperature(self, value: float):
        """Set the supply temperature via a descriptive alias."""
        self.t_supply = value

    @property
    def target_temperature(self) -> Optional[StreamValueAccessor]:
        """Alias for the target temperature."""
        return self.t_target

    @target_temperature.setter
    def target_temperature(self, value: float):
        """Set the target temperature via a descriptive alias."""
        self.t_target = value

    @property
    def minimum_temperature(self) -> Optional[StreamValueAccessor]:
        """Alias for the minimum stream temperature."""
        return self.t_min

    @property
    def maximum_temperature(self) -> Optional[StreamValueAccessor]:
        """Alias for the maximum stream temperature."""
        return self.t_max

    @property
    def shifted_minimum_temperature(self) -> Optional[StreamValueAccessor]:
        """Alias for the shifted minimum stream temperature."""
        return self.t_min_star

    @property
    def shifted_maximum_temperature(self) -> Optional[StreamValueAccessor]:
        """Alias for the shifted maximum stream temperature."""
        return self.t_max_star

    @property
    def supply_pressure(self) -> Optional[StreamValueAccessor]:
        """Alias for the supply pressure."""
        return self.P_supply

    @supply_pressure.setter
    def supply_pressure(self, value: float):
        """Set the supply pressure via a descriptive alias."""
        self.P_supply = value

    @property
    def target_pressure(self) -> Optional[StreamValueAccessor]:
        """Alias for the target pressure."""
        return self.P_target

    @target_pressure.setter
    def target_pressure(self, value: float):
        """Set the target pressure via a descriptive alias."""
        self.P_target = value

    @property
    def supply_enthalpy(self) -> Optional[StreamValueAccessor]:
        """Alias for the supply enthalpy."""
        return self.h_supply

    @supply_enthalpy.setter
    def supply_enthalpy(self, value: float):
        """Set the supply enthalpy via a descriptive alias."""
        self.h_supply = value

    @property
    def target_enthalpy(self) -> Optional[StreamValueAccessor]:
        """Alias for the target enthalpy."""
        return self.h_target

    @target_enthalpy.setter
    def target_enthalpy(self, value: float):
        """Set the target enthalpy via a descriptive alias."""
        self.h_target = value

    @property
    def delta_t_contribution(self) -> StreamValueAccessor:
        """Alias for the base shifted-temperature contribution."""
        return self.dt_cont

    @delta_t_contribution.setter
    def delta_t_contribution(self, value: float):
        """Set the base shifted-temperature contribution via an alias."""
        self.dt_cont = value

    @property
    def effective_delta_t_contribution(self) -> StreamValueAccessor:
        """Alias for the effective shifted-temperature contribution."""
        return self.dt_cont_act

    @effective_delta_t_contribution.setter
    def effective_delta_t_contribution(self, value: float):
        """Set the effective shifted-temperature contribution via an alias."""
        self.dt_cont_act = value

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

    @property
    def utility_cost(self) -> float:
        """Alias for the derived utility cost."""
        return self.ut_cost

    @property
    def heat_capacity_flow_rate(self) -> float:
        """Alias for the derived stream heat-capacity flow rate."""
        return self.CP

    @property
    def resistance_capacity_product(self) -> Optional[float]:
        """Alias for the stream resistance-capacity product."""
        return self.rCP

    # === Methods ===

    def _set_value_attribute(self, attr_name: str, value, update: bool = True) -> None:
        setattr(self, attr_name, self._coerce_to_value(value, attr_name))
        del update

    def _coerce_to_value(self, value, attr_name: str) -> Value | None:
        if value is None:
            return None
        if isinstance(value, StreamValueAccessor):
            return value.raw_value
        incoming_state_ids, incoming_weights = extract_state_context(value)
        if incoming_state_ids is not None:
            active_state_ids = self.state_ids
            active_weights = self.weights
            if active_state_ids is None:
                self._set_local_state_context(incoming_state_ids, incoming_weights)
            elif incoming_state_ids != active_state_ids:
                raise ValueError(
                    f"state_ids for {attr_name[1:]} "
                    f"must align with the stream state_ids."
                )
            elif not np.allclose(
                incoming_weights,
                active_weights,
                rtol=_STATE_WEIGHT_RTOL,
                atol=_STATE_WEIGHT_ATOL,
            ):
                raise ValueError(
                    f"weights for {attr_name[1:]} must align with the stream weights."
                )
        parsed = Value(value)
        target_unit = self._VALUE_UNITS[attr_name]
        if target_unit is None:
            return parsed
        return Value(parsed, unit=target_unit)

    def _to_accessor(self, value, attr_name: str, *, writable: bool):
        if isinstance(value, Value):
            state_ids, weights = self._current_state_context()
            return StreamValueAccessor(
                self,
                attr_name,
                value,
                writable=writable,
                state_ids=state_ids,
                weights=weights,
            )
        return value

    def _normalise_existing_value(self, value, attr_name: str) -> Value | None:
        if value is None:
            return None
        if isinstance(value, StreamValueAccessor):
            return value.raw_value
        if isinstance(value, Value):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool
        ):
            return Value(value, unit=self._VALUE_UNITS[attr_name])
        return None

    def _get_state_context(
        self,
        field_names: tuple[str, ...] | None = None,
    ) -> tuple[list[str] | None, np.ndarray | None]:
        fields = field_names or self._CORE_STATE_FIELDS
        return self._resolve_state_context(
            {field_name: getattr(self, field_name) for field_name in fields}
        )

    def _resolve_state_context(self, values_by_name: dict[str, Any]):
        stateful_values: list[tuple[str, Value]] = []
        for name, value in values_by_name.items():
            value_obj = self._normalise_existing_value(value, name)
            if value_obj is not None and len(value_obj.state_values) > 1:
                stateful_values.append((name, value_obj))

        if not stateful_values:
            return None, None

        ref_name, ref_value = stateful_values[0]
        ref_len = len(ref_value.state_values)

        for name, value_obj in stateful_values[1:]:
            if len(value_obj.state_values) != ref_len:
                raise ValueError(
                    f"state count for {name[1:]} must align with {ref_name[1:]}."
                )

        active_state_ids, active_weights = self._current_state_context()
        if active_state_ids is not None:
            if len(active_state_ids) != ref_len:
                raise ValueError(
                    "Stream state_ids must align with the stored state count."
                )
            return active_state_ids, active_weights

        state_ids = [str(idx) for idx in range(ref_len)]
        weights = np.ones(ref_len, dtype=float) / ref_len
        return state_ids, weights

    def _broadcast_value_object(
        self,
        value_obj: Value | None,
        state_ids: list[str] | None,
        attr_name: str,
    ) -> np.ndarray | None:
        if value_obj is None:
            return None
        if state_ids is None:
            return np.asarray([float(value_obj.value)], dtype=float)
        if len(value_obj.state_values) == 1:
            return np.full(len(state_ids), float(value_obj.value), dtype=float)
        if len(value_obj.state_values) != len(state_ids):
            raise ValueError(
                f"state count for {attr_name[1:]} must align with the stream state_ids."
            )
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
        del weights
        return Value(values=arr, unit=self._VALUE_UNITS[attr_name])

    def _scale_value(self, value: Value | None, factor: float) -> Value | None:
        if value is None:
            return None
        scaled = np.asarray(value.state_values, dtype=float) * float(factor)
        if len(value.state_values) == 1:
            return Value(float(scaled[0]), unit=value.unit)
        return Value(values=scaled, unit=value.unit)

    def _coerce_to_stream_state_context(self, value, attr_name: str) -> Value | None:
        parsed = self._coerce_to_value(value, attr_name)
        if parsed is None:
            return None

        state_ids, weights = self._get_state_context()
        if state_ids is None:
            return parsed

        if len(parsed.state_values) == 1:
            return Value(
                values=np.full(len(state_ids), float(parsed.value), dtype=float),
                unit=parsed.unit,
            )

        if len(parsed.state_values) != len(state_ids):
            raise ValueError(
                f"state count for {attr_name[1:]} must align with the stream state_ids."
            )
        return parsed

    @property
    def state_ids(self) -> list[str] | None:
        state_ids, _ = self._current_state_context()
        if state_ids is None:
            state_ids, _ = self._get_state_context()
        return state_ids

    @property
    def weights(self) -> np.ndarray | None:
        _state_ids, weights = self._current_state_context()
        if weights is None:
            _state_ids, weights = self._get_state_context()
        return None if weights is None else weights.copy()

    def _current_state_context(
        self,
    ) -> tuple[list[str] | None, np.ndarray | None]:
        if (
            self._state_collection is not None
            and self._state_collection._state_ids is not None
        ):
            return (
                list(self._state_collection._state_ids),
                self._state_collection._weights.copy(),
            )
        if self._state_ids is None:
            return None, None
        return list(self._state_ids), self._weights.copy()

    def _set_local_state_context(
        self,
        state_ids: list[str] | None,
        weights: np.ndarray | list[float] | None,
    ) -> None:
        if state_ids is None:
            self._state_ids = None
            self._weights = None
            return

        state_ids_list = [str(state_id) for state_id in state_ids]
        weights_arr = np.asarray(weights, dtype=float).reshape(-1)
        if len(state_ids_list) != len(weights_arr):
            raise ValueError("state_ids and weights must have the same length.")
        if not np.allclose(
            weights_arr.sum(), 1.0, rtol=_STATE_WEIGHT_RTOL, atol=_STATE_WEIGHT_ATOL
        ):
            weights_arr = weights_arr / float(weights_arr.sum())
        self._state_ids = state_ids_list
        self._weights = weights_arr.copy()

    def bind_state_collection(self, collection: StreamCollection | None) -> None:
        """Bind this stream to a collection-owned shared state model."""
        self._state_collection = collection

    def clear_state_context(self) -> None:
        """Detach any bound collection state model and local fallback context."""
        self._state_collection = None
        self._state_ids = None
        self._weights = None

    def resolve_attr(
        self,
        attr_name: str,
        state_id: str | None = None,
        *,
        default_allowed: bool = True,
    ) -> float | None:
        """Resolve one stream attribute to a scalar for the selected state."""
        internal_name = (
            attr_name
            if attr_name.startswith("_")
            else f"_{attr_name}"
            if hasattr(self, f"_{attr_name}")
            else attr_name
        )
        value = getattr(self, internal_name)
        return resolve_value_for_state(
            value,
            state_id=state_id,
            state_ids=self.state_ids,
            default_allowed=default_allowed,
        )

    def set_attr_for_state(
        self,
        attr_name: str,
        value,
        *,
        state_id: str | None,
    ) -> None:
        """Update one state of a value-backed attribute while preserving others."""
        if not attr_name.startswith("_"):
            public_name = attr_name
            internal_name = (
                f"_{attr_name}" if hasattr(self, f"_{attr_name}") else attr_name
            )
        else:
            internal_name = attr_name
            public_name = attr_name[1:]

        existing = self._normalise_existing_value(
            getattr(self, internal_name), internal_name
        )
        state_ids = self.state_ids
        if existing is None or state_ids is None or len(existing.state_values) <= 1:
            setattr(self, public_name, value)
            return

        if state_id is None:
            state_id = "0" if "0" in state_ids else state_ids[0]
        if state_id not in state_ids:
            raise ValueError(
                f"Unknown state_id {state_id!r} for stream {self.name!r}. "
                f"Available states: {', '.join(state_ids)}."
            )

        parsed = self._coerce_to_value(value, internal_name)
        scalar_value = resolve_value_for_state(
            parsed, state_id=state_id, state_ids=state_ids
        )
        state_values = existing.state_values
        state_values[state_ids.index(state_id)] = float(scalar_value)
        updated = Value(values=state_values, unit=existing.unit)
        setattr(self, internal_name, updated)

    def _derive_stream_state(self) -> _DerivedStreamState | None:
        """Derive temperature bounds and transport properties from core values."""
        t_supply_value = self._normalise_existing_value(self._t_supply, "_t_supply")
        t_target_value = self._normalise_existing_value(self._t_target, "_t_target")
        htc_value = self._normalise_existing_value(self._htc, "_htc")
        heat_flow_value = self._normalise_existing_value(self._heat_flow, "_heat_flow")
        dt_cont_act_value = self._normalise_existing_value(
            self._dt_cont_act, "_dt_cont_act"
        )
        price_value = self._normalise_existing_value(self._price, "_price")

        if t_supply_value is None or t_target_value is None or htc_value is None:
            return None

        state_ids, weights = self._resolve_state_context(
            {
                "_t_supply": t_supply_value,
                "_t_target": t_target_value,
                "_heat_flow": heat_flow_value,
                "_dt_cont": self._dt_cont,
                "_dt_cont_act": dt_cont_act_value,
                "_htc": htc_value,
                "_price": price_value,
            }
        )

        t_supply_arr = self._broadcast_value_object(
            t_supply_value, state_ids, "_t_supply"
        )
        t_target_arr = self._broadcast_value_object(
            t_target_value, state_ids, "_t_target"
        )
        heat_flow_arr = self._broadcast_value_object(
            heat_flow_value, state_ids, "_heat_flow"
        )
        if heat_flow_arr is None:
            heat_flow_arr = np.zeros_like(t_supply_arr, dtype=float)
        dt_cont_act_arr = self._broadcast_value_object(
            dt_cont_act_value, state_ids, "_dt_cont_act"
        )
        if dt_cont_act_arr is None:
            dt_cont_act_arr = np.zeros_like(t_supply_arr, dtype=float)
        htc_arr = self._broadcast_value_object(htc_value, state_ids, "_htc")
        if htc_arr is None:
            return

        equal_mask = np.isclose(
            t_supply_arr, t_target_arr, atol=_TEMPERATURE_EQUAL_TOL, rtol=0.0
        )
        if np.any(equal_mask):
            adjusted_target_arr = t_target_arr.copy()
            cold_mask = equal_mask & (heat_flow_arr > 0.0)
            hot_mask = equal_mask & (heat_flow_arr < 0.0)
            adjusted_target_arr[cold_mask] = t_supply_arr[cold_mask] + 0.01
            adjusted_target_arr[hot_mask] = t_supply_arr[hot_mask] - 0.01
            t_target_arr = adjusted_target_arr

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
            stream_type = ST.Hot.value
            t_min_arr = t_target_arr
            t_max_arr = t_supply_arr
            t_min_star_arr = t_min_arr - dt_cont_act_arr
            t_max_star_arr = t_max_arr - dt_cont_act_arr
        elif np.any(cold_states):
            stream_type = ST.Cold.value
            t_min_arr = t_supply_arr
            t_max_arr = t_target_arr
            t_min_star_arr = t_min_arr + dt_cont_act_arr
            t_max_star_arr = t_max_arr + dt_cont_act_arr
        else:
            stream_type = ST.Both.value
            t_min_arr = t_supply_arr
            t_max_arr = t_target_arr
            t_min_star_arr = t_min_arr.copy()
            t_max_star_arr = t_max_arr.copy()

        dt_arr = t_max_arr - t_min_arr
        cp_arr = np.zeros_like(dt_arr, dtype=float)
        non_zero_dt = np.abs(dt_arr) > _TEMPERATURE_EQUAL_TOL
        cp_arr[non_zero_dt] = heat_flow_arr[non_zero_dt] / dt_arr[non_zero_dt]

        htr_arr = np.zeros_like(htc_arr, dtype=float)
        positive_mask = htc_arr > 0.0
        htr_arr[positive_mask] = 1.0 / htc_arr[positive_mask]
        price_arr = self._broadcast_value_object(price_value, state_ids, "_price")
        if price_arr is None:
            price_arr = np.zeros_like(heat_flow_arr, dtype=float)
        ut_cost_arr = (heat_flow_arr / 1000.0) * price_arr

        rcp_arr = np.zeros_like(cp_arr, dtype=float)
        rcp_arr[positive_mask] = cp_arr[positive_mask] * htr_arr[positive_mask]

        return _DerivedStreamState(
            state_ids=state_ids,
            weights=weights,
            stream_type=stream_type,
            t_min=t_min_arr,
            t_max=t_max_arr,
            t_min_star=t_min_star_arr,
            t_max_star=t_max_star_arr,
            cp=cp_arr,
            htr=htr_arr,
            ut_cost=ut_cost_arr,
            rcp=rcp_arr,
        )

    def _derived_value(self, attr_name: str) -> Value | None:
        derived = self._derive_stream_state()
        if derived is None:
            return None

        magnitudes = {
            "_t_min": derived.t_min,
            "_t_max": derived.t_max,
            "_t_min_star": derived.t_min_star,
            "_t_max_star": derived.t_max_star,
            "_CP": derived.cp,
            "_htr": derived.htr,
            "_ut_cost": derived.ut_cost,
            "_RCP_prod": derived.rcp,
        }[attr_name]
        return self._value_from_array(
            magnitudes,
            attr_name=attr_name,
            state_ids=derived.state_ids,
            weights=derived.weights,
        )

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

        self._is_process_stream = True
