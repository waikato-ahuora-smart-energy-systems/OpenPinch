"""Shared numerical helpers."""

from typing import Any, Iterable, Tuple, Union

import numpy as np

from ..classes.value import Value
from ..lib.config import tol
from ..lib.schemas.common import MaybeVU, StatefulValueWithUnit, ValueWithUnit

__all__ = [
    "clean_composite_curve",
    "clean_composite_curve_ends",
    "delta_vals",
    "delta_with_zero_at_start",
    "g_ineq_penalty",
    "get_state_index",
    "get_value",
    "graph_simple_cc_plot",
    "interp_with_plateaus",
    "linear_interpolation",
    "make_monotonic",
    "resolve_stream_attr",
    "resolve_stream_attr_array",
]


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "Plotly is required for graph_simple_cc_plot. "
            "Install it directly or reinstall OpenPinch with "
            "'pip install openpinch[notebook]' or 'pip install openpinch[dashboard]'."
        ) from exc
    return go


def resolve_value_for_state(
    val: Any,
    state_id: str | None = None,
    *,
    state_ids: dict[str, int] | list[str] | None = None,
    default_allowed: bool = True,
) -> float | None:
    """Return one scalar magnitude from a scalar or stateful value-like object."""
    if val is None:
        return None
    if isinstance(val, Value):
        raw_value = val
    else:
        raw_value = Value(val)

    if len(raw_value.state_values) <= 1:
        return float(raw_value.value)

    if state_ids is None and isinstance(val, dict) and val.get("state_ids") is not None:
        state_ids = {str(sid): idx for idx, sid in enumerate(val["state_ids"])}

    if isinstance(state_ids, dict):
        state_lookup = {str(sid): int(idx) for sid, idx in state_ids.items()}
    elif state_ids:
        state_lookup = {str(sid): idx for idx, sid in enumerate(state_ids)}
    else:
        state_lookup = None

    if state_lookup is None:
        if not default_allowed and state_id is not None:
            raise ValueError("state_ids are required for stateful values.")
        return float(raw_value[0].value)

    resolved_state_id = None if state_id is None else str(state_id)
    if resolved_state_id is None:
        if not default_allowed:
            raise ValueError("state_id is required for stateful values.")
        resolved_state_id = "0" if "0" in state_lookup else next(iter(state_lookup))

    if resolved_state_id not in state_lookup:
        raise ValueError(
            f"Unknown state_id {resolved_state_id!r}. "
            f"Available states: {', '.join(state_lookup)}."
        )
    return float(raw_value[state_lookup[resolved_state_id]].value)


def resolve_stream_attr(
    stream: Any,
    attr_name: str,
    state_id: str | None = None,
    *,
    default_allowed: bool = True,
) -> float | None:
    """Resolve one stream attribute to a scalar for the selected state."""
    if not hasattr(stream, attr_name):
        raise AttributeError(f"Stream {stream!r} has no attribute {attr_name!r}.")
    return resolve_value_for_state(
        getattr(stream, attr_name),
        state_id=state_id,
        state_ids=getattr(stream, "state_ids", None),
        default_allowed=default_allowed,
    )


def resolve_stream_attr_array(
    streams: Iterable[Any],
    attr_name: str,
    state_id: str | None = None,
    *,
    default_allowed: bool = True,
) -> np.ndarray:
    """Resolve one attribute across a stream iterable into a float array."""
    return np.asarray(
        [
            resolve_stream_attr(
                stream,
                attr_name,
                state_id=state_id,
                default_allowed=default_allowed,
            )
            for stream in streams
        ],
        dtype=float,
    )


def get_value(
    val: Union[float, int, str, dict, "ValueWithUnit", None],
    val2: Union[float, int, str, None] = None,
    zone_name: str = None,
    state_id: str | None = None,
) -> float:
    """Extract a numeric value from supported scalars and payload wrappers."""
    if isinstance(val, bool):
        raise TypeError(
            "Unsupported type: "
            f"{type(val)}. Expected float, int, numeric string, dict, "
            "or ValueWithUnit."
        )
    elif isinstance(val, Value):
        return resolve_value_for_state(val, state_id=state_id)
    elif isinstance(val, (float, int)):
        return float(val)
    elif hasattr(val, "model_dump"):
        return get_value(
            val.model_dump(mode="python"),
            val2=val2,
            zone_name=zone_name,
            state_id=state_id,
        )
    elif isinstance(val, dict):
        if zone_name in val:
            return get_value(val[zone_name], val2=val2, state_id=state_id)

        if _is_stateful_value_payload(val):
            return resolve_value_for_state(val, state_id=state_id)

        payload = val.copy()
        if "value" not in payload:
            if val2 is None:
                raise KeyError("value")
            payload["value"] = val2

        if len(payload) > 2:
            raise ValueError(
                "Invalid payload: more than one operation specified. Payload "
                "must contain only 'value' and at most one of "
                "'multiplier', 'multiply', 'add', 'subtract', 'divide', "
                "'power', 'log', 'exp', 'abs', 'min', or 'max'."
            )

        value = get_value(payload["value"], state_id=state_id)

        if "multiplier" in payload:
            return value * get_value(payload["multiplier"], state_id=state_id)
        elif "multiply" in payload:
            return value * get_value(payload["multiply"], state_id=state_id)
        elif "add" in payload:
            return value + get_value(payload["add"], state_id=state_id)
        elif "subtract" in payload:
            return value - get_value(payload["subtract"], state_id=state_id)
        elif "divide" in payload:
            return (
                value / get_value(payload["divide"], state_id=state_id)
                if value != 0
                else 0.0
            )
        elif "power" in payload:
            return value ** get_value(payload["power"], state_id=state_id)
        elif "log" in payload:
            base = payload["log"] if isinstance(payload["log"], float) else np.e
            return np.log(value) / np.log(base) if value > 0 else 0.0
        elif "exp" in payload:
            base = payload["exp"] if isinstance(payload["exp"], float) else np.e
            return base**value if value > 0 else 0.0
        elif "abs" in payload:
            return abs(value)
        elif "min" in payload:
            return min(value, get_value(payload["min"], state_id=state_id))
        elif "max" in payload:
            return max(value, get_value(payload["max"], state_id=state_id))
        else:
            return value
    elif _is_value_with_unit(val):
        return val.value
    elif isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            raise TypeError(
                f"Unsupported string value: {val}. String must be convertible to float."
            )
    elif val is None and val2 is not None:
        return get_value(val2, zone_name=zone_name, state_id=state_id)
    elif val is None:
        return None
    else:
        raise TypeError("Unsupported type")


def get_values(obj: MaybeVU) -> np.ndarray:
    if isinstance(obj, ValueWithUnit):
        return np.asarray(obj.value)
    elif isinstance(obj, (float, int)):
        return np.asarray([float(obj)])
    elif isinstance(obj, StatefulValueWithUnit):
        return np.asarray(obj.values)
    elif obj is None:
        return np.array([])
    else:
        raise TypeError("Unsupported type")


def _is_value_with_unit(val: Any) -> bool:
    """Return ``True`` for objects that look like ``ValueWithUnit`` containers."""
    return hasattr(val, "value") and hasattr(val, "unit")


def _is_stateful_value_payload(val: Any) -> bool:
    """Return ``True`` for dict-like stateful value payloads."""
    if not isinstance(val, dict):
        return False
    keys = set(val)
    return keys.issubset({"values", "state_ids", "weights", "unit"}) and (
        "values" in keys or "state_ids" in keys or "weights" in keys
    )


def get_state_index(state_ids, args: dict) -> Tuple[int, str]:
    sid = None if not isinstance(args, dict) else args.get("state_id")
    if isinstance(state_ids, dict):
        if sid is None:
            idx = 0
        elif sid in state_ids.keys():
            idx = state_ids[sid]
        else:
            raise ValueError(
                f"state_id {sid!r} was not found on this collection. "
                f"Available states: {', '.join(state_ids.keys())}."
            )
    elif state_ids:
        lookup = [str(state_id) for state_id in state_ids]
        sid = None if sid is None else str(sid)
        if sid is None:
            idx = 0
        elif sid in lookup:
            idx = lookup.index(sid)
        else:
            raise ValueError(
                f"state_id {sid!r} was not found on this collection. "
                f"Available states: {', '.join(lookup)}."
            )
    else:
        idx = 0
    return idx, sid


def linear_interpolation(
    xi: float, x1: float, x2: float, y1: float, y2: float
) -> float:
    """Estimate ``y`` at ``xi`` using two known points and linear interpolation."""
    if x1 == x2:
        raise ValueError(
            "Cannot perform interpolation when x1 == x2 (undefined slope)."
        )
    m = (y1 - y2) / (x1 - x2)
    c = y1 - m * x1
    yi = m * xi + c
    return yi


def delta_with_zero_at_start(x: np.ndarray) -> np.ndarray:
    """Compute successive differences and prepend a zero entry."""
    return np.insert(delta_vals(x), 0, 0.0)


def delta_vals(x: np.ndarray, descending_vals: bool = True) -> np.ndarray:
    """Compute difference between successive entries in a column."""
    deltas = x[:-1] - x[1:] if descending_vals else x[1:] - x[:-1]
    deltas[np.abs(deltas) <= tol] = 0.0
    return deltas


def clean_composite_curve_ends(
    y_vals: np.ndarray | list, x_vals: np.ndarray | list
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove redundant points in composite curves."""
    y_vals = np.array(y_vals)
    x_vals = np.array(x_vals)

    if np.all(np.isclose(x_vals, 0.0, atol=tol)) or np.abs(x_vals.var()) < tol:
        return np.array([]), np.array([])

    mask_0 = ~np.isclose(x_vals, x_vals[0] * np.ones(len(x_vals)), atol=tol)
    start = np.flatnonzero(mask_0)[0] - 1
    mask_1 = ~np.isclose(x_vals, x_vals[-1] * np.ones(len(x_vals)), atol=tol)
    end = np.flatnonzero(mask_1)[-1] + 1

    x_clean = x_vals[start : end + 1]
    y_clean = y_vals[start : end + 1]
    return y_clean, x_clean


def clean_composite_curve(
    y_array: np.ndarray | list, x_array: np.ndarray | list
) -> Tuple[np.ndarray | list]:
    """Remove redundant points in composite curves."""

    # Round to avoid tiny numerical errors
    y_vals, x_vals = clean_composite_curve_ends(y_array, x_array)

    if len(x_vals) <= 2:
        return y_vals, x_vals

    x_clean, y_clean = [x_vals[0]], [y_vals[0]]

    for i in range(1, len(x_vals) - 1):
        x1, x2, x3 = x_vals[i - 1], x_vals[i], x_vals[i + 1]
        y1, y2, y3 = y_vals[i - 1], y_vals[i], y_vals[i + 1]

        if x1 == x3:
            # All three x are the same; keep x2 only if y2 is different
            if x1 != x2:
                x_clean.append(x2)
                y_clean.append(y2)
        else:
            # Linear interpolation check
            y_interp = y1 + (y3 - y1) * (x2 - x1) / (x3 - x1)
            if abs(y2 - y_interp) > tol:
                x_clean.append(x2)
                y_clean.append(y2)

    x_clean.append(x_vals[-1])
    y_clean.append(y_vals[-1])

    if abs(x_clean[0] - x_clean[1]) < tol:
        x_clean.pop(0)
        y_clean.pop(0)

    i = len(x_clean) - 1
    if abs(x_clean[i] - x_clean[i - 1]) < tol:
        x_clean.pop(i)
        y_clean.pop(i)

    return np.asarray(y_clean), np.asarray(x_clean)


def graph_simple_cc_plot(Tc, Hc, Th, Hh):
    """Render a quick Plotly plot of hot/cold composite curves for debugging."""
    go = _require_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Hc,
            y=Tc,
            mode="lines",
            name="Cold composite",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Hh,
            y=Th,
            mode="lines",
            name="Hot composite",
        )
    )
    fig.update_layout(
        title="Balanced Composite Curves",
        xaxis_title="Enthalpy",
        yaxis_title="Temperature",
        template="plotly_white",
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)")
    fig.show()
    return fig


def interp_with_plateaus(
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    targets: np.ndarray,
    side: str,
    tol: float = 1e-6,
) -> np.ndarray:
    """Interpolate temperatures while respecting vertical curve segments."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    h_vals = np.asarray(h_vals, dtype=float)
    t_vals = np.asarray(t_vals, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if h_vals.size == 1:
        return np.full_like(targets, t_vals[0], dtype=float)

    h_monotonic = make_monotonic(h_vals, side, tol)
    return np.interp(targets, h_monotonic, t_vals)


def make_monotonic(h_vals: np.ndarray, side: str, tol: float = 1e-6) -> np.ndarray:
    """Adjust repeated values to become strictly increasing for interpolation."""
    adjusted = np.asarray(h_vals, dtype=float).copy()
    if adjusted.size <= 1:
        return adjusted

    eps = tol * 0.5
    # Identify the start of each strictly increasing block
    diff = np.abs(np.diff(adjusted)) > tol
    starts = np.flatnonzero(np.concatenate(([True], diff)))
    n = adjusted.size
    lengths = np.diff(np.append(starts, n))

    if np.all(lengths == 1):
        return adjusted

    # Compute position within each block using vectorised repetition
    within_block = np.arange(n) - np.repeat(starts, lengths)
    block_lengths = np.repeat(lengths, lengths)
    mask = block_lengths > 1

    offsets = np.zeros_like(adjusted)
    if side == "right":
        offsets[mask] = (block_lengths[mask] - 1 - within_block[mask]) * eps
        adjusted[mask] -= offsets[mask]
    else:  # side == "left"
        offsets[mask] = within_block[mask] * eps
        adjusted[mask] += offsets[mask]

    return adjusted


def g_ineq_penalty(
    g: float | list | np.ndarray,
    *,
    eta: float = 0.01,
    rho: float = 10,
    form: str = "square",
) -> np.float64:
    """Return a penalty value for an inequality-constraint residual."""
    g = np.asarray(g, dtype=float)
    if (
        form.lower() == "square_root_smoothing"
        or form.lower() == "square root smoothing"
    ):
        p = 0.5 * rho * (g + ((g) ** 2 + (eta) ** 2) ** 0.5)
    elif form.lower() == "square":
        p = rho * (g**2)
    else:
        raise ValueError("Unrecognised penalty function form selection.")

    if isinstance(p, float):
        return np.float64(p)
    elif isinstance(p, np.ndarray):
        return p.sum()
    else:
        raise ValueError(
            "Return of the penalty function failed due to unrecognised type."
        )
