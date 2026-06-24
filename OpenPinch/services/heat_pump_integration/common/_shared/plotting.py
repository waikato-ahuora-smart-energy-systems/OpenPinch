"""Private plotting helpers for HPR targeting."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - optional dependency guard
    go = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None

from .....classes.stream_collection import StreamCollection
from .....lib.enums import PT
from .....utils.optional_dependencies import optional_dependency_error
from ....common.graph_data import clean_composite_curve_ends
from ....common.problem_table_analysis import (
    create_problem_table_with_t_int,
    get_utility_heat_cascade,
)


def plot_multi_hp_profiles_from_results(
    T_hot: np.ndarray = None,
    H_hot: np.ndarray = None,
    T_cold: np.ndarray = None,
    H_cold: np.ndarray = None,
    hpr_hot_streams: StreamCollection = None,
    hpr_cold_streams: StreamCollection = None,
    idx: int = 0,
    title: str = None,
) -> "go.Figure":  # type: ignore
    """Plot background source/sink profiles alongside solved HPR cycle streams."""
    go = _require_plotly()
    fig = go.Figure()

    if T_hot is not None and H_hot is not None:
        T_hot, H_hot = clean_composite_curve_ends(T_hot, H_hot)
        fig.add_trace(
            go.Scatter(
                x=H_hot,
                y=T_hot,
                mode="lines",
                name="Sink",
                line={"color": "red", "width": 2},
            )
        )

    if T_cold is not None and H_cold is not None:
        T_cold, H_cold = clean_composite_curve_ends(T_cold, H_cold)
        fig.add_trace(
            go.Scatter(
                x=H_cold,
                y=T_cold,
                mode="lines",
                name="Source",
                line={"color": "blue", "width": 2},
            )
        )

    if hpr_hot_streams is not None and hpr_cold_streams is not None:
        T_hpr_arr, H_hpr_hot, H_hpr_cold = _get_hpr_cascade(
            hpr_hot_streams,
            hpr_cold_streams,
            idx=idx,
        )
        T_hpr_hot, H_hpr_hot = clean_composite_curve_ends(T_hpr_arr, H_hpr_hot)
        T_hpr_cold, H_hpr_cold = clean_composite_curve_ends(T_hpr_arr, H_hpr_cold)
        fig.add_trace(
            go.Scatter(
                x=H_hpr_hot,
                y=T_hpr_hot,
                mode="lines",
                name="Condenser",
                line={"color": "darkred", "width": 1.8, "dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=H_hpr_cold,
                y=T_hpr_cold,
                mode="lines",
                name="Evaporator",
                line={"color": "darkblue", "width": 1.8, "dash": "dash"},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Heat Flow / kW",
        yaxis_title="Temperature / degC",
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)", zeroline=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.2)")
    fig.add_vline(x=0.0, line_color="black", line_width=2)
    return fig


def _require_plotly():
    if _PLOTLY_IMPORT_ERROR is not None:
        raise ImportError(
            optional_dependency_error(
                package="Plotly",
                purpose="HPR profile plotting",
                extras=("notebook", "dashboard"),
                docs="the heat-pump workflows guide",
            )
        ) from _PLOTLY_IMPORT_ERROR
    return go


def _get_hpr_cascade(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    *,
    idx: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = idx or 0
    pt = create_problem_table_with_t_int(
        streams=hot_streams + cold_streams,
        is_shifted=False,
        idx=idx,
    )
    pt.update(
        **get_utility_heat_cascade(
            pt[PT.T],
            hot_streams,
            cold_streams,
            is_shifted=False,
            idx=idx,
        )
    )
    return pt[PT.T], pt[PT.H_HOT_UT], pt[PT.H_COLD_UT]
