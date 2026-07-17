"""Plotly helper routines for quick temperature-enthalpy visual checks."""

import numpy as np

from ...adapters.optional_dependencies import optional_dependency_error

__all__ = [
    "graph_simple_cc_plot",
    "plot_t_h_curve",
    "plot_t_h_curve_with_piecewise_and_bounds",
]


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            optional_dependency_error(
                package="Plotly",
                purpose="graph_simple_cc_plot",
                extras=("notebook", "dashboard"),
                docs="the graphing and exporting results guides",
            )
        ) from exc
    return go


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


def plot_t_h_curve(points, title: str = "Temperature vs. Enthalpy") -> None:
    """
    Plot Temperature vs. Enthalpy.
    :param points: tuple with columns 'Temperature (K)' and 'Enthalpy (kJ/mol)'.
    :param title: Title of the graph.
    :returns: None
    """
    go = _require_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines+markers",
            name="T-H Curve",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Heat Flow / kW",
        yaxis_title="Temperature / \N{DEGREE SIGN}C",
        template="plotly_white",
    )
    fig.show()


def plot_t_h_curve_with_piecewise_and_bounds(
    points: np.array,
    piecewise_points: np.array,
    epsilon: float,
    title: str = "Temperature vs. Enthalpy",
) -> None:
    """
    Plot the TH curve, its piecewise linearization, and a shaded ±epsilon band.
    :param points: Original TH curve points.
    :param piecewise_points: Simplified piecewise linear curve points.
    :param epsilon: Epsilon value for shading.
    :param title: Title of the graph.
    """
    go = _require_plotly()
    enthalpies, temperatures = points[:, 0], points[:, 1]
    upper_bound = [e + epsilon for e in temperatures]
    lower_bound = [e - epsilon for e in temperatures]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=enthalpies,
            y=temperatures,
            mode="lines",
            name="TH Curve",
            line={"color": "red", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=piecewise_points[:, 0],
            y=piecewise_points[:, 1],
            mode="lines+markers",
            name="Piecewise Curve",
            line={"color": "blue", "width": 2, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(enthalpies) + list(reversed(enthalpies)),
            y=upper_bound + list(reversed(lower_bound)),
            fill="toself",
            fillcolor="rgba(135, 206, 250, 0.3)",
            line={"color": "rgba(135, 206, 250, 0.3)"},
            hoverinfo="skip",
            showlegend=True,
            name=f"±{epsilon} Bounds",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Heat Flow / kW",
        yaxis_title="Temperature / \N{DEGREE SIGN}C",
        template="plotly_white",
    )
    fig.show()
