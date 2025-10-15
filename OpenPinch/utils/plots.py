import plotly.graph_objects as go
import numpy as np


def plot_t_h_curve(points, title: str = "Temperature vs. Enthalpy") -> None:
    """
    Plot Temperature vs. Enthalpy.
    :param points: tuple with columns 'Temperature (K)' and 'Enthalpy (kJ/mol)'.
    :param title: Title of the graph.
    :returns: None
    """
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
    Plot the TH curve, its piecewise linearization, and a shaded region ±epsilon around the TH curve.
    :param points: Original TH curve points.
    :param piecewise_points: Simplified piecewise linear curve points.
    :param epsilon: Epsilon value for shading.
    :param title: Title of the graph.
    """
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
