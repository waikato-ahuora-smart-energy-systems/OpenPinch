import matplotlib.pyplot as plt
import numpy as np


def plot_t_h_curve(points, title: str = "Temperature vs. Enthalpy") -> None:
    """
    Plot Temperature vs. Enthalpy.
    :param points: tuple with columns 'Temperature (K)' and 'Enthalpy (kJ/mol)'.
    :param title: Title of the graph.
    :returns: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], marker="o", linestyle="-")
    plt.title(title, fontsize=16)
    plt.ylabel("Temperature (K)", fontsize=14)
    plt.xlabel("Enthalpy (kJ/mol)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()


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

    plt.figure(figsize=(10, 6))
    plt.plot(enthalpies, temperatures, label="TH Curve", color="red", linewidth=1.5)
    plt.plot(
        piecewise_points[:, 0],
        piecewise_points[:, 1],
        label="Piecewise Curve",
        color="blue",
        linestyle="--",
        marker="o",
        linewidth=2,
    )  ## pwl curve should be dashed and blue
    plt.fill_between(
        enthalpies,
        lower_bound,
        upper_bound,
        color="lightblue",
        alpha=0.3,
        label=f"±{epsilon} Bounds",
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Enthalpy (J/mol)", fontsize=14)
    plt.ylabel("Temperature (K)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
