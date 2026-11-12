"""Compressor isentropic efficiency models for vapour compression HPR cycles.

The correlations follow HeatPumpLab (Apache-2.0,
https://github.com/waikato-ahuora-smart-energy-systems/HeatPumpLab,
``HeatPumpLab.components.compressor``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

# Pareto front of design compressor efficiency against pressure ratio (HeatPumpLab,
# October 2026): eta(PR) = eta_1 - a ln(PR)^b - d / (1 + exp(-(ln(PR) - L0) / w)),
# a lower envelope of the best published shaft-basis isentropic efficiency for
# 1 <= PR <= 20.
_PARETO_FRONT_COEFFS = (0.8695, 0.025392, 1.476, 0.29915, 2.9599, 0.2)


def pareto_front_efficiency(pressure_ratio: float) -> float:
    """Return the shaft-basis isentropic efficiency on the Pareto front."""
    log_pr = math.log(max(float(pressure_ratio), 1.0))
    eta_1, a, b, d, L0, w = _PARETO_FRONT_COEFFS
    return eta_1 - a * log_pr**b - d / (1.0 + math.exp(-(log_pr - L0) / w))


def wang_cleland_efficiency(pressure_ratio: float) -> float:
    """Return the shaft-basis isentropic efficiency of Wang and Cleland (2018).

    The overall efficiency polynomial is divided by a motor efficiency of 0.96.
    """
    pr = float(pressure_ratio)
    return (0.65 + 0.015 * pr - 0.0015 * pr**2) / 0.96


def brendel_efficiency(pressure_ratio: float, suction_pressure: float) -> float:
    """Return the isentropic efficiency of Brendel et al.; pressure in kPa.

    The correlation takes the suction pressure in MPa; without compression
    (pressure ratio at most 1) it returns 0.4.
    """
    pr = float(pressure_ratio)
    if pr <= 1.0:
        return 0.4
    P_mpa = float(suction_pressure) / 1000.0
    return 0.66981 - 0.6 / (pr - 0.01466) ** (8.38 * P_mpa) - 0.00102 * pr**1.8


@dataclass(frozen=True)
class CompressorConditions:
    """Operating point handed to a compressor efficiency model.

    Pressures are in kPa, temperatures in degC and densities in kg/m3. The dew
    points are those of the refrigerant at the suction and discharge pressures,
    the basis of manufacturer maps in evaporating and condensing temperature.
    A discharge above the critical pressure, which has no dew point, is given
    the critical point.
    """

    refrigerant: str
    P_suction: float
    P_discharge: float
    T_suction: float
    T_evap_dew: float
    T_cond_dew: float
    rho_suction: float
    rho_evap_dew: float
    rho_cond_dew: float

    @property
    def pressure_ratio(self) -> float:
        """Discharge over suction pressure."""
        return self.P_discharge / self.P_suction


CompressorEfficiencyFunction = Callable[[CompressorConditions], float]

# HeatPumpLab bounds every correlation to this range.
_CORRELATION_BOUNDS = (0.4, 1.0)


@dataclass(frozen=True)
class _Correlation:
    """A built-in correlation by name; a plain class so it pickles."""

    name: str

    def __call__(self, conditions: CompressorConditions) -> float:
        pr = conditions.pressure_ratio
        if self.name == "pareto_front":
            eta = pareto_front_efficiency(pr)
        elif self.name == "wang_cleland":
            eta = wang_cleland_efficiency(pr)
        else:
            eta = brendel_efficiency(pr, conditions.P_suction)
        lo, hi = _CORRELATION_BOUNDS
        return min(max(eta, lo), hi)


def compressor_efficiency_model(
    model: str | CompressorEfficiencyFunction, eta_constant: float
) -> float | CompressorEfficiencyFunction:
    """Return a constant efficiency or a function of ``CompressorConditions``.

    ``"constant"`` gives ``eta_constant``; a built-in name gives its bounded
    correlation; a function is returned as given.
    """
    if callable(model):
        return model
    if model == "constant":
        return float(eta_constant)
    return _Correlation(model)


class CompressorModelError(RuntimeError):
    """A compressor model returned an unusable efficiency."""


def evaluate_compressor_model(
    model: CompressorEfficiencyFunction, conditions: CompressorConditions
) -> float:
    """Return the model's isentropic efficiency, which must lie in (0, 1].

    A ``ValueError`` raised by the model marks the operating point infeasible
    for the targeting search; an efficiency out of range raises
    ``CompressorModelError``, which stops it.
    """
    eta = float(model(conditions))
    if not 0.0 < eta <= 1.0:
        name = getattr(model, "name", None) or getattr(
            model, "__name__", type(model).__name__
        )
        raise CompressorModelError(
            f"Compressor model {name} returned an isentropic efficiency of {eta}; "
            "it must lie in (0, 1]."
        )
    return eta
