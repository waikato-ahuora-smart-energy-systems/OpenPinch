"""Numerical inputs for ORC targeting, independent of the zone model."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OrcTargetInputs"]


@dataclass(frozen=True)
class OrcTargetInputs:
    """Settings for one ORC target, in kW, degC, K, $/MWh and $.

    ``capital_recovery_factor`` annualises installed capital (1/y); it is
    computed by the caller from the discount rate and service life.
    """

    n_stages: int = 1
    eta_ii: float = 0.5
    dt_cont: float = 5.0
    T_cond: float = 30.0
    min_lift: float = 10.0
    dt_phase_change: float = 0.01
    load_fraction: float = 1.0
    ele_price: float = 100.0
    cooling_price: float = 2.5
    annual_hours: float = 8300.0
    equipment_cost: float = 2.3e6
    installation_factor: float = 1.3
    cost_exp: float = 0.75
    capital_recovery_factor: float = 0.0944
    max_multistart: int = 5
    bb_minimiser: str = "dual_annealing"
    maximum_iterations: int = 300
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.n_stages) < 1:
            raise ValueError("An ORC needs at least one stage.")
        if not 0.0 < float(self.eta_ii) <= 1.0:
            raise ValueError("The ORC second-law efficiency must be in (0, 1].")
        if not 0.0 <= float(self.load_fraction) <= 1.0:
            raise ValueError("The ORC load fraction must be in [0, 1].")
        for name in ("dt_cont", "min_lift", "dt_phase_change", "annual_hours"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must not be negative.")

    @property
    def T_evap_shifted_min(self) -> float:
        """Coldest shifted evaporating temperature: condenser plus lift."""
        return float(self.T_cond + self.min_lift + self.dt_cont)

    @property
    def power_value(self) -> float:
        """Value of 1 kW of net power for a year, in $/y."""
        return self.ele_price * self.annual_hours / 1000.0

    @property
    def cooling_value(self) -> float:
        """Cost of 1 kW of cooling for a year, in $/y."""
        return self.cooling_price * self.annual_hours / 1000.0
