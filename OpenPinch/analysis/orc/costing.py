"""Installed ORC capital and the annual cost of an ORC design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inputs import OrcTargetInputs

__all__ = ["OrcCosts", "orc_costs", "orc_machine_capital_costs"]

# Units below this net power (kW) are not built and cost nothing.
_MIN_UNIT_POWER = 1e-6


def orc_machine_capital_costs(
    W_net: np.ndarray,
    inputs: OrcTargetInputs,
) -> np.ndarray:
    """Installed capital of each parallel unit, in $.

    ``C = F_inst * C_eq * (W_net / 1 MW)^n`` per unit. The defaults put an
    installed 1 MW unit at about $3,000/kW (2025 USD), between the US EPA
    (2021) range of $1,900-4,500/kW and Lemmens (2016) heat-recovery
    projects, with n = 0.75 matching large-unit market averages of about
    $1,500/kW (Tartiere and Astolfi, 2017). Treat it as accurate to about
    +/-30 %.
    """
    W = np.maximum(np.asarray(W_net, dtype=float), 0.0)
    scale = float(inputs.installation_factor) * float(inputs.equipment_cost)
    costs = scale * (W / 1000.0) ** float(inputs.cost_exp)
    return np.where(W > _MIN_UNIT_POWER, costs, 0.0)


@dataclass(frozen=True)
class OrcCosts:
    """Annual cost change from building the ORC, in $/y (capital in $)."""

    machine_capital_costs: tuple[float, ...]
    capital_cost: float
    annualized_capital_cost: float
    power_value: float
    cooling_cost_change: float
    total_annualized_cost_change: float


def orc_costs(
    *,
    W_net: np.ndarray,
    Q_in: float,
    Q_out: float,
    inputs: OrcTargetInputs,
) -> OrcCosts:
    """Return the annual cost change of an ORC against no ORC.

    The ORC takes ``Q_in`` from the process, which then needs that much less
    cooling, and its condenser needs ``Q_out`` of cooling. Its net power is
    valued at the electricity price. A negative total means it pays.
    """
    machine = orc_machine_capital_costs(W_net, inputs)
    capital = float(machine.sum())
    annualized = capital * float(inputs.capital_recovery_factor)
    W_total = float(np.maximum(np.asarray(W_net, dtype=float), 0.0).sum())
    power_value = W_total * inputs.power_value
    cooling_change = (float(Q_out) - float(Q_in)) * inputs.cooling_value
    return OrcCosts(
        machine_capital_costs=tuple(float(c) for c in machine),
        capital_cost=capital,
        annualized_capital_cost=annualized,
        power_value=power_value,
        cooling_cost_change=cooling_change,
        total_annualized_cost_change=annualized - power_value + cooling_change,
    )
