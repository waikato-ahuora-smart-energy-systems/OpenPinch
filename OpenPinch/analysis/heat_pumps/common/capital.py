"""Installed capital of heat pump components.

Purchased equipment cost (PEC) follows ``PEC = PEC_ref (X / X_ref)^a`` with the
data of Ommen et al. (2015), "Technical and economic working domains of
industrial heat pumps: Part 1", Int. J. Refrig. 55, Table 4 (2013 EUR):
screw compressors on suction volume flow, motors on shaft power and plate heat
exchangers on area. Water vapour has no Ommen correlation: its turbo compressors
are anchored on Zuhlsdorf et al. (2018), "Design recommendations for R-718 heat
pumps in high temperature applications", 13th IIR Gustav Lorentzen Conference,
DOI 10.18462/iir.gl.2018.1367: two of 1231 and 468 m3/h cost 50,000 EUR (2018)
with their motors, gears and inverters, scaled by Ommen's hydrocarbon exponent.
Installed capital is 4.16 x PEC (Bejan et al. 1996, as
Ommen), scaled by ``COST_LEVEL`` so that Ommen's worked example, a 1 MW
ammonia heat pump, costs the IEA HPT Annex 68 level for 1 MW.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ....domain.configuration import tol
from ....domain.fluids import build_coolprop_abstract_state
from ...heat_transfer import compute_LMTD_from_dts
from ...targeting.temperature_driving_force import get_temperature_driving_forces

__all__ = [
    "ANNEX_68_COST_AT_1_MW",
    "COST_LEVEL",
    "REFERENCE_MACHINE",
    "CompressorDuty",
    "MachineComponents",
    "compressor_pec",
    "exchanger_pec",
    "installed_capital",
    "matched_area",
    "motor_pec",
]

# The previous OpenPinch fit to the IEA HPT Annex 68 cost data: installed
# capital of a 1 MW single-stage machine below 75 degC (2025 USD).
ANNEX_68_COST_AT_1_MW = 2.3 * 485_000.0

# Installed capital per unit of purchased equipment cost (Bejan et al. 1996).
TOTAL_CAPITAL_PER_PEC = 4.16

# Zuhlsdorf et al. (2018): the suction volume flows (m3/h) and the joint cost
# (EUR, motors included) of a two-stage R718 turbo compressor pair.
_WATER_ANCHOR_FLOWS = (1231.0, 468.0)
_WATER_ANCHOR_COST = 50_000.0
_WATER_EXPONENT = 0.73

# (PEC_ref EUR, X_ref m3/h, exponent, largest compressor m3/h) per family.
_COMPRESSORS = {
    "ammonia": (11_914.0, 178.4, 0.66, 280.0),
    "hydrocarbon": (19_850.0, 279.8, 0.73, 280.0),
    "carbon_dioxide": (12_109.0, 25.6, 0.42, 25.6),
    "r134a": (10_631.0, 178.4, 0.79, 280.0),
    # A single turbo compressor or fan moves up to about 280,000 m3/h of steam.
    "water": (
        _WATER_ANCHOR_COST
        / sum(
            (flow / _WATER_ANCHOR_FLOWS[0]) ** _WATER_EXPONENT
            for flow in _WATER_ANCHOR_FLOWS
        ),
        _WATER_ANCHOR_FLOWS[0],
        _WATER_EXPONENT,
        280_000.0,
    ),
}
# Families whose compressor price includes the motor.
_MOTOR_INCLUDED = frozenset({"hydrocarbon", "water"})
_HYDROCARBONS = frozenset(
    {
        "n-Propane",
        "Propylene",
        "IsoButane",
        "n-Butane",
        "IsoButene",
        "Isopentane",
        "n-Pentane",
        "Neopentane",
        "Cyclopentane",
        "n-Hexane",
        "Ethane",
        "Ethylene",
    }
)
_MOTOR = (10_710.0, 250.0, 0.65)
_PLATE_EXCHANGER = (15_526.0, 42.0, 0.8)


@dataclass(frozen=True)
class CompressorDuty:
    """One compression stage: suction volume flow in m3/h, shaft power in kW."""

    fluid: str
    volume_flow: float
    shaft_power: float


@dataclass(frozen=True)
class MachineComponents:
    """The compressors and the exchanger areas (m2) of one machine."""

    compressors: tuple[CompressorDuty, ...] = ()
    exchanger_areas: tuple[float, ...] = ()


def _family(fluid: str) -> str:
    try:
        names = build_coolprop_abstract_state(fluid).fluid_names()
    except ValueError:
        return "r134a"
    if names == ["Ammonia"]:
        return "ammonia"
    if names == ["CarbonDioxide"]:
        return "carbon_dioxide"
    if names == ["Water"]:
        return "water"
    if len(names) == 1 and names[0] in _HYDROCARBONS:
        return "hydrocarbon"
    return "r134a"


def compressor_pec(fluid: str, volume_flow: float) -> float:
    """Return the PEC (2013 EUR) of the compressors for a suction volume flow.

    A flow beyond the largest compressor is split over several equal ones in
    parallel. Fluids outside Ommen's families and water are priced as R134a.
    """
    volume_flow = max(float(volume_flow), 0.0)
    if volume_flow <= 0.0:
        return 0.0
    reference, flow_ref, exponent, largest = _COMPRESSORS[_family(fluid)]
    units = max(math.ceil(volume_flow / largest), 1)
    return units * reference * (volume_flow / units / flow_ref) ** exponent


def motor_pec(fluid: str, shaft_power: float) -> float:
    """Return a compressor motor's PEC (2013 EUR).

    Hydrocarbon and water compressor prices already include the motor.
    """
    if _family(fluid) in _MOTOR_INCLUDED or shaft_power <= 0.0:
        return 0.0
    reference, power_ref, exponent = _MOTOR
    return reference * (float(shaft_power) / power_ref) ** exponent


def exchanger_pec(area: float) -> float:
    """Return a plate heat exchanger's PEC (2013 EUR) for an area in m2."""
    if area <= 0.0:
        return 0.0
    reference, area_ref, exponent = _PLATE_EXCHANGER
    return reference * (float(area) / area_ref) ** exponent


def _purchased_equipment_cost(machine: MachineComponents) -> float:
    return sum(
        compressor_pec(c.fluid, c.volume_flow) + motor_pec(c.fluid, c.shaft_power)
        for c in machine.compressors
    ) + sum(exchanger_pec(area) for area in machine.exchanger_areas)


# Ommen's worked example (Table A.4): ammonia, 467 m3/h, 146 kW, a 73.2 m2
# evaporator and a 44.1 m2 condenser.
REFERENCE_MACHINE = MachineComponents(
    compressors=(CompressorDuty("Ammonia", 467.0, 146.0),),
    exchanger_areas=(73.2, 44.1),
)

COST_LEVEL = ANNEX_68_COST_AT_1_MW / (
    TOTAL_CAPITAL_PER_PEC * _purchased_equipment_cost(REFERENCE_MACHINE)
)


def matched_area(
    T_hot: np.ndarray,
    H_hot: np.ndarray,
    T_cold: np.ndarray,
    H_cold: np.ndarray,
    *,
    extra_dt: float,
    heat_transfer_coefficient: Callable[..., np.ndarray],
) -> float:
    """Return the area (m2) to transfer between two balanced curves.

    The curves run hottest first with heat flows (kW) falling to zero at the
    cold end, as the composite curves of an area target. Each vertical
    interval is a counter-current exchanger: its area is its duty over
    ``U x LMTD``, with ``extra_dt`` added to both end differences and ``U``
    (W/m2K) from ``heat_transfer_coefficient(t_h1, t_h2, t_c1, t_c2)``.
    Curves that touch need an infinite area.
    """
    tdf = get_temperature_driving_forces(T_hot, H_hot, T_cold, H_cold)
    duty = np.asarray(tdf["dh_vals"], dtype=float)
    dt1 = np.asarray(tdf["delta_T1"], dtype=float) + float(extra_dt)
    dt2 = np.asarray(tdf["delta_T2"], dtype=float) + float(extra_dt)
    carries_heat = duty > tol
    if np.any(carries_heat & ((dt1 <= tol) | (dt2 <= tol))):
        return float("inf")
    duty, dt1, dt2 = duty[carries_heat], dt1[carries_heat], dt2[carries_heat]
    u = np.asarray(
        heat_transfer_coefficient(
            *(
                np.asarray(tdf[key])[carries_heat]
                for key in ("t_h1", "t_h2", "t_c1", "t_c2")
            )
        ),
        dtype=float,
    )
    return float(np.sum(duty * 1000.0 / (u * compute_LMTD_from_dts(dt1, dt2))))


def installed_capital(machine: MachineComponents) -> float:
    """Return a machine's installed capital at the Annex 68 cost level."""
    return COST_LEVEL * TOTAL_CAPITAL_PER_PEC * _purchased_equipment_cost(machine)
