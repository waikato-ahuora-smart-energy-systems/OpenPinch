"""Convenience utilities for evaluating a simple heat pump cycle."""

from CoolProp.Plots import SimpleCompressionCycle


def get_hp_cycle_COP(
    fld = 'Ammonia',
    Te = 10,
    Tc = 60,
    dT_sh=10,
    dT_sc=5, 
    eta_com=0.7,
):
    """Compute the heating coefficient of performance (COP).

    Args:
        fld: Working fluid name understood by CoolProp.
        Te: Evaporator saturation temperature in Celsius.
        Tc: Condenser saturation temperature in Celsius.
        dT_sh: Superheat at evaporator outlet in Kelvin.
        dT_sc: Subcooling at condenser outlet in Kelvin.
        eta_com: Compressor isentropic efficiency (0–1).
    Return:
        COP_heating: Coefficient of Performance for heating.
    """
    Te += 273.15
    Tc += 273.15
    cycle = SimpleCompressionCycle(f'HEOS::{fld}', 'PH', unit_system='EUR')
    cycle.simple_solve_dt(
        Te=Te, 
        Tc=Tc, 
        dT_sh=dT_sh, 
        dT_sc=dT_sc, 
        eta_com=eta_com, 
        fluid=f'HEOS::{fld}', 
        SI=True
    )
    COP_heating = cycle.COP_heating()
    return COP_heating

print(
    get_hp_cycle_COP()
)
