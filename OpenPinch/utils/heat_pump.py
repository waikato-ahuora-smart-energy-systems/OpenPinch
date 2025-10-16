from CoolProp.Plots import SimpleCompressionCycle


def get_hp_cycle_COP(
    fld = 'Ammonia',
    Te = 10,
    Tc = 60,
    dT_sh=10,
    dT_sc=5, 
    eta_com=0.7,
):
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
    print(f"COP heating:{cycle.COP_heating()}")
get_hp_cycle_COP()