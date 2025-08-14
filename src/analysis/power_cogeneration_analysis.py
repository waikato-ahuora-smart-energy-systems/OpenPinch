from __future__ import annotations

from typing import TYPE_CHECKING
from ..utils import *
from ..classes import *
from .support_methods import *

if TYPE_CHECKING:
    from ..classes import *

__all__ = ["get_power_cogeneration_above_pinch"]

#######################################################################################################
# Public API --- TODO
#######################################################################################################

def get_power_cogeneration_above_pinch(z: Zone):
    """Calculate the power cogeneration potential above pinch for a given process zone."""
    
    # === Step 1: Prepare turbine and model parameters ===
    turbine_params = _prepare_turbine_parameters(z.config)
    if turbine_params is None:
        return z  

    # === Step 2: Preprocess utilities ===
    utility_data = _preprocess_utilities(z, turbine_params)
    if utility_data is None:
        return z  

    # === Step 3: Solve turbine work and efficiency ===
    w_total, Wmax = _solve_turbine_work(turbine_params, utility_data)

    # === Step 4: Assign back to zone ===
    z.work_target = w_total
    z.turbine_efficiency_target = w_total / Wmax if Wmax > ZERO else 0.0

    return z


#######################################################################################################
# Helper functions
#######################################################################################################

def _prepare_turbine_parameters(config: Configuration):
    """Load and sanitize turbine parameters from config."""

    P_in = float(config.P_TURBINE_BOX)     # bar
    T_in = float(config.T_TURBINE_BOX)      # °C
    min_eff = float(config.MIN_EFF)         # minimum isentropic efficiency (decimal)
    model = config.COMBOBOX

    load_frac = min(max(config.LOAD, 0), 1)     # clamp between 0 and 1
    mech_eff = min(max(config.MECH_EFF, 0), 1)  # clamp between 0 and 1

    return {
        "P_in": P_in,
        "T_in": T_in,
        "min_eff": min_eff,
        "model": model,
        "load_frac": load_frac,
        "mech_eff": mech_eff,
        "CONDESATE_FLASH_CORRECTION": getattr(config, "CONDESATE_FLASH_CORRECTION", False),  # optional
    }


def _preprocess_utilities(z: Zone, turbine_params: dict):
    """Identify eligible hot utilities above pinch and prepare initial turbine stream data."""

    P_in = turbine_params["P_in"]

    Q_HU = 0.0
    HU_num = 0
    u: Stream
    for u in z.hot_utilities:
        if u.t_supply < T_CRIT and u.heat_flow > ZERO:
            HU_num += 1
            Q_HU += u.heat_flow

    if Q_HU < ZERO:
        return None  # No turbine opportunity

    # Initialize arrays
    P_out = [P_in]
    Q_users = [0]
    w_k = [0]
    w_isen_k = [0]
    m_k = [0]
    eff_k = [0]
    dh_is_k = [0]
    h_out = [h_pT(P_in, turbine_params["T_in"])]
    h_tar = [hL_p(P_in)]
    h_sat = [hV_p(P_in)]
    turbine = [0]

    m_in_est = 0.0  # estimated steam inlet mass flow
    Q_boiler = 0.0  # total boiler heat required

    s = 0
    for u in z.hot_utilities:
        if u.t_supply < T_CRIT and u.heat_flow > ZERO:
            P_out_n = psat_T(u.t_target) if abs(u.t_supply - u.t_target) < 1 + ZERO else psat_T(u.t_target + u.dt_cont * 2)

            if P_in >= P_out_n:
                s += 1
                for lst in [w_k, w_isen_k, eff_k, turbine]:
                    lst.append(0)

                P_out.append(P_out_n)
                Q_users.append(u.heat_flow)
                Q_boiler += u.heat_flow

                h_out.append(hV_p(P_out_n))
                h_tar.append(hL_p(P_out_n))
                h_sat.append(hV_p(P_out_n))

                dh_is = h_out[0] - h_ps(P_out_n, s_ph(P_out[0], h_out[0]))
                dh_is_k.append(dh_is)
                mflow = u.heat_flow / (h_out[0] - dh_is - h_tar[-1])
                m_k.append(mflow)
                m_in_est += mflow

    return {
        "P_out": P_out,
        "Q_users": Q_users,
        "w_k": w_k,
        "w_isen_k": w_isen_k,
        "m_k": m_k,
        "eff_k": eff_k,
        "dh_is_k": dh_is_k,
        "h_out": h_out,
        "h_tar": h_tar,
        "h_sat": h_sat,
        "turbine": turbine,
        "m_in_est": m_in_est,
        "Q_boiler": Q_boiler,
        "s": s + 1,
    }


def _solve_turbine_work(turbine_params: dict, utility_data: dict):
    """Solve turbine mass flow, work production, and efficiency based on utilities and turbine parameters."""

    P_out = utility_data["P_out"]
    Q_users = utility_data["Q_users"]
    w_k = utility_data["w_k"]
    w_isen_k = utility_data["w_isen_k"]
    m_k = utility_data["m_k"]
    eff_k = utility_data["eff_k"]
    dh_is_k = utility_data["dh_is_k"]
    h_out = utility_data["h_out"]
    h_tar = utility_data["h_tar"]
    h_sat = utility_data["h_sat"]
    s = utility_data["s"]

    model = turbine_params["model"]
    load_frac = turbine_params["load_frac"]
    n_mech = turbine_params["mech_eff"]
    min_eff = turbine_params["min_eff"]
    flash_correction = turbine_params.get("CONDESATE_FLASH_CORRECTION", False)

    m_in_est = utility_data["m_in_est"]
    i = 0

    while True:
        m_in = m_in_est
        m_in_k = m_in_est
        m_in_est = 0

        for j in range(1, s):
            m_in_k -= m_k[j - 1]
            dh_is_k[j] = h_out[j - 1] - h_ps(P_out[j], s_ph(P_out[j - 1], h_out[j - 1]))
            w_isen_k[j] = m_in_k * dh_is_k[j]
            m_max = m_in_k / load_frac if load_frac > 0 else 0

            if m_in_k > 0:
                if model == 'Sun & Smith (2015)':
                    w_k[j] = Work_SunModel(P_out[j-1], h_out[j-1], P_out[j], h_sat[j], m_in_k, m_max, dh_is_k[j], n_mech)
                elif model == 'Medina-Flores et al. (2010)':
                    w_k[j] = Work_MedinaModel(P_out[j-1], m_in_k, dh_is_k[j])
                elif model == 'Varbanov et al. (2004)':
                    w_k[j] = Work_THM(P_out[j-1], h_out[j-1], P_out[j], h_sat[j], m_in_k, dh_is_k[j], n_mech)
                elif model == 'Fixed Isentropic Turbine':
                    w_k[j] = w_isen_k[j] * min_eff
            else:
                w_k[j] = 0

            if w_isen_k[j] > 0:
                eff_k[j] = w_k[j] / w_isen_k[j]
            else:
                eff_k[j] = min_eff

            if eff_k[j] <= min_eff:
                w_k[j] = min_eff * w_isen_k[j]

            if m_in_k > 0:
                h_out[j] = h_out[j-1] - w_k[j] / (m_in_k * n_mech)
            else:
                h_out[j] = h_out[j-1]

            if m_in_k > 0:
                if flash_correction:
                    Q_flash = m_k[j-1] * (h_tar[j-1] - h_tar[j])
                    m_k[j] = (Q_users[j] - Q_flash) / (h_out[j] - h_tar[j])
                else:
                    m_k[j] = Q_users[j] / (h_out[j] - h_tar[j])
            else:
                m_k[j] = 0

            m_in_est += m_k[j]

        i += 1
        if abs(m_in - m_in_est) < ZERO or i >= 3:
            break

    w_total = sum(w_k)
    Wmax = sum(w_isen_k)

    return w_total, Wmax




# def get_power_cogeneration_above_pinch(z: Zone): # type: ignore
#     config: Configuration = z.config
#     P_in = float(config.P_TURBINE_BOX) # bar
#     T_in = float(config.T_TURBINE_BOX) # C
#     MinEff = float(config.MIN_EFF)     # %
#     SelectedModel = config.COMBOBOX

#     load_frac = config.LOAD
#     n_mech = config.MECH_EFF

#     if load_frac > 1:
#         load_frac = 1
#     elif load_frac < 0:
#         load_frac = 0

#     if n_mech > 1:
#         n_mech = 1
#     elif n_mech < 0:
#         n_mech = 0
    
#     Q_HU = 0
#     for Utility_k in z.hot_utilities:
#         Q_HU += Utility_k.heat_flow
#     HU_num = len(z.hot_utilities)
#     if Q_HU < ZERO:
#         return z
                    
#     if SelectedModel == 'Sun & Smith (2015)':
#         SunCoef = [ [ [None for k in range(3)] for j in range(3)] for i in range(2)]
#         Set_Coeff(SunCoef, None)
#     elif SelectedModel == 'Varbanov et al. (2004)':
#         VarCoef = [ [ [None for k in range(4)] for j in range(2)] for i in range(2)]
#         Set_Coeff(None, VarCoef)

#     P_out = [P_in] # bar
#     Q_users = [0]
#     w_k = [0]
#     w_isen_k = [0]
#     m_k = [0] # kg/s
#     eff_k = [0]
#     dh_is_k = [0]
#     h_out = [h_pT(P_in, T_in)] # kJ/kg for turbine
#     h_tar = [hL_p(P_out[0])] # kJ/kg for after condensing
#     h_sat = [hV_p(P_out[0])] # kJ/kg
#     turbine = [0]

#     m_in_est = 0             # kg/s
#     Q_boiler = 0             # kW
    
#     # Preparation of data for turbine calculation
#     s = 0
#     for i in range(HU_num):
#         Utility_i = z.hot_utilities[i]
#         if Utility_i.t_supply < T_CRIT:
#             P_out_n = psat_T(Utility_i.t_target) if abs(Utility_i.t_supply - Utility_i.t_target) < 1 + ZERO \
#                 else psat_T(Utility_i.t_target + Utility_i.dt_cont * 2) # bar
#             if P_in >= P_out_n and Utility_i.heat_flow > ZERO:
#                 s += 1
#                 for l in [w_k, w_isen_k, eff_k, turbine]:
#                     l.append(0)

#                 P_out.append(P_out_n)                      # bar
#                 Q_users.append(Utility_i.heat_flow)        # kW
#                 Q_boiler += Q_users[s]                  # kW

#                 h_out.append(hV_p(P_out[s]))               # kJ/kg
#                 h_tar.append(hL_p(P_out[s]))               # kJ/kg
#                 h_sat.append(hV_p(P_out[s]))               # kJ/kg

#                 dh_is_k.append(h_out[0] - h_ps(P_out[s], s_ph(P_out[0], h_out[0])))    # kJ/kg
#                 m_k.append(Q_users[s] / (h_out[0] - dh_is_k[s] - h_tar[s]))            # kg/s
#                 m_in_est += m_k[s]                                                     # kg/s
#     s += 1

#     # Turbine calculation
#     i = 0
#     while True:
#         m_in = m_in_est
#         m_in_k = m_in_est
#         m_in_est = 0
        
#         for j in range(1, s):
#             m_in_k -= m_k[j - 1]                                                    # kg/s
#             dh_is_k[j] = h_out[j - 1] - h_ps(P_out[j], s_ph(P_out[j - 1], h_out[j - 1]))    # kJ/kg
#             w_isen_k[j] = m_in_k * dh_is_k[j]                                               # kW
#             m_max = m_in_k / load_frac                                                      # kg/s
            
#             # Determine the work production based on various Turbine Models
#             if m_in_k > 0:
#                 if SelectedModel == 'Sun & Smith (2015)':
#                     w_k[j] = Work_SunModel(P_out[j - 1], h_out[j - 1], P_out[j], h_sat[j], m_in_k, m_max, dh_is_k[j], n_mech, SunCoef)      # kW
#                 elif SelectedModel == 'Medina-Flores et al. (2010)':
#                     w_k[j] = Work_MedinaModel(P_out[j - 1], m_in_k, dh_is_k[j])                                                             # kW
#                 elif SelectedModel == 'Varbanov et al. (2004)':
#                     w_k[j] = Work_THM(P_out[j - 1], h_out[j - 1], P_out[j], h_sat[j], m_in_k, dh_is_k[j], n_mech, VarCoef)                  # kW
#                 elif SelectedModel == 'Fixed Isentropic Turbine':
#                     w_k[j] = w_isen_k[j] * MinEff                                                                                           # kW
#             else:
#                 w_k[j] = 0

#             if w_isen_k[j] > 0:
#                 eff_k[j] = w_k[j] / w_isen_k[j]
#             else:
#                 eff_k[j] = MinEff
#             if eff_k[j] <= MinEff:
#                 w_k[j] = MinEff * w_isen_k[j]                                                # kW
#             if m_in_k > 0:
#                 h_out[j] = h_out[j - 1] - w_k[j] / (m_in_k * n_mech)
#             else:
#                 h_out[j] = h_out[j - 1]    # kJ/kg
            
#             # Correct for high pressure condensate flash, if selected
#             if m_in_k > 0:
#                 if config.CONDESATE_FLASH_CORRECTION:
#                     Q_flash = m_k[j - 1] * (h_tar[j - 1] - h_tar[j])
#                     m_k[j] = (Q_users[j] - Q_flash) / (h_out[j] - h_tar[j])
#                 else:
#                     m_k[j] = Q_users[j] / (h_out[j] - h_tar[j])
#             else:
#                 m_k[j] = 0                  # kg/s
            
#             m_in_est += m_k[j]    # kg/s

#         i += 1
#         if (abs(m_in - m_in_est) < ZERO and i >= 100) or i >= 3:
#             break
    
#     w = 0
#     Wmax = 0
#     for j in range(s):
#         w += w_k[j]
#         Wmax += w_isen_k[j]

#     z.work_target = w        # kW
#     if Wmax > 0:
#         z.turbine_efficiency_target = w / Wmax
#     else:
#         z.turbine_efficiency_target = 0 # %

#     return z

# #######################################################################################################
# # Helper Functions
# #######################################################################################################

def Work_MedinaModel(P_in, m, dh_is):
    """'Determines power generation based on the thermodynamic model of Medina-Flores & Picón-Núñez (2010).
    """
    # Reference for Turbine Model 1: Medina-Flores & Picón-Núñez (2010)
    #                               Modelling the power production of single and multiple extraction steam turbines.
    #                               Chemical Engineering Science 65, 2811-2820
    # Part load not used
    A0 = 185.4 + 43.3 * (P_in * 0.1)        # a0 in kW, P in bar
    b0 = 1.2057 + 0.0075 * (P_in * 0.1)     # b0 in dimensionless, P in bar
    return (m * dh_is - A0) / b0            # kW

def Work_SunModel(P_in, h_in, P_out, h_sat, m, m_max, dh_is, n_mech, t_type=1):
    """Determines power generation based on the correlation model of Sun and Smith (2015).
    """
    # 'Reference for Turbine Model 2: Sun & Smith (2015)
    # '                               Performance Modeling of New and Existing Steam Turbines
    # '                               I&CE 54, 1908-1915

    coeff = {
        "BPST": {
            "a": [1.18795366, -0.00029564, 0.004647288],
            "b": [449.9767142, 5.670176939, -11.5045814],
            "c": [0.205149333, -0.000695171, 0.002844611],
        },
        "CT": {
            "a": [1.314991261, -0.001634725, -0.367975103],
            "b": [-437.7746025, 29.00736723, 10.35902331],
            "c": [0.07886297, 0.000528327, -0.703153891],
        }
    }

    # Model coefficients where P in bar
    A0 = coeff[t_type]["a"][0] + coeff[t_type]["a"][1] * (P_in) + coeff[t_type]["a"][2] * (P_out)
    b0 = coeff[t_type]["b"][0] + coeff[t_type]["b"][1] * (P_in) + coeff[t_type]["b"][2] * (P_out)
    c0 = coeff[t_type]["c"][0] + coeff[t_type]["c"][1] * (P_in) + coeff[t_type]["c"][2] * (P_out)
    
    # Willan's line coefficients and predicted work (after isentropic and mechanical efficiency loss)
    W_int = c0 / A0 * (m_max * dh_is - b0)
    n = (1 + c0) / A0 * (dh_is - b0 / m_max)
    w_act = n * m - W_int
    h_out = h_in - w_act / (n_mech * m)
    
    if h_out <= h_sat + ZERO and t_type == 1:
        w_act = Work_SunModel(P_in, h_in, P_out, h_sat, m, m_max, dh_is, n_mech, 2)
    return w_act

def Work_THM(P_in, h_in, P_out, h_sat, m, dh_is, n_mech, t_size=1, t_type=1):
    """Determines power generation based on the Turbine Hardware Model of Varbanov et al. (2004).
    """
    # 'Reference for Turbine Model 3: Varbanov et al. (2004)
    # '                               Modelling and Optimization of Utility Systems
    # '                               Trans IChemE, Part A, Chemical Engineering Research and Design, 2004, 82(A5): 561–578
    # Part load not used
    coeff = {
        "BPST": {
            "<2MW": [0, 0.00108, 1.097, 0.00172],
            ">2MW": [0, 0.00423, 1.155, 0.000538],
        },
        "CT": {
            "<2MW": [0, 0.000662, 1.191, 0.000759],
            ">2MW": [-0.463, 0.00353, 1.22, 0.000148],
        }
    }

    dT_sat = Tsat_p(P_in) - Tsat_p(P_out)
    a = (coeff[t_type][t_size][0] + coeff[t_type][t_size][1] * dT_sat) * 1000
    b = coeff[t_type][t_size][2] + coeff[t_type][t_size][3] * dT_sat
    w_max = (dh_is * m - a) / b

    if w_max > 2000 and t_size == 1:
        t_size = 2
        w_max = Work_THM(P_in, h_in, P_out, h_sat, m, dh_is, n_mech, t_size)

    h_out = h_in - w_max / (n_mech * m)
    if h_out <= h_sat + ZERO and t_type == 1:
        t_type = 2
        w_max = Work_THM(P_in, h_in, P_out, h_sat, m, dh_is, n_mech, t_size, t_type)

    return w_max



def Set_Coeff(SunCoef=None, VarCoef=None):
    """Sets the model coefficients."""


    if VarCoef != None:
        # Model coefficients for the Varbanov et al. model
        # BPST
        # < 2MW
        VarCoef[0][0][0] = 0
        VarCoef[0][0][1] = 0.00108
        VarCoef[0][0][2] = 1.097
        VarCoef[0][0][3] = 0.00172
        
        # > 2MW
        VarCoef[0][1][0] = 0
        VarCoef[0][1][1] = 0.00423
        VarCoef[0][1][2] = 1.155
        VarCoef[0][1][3] = 0.000538
        
        # CT
        # < 2MW
        VarCoef[1][0][0] = 0
        VarCoef[1][0][1] = 0.000662
        VarCoef[1][0][2] = 1.191
        VarCoef[1][0][3] = 0.000759
        
        # > 2MW
        VarCoef[1][1][0] = -0.463
        VarCoef[1][1][1] = 0.00353
        VarCoef[1][1][2] = 1.22
        VarCoef[1][1][3] = 0.000148

