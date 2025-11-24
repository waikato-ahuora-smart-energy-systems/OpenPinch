# """Convenience utilities for evaluating a simple heat pump cycle."""

# #from CoolProp.Plots import SimpleCompressionCycle
# from CoolProp.CoolProp import PropsSI
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# def get_hp_cycle_COP(
#     fld = 'Ammonia',
#     Te = 10,
#     Tc = 60,
#     dT_sh=10,
#     dT_sc=5, 
#     eta_com=0.7,
# ):
#     """Compute the heating coefficient of performance (COP).

#     Args:
#         fld: Working fluid name understood by CoolProp.
#         Te: Evaporator saturation temperature in Celsius.
#         Tc: Condenser saturation temperature in Celsius.
#         dT_sh: Superheat at evaporator outlet in Kelvin.
#         dT_sc: Subcooling at condenser outlet in Kelvin.
#         eta_com: Compressor isentropic efficiency (0-1).
#     Return:
#         COP_heating: Coefficient of Performance for heating.
#     """
#     Te += 273.15
#     Tc += 273.15
#     cycle = SimpleCompressionCycle(f'HEOS::{fld}', 'PH', unit_system='EUR')
#     cycle.simple_solve_dt(
#         Te=Te, 
#         Tc=Tc, 
#         dT_sh=dT_sh, 
#         dT_sc=dT_sc, 
#         eta_com=eta_com, 
#         fluid=f'HEOS::{fld}', 
#         SI=True
#     )
#     COP_heating = cycle.COP_heating()
#     return COP_heating


# def get_hp_cycle(fluid, T_evap, T_cond, eta_is=0.75, subcooling=5.0):
#     """
#     Compute main state points for a simple vapor-compression HP.

#     Returns states (dict), COP, q_cond, q_evap, w_comp
#     """

#     T_evap_K = T_evap + 273.15
#     T_cond_K = T_cond + 273.15
#     T_subcool_K = T_cond_K - subcooling

#     # Saturation pressures
#     p_evap = PropsSI('P', 'T', T_evap_K, 'Q', 1, fluid)
#     p_cond = PropsSI('P', 'T', T_cond_K, 'Q', 0, fluid)

#     # State 1: evaporator outlet (sat. vapor)
#     h1 = PropsSI('H', 'T', T_evap_K, 'Q', 1, fluid)
#     s1 = PropsSI('S', 'T', T_evap_K, 'Q', 1, fluid)

#     # State 2s: isentropic compression
#     h2s = PropsSI('H', 'P', p_cond, 'S', s1, fluid)
#     h2 = h1 + (h2s - h1) / eta_is
#     T2 = PropsSI('T', 'P', p_cond, 'H', h2, fluid)

#     # State 3: saturated liquid after condensation
#     h3 = PropsSI('H', 'T', T_cond_K, 'Q', 0, fluid)

#     # State 4: after throttling
#     h4 = h3
#     T4 = PropsSI('T', 'P', p_evap, 'H', h4, fluid)

#     # Subcooled outlet
#     h3_sub = PropsSI('H', 'T', T_subcool_K, 'P', p_cond, fluid)

#     # Key energy quantities
#     q_cond = h2 - h3_sub
#     q_evap = h1 - h4
#     w_comp = h2 - h1
#     COP = q_cond / w_comp

#     # Create condenser curve (four T-Q points)
#     # 1. Compressor outlet -> start condensing (dew)
#     h_ddew = PropsSI('H', 'T', T_cond_K, 'Q', 1, fluid)
#     T_ddew = T_cond_K

#     condenser_points = [
#         (h2, T2),          # compressor outlet
#         (h_ddew, T_ddew),  # start condensation
#         (h3, T_cond_K),    # end condensation
#         (h3_sub, T_subcool_K)  # after subcooling
#     ]

#     states = {
#         '1': {'T': T_evap_K, 'p': p_evap, 'h': h1},
#         '2': {'T': T2, 'p': p_cond, 'h': h2},
#         '3': {'T': T_cond_K, 'p': p_cond, 'h': h3},
#         '3_sub': {'T': T_subcool_K, 'p': p_cond, 'h': h3_sub},
#         '4': {'T': T4, 'p': p_evap, 'h': h4}
#     }

#     return states, condenser_points, COP, q_cond, q_evap, w_comp

# def simulate_multi_hp(hp_list):
#     """
#     hp_list = [
#         {'fluid':'R600a', 'T_evap':10, 'T_cond':50, 'eta_is':0.75, 'subcooling':5, 'Q_sink':1000},
#         ...
#     ]
#     Returns:
#         - composite_sink: dict with composite T-Q (°C, MW)
#         - results: list of performance summaries
#         - hp_segments: list of DataFrames, one per HP, with condenser segment data
#     """

#     hp_segments = []
#     results = []

#     for hp in hp_list:
#         # Simulate single HP
#         states, cond_curve, COP, q_cond, q_evap, w_comp = get_hp_cycle(
#             hp['fluid'], hp['T_evap'], hp['T_cond'], hp['eta_is'], hp['subcooling']
#         )

#         Q_sink = hp['Q_sink'] * 1e3  # kW → W
#         m_dot = Q_sink / q_cond      # mass flow rate

#         # Extract enthalpies and temperatures
#         h_vals = np.array([pt[0] for pt in cond_curve])
#         T_vals = np.array([pt[1] - 273.15 for pt in cond_curve])  # °C

#         # Convert to incremental Q per section
#         Q_points = (h_vals[0] - h_vals) * m_dot / 1e3  # kJ/s = kW

#         # Build 3 segments: desuperheating, condensation, subcooling
#         seg_data = []
#         for i in range(3):
#             seg_data.append({
#                 'Segment': ['Desuperheating', 'Condensation', 'Subcooling'][i],
#                 'T_in (°C)': T_vals[i],
#                 'T_out (°C)': T_vals[i+1],
#                 'Q (kW)': Q_points[i+1] - Q_points[i],
#             })

#         df_segments = pd.DataFrame(seg_data)
#         hp_segments.append(df_segments)

#         results.append({
#             'fluid': hp['fluid'],
#             'COP': COP,
#             'T_evap': hp['T_evap'],
#             'T_cond': hp['T_cond'],
#             'Q_sink_kW': hp['Q_sink'],
#             'W_comp_kW': hp['Q_sink'] / COP,
#         })


#     return results, hp_segments



# # ============================================================
# # Plotting function
# # ============================================================
# def plot_sink_curves(all_sink_curves, composite_sink):
#     plt.figure(figsize=(7,5))
#     for i, (Q, T) in enumerate(all_sink_curves):
#         plt.plot(Q/1e3, T, '--', label=f'HP{i+1} sink')  # MW
#     plt.plot(composite_sink['Q'], composite_sink['T'], 'k-', lw=2.5, label='Composite')
#     plt.xlabel('Cumulative Heat [MW]')
#     plt.ylabel('Temperature [°C]')
#     plt.title('Detailed Condenser (Sink) Curves of Multiple Heat Pumps')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# # ============================================================
# # Example
# # ============================================================
# if __name__ == "__main__":
#     hp_list = [
#         {'fluid':'R600a', 'T_evap':10, 'T_cond':50, 'eta_is':0.75, 'subcooling':5, 'Q_sink':500},
#         {'fluid':'R134a', 'T_evap':25, 'T_cond':70, 'eta_is':0.75, 'subcooling':5, 'Q_sink':1000},
#         {'fluid':'Ammonia', 'T_evap':40, 'T_cond':90, 'eta_is':0.8, 'subcooling':3, 'Q_sink':1500}
#     ]

#     results, hp_segments = simulate_multi_hp(hp_list)

#     print("\n=== Heat Pump Results ===")
#     for r in results:
#         print(f"{r['fluid']:10s} | COP={r['COP']:.2f} | Qsink={r['Q_sink_kW']:.0f} kW | Wcomp={r['W_comp_kW']:.0f} kW")

